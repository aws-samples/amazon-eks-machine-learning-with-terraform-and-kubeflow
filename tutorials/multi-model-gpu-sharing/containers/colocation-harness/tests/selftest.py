"""Offline self-test. No cluster, no GPU, no network.

Covers the parts where a silent bug would produce a plausible-looking number: the
estimator's "not enough data" case, the delivered-fraction gate, the DCGM UUID join,
and record assembly. Run inside the image:

    docker run --rm --entrypoint python <image> -m tests.selftest
"""

from __future__ import annotations

import inspect
import sys
import types

import asyncio

from harness.batchmeans import batch_means_ci, relative_halfwidth
from harness import config as _cfgmod
from harness import engines as _engines
from harness import scrape as _scrape
from harness.config import RunConfig, TenantConfig
from harness.controller import (PHASE_POSITIONS, ROTATE_STEP, TENANT_CADENCE_S, Sample,
                                Sampler, StabilityThresholds, backlog_series,
                                steady_state_check, warmup_samples)
from harness.load import LoadDriver, RequestResult
from harness.observation import (VERDICT_DELIVERY, VERDICT_MEASURED, build_record,
                                 delivered_fraction)

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


# --- batch means -------------------------------------------------------------
check("too few samples -> None (not a tight interval)",
      batch_means_ci(list(range(19))) is None)
check("enough samples -> a number", isinstance(batch_means_ci(list(range(200))), float))
check("constant series -> zero half-width", batch_means_ci([5.0] * 100) == 0.0)
check("relative half-width of a constant series is 0",
      relative_halfwidth([5.0] * 100) == 0.0)
check("zero mean with spread -> inf, not 0",
      relative_halfwidth([-1.0] * 50 + [1.0] * 50) == float("inf"))
# Batch means aliases: a series alternating every sample has zero spread *between*
# batch means, because each batch averages to the same value. Not a bug -- it is what
# averaging within batches does -- but it means the estimator is blind to oscillation
# whose period divides the batch length. At a 5s cadence and k=10 that is oscillation
# faster than the window/10, which is why the steady-state test also watches the
# in-flight count, whose drift under overload is monotone rather than periodic.
check("batch means is blind to per-sample oscillation (recorded, not a bug)",
      relative_halfwidth([-1.0, 1.0] * 50) == 0.0)

# --- delivered fraction gate -------------------------------------------------
def rr(tokens: int, t: float = 1.0, ok: bool = True,
       chunks: int | None = None) -> RequestResult:
    """One result. `tokens` is the server's count; `chunks` defaults to matching it.

    Pass a lower `chunks` to model the real defect this fix exists for: the client
    observing fewer chunks than the server says it sent.
    """
    r = RequestResult(tenant="x", dispatch_t=t, scheduled_t=t, ok=ok,
                      chunks=tokens if chunks is None else chunks,
                      usage_tokens=tokens)
    r.ttft_s = 0.05
    r.itl_mean_s = 0.02
    r.itl_from_gaps_s = 0.021
    return r


# The gate must read the server's count, not the client's observation. A run where
# every request lost 5 chunks in the client's own path is a full-delivery run.
check("tokens prefers the server count over a short chunk count",
      rr(128, chunks=123).tokens == 128)
check("tokens falls back to chunks when the server sent no usage",
      RequestResult(tenant="x", dispatch_t=0, scheduled_t=0, ok=True, chunks=123).tokens == 123)

full = [rr(128) for _ in range(10)]
check("full delivery -> 1.0", delivered_fraction(full, 128, 10) == 1.0)
# Half the dispatched requests never came back at all. The missing tokens must land
# in the denominator or the shortfall is invisible.
check("half the requests never returned -> 0.5",
      delivered_fraction(full, 128, 20) == 0.5)
check("no dispatches -> None, not a divide-by-zero",
      delivered_fraction([], 128, 0) is None)

from harness.observation import chunk_provenance  # noqa: E402

cp = chunk_provenance([rr(128), rr(128), rr(120), rr(128)], 128)
check("token provenance separates a few short requests from many",
      cp["requests_short_of_osl"] == 1 and cp["tokens_min"] == 120
      and cp["tokens_short_examples"] == [120], str(cp))
check("token provenance is clean when OSL is honoured",
      chunk_provenance([rr(128)] * 5, 128)["requests_short_of_osl"] == 0)
# A client-side chunk loss must show up as a chunks/tokens divergence in the record
# and NOT as a delivery shortfall -- that conflation is the bug being fixed.
cp = chunk_provenance([rr(128, chunks=123)] * 4, 128)
check("client chunk loss is recorded but does not read as a shortfall",
      cp["requests_short_of_osl"] == 0 and cp["chunks_min"] == 123
      and cp["tokens_min"] == 128, str(cp))
check("a run with no usage chunk at all is flagged",
      chunk_provenance(
          [RequestResult(tenant="x", dispatch_t=0, scheduled_t=0, ok=True, chunks=123)],
          128)["requests_missing_usage"] == 1)
check("full delivery despite short chunk counts",
      delivered_fraction([rr(128, chunks=123)] * 10, 128, 10) == 1.0)

# --- sub-sample aggregation --------------------------------------------------
# The fix for the first live run's silent failure. A gauge read once per 5s can be
# phase-locked to the arrival schedule and read 0 while the GPU is 47% busy; averaging
# many sub-samples at a rotating phase is what removes that bias.
class _Drv:
    def __init__(self):
        self._n = 0

    def in_flight(self):
        return self._n

    delivered_chunks = 0
    issued = 0                      # the driver's own cumulative launch count


_drv_init = inspect.getsource(LoadDriver.__init__)
check("the fast loop reads only counters the real driver keeps in O(1)",
      callable(getattr(LoadDriver, "in_flight", None))
      and all(f"self.{a} = 0" in _drv_init for a in ("delivered_chunks", "issued")),
      "a read that walks a growing list -- `launched_in` over `_launched` -- makes the sampler "
      "miss its own cadence, which is the failure the two-loop split exists to avoid")


s = Sampler(None, {"a": _Drv()})
# In-flight alternates 0,1,0,1,... -- the case a point read gets wrong 100% of the time
# depending only on which phase it lands in.
for i in range(36):
    s._bucket.append({"in_flight": {"a": i % 2}, "delivered": {"a": 10 * i}})
sample = s._close_bucket({})
check("cumulative delivered takes the last value", sample.delivered["a"] == 350)
check("in-flight is the bucket mean, not a point read",
      abs(sample.in_flight["a"] - 0.5) < 1e-9, f"got {sample.in_flight['a']}")
check("sub-sample count recorded", sample.subsamples == 36)
check("bucket emptied after close", s._bucket == [])

# An empty bucket must not fabricate a value.
empty = s._close_bucket({})
check("empty bucket yields no client-side series", empty.in_flight == {}, str(empty.in_flight))

# --- nothing in a Sample comes from a server -------------------------------------------------
# The sampler used to hold tenant base URLs and an engine per tenant so it could scrape each
# server's /metrics. impl §14.6 deleted that, and this is the check that keeps it deleted: the
# constructor takes no tenant endpoint, so there is no wiring through which a server could
# thin the judged count or add a field to a record.
check("the sampler cannot be given a tenant endpoint",
      set(inspect.signature(Sampler.__init__).parameters)
      == {"self", "exporter_url", "drivers", "cadence_s", "tenant_cadence_s"},
      str(list(inspect.signature(Sampler.__init__).parameters)))
check("  and a Sample carries no per-tenant server reading",
      not any(f in Sample.__dataclass_fields__ for f in ("tenants", "engine_reads")),
      str(list(Sample.__dataclass_fields__)))
check("  and the engine adapter has no metric vocabulary to map",
      not any(hasattr(_engines.Engine, n)
              for n in ("gauges", "counters", "metrics_path", "metric_name", "scraped",
                        "cache_info_families", "latency_histograms"))
      and not any(hasattr(_engines, n) for n in ("first_present", "spellings", "KV_USAGE",
                                                 "WAITING", "RUNNING", "PREEMPTIONS")))
check("  and there is no tenant scrape left to call",
      not hasattr(_scrape, "scrape_tenant"))


# Readiness is the one place a tenant's `/metrics` was still fetched, as a proxy for the scrape that
# followed. With no scrape, requiring it would refuse to start a run over an endpoint nothing reads --
# so a tenant whose `/metrics` is unreachable, or absent entirely, must come up exactly as before.
# Driven against a stub client that fails anything but the health path, because the property is about
# which requests are made and asserting on a signature would not catch a second one being added back.
async def _drive_ready() -> tuple:
    from harness.__main__ import check_ready

    asked = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {}

    class _Client:
        async def get(self, url, timeout=None):
            asked.append(url)
            if not url.endswith("/health"):
                raise ConnectionError(f"nothing serves {url}")
            return _Resp()

    cfg = RunConfig(run_id="ready", validity_key_hash="k", tenants=[
        TenantConfig(name="a", base_url="http://x", served_name="m", rate_hz=1.0,
                     isl_tokens=8, osl_tokens=4, gpu_uuids=["GPU-ready"])])
    return await check_ready(_Client(), cfg), asked


_ready_bad, _ready_asked = asyncio.run(_drive_ready())
check("a tenant whose /metrics is unreachable is still Ready, because nothing reads it",
      _ready_bad == [], str(_ready_bad))
check("  and exactly one request was made per tenant, to the engine's health path",
      _ready_asked == ["http://x/health"], str(_ready_asked))

# The fast loop must reach the store's MIN_SUBSAMPLES of 2 on its own, with no network in it at
# all. Driven live rather than by hand-filling `_bucket`, because the property is a timing one.
async def _drive_fast_loop():
    smp = Sampler(None, {"a": _Drv()}, cadence_s=0.25, tenant_cadence_s=0.01)
    task = asyncio.create_task(smp.run(None))
    await asyncio.sleep(0.6)
    smp.stop()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
    return smp


_fast = asyncio.run(_drive_fast_loop())
check("the driver loop fills its bucket with no endpoint reachable anywhere",
      sum(x.subsamples for x in _fast.samples) >= 20,
      f"samples={[x.subsamples for x in _fast.samples]}")
check("  and every closed sample clears the store's MIN_SUBSAMPLES of 2, so no row can be "
      "disqualified for a server's silence",
      all(x.subsamples >= 2 for x in _fast.samples),
      f"{[x.subsamples for x in _fast.samples]}")

# --- the backlog direction: the replacement for the engine's queue gauge ----------------------
# `too_narrow_reason` needs to know whether the server fell behind, and used to ask the engine.
# It now reads the direction of this series, which the steady-state gate already reads for
# flatness. Same numbers, two questions (impl §14.6).
def _sm(t, v):
    return Sample(t=t, in_flight={"a": v})


_rising = [_sm(float(i), float(i)) for i in range(20)]
_bl = backlog_series(_rising, "a", 0.0, 100.0)
check("a rising in-flight series reports a last quarter above its first",
      _bl["in_flight_last_quarter_mean"] > _bl["in_flight_first_quarter_mean"], str(_bl))
_flat = [_sm(float(i), 4.0) for i in range(20)]
_bf = backlog_series(_flat, "a", 0.0, 100.0)
check("a flat one reports them equal, so flat-but-slow cannot widen a degree",
      _bf["in_flight_last_quarter_mean"] == _bf["in_flight_first_quarter_mean"] == 4.0, str(_bf))
check("  and both carry the sample count, so 'one quarter each' is checkable",
      _bl["in_flight_samples"] == _bf["in_flight_samples"] == 20)
check("a window with no samples in it yields no backlog block at all, not zeros",
      backlog_series(_flat, "a", 500.0, 600.0) == {})
check("a tenant absent from the series yields nothing rather than a zero level",
      backlog_series(_flat, "b", 0.0, 100.0) == {})
check("the window bounds are honoured", backlog_series(_rising, "a", 15.0, 100.0)
      ["in_flight_mean"] == sum(range(15, 20)) / 5)

# The phase rotation must visit every position, or a rate commensurate with the
# sub-cadence would re-create the alias the fix exists to remove.
seen = {(i * ROTATE_STEP) % PHASE_POSITIONS for i in range(PHASE_POSITIONS)}
check("phase rotation covers every position", seen == set(range(PHASE_POSITIONS)),
      f"{sorted(seen)}")
check("tenant cadence is a poor divisor of the usual request intervals",
      all(abs((iv / TENANT_CADENCE_S) - round(iv / TENANT_CADENCE_S)) > 0.15
          for iv in (0.05, 0.1, 0.5, 1.0)),
      f"cadence={TENANT_CADENCE_S}s")

# --- steady state ------------------------------------------------------------
# The gate reads one series two ways: a half-width on the LEVEL and a drift normalised by the
# ARRIVALS over the same span. Every fixture here therefore has to carry `launched` -- without it
# there is no offered load to normalise against, and a series with no arrivals is refused by name
# rather than read as flat (which is the whole point of the block further down).
th = StabilityThresholds()
flat = []
for i in range(60):
    s = Sample(t=float(i) * 5.0, subsamples=36)
    s.in_flight["a"] = 4
    s.delivered["a"] = i * 100
    s.launched["a"] = i * 4          # arrival and completion in step: in-flight holds at 4
    flat.append(s)
settled, detail = steady_state_check(flat, th)
check("a flat series settles", settled, str(detail.get("signals_ok")))
check("  and it is flat in the sense the gate claims: no drift against the offered load",
      abs(detail["in_flight_drift_frac"]) < 1e-9 and detail["arrivals_judged"] > 0,
      str({k: detail[k] for k in ("in_flight_rel", "in_flight_drift_frac", "arrivals_judged")}))

short = flat[:5]
settled, detail = steady_state_check(short, th)
check("too few samples never settles", not settled, detail.get("reason", ""))

drifting = []
for i in range(60):
    s = Sample(t=float(i) * 5.0, subsamples=36)
    s.in_flight["a"] = i             # backlog growing without bound
    s.delivered["a"] = i * 100
    s.launched["a"] = i * 4          # arriving 4x faster than the queue is clearing
    drifting.append(s)
settled, detail = steady_state_check(drifting, th)
check("a growing backlog does not settle", not settled,
      str(detail.get("signals_ok")))
check("  and the record names which signal moved",
      detail.get("signals_ok", {}).get("in_flight") is False)

# The case the level test cannot see, and the reason the drift term exists. A half-width is a
# statement about a LEVEL; "flat" is a statement about a derivative. For a monotonic rise of total
# size R, k=10 batches at z=1.96 give hw ~= 0.2085 R, so a ramp clears a 0.15 gate whenever
# R <= 0.72 * mean -- and because the tolerance is relative to the mean, the absolute drift waved
# through GROWS with the in-flight level, which is backwards for a runaway queue. Measured on
# 2026-09-21: `qwen35-multimodal-2`'s `docqa` at 0.5 Hz ran to in-flight 117.6 and was declared
# settled at 330s with 0 of 30 requests completed, three times over. Reproduced here exactly: 0.5 Hz
# is 2.5 arrivals per 5s bucket, nothing completes, so in-flight rises 2.5 per sample off a base
# high enough to hide the ramp inside the half-width.
_stall = []
for i in range(40):
    s = Sample(t=float(i) * 5.0, subsamples=36)
    s.in_flight["a"] = 104.0 + 2.5 * i
    s.delivered["a"] = 6400          # cumulative and NOT moving: nothing is coming back
    s.launched["a"] = 2.5 * i
    _stall.append(s)
_st_settled, _st_detail = steady_state_check(_stall, th)
check("a queue running away at a high level does NOT settle, which is the defect this term fixes",
      not _st_settled, str(_st_detail.get("signals_ok")))
check("  and the level's half-width on its own would have waved it through",
      _st_detail["in_flight_rel"] <= th.in_flight_rel,
      f"in_flight_rel={_st_detail['in_flight_rel']:.3f} against {th.in_flight_rel}")
check("  so it is the drift that caught it, and the drift reads as a completion deficit: nothing "
      "completed, so the whole of the offered load accumulated",
      abs(_st_detail["in_flight_drift_frac"] - 1.0) < 0.02,
      f"drift={_st_detail['in_flight_drift_frac']:.3f}")
check("  and the reason says completion is not keeping pace with arrival, not 'it is noisy'",
      "not keeping pace with arrival" in _st_detail["reason"], _st_detail["reason"])
# And the term does not scale with the level, which is the property the half-width lacked: the same
# stall parked an order of magnitude higher is caught just as hard.
_st_hi = []
for i in range(40):
    s = Sample(t=float(i) * 5.0, subsamples=36)
    s.in_flight["a"] = 1040.0 + 2.5 * i
    s.delivered["a"] = 6400
    s.launched["a"] = 2.5 * i
    _st_hi.append(s)
_hi_settled, _hi_detail = steady_state_check(_st_hi, th)
check("  and raising the level tenfold does not buy the same stall a pass, as it did before",
      not _hi_settled
      and abs(_hi_detail["in_flight_drift_frac"] - _st_detail["in_flight_drift_frac"]) < 1e-9
      and _hi_detail["in_flight_rel"] < _st_detail["in_flight_rel"],
      f"rel {_st_detail['in_flight_rel']:.3f} -> {_hi_detail['in_flight_rel']:.3f}, "
      f"drift {_hi_detail['in_flight_drift_frac']:.3f} unchanged")

# Flatness with no arrivals at all is vacuous in the same way a series pinned at zero is: a driver
# that issued nothing over 150s is not a settled fleet, and there is nothing to normalise a drift
# against. Refused by name rather than defaulted -- vouching for it would be vouching for the
# absence of the evidence.
_noarr = [Sample(t=float(i) * 5.0, subsamples=36) for i in range(60)]
for i, s in enumerate(_noarr):
    s.in_flight["a"] = 4
    s.delivered["a"] = i * 100
    s.launched["a"] = 7              # constant: perfectly flat, and nothing arrived
_na_settled, _na_detail = steady_state_check(_noarr, th)
check("a flat series with no arrivals in it is refused rather than read as settled",
      not _na_settled and "no request arrived" in _na_detail["reason"],
      _na_detail.get("reason", ""))
check("  and it records a null drift rather than a zero one, since zero would read as a pass",
      _na_detail["in_flight_drift_frac"] is None and _na_detail["arrivals_judged"] == 0.0)


# --- the warm-up, which is not the run ---------------------------------------
# A slow model delivers nothing for its first service time: 1024 output tokens at
# 55 ms is ~56s, or ~11 samples at a 5s cadence. Those samples say the fleet had not
# started, not that it was unsettled, and judging them alongside the rest is what made
# GLM-4.5-Air at tp4 come back `infeasible_no_steady_state` on a still machine.
def _series(n: int, dead: int, per_sample: int = 100, in_flight=None) -> list[Sample]:
    out, delivered = [], 0
    for i in range(n):
        s = Sample(t=float(i) * 5.0, subsamples=36)
        # In-flight climbs while nothing is coming back, then holds: Little's law, with the
        # plateau arriving as completions do.
        s.in_flight["a"] = ((i if i < dead else 4) if in_flight is None else in_flight(i))
        if i >= dead:
            delivered += per_sample
        s.delivered["a"] = delivered
        # Arrivals run at a steady rate throughout, warm-up included: the schedule is open-loop, so
        # the driver keeps issuing whether or not anything has come back yet (`load.py`). This is
        # what the drift term normalises against, so a fixture without it would be refused for want
        # of an offered load rather than judged on the thing it was built to test.
        s.launched["a"] = 2 * i
        out.append(s)
    return out


_warm = _series(60, dead=11)
check("the warm-up is counted from the data, not from a configured number of seconds",
      warmup_samples(_warm) == 11, "11 samples went by with nothing delivered")
settled, detail = steady_state_check(_warm, th)
check("a still fleet that spent its first minute delivering nothing DOES settle", settled,
      str(detail.get("signals_ok") or detail.get("reason")))
check("  and the record says how many samples were discarded and how many were judged",
      (detail["warmup_samples_discarded"], detail["samples_judged"], detail["samples_total"])
      == (11, 49, 60), str(detail))
# Not a trailing window. The original reason was `delivered_rate`'s per-completion lumpiness,
# which needed ~100 samples to average out; that signal is gone, and this behaviour is kept
# deliberately rather than by inheritance (`controller.StabilityThresholds`).
check("  and the whole post-warm-up series is judged, not a recent slice of it",
      steady_state_check(_series(200, dead=11), th)[1]["samples_judged"] == 189)

# The other half of the same rule: a mid-run dropout is a bucket that collected no readings, so
# `Sampler._close_bucket` fills it with zeros -- zero in-flight AND zero delivered, since it has
# nothing else to give. Treating it as warm-up would discard every sample before it, so only the
# leading run counts and the dropout stays in the window, where it must cost the run its verdict.
# In-flight's flatness does NOT catch it, which is the point of checking it by name: one zero among
# sixty samples at a level of 4 leaves a relative half-width near 0.04, well inside the gate.
_dropout = _series(60, dead=0)
_dropout[40].delivered["a"] = 0
_dropout[40].in_flight["a"] = 0.0
_dropout[40].subsamples = 0
check("a zero in the middle of a run is not warm-up", warmup_samples(_dropout) == 0)
check("  and it costs the run its verdict rather than being discarded",
      not steady_state_check(_dropout, th)[0])
check("  and the reason names a gap in the series rather than an unsettled signal",
      "collected no readings" in steady_state_check(_dropout, th)[1]["reason"],
      steady_state_check(_dropout, th)[1].get("reason", ""))
check("  which the surviving stability signal on its own would have waved through",
      relative_halfwidth([float(s.in_flight["a"]) for s in _dropout]) <= th.in_flight_rel,
      f"in_flight_rel={relative_halfwidth([float(s.in_flight['a']) for s in _dropout]):.3f}")

_never = _series(40, dead=40)
settled, detail = steady_state_check(_never, th)
check("a fleet that has delivered nothing at all cannot be settled, and is told apart from "
      "a fleet that is merely early", not settled and "completed a request" in detail["reason"],
      detail["reason"])
check("  and one that has delivered only briefly says so in those terms",
      "since delivery began" in steady_state_check(_series(40, dead=20), th)[1]["reason"])

# The discard must not become an excuse to ignore a backlog: a fleet whose in-flight count
# climbs from the moment it starts delivering is not settled, warm-up or no warm-up.
_climbing = _series(60, dead=11, in_flight=lambda i: max(1, i))
check("a backlog that grows after the warm-up still fails, and names in-flight",
      not steady_state_check(_climbing, th)[0]
      and steady_state_check(_climbing, th)[1]["signals_ok"]["in_flight"] is False)

# --- record assembly + UUID join --------------------------------------------
UUID_A = "GPU-aaaa"
UUID_OTHER = "GPU-zzzz"          # idle GPU claimed by no tenant


class FakeDriver:
    def __init__(self, results, dispatched, launched=None):
        self.results = results
        self._d = dispatched
        # Defaults to `dispatched` so the existing cases read unchanged, and is settable because
        # the two diverge in the one case that matters: under overload nothing comes back, so
        # `dispatched_in` reports 0 while the driver did send requests.
        self._l = dispatched if launched is None else launched
        self.hit_cap = False

    def completed_in(self, a, b):
        return [r for r in self.results if r.ok and a <= r.dispatch_t < b]

    def dispatched_in(self, a, b):
        return self._d

    def launched_in(self, a, b):
        return self._l


def make_cfg(threshold=0.99):
    t = TenantConfig(name="a", base_url="http://x", served_name="m", rate_hz=1.0,
                     isl_tokens=512, osl_tokens=128,
                     gpu_uuids=[UUID_A], tp=1, envelope_gib=8.0,
                     delivered_fraction_threshold=threshold)
    return RunConfig(run_id="selftest", tenants=[t],
                     validity_key_hash="selftest", exporter_url="http://y")


samples = []
for i in range(30):
    s = Sample(t=1.0 + i * 5.0, subsamples=36)
    s.in_flight["a"] = 3.0
    s.delivered["a"] = i * 100
    s.gpus = {UUID_A: {"DCGM_FI_PROF_SM_ACTIVE": 0.62,
                       "DCGM_FI_DEV_POWER_VIOLATION": 0.0},
              UUID_OTHER: {"DCGM_FI_PROF_SM_ACTIVE": 0.0}}
    samples.append(s)
sampler = types.SimpleNamespace(samples=samples, cadence_s=5.0,
                                tenant_cadence_s=TENANT_CADENCE_S,
                                subsample_lag_max_s=0.004)

# Good run.
results = [rr(128, t=1.0 + i * 0.5) for i in range(200)]
drivers = {"a": FakeDriver(results, 200)}
rec = build_record(make_cfg(), drivers, sampler, 0.0, 200.0, verdict=VERDICT_MEASURED,
                   steady_state_s=47.2, steady_state_detail={}, window_detail={})
a = rec["per_tenant"]["a"]
check("full delivery keeps verdict=measured", rec["verdict"] == VERDICT_MEASURED)
check("latency emitted when delivery passes", "itl_mean_ms" in a,
      f"itl_mean_ms={a.get('itl_mean_ms')}")
# `preemptions` used to be checked here, as a window delta rather than vLLM's cumulative counter --
# a real bug, fixed, and then the field it was computed from went away with the scrape (impl §14.6).
# What the record carries instead is the direction of the driver's own in-flight level, and a flat
# one is the thing a settled window is *supposed* to show.
check("the record carries the in-flight level's direction, not a server's queue depth",
      a.get("in_flight_samples") == 30 and a["in_flight_mean"] == 3.0
      and a["in_flight_last_quarter_mean"] == a["in_flight_first_quarter_mean"],
      str({k: v for k, v in a.items() if k.startswith("in_flight")}))
check("UUID join keeps the claimed GPU", UUID_A in rec["per_gpu"])
check("UUID join drops the unclaimed GPU", UUID_OTHER not in rec["per_gpu"],
      f"per_gpu keys={list(rec['per_gpu'])}")
check("throttle field present even at zero",
      rec["per_gpu"][UUID_A].get("power_violation_max") == 0.0)
check("sampling provenance recorded so a phase-locked record is identifiable",
      rec["sampling"]["subsamples_per_sample_mean"] == 36
      and rec["sampling"]["tenant_cadence_s"] == TENANT_CADENCE_S,
      str(rec["sampling"]))

# Under-delivering run: half the dispatched requests never returned.
drivers = {"a": FakeDriver(results, 400)}
rec = build_record(make_cfg(), drivers, sampler, 0.0, 200.0, verdict=VERDICT_MEASURED,
                   steady_state_s=47.2, steady_state_detail={}, window_detail={})
a = rec["per_tenant"]["a"]
check("delivery gate overrides a nominally measured verdict",
      rec["verdict"] == VERDICT_DELIVERY, rec["verdict"])
check("no latency emitted when delivery fails", "itl_mean_ms" not in a)
check("  and the suppression is stated in the record", "latency_suppressed" in a)
# The cause signals must survive a failed delivery gate: a row that cannot be cited about latency is
# exactly the row the walk asks "was the degree too narrow" about, and it answers from these.
check("cause signals still emitted when delivery fails", "in_flight_mean" in a,
      str(sorted(k for k in a if k.startswith("in_flight"))))
check("delivery gate failure names the tenant", rec["delivery_gate_failed"] == ["a"])

# A bad verdict string must not reach the store.
try:
    build_record(make_cfg(), drivers, sampler, 0.0, 1.0, verdict="probably_fine",
                 steady_state_s=None, steady_state_detail={}, window_detail={})
    check("unknown verdict rejected", False)
except ValueError:
    check("unknown verdict rejected", True)

# --- workload (approximate path; no tokenizer offline) -----------------------
from harness.workload import Workload  # noqa: E402

w = Workload(64, 128, None)
p = w.payload("m")
check("OSL pinned by min_tokens + ignore_eos",
      p["min_tokens"] == 128 and p["max_tokens"] == 128 and p["ignore_eos"] is True)
check("streaming on -- ITL is impossible without it", p["stream"] is True)
# Without this the server sends no usage chunk, the delivered-fraction gate silently
# falls back to client chunk counts, and the whole token-count fix is inert.
check("usage requested -- the gate depends on the server's own token count",
      p.get("stream_options", {}).get("include_usage") is True)
check("no tokenizer is recorded, not hidden", w.describe()["isl_exact"] is False)

# --- per-tenant sequence shape (impl §6.2) -----------------------------------
from harness.__main__ import build_workloads  # noqa: E402


def _tenant(name, isl, osl, rate=1.0, tokenizer=None):
    return TenantConfig(name=name, base_url="http://x", served_name=f"m-{name}",
                        rate_hz=rate, isl_tokens=isl, osl_tokens=osl, gpu_uuids=[UUID_A],
                        tokenizer_path=tokenizer)


mixed = RunConfig(run_id="mixed", validity_key_hash="selftest",
                  tenants=[_tenant("long", 2048, 64), _tenant("chat", 128, 512),
                           _tenant("chat2", 128, 512)])
ws = build_workloads(mixed)
check("each tenant gets a workload of its own shape",
      (ws["long"].isl_tokens, ws["long"].osl_tokens) == (2048, 64)
      and (ws["chat"].isl_tokens, ws["chat"].osl_tokens) == (128, 512))
check("tenants that agree on shape share one prompt", ws["chat"] is ws["chat2"])
check("tenants that differ do not", ws["long"] is not ws["chat"])

# The tokenizer is part of what identifies a prompt, not a run-wide constant. Two tenants asking
# for the same shape with different tokenizers are asking for different prompts: 128 tokens of
# one vocabulary is not 128 tokens of another. Sharing them would make ISL exact for one tenant
# and approximate for the other while the record showed the same requested number for both.
#
# `Workload` is stubbed for the keying check because a real one loads the tokenizer from disk, and
# a path that resolves offline would defeat the point of naming two different ones. What is under
# test here is which prompts get built, not how one is built -- that is covered above.
_forked = RunConfig(run_id="forked", validity_key_hash="selftest",
                    tenants=[_tenant("a", 128, 512, tokenizer="/models/qwen"),
                             _tenant("b", 128, 512, tokenizer="/models/llama"),
                             _tenant("c", 128, 512, tokenizer="/models/qwen")])
import harness.__main__ as _hmain  # noqa: E402

_built: list = []


class _StubWorkload:
    def __init__(self, isl, osl, tokenizer_path, image_count=0, image_pixels=0, engine=None):
        self.tokenizer_path = tokenizer_path
        _built.append((isl, osl, tokenizer_path, image_count, image_pixels))

    def describe(self):
        return {"tokenizer_path": self.tokenizer_path}


_real_workload, _hmain.Workload = _hmain.Workload, _StubWorkload
try:
    fw = build_workloads(_forked)
finally:
    _hmain.Workload = _real_workload
check("a tenant's tokenizer is part of what identifies its prompt, not a run-wide constant",
      fw["a"] is not fw["b"] and fw["a"] is fw["c"])
check("  and only one prompt is built per (shape, tokenizer), not one per tenant",
      _built == [(128, 512, "/models/qwen", 0, 0), (128, 512, "/models/llama", 0, 0)],
      str(_built))
check("  and a workload reports the tokenizer it was built with, so the record can name it",
      Workload(8, 4, None).describe()["tokenizer_path"] is None)
check("the payload pins each tenant's own OSL",
      ws["long"].payload("m")["min_tokens"] == 64
      and ws["chat"].payload("m")["min_tokens"] == 512)

# The gate is per tenant. Each tenant here delivers exactly its own OSL on every
# request, so every delivered fraction must read 1.0. Under one fleet-wide OSL of 64
# the same run would read 8.0 for `chat` and, at 512, 0.125 for `long` -- one tenant
# apparently over-delivering and the other failing the gate, from a healthy run.
mixed_rec = build_record(
    mixed,
    {"long": FakeDriver([rr(64, t=1.0 + i) for i in range(10)], 10),
     "chat": FakeDriver([rr(512, t=1.0 + i) for i in range(10)], 10),
     "chat2": FakeDriver([rr(512, t=1.0 + i) for i in range(10)], 10)},
    sampler, 0.0, 200.0, verdict=VERDICT_MEASURED, steady_state_s=1.0,
    steady_state_detail={}, window_detail={})
check("the delivery gate uses each tenant's own OSL",
      all(mixed_rec["per_tenant"][n]["delivered_fraction"] == 1.0
          for n in ("long", "chat", "chat2")),
      str({n: r.get("delivered_fraction") for n, r in mixed_rec["per_tenant"].items()}))
check("  and a healthy mixed fleet is not read as a delivery failure",
      mixed_rec["verdict"] == VERDICT_MEASURED and mixed_rec["delivery_gate_failed"] == [])
check("  and each tenant's expected token count is its own",
      [mixed_rec["per_tenant"][n]["tokens_expected"] for n in ("long", "chat")] == [64, 512])
# The image payload is part of the shape, not an extra: the agent's validity key reads this
# block back as a 4-tuple, so a record that named only the token pair would be filed under a
# key it does not describe.
check("the record carries each tenant's shape in configuration, payload included",
      {t["name"]: (t["isl_tokens"], t["osl_tokens"], t["image_count"], t["image_pixels"])
       for t in mixed_rec["configuration"]["tenants"]}
      == {"long": (2048, 64, 0, 0), "chat": (128, 512, 0, 0), "chat2": (128, 512, 0, 0)},
      str(mixed_rec["configuration"]["tenants"]))
check("the workload block is keyed by tenant, not fleet-wide",
      set(mixed_rec["workload"]["tenants"]) == {"long", "chat", "chat2"}
      and "isl_tokens" not in mixed_rec["workload"])
# The tokenizer too, and it is what makes the ISL above readable: the same requested 128 means a
# different prompt per tenant unless the tokenizer is named beside it.
_forked_rec = build_record(
    _forked, {n: FakeDriver([rr(512, t=1.0 + i) for i in range(10)], 10)
              for n in ("a", "b", "c")},
    sampler, 0.0, 200.0, verdict=VERDICT_MEASURED, steady_state_s=1.0,
    steady_state_detail={}, window_detail={})
check("  and it names each tenant's tokenizer, with no fleet-wide one beside it",
      {n: v["tokenizer_path"] for n, v in _forked_rec["workload"]["tenants"].items()}
      == {"a": "/models/qwen", "b": "/models/llama", "c": "/models/qwen"}
      and "tokenizer_path" not in _forked_rec["workload"])

# impl §16: an SM cap is part of the placement, so the record carries it -- and only on a tenant
# that had one, so an uncapped record reads exactly as one written before the field existed.
_capped = RunConfig(run_id="capped", validity_key_hash="selftest",
                    tenants=[TenantConfig(**{**vars(_tenant("a", 128, 512)), "active_thread_percentage": 25}),
                             _tenant("b", 128, 512)])
_capped_rec = build_record(
    _capped, {n: FakeDriver([rr(512, t=1.0 + i) for i in range(10)], 10) for n in ("a", "b")},
    sampler, 0.0, 200.0, verdict=VERDICT_MEASURED, steady_state_s=1.0,
    steady_state_detail={}, window_detail={})
_cap_cfg = {t["name"]: t for t in _capped_rec["configuration"]["tenants"]}
check("a capped tenant's SM cap is in the record's configuration, and an uncapped one has no key",
      _cap_cfg["a"].get("active_thread_percentage") == 25
      and "active_thread_percentage" not in _cap_cfg["b"], str(_cap_cfg))

# A run config from before the shape moved is a config for a different experiment.
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    legacy = Path(d) / "legacy.yaml"
    legacy.write_text(
        "run_id: old\nisl_tokens: 512\nosl_tokens: 128\n"
        "tenants:\n  - {name: a, base_url: 'http://x', served_name: m, rate_hz: 1.0,\n"
        "     isl_tokens: 512, osl_tokens: 128}\n")
    try:
        RunConfig.load(str(legacy))
        check("a run-level isl_tokens is refused, not applied fleet-wide", False)
    except ValueError as exc:
        check("a run-level isl_tokens is refused, not applied fleet-wide",
              "per tenant" in str(exc).lower() or "onto each tenant" in str(exc))

    # And the same for the tokenizer, which is the field this change moved. Refused rather than
    # lifted onto every tenant: a config written when it was fleet-wide meant "build every prompt
    # with this one", and honouring that now would reintroduce exactly the fiction being removed.
    old_tok = Path(d) / "old_tok.yaml"
    old_tok.write_text(
        "run_id: old\ntokenizer_path: /models/qwen\n"
        "tenants:\n  - {name: a, base_url: 'http://x', served_name: m, rate_hz: 1.0,\n"
        "     isl_tokens: 512, osl_tokens: 128}\n")
    try:
        RunConfig.load(str(old_tok))
        check("a run-level tokenizer_path is refused, not applied fleet-wide", False)
    except ValueError as exc:
        check("a run-level tokenizer_path is refused, not applied fleet-wide",
              "on each tenant" in str(exc))

# --- input length is measured, not computed (workload.calibrate) -------------
# The rule the whole path exists for: the SERVER says what a request costs. A tokenizer can
# count text; nothing in this container can count what an image plus a chat template costs,
# and putting a copy of each model family's preprocessor in the load driver is the mistake
# being refused. So the loop below is the instrument, and these tests are its calibration.
from harness.workload import _MAX_PROBES  # noqa: E402


def _server(overhead):
    """A fake engine that charges `overhead` tokens on top of the text it was sent.

    `overhead` stands in for visual tokens plus the chat template -- the constant nothing in
    this container can compute, which is why it is absorbed rather than predicted. An int is
    a well-behaved server; a callable of the probe number is one whose cost will not settle.

    The text is counted by splitting on spaces, which is exactly right for the no-tokenizer
    path: `_build_text(n)` joins n words, so one added word is one added token, the same
    first-order relation the real loop relies on.
    """
    seen: list = []

    async def price(payload):
        n = overhead(len(seen)) if callable(overhead) else overhead
        seen.append(payload)
        text = payload.get("prompt")
        if text is None:
            text = [q for q in payload["messages"][0]["content"]
                    if q["type"] == "text"][0]["text"]
        return len(text.split()) + n
    return price, seen


_w = Workload(512, 128, None)
_price, _seen_probes = _server(3)
asyncio.run(_w.calibrate(_price, "m"))
check("the probe settles ISL on the server's own count, not the client's",
      _w.isl_exact and _w.isl_achieved == 512, f"achieved={_w.isl_achieved}")
check("  in two probes -- the correction is exact to first order",
      len(_w.probes) == 2, str(_w.probes))
check("  by shortening the text by exactly what the template cost", _w.text_tokens == 509)
check("  and every probe is in the record, so an unexpected cost is readable afterwards",
      [p["prompt_tokens"] for p in _w.probes] == [515, 512], str(_w.probes))
check("  and the probe priced the request that will be driven, not a different one",
      _w._probe_payload("m")["prompt"] == _w.payload("m")["prompt"]
      and _w._probe_payload("m")["max_tokens"] == 1
      and "stream" not in _w._probe_payload("m"))

# An unsettled ISL is recorded as unsettled. It must not be rounded off and carried into the
# run: an approximate input length makes prefill cost a hidden variable, which is the one
# thing this module exists to prevent.
_w2 = Workload(512, 128, None)
_price2, _ = _server(lambda i: 2 if i % 2 == 0 else 4)
asyncio.run(_w2.calibrate(_price2, "m"))
check("a cost that will not settle leaves isl_exact False rather than being rounded off",
      _w2.calibrated and not _w2.isl_exact and len(_w2.probes) == _MAX_PROBES,
      str([p["prompt_tokens"] for p in _w2.probes]))
check("  and the record still says what the server last charged",
      _w2.describe()["isl_tokens_achieved"] in (510, 514)
      and _w2.describe()["isl_exact"] is False)

# A server that reports no usage at all: unverified, and said so, rather than assumed good.
_w3 = Workload(512, 128, None)


async def _no_usage(payload):
    return None


asyncio.run(_w3.calibrate(_no_usage, "m"))
check("a server that reports no prompt_tokens leaves the ISL unverified, not assumed",
      _w3.calibrated and not _w3.isl_exact and _w3.isl_achieved == -1
      and _w3.probes[0]["prompt_tokens"] is None)

# --- the image path: /v1/chat/completions, because nothing else carries an image ---
_iw = Workload(4096, 64, None, image_count=3, image_pixels=65536)
check("a text tenant stays on completions, where nothing sits between prompt and tokens",
      _w.endpoint == "/v1/completions")
check("an image tenant is driven through chat, the only route that carries an image",
      _iw.endpoint == "/v1/chat/completions")
_p = _iw.payload("m", 0)
check("the image request carries no prompt field and exactly one message",
      "prompt" not in _p and len(_p["messages"]) == 1)
_parts = _p["messages"][0]["content"]
check("  carrying the stated number of images, then the text",
      [q["type"] for q in _parts] == ["image_url"] * 3 + ["text"],
      str([q["type"] for q in _parts]))
check("  and the OSL pin and the streaming options survive the change of endpoint",
      _p["min_tokens"] == 64 and _p["max_tokens"] == 64 and _p["ignore_eos"] is True
      and _p["stream"] is True and _p["stream_options"]["include_usage"] is True)
check("  and the probe prices that same request, one token out",
      _iw._probe_payload("m")["messages"] == _p["messages"]
      and _iw._probe_payload("m")["max_tokens"] == 1)

_urls0 = [q["image_url"]["url"] for q in _parts if q["type"] == "image_url"]
check("a data URL is what a vLLM chat request accepts",
      _urls0[0].startswith("data:image/png;base64,"))
# Both halves of why the pool is sized `image_count + _POOL`. Either kind of repeat lets an
# engine that hashes multimodal input serve the vision tower's work out of a cache, and then
# the record says images were sent while the tower did nothing.
check("no request repeats an image within itself", len(set(_urls0)) == 3, str(len(set(_urls0))))
_urls1 = [q["image_url"]["url"] for q in _iw.payload("m", 1)["messages"][0]["content"]
          if q["type"] == "image_url"]
check("  and successive requests do not send the same set either",
      set(_urls0).isdisjoint(_urls1))
check("  which is what the pool being bigger than one request's demand buys",
      len(_iw.images.variants) == 3 + 8)
check("images are square, and at or under the stated pixels -- never over, which is the side "
      "that would trip max_pixels and make the engine resize",
      _iw.images.side == 256 and _iw.images.actual_pixels == 65536)
check("every variant is a different image at an identical cost",
      len(set(_iw.images.variants)) == len(_iw.images.variants))
check("the record says what was rendered, not only what was asked for",
      _iw.describe()["image_pixels_rendered"] == 65536
      and _iw.describe()["image_count"] == 3
      and _iw.describe()["endpoint"] == "/v1/chat/completions", str(_iw.describe()))

try:
    Workload(4096, 64, None, image_count=1, image_pixels=0)
    check("an image with no pixel count is refused -- the pixels ARE the specification", False)
except ValueError as exc:
    check("an image with no pixel count is refused -- the pixels ARE the specification",
          "image_pixels" in str(exc))

# A tenant whose images already cost more than its stated ISL is asking for a shape that does
# not exist. Driving the nearest one that does would measure a different workload under the
# stated one's name, so this raises instead of clamping.
_tight = Workload(8, 16, None, image_count=1, image_pixels=1024)


async def _dear(payload):
    return 5000


try:
    asyncio.run(_tight.calibrate(_dear, "m"))
    check("images costing more than the stated ISL is refused, not clamped", False)
except ValueError as exc:
    check("images costing more than the stated ISL is refused, not clamped",
          "no room for text" in str(exc), str(exc))

# The chat endpoint puts the token somewhere else. Counting only `choices[0].text` made every
# multimodal request look like it delivered nothing -- which reads as a delivery failure
# rather than as a parser that was never taught the other shape.
from harness.load import LoadDriver, _chunk_carries_text  # noqa: E402

check("a chat chunk's token lives in delta.content, and a completions chunk's in text",
      _chunk_carries_text({"choices": [{"delta": {"content": "x"}}]})
      and _chunk_carries_text({"choices": [{"text": "x"}]})
      and not _chunk_carries_text({"choices": [{"delta": {}}]})
      and not _chunk_carries_text({"choices": []}))

# --- the payload is a factory, and the driver passes the index through --------
_issued: list = []


async def _fake_issue(client, url, payload, tenant, scheduled_t):
    _issued.append(payload)
    return RequestResult(tenant=tenant, dispatch_t=scheduled_t, scheduled_t=scheduled_t,
                         ok=True, chunks=1, usage_tokens=1)


async def _drive_three():
    d = LoadDriver("img", "http://x/v1/chat/completions",
                   lambda i: _iw.payload("m", i), 1000.0, max_requests=3)
    await d.run(None)
    await asyncio.gather(*d._tasks)
    return d


import harness.load as _load  # noqa: E402

_real_issue, _load.issue = _load.issue, _fake_issue
try:
    _drv = asyncio.run(_drive_three())
finally:
    _load.issue = _real_issue
check("the driver builds each body from the request index rather than reusing one",
      len(_issued) == 3 and _drv.issued == 3, f"{len(_issued)} bodies, {_drv.issued} issued")
_sets = [tuple(q["image_url"]["url"] for q in p["messages"][0]["content"]
               if q["type"] == "image_url") for p in _issued]
check("  so successive requests differ in their image bytes while costing identical work",
      len(set(_sets)) == 3 and {len(s) for s in _sets} == {3}, str([len(s) for s in _sets]))

# `dispatched_in` walks the task list and skips anything not `done()`, so a window in which every
# request was still streaming when the drain timed out reports ZERO requests dispatched -- in the
# one situation where the count decides something. That is also the null denominator behind a null
# `delivered_fraction`. It is not a number to fix in place: the delivery gate's denominator is
# deliberately "requests we hold a result for". `launched_in` is the driver's own books, and it is
# what an overload argument rests on (`SoloProfile.too_narrow_reason`).


async def _drive_hanging():
    """Three requests that never return, as a saturated server produces."""
    async def _never(client, url, payload, tenant, scheduled_t):
        await asyncio.sleep(3600)

    d = LoadDriver("hang", "http://x/v1/completions", lambda i: {"prompt": "x"}, 1000.0,
                   max_requests=3)
    _load.issue = _never
    try:
        await d.run(None)
        await asyncio.sleep(0)                  # let the tasks start, none of them finish
        counts = (d.launched_in(0.0, 1e9), d.dispatched_in(0.0, 1e9), d.in_flight())
    finally:
        for t in d._tasks:
            t.cancel()
        _load.issue = _real_issue
    return counts


_hang = asyncio.run(_drive_hanging())
check("a window where nothing came back counts what the driver SENT, not what it heard back",
      _hang == (3, 0, 3), str(_hang))
check("  which is the difference between an overloaded server and a driver that sent nothing -- "
      "they need opposite responses and dispatched_in reads 0 for both", _hang[1] == _hang[0] - 3)

# --- calibration is per distinct workload, and blocks the run when it fails ---
_cal = RunConfig(run_id="cal", validity_key_hash="selftest",
                 tenants=[_tenant("a", 512, 128), _tenant("b", 512, 128),
                          _tenant("c", 256, 128)])
_cal_w = build_workloads(_cal)
_probe_urls: list = []


def _patched_price(overhead):
    inner, _ = _server(overhead)

    async def price_prompt(client, url, payload):
        _probe_urls.append(url)
        return await inner(payload)
    return price_prompt


_real_price, _hmain.price_prompt = _hmain.price_prompt, _patched_price(3)
try:
    _bad = asyncio.run(_hmain.calibrate_workloads(None, _cal, _cal_w))
finally:
    _hmain.price_prompt = _real_price
check("every tenant's ISL is settled before any load exists to perturb it", _bad == [], str(_bad))
check("  once per distinct workload, not once per tenant",
      len(_probe_urls) == 4, f"{len(_probe_urls)} probes for 3 tenants / 2 shapes")
check("  against the tenant's own endpoint", set(_probe_urls) == {"http://x/v1/completions"},
      str(set(_probe_urls)))
check("  and both shapes came out exact on the server's count",
      (_cal_w["a"].isl_achieved, _cal_w["c"].isl_achieved) == (512, 256))

# The two ways calibration fails are both run-blocking, and neither is a crash: the agent's
# vocabulary is four verdicts, and what has happened is that this placement cannot be driven
# at the shape that was asked for.
_cal2 = RunConfig(run_id="cal2", validity_key_hash="selftest", tenants=[_tenant("a", 512, 128)])
_real_price, _hmain.price_prompt = _hmain.price_prompt, _patched_price(
    lambda i: 2 if i % 2 == 0 else 4)
try:
    _bad2 = asyncio.run(_hmain.calibrate_workloads(None, _cal2, build_workloads(_cal2)))
finally:
    _hmain.price_prompt = _real_price
check("an ISL that will not settle blocks the run and names the tenant and the shortfall",
      len(_bad2) == 1 and _bad2[0].startswith("a: isl_tokens=512 not achieved"), str(_bad2))

_real_price, _hmain.price_prompt = _hmain.price_prompt, _patched_price(50_000)
try:
    _bad3 = asyncio.run(_hmain.calibrate_workloads(None, _cal2, build_workloads(_cal2)))
finally:
    _hmain.price_prompt = _real_price
check("  and an impossible shape is reported as a verdict, not raised through the run",
      len(_bad3) == 1 and "ValueError" in _bad3[0], str(_bad3))

# --- the image payload is part of what identifies a prompt -------------------
# Two tenants at one ISL, one sending a megapixel image and one sending none, are not the same
# request: the image one spends most of its prompt on visual tokens. Sharing them would
# calibrate once and report the result twice.
def _itenant(name, count, pixels):
    return TenantConfig(name=name, base_url="http://x", served_name=f"m-{name}", rate_hz=1.0,
                        isl_tokens=4096, osl_tokens=64, gpu_uuids=[UUID_A],
                        image_count=count, image_pixels=pixels)


_built.clear()
_pair = RunConfig(run_id="pair", validity_key_hash="selftest",
                  tenants=[_itenant("txt", 0, 0), _itenant("img", 1, 65536),
                           _itenant("img2", 1, 65536), _itenant("big", 1, 262144)])
_real_workload, _hmain.Workload = _hmain.Workload, _StubWorkload
try:
    _pw = build_workloads(_pair)
finally:
    _hmain.Workload = _real_workload
check("an image tenant does not share a prompt with a text tenant at the same shape",
      _pw["txt"] is not _pw["img"])
check("  nor with one sending the same count at a different resolution",
      _pw["img"] is not _pw["big"])
check("  and tenants whose payload agrees in every field do share one", _pw["img"] is _pw["img2"])
check("  so images are rendered once per distinct payload, not once per tenant",
      _built == [(4096, 64, None, 0, 0), (4096, 64, None, 1, 65536),
                 (4096, 64, None, 1, 262144)], str(_built))

# --- soak: an hour of the client's view, selected as a window is -------------
from harness import soak as _soak  # noqa: E402


async def _drive_soak():
    d = LoadDriver("s", "http://x/v1/completions", lambda i: {"prompt": "x"}, 1000.0,
                   max_requests=4)

    async def _two_ok(client, url, payload, tenant, scheduled_t):
        n = len(d._launched)
        if n > 2:
            return RequestResult(tenant=tenant, dispatch_t=scheduled_t, scheduled_t=scheduled_t,
                                 ok=False, error="HTTP 400")
        return RequestResult(tenant=tenant, dispatch_t=scheduled_t, scheduled_t=scheduled_t,
                             ok=True, usage_tokens=8, prompt_tokens=100, ttft_s=0.01 * n,
                             itl_mean_s=0.005)
    _load.issue = _two_ok
    try:
        await d.run(None)
        await asyncio.gather(*d._tasks)
    finally:
        _load.issue = _real_issue
    return _soak.hour_summary("s", d, 0.0, 1e12, 7200.0)


_hs = asyncio.run(_drive_soak())
check("a soak hour counts every request sent and summarizes only the ones that came back",
      (_hs["launched"], _hs["completed"], _hs["not_completed"]) == (4, 2, 2)
      and _hs["isl_tokens"] == 100 and _hs["osl_tokens"] == 8 and _hs["itl_mean_ms"] == 5.0
      and _hs["hour_start"] == "1970-01-01T02:00:00Z", str(_hs))
check("  and a p95 of two is the larger of them, never an interpolation past the data",
      _hs["ttft_p95_ms"] == 20.0, str(_hs))

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all self-tests passed")


# ---------------------------------------------------------------------------------------------
# The engine seam (impl §13). Everything below is about the *seam*, not about vLLM: the checks
# that matter are the ones a second engine has to keep passing, so each one is written against a
# throwaway engine registered here rather than against the shipped adapter.
# ---------------------------------------------------------------------------------------------
from harness import engines as _eng                                            # noqa: E402
from harness import scrape as _scrape                                          # noqa: E402


def _refused(fn) -> bool:
    """True if calling `fn` raises. For the checks whose subject is a refusal."""
    try:
        fn()
    except Exception:                                                          # noqa: BLE001
        return True
    return False


check("the harness resolves its engine by name and the shipped one is vLLM's OpenAI server",
      _eng.get(None).name == "vllm-openai" and _eng.get("vllm-openai") is _eng.get(None))
try:
    _eng.get("triton-trtllm")
    _unknown = None
except KeyError as _exc:
    _unknown = _exc.args[0]
check("  and a run config naming an engine this image has no adapter for is REFUSED, not "
      "defaulted",
      _unknown is not None and "vllm-openai" in _unknown and "fallback" in _unknown,
      (_unknown or "")[:70])

# A stub engine with no way to pin output -- the property a second engine is most likely to have,
# and it has to degrade rather than break. It used to publish no metrics either; that is no longer
# a property an engine can have, because no engine is asked (impl §14.6).
_BLIND = _eng.register(_eng.Engine(
    name="test-blind", health_path="/v2/health/ready",
    endpoint=lambda n: "/v2/generate", pin_output=lambda n: {}, pins_output=False,
    stream_fields=lambda: {"stream": True}))

# Absent-is-not-zero was enforced on two helpers that fed the steady-state gate, then on an engine's
# own metric mapping. Both are gone: there is no mapping, because there is no scrape. What is left to
# assert is that an adapter *cannot* carry one -- the check is in the structural block above, and the
# strong form (a complete judgement with an engine that publishes nothing) is at the end of this file.
check("an engine adapter is three questions about talking to a server, and none about reading it",
      set(_eng.Engine.__dataclass_fields__)
      == {"name", "health_path", "ready_ok", "endpoint", "pin_output", "pins_output",
          "stream_fields"},
      str(sorted(_eng.Engine.__dataclass_fields__)))

# Output pinning: the one thing this seam is not allowed to fake.
_pin = Workload(8, 4, None)
_nopin = Workload(8, 4, None, engine="test-blind")
check("vLLM pins output length, and the record says so",
      _pin.osl_pinned is True and _pin.describe()["osl_pinned"] is True
      and _pin.payload("m")["min_tokens"] == 4 and _pin.payload("m")["ignore_eos"] is True)
check("  while an engine that cannot pin it reports that, rather than a pinned-looking number",
      _nopin.osl_pinned is False and _nopin.describe()["osl_pinned"] is False
      and "min_tokens" not in _nopin.payload("m")
      and "ignore_eos" not in _nopin.payload("m"))
check("  and streaming is on either way -- ITL is impossible without it",
      _pin.payload("m")["stream"] is True and _nopin.payload("m")["stream"] is True)
check("the engine decides where a request goes, and the record names both",
      _nopin.endpoint == "/v2/generate" and _pin.endpoint == "/v1/completions"
      and _nopin.describe()["engine"] == "test-blind"
      and _pin.describe()["endpoint"] == "/v1/completions")

# The sharing key. Two tenants alike in every other way but served by different stacks must not
# share one workload: a shared one would build a single body and report it as both.
_a = TenantConfig(name="a", base_url="http://a", served_name="m-a", rate_hz=1.0,
                  isl_tokens=8, osl_tokens=4, gpu_uuids=[UUID_A])
_b = TenantConfig(name="b", base_url="http://b", served_name="m-b", rate_hz=1.0,
                  isl_tokens=8, osl_tokens=4, gpu_uuids=[UUID_A], engine="test-blind")
_mixed_eng = build_workloads(RunConfig(run_id="mixed-eng", validity_key_hash="selftest",
                                       tenants=[_a, _b]))
check("two tenants of identical shape on different engines do NOT share one workload",
      _mixed_eng["a"] is not _mixed_eng["b"]
      and _mixed_eng["a"].endpoint == "/v1/completions"
      and _mixed_eng["b"].endpoint == "/v2/generate")
check("  while the default is unchanged: a tenant that names no engine is on vLLM's server",
      _a.engine == "vllm-openai" and _mixed_eng["a"].engine.name == "vllm-openai")
check("a run config naming an engine with no adapter is refused while parsing, before any load",
      _refused(lambda: RunConfig(run_id="x", validity_key_hash="k", tenants=[
          TenantConfig(name="a", base_url="http://a", served_name="m", rate_hz=1.0,
                       isl_tokens=8, osl_tokens=4, engine="nope")]).validate()))

# Readiness follows the engine's own path, not vLLM's. There is no second path to check: the
# `/metrics` half of readiness went with the scrape it was a proxy for (impl §14.6).
check("readiness is read at the engine's own path",
      _BLIND.health_path == "/v2/health/ready" and _eng.get(None).health_path == "/health")
check("  and `scrape` offers no way to read a tenant at all",
      not hasattr(_scrape, "scrape_tenant")
      and [n for n in dir(_scrape) if n.startswith("TENANT")] == [],
      str([n for n in dir(_scrape) if not n.startswith("_")]))

# The second SHIPPED adapter. Everything above is written against a throwaway engine on purpose;
# these are about the one property the Triton adapter exists to have, which no stub can stand in for:
# the two shipped stacks must send the SAME bytes to the SAME path, or a difference measured between
# them is partly a difference between two protocols.
_TIS = "vllm-triton-inference-server"
_w_vllm = Workload(64, 16, None)
_w_tis = Workload(64, 16, None, engine=_TIS)
check("the two shipped stacks are driven identically: same path, same body",
      _w_tis.endpoint == _w_vllm.endpoint == "/v1/completions"
      and _w_tis.payload("m") == _w_vllm.payload("m"),
      _w_tis.endpoint)
check("  including the output pin, so OSL is an input on both and the record says so on both",
      _w_tis.osl_pinned is True and _w_tis.payload("m")["min_tokens"] == 16
      and _w_tis.payload("m")["ignore_eos"] is True)
check("  while readiness is at the frontend's own path, which is where the two do differ",
      _eng.get(_TIS).health_path == "/health/ready"
      and _eng.get(None).health_path == "/health")
# The backstop for the fleet loader's offline refusal. It has to raise while the workloads are being
# assembled, not at the first request: by then the node is leased and the weights are on the card.
_img_err = None
try:
    Workload(2048, 16, None, image_count=2, image_pixels=1024, engine=_TIS)
except ValueError as _exc:
    _img_err = str(_exc)
check("an image tenant on a stack that cannot serve images is refused as the workload is built",
      _img_err is not None and "cannot serve images" in _img_err and "vllm-openai" in _img_err,
      (_img_err or "")[:60])
check("  and the same tenant is built without complaint on the stack that can",
      Workload(2048, 16, None, image_count=2, image_pixels=1024).image_count == 2)
check("  which is a refusal and never a quiet reroute to a path that would collect 400s",
      _refused(lambda: _eng.get(_TIS).endpoint(1))
      and _eng.get(_TIS).endpoint(0) == "/v1/completions")

# The third, Dynamo + vLLM. The request is the siblings' again; what differs is readiness, because a
# Dynamo frontend is its own process and answers 200 on every path before its worker has registered.
_DYN = "dynamo-vllm"
_w_dyn = Workload(64, 16, None, engine=_DYN)
check("the Dynamo stack is driven identically too: same path, same body, same output pin",
      _w_dyn.endpoint == _w_vllm.endpoint and _w_dyn.payload("m") == _w_vllm.payload("m")
      and _w_dyn.osl_pinned is True)
check("  while readiness is a model LISTED at /v1/models, not a 200 from it",
      _eng.get(_DYN).health_path == "/v1/models"
      and _eng.get(_DYN).ready_ok({"object": "list", "data": [{"id": "m"}]})
      and not _eng.get(_DYN).ready_ok({"object": "list", "data": []})
      and not _eng.get(_DYN).ready_ok(None))
check("  and an image tenant is refused as the workload is built",
      _refused(lambda: Workload(2048, 16, None, image_count=2, image_pixels=1024, engine=_DYN)))

# --- Phase C: a complete judgement with an engine that publishes nothing ------
# The strong form of the engine-blind requirement, and the reason the steady-state gate was cut to
# one client-side signal. Since impl §14.6 this is the *only* form -- no sample carries any server
# reading, for any engine -- so the block now proves the general case rather than a degraded one. A
# run must still settle, still assemble a record, still emit latency, and still be vouched for. If
# any of these four regress, a second serving engine stops being measurable rather than being
# measured less precisely.
_blind_samples = []
for i in range(40):
    _s = Sample(t=1.0 + i * 5.0, subsamples=36)
    _s.in_flight["a"] = 4
    _s.delivered["a"] = i * 128
    _s.launched["a"] = i * 4
    _s.gpus = {UUID_A: {"DCGM_FI_PROF_SM_ACTIVE": 0.62,
                        "DCGM_FI_DEV_POWER_VIOLATION": 0.0}}
    _blind_samples.append(_s)

_blind_settled, _blind_detail = steady_state_check(_blind_samples, th)
check("a fleet whose engine publishes no metrics at all still reaches steady state",
      _blind_settled, str(_blind_detail.get("signals_ok") or _blind_detail.get("reason")))
check("  and the gate that decided it read exactly one signal, the driver's own",
      set(_blind_detail["signals_ok"]) == {"in_flight"}, str(_blind_detail["signals_ok"]))
check("  so no retired threshold can be reintroduced by accident",
      not (set(vars(th)) & _cfgmod.RETIRED_STABILITY_KEYS), str(sorted(vars(th))))

_blind_sampler = types.SimpleNamespace(samples=_blind_samples, cadence_s=5.0,
                                       tenant_cadence_s=TENANT_CADENCE_S,
                                       subsample_lag_max_s=0.004)
_blind_cfg = RunConfig(
    run_id="blind", validity_key_hash="selftest", exporter_url="http://y",
    tenants=[TenantConfig(name="a", base_url="http://x", served_name="m", rate_hz=1.0,
                          isl_tokens=512, osl_tokens=128, gpu_uuids=[UUID_A], tp=1,
                          envelope_gib=8.0, engine="test-blind")])
_blind_cfg.validate()
_blind_rec = build_record(_blind_cfg, {"a": FakeDriver([rr(128, t=1.0 + i * 0.5)
                                                        for i in range(200)], 200)},
                          _blind_sampler, 0.0, 200.0, verdict=VERDICT_MEASURED,
                          steady_state_s=47.2, steady_state_detail=_blind_detail,
                          window_detail={})
_blind_a = _blind_rec["per_tenant"]["a"]
check("  and the run is vouched for, not downgraded for want of server metrics",
      _blind_rec["verdict"] == VERDICT_MEASURED, _blind_rec["verdict"])
check("  with latency measured from the client's own token arrivals", "itl_mean_ms" in _blind_a,
      f"itl_mean_ms={_blind_a.get('itl_mean_ms')}")
check("  and the device still observed engine-blind, off DCGM", UUID_A in _blind_rec["per_gpu"])
# Those signals used to live in the record's cause block, absent rather than zeroed when an engine
# did not publish them. Now they are absent from *every* record, on every engine, and what stands in
# their place is the driver's own in-flight level -- which is why this check is about presence of the
# replacement and not just absence of what it replaced. Absence alone would also pass on a record
# that says nothing about backlog at all, and that record cannot be widened from.
check("  while no server reading reaches the record, on this or any engine",
      not any(k in _blind_a for k in ("kv_usage_mean", "kv_usage_max", "preemptions",
                                      "num_requests_waiting_mean", "num_requests_running_mean",
                                      "num_gpu_blocks"))
      and not any(k in rec["per_tenant"]["a"] for k in ("kv_usage_mean", "preemptions")),
      str(sorted(k for k in _blind_a if "kv" in k or "preempt" in k or "requests" in k)))
check("  and the backlog question is answered anyway, from the driver's own in-flight level",
      _blind_a.get("in_flight_samples") == 40
      and _blind_a["in_flight_first_quarter_mean"] == 4.0
      and _blind_a["in_flight_last_quarter_mean"] == 4.0,
      str({k: v for k, v in _blind_a.items() if k.startswith("in_flight")}))

# A run config written against an earlier image can name a threshold that no longer exists. It is
# refused by name, because the bare TypeError from the dataclass reads as a typo rather than as a
# signal that was deliberately taken out.
import tempfile as _tf, os as _os, json as _json                       # noqa: E402

for _retired in sorted(_cfgmod.RETIRED_STABILITY_KEYS):
    with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as _fh:
        _json.dump({"run_id": "old", "validity_key_hash": "k", "exporter_url": "http://y",
                    "stability": {_retired: 0.1},
                    "tenants": [{"name": "a", "base_url": "http://x", "served_name": "m",
                                 "rate_hz": 1.0, "isl_tokens": 8, "osl_tokens": 4,
                                 "gpu_uuids": [UUID_A]}]}, _fh)
        _p = _fh.name
    try:
        _err = None
        try:
            RunConfig.load(_p)
        except Exception as e:                                          # noqa: BLE001
            _err = e
        check(f"a run config still setting {_retired} is refused, and told why",
              isinstance(_err, ValueError) and _retired in str(_err)
              and "in-flight" in str(_err), repr(_err)[:160])
    finally:
        _os.unlink(_p)
