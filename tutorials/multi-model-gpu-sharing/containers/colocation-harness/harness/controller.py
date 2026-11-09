"""Sampling, steady-state detection, and the window-length rule (impl §6.4).

**Two cadences, because the two sources have opposite requirements.**

*DCGM is scraped at exactly 5s, and that is not a free parameter.* `dcgm-exporter`
collects into a cache on a background timer (--collect-interval, pinned to 5000ms) and
every scrape at any rate re-reads that cache. So:

  - Scraping FASTER returns duplicate values reported as independent samples, which is
    false precision -- measured, SM_ACTIVE returned the byte-identical value for nine
    consecutive scrapes 3.5s apart.
  - Scraping SLOWER reintroduces a bias worse than the staleness: a metric window
    longer than a batch straddles batch boundaries. The first fully-loaded 30s window
    read 0.063 where five consecutive 5s windows read 0.3065 +/- 0.0003. Same load,
    5x error, and no way to tell from the value alone.

Each DCGM field is already a time-average over its own 5s collection window, so one
read per window is the right number of reads.

*The in-flight count must be sampled far faster than that, and at a cadence
incommensurate with the request interval.* It is an instantaneous read, not a windowed
average, and the first live run got this catastrophically wrong. Driving 4 tenants at
2 req/s (a 0.5s interval) and sampling at 5.0s -- an exact 10x multiple -- phase-locked
every read to the dispatch instant, when the previous request had finished and the next
had not started. Occupancy read exactly 0.0 on all 44 samples for all 4 tenants while
DCGM reported the GPU 47% busy. Measured against the same tenant under the same load:

    cadence   samples   mean requests in flight   nonzero
    5.0s      9         0.000                     0/9
    0.137s    293       0.669                     196/293

True occupancy was 0.67 and the harness recorded 0. That run read the figure off the
engine's own `num_requests_running` gauge, because the harness scraped tenant metrics
then; it no longer does (impl §14.6), and the lesson survives the source unchanged --
the driver's `launched - completed` is just as instantaneous and just as phase-lockable.
So the fast series is sampled at TENANT_CADENCE_S with a deterministic phase rotation,
and each 5s Sample carries the time-average of the sub-samples collected during that
interval. The level is averaged; cumulative counts take the last value in the bucket,
since they are used for deltas.

The rotation matters as much as the cadence: any fixed sub-cadence is commensurate with
*some* arrival rate, and rate is a config knob the harness cannot constrain. Rotating
the phase deterministically guarantees coverage regardless of the rate chosen.

**Nothing here scrapes a serving engine.** Two loops, both reading instruments the engine
does not own: this process's own counters, and DCGM's account of the device. There is no
third source, so there is no signal in a record whose presence depends on which server
answered -- which is what makes a row from one engine comparable with a row from another
(impl §6.5, §14.6).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from .batchmeans import batch_means_ci, relative_halfwidth
from .scrape import scrape_dcgm

log = logging.getLogger(__name__)

DCGM_CADENCE_S = 5.0

# Tenant sub-cadence. Chosen to be a poor divisor of the request intervals a run is
# likely to use (0.5s, 0.1s, 0.05s, 1.0s) rather than for any property of its own.
# Combined with the phase rotation below, ~36 sub-samples land in each 5s bucket at
# rotating phases relative to dispatch.
TENANT_CADENCE_S = 0.137
# Number of phase positions the sub-cadence rotates through. Coprime with the small
# integers, so `(i * ROTATE_STEP) % PHASE_POSITIONS` visits every position.
PHASE_POSITIONS = 7
ROTATE_STEP = 3

@dataclass
class Sample:
    """One 5s observation. `in_flight` is a bucket time-average, not a point read."""
    t: float
    gpus: dict = field(default_factory=dict)      # uuid -> field dict
    in_flight: dict = field(default_factory=dict)  # name -> mean client-side in-flight
    delivered: dict = field(default_factory=dict)  # name -> cumulative ok chunks
    # name -> cumulative requests this driver has put on the wire. The driver's own O(1) counter
    # (`LoadDriver.issued`), not a walk of the task list, and cumulative for the same reason
    # `delivered` is: a difference across a window is the arrivals in that window. It is what turns
    # a drift in `in_flight` into a *fraction of the offered load* rather than a bare count, which
    # is what `steady_state_check` needs to test the claim this gate actually makes.
    launched: dict = field(default_factory=dict)
    # Driver reads behind this Sample. Counts the fast loop's iterations, which is what makes it
    # safe to judge a row on: no server can influence it. See `Sampler`'s docstring.
    subsamples: int = 0


@dataclass
class StabilityThresholds:
    """The steady-state gate: one signal, and it is the driver's own (impl §6.4).

    In-flight is `launched - completed` on the client's books. Requests go onto the wire on an
    open-loop fixed-rate schedule with no coupling to whether earlier ones finished (`load.py`), so
    arrivals happen at the configured rate whether or not the server keeps up. That makes a flat
    in-flight count *the same fact* as "completions are keeping pace with the offered rate" -- not
    corroborating evidence for it, the same arithmetic: `d(in_flight)/dt == 0` is
    `completion_rate == arrival_rate`, and the arrival rate is a constant the driver sets.

    **Two terms, because "flat" is a claim about the derivative and a half-width is not.** The
    arithmetic above is about `d(in_flight)/dt`; a batch-means relative half-width is a confidence
    interval on the *level*, and the two are not the same test. `in_flight_rel` alone passed a run
    whose queue was running away, and it could not have done otherwise:

      - For a monotonic rise of total size `R` across the judged window, `k=10` batches at `z=1.96`
        put the half-width at `~0.2085 R`, so a ramp passes a 0.15 gate whenever `R <= 0.72 * mean`.
      - The tolerance is *relative to the mean*, so the absolute drift waved through GROWS with the
        in-flight level -- backwards, since a high level is the symptom of the thing being tested
        for. Past `mean >= arrival_rate * judged_seconds / 0.72` even a total stall is inside the
        gate: at 0.5 Hz over 150s judged, anything above ~104 in flight.
      - Measured, 2026-09-21: `qwen35-multimodal-2`'s `docqa` at 0.5 Hz ran to in-flight 117.6 and
        was declared settled at 330s with **0 of 30 requests completed**, three times over. No wrong
        number was published -- the downstream delivery gate suppressed the latency as survivorship
        bias (impl §6.3) -- but this gate vouched for a dead run, and the profiler read the same
        series as "this degree is too narrow" and spent tp2, tp4 and tp8 re-learning it.

    So `in_flight_drift_frac` states the derivative directly, and normalises it by the offered load
    rather than by the level: the rise across the window over the ARRIVALS across the same span.
    Since `in_flight == launched - completed`, that ratio is exactly `1 - completed/launched` over
    the window, so the threshold reads as what the paragraph above claims the gate tests -- 0.10 is
    "completions kept pace with arrivals to within 10%". Nothing about it scales with the level, and
    a stall reads 1.0 whatever the level is.

    Both terms judge the SAME series, `in_flight`, read two ways. That is why `signals_ok` still has
    one entry: the count of signals in this gate is one, and it is the driver's own.

    **Why nothing else is here, including the engine's own gauges.** A gate that reads backend
    internals -- KV occupancy, queue depth, preemption counters -- redefines "settled" every time a
    server changes its internals, blocks entirely on an engine that has no analogue of the concept,
    and judges two engines by different criteria. It also buys nothing: anything happening inside
    that matters to what this harness measures (latency, delivered throughput) must show up in
    arrival, completion or in-flight, and anything that shows up in none of them did not affect the
    measurement. Peering in buys earliness, not correctness. Those signals are not recorded either
    any more (impl §14.6): a number no decision may read has no reader here, since the consumers of
    a record are this pipeline and an agent, not somebody forming a hunch.

    **The same series answers the opposite question, and that is deliberate.** Flat while delivering
    is settled; *rising* is the server falling behind the offered rate, which is what
    `SoloProfile.too_narrow_reason` needs in order to widen a degree. One measurement read in two
    directions, rather than two instruments that can disagree -- and it is the driver's, so it cannot
    be defeated by a server that stops answering a scrape under exactly the load being characterised.

    **Why not a second client-side series either.** `delivered_rate` used to be a gate term and was
    removed as redundant-and-noisier, not as unwanted: with output length pinned, delivered tokens
    are request completions times a constant, so its stationarity is in-flight's flatness restated
    through a difference of cumulative counts -- which amplifies the lumpiness of per-completion
    arrivals instead of measuring a level. It produced a false negative on real hardware: GLM-4.5-Air
    at tp4 was filed `infeasible_no_steady_state` on `delivered_rate_rel` 0.164 against a 0.10
    threshold while in-flight was flat, nothing was queued and everything was delivered. The
    single-signal gate judges that run correctly. Where output length is *not* pinned the argument
    gets stronger, not weaker: reply lengths then vary, so token rate wanders while request flow is
    genuinely steady, and in-flight is the signal that is still about the right thing.

    `min_samples` is both the floor before any verdict and the length of the window the verdict is
    read over: the check judges every sample after the warm-up (`steady_state_check`).
    """
    in_flight_rel: float = 0.15
    # The drift term above: |rise in in-flight| as a fraction of the requests that arrived over the
    # same span, which is the completion deficit. 0.10 is the same tolerance the retired
    # `delivered_rate_rel` carried, chosen so this does not tighten the gate for a run that was
    # already passing honestly -- it closes a hole rather than moving a bar.
    in_flight_drift_frac: float = 0.10
    consecutive: int = 2                  # sub-batches that must all pass
    min_samples: int = 30                 # 30 * 5s = 150s judged, and not before


class Sampler:
    """Owns both series; nothing else samples anything.

    Two loops, neither of which reads a serving engine.

      - `_driver_loop` reads the drivers' own counters at the fast cadence. Two dict lookups,
        no network, no engine: it cannot fail and it cannot be made slow by anything outside
        this process. `Sample.subsamples` counts its iterations.
      - `_dcgm_loop` reads DCGM every 5s, closes the fast bucket, and emits one Sample.

    Downstream code sees a single 5s series, so the batch-means thresholds keep meaning what
    they meant.

    **Why the fast loop touches no network, stated as a requirement.** The steady-state gate
    judges one number, the mean in-flight count, and a row is disqualified (`sampling_suspect`)
    when too few sub-samples stand behind that mean -- rightly, because a point read per 5s
    bucket can sit at a fixed phase against the arrival schedule and read as flat while nothing
    of the kind is true. That guard is only fair if the sub-sample count is the driver's own
    property. For one day it was not: this loop also scraped the tenants' metrics endpoints,
    awaiting the network with a 10s timeout on a 0.137s cadence, so one tenant whose endpoint
    hung -- or an engine that simply did not serve the path its adapter named -- collapsed the
    count for every tenant and got the row thrown out for the *server's* silence rather than the
    driver's. It was split into two loops, and then impl §14.6 deleted the scrape outright, which
    is why the property now holds by construction instead of by a separation that had to be kept.
    """

    def __init__(self, exporter_url: str | None, drivers: dict,
                 cadence_s: float = DCGM_CADENCE_S,
                 tenant_cadence_s: float = TENANT_CADENCE_S):
        self.exporter_url = exporter_url
        self.drivers = drivers            # name -> LoadDriver
        self.cadence_s = cadence_s
        self.tenant_cadence_s = tenant_cadence_s
        self.samples: list[Sample] = []
        self._bucket: list[dict] = []          # driver reads
        self._stop = asyncio.Event()
        self.subsample_lag_max_s = 0.0

    async def run(self, client: httpx.AsyncClient) -> None:
        await asyncio.gather(self._driver_loop(), self._dcgm_loop(client))

    async def _sleep_until(self, target: float) -> bool:
        """True to continue, False if stopped. Absolute targets, so no drift."""
        delay = target - time.perf_counter()
        if delay > 0:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                return False
            except asyncio.TimeoutError:
                pass
        return not self._stop.is_set()

    def _phase_target(self, t0: float, i: int) -> float:
        """The i'th fast-cadence target, phase-rotated.

        Deterministic rotation. No randomness: Math.random-style jitter would make a run
        unreproducible, and reproducibility is the point of pinning everything else.
        """
        base = self.tenant_cadence_s
        return t0 + i * base + ((i * ROTATE_STEP) % PHASE_POSITIONS) * (base / PHASE_POSITIONS)

    async def _driver_loop(self) -> None:
        """The judged series. Client-side only, so nothing outside this process can thin it."""
        t0 = time.perf_counter()
        i = 0
        while True:
            target = self._phase_target(t0, i)
            if not await self._sleep_until(target):
                return
            self.subsample_lag_max_s = max(
                self.subsample_lag_max_s, time.perf_counter() - target)
            reading: dict = {"in_flight": {}, "delivered": {}, "launched": {}}
            for n, d in self.drivers.items():
                # O(1) reads. Counting over the task list here would grow with the
                # run and eventually make the sampler miss its own cadence, which is
                # the failure this whole loop exists to avoid. `issued` is the driver's
                # running total for exactly that reason -- `launched_in` walks `_launched`
                # and belongs in the record assembly, not on this cadence.
                reading["in_flight"][n] = d.in_flight()
                reading["delivered"][n] = d.delivered_chunks
                reading["launched"][n] = d.issued
            self._bucket.append(reading)
            i += 1

    async def _dcgm_loop(self, client: httpx.AsyncClient) -> None:
        t0 = time.perf_counter()
        # Start at i=1: the first bucket needs a full interval of sub-samples in it.
        i = 1
        while True:
            if not await self._sleep_until(t0 + i * self.cadence_s):
                return
            gpus = {}
            if self.exporter_url:
                gpus = await scrape_dcgm(client, self.exporter_url)
            self.samples.append(self._close_bucket(gpus))
            i += 1

    def _close_bucket(self, gpus: dict) -> Sample:
        bucket, self._bucket = self._bucket, []
        s = Sample(t=time.perf_counter(), gpus=gpus, subsamples=len(bucket))
        if not bucket:
            # Should not happen. If it does, the sample is left with no client-side series at
            # all, which `steady_state_check`'s no-readings guard refuses by name rather than
            # reading as flat.
            log.warning("no driver sub-samples in this interval")

        for name in self.drivers if bucket else ():
            vals = [r["in_flight"][name] for r in bucket if name in r["in_flight"]]
            s.in_flight[name] = sum(vals) / len(vals) if vals else 0.0
            dl = [r["delivered"][name] for r in bucket if name in r["delivered"]]
            # Cumulative: last, not mean, or the window deltas come out halved.
            s.delivered[name] = dl[-1] if dl else 0
            ll = [r["launched"][name] for r in bucket if name in r.get("launched", {})]
            s.launched[name] = ll[-1] if ll else 0          # cumulative, as above
        return s

    def stop(self) -> None:
        self._stop.set()


def _series(samples: list[Sample], fn) -> list[float]:
    out = []
    for s in samples:
        v = fn(s)
        if v is not None:
            out.append(v)
    return out


def backlog_series(samples: list[Sample], name: str,
                   w0: float, w1: float) -> dict:
    """This tenant's in-flight level over the window, described so its *direction* can be read.

    The same series `steady_state_check` reads for flatness, read for slope instead. Under the
    open-loop schedule of impl §6.2 requests go onto the wire at a fixed rate whatever the server
    is doing, so a rising in-flight level is completion falling behind arrival -- by arithmetic,
    not by inference. That is the fact an engine's `requests_waiting` gauge used to supply, and
    this is the same fact observed from outside the server (impl §14.6).

    Quarter means rather than a fitted slope. A least-squares line over ~30 samples is dominated
    by whichever end is noisier and reports a small non-zero slope on any real series, which would
    put the whole weight of the judgement on a threshold. Comparing the first quarter's mean with
    the last quarter's asks a blunter question -- is the level higher at the end than at the start
    -- and blunt is what is wanted for a signal whose job is to be unambiguous.

    No thresholds here: this is an extractor, and `SoloProfile.too_narrow_reason` judges it, so a
    row written today is re-judged under today's rule (impl §8).
    """
    vals = [s.in_flight[name] for s in samples
            if w0 <= s.t < w1 and name in s.in_flight]
    if not vals:
        return {}
    q = max(1, len(vals) // 4)
    return {
        "in_flight_mean": float(sum(vals) / len(vals)),
        "in_flight_max": float(max(vals)),
        "in_flight_first_quarter_mean": float(sum(vals[:q]) / q),
        "in_flight_last_quarter_mean": float(sum(vals[-q:]) / q),
        # So a reader can tell "one quarter each" from "the whole window was six samples".
        "in_flight_samples": len(vals),
    }


def warmup_samples(samples: list[Sample]) -> int:
    """How many leading samples were taken before the fleet produced any output.

    A sample taken before the first request has finished is not a measurement of a
    system in steady state; it is a measurement of a system that has not started
    delivering yet. Its delivered rate is exactly zero by construction, and zero is
    as far from the steady rate as the series ever gets, so leaving those samples in
    guarantees a wide half-width no matter how still the machine is.

    Data-driven rather than a configured number of seconds, because the length of the
    transient is set by the model: one request of 1024 output tokens at 55 ms/token
    takes ~56s to finish, so a 5s cadence spends ~11 samples at zero. A faster model
    spends one or two, which is why this never showed up before GLM-4.5-Air.

    The boundary sample itself is discarded too: the delta across it spans the moment
    delivery started, so it is a partial-ramp rate rather than a steady one.

    Only the LEADING run of zeros counts. A sample whose driver read failed also
    reports zero (`Sampler._close_bucket`), and treating that as warm-up would throw
    away every sample before it -- so a mid-run dropout stays in the window, where it
    shows up as an unsettled rate and fails the check. That is the safe direction:
    the check is meant to withhold a verdict when the signal is doubtful.
    """
    warm = 0
    for s in samples:
        if sum(s.delivered.values()) > 0:
            break
        warm += 1
    return warm


def steady_state_check(samples: list[Sample], th: StabilityThresholds) -> tuple[bool, dict]:
    """Is in-flight flat AND not drifting, over every sample since delivery began? (impl §6.4)

    Two readings of one series -- a half-width on the level and a drift normalised by the arrivals
    over the same span -- and both have to hold. `StabilityThresholds` is where that is argued; the
    short version is that a half-width is a statement about a level and "flat" is a statement about
    a derivative, and a gate that tested only the first vouched for a stalled queue.

    Returns (settled, detail) so a run that never settles records what the half-width
    actually was. "Did not converge" without the number is not actionable.

    **Flatness alone is not enough, because zero is flat.** An in-flight series pinned
    at 0 -- nothing sent, or nothing completing -- has a perfect half-width, so the
    non-vacuity guards below are load-bearing rather than defensive housekeeping, and
    more so now that they are the only thing standing between a dead run and a
    `settled` verdict. Four of them, in order: the warm-up discard drops the leading
    run of samples before any tenant delivered anything; an empty remainder is refused
    by name; `min_samples` refuses a window too short to have a half-width worth
    quoting; and a window over which nothing ARRIVED is refused too, since the drift
    term has nothing to normalise against and a driver that issued nothing is not a
    settled fleet. A run that delivers nothing at all never gets past the second.

    The warm-up is discarded and everything after it is kept (`warmup_samples`), rather
    than a trailing window. That was originally forced by `delivered_rate`, whose
    per-completion lumpiness needed ~100 samples to average out; in-flight is a level
    and is far better behaved, so a trailing window would probably now work. Probably
    is not a reason to shorten a measurement window -- that is a change to make against
    data, not against an argument -- so the accumulate-everything behaviour stands.

    What the discard fixes: GLM-4.5-Air at tp4 came back `infeasible_no_steady_state`
    while in-flight was flat and nothing was waiting in any queue. 11 of its 132 samples
    were taken before the first 1024-token reply could finish, and one deviant batch of
    zeros is enough to hold a half-width above a gate indefinitely.
    """
    detail: dict = {}
    warm = warmup_samples(samples)
    detail["warmup_samples_discarded"] = warm
    detail["samples_total"] = len(samples)
    samples = samples[warm:]
    detail["samples_judged"] = len(samples)
    if not samples:
        return False, {**detail,
                       "reason": f"no tenant has completed a request yet, after "
                                 f"{warm} samples"}
    if len(samples) < th.min_samples:
        return False, {**detail,
                       "reason": f"only {len(samples)} samples since delivery began, "
                                 f"need {th.min_samples}"}

    # A bucket that collected no readings at all is a hole in the series, and `_close_bucket` fills
    # it with zeros because it has nothing else to give -- zero in-flight, zero delivered. Under the
    # old four-signal gate a mid-run dropout was caught as a side effect: `delivered_rate` saw the
    # cumulative count fall and the run lost its verdict, which is the safe direction and was
    # deliberate. In-flight does not catch it -- one zero among sixty samples at a level of 4 leaves
    # a relative half-width near 0.04, well inside the gate -- so the protection is stated here
    # instead of being inherited from a redundant signal. It is the driver's own book (did a read
    # come back?) and not the engine's, so it belongs in a client-side gate.
    holes = sum(1 for s in samples if s.subsamples == 0)
    if holes:
        return False, {**detail, "sample_holes": holes,
                       "reason": f"{holes} of {len(samples)} judged samples collected no readings "
                                 f"at all, so the series has gaps that read as zero load"}

    # Aggregate across tenants: the node is the unit under test, and a single
    # unsettled tenant means the node is not in steady state.
    in_flight = _series(samples, lambda s: sum(s.in_flight.values()) or 0.0)
    hw = relative_halfwidth(in_flight)
    detail["in_flight_rel"] = hw
    ok_level = hw is not None and hw <= th.in_flight_rel

    # The drift term. Quarter means on both series rather than a fitted slope, for
    # `backlog_series`'s reason -- a least-squares line over ~30 samples is dominated by whichever
    # end is noisier -- and reduced the SAME way on both so the two spans match. A cumulative
    # counter's mean over a quarter is its value at that quarter's centre, so the difference of the
    # quarter means is the arrivals between the same two instants the rise is measured across. Get
    # that wrong in either direction and the ratio stops being the completion deficit it is
    # thresholded as.
    launched = _series(samples, lambda s: float(sum(s.launched.values())))
    q = max(1, len(in_flight) // 4)
    rise = (sum(in_flight[-q:]) - sum(in_flight[:q])) / q
    arrivals = ((sum(launched[-q:]) - sum(launched[:q])) / q if len(launched) == len(in_flight)
                else 0.0)
    detail["in_flight_rise"] = rise
    detail["arrivals_judged"] = arrivals
    if arrivals <= 0.0:
        # No arrival information, so there is nothing to normalise against and flatness here is
        # vacuous in the same way a series pinned at zero is: a driver that launched nothing over
        # 150s is not a settled fleet. Refused by name rather than defaulted, because the only ways
        # to get here are a driver that stopped issuing and a caller that did not record the
        # counter, and vouching for either would be vouching for the absence of the evidence.
        detail["signals_ok"] = {"in_flight": False}
        return False, {**detail,
                       "in_flight_drift_frac": None,
                       "reason": f"no request arrived over the {len(samples)} judged samples, so a "
                                 "drift in in-flight cannot be read as a fraction of the offered "
                                 "load and flatness says nothing"}
    drift = abs(rise) / arrivals
    detail["in_flight_drift_frac"] = drift
    ok_drift = drift <= th.in_flight_drift_frac

    # One signal, read two ways: see `StabilityThresholds`. Both have to hold, and the reason names
    # which one did not, because "did not converge" without that is a run nobody can act on.
    ok_if = ok_level and ok_drift
    detail["signals_ok"] = {"in_flight": ok_if}
    if not ok_drift:
        detail["reason"] = (
            f"in-flight drifted by {rise:+.1f} over {arrivals:.0f} arrivals "
            f"({drift:.3f} of the offered load, over {th.in_flight_drift_frac:.2f}): completion is "
            "not keeping pace with arrival, whatever the level's half-width says")
    return ok_if, detail


# There were two more helpers here -- `_mean_kv` and `_sum_preempt` -- that averaged the engine's KV
# occupancy and differenced its preemption counter so `steady_state_check` could gate on both. They
# are gone rather than unused: Phase B went to some trouble to make them return None instead of 0.0
# so an engine with no such gauge could not *satisfy* a threshold it never reported against, and
# deleting the dependency is strictly better than guarding it. The signals themselves are no longer
# recorded either (impl §14.6) -- an unreadable number with no reader is not provenance.


async def await_steady_state(sampler: Sampler, th: StabilityThresholds,
                             ceiling_s: float) -> tuple[bool, float, dict]:
    """Poll until settled or the ceiling trips.

    Hitting the ceiling is a feasibility verdict, not a failed measurement
    (impl §6.4, design §8.2) -- the caller records it as such.
    """
    t_start = time.perf_counter()
    passes = 0
    last: dict = {}
    while time.perf_counter() - t_start < ceiling_s:
        await asyncio.sleep(DCGM_CADENCE_S)
        settled, detail = steady_state_check(sampler.samples, th)
        last = detail
        passes = passes + 1 if settled else 0
        if settled:
            log.info("steady-state check passed (%d/%d consecutive)", passes, th.consecutive)
        if passes >= th.consecutive:
            return True, time.perf_counter() - t_start, detail
    log.error("steady state not reached within %.0fs ceiling: %s", ceiling_s, last)
    return False, time.perf_counter() - t_start, last


async def await_window(drivers: dict, window_start_t: float, target_halfwidth_s: float | None,
                       min_s: float, ceiling_s: float) -> tuple[bool, float, dict]:
    """Extend the window until the ITL half-width meets the precision target.

    The target is half the between-run reproducibility figure for the regime, read
    from the calibration record. During calibration itself no such record exists, so
    a fixed window is used instead and which rule applied is recorded.
    """
    detail: dict = {"rule": "precision" if target_halfwidth_s else "fixed"}
    while True:
        elapsed = time.perf_counter() - window_start_t
        if elapsed >= ceiling_s:
            detail["reason"] = f"window ceiling {ceiling_s:.0f}s reached"
            return False, elapsed, detail
        if elapsed < min_s:
            await asyncio.sleep(DCGM_CADENCE_S)
            continue
        if target_halfwidth_s is None:
            detail["reason"] = "fixed window satisfied"
            return True, elapsed, detail

        worst = None
        for name, d in drivers.items():
            rs = d.completed_in(window_start_t, time.perf_counter())
            itls = [r.itl_mean_s for r in rs if r.itl_mean_s is not None]
            hw = batch_means_ci(itls) if len(itls) >= 20 else None
            detail[f"itl_halfwidth_s.{name}"] = hw
            if hw is None:
                worst = float("inf")
                break
            worst = hw if worst is None else max(worst, hw)
        if worst is not None and worst <= target_halfwidth_s:
            detail["reason"] = "precision target met"
            return True, time.perf_counter() - window_start_t, detail
        await asyncio.sleep(DCGM_CADENCE_S)
