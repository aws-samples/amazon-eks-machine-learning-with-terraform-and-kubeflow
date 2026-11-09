"""Building and writing the observation record (impl §8).

The record is the only durable output. Files on EFS, not a database: readable by the
agent, diffable, versionable alongside the recipes that produced them.

Two rules hold throughout:

  - Delivered fraction is computed FIRST and gates everything (impl §6.3, design §8.5).
    Under open-loop arrival an overloaded server does not fail -- it just delivers
    fewer tokens. Reading the latency of the requests that survived is survivorship
    bias, so below threshold no latency figures are emitted at all.
  - Infeasible observations are stored, never discarded (impl §8). They are what stops
    the agent re-proposing a placement that already failed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .batchmeans import batch_means_ci
from .controller import backlog_series

log = logging.getLogger(__name__)

VERDICT_MEASURED = "measured"
VERDICT_DELIVERY = "infeasible_delivery"
VERDICT_NO_STEADY_STATE = "infeasible_no_steady_state"
VERDICT_DEPLOY = "infeasible_deploy"
VERDICTS = (VERDICT_MEASURED, VERDICT_DELIVERY, VERDICT_NO_STEADY_STATE, VERDICT_DEPLOY)


def delivered_fraction(results, osl_tokens: int, dispatched: int) -> float | None:
    """tokens received / (requests dispatched x OSL).

    Denominator is requests *dispatched* in the window, not requests that came back.
    That is the whole point: an overloaded tenant's missing tokens have to appear in
    the denominator or the shortfall is invisible.
    """
    if dispatched <= 0 or osl_tokens <= 0:
        return None
    return sum(r.tokens for r in results) / (dispatched * osl_tokens)


def chunk_provenance(results, osl_tokens: int) -> dict:
    """Distribution of per-request delivered token counts.

    OSL is supposed to be an input, pinned by min_tokens + ignore_eos, so every
    request should return exactly `osl_tokens`. Recorded rather than assumed: the
    first corrected run read a delivered fraction of 0.991 where the arithmetic said
    1.000, and with only a total there was no way to tell a few short requests from
    many slightly short ones. Verified separately that one SSE chunk carries exactly
    one token on this build, including under batching, so a shortfall here is a real
    shortfall and not a counting artifact.
    """
    if not results:
        return {}
    counts = [r.tokens for r in results]
    short = [c for c in counts if c != osl_tokens]
    chunks = [r.chunks for r in results]
    no_usage = sum(1 for r in results if r.usage_tokens is None)
    return {
        "tokens_min": min(counts),
        "tokens_max": max(counts),
        "tokens_expected": osl_tokens,
        "requests_short_of_osl": len(short),
        "tokens_short_examples": sorted(set(short))[:10],
        # Client-observed chunk counts, kept only as a cross-check on the server's
        # count. A gap between these two is a defect in the harness's observation
        # path, not in the placement being measured.
        "chunks_min": min(chunks),
        "chunks_max": max(chunks),
        "requests_missing_usage": no_usage,
    }


def _tenant_effects(results) -> dict:
    ttfts = [r.ttft_s for r in results if r.ttft_s is not None]
    itls = [r.itl_mean_s for r in results if r.itl_mean_s is not None]
    out: dict = {"requests_measured": len(results)}
    if ttfts:
        out["ttft_mean_ms"] = float(np.mean(ttfts)) * 1000.0
        # p95 on TTFT is fine and is what §8 stores. The mean-not-p95 rule is about
        # ITL specifically (design §1.8).
        out["ttft_p95_ms"] = float(np.percentile(ttfts, 95)) * 1000.0
    gap_itls = [r.itl_from_gaps_s for r in results if r.itl_from_gaps_s is not None]
    if gap_itls:
        # Cross-check only. Rises above itl_mean_ms by exactly the fraction of chunks
        # the client failed to observe, so the pair localises the error.
        out["itl_from_gaps_mean_ms"] = float(np.mean(gap_itls)) * 1000.0
    if itls:
        out["itl_mean_ms"] = float(np.mean(itls)) * 1000.0
        hw = batch_means_ci(itls)
        out["itl_ci_halfwidth_ms"] = None if hw is None else hw * 1000.0
    return out


# This is where `_tenant_causes` was. It filed whichever of the engine's own gauges the server
# published -- KV occupancy, requests running and waiting, a preemption delta, the KV pool's block
# count -- on the grounds that they were recorded and nothing decided on them. Both halves of that
# were wrong (impl §14.6): `too_narrow_reason` read three of them to justify widening a degree, and
# where it did not, the "reader who asks why" the fields existed for does not exist -- the consumers
# of a record are this pipeline and an agent, neither of which may read an engine's internals. The
# question they were consulted for ("is the server falling behind?") is answered by
# `controller.backlog_series` from the driver's own in-flight level, which is engine-blind and is
# the same series the steady-state gate already reads.


def _per_gpu(samples, w0: float, w1: float, uuids: set[str]) -> dict:
    """Per-GPU cause signals, keyed by UUID.

    A per-GPU signal answers "did this device saturate", never "which co-tenant
    saturated it" -- see scrape.scrape_dcgm for why that limitation is structural
    rather than a gap to be closed later.
    """
    acc: dict[str, dict[str, list[float]]] = {}
    for s in samples:
        if not (w0 <= s.t < w1):
            continue
        for uuid, fields in s.gpus.items():
            if uuids and uuid not in uuids:
                continue
            for k, v in fields.items():
                acc.setdefault(uuid, {}).setdefault(k, []).append(v)

    out: dict = {}
    for uuid, fields in acc.items():
        rec: dict = {"samples": max(len(v) for v in fields.values())}
        for k, vals in fields.items():
            short = k.replace("DCGM_FI_PROF_", "").replace("DCGM_FI_DEV_", "").lower()
            rec[f"{short}_mean"] = float(np.mean(vals))
            rec[f"{short}_max"] = float(np.max(vals))
        # Recorded on every GPU whether or not it tripped. A clock drop caused by a
        # power cap is indistinguishable from one caused by co-location unless the
        # cap is in the record.
        for viol in ("power_violation_max", "thermal_violation_max"):
            rec.setdefault(viol, None)
        out[uuid] = rec
    return out


def _sampling_provenance(sampler, w0: float, w1: float) -> dict:
    """How the window was sampled. `subsamples_*` is judged downstream (impl §6.4).

    `subsamples_per_sample_*` counts the fast loop's iterations, so it is a statement about this
    process and `SoloProfile.unusable_reason` may hold a row to it. There is no second count here:
    an `engine_reads_per_sample_*` pair existed for one day, alongside the metrics scrape whose
    thinness it was meant to keep out of the judged number, and both went with the scrape (§14.6).
    Nothing outside this process can influence any figure in this block.
    """
    inwin = [s for s in getattr(sampler, "samples", []) if w0 <= s.t < w1]
    subs = [s.subsamples for s in inwin if getattr(s, "subsamples", 0)]
    return {
        "dcgm_cadence_s": getattr(sampler, "cadence_s", None),
        "tenant_cadence_s": getattr(sampler, "tenant_cadence_s", None),
        "samples_in_window": len(inwin),
        "subsamples_per_sample_mean": float(np.mean(subs)) if subs else 0,
        "subsamples_per_sample_min": min(subs) if subs else 0,
        "subsample_lag_max_ms": getattr(sampler, "subsample_lag_max_s", 0.0) * 1000.0,
    }


def build_record(cfg, drivers, sampler, w0: float, w1: float, *, verdict: str,
                 steady_state_s: float | None, steady_state_detail: dict,
                 window_detail: dict, notes: dict | None = None) -> dict:
    """Assemble the record. Latency is emitted only where delivery passed."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")

    per_tenant: dict = {}
    delivery_failed = []
    for t in cfg.tenants:
        d = drivers.get(t.name)
        rec: dict = {}
        if d is None:
            per_tenant[t.name] = {"error": "no driver"}
            continue

        results = d.completed_in(w0, w1)
        dispatched = d.dispatched_in(w0, w1)
        # This tenant's own OSL. A fleet-wide one would compute the delivered fraction of a
        # tenant against a token count nobody asked it for, and the gate is what decides whether
        # any latency in this record may be read at all (impl §6.3).
        frac = delivered_fraction(results, t.osl_tokens, dispatched)

        rec["requests_dispatched"] = dispatched
        # What the driver actually sent, including requests that never came back. Under overload
        # `requests_dispatched` goes to zero -- it can only count requests it holds a result for --
        # so without this the record cannot distinguish a server that was offered load and served
        # none from a driver that never sent anything. Those need opposite responses, and the walk
        # in `agent/profiler.py` is the caller that has to tell them apart.
        rec["requests_launched"] = d.launched_in(w0, w1)
        rec["requests_completed"] = len(results)
        rec["requests_failed"] = sum(
            1 for r in d.results if not r.ok and w0 <= r.dispatch_t < w1)
        rec["delivered_fraction"] = frac
        rec["delivered_fraction_threshold"] = t.delivered_fraction_threshold
        # Emitted whether or not the gate passed: it is how you tell "a few requests
        # were cut short" from "every request was slightly short", and those have
        # different causes.
        rec.update(chunk_provenance(results, t.osl_tokens))

        # If the driver could not keep to its own schedule the run is invalid for
        # reasons that have nothing to do with the server. Recorded, not inferred.
        lags = [r.schedule_lag_s for r in d.results if w0 <= r.dispatch_t < w1]
        if lags:
            rec["schedule_lag_max_ms"] = float(np.max(lags)) * 1000.0
            rec["schedule_lag_mean_ms"] = float(np.mean(lags)) * 1000.0
        rec["issue_cap_hit"] = d.hit_cap

        passes = frac is not None and frac >= t.delivered_fraction_threshold
        if passes:
            rec.update(_tenant_effects(results))
        else:
            delivery_failed.append(t.name)
            rec["latency_suppressed"] = (
                "delivered fraction below threshold: latency over the surviving "
                "requests would be survivorship bias (impl §6.3)")

        # The backlog block: this tenant's in-flight level over the window, unjudged. It is what
        # `too_narrow_reason` reads to decide whether the server fell behind, and it replaced the
        # engine's queue gauge rather than supplementing it (impl §14.6).
        rec.update(backlog_series(sampler.samples, t.name, w0, w1))
        per_tenant[t.name] = rec

    # Delivery is a gate, and it outranks a nominally successful measurement.
    if verdict == VERDICT_MEASURED and delivery_failed:
        log.error("delivery gate failed for %s -- verdict is %s",
                  delivery_failed, VERDICT_DELIVERY)
        verdict = VERDICT_DELIVERY

    uuids = {u for t in cfg.tenants for u in t.gpu_uuids}
    record = {
        "run_id": cfg.run_id,
        "validity_key_hash": cfg.validity_key_hash,
        "node": cfg.node,
        "verdict": verdict,
        "delivery_gate_failed": delivery_failed,
        "steady_state_reached_s": steady_state_s,
        "steady_state_detail": steady_state_detail,
        "window_s": (w1 - w0) if w1 > w0 else 0.0,
        "window_rule": window_detail,
        # Per tenant, and deliberately duplicated into `configuration.tenants` below. This block
        # is the workload as *asked for and achieved* -- including which tokenizer built the
        # prompt and whether the exact ISL was hit -- while the configuration block is what a
        # reader compares two runs by. There is no fleet-wide tokenizer here because there is no
        # fleet-wide tokenizer: `__main__` merges each `Workload.describe()` in, which carries
        # the one that tenant actually used.
        "workload": {"tenants": {t.name: {"isl_tokens": t.isl_tokens,
                                          "osl_tokens": t.osl_tokens,
                                          "tokenizer_path": t.tokenizer_path,
                                          "image_count": t.image_count,
                                          "image_pixels": t.image_pixels}
                                 for t in cfg.tenants}},
        "configuration": {"tenants": [
            {"name": t.name, "model_id": t.model_id, "tp": t.tp,
             "envelope_gib": t.envelope_gib, "rate_hz": t.rate_hz,
             "isl_tokens": t.isl_tokens, "osl_tokens": t.osl_tokens,
             # In the comparison block and not only in the workload block above: two runs
             # of one fleet differing in image payload are two different experiments, and a
             # reader comparing them by `configuration` must be able to see that here.
             "image_count": t.image_count, "image_pixels": t.image_pixels,
             "gpu_uuids": t.gpu_uuids,
             # Only when set, so an uncapped record reads exactly as one written before impl §16.
             **({"active_thread_percentage": t.active_thread_percentage}
                if t.active_thread_percentage is not None else {})} for t in cfg.tenants]},
        "per_tenant": per_tenant,
        "per_gpu": _per_gpu(sampler.samples, w0, w1, uuids),
        "samples_collected": len(sampler.samples),
        # Recorded because getting this wrong once produced a record that looked
        # perfectly healthy while reporting zero occupancy. A record whose
        # subsamples_per_sample is 1 is a record whose tenant gauges are point reads
        # and may be phase-locked to the arrival schedule; see controller.py.
        "sampling": _sampling_provenance(sampler, w0, w1),
        "provenance": cfg.provenance,
        "notes": notes or {},
    }
    return record


def write_record(record: dict, out_dir: str, validity_key_hash: str, run_id: str) -> Path:
    """Write to the §8 layout. Never overwrite: a repeat is a new observation.

    On any filesystem failure the record is dumped to the log before the exception
    propagates. Learned the hard way: the first live run measured a full window and
    then lost every number to a permission error on the store root, because the write
    was the last thing that could fail and nothing had been reported yet. A run costs
    minutes of GPU time; recovering it from Job logs must always be possible.
    """
    payload = json.dumps(record, indent=2, sort_keys=True, default=str) + "\n"
    d = Path(out_dir) / "observations" / validity_key_hash / "observations"
    try:
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{run_id}.json"
        n = 1
        while path.exists():
            path = d / f"{run_id}-{n}.json"
            n += 1
        path.write_text(payload)
    except OSError as exc:
        log.error("could not write the record to %s: %s. Dumping it below so the run "
                  "is recoverable from these logs.", d, exc)
        log.error("BEGIN OBSERVATION RECORD %s\n%s\nEND OBSERVATION RECORD %s",
                  run_id, payload, run_id)
        raise
    log.info("wrote %s", path)
    return path
