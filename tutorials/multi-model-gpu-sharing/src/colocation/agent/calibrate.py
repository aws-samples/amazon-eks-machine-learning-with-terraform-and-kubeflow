"""The between-run noise floor, computed from repeats already in the store (impl §12 step 2).

The resolution of a measurement on this node: two runs of one placement that differ by less than
`Calibration.between_run_itl_halfwidth_ms` differ by nothing. `read_calibration` deliberately
returns None rather than a default when it is absent, because a default noise floor is a number
nobody measured being used to decide that two placements are the same.

It is evidence handed to the reasoner, not a stop rule applied above it. Nothing here decides
anything: a reasoner told the floor is 0.4 ms and shown two runs 0.2 ms apart can say they are
indistinguishable, and can also say why that is not the question it was asked.

Nothing here runs a measurement either. The quantity is already in the store: any run that
measured one placement twice has measured the between-run spread, and the difference between those
passes *is* it. This turns that into the sidecar.

Three choices worth stating:

**Only a repeated placement counts.** Two runs of different placements differ for a reason, and
folding them together would report a placement effect as noise -- inflating the floor until the
search declares real differences unresolvable and stops immediately. Placements are grouped by
assignment the same way `policy.canonical` does it, on GPU identity and not on envelope.

**The half-width is the worst pair, not the average one.** It is a claim about what the agent can
resolve, and an average would declare differences resolvable that an observed pair already showed
are not. Maximum over every tenant of every repeated group in the bucket.

**The bucket is the placement's co-residency degree, computed from the record's own GPU UUIDs.**
It used to be `provenance.predicted_regime`, a label the deleted interference model attached to a
placement before measuring it -- so the floor was bucketed by a forecast, and a record whose
forecast was missing could not be bucketed at all. The degree is a fact about what was deployed,
readable from any record the harness wrote, and it separates the cases the bucket exists to
separate: a placement sharing GPUs and one that shares none do not reproduce alike.
"""

from __future__ import annotations

import logging
import statistics

from ..retrieval import sm_caps_of
from ..solver.topology import coresidency
from ..store.schema import Calibration, SuppressedMetric

log = logging.getLogger(__name__)


def placement_key(obs) -> tuple | None:
    """A hashable identity for what a record measured, or None if it cannot be determined.

    GPU UUIDs rather than indices, because that is what the harness records. Envelopes are
    excluded for the same reason `policy.canonical` excludes them: two runs differing by one
    envelope step are the same experiment as far as co-residency goes.
    """
    per = obs.per_tenant
    if not per:
        return None
    out = []
    for name in sorted(per):
        uuids = obs.tenant_config(name).get("gpu_uuids") or []
        if not uuids:
            return None
        out.append((name, tuple(sorted(uuids))))
    # SM caps join the identity as `policy.canonical` joins them (impl §16): a capped run is not a
    # repeat of the same placement uncapped. Uncapped, the key is the one it always was.
    caps = sm_caps_of(obs)
    return (tuple(out), tuple(sorted(caps.items()))) if caps else tuple(out)


def bucket(obs) -> str | None:
    """The co-residency bucket this record belongs in, or None if it cannot be determined.

    The fleet's worst degree, not per tenant: the question is how reproducible a *placement* is,
    and a placement is one thing. Derived from the record, so it needs no agent provenance and
    works on records written before the agent existed.
    """
    per = obs.per_tenant
    if not per:
        return None
    placement = {n: (obs.tenant_config(n).get("gpu_uuids") or []) for n in per}
    if not all(placement.values()):
        return None
    return f"coresident={max(coresidency(placement).values())}"


def between_run_halfwidth(observations, *, notes: dict | None = None) -> Calibration | None:
    """Derive a `Calibration` from repeated placements. None when nothing was repeated.

    None is the honest answer to "what is the noise floor" when no placement has been measured
    twice, and it keeps `read_calibration`'s contract: absent means uncalibrated.
    """
    groups: dict = {}
    for obs in observations:
        if not obs.usable_as_evidence:
            continue
        degree = bucket(obs)
        key = placement_key(obs)
        if degree is None or key is None:
            log.debug("%s not used for calibration: bucket=%r placement=%s",
                      obs.run_id, degree, "unknown" if key is None else "known")
            continue
        groups.setdefault((degree, key), []).append(obs)

    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    if not repeated:
        return None

    halfwidth: dict = {}
    detail: dict = {}
    for (degree, _key), runs in sorted(repeated.items(), key=lambda kv: str(kv[0])):
        for name in sorted(runs[0].per_tenant):
            itls = []
            for obs in runs:
                try:
                    itls.append(obs.per_tenant[name].itl_mean_ms)
                except (KeyError, SuppressedMetric):
                    itls = []
                    break
            if len(itls) < 2:
                continue
            # Half the full range. With two passes that is the only estimator available, and it
            # is biased low -- two draws rarely straddle the true extremes -- which is why
            # `repeats` is recorded next to it rather than left implicit.
            hw = (max(itls) - min(itls)) / 2.0
            if hw > halfwidth.get(degree, -1.0):
                halfwidth[degree] = hw
                detail[degree] = (f"widest pair: {name} over {len(itls)} passes of "
                                  f"{' '.join(sorted(runs[0].per_tenant))}, "
                                  f"{min(itls):.3f}-{max(itls):.3f} ms "
                                  f"(mean {statistics.fmean(itls):.3f} ms)")

    if not halfwidth:
        return None
    repeats = min(len(v) for v in repeated.values())
    return Calibration(
        between_run_itl_halfwidth_ms={k: round(v, 4) for k, v in halfwidth.items()},
        # Not measured here and not defaulted. Solo saturation needs a rate sweep against one
        # tenant alone, which no record in the store is; an invented value would be read as
        # measured by everything downstream.
        solo_saturation_rate_hz={},
        repeats=repeats,
        notes=dict(notes or {}, **{
            "derived_from": "repeated placements already in this store; no new measurement",
            "estimator": ("half the full range of a tenant's ITL across passes of one "
                          "placement, maximised over tenants and placements in the bucket"),
            "repeated_placements": len(repeated),
            "biased_low": (f"{repeats} passes per placement. Two draws rarely straddle the "
                           "true extremes, so this floor is a lower bound on between-run "
                           "spread -- it will call differences resolvable that more passes "
                           "would not."),
            "solo_saturation_rate_hz": "not measured; impl §12 step 2 proper is not run yet",
            "widest_pair_per_bucket": detail,
        }),
    )
