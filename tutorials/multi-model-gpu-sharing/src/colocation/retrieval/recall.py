"""Prior runs that bear on a candidate (impl §10).

The query, and the four types it returns. Read `retrieval/__init__.py` first for why this is a
query and not a model.

Pure standard library plus `solver/topology.py`, which is stdlib too. No OR-Tools, nothing from
`agent/`: retrieval is what the agent calls, so the dependency may not point back. It also means a
record can be read and judged on a laptop with nothing installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..solver.topology import coresidency
# The same constant `too_narrow_reason` judges on, deliberately: a qualifier that warned about a
# backlog the walk did not consider risen would be two thresholds for one question.
from ..store.profile import IN_FLIGHT_RISE_FRACTION
from ..store.schema import Observation, SuppressedMetric

# Reasons a record was set aside, and the text each is reported under. Every one of them is
# printed with the run ids it applied to: a record silently dropped is a record the reasoner
# believes does not exist, and "no evidence" then means two different things.
EXCLUDED_KEY = "under a different validity key, so it is not comparable"
EXCLUDED_TENANTS = "names none of the tenants asked about"
EXCLUDED_SUSPECT = ("its client-side series is a point read per sample and may be phase-locked to "
                    "the arrival schedule (sampling.subsamples_per_sample_mean below 2)")


def placement_of(obs: Observation, gpu_of: dict | None = None) -> dict | None:
    """`{tenant: (gpu, ...)}` as this run actually ran, or None if it cannot be resolved.

    Records name GPUs by UUID; a candidate names them by host ordinal. `gpu_of` is the
    `{uuid: ordinal}` map that relates the two, and it is the *only* thing that does — which is
    why it is passed in rather than derived here. Without it the UUIDs are returned as they stand,
    which is still a usable identity for comparing two records with each other.

    None, not a partial map, when any tenant's GPUs are missing or off this node. A half-resolved
    placement would claim a structural relation to the candidate that was never checked.
    """
    out: dict = {}
    for name in obs.per_tenant:
        uuids = obs.tenant_config(name).get("gpu_uuids") or []
        if not uuids:
            return None
        if gpu_of is None:
            out[name] = tuple(uuids)
            continue
        gpus = tuple(gpu_of[u] for u in uuids if u in gpu_of)
        if len(gpus) != len(uuids):
            return None
        out[name] = gpus
    return out or None


def sm_caps_of(obs: Observation) -> dict:
    """`{tenant: percent}` for the tenants this run capped (impl §16); empty when none was.

    Part of a run's placement as much as its GPUs are: a tenant held to 25% of the SMs is a
    different experiment from the same tenant on the same card uncapped. A record written before
    the field existed has no cap anywhere and reads as uncapped, which is what it was.
    """
    out = {}
    for name in obs.per_tenant:
        cap = obs.tenant_config(name).get("active_thread_percentage")
        if cap is not None:
            out[name] = int(cap)
    return out


def coresident_pairs(placement: dict) -> frozenset:
    """The unordered tenant pairs that shared at least one GPU.

    Pairs rather than a count, because a count says how many neighbours a tenant had and not which
    ones. **But this is still a lossy summary, and the store holds the case that proves it.** The
    two placements that measured 1.89x apart have the *same* pair set -- {(r2,t2)} -- and the same
    degree; they differ in that one shares both of r2's GPUs with t2 and the other shares one. No
    summary of structure separates them. That is why a `RetrievedRun` carries the assignment itself
    and `Relation` names the tenants whose GPU sets differ, and why this function is a convenience
    for describing a run rather than the thing a run is compared by.
    """
    names = sorted(placement)
    out = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(placement[a]) & set(placement[b]):
                out.add((a, b))
    return frozenset(out)


@dataclass(frozen=True)
class Relation:
    """How a prior run relates to what was asked. Stated, never scored.

    Every field is a fact that can be checked against the two placements. There is deliberately no
    number here and no ordering over relations: the moment retrieval ranks by "closeness" it has
    written the equivalence class impl §10 refuses to write, and the reasoner is back to trusting a
    rule it cannot see. Identity is the one exception, because identity is not similarity -- two
    runs of the same assignment are the same experiment, and saying so takes no judgement.
    """

    tenants_in_common: tuple = ()
    same_placement: bool = False
    alike: tuple = ()                  # tenants whose GPU set is identical in both
    differs: tuple = ()                # (tenant, run's GPUs, candidate's GPUs) where they differ
    shared_pairs: tuple = ()           # co-resident pairs present in both
    run_only_pairs: tuple = ()         # pairs this run had and the candidate does not
    candidate_only_pairs: tuple = ()   # pairs the candidate has and this run did not
    caps_differ: tuple = ()            # (tenant, run's SM cap, candidate's) where they differ
    placement_resolved: bool = True

    @property
    def text(self) -> str:
        if not self.placement_resolved:
            return ("its placement cannot be resolved on this node, so its numbers bear here but "
                    "its structure cannot be compared")
        if self.same_placement:
            return "THE SAME PLACEMENT"
        parts = []
        if self.alike:
            parts.append(f"same GPUs for {', '.join(self.alike)}")
        # Named per tenant, with both GPU sets, because this is the level at which the store's two
        # 1.89x-apart placements actually differ -- their degree and their pair set are identical.
        for name, mine, theirs in self.differs:
            parts.append(f"{name} on {_gpus(mine)} here vs {_gpus(theirs)} in the candidate")
        if self.shared_pairs:
            parts.append(f"also shared {_pairs(self.shared_pairs)}")
        if self.run_only_pairs:
            parts.append(f"shared {_pairs(self.run_only_pairs)} where the candidate does not")
        if self.candidate_only_pairs:
            parts.append(f"did not share {_pairs(self.candidate_only_pairs)}")
        for name, mine, theirs in self.caps_differ:
            parts.append(f"{name} SM cap {_cap(mine)} here vs {_cap(theirs)} in the candidate")
        if not parts:
            parts.append("no GPU set and no co-resident pair in common")
        return "; ".join(parts)


def _pairs(pairs) -> str:
    return " ".join(f"({a},{b})" for a, b in pairs)


def _gpus(gpus) -> str:
    return ",".join(str(g) for g in sorted(gpus))


def _cap(cap) -> str:
    return "none" if cap is None else f"{cap}%"


@dataclass(frozen=True)
class TenantRun:
    """One tenant's numbers in one prior run, with what is required of it *now* beside them.

    `met` is computed here, at read time, against the targets in force in this session -- never
    read from the record. A record stamped met/missed when it was written would be answering a
    question about a threshold that has since changed, and design §2 makes the target an input to
    the search, so it can change between two reads of the same run.
    """

    name: str
    gpus: tuple = ()
    model_id: str | None = None
    tp: int | None = None
    sm_cap: int | None = None           # impl §16; None is uncapped
    rate_hz: float | None = None
    shape: tuple | None = None          # (isl, osl) this run actually offered
    coresident: int | None = None
    itl_mean_ms: float | None = None
    ttft_p95_ms: float | None = None
    latency_note: str | None = None     # why there is no latency, when there is none
    # Where the driver's in-flight count started and ended, which is this row's account of whether
    # the backlog drained. A preemption count and a KV high-water mark were here until impl §14.6;
    # both were the server's account of itself, and a reasoning agent reading them was the last
    # consumer keeping the scrape alive.
    in_flight_from: float | None = None
    in_flight_to: float | None = None
    met: bool | None = None             # True, False, or None for "this run cannot answer"
    met_detail: str = ""

    @property
    def line(self) -> str:
        itl = "suppressed" if self.itl_mean_ms is None else f"{self.itl_mean_ms:.3f}ms"
        shape = "?" if self.shape is None else f"{self.shape[0]}+{self.shape[1]}"
        rate = "?" if self.rate_hz is None else f"{self.rate_hz:g}Hz"
        verdict = {True: "met", False: "MISSED", None: "no verdict"}[self.met]
        cap = "" if self.sm_cap is None else f" sm-cap {self.sm_cap}%"
        return (f"{self.name:<8} tp{self.tp or '?'}{cap} gpus={list(self.gpus)!s:<12} "
                f"{rate:>7} {shape:>9}  coresident={self.coresident if self.coresident is not None else '?'}"
                f"  ITL {itl:>10}  {verdict} ({self.met_detail})")


@dataclass(frozen=True)
class RetrievedRun:
    """One prior run, intact.

    Intact is the point: the assignment it actually ran, each tenant's own offered rate and shape,
    each measured number, and the signals that qualify them. Nothing is folded into a summary
    statistic, because folding is what destroys the structure that decides the answer.
    """

    run_id: str
    verdict: str
    relation: Relation
    tenants: dict = field(default_factory=dict)      # name -> TenantRun
    placement: dict | None = None
    sm_caps: dict = field(default_factory=dict)      # impl §16; part of the placement
    qualifiers: tuple = ()                           # §6.5 signals about the whole run
    latency_evidence: bool = True                    # False for an infeasible record
    source_path: str | None = None

    @property
    def summary_lines(self) -> list:
        """`recall`'s breadth: one header and one line per tenant.

        Depth is `read_run` (impl §11.1), and the split is deliberate rather than a cap: impl §13
        notes that a cap on how many runs come back would be the similarity rule §10 refuses to
        write, only cruder -- a truncation by recency or count instead of by relevance. So every
        run that bears on the query is listed, and the reasoner spends its own context deciding
        which to open.
        """
        head = f"  {self.run_id}  [{self.relation.text}]"
        if not self.latency_evidence:
            head += f"  -- {self.verdict}: evidence about feasibility, not about latency"
        out = [head]
        out += [f"    {t.line}" for _, t in sorted(self.tenants.items())]
        for q in self.qualifiers:
            out.append(f"    ! {q}")
        return out


@dataclass(frozen=True)
class Evidence:
    """What retrieval found, including when it found nothing.

    `no evidence` is a first-class answer, not an empty list. An empty result reads as agreement --
    "nothing contradicts this candidate" -- which is the opposite of what it means, and it is
    precisely the answer a predictor cannot give: a lookup key returns a confident number for a
    neighbourhood nothing in the store resembles and has no way to say so.
    """

    asked: str
    runs: tuple = ()
    considered: int = 0
    excluded: dict = field(default_factory=dict)     # reason -> (run_id, ...)
    repeats: tuple = ()                              # groups of runs of one identical placement
    calibration_note: str | None = None

    @property
    def has_evidence(self) -> bool:
        return bool(self.runs)

    @property
    def latency_runs(self) -> tuple:
        return tuple(r for r in self.runs if r.latency_evidence)

    def lines(self) -> list:
        out = [f"prior runs bearing on {self.asked}:"]
        if not self.runs:
            # Said in these words on purpose. The reasoner has to be able to distinguish this
            # from a weak result, and from a strong one.
            out.append("  NO EVIDENCE. Nothing in the store under this validity key bears on it, "
                       "so anything said about what it will measure is a guess and should be "
                       "labelled as one.")
        for run in self.runs:
            out += run.summary_lines
        for group in self.repeats:
            out += _repeat_lines(group, self.calibration_note)
        if self.excluded:
            out.append(f"  {self.considered} record(s) considered; set aside:")
            for reason, ids in sorted(self.excluded.items()):
                out.append(f"    {len(ids)}: {reason}")
                out.append(f"       {', '.join(ids)}")
        return out


def _repeat_lines(group, calibration_note) -> list:
    """Runs of one identical placement, side by side, never averaged (impl §10).

    Where two runs of the same thing disagree, both figures go in front of the reasoner. The
    calibrated between-run half-width is printed beside the observed spread when the store has
    one, as a measured resolution rather than as a test -- nothing here decides whether a
    difference is real, which is the judgement being delegated.
    """
    ids = [r.run_id for r in group]
    out = [f"  the same placement was measured {len(group)} times: {', '.join(ids)}"]
    for name in sorted(group[0].tenants):
        vals = [r.tenants[name].itl_mean_ms for r in group
                if name in r.tenants and r.tenants[name].itl_mean_ms is not None]
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        shown = ", ".join(f"{v:.3f}" for v in vals)
        out.append(f"    {name:<8} ITL {shown} ms  -> spread {spread:.3f} ms")
    if calibration_note:
        out.append(f"    {calibration_note}")
    return out


# --- judging a run against the targets in force now --------------------------

def targets_met(obs: Observation, targets: dict) -> dict:
    """`{tenant: True | False | None}`, judged now, never stamped into the record.

    `targets` maps a tenant name to something with `ttft_p95_ms` and `itl_mean_ms` attributes, or
    to None for a tenant that stated no target. `agent.spec.TenantSlo` is that something; it is
    duck-typed here so retrieval does not depend on the agent package.

    **None is neither a pass nor a failure**, and collapsing it either way is a real error. It
    means this run cannot answer for that tenant: no target was stated, or the tenant is not in the
    record, or the harness suppressed the latency it would be judged on (impl §6.3). Collapsing
    None into False reports a fleet as missing targets nobody set. Collapsing it into True
    certifies a fleet on a measurement that does not exist.

    A tenant that stated no target is None here for a further reason: impl §11.2 forbids declaring
    that such a tenant holds one, and this is the function that would otherwise let it.
    """
    return {name: _met_one(obs, name, slo)[0] for name, slo in targets.items()}


def required_relaxation(obs: Observation, targets: dict) -> dict:
    """How much each target would have to move for this record to meet every one of them.

    `{"factor": f, "per_tenant": {name: {...}}, "blocked": (name, ...)}`. The factor is the largest
    `measured / ceiling` over every tenant and every stated ceiling, so 1.029 means "one target
    exceeded by 2.9%, none by more" and anything <= 1.0 means the record already meets them all.

    This exists because "3 GPUs missed" and "3 GPUs missed by 1.45 ms on one tenant out of five" are
    different findings, and the second is the one somebody can act on -- a target is an input to this
    search (design §2, `TenantSlo`), so what a denser placement costs in requirements is a number
    the person who wrote the requirement should see. It is arithmetic over a record already taken:
    no measurement, and nothing here relaxes anything. It reports what relaxing would take.

    `blocked` names the tenants no relaxation can help: no target stated, not in this record, or a
    suppressed latency. A suppressed latency is the important one -- the delivery gate failed, so
    the run served less than it was asked to and measured nothing about latency. Moving a latency
    ceiling cannot supply a measurement that was never taken, and a factor that quietly ignored
    those tenants would read as "this placement is 2.9% away" when the truth is that it is
    unmeasured. When `blocked` is non-empty the factor speaks only for the tenants it could judge.
    """
    per, blocked, factor = {}, [], 1.0
    for name, slo in sorted(targets.items()):
        verdict, _ = _met_one(obs, name, slo)
        if verdict is None:
            blocked.append(name)
            continue
        got = obs.per_tenant[name]
        for attr, label in (("itl_mean_ms", "ITL mean"), ("ttft_p95_ms", "TTFT p95")):
            ceiling = getattr(slo, attr, None)
            if ceiling is None:
                continue
            value = getattr(got, attr)          # cannot raise: _met_one gave a verdict
            ratio = float(value) / float(ceiling)
            if ratio <= 1.0:
                continue
            # Per tenant, the binding ceiling only. A tenant over on both is over by the worse of
            # the two, and reporting the slacker one would understate what it would take.
            if name not in per or ratio > per[name]["factor"]:
                per[name] = {"factor": round(ratio, 6), "metric": label,
                             "measured": float(value), "ceiling": float(ceiling),
                             "over_by": round(float(value) - float(ceiling), 6)}
            factor = max(factor, ratio)
    return {"factor": round(factor, 6), "per_tenant": per, "blocked": tuple(blocked)}


def _met_one(obs: Observation, name: str, slo) -> tuple:
    """`(verdict, detail)` for one tenant. The detail is what a reader needs to check it."""
    if slo is None:
        return None, "no target stated, so no measurement can certify it"
    if name not in obs.per_tenant:
        return None, "this run did not include this tenant"
    got = obs.per_tenant[name]
    checks, missing = [], []
    for attr, label in (("itl_mean_ms", "ITL mean"), ("ttft_p95_ms", "TTFT p95")):
        ceiling = getattr(slo, attr, None)
        if ceiling is None:
            continue
        try:
            value = getattr(got, attr)
        except SuppressedMetric:
            # The schema raises rather than returning None, and catching it here is the whole
            # reason it raises: the alternative is a zero flowing into a comparison, and an
            # under-delivering run certifying itself as the fastest in the store.
            missing.append(f"{label} was suppressed ({got.latency_suppressed})")
            continue
        checks.append((label, value, ceiling))
    if missing or not checks:
        return None, "; ".join(missing) or "no comparable target"
    detail = ", ".join(f"{label} {v:.3f} vs {c:g}" for label, v, c in checks)
    return all(v <= c for _, v, c in checks), detail


def target_lines(obs: Observation, targets: dict) -> list:
    """The margin block for one record, against the targets in force now. For `read_run`."""
    out = ["  measured against the targets in force now:"]
    for name in sorted(targets):
        verdict, detail = _met_one(obs, name, targets[name])
        label = {True: "met", False: "MISSED", None: "no verdict"}[verdict]
        out.append(f"    {name:<8} {label:<10} {detail}")
    return out


def _figure(t, attr: str, fmt: str) -> str:
    """One latency figure as text, or "not recorded": a record can carry an ITL mean and still lack
    a figure beside it, and the run is read either way."""
    try:
        return fmt.format(getattr(t, attr))
    except SuppressedMetric:
        return "not recorded"


def read_run(obs: Observation, targets: dict, gpu_of: dict | None = None) -> list:
    """One record in full: raw metrics, margins against the targets in force now, §6.5 signals.

    The depth half of the breadth/depth split (impl §11.1). `recall` says which runs exist and what
    they measured in one line each; this is what a reasoner opens when a line looks decisive, and it
    is what makes the absence of a cap on `recall` affordable.

    Raw numbers first and derived judgements after, in that order, with the qualifying signals last
    where they cannot be skimmed past. Nothing here is rounded away or omitted for brevity: the
    reader is deciding whether this run speaks for a candidate, and the thing that decides it is
    routinely a detail -- a preemption count, a delivered fraction, one GPU at a power cap.
    """
    # Two nodes, both named and both labelled. `obs.node` alone was here once, and a live search
    # read it as the machine under test: the reasoner hedged a reproduced result on the grounds that
    # "all four runs executed on <node>, not the <node> named in the problem" -- the harness Job's
    # cpu-compute node against the leased g6e, which are different machines by design. The GPUs are
    # in provenance, written by the agent from the lease (`loop.py:_provenance`), and an unlabelled
    # `node=` invites exactly that reading. `inspect.py` already separates them for a human reader.
    out = [f"run {obs.run_id}  verdict={obs.verdict}  key={obs.validity_key_hash}",
           f"  gpu node under test={obs.provenance.get('gpu_node') or 'not recorded'}"
           f"  load driver ran on={obs.node or '?'} (a cpu node, not the GPUs')",
           f"  window={obs.window_s:.1f}s  steady state at "
           + ("not recorded" if obs.steady_state_reached_s is None
              else f"{obs.steady_state_reached_s:.1f}s")]
    if not obs.usable_as_evidence:
        why = ("it was not measured" if not obs.measured
               else "its gauges may be phase-locked to the arrival schedule")
        out.append(f"  NOT LATENCY EVIDENCE: {why}. What it can still settle is whether this "
                   "configuration deploys and serves at all")
    placement = placement_of(obs, gpu_of)
    if placement is None:
        out.append("  its GPUs cannot be resolved on this node, so its placement cannot be "
                   "compared with a candidate's -- only its numbers carry over")
    else:
        degrees = coresidency(placement)
        pairs = coresident_pairs(placement)
        out.append(f"  placement: {_asked_text(placement, (), sm_caps_of(obs))}")
        out.append(f"  co-resident pairs: {_pairs(sorted(pairs)) or 'none, every tenant alone'}")
        out.append(f"  co-residency degree: {dict(sorted(degrees.items()))}")

    for name in sorted(obs.per_tenant):
        t = obs.per_tenant[name]
        cfg = obs.tenant_config(name)
        shape = obs.sequence_shape(name)
        out.append(f"  {name}:")
        out.append(f"    offered   {obs.rate_hz(name)} Hz, "
                   f"shape {'not recorded per tenant' if shape is None else f'{shape[0]}+{shape[1]}'}"
                   f", tp{cfg.get('tp')}, envelope {cfg.get('envelope_gib')} GiB")
        out.append(f"    delivered {t.requests_completed}/{t.requests_dispatched} completed, "
                   f"{t.requests_failed} failed, fraction {t.delivered_fraction} "
                   f"(threshold {t.delivered_fraction_threshold})")
        if t.latency_available:
            out.append(f"    latency   ITL mean {t.itl_mean_ms:.4f} ms "
                       f"({_figure(t, 'itl_ci_halfwidth_ms', '+-{:.4f}')} within-run), "
                       f"TTFT mean {_figure(t, 'ttft_mean_ms', '{:.3f} ms')}, "
                       f"p95 {_figure(t, 'ttft_p95_ms', '{:.3f} ms')}")
        else:
            out.append(f"    latency   SUPPRESSED: {t.latency_suppressed}")
        # This was a `server` line: KV usage, pool size, running, waiting and preemptions, all read
        # off the tenant's own /metrics. It is now the driver's own backlog level (impl §14.6), which
        # answers the same question -- did work pile up -- without asking the server about itself.
        out.append(f"    backlog   in flight {t.in_flight_first_quarter_mean} -> "
                   f"{t.in_flight_last_quarter_mean} (mean {t.in_flight_mean}, "
                   f"max {t.in_flight_max}, over {t.in_flight_samples} samples)")

    out += target_lines(obs, targets)

    quals = qualifiers(obs)
    out.append("  qualifying signals (impl §6.5):")
    out += [f"    ! {q}" for q in quals] or ["    none"]
    chosen = obs.provenance.get("chosen_by") or obs.provenance.get("policy_kind")
    if chosen:
        out.append(f"  chosen by: {chosen}")
        for key in ("reasoner_why", "why", "expectation", "expectation_outcome"):
            if obs.provenance.get(key):
                out.append(f"    {key}: {obs.provenance[key]}")
    return out


# --- the query ---------------------------------------------------------------

def recall(observations, *, targets: dict, assignment: dict | None = None,
           tenants=(), gpu_of: dict | None = None, key_hash: str | None = None,
           calibration=None, sm_caps: dict | None = None) -> Evidence:
    """Prior runs bearing on a candidate placement, or on named tenants.

    `assignment` is `{tenant: (gpu, ...)}` in the same label space as `gpu_of`'s values -- the
    solver's ordinals, normally. `tenants` narrows a query that names no placement. Give one or
    the other; giving neither asks about the whole fleet, which is legitimate and returns
    everything that shares a tenant with the store. `sm_caps` is the candidate's SM ceilings
    (impl §16), and only matters beside an `assignment`: a run is the same placement only with the
    same caps.

    **What is included is stated, not scored.** Every record under this validity key that names at
    least one tenant asked about comes back, with the structural relation spelled out per run.
    There is no distance function and no cap. What is excluded is excluded for a reason that can be
    checked -- a different validity key, no tenant in common, gauges that may be phase-locked --
    and every exclusion is reported with the run ids it covered.

    Infeasible records come back too, marked as evidence about feasibility rather than about
    latency. They are what stops a placement that already failed to deploy being proposed again
    (impl §8), and dropping them here would hide that. They are marked rather than mixed because
    §10 is explicit that retrieval feeds proposals and cuts and never feasibility: a wrong belief
    must cost iterations, not correctness.
    """
    sm_caps = dict(sm_caps or {})
    asked = _asked_text(assignment, tenants, sm_caps)
    wanted = set(tenants) or (set(assignment) if assignment else set(targets))
    cand_pairs = coresident_pairs(assignment) if assignment else frozenset()

    runs, excluded, considered = [], {}, 0
    for obs in observations:
        considered += 1
        if key_hash is not None and obs.validity_key_hash != key_hash:
            excluded.setdefault(EXCLUDED_KEY, []).append(obs.run_id)
            continue
        common = tuple(sorted(set(obs.per_tenant) & wanted))
        if not common:
            excluded.setdefault(EXCLUDED_TENANTS, []).append(obs.run_id)
            continue
        if obs.measured and obs.sampling_suspect:
            # Excluded rather than marked: unlike an infeasible record, this one *looks* like a
            # measurement. Live run 2 read occupancy of exactly 0.0 on every sample while DCGM had
            # the GPU 47% busy, and nothing on its face said so.
            excluded.setdefault(EXCLUDED_SUSPECT, []).append(obs.run_id)
            continue
        runs.append(_retrieve_one(obs, targets, assignment, cand_pairs, gpu_of, common, sm_caps))

    # Run-id order, deliberately. It is arbitrary, which is the property wanted: any other order
    # would be a ranking, and a ranking is the judgement being delegated. The one thing pulled out
    # of it is exact identity, below, because that is a fact rather than a similarity.
    runs.sort(key=lambda r: r.run_id)
    return Evidence(asked=asked, runs=tuple(runs), considered=considered,
                    excluded={k: tuple(v) for k, v in excluded.items()},
                    repeats=_repeat_groups(runs),
                    calibration_note=_calibration_note(runs, calibration))


def _asked_text(assignment, tenants, sm_caps=None) -> str:
    if assignment:
        caps = sm_caps or {}
        return " ".join(f"{n}:{','.join(str(g) for g in sorted(gs))}"
                        + (f"@{caps[n]}%" if n in caps else "")
                        for n, gs in sorted(assignment.items()))
    if tenants:
        return f"tenant(s) {', '.join(sorted(tenants))}"
    return "this fleet, any placement"


def _retrieve_one(obs, targets, assignment, cand_pairs, gpu_of, common, sm_caps) -> RetrievedRun:
    placement = placement_of(obs, gpu_of)
    caps = sm_caps_of(obs)
    degrees = coresidency(placement) if placement else {}
    relation = _relate(placement, assignment, cand_pairs, common, caps, sm_caps)

    tenants = {}
    for name in sorted(obs.per_tenant):
        t = obs.per_tenant[name]
        cfg = obs.tenant_config(name)
        verdict, detail = _met_one(obs, name, targets.get(name))
        itl = ttft = None
        note = t.latency_suppressed
        if t.latency_available:
            itl = t.itl_mean_ms
            try:
                ttft = t.ttft_p95_ms
            except SuppressedMetric:
                ttft = None
        tenants[name] = TenantRun(
            name=name, gpus=tuple((placement or {}).get(name, ())),
            model_id=cfg.get("model_id") or cfg.get("model"), tp=cfg.get("tp"),
            sm_cap=caps.get(name), rate_hz=obs.rate_hz(name), shape=obs.sequence_shape(name),
            coresident=degrees.get(name), itl_mean_ms=itl, ttft_p95_ms=ttft,
            latency_note=note, in_flight_from=t.in_flight_first_quarter_mean,
            in_flight_to=t.in_flight_last_quarter_mean,
            met=verdict, met_detail=detail)

    return RetrievedRun(run_id=obs.run_id, verdict=obs.verdict, relation=relation,
                        tenants=tenants, placement=placement, sm_caps=caps,
                        qualifiers=qualifiers(obs), latency_evidence=obs.usable_as_evidence,
                        source_path=obs.source_path)


def _relate(placement, assignment, cand_pairs, common, run_caps, cand_caps) -> Relation:
    if placement is None or assignment is None:
        return Relation(tenants_in_common=common, placement_resolved=placement is not None)
    run_pairs = coresident_pairs(placement)
    alike, differs = [], []
    for n in common:
        if n not in placement or n not in assignment:
            continue
        mine, theirs = set(placement[n]), set(assignment[n])
        (alike if mine == theirs else differs).append(
            n if mine == theirs else (n, tuple(sorted(mine)), tuple(sorted(theirs))))
    caps_differ = tuple((n, run_caps.get(n), cand_caps.get(n)) for n in common
                        if run_caps.get(n) != cand_caps.get(n))
    same = (set(placement) == set(assignment)
            and all(set(placement[n]) == set(assignment[n]) for n in assignment)
            and run_caps == cand_caps)
    return Relation(tenants_in_common=common, same_placement=same, alike=tuple(alike),
                    differs=tuple(differs), caps_differ=caps_differ,
                    shared_pairs=tuple(sorted(run_pairs & cand_pairs)),
                    run_only_pairs=tuple(sorted(run_pairs - cand_pairs)),
                    candidate_only_pairs=tuple(sorted(cand_pairs - run_pairs)))


def qualifiers(obs: Observation) -> tuple:
    """The §6.5 signals that qualify a run's numbers, as short statements.

    Printed with every run rather than only on request. A throttled GPU or a run that never
    reached steady state produces a latency that is about the cap or the ramp, and a reasoner
    comparing it with a clean run would be comparing two different experiments.
    """
    out = []
    throttled = sorted(u for u, g in obs.per_gpu.items() if g.throttled)
    if throttled:
        out.append(f"{len(throttled)} GPU(s) hit a power or thermal cap during this run: any "
                   "latency difference here may be the cap and not the placement")
    if obs.measured and obs.steady_state_reached_s is None:
        out.append("no steady-state timestamp, so the window may include the ramp")
    diverged = []
    for name in sorted(obs.per_tenant):
        t = obs.per_tenant[name]
        # A preemption count was the qualifier here, and it said the KV pool was too small for what
        # the tenant was offered -- so its ITL was partly memory rather than only contention. The
        # observable version of the same warning is the driver's own backlog climbing across a window
        # that is supposed to be settled: whatever the cause, the latency in this row was measured
        # while work was piling up, so it is not the row's steady-state latency (impl §14.6).
        a, b = t.in_flight_first_quarter_mean, t.in_flight_last_quarter_mean
        if a is not None and b is not None and a > 0.0 and b >= a * (1.0 + IN_FLIGHT_RISE_FRACTION):
            out.append(f"{name}: requests in flight climbed {a:.3g} -> {b:.3g} across the window, so "
                       "completions ran below arrivals and any latency here was measured while work "
                       "was accumulating")
        if t.chunk_divergence:
            diverged.append(f"{name} {t.chunk_divergence}")
        if not t.delivery_passed and obs.measured:
            out.append(f"{name}: delivered {t.delivered_fraction} of what was offered")
    if diverged:
        # One line for the whole run rather than one per tenant. It is the same signal about the
        # same observation path every time, and repeated four times per run it crowds out the
        # qualifiers that differ between runs -- which are the ones a reader is scanning for.
        out.append(f"tokens the server sent that the client never saw as chunks "
                   f"({', '.join(diverged)}): an unexplained divergence in the harness's "
                   "observation path, not in the placement")
    return tuple(out)


def _repeat_groups(runs) -> tuple:
    """Runs of one identical placement, grouped. The input to "both figures, not a summary"."""
    by_placement: dict = {}
    for r in runs:
        if r.placement is None or not r.latency_evidence:
            continue
        key = (tuple(sorted((n, tuple(sorted(g))) for n, g in r.placement.items())),
               tuple(sorted(r.sm_caps.items())))
        by_placement.setdefault(key, []).append(r)
    return tuple(tuple(v) for _, v in sorted(by_placement.items()) if len(v) > 1)


def _calibration_note(runs, calibration) -> str | None:
    """The measured resolution, stated beside a spread. Never applied to it."""
    if calibration is None:
        return None
    hw = calibration.tightest_ms
    if hw is None:
        return None
    return (f"the store's between-run ITL half-width is {hw:.4f} ms over {calibration.repeats} "
            "repeat(s): two runs of one placement differing by less than that differ by nothing "
            "measurable here. What follows from that is yours to decide")
