"""Measuring one tenant alone, at ascending TP degrees, until its target is met (impl §0).

This is the actuator for **minimal TP**, which §0 puts in the baseline rather than in the search.
The search decides which GPUs a fleet's tenants share; it never decides how many GPUs a tenant is
sharded across. So a fleet whose TP degrees are wider than they need to be has already conceded the
GPUs that co-location would otherwise have been credited with saving, and nothing inside the search
can notice. What settles a degree is a measurement taken *outside* the search, of one tenant, alone.

**Alone at the full envelope, and both halves matter.** Alone means no other tenant on those GPUs,
so no SM contention: this is the fastest the model can go on this hardware at this rate and shape.
The full envelope means the KV pool is as large as one GPU allows, so the tenant is not being
starved of cache either. A degree that misses the target under those conditions cannot be rescued by
any placement -- so the miss is permanent for that target, and the pass is the reference every
co-resident measurement is compared against ("vs solo").

**Ascending from the memory floor, stopping at the first pass.** The floor (`solver/baseline.py`) is
arithmetic and free; below it a shard does not fit on one GPU at any envelope, so there is nothing to
measure. Above it each degree costs a deploy-and-drive cycle, and the first one that passes is the
answer -- so the cost of establishing minimal TP is (degrees ruled out) + 1 measurements per tenant,
not a full sweep. A tenant whose floor is already the widest degree the node offers is resolved by
arithmetic alone and never measured, because there is no alternative to compare it against.

**Each round is its own validity key, and that is why the profile store exists.** TP is per tenant
and inside `ValidityKey`, so a round at TP=2 and the same tenant's round at TP=4 are two different
keys with two different `observations/` directories -- correctly, since they are different
experiments. The comparison across them therefore cannot live in either, and it lives in
`store/profile.py` under a key naming only the environment. The observation stays where an
observation belongs, append-only under its own key; the profile row is an index into them.

Nothing here decides anything about a fleet. It reuses `SearchAgent.execute` and
`PlacementSolver.solve_pinned` rather than reimplementing deploy-gate-drive-record, which is the
same reason `tools.measure` reuses them: a second implementation is a second thing to get wrong,
and the one that would be wrong is the one that writes to the store.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone

from ..solver.baseline import candidate_tp_degrees, memory_floor_tp, weight_floor_tp
from ..solver.cpsat import Infeasible, PlacementSolver
from ..store.profile import RESOLVED, SoloProfile, TpResolution, minimal_tp
from ..store.schema import MIN_SUBSAMPLES
from ..store.store import ObservationStore
from .deploy import NOT_READY_ENGINE
from .policy import Proposal
from .spec import AGENT_ENVELOPE_STEP_MIB, FleetSpec

log = logging.getLogger(__name__)

# Why a round was not measured. Distinct from `store.profile`'s statuses, which describe what the
# *store* can say; these describe what this run did.
DID_MEASURE = "measured"
ALREADY_PROFILED = "already profiled"
FORCED_BY_MEMORY = "forced by memory alone"
NOT_READY = "did not come up"
NOT_PLACEABLE = "no envelope fits"


def solo_fleet(fleet: FleetSpec, tenant, tp: int) -> FleetSpec:
    """The one-tenant fleet a profile round deploys, on the node the run holds.

    The workload block travels with it, because the delivery gate is what decides whether the
    round's latency may be read at all (impl §6.3) and a profile taken under a looser gate than the
    search's would not be comparable with it.
    """
    return FleetSpec(tenants=(dataclasses.replace(tenant, tp=int(tp)),),
                     node=fleet.node, workload=fleet.workload)


def solo_key(key, tenant, tp: int):
    """The validity key for one tenant at one degree, from the fleet's own key.

    Everything about the environment is inherited and every per-tenant map is narrowed to this
    tenant, so a profile round is keyed by the same recipe as any other run rather than by a
    parallel one. `alpha` is inherited too: if the fleet's rates are expressed as a fraction of a
    measured saturation rate, this round is driven at the same fraction.
    """
    return dataclasses.replace(
        key,
        sequence_shapes={tenant.name: (tenant.isl_tokens, tenant.osl_tokens,
                                       tenant.image_count, tenant.image_pixels)},
        tensor_parallel={tenant.name: int(tp)},
        serving={tenant.name: tenant.serving},
        offered_rates_hz={tenant.name: tenant.rate_hz},
        model_revisions=({tenant.served_name: tenant.model_revision}
                         if tenant.model_revision else {}))


def measured_reserve_gib(record: dict, tenant_name: str, device_total_gib: float,
                         util: float | None = None) -> float | None:
    """Per-shard memory the engine used that `gpu_memory_utilization` did not account for.

    `nvidia-smi resident - util x device_total`, averaged over the shard's own GPUs, which is the
    same derivation our own measurement runs used to calibrate `mps_reserve_gib` -- deliberately, so
    the two are comparable. It includes the MPS server's own context and allocator rounding, so it
    is only ever compared against figures derived this way.

    `device_total_gib` must be **the figure the driver reports and the engine multiplied its
    utilisation by**, because that product is literally the budget the engine allocated inside; any
    smaller "usable" figure here understates that budget and the whole difference lands in this
    residual, where it reads as overhead the engine incurred rather than as the wrong denominator.
    It is the same requirement `memory.gpu_memory_utilization` states, for the same reason, and it
    is why there is now only one such number (`NodeSpec.device_total_gib`).

    Valid **only** because the tenant was alone on those GPUs: with a co-tenant, `fb_used` is the
    sum of both and MPS gives no per-process attribution. That is what makes a solo round the only
    place this can be measured, and why nothing tries to read it out of a search's records.

    None when the record does not carry the fields, rather than a zero -- a missing measurement and
    a measured absence of overhead are different things, and one of them would tighten a safety
    margin. **Also None when the derivation comes out at or below zero**, which is the same
    statement arrived at from the other side: the engine claimed less than its budget, so this run
    put no lower bound on the overhead and has measured nothing about it. Storing the negative
    number would be worse than storing nothing, because `for_fleet_file` offers it to a person to
    paste and the solver reads it as extra room.
    """
    per_gpu = record.get("per_gpu") or {}
    uuids, envelope = None, None
    for t in (record.get("configuration") or {}).get("tenants") or []:
        if t.get("name") == tenant_name:
            uuids = t.get("gpu_uuids") or []
            if util is None:
                util = t.get("util")
            envelope = t.get("envelope_gib")
            break
    else:
        return None
    if not uuids:
        return None
    if util is None:
        if envelope is None:
            return None
        # The chart is given a utilization fraction and the record carries the envelope it came
        # from, so recovering one from the other is arithmetic, not an estimate. Only a fallback:
        # the placement's own `util` is what the chart was actually given, and the caller has it.
        util = float(envelope) / float(device_total_gib)
    extras = []
    for u in uuids:
        used_mib = (per_gpu.get(u) or {}).get("fb_used_mean")
        if used_mib is None:
            continue
        extras.append(float(used_mib) / 1024.0 - float(util) * float(device_total_gib))
    if not extras:
        return None
    reserve = round(sum(extras) / len(extras), 4)
    if reserve <= 0.0:
        # Resident memory came in at or under the budget, so this run saw no overhead beyond it and
        # has therefore measured nothing about how much there is. Reported as unmeasured for the
        # same reason a missing field is: `for_fleet_file` offers these numbers to a person to paste
        # into `models:`, where a negative reserve does not mean "no overhead" -- it *widens* the KV
        # pool the solver will price, pushing a later placement past what the GPU holds. The safe
        # default stays in force instead. It is not a failed round: the latency is unaffected.
        log.info("%s: derived reserve %.4f GiB is not positive, so this round measured no overhead "
                 "beyond the envelope and the row states none. The default stands.",
                 tenant_name, reserve)
        return None
    return reserve


def saturation_evidence(record: dict, tenant_name: str) -> dict:
    """The numbers that decide whether a round that produced no latency still ruled its degree out.

    This is deliberately an *extractor*, not a judge: it copies raw counts and gauges out of the
    record and applies no thresholds. `SoloProfile.too_narrow_reason` does the judging, so a row
    written today is re-judged under today's rule when it is read back rather than being frozen at
    whatever the rule was when it was written. That is the opposite of the `sampling_suspect`
    convention two dozen lines below, and the difference is not an inconsistency: `MIN_SUBSAMPLES`
    is applied to a number the row does not carry, so that conclusion *has* to be stored, while
    every number here is one this function can put in the row.

    Everything gathered here is client-side or engine-blind, with no exceptions. That is a Phase B
    requirement rather than a preference: the walk's authority to widen must not rest on gauges only
    one engine publishes, or the second engine arrives and the walk silently loses the ability to
    widen at all. There *were* four exceptions -- `num_requests_waiting_mean`,
    `num_requests_running_mean`, `kv_usage_max` and `preemptions`, each labelled here as engine
    specific and defended as "the sharpest backlog evidence when an engine does publish them". They
    were also, as it turned out, the first three things `too_narrow_reason` consulted, so the
    authority to widen did rest on them. They are gone (impl §14.6) and the in-flight direction
    below answers the same question from the driver's own books.
    """
    per = (record.get("per_tenant") or {}).get(tenant_name) or {}
    ev = {
        # -- client side. The load driver's own books, true of any server that speaks HTTP.
        # `requests_launched` is what the driver sent; `requests_dispatched` is what it got a
        # result for, and reads 0 under the overload this evidence is about. Both are kept
        # because the pair is itself the shortfall.
        "requests_launched": per.get("requests_launched"),
        "requests_dispatched": per.get("requests_dispatched"),
        "requests_completed": per.get("requests_completed"),
        "requests_failed": per.get("requests_failed"),
        # Whether the driver kept its own schedule. Without this the offered rate is a claim: a
        # driver that fell behind offered less load than the fleet file says, and the server it was
        # pointed at was never actually asked for the rate the row is about to be judged against.
        "schedule_lag_max_ms": per.get("schedule_lag_max_ms"),
        # -- backlog, from the same client-side series the steady-state gate reads. Load is offered
        # open-loop, so completion below arrival shows up as the in-flight count climbing, by
        # arithmetic rather than by inference. The two quarter means are the direction; the whole-
        # window mean and max are context for a reader, and `in_flight_samples` is the guard that
        # stops an empty series from reading as a flat one.
        "in_flight_mean": per.get("in_flight_mean"),
        "in_flight_max": per.get("in_flight_max"),
        "in_flight_first_quarter_mean": per.get("in_flight_first_quarter_mean"),
        "in_flight_last_quarter_mean": per.get("in_flight_last_quarter_mean"),
        "in_flight_samples": per.get("in_flight_samples"),
    }
    # Whether the request shape was priced by the server itself. A probe that landed exactly proves
    # the server accepted this payload and told us what it costs -- so a round that then delivered
    # nothing was refused service under load, not refused at the door. Without it, "nothing came
    # back" is equally well explained by a request the server would never have accepted.
    wl = ((record.get("workload") or {}).get("tenants") or {}).get(tenant_name) or {}
    ev["isl_exact"] = wl.get("isl_exact")
    ev["isl_calibrated"] = wl.get("isl_calibrated")

    # The engine-blind busy signal: DCGM reads the device, not the server. This is what lets the
    # walk keep its authority across engines -- a card at 99% engine-active is doing work whoever
    # asked it to. Averaged over the shard's own GPUs, found the same way `measured_reserve_gib`
    # finds them, so a co-tenant's card can never be read as this tenant's.
    uuids = []
    for t in (record.get("configuration") or {}).get("tenants") or []:
        if t.get("name") == tenant_name:
            uuids = t.get("gpu_uuids") or []
            break
    per_gpu = record.get("per_gpu") or {}
    for field_name in ("gr_engine_active_mean", "sm_active_mean"):
        vals = [float(per_gpu[u][field_name]) for u in uuids
                if (per_gpu.get(u) or {}).get(field_name) is not None]
        ev[field_name] = round(sum(vals) / len(vals), 4) if vals else None
    return ev


def profile_from_record(record: dict, tenant, tp: int, *, envelope_gib: float,
                        device_total_gib: float, run_id: str, run_token: str = "",
                        key_hash: str = "", gpu_ids=(), util: float | None = None,
                        measured_at: str | None = None, descent: dict | None = None,
                        boot_failed: dict | None = None) -> SoloProfile:
    """One `SoloProfile` row out of one harness record.

    Latency comes back as None when the harness suppressed it -- a failed delivery gate -- and is
    left as None rather than filled in. `SoloProfile.meets` then answers None for that row, so a
    suppressed round cannot rule a TP degree out. The alternative, reading it as a miss, would push
    the tenant to a wider shard count on the strength of a measurement that does not exist.

    `measured_at` is this machine's clock at the moment the row is built, because a §6 record
    carries no wall-clock time of its own -- it dates the *row*, not the window, and nothing reads
    it as a measurement. The run id and the observation key hash are what tie the row to the
    window it came from.
    """
    per = (record.get("per_tenant") or {}).get(tenant.name) or {}
    delivered = per.get("delivered_fraction")
    threshold = per.get("delivered_fraction_threshold")
    passed = (None if delivered is None or threshold is None
              else float(delivered) >= float(threshold))
    if measured_at is None:
        measured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return SoloProfile(
        model_id=tenant.model_id, tp=int(tp), isl_tokens=tenant.isl_tokens,
        osl_tokens=tenant.osl_tokens, rate_hz=tenant.rate_hz, envelope_gib=round(envelope_gib, 4),
        # Read off the tenant that was actually deployed, so the row's identity and the engine that
        # produced it cannot disagree -- the renderer builds vLLM's command line from these same
        # fields (`store/serving.py`).
        serving=tenant.serving,
        # Read off the deployed tenant for `serving`'s reason: the row must say what was sent, and
        # the harness's payload is built from these same two fields (`render.py`).
        image_count=tenant.image_count, image_pixels=tenant.image_pixels,
        itl_mean_ms=per.get("itl_mean_ms"), ttft_p95_ms=per.get("ttft_p95_ms"),
        delivered_fraction=delivered, delivery_passed=passed,
        # Always None, and permanently so rather than pending a better record. The reserve below is
        # resident memory minus the utilisation budget -- a difference of two numbers the driver and
        # the placement both state. The activation term is inside that budget, and the engine fills
        # what activation leaves with KV cache, so from outside the budget the two are one number.
        # Separating them needs the engine's own account of its start-up memory, which is not
        # evidence this design reads (impl §14.2). The safe fallback carries it instead, and what
        # that costs is visible: dropping the fitted figures from the shipped fleets raised the floor
        # envelopes by 0.23 to 0.81 GiB/shard and moved no fleet's GPU count. It did cost one
        # arrangement, which is the honest way to state the price: in `glm-air-anchor-riders` the
        # largest rider now needs a sixth 1 GiB envelope step and so pairs with no other rider on an
        # anchor shard, and the full enumeration falls from 808 feasible placements to 600.
        activation_gib=None,
        reserve_gib=measured_reserve_gib(record, tenant.name, device_total_gib, util=util),
        # The conditions this row was DRIVEN under, taken from the provenance the agent wrote when it
        # chose the placement (`loop._provenance`) rather than from anything the engine said. They
        # are what let `_vs_solo` tell a comparison from a subtraction: the reserve above is measured
        # BY a solo round, so the first row at a degree is necessarily driven on the default and
        # every later run on the measured one, and a ratio taken across that boundary is not a
        # controlled comparison. See `SoloProfile.kv_tokens`.
        kv_tokens=((record.get("provenance") or {}).get("kv_tokens") or {}).get(tenant.name),
        reserve_source=(((record.get("provenance") or {}).get("mps_reserve_source") or {})
                        .get(tenant.name) or ""),
        # `provenance.gpu_node`, not the record's own `node`. They are different machines, and
        # `loop.py:_provenance` writes the former precisely because of it: `node` is where the
        # harness Job's pod ran, a cpu-compute node Karpenter reclaims within the hour, while the
        # GPUs under test were on the node the agent held. This row's whole provenance is
        # (node, gpu_ids), so taking the harness's node makes `gpu_ids: [0]` point at a machine the
        # measurement never touched -- and by the time anyone reads the row, at no machine at all.
        # The fallback covers a synthesized deploy-failure record, whose top-level `node` IS the
        # GPU node because the agent wrote it.
        run_id=run_id, run_token=run_token,
        gpu_node=((record.get("provenance") or {}).get("gpu_node")
                  or record.get("node") or ""),
        gpu_ids=tuple(gpu_ids), observation_key_hash=key_hash,
        measured_at=measured_at,
        # `verdict` and `sampling_suspect` are not decoration: `SoloProfile.unusable_reason` reads
        # both, and they are what stop a run the harness declined to vouch for from setting minimal
        # TP. `sampling_suspect` is derived here rather than stored raw because the row should carry
        # the *conclusion* -- a reader of the row cannot re-apply MIN_SUBSAMPLES to a number that is
        # not there, and rows written before this existed carry neither.
        notes={"verdict": record.get("verdict"),
               "sampling_suspect": (float((record.get("sampling") or {})
                                          .get("subsamples_per_sample_mean") or 0.0)
                                    < MIN_SUBSAMPLES),
               "latency_suppressed": per.get("latency_suppressed"),
               "reserve_derivation": "fb_used_mean - util x device_total_gib, mean over the shard's "
                                     "GPUs, tenant alone on them",
               # Raw, unjudged, and here so the ROW alone can answer whether its degree was too
               # narrow. Without it that question could only be asked while the record was still to
               # hand, which is the wrong lifetime: the walk reads rows, including rows a previous
               # run wrote. See `SoloProfile.too_narrow_reason`.
               "saturation": saturation_evidence(record, tenant.name),
               # Present only on a descent round, and it is what says this row was measured with a KV
               # pool below the minimum its own fleet file asks for. That is the experiment: the
               # stated minimum is a derived number (concurrency x span) and the descent is asking
               # whether it was over-stated, so both figures are kept and neither is judged here.
               "descent": descent,
               # Present only when a boot failure is evidence about this DEGREE rather than about the
               # run -- see `SoloProfile.too_narrow_reason`, which is the only reader.
               "boot_failed": boot_failed,
               "solo": True})


@dataclasses.dataclass
class Round:
    """What one (tenant, TP degree) round did. One per attempt, measured or not."""

    tenant: str
    tp: int
    status: str
    profile: SoloProfile | None = None
    run_id: str = ""
    detail: str = ""
    # The readiness gate's own classification when this round did not come up (`agent/deploy.py`:
    # `engine_failed`, `verification_failed`, `timeout`, `unschedulable`), and "" otherwise. It is
    # what separates a boot that tested this degree and failed from a run that broke before testing
    # it, and those two have opposite consequences for the walk.
    gate: str = ""
    # True for a round the DESCENT spent, below the degree the ascent stopped at (impl §14.2). Worth
    # keeping beside the degree because a descent round is not the same experiment as an ascent one:
    # its KV floor was relaxed, so it was driven with a smaller pool than its fleet file asks for.
    descent: bool = False

    def meets(self, slo):
        """Whether this round's row meets a target, or None when it cannot say."""
        return None if self.profile is None else self.profile.meets(slo)

    def as_dict(self) -> dict:
        return {"tenant": self.tenant, "tp": self.tp, "status": self.status,
                "run_id": self.run_id, "detail": self.detail, "gate": self.gate,
                "descent": self.descent,
                "profile": None if self.profile is None else self.profile.to_dict()}


class SoloProfiler:
    """Drives profile rounds for a fleet's tenants and files the rows they produce.

    `make_agent(solo_fleet, store)` returns a `SearchAgent` bound to this run's lease -- passed in
    rather than constructed here, because the deployer, the harness and the lease are the CLI's to
    assemble and this class has no business knowing how a Helm release is named.
    """

    def __init__(self, fleet: FleetSpec, key, hardware, profiles, *, make_agent,
                 store_root, profile_mirror=None, envelope_step_mib: int = AGENT_ENVELOPE_STEP_MIB,
                 solver_seconds: float = 10.0, reprofile: bool = False,
                 budget: int | None = None):
        self.fleet = fleet
        self.key = key
        self.hardware = hardware
        self.profiles = profiles
        self.make_agent = make_agent
        self.store_root = store_root
        self.profile_mirror = profile_mirror
        self.envelope_step_mib = envelope_step_mib
        self.solver_seconds = solver_seconds
        self.reprofile = reprofile
        # Measurements, not rounds: a degree the store already answers for costs nothing and does
        # not count against it. None means unbounded, which is only right when a caller has already
        # decided how much GPU time the answer is worth.
        self.budget = budget
        self.candidates = candidate_tp_degrees(hardware.gpu_count)
        # The agent of the most recent round, kept only so `run` can put the node back. Each round
        # builds its own -- a round is a different one-tenant fleet under a different key -- and the
        # next round replaces what is standing: the same tenant at a wider degree is an upgrade of
        # the same release, and a different tenant is uninstalled by that round's `_clear_stale`.
        # Either way exactly one is left when the rounds stop, and nothing after them removes it. It
        # holds an MPS envelope on its GPUs with nothing left to measure, so whatever runs next on
        # this node is measured against a memory budget nobody stated.
        self._agent = None

    # --- planning, which costs nothing ---------------------------------------

    def floor_for(self, tenant) -> object:
        """This tenant's memory floor, as a `TenantBaseline`. Arithmetic, no node needed."""
        provisional = dataclasses.replace(tenant, tp=1)
        return memory_floor_tp(provisional.request, device_total_gib=self.hardware.device_total_gib,
                               envelope_step_mib=self.envelope_step_mib,
                               gpu_count=self.hardware.gpu_count)

    def weight_floor_for(self, tenant) -> object:
        """The narrowest degree this tenant's WEIGHTS alone permit, as a `WeightFloor`.

        The only bound in this class a degree may be *eliminated* on (impl §14.2). Weights are the
        one term of the envelope that cannot come back smaller than the arithmetic states: the KV
        pool's page size was 3.27x too large the one time it met a booted engine, the activation
        fallback 4.4x, and the reserve is a constant. So a degree the weights rule out is ruled out,
        and a degree only the computed terms rule out is a degree nobody has measured.
        """
        provisional = dataclasses.replace(tenant, tp=1)
        return weight_floor_tp(provisional.request.mem,
                               device_total_gib=self.hardware.device_total_gib,
                               candidates=self.candidates)

    def degrees_for(self, tenant) -> tuple:
        """Every degree this tenant could still be served at, narrowest first (impl §14.2).

        Bounded below by the WEIGHT floor, not the memory floor. Until 2026-09-18 this started at
        the memory floor and said, in as many words, that "a round there would fail to come up and
        would record a feasibility fact the arithmetic already had" -- which is true of the weight
        term and false of everything else in the envelope. A degree the memory floor excludes but
        the weights allow is a degree arithmetic *doubts*, on numbers that have each been measured
        wrong in the optimistic direction, and only a boot settles it.

        Narrowest-first is the order of the list, not the order of the walk: see `start_tp_for`.
        """
        wf = self.weight_floor_for(tenant)
        if wf.floor_tp is None:
            return ()
        return tuple(k for k in self.candidates if k >= wf.floor_tp)

    def start_tp_for(self, tenant) -> int | None:
        """Where in `degrees_for` the walk begins. An ordering choice, not an elimination.

        The memory floor, which is the narrowest degree the *full* computed envelope fits at. That
        makes it the best guess at where a boot will first succeed, and guessing is all it is used
        for here: the degrees below it stay in `degrees_for` and stay reachable (impl §14.2).

        With no memory floor at all -- the computed envelope fits nowhere, yet the weights fit
        somewhere -- the widest degree the node offers is the best guess instead, and the walk has
        only downward to go from there.
        """
        degrees = self.degrees_for(tenant)
        if not degrees:
            return None
        floor = self.floor_for(tenant)
        if floor.floor_tp is None or floor.floor_tp not in degrees:
            return degrees[-1]
        return floor.floor_tp

    def ascent_for(self, tenant) -> tuple:
        """The degrees the upward walk measures: `start_tp_for` and everything wider.

        Upward is the *latency* question -- this degree met the target or it did not -- and latency
        is not monotone in TP (GLM-4.5-Air was measured worse at tp8 than tp4), so the walk stops at
        the first pass rather than continuing to look for a better one. Downward is the *memory*
        question, which is monotone, and belongs to the descent below the first pass.
        """
        start = self.start_tp_for(tenant)
        if start is None:
            return ()
        return tuple(k for k in self.degrees_for(tenant) if k >= start)

    def band_for(self, tenant) -> tuple:
        """The degrees the downward walk measures: everything the ascent will not reach.

        The weights allow them and the computed envelope refuses them, which is a doubt and not an
        elimination (impl §14.2), so only a boot settles them. Narrowest first, like `degrees_for`;
        the descent walks it backwards.
        """
        return tuple(k for k in self.degrees_for(tenant) if k not in self.ascent_for(tenant))

    def plan(self) -> list:
        """`(tenant, floor, ascent, already_profiled, band)` per tenant, for printing before spending.

        `band` is the descent's cost: the degrees below the memory floor that the weights allow, which
        `run` only reaches if the ascent finds a pass (impl §14.2). It is a *ceiling* on that cost and
        not a count -- the descent stops at the first degree that cannot hold the engine -- and it is
        listed separately because a reader deciding `--budget` needs to see that the walk can cost
        more rounds than the ascent alone.

        `already_profiled` spans BOTH walks, for the same reason: a band degree the store already
        holds a usable row for costs no round either, and a preview that counted it would overstate
        what a second `profile` of the same fleet spends.
        """
        out = []
        for t in self.fleet.tenants:
            have = self.profiles.for_model(t.model_id, t.isl_tokens, t.osl_tokens, t.rate_hz,
                                           t.serving, t.image_count, t.image_pixels)
            out.append((t, self.floor_for(t), self.ascent_for(t),
                        tuple(k for k in self.degrees_for(t) if k in have and have[k].usable),
                        self.band_for(t)))
        return out

    # --- one round, which does -----------------------------------------------

    def measure(self, tenant, tp: int, iteration: int, *,
                descend: bool = False, booted_at: int | None = None) -> Round:
        """Deploy this tenant alone at this degree, drive it, and file the row.

        The GPUs are the node's first `tp` ordinals rather than a choice. A solo tenant has no
        co-tenant for the arrangement to matter against, and the one thing that does vary --
        whether the shard straddles a NUMA boundary -- is recorded in the row rather than
        optimized here: on a g6e.48xlarge gpu0-3 and gpu4-7 are separate NUMA nodes, so taking
        them in order keeps a tp<=4 shard on one side, which is the best case a solo bound wants.

        `descend` makes this a **descent** round, below the memory floor and into the band the
        computed envelope refuses (impl §14.2). It changes one thing here, and `booted_at` changes a
        second.

        The KV floors are **relaxed to nothing** for the solve. Not a weaker experiment: a solo round
        is handed the whole card either way -- the solver's third objective spends the envelope up to
        what one GPU holds -- so the floors are the only thing that can refuse a degree here, and
        refusing is precisely what must not happen below the memory floor. `min_kv_tokens` and the
        SLO-derived floor are both computed from the demand, and a computed number is not entitled to
        close a degree the weights allow. What the round then measures is whether the tenant meets its
        own target with the pool the card actually leaves it, which is the question the floors were
        standing in for. Both figures go into the row.

        And with `booted_at`, a boot that fails for lack of room becomes a **witness**: `booted_at` is
        a degree that came up minutes earlier IN THIS RUN, with the same image, chart, model path and
        flags, so a crash here is attributable to the shard count rather than to the fleet. That
        control is what makes the row safe to write (`SoloProfile.too_narrow_reason`), and without it
        the same failure is recorded as nothing at all -- which is why it is a separate argument from
        `descend` rather than implied by it. A descent that follows a pass read out of the STORE has
        no such control: that row came up under some earlier run's chart, and `--reprofile` exists
        precisely because a chart can change under an unchanged key.
        """
        fleet = solo_fleet(self.fleet, tenant, tp)
        key = solo_key(self.key, tenant, tp)
        gpus = tuple(self.fleet.node.gpu_ids[:tp])
        requests = fleet.requests
        stated_floor = requests[0].floor_tokens
        stated_detail = requests[0].floor_detail
        if descend:
            requests = tuple(dataclasses.replace(
                r, min_kv_tokens=0, kv_floor_tokens=0,
                floor_source="relaxed for a descent round below the memory floor (impl §14.2)")
                for r in requests)
        solver = PlacementSolver(fleet.node.node_spec, requests,
                                 envelope_step_mib=self.envelope_step_mib)
        try:
            # Pinned rather than solved: which GPUs is not the question, and the solver's third
            # objective then spends the envelope down to what one GPU holds -- which is exactly the
            # full-envelope condition that makes this a bound.
            candidate = solver.solve_pinned({tenant.name: gpus}, max_seconds=self.solver_seconds)
        except Infeasible as exc:
            # The arithmetic refuses this degree before anything is deployed, so no GPU time was
            # spent and no boot happened. Whether that refusal is a fact about the fleet or only
            # about the arithmetic is `Unproven`'s distinction (impl §14.2) and it is carried in the
            # detail; the walk's response is the same either way, because memory feasibility IS
            # monotone in TP: if a shard does not fit here, a narrower one does not either, and the
            # only direction left is wider. What the walk must not do is stop -- that would leave a
            # tenant unresolved on a number nobody measured.
            log.warning("%s at tp%d cannot be placed at all: %s", tenant.name, tp, exc)
            return Round(tenant.name, tp, NOT_PLACEABLE, detail=str(exc), descent=descend)

        store = ObservationStore(self.store_root, key)
        store.write_key()
        agent = self._agent = self.make_agent(fleet, store, key)
        envelope = candidate.placement.envelope_gib[tenant.name]
        offered = candidate.placement.kv_tokens[tenant.name]
        descent_note = None
        if descend:
            descent_note = {"relaxed_floor": True,
                            "booted_at_tp": None if booted_at is None else int(booted_at),
                            "stated_floor_tokens": int(stated_floor),
                            "stated_floor_detail": stated_detail,
                            "offered_tokens": int(offered),
                            "clears_stated_floor": bool(offered >= stated_floor)}
            log.info("descending: %s at tp%d gets the whole card (%.2f GiB/shard, %d KV tokens) "
                     "against a stated floor of %d (%s). The floor is computed and this round is "
                     "what checks it: if the target is met with this pool, the floor was larger than "
                     "the tenant needed.", tenant.name, tp, envelope, int(offered),
                     int(stated_floor), stated_detail)
        log.info("profiling %s alone at tp%d on gpus %s, envelope %.2f GiB/shard, key %s",
                 tenant.name, tp, list(gpus), envelope, key.hash)
        attempt = agent.execute(Proposal(
            candidate=candidate, label=f"solo-tp{tp}",
            reason=(f"solo profile of {tenant.name} at tp{tp}: alone on gpus {list(gpus)} with the "
                    "whole of each to itself, which is the best case for this model at this rate "
                    "and shape and therefore a bound on every co-resident placement"),
            provenance={"kind": "solo-profile", "tenant": tenant.name, "tp": int(tp),
                        "profile_key_hash": self.profiles.key_hash}), iteration)
        if attempt.record is None or not attempt.record.get("per_tenant"):
            # The readiness gate's classification travels with the round, because the walk's next
            # move depends on it (impl §14.5): `engine_failed` means the envelope was applied and
            # the engine could not live in it, which is evidence about this DEGREE, and everything
            # else means the run broke without testing the degree at all. Read off the synthesized
            # record rather than parsed out of `outcome`, so the round and the stored observation
            # cannot disagree about what happened.
            notes = (attempt.record or {}).get("notes") or {}
            gate = str(notes.get("readiness_gate") or "")
            witness = None
            if descend and booted_at is not None and gate == NOT_READY_ENGINE:
                # The one boot failure that gets a row. It is a witness because a wider degree came
                # up in this same run (`booted_at`), so the image, the chart, the model path and
                # the flags are all known good -- see this method's docstring and
                # `SoloProfile.too_narrow_reason`. Writing it is what lets the answer above stop being
                # an upper bound: without a row the degree reads as unmeasured forever, and a hedge
                # that can never clear is one nobody reads. The row is superseded by any later
                # measurement at the same identity, so `--reprofile` still retries it.
                witness = self._file(profile_from_record(
                    attempt.record or {}, tenant, tp, envelope_gib=envelope,
                    device_total_gib=self.fleet.node.device_total_gib, run_id=attempt.run_id,
                    run_token=getattr(getattr(agent, "lease", None), "token", "") or "",
                    key_hash=key.hash, gpu_ids=gpus,
                    util=candidate.placement.util.get(tenant.name),
                    descent=descent_note,
                    boot_failed={"gate": gate, "detail": str(notes.get("detail") or ""),
                                 "booted_at_tp": int(booted_at)}), tenant, tp)
            return Round(tenant.name, tp, NOT_READY, profile=witness, run_id=attempt.run_id,
                         detail=attempt.outcome, gate=gate, descent=descend)
        profile = self._file(profile_from_record(
            attempt.record, tenant, tp, envelope_gib=envelope,
            device_total_gib=self.fleet.node.device_total_gib, run_id=attempt.run_id,
            run_token=getattr(getattr(agent, "lease", None), "token", "") or "",
            key_hash=key.hash, gpu_ids=gpus,
            util=candidate.placement.util.get(tenant.name),
            descent=descent_note), tenant, tp)
        return Round(tenant.name, tp, DID_MEASURE, profile=profile, run_id=attempt.run_id,
                     detail=attempt.outcome, descent=descend)

    def _file(self, profile: SoloProfile, tenant, tp: int) -> SoloProfile:
        """Write a row to the local store and mirror it. Returns the row, for chaining."""
        self.profiles.write(profile)
        if self.profile_mirror is not None:
            try:
                self.profile_mirror.push_profile(profile.id, profile.to_dict())
            except Exception:                                   # noqa: BLE001
                # Loud but never fatal. The observation behind this row is already on EFS, written
                # by the harness Job itself, so a failed push costs the index and not the
                # measurement -- and the round cost real GPU time.
                log.exception("could not push the profile row for %s at tp%d; it is in the local "
                              "store only", tenant.name, tp)
        return profile

    # --- the two walks --------------------------------------------------------

    def _ascend(self, tenant, rounds: list, spent: int, budget: int | None) -> tuple:
        """Walk degrees upward from the memory floor, stopping at the first pass.

        Returns `(spent, passed_at, booted_at)`. `passed_at` is the degree whose row meets the target,
        and None means the walk ran out of degrees, budget, or evidence. `booted_at` is the narrowest
        degree that actually CAME UP in this run, or None if nothing was deployed -- the descent needs
        it as the control for a failed boot, and a pass read out of the store is not one (`measure`).
        Narrowest, because this walk goes upward: the first round that comes up is the closest control
        to the band below.

        The upward walk starts at the memory floor rather than at the narrowest degree the weights
        allow: the floor is the best guess at where a boot first comes up, and the degrees below it
        are `_descend`'s business (impl §14.2), not this walk's.

        A degree can be visited twice. A boot failure is the one outcome that is genuinely ambiguous
        between the shard count and the run, so the walk pays for one repeat of the same degree
        before it will read a failed boot as evidence at all -- `_retry_once` below is where that is
        argued.
        """
        booted_at = None
        # Not `for tp in ...`: a boot failure re-queues its own degree once, so the sequence of
        # degrees visited is decided inside the loop rather than by the iterator.
        pending = list(self.ascent_for(tenant))
        retried: set = set()
        while pending:
            tp = pending.pop(0)
            have = self.profiles.read(tenant.model_id, tp, tenant.isl_tokens,
                                      tenant.osl_tokens, tenant.rate_hz, tenant.serving,
                                      tenant.image_count, tenant.image_pixels)
            if have is not None and have.usable and not self.reprofile:
                rounds.append(Round(tenant.name, tp, ALREADY_PROFILED, profile=have,
                                    run_id=have.run_id))
                if have.meets(tenant.slo):
                    return spent, tp, booted_at
                continue
            if budget is not None and spent >= budget:
                log.warning("budget of %d measurement(s) spent; %s at tp%d and anything wider "
                            "is still unmeasured", budget, tenant.name, tp)
                return spent, None, booted_at
            r = self.measure(tenant, tp, len(rounds))
            if r.status != NOT_PLACEABLE:
                # A round the arithmetic refused deployed nothing and held no GPU, so charging
                # it would spend the budget on a subtraction. Every other round reached a node.
                spent += 1
            rounds.append(r)
            if r.profile is not None and booted_at is None:
                # It came up. Not "it passed": the descent needs a degree whose ENGINE started under
                # this run's image and chart, which a missed target is just as much evidence of.
                booted_at = tp
            if r.profile is not None and r.profile.meets(tenant.slo):
                return spent, tp, booted_at
            # Ascending is earned four ways and no others, and a round that earns it none of them
            # stops the walk. They are not variants of one test, and two of the four do not
            # produce a profile row at all:
            #
            #   - the arithmetic would not place this degree (`NOT_PLACEABLE`). Memory
            #     feasibility IS monotone in TP -- a wider shard is a smaller shard -- so the
            #     narrower degree being over the card says nothing against the wider one, and
            #     no GPU time was spent finding out.
            #   - the engine was deployed and could not live in the envelope
            #     (`engine_failed`), AND it failed that way twice at the same degree. The
            #     verification init container passed, so the MPS limit was applied as asked, and
            #     the tenant had the whole card to itself. That is `executed` evidence this degree
            #     is too NARROW, arrived at the only way the arithmetic's soft terms can be
            #     settled -- by a boot (impl §14.5). The repeat is what makes it evidence rather
            #     than a guess; `_retry_once` below is why.
            #   - a measured miss says the degree is too SLOW.
            #   - a too-narrow row says the degree is too SMALL on its own evidence, which is
            #     available exactly when the miss is not: a server driven flat out that
            #     completed nothing has no latency to be slow with.
            #
            # Everything else -- a verification failure, a timeout, an unschedulable pod, a row
            # that came back unusable for neither reason -- is a property of the RUN and not of
            # the shard count, and widening on it buys the baseline a degree nothing ruled the
            # narrower one out at. Those stop.
            if r.status == NOT_PLACEABLE:
                continue                    # widened by arithmetic; see above. No row written.
            if r.profile is None:
                if r.gate == NOT_READY_ENGINE:
                    if tp not in retried:
                        # `_retry_once`. A failed boot is the only round in this walk whose cause is
                        # ambiguous between the shard count and the run itself, and the two call for
                        # opposite answers: too narrow means widen, a transient fault means measure
                        # this degree again. One failure cannot tell them apart -- the engine's exit
                        # is not read (`judge from observations, not backend internals`), and it
                        # would not settle it anyway, since an out-of-memory abort and a failed
                        # weight load can both arrive as a nonzero exit.
                        #
                        # So the walk buys the distinguishing evidence instead of guessing at it,
                        # and one round is the whole price. Too narrow is a property of the
                        # arithmetic and repeats: the same degree, the same envelope, the same
                        # image, so the second boot fails the same way and the widening below
                        # happens one round later with the evidence actually in hand. A transient
                        # fault does not repeat, and the retry measures the degree the fleet
                        # wanted.
                        #
                        # Measured, 2026-09-21: `qwen35-multimodal-2`'s `docqa` failed to boot at
                        # tp1 once (`vllm exited 1`). Read as evidence, it widened the walk to tp2,
                        # tp4 and tp8 -- three rounds, ~45 min of g6e -- all of which came up and
                        # then stalled. Retried at tp1 it measured 56.253 ms against a 60 ms
                        # target. One round would have bought what three rounds could not, and a
                        # widened degree that nothing ruled the narrower one out at is the §0 error
                        # `tp:` was deleted for: the baseline goes up by a shard on no evidence.
                        retried.add(tp)
                        pending.insert(0, tp)
                        log.warning("%s at tp%d was deployed alone at the full envelope and the "
                                    "engine could not come up in it (%s). NOT widening on that: "
                                    "one failed boot does not separate a shard too narrow from a "
                                    "fault in this run, and those want opposite answers. Measuring "
                                    "tp%d once more -- if it fails the same way the walk widens "
                                    "with evidence, and if it comes up the degree was never the "
                                    "problem.", tenant.name, tp, r.detail or "no detail", tp)
                        continue
                    log.warning("%s at tp%d was deployed alone at the full envelope and the "
                                "engine could not come up in it twice (%s), so the walk widens: "
                                "a failure that repeats at the same degree, envelope and image is "
                                "measured evidence tp%d is too narrow, which is the only kind the "
                                "computed terms of the envelope can be settled by. No profile row "
                                "is written, so this degree stays UNMEASURED for latency and a "
                                "plain re-run retries it.",
                                tenant.name, tp, r.detail or "no detail", tp)
                    continue
                log.error("%s at tp%d did not come up, and not for lack of room: the readiness "
                          "gate says %r (%s). NOT measuring a wider degree -- that is a "
                          "property of this run and not of the shard count, so the wider "
                          "degree would most likely break the same way, having spent the "
                          "budget to find out. Fix the run and re-run: this same degree has no "
                          "row, so it is measured again without --reprofile.",
                          tenant.name, tp, r.gate or "unclassified", r.detail)
                return spent, None, booted_at
            if r.profile.usable:
                continue                    # a measured miss: `meets` was False above. Widen.
            narrow = r.profile.too_narrow_reason
            if narrow:
                code, why = r.profile.unusable_reason
                log.warning("%s at tp%d has no latency to judge (%s: %s), and does not need "
                            "one: %s. That rules tp%d out as too narrow on its own evidence, so "
                            "the walk widens. The row stays unusable for latency -- it sets no "
                            "minimum and is never quoted as one.",
                            tenant.name, tp, code, why, narrow, tp)
                continue
            # Neither. A row that came back unusable did not tell us this degree is too slow --
            # it told us this degree was not measured -- and going wider on that is the mirror
            # image of the §0 error `tp:` was deleted for: it spends a GPU round to hand the
            # baseline a degree nothing ruled the narrower one out at. Whatever broke here (the
            # harness declining to vouch, a suppressed latency, a failed delivery gate) is a
            # property of the run and not of the shard count, so the wider degree would most
            # likely break the same way, having spent the budget to find out. Stop, and let the
            # resolution below report the tenant unresolved.
            code, why = r.profile.unusable_reason
            log.error("%s at tp%d produced a row that cannot accept or reject a degree "
                      "(%s: %s), and does not show a server that was working and behind "
                      "either. NOT measuring a wider degree: nothing here says tp%d is too "
                      "slow or too small, only that it was not measured, and a wider degree "
                      "bought on that reasoning would inflate the baseline. Fix the run and "
                      "re-run: the walk up re-measures a degree whose row is unusable, so "
                      "this needs no --reprofile.",
                      tenant.name, tp, code, why, tp)
            return spent, None, booted_at
        return spent, None, booted_at

    def _descend(self, tenant, rounds: list, spent: int, budget: int | None, *,
                 below: int, booted_at: int | None) -> int:
        """Walk downward from a degree that passed, into the band the arithmetic only doubts.

        `_ascend` starts at the MEMORY floor, so every degree below it -- `band_for` -- is a degree
        the envelope's computed terms (activation, reserve, KV page size, floor tokens, headroom)
        refused, and every one of those terms has come back smaller than stated at least once.
        Refusing on them is a guess; a boot is not. So once a degree has passed, this walks the band
        down rung by rung and lets a boot settle each one (impl §14.2). Only the band: any degree
        between the memory floor and `below` was already measured by the ascent, which walked up
        through it.

        Downward and not upward because the two questions have opposite monotonicity. Memory
        feasibility IS monotone in TP -- a narrower shard is a bigger shard, so the first degree that
        cannot hold the engine rules out every degree below it and the walk stops. Latency is NOT:
        a narrower degree that misses says nothing about a narrower one still, so a miss keeps
        walking. That is why the descent is a separate walk and not the ascent run backwards.

        `below` is the degree that passed, and it is the first rung's control: `measure` names it in
        the witness row a failed boot writes, and what makes that row evidence about the DEGREE is
        that this same tenant, image, chart and model path came up there in this same run. Each rung
        that comes up becomes the control for the next one down, so the control is never further than
        one rung away from the degree it vouches for.

        Returns `spent`. The narrowest passing row is picked by `minimal_tp` from the store, so this
        walk has no answer of its own to return: every degree it settles, it settles by writing a
        row (or by writing nothing, which leaves the degree unmeasured and the baseline an upper
        bound -- which is what it already was).
        """
        for tp in reversed([k for k in self.band_for(tenant) if k < below]):
            have = self.profiles.read(tenant.model_id, tp, tenant.isl_tokens, tenant.osl_tokens,
                                      tenant.rate_hz, tenant.serving, tenant.image_count,
                                      tenant.image_pixels)
            if have is not None and not self.reprofile:
                # Unlike the ascent, an UNUSABLE row counts here if it carries a too-narrow reason:
                # that is a degree already ruled out, and memory is monotone, so nothing narrower
                # can come up either.
                # A stored row does NOT become the control for the rung below it: it came up under
                # some earlier run's chart, and `--reprofile` exists because a chart can change under
                # an unchanged key. The control has to be a boot this run watched.
                rounds.append(Round(tenant.name, tp, ALREADY_PROFILED, profile=have,
                                    run_id=have.run_id, descent=True))
                if have.too_narrow_reason is not None:
                    return spent
                if have.usable:
                    continue
                # Unusable for some other reason: that row says this degree was not measured, and
                # descending past it would credit a narrower degree nothing checked this one at.
                code, why = have.unusable_reason
                log.error("%s at tp%d: the store's row cannot accept or reject this degree "
                          "(%s: %s), so the descent stops here rather than crediting a narrower "
                          "one. --reprofile this degree once whatever broke is fixed.",
                          tenant.name, tp, code, why)
                return spent
            if budget is not None and spent >= budget:
                log.warning("budget of %d measurement(s) spent, so the descent below tp%d stops "
                            "with tp%d unmeasured: whatever minimum is reported for %s is an UPPER "
                            "BOUND on it and not the minimum. Raise --budget to settle it.",
                            budget, below, tp, tenant.name)
                return spent
            r = self.measure(tenant, tp, len(rounds), descend=True, booted_at=booted_at)
            if r.status != NOT_PLACEABLE:
                spent += 1
            rounds.append(r)
            if r.status == NOT_PLACEABLE:
                # Refused with the KV floors already relaxed to nothing, so what is over the card is
                # the shard itself. Monotone: narrower is worse. Done.
                log.info("%s at tp%d does not fit even with its pool relaxed to nothing (%s), so "
                         "no narrower degree can, and the descent ends. What is over the card here "
                         "is the shard itself and not the pool.", tenant.name, tp, r.detail)
                return spent
            if r.status == NOT_READY:
                # Either a witness -- the engine could not start in the whole card at this degree,
                # while `booted_at` came up in the same run with the same image and chart, which is
                # `executed` evidence about the DEGREE and gets a disqualifying row -- or a
                # stop-class gate, which is about the run and gets nothing. `measure` decides which
                # and writes accordingly; both end the descent, because memory is monotone and a
                # broken run will not get better one degree narrower.
                return spent
            booted_at = tp                  # it came up. The control for the next rung down.
            if r.profile is None:
                return spent                # unreachable in practice; never descend on no evidence
            if r.profile.usable:
                # Meets or misses, keep walking: latency is not monotone in TP, so a narrower degree
                # is not ruled out by this one missing, and `minimal_tp` picks the narrowest row that
                # both came up and met the target.
                continue
            if r.profile.too_narrow_reason is not None:
                code, why = r.profile.unusable_reason
                log.warning("%s at tp%d came up and has no latency to judge (%s: %s), and does not "
                            "need one: %s. Narrower is smaller still, so the descent ends here.",
                            tenant.name, tp, code, why, r.profile.too_narrow_reason)
                return spent
            code, why = r.profile.unusable_reason
            log.error("%s at tp%d produced a row that cannot accept or reject this degree (%s: %s). "
                      "The descent stops rather than crediting a narrower degree nothing checked "
                      "this one at; --reprofile it once the run is fixed.",
                      tenant.name, tp, code, why)
            return spent
        return spent

    # --- the whole thing ------------------------------------------------------

    def run(self, *, budget: int | None = None, teardown_at_end: bool = True) -> dict:
        """Profile every tenant: ascend to a pass, then descend into the band below it.

        Returns `{"rounds": [...], "resolutions": {tenant: TpResolution}, "spent": n}`. Failing to
        resolve a tenant is a result, not an exception: the other tenants' rounds are still worth
        having, and the resolution says exactly which degrees are still unmeasured.

        `teardown_at_end` defaults to True, unlike `search`'s, and the asymmetry is deliberate. A
        search ends on a placement somebody may want to look at, keep serving or measure again; a
        solo round ends on a one-tenant fleet whose only output is a row that is already written. So
        leaving it up has no diagnostic value and one real cost: it holds its envelope on a GPU that
        the next round, the next search, or the node holder is about to be measured on. A round that
        failed to come up is diagnosed from its record, which says which gate failed.
        """
        budget = self.budget if budget is None else budget
        rounds, spent = [], 0
        for tenant in self.fleet.tenants:
            degrees = self.degrees_for(tenant)
            if not degrees:
                # Its WEIGHTS do not fit one GPU at any legal degree (impl §14.2), which is the one
                # arithmetic entitled to say "there is nothing to profile": no envelope, no page
                # size and no measurement puts a shard under the card once the weights are over it.
                log.warning("%s fits at no TP degree on a %s, so there is nothing to profile: %s",
                            tenant.name, self.hardware.instance_type,
                            self.weight_floor_for(tenant).why)
                continue
            if len(degrees) == 1:
                # Its WEIGHT floor is the widest degree the node offers, so every legal degree is
                # this one. Measuring it would cost a round and could not change the answer. Keyed
                # to the weight floor and not the memory floor since 2026-09-18: a memory floor at
                # the widest degree leaves narrower degrees that computed terms doubt and nothing
                # has measured, and calling those "forced" spent no round while closing the question.
                rounds.append(Round(tenant.name, degrees[0], FORCED_BY_MEMORY,
                                    detail=self.weight_floor_for(tenant).why))
                log.info("%s: tp%d is forced by memory alone, so no round is spent on it",
                         tenant.name, degrees[0])
                continue
            if tenant.slo is None or tenant.slo.itl_mean_ms is None:
                log.warning("%s states no itl_mean_ms target, so no degree can be judged against "
                            "one and nothing here will pick a minimum for it. Profiling it anyway "
                            "would measure without a criterion.", tenant.name)
                continue
            # Two walks, and the second one only ever runs because the first one succeeded. The
            # ascent starts at the memory floor and finds a degree that comes up and meets the
            # target; the descent then tests the band below it, which the envelope's computed terms
            # refuse and nothing has measured (impl §14.2). Skipped when there is no band -- the
            # memory floor is already the narrowest degree the weights allow.
            spent, passed_at, booted_at = self._ascend(tenant, rounds, spent, budget)
            if passed_at is not None and self.band_for(tenant):
                spent = self._descend(tenant, rounds, spent, budget,
                                      below=passed_at, booted_at=booted_at)
        if teardown_at_end and self._agent is not None:
            # Never fatal. Every row is written and pushed by this point, so a failed uninstall
            # costs a held GPU and not a measurement -- and saying which release is still up is
            # what lets somebody finish the job by hand.
            try:
                self._agent.teardown()
            except Exception:                                   # noqa: BLE001
                log.exception("could not tear down the last profile round's tenant. It still holds "
                              "its envelope on this node: uninstall it before measuring here again")

        resolutions = {}
        for tenant in self.fleet.tenants:
            floor = self.floor_for(tenant)
            degrees = self.degrees_for(tenant)
            if len(degrees) == 1:
                # The weights leave exactly one legal degree, so no measurement could choose
                # differently and `certain` here is honest arithmetic (impl §14.2). It is a claim
                # about the DEGREE and not about whether the tenant comes up at it: if the computed
                # envelope doubts this degree too, that doubt is the boot's to settle, not this
                # resolution's -- and there is no narrower degree for it to settle in favour of.
                resolutions[tenant.name] = TpResolution(
                    tenant=tenant.name, model_id=tenant.model_id, status=RESOLVED,
                    tp=degrees[0], floor_tp=floor.floor_tp, certain=True)
                continue
            # `floor_tp` here is the floor a degree may be ELIMINATED at, which is the weight floor
            # (impl §14.2): `minimal_tp` marks everything below it "does not fit" -- a checked
            # ground, which keeps the answer a minimum -- and everything above it that nobody
            # measured "not profiled", which makes the answer an upper bound and says which degrees
            # would settle it. Passing the memory floor instead reported the doubted band below it as
            # ruled out, so an upper bound read as a minimum.
            resolutions[tenant.name] = minimal_tp(self.profiles, tenant,
                                                  floor_tp=degrees[0],
                                                  candidates=self.candidates)
        return {"rounds": rounds, "resolutions": resolutions, "spent": spent}
