"""The agent loop (impl §11).

    load the store for this validity key
    repeat:
      feasible    = solver.solve(cap)                   <- constraints only; nothing predicted
      choice      = policy.next(...)                    <- the one genuinely agentic decision
      values      = render(choice)
      deploy, wait for the readiness gate
      observation = harness.measure()
      store.append(observation)
    until the policy stops

Everything except `policy.next` is mechanical, which is why the policy is a parameter and the
loop is not. Calibration and search are the same code path with different policies -- so the
first calibration run is also the first end-to-end test of the deploy path, the readiness gate
and the store write, rather than those being trusted for the first time during a search whose
results depend on them.

Three decisions in here are worth stating, because each one is the opposite of the convenient
choice:

**A placement that fails its readiness gate becomes an observation.** impl §8 requires
infeasible observations in the store: they are what stops the agent re-proposing something that
already failed. The record is synthesized here, with `verdict: infeasible_deploy` and the gate's
own classification in `notes`, because the harness never ran and so cannot write one. It is
marked `synthesized_by: agent` so it is never mistaken for a measurement.

**The store is re-read every iteration, and nothing is carried between them.** There used to be
an interference model here, rebuilt from the whole store on each pass so that the agent reasoned
with exactly the model a fresh process would have built. The model is gone (impl §9 is
prediction-free), and what is left is simpler and has the same property for the same reason: the
records are read fresh, the solver holds no history at all, and a resumed search sees what an
uninterrupted one would.

**The fleet is torn down only when asked.** Leaving the last placement up is the useful default
for an interactive session, and tearing it down is what an unattended sweep wants. Neither is
safe to guess, so it is an argument with no default that flatters one use.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..solver.cpsat import Infeasible, PlacementSolver
from ..store.schema import VERDICT_DEPLOY, VERDICT_MEASURED
from ..store.store import ObservationStore
from .deploy import FleetDeployer
from .measure import HarnessRunner
from .policy import Attempt, Context, Policy
from .render import check_agreement, render_releases, render_run_config
from .spec import AGENT_ENVELOPE_STEP_MIB, FleetSpec

log = logging.getLogger(__name__)

OUTCOME_MEASURED = "measured"
OUTCOME_NOT_READY = "not_ready"
OUTCOME_SOLVER_INFEASIBLE = "solver_infeasible"


@dataclass
class SearchResult:
    attempts: list
    stopped_because: str

    @property
    def measured(self) -> list:
        return [a for a in self.attempts if a.outcome == OUTCOME_MEASURED]


def closest_to_targets(attempts: list, targets: dict):
    """The attempt that came nearest its targets, whether or not it met them.

    **This is not a certification.** It answers "what is the best this fleet did on this node",
    which a search has to report even when the answer is that nothing passed -- otherwise a failed
    search records only that it failed, and the next one re-measures the same ground to rediscover
    which direction was promising.

    Ranked by the *worst tenant's margin ratio* (measured ITL over its target), not by raw ms. Raw
    ms across tenants with different targets is not a ranking of anything: on the 14B/30B fleet it
    put a 41.954 ms placement ahead of a 53.410 ms one, when by ratio the second was the closer of
    the two (1.335x against 1.398x) and used four GPUs rather than seven. Ties on ratio break toward
    fewer GPUs, because two placements equally far from the targets are not equally good.

    Filtered on the *record's* verdict, and that is the substantive part. Its predecessor -- a
    `SearchResult.best()` that was dead code, so the mistake never surfaced -- filtered on the
    loop's `outcome == "measured"`, which means only that the harness ran and returned. A record
    whose verdict is `infeasible_no_steady_state` or a delivery failure carries suppressed or
    meaningless latency, and `best()` would have returned one as the best result of a search: on the
    14B/30B run it picks the placement whose reason tenant took 28.7 s to first token.

    A free function rather than a method, because the two result types a search produces
    (`SearchResult` and `packer.PackerResult`) disagree about what `measured` means and neither
    disagreement is the right filter here.

    Returns `(attempt, worst_ratio)`, or None when nothing measured cleanly, and also when nothing
    that did has a target to be near -- an untargeted fleet has no "closest", only results.
    """
    ranked = []
    for a in attempts:
        if not a.record or a.record.get("verdict") != VERDICT_MEASURED:
            continue
        per = a.record.get("per_tenant") or {}
        ratios: list | None = []
        for name, target in targets.items():
            want = getattr(target, "itl_mean_ms", None) if target else None
            if want is None:
                continue
            got = (per.get(name) or {}).get("itl_mean_ms")
            if got is None:
                # Suppressed latency is not a fast tenant (see `_worst_itl_ms`). A record that
                # cannot answer for a targeted tenant cannot be the closest to the targets.
                ratios = None
                break
            ratios.append(float(got) / float(want))
        if not ratios:
            continue
        ranked.append((max(ratios), a.candidate.gpus_used, a))
    if not ranked:
        return None
    worst_ratio, _, attempt = min(ranked, key=lambda r: (r[0], r[1]))
    return attempt, worst_ratio


class SearchAgent:
    def __init__(self, fleet: FleetSpec, store: ObservationStore, deployer: FleetDeployer,
                 harness: HarnessRunner, *, lease=None, run_prefix: str = "search",
                 top_k: int = 4, solver_seconds: float = 10.0,
                 ready_timeout_s: float = 1800.0, measure_timeout_s: float = 5400.0,
                 envelope_step_mib: int = AGENT_ENVELOPE_STEP_MIB,
                 mirror=None,
                 render_kwargs: dict | None = None,
                 run_config_kwargs: dict | None = None):
        self.fleet = fleet
        self.store = store
        # Anything with `push_record(run_id, record)` -- normally an `EfsMirror`. Only
        # synthesized records go through it; measured ones are already on EFS, written there by
        # the harness Job itself.
        self.mirror = mirror
        self.deployer = deployer
        self.harness = harness
        # The `capture.NodeLease` this search holds, or None for a hand-run against whatever is
        # already installed. It is not another way to name the node -- `fleet.node` came out of
        # it -- it is the run's IDENTITY: the prefix every release it installs carries, and the
        # token that keeps two searches sharing one store from writing each other's run_ids.
        self.lease = lease
        self.tenant_prefix = lease.tenant_prefix if lease is not None else ""
        self.run_prefix = run_prefix
        self.top_k = top_k
        self.solver_seconds = solver_seconds
        self.ready_timeout_s = ready_timeout_s
        self.measure_timeout_s = measure_timeout_s
        self.envelope_step_mib = envelope_step_mib
        self.render_kwargs = dict(render_kwargs or {})
        self.run_config_kwargs = dict(run_config_kwargs or {})
        for label, kwargs in (("render_kwargs", self.render_kwargs),
                              ("run_config_kwargs", self.run_config_kwargs)):
            if "prefix" in kwargs:
                # Refused rather than honoured, because the release prefix and the deployer's
                # guard have to be the same string: `HelmDeployer` refuses anything not carrying
                # `lease.tenant_prefix`, so a prefix passed in here that disagreed would make
                # every apply fail -- or, worse, agree with the guard while disagreeing with the
                # lease, and install this run's tenants under another run's name.
                raise ValueError(
                    f"{label} must not carry 'prefix': the release prefix is the run's lease "
                    "token, so it comes from `lease=` and nowhere else.")

    # --- the loop ------------------------------------------------------------

    def run(self, policy: Policy, *, teardown_at_end: bool) -> SearchResult:
        history: list = []
        stopped = "policy stopped"
        iteration = 0
        while True:
            observations = self._observations()
            solver = PlacementSolver(self.fleet.node.node_spec, self.fleet.requests,
                                     envelope_step_mib=self.envelope_step_mib)
            ctx = Context(fleet=self.fleet, history=history,
                          calibration=self.store.read_calibration(), iteration=iteration,
                          observations=observations,
                          solve=lambda k, avoid=(): solver.solve(
                              cap=k, max_seconds=self.solver_seconds, avoid=avoid),
                          solve_pinned=lambda a: solver.solve_pinned(
                              a, max_seconds=self.solver_seconds))
            try:
                proposal = policy.next(ctx)
            except Infeasible as exc:
                # The policy asked for something the node cannot host. For a designed sweep
                # that is a bug in the sweep, not a result, so it stops the run rather than
                # being filed as a feasibility fact about a placement that was never tried.
                log.error("iteration %d: the policy's request is infeasible: %s",
                          iteration, exc)
                stopped = f"infeasible request: {exc}"
                break
            if proposal is None:
                break

            log.info("iteration %d: trying %s -- %s", iteration, proposal.label,
                     proposal.reason)
            attempt = self.execute(proposal, iteration)
            history.append(attempt)
            iteration += 1

        if teardown_at_end:
            self.teardown()
        log.info("search finished after %d iterations: %s", len(history), stopped)
        return SearchResult(history, stopped)

    def execute(self, proposal, iteration: int) -> Attempt:
        """One placement, from render to record. Public because it is the `measure` tool's body.

        Named rather than private on purpose: `tools.measure` is exactly this, and there must not
        be a second implementation of deploy-gate-drive-record for the reasoner-driven path. That
        it decides nothing is what makes it safe to expose -- the choice arrived as a `Proposal`.
        """
        candidate = proposal.candidate
        placement = candidate.placement
        run_id = self._run_id(iteration, proposal.label)

        sm_caps = dict(getattr(proposal, "sm_caps", None) or {})
        releases = render_releases(placement, self.fleet, prefix=self.tenant_prefix,
                                   sm_caps=sm_caps, **self.render_kwargs)
        run_config = render_run_config(
            placement, self.fleet, run_id, self.store.key_hash,
            prefix=self.tenant_prefix, sm_caps=sm_caps,
            itl_target_halfwidth_ms=self._calibrated_halfwidth(candidate),
            provenance=self._provenance(proposal, candidate),
            **self.run_config_kwargs)
        # The device total travels from the fleet's own inventory, which got it from the hardware
        # store: the same number the solver priced this placement against, re-applied to what is
        # about to be deployed.
        check_agreement(releases, run_config, self.tenant_prefix,
                        device_total_gib=self.fleet.node.device_total_gib,
                        mps_reserve_gib=placement.mps_reserve_gib)

        self._clear_stale(releases)
        self.deployer.apply(releases)
        ready = self.deployer.wait_ready(releases, self.ready_timeout_s)
        if not ready.ok:
            record = self._deploy_failure_record(run_id, run_config, ready)
            self.store.append_observation(record)
            if self.mirror is not None:
                # No harness Job ran, so this record exists nowhere but the local mirror. It is
                # the record that stops the *next* agent re-proposing this placement, so it has
                # to reach the shared store.
                try:
                    self.mirror.push_record(run_id, record)
                except Exception:       # noqa: BLE001 - never lose a search to an upload
                    log.exception("could not push the synthesized record for %s; it is in the "
                                  "local store only", run_id)
            return Attempt(run_id, proposal.label, candidate, record,
                           f"{OUTCOME_NOT_READY}:{ready.state}", sm_caps)

        record = self.harness.measure(run_config, run_id, self.measure_timeout_s)
        self.store.append_observation(record)
        return Attempt(run_id, proposal.label, candidate, record, OUTCOME_MEASURED, sm_caps)

    # --- pieces --------------------------------------------------------------

    def _observations(self) -> list:
        """Every record under this key that may be cited as evidence about latency.

        The records themselves, not a summary of them. That is the whole of what impl §10 leaves
        for anything downstream to reason from -- there is no table, no fit and no index -- so
        this hands them over as they were written. `Context.observations` carries them; nothing
        is obliged to look.
        """
        report = self.store.load()
        if report.unreadable:
            # Not tolerated silently. A store that returns fewer records than it holds makes
            # the agent re-propose placements that already failed, and the symptom looks like
            # a bad exploration policy rather than a broken read (impl §8).
            raise RuntimeError(
                f"{len(report.unreadable)} records under {self.store.key_hash} could not be "
                f"read: {report.unreadable}. Fix or move them; searching against a partial "
                "history silently changes what the agent believes.")
        return [o for o in report.observations if o.usable_as_evidence]

    def _calibrated_halfwidth(self, candidate) -> float | None:
        """The resolution the harness should size its window for, for *this* placement.

        Looked up by the candidate's own co-residency bucket, which is a fact about the
        assignment rather than the forecast the bucket used to be keyed by. When the store has no
        half-width for that bucket -- an arrangement nobody has repeated yet -- it falls back to
        the tightest one measured, because that demands the longest window, and a window that was
        too short is the one error a run cannot be repaired from afterwards.
        """
        cal = self.store.read_calibration()
        if cal is None:
            return None
        degrees = candidate.coresidency.values()
        bucket = f"coresident={max(degrees)}" if degrees else None
        return (cal.threshold_ms(bucket) if bucket else None) or cal.tightest_ms

    def _run_id(self, iteration: int, label: str) -> str:
        # Doubles as the Job name and the record filename, so it must be a DNS-1123 label and
        # unique. The store appends a suffix on collision, but the Job would not -- so a
        # collision is resolved here rather than discovered as a failed apply.
        #
        # The lease token is in it because the collision check below is a `.exists()` on a shared
        # EFS directory, and two searches under the same validity key both start at iteration 0
        # with the same policy labels. Each would see no file, both would pick the same run_id,
        # and the loser's record would be overwritten by the winner's -- silently, since the
        # verdict and the shape are the same and only the numbers differ.
        head = f"{self.run_prefix}-{self.lease.token}" if self.lease is not None \
            else self.run_prefix
        base = f"{head}-{iteration:03d}-{label}".lower()[:52]
        candidate = base
        n = 1
        while (self.store.observations_dir / f"{candidate}.json").exists():
            candidate = f"{base}-r{n}"
            n += 1
        return candidate

    def _provenance(self, proposal, candidate) -> dict:
        """What was required of this placement when it was chosen. Travels into the record.

        **Nothing here is a prediction.** The keys this used to carry -- `predicted_itl_ms`,
        `predicted_worst_occupancy`, `predicted_regime`, `itl_sources`, `all_measured`,
        `surrogate_occupancy` -- were all the interference model's, and they are gone with it.
        What replaces them is the *requirement* the placement was solved against: each tenant's
        KV floor, where that floor came from, and the margin the placement gave it. That is what
        makes a record checkable later. A prediction next to a measurement tells you whether the
        model was wrong; a floor next to a measurement tells you whether the fleet was certified,
        which is the question the design actually asks.

        Records written before this change still carry the prediction keys. Nothing rewrites them:
        they are what that run believed, and editing them would be forging a history.

        **`gpu_indices` and `gpu_ordinal_by_uuid` exist because a UUID is not portable.** The
        harness can only see the GPUs it was given, so a record's per-tenant config and its
        per-GPU gauges are keyed by UUID -- and a UUID belongs to one physical device on one
        machine. Once the node is released the mapping back to a host ordinal survives nowhere:
        it lived in the leased node's ConfigMap, which went with it. That makes anything ordinal
        -shaped unrecoverable from the store, and "ordinal-shaped" turns out to include real
        effects. Runs 000 and 003 of 2026-09-04 measured the same tenant, at the same rate, with
        the same envelope and no co-residency, 7 ms apart; the only difference was which four
        GPUs, and a reader of those two records cannot see it. Written here, while the lease is
        still standing and it costs nothing, the comparison stays available on any later node.

        Deliberately the *facts* and not the inference. Nothing here says which ordinals are
        near each other, because that is a property of a machine this agent does not model
        (cpsat.py's uniform-topology assumption) and a reasoner can find it in the records.
        """
        assignment = candidate.placement.assignment
        node = self.fleet.node
        return {
            # The policy's own provenance goes first so it cannot overwrite anything below it.
            # What the agent required of the placement is a fact about the run; what a policy says
            # about its own choice is a claim, and a claim must not be able to shadow a fact.
            **dict(proposal.provenance or {}),
            "chosen_by": "agent",
            # The record's own `node` field is written by the harness from its own environment,
            # so it names the *load driver's* node -- a cpu-compute node Karpenter reclaims,
            # which means the field points at a machine that no longer exists by the time anyone
            # reads it. The node under test is this one, and only the agent knows it.
            "gpu_node": self.fleet.node.node_name,
            # Which lease produced this record. A node name is not enough once nodes are captured
            # and released: the same hostname can be two different leases hours apart, and two
            # leases on two nodes write into one store under one key. The token is what says
            # "these records came from one continuous hold of one machine".
            **({"run_token": self.lease.token,
                "daemon_release": self.lease.release,
                "gpu_node_instance_type": self.lease.hardware.instance_type}
               if self.lease is not None else {}),
            "policy_label": proposal.label,
            "policy_reason": proposal.reason,
            "kv_floor_tokens": dict(candidate.floor_tokens),
            "kv_floor_source": dict(candidate.floor_detail),
            "kv_tokens": dict(candidate.placement.kv_tokens),
            "kv_margin": dict(candidate.margin),
            "tightest_margin": candidate.min_margin,
            "tightest_tenant": candidate.tightest_tenant,
            "gpus_used": candidate.gpus_used,
            # Host ordinals per tenant, in shard order, and the map that joins the UUIDs the
            # harness recorded back onto them. Both, not either: the first is what a placement
            # *is*, the second is what makes the per-GPU gauges in the same record readable
            # after the node is gone.
            "gpu_indices": {n: list(g) for n, g in assignment.items()},
            **({"gpu_ordinal_by_uuid": {u: g for g, u in zip(node.gpu_ids, node.uuids)}}
               if node.uuids else {}),
            "coresidency": dict(candidate.coresidency),
            "envelope_gib": dict(candidate.placement.envelope_gib),
            "mps_reserve_source": dict(candidate.placement.reserve_source),
        }

    def _clear_stale(self, releases) -> None:
        """Uninstall tenant releases this placement does not include.

        For a fixed fleet this is normally a no-op -- the release set does not change, only its
        values do. It matters when a fleet shrinks between runs: a leftover tenant still holds
        its MPS envelope on its GPU, so the next placement would be measured against a memory
        budget the solver did not know about.
        """
        wanted = {name for name, _ in releases}
        stale = [r for r in self.deployer.live_tenant_releases() if r not in wanted]
        if stale:
            log.info("uninstalling %s: not part of this placement", stale)
            self.deployer.destroy(stale)

    def teardown(self) -> None:
        """Uninstall every tenant release this agent's deployer can see.

        Public for the same reason `execute` is: `SoloProfiler` ends a run of many one-tenant
        placements and has to be able to put the node back, and a second implementation of
        "uninstall what we deployed" is the kind of duplicate that leaves a GPU held. It does not
        touch the lease -- the node is released deliberately or not at all.
        """
        live = self.deployer.live_tenant_releases()
        if live:
            log.info("tearing down %s", live)
            self.deployer.destroy(live)

    def _deploy_failure_record(self, run_id: str, run_config: dict, ready) -> dict:
        """An observation for a placement that never came up.

        Deliberately synthesized rather than skipped. impl §8: infeasible observations are what
        keep the agent from re-proposing a placement that already failed. `synthesized_by`
        marks it so it is never read as something the harness measured -- and it carries no
        `sampling` block, so `usable_as_evidence` is false for two independent reasons.
        """
        return {
            "run_id": run_id,
            "validity_key_hash": self.store.key_hash,
            "node": self.fleet.node.node_name,
            "verdict": VERDICT_DEPLOY,
            "steady_state_reached_s": None,
            "window_s": 0.0,
            "configuration": {"tenants": run_config["tenants"]},
            # Per tenant, matching what the harness writes, so a synthesized record and a
            # measured one can be read by the same code.
            "workload": {"tenants": {t.name: {"isl_tokens": t.isl_tokens,
                                              "osl_tokens": t.osl_tokens,
                                              "tokenizer_path": t.tokenizer_path}
                                     for t in self.fleet.tenants}},
            "per_tenant": {},
            "per_gpu": {},
            "provenance": dict(run_config.get("provenance") or {},
                               synthesized_by="agent",
                               synthesized_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                            time.gmtime())),
            "notes": {
                "readiness_gate": ready.state,
                "readiness_waited_s": round(ready.waited_s, 1),
                "per_tenant_status": ready.per_tenant,
                "detail": ready.detail,
                "why_recorded": (
                    "the readiness gate failed, so the harness never ran and could not write "
                    "a record. Filed by the agent so this placement is not proposed again."),
            },
        }


def _worst_itl_ms(record: dict) -> float:
    """The slowest tenant's measured ITL. inf when any tenant has none.

    A missing ITL is not a zero. The harness suppresses latency entirely when the delivery
    gate fails (impl §6.3), so treating absence as fast would rank an overloaded placement
    first -- the exact survivorship bias the suppression exists to prevent.
    """
    per = record.get("per_tenant") or {}
    if not per:
        return float("inf")
    worst = 0.0
    for r in per.values():
        v = r.get("itl_mean_ms")
        if v is None:
            return float("inf")
        worst = max(worst, float(v))
    return worst
