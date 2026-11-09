"""The six tools the reasoner drives the search with (impl §11.1).

    solve            the feasible set. Cheap, deterministic, a scratchpad.
    recall           prior runs bearing on a placement. Runs, never a number.
    read_run         one record in full.
    measure          deploy, gate, drive load, record, tear down. The only expensive tool, the
                     only mutating one, and the only one behind the budget.
    cluster_status   what is running right now.
    submit           end the run.

**The invariants live here, not in the prompt.** impl §11.2 lists what the reasoner must not be
able to do, and every one of them is enforced by the tool that would otherwise do it:

  - a pinned assignment is checked structurally (`prompt.check_assignment`), then by the solver
    (`solve_pinned`), then independently by `Placement.verify()` in plain arithmetic. Three
    checks, because the first cannot see memory, and the second and third share no code -- a
    constraint expressed wrongly in the CP-SAT model is exactly what the third one catches.
  - the fleet, the targets, the offered rates and the request shapes are not arguments to any
    tool. There is nothing to enforce because there is nothing to change.
  - `measure` requires an `expectation` for every tenant, and refuses `hold` or `miss` for a
    tenant that stated no target.
  - `submit` refuses `verdict: "met"` unless a record from this run actually shows every stated
    target held. The reasoner cannot certify; the loop derives certification from a measurement.
  - the budget is checked at the top of `measure`, before anything is deployed. Being told the
    budget in the prompt is not a control.

A wrong belief is allowed to cost iterations and must not cost correctness. So `avoid` -- the
reasoner's own cuts -- narrows what `solve` offers and is never written to the store, and
`recall` feeds proposals only: no tool consults it to decide whether a placement is feasible.

Every tool returns **text**, and it is text written to be read by a model rather than parsed.
That is what makes these objects runtime-agnostic: the same six functions serve the loop in
`packer.py` today and an SDK tool decorator in Phase 7 (impl §11.4) without changing what a tool
means. `ToolError` is raised for a call that is understood and must not run, and its message is
addressed to the reasoner -- a refusal it cannot read is a refusal it will repeat.
"""

from __future__ import annotations

import logging

from ..retrieval import placement_of, read_run, recall, sm_caps_of, targets_met
from ..solver.cpsat import Infeasible, PlacementSolver
from ..store.schema import Observation
from . import prompt as P
from .policy import Proposal, canonical
from .spec import AGENT_ENVELOPE_STEP_MIB

log = logging.getLogger(__name__)

SUBMITTED = "submitted"


class Tools:
    """The tool set for one run. Holds the run-scoped state a tool call may see.

    Stateful on purpose, and the state is exactly three things: how much budget is spent, the
    cuts the reasoner has declared, and what this run has measured. Nothing else persists between
    calls -- the store is re-read on every call that touches it, so a tool sees what a fresh
    process would see, and a resumed run cannot be told a different history than a new one.
    """

    def __init__(self, fleet, store, agent, *, budget: int, top_k: int = 4,
                 solver_seconds: float = 10.0, envelope_step_mib: int = AGENT_ENVELOPE_STEP_MIB,
                 reasoner_name: str = "unknown"):
        self.fleet = fleet
        self.store = store
        self.agent = agent                 # SearchAgent: the mechanical executor of `measure`
        self.reasoner_name = reasoner_name
        self.budget = int(budget)
        self.top_k = int(top_k)
        self.solver_seconds = float(solver_seconds)
        self.envelope_step_mib = int(envelope_step_mib)

        self.spent = 0
        self.avoid: tuple = ()             # run-scoped cuts; never written to the store
        self.attempts: list = []           # Attempt, in order
        self.expectations: dict = {}       # run_id -> {tenant: hold|miss|unknown}
        self.last_feasible = None          # so `measure` can resolve "candidate N" as offered
        self.submitted: dict | None = None
        self.calls: list = []              # (tool, args) in order, for the run summary

    # --- what the tools share ------------------------------------------------

    @property
    def targets(self) -> dict:
        """`{tenant: TenantSlo | None}`. The targets in force now, from the fleet file."""
        return {t.name: t.slo for t in self.fleet.tenants}

    @property
    def gpu_of(self) -> dict:
        return {u: g for g, u in zip(self.fleet.node.gpu_ids, self.fleet.node.uuids)}

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.spent)

    def _solver(self) -> PlacementSolver:
        return PlacementSolver(self.fleet.node.node_spec, self.fleet.requests,
                               envelope_step_mib=self.envelope_step_mib)

    def records(self) -> list:
        """Every readable record under this key, including the ones that never measured.

        Deliberately *not* `SearchAgent._observations`, which keeps only records usable as evidence
        about latency. A placement that failed its readiness gate is evidence about feasibility
        (impl §8) and is what stops it being proposed a second time, so retrieval has to see it --
        and retrieval is what marks it as feasibility-only rather than mixing it in.
        """
        report = self.store.load()
        if report.unreadable:
            raise RuntimeError(
                f"{len(report.unreadable)} records under {self.store.key_hash} could not be read: "
                f"{report.unreadable}. Fix or move them; searching against a partial history "
                "silently changes what the agent believes.")
        return list(report.observations)

    def measured_keys(self) -> set:
        """Placements the store already holds, as far as a record's UUIDs resolve to this node."""
        of = self.gpu_of
        return {canonical(p, sm_caps_of(o))
                for o, p in ((o, placement_of(o, of)) for o in self.records()) if p}

    def dispatch(self, tool: str, args: dict) -> str:
        fn = getattr(self, f"tool_{tool}", None)
        if fn is None:                     # `parse_call` already rejected unknown names
            raise P.ToolError(f"there is no tool called {tool!r}.")
        self.calls.append((tool, args))
        return fn(args)

    # --- solve ---------------------------------------------------------------

    def tool_solve(self, args: dict) -> str:
        """The feasible set, or one pinned assignment checked. Deterministic and repeatable.

        `avoid` here is additive with the run's standing cuts: a cut declared on a `measure` call
        stays in force for the rest of the run, and one passed to `solve` applies to that call.
        Both are the reasoner's beliefs, so neither reaches the store.
        """
        assign = args.get("assignment")
        if assign is not None:
            checked = P.check_assignment(assign, self.fleet)
            candidate = self._pin(checked)
            # `last_feasible` is left as it was rather than replaced by this one candidate:
            # "candidate 2" must keep meaning what the last `solve` printed, or a pin would
            # silently renumber the list the reasoner is holding in its head.
            out = ["the placement you named is FEASIBLE."]
            out += P.candidate_lines([candidate], measured=self.measured_keys())
            out.append("")
            out.append("  It is not in the numbered candidate list; to measure it, pass the same "
                       "\"assignment\" to `measure`.")
            return "\n".join(out)

        cap = args.get("cap", self.top_k)
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            raise P.ToolError(f"'cap' must be a positive integer; got {cap!r}.")
        extra = P.check_avoid(args.get("avoid"), self.fleet)
        cuts = tuple(dict.fromkeys(self.avoid + extra))
        try:
            feasible = self._solver().solve(cap=cap, max_seconds=self.solver_seconds,
                                            avoid=cuts)
        except Infeasible as exc:
            # An answer about the fleet, not an error. And it is only an answer about the fleet
            # *under these cuts*: a cut the reasoner invented can make a hostable fleet look
            # unhostable, which would be a wrong conclusion drawn from its own belief.
            self.last_feasible = None
            return (f"the solver found nothing: {exc}\n"
                    + (f"  Your cuts {_pairs(cuts)} were in force. Feasibility under your own cuts "
                       "is not feasibility; drop them and solve again before concluding anything "
                       "about the fleet.\n" if cuts else "")
                    + "  This is a statement about capacity and floors, not about latency.")
        self.last_feasible = feasible
        head = ["the feasible set" + (f", with your cuts {_pairs(cuts)} in force" if cuts else "")
                + f" (cap {cap}):"]
        return "\n".join(head + P.candidate_lines(feasible, measured=self.measured_keys()))

    def _pin(self, checked: dict):
        """A named assignment, checked three independent ways. Raises `ToolError`.

        The third check is not redundancy for its own sake. `solve_pinned` asks the CP-SAT model
        whether the assignment satisfies the constraints *as expressed in that model*, and
        `verify` re-derives them in plain arithmetic; only disagreement between the two can catch a
        constraint that was written down wrongly. `SolverBug` is deliberately not converted into a
        `ToolError` -- it is not something the reasoner can fix by proposing differently, and
        telling it to try again would bury the bug under a retry.
        """
        try:
            candidate = self._solver().solve_pinned(checked, max_seconds=self.solver_seconds)
        except Infeasible as exc:
            raise P.ToolError(
                f"that placement cannot be hosted on this node: {exc}. Every tenant needs an "
                "envelope large enough for its weights, its KV pool and its activations, and the "
                "envelopes on any one GPU must fit inside the measured device total. Propose a "
                "different placement.") from None
        candidate.placement.verify(self.fleet.node.node_spec, self.fleet.requests)
        return candidate

    # --- recall / read_run ---------------------------------------------------

    def tool_recall(self, args: dict) -> str:
        """Prior runs. No average, no ranking, and "no evidence" when there is none."""
        assign, tenants = args.get("assignment"), args.get("tenants") or ()
        if args.get("candidate") is not None:
            # As lists, because it goes through `check_assignment` below like a named one: the
            # candidate came from the solver, but the checker's contract is what the model sends.
            offered = self._as_offered(args["candidate"]).placement.assignment
            assign = {n: [int(g) for g in gpus] for n, gpus in offered.items()}
        if assign is not None and not isinstance(assign, dict):
            raise P.ToolError("'assignment' must be a JSON object mapping tenant to GPU list.")
        if assign is not None:
            assign = P.check_assignment(assign, self.fleet)
        if args.get("sm_caps") is not None and assign is None:
            raise P.ToolError("'sm_caps' needs a placement beside it -- an \"assignment\" or a "
                              "\"candidate\" -- because a cap is part of a placement.")
        sm_caps = P.check_sm_caps(args.get("sm_caps"), self.fleet, assign) if assign else {}
        if tenants:
            if not isinstance(tenants, list) or not all(isinstance(t, str) for t in tenants):
                raise P.ToolError("'tenants' must be a list of tenant names.")
            unknown = sorted(set(tenants) - {t.name for t in self.fleet.tenants})
            if unknown:
                raise P.ToolError(f"'tenants' names {unknown}, which is not in this fleet.")
        evidence = recall(self.records(), targets=self.targets, assignment=assign,
                          tenants=tuple(tenants), gpu_of=self.gpu_of,
                          key_hash=self.store.key_hash,
                          calibration=self.store.read_calibration(), sm_caps=sm_caps)
        return "\n".join(evidence.lines())

    def tool_read_run(self, args: dict) -> str:
        run_id = args.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise P.ToolError("'run_id' must be the id of a run, as `recall` printed it.")
        run_id = run_id.strip()
        for obs in self.records():
            if obs.run_id == run_id:
                return "\n".join(read_run(obs, self.targets, self.gpu_of))
        known = sorted(o.run_id for o in self.records())
        raise P.ToolError(f"no run called {run_id!r} is in the store under "
                          f"{self.store.key_hash}. It holds: {', '.join(known) or 'nothing'}.")

    # --- measure -------------------------------------------------------------

    def tool_measure(self, args: dict, *, provenance_extra: dict | None = None) -> str:
        """The only tool that spends anything. Budget first, then the invariants, then the GPUs.

        Order matters here. The budget is checked before the arguments are validated, so a
        malformed call cannot be the thing that discovers the budget is gone -- and before
        anything is deployed, because a gate that fires after `helm install` is not a budget.

        `provenance_extra` is a Python argument, not a tool argument: it cannot appear in `args`
        and so the reasoner cannot set it. It exists for the one caller that needs to overwrite
        `policy_kind` -- the loop's fallback, which measures something the reasoner did not choose
        and has to say so *in the record*, not only in the run summary.
        """
        if self.remaining <= 0:
            raise P.ToolError(
                f"the budget is spent: {self.spent} of {self.budget} measurement(s) used. No "
                "further measurement will run. Submit your conclusion -- what the runs you have "
                "settle, and what they do not.")
        choice, source = self._choice(args.get("choice"))
        expectation = P.check_expectation(args.get("expectation"), self.fleet)
        rationale = P.check_rationale(args.get("rationale"))
        label = P.check_label(args.get("label", f"m{len(self.attempts):02d}"))
        cuts = P.check_avoid(args.get("avoid"), self.fleet)
        sm_caps = P.check_sm_caps(args.get("sm_caps"), self.fleet, choice.placement.assignment)
        if cuts:
            # Kept for the rest of the run, and only for the rest of the run (impl §11.2).
            self.avoid = tuple(dict.fromkeys(self.avoid + cuts))

        key = canonical(choice.placement.assignment, sm_caps)
        repeat = key in self.measured_keys()
        proposal = Proposal(choice, label, f"{rationale} [{source}]", sm_caps=sm_caps, provenance={
            "policy_kind": "model-packer-agent",
            "chosen_by_reasoner": self.reasoner_name,
            "reasoner_rationale": rationale,
            "reasoner_choice_source": source,
            # The falsifiable half of impl §11.2, written *before* the measurement. Whether it
            # turned out right is not written here: it is derived at read time from this field and
            # the record's own metrics, the same way a target verdict is (retrieval, design §2).
            # A stored outcome would freeze a comparison against the targets of the day.
            "reasoner_expectation": dict(expectation),
            "reasoner_declared_repeat": repeat,
            # The cuts in force are deliberately NOT here. A cut is a belief the reasoner formed,
            # and impl §11.2 keeps it run-scoped: a record that carried it would hand a later run
            # a belief wearing the authority of a measurement. Where the reasoner's reasons belong
            # is `reasoner_rationale`, in its own words, as a claim.
            "measurement_index": self.spent,
            "measurement_budget": self.budget,
            **dict(provenance_extra or {}),
        })

        self.spent += 1
        attempt = self.agent.execute(proposal, len(self.attempts))
        self.attempts.append(attempt)
        self.expectations[attempt.run_id] = dict(expectation)
        return "\n".join(self._measured_lines(attempt, expectation))

    def _choice(self, choice) -> tuple:
        if not isinstance(choice, dict) or not choice:
            raise P.ToolError(
                "'choice' must be a JSON object: either {\"candidate\": <n>} naming one of the "
                "candidates `solve` last printed, or {\"assignment\": {\"<tenant>\": [<gpu>, "
                "...]}} naming a placement yourself.")
        if "candidate" in choice and "assignment" in choice:
            raise P.ToolError("'choice' names both a candidate and an assignment. Send one: they "
                              "would otherwise disagree and nothing here would know which you "
                              "meant.")
        if "assignment" in choice:
            checked = P.check_assignment(choice["assignment"], self.fleet)
            return self._pin(checked), "a placement it named itself"
        candidate = self._as_offered(choice.get("candidate"))
        return candidate, f"candidate {choice['candidate']} of the last solve, as offered"

    def _as_offered(self, n):
        if self.last_feasible is None:
            raise P.ToolError("there is no candidate list to index: call `solve` first, or name a "
                              "placement yourself with \"assignment\".")
        if not isinstance(n, int) or isinstance(n, bool):
            raise P.ToolError(f"'candidate' must be an integer, 1 to {len(self.last_feasible)}; "
                              f"got {n!r}.")
        if not 1 <= n <= len(self.last_feasible):
            raise P.ToolError(f"there is no candidate {n}: the last solve returned "
                              f"{len(self.last_feasible)}.")
        return self.last_feasible[n - 1]

    def _measured_lines(self, attempt, expectation) -> list:
        out = [f"run {attempt.run_id} finished: {attempt.outcome}",
               f"  {self.remaining} of {self.budget} measurement(s) left."]
        if not attempt.record:
            return out + ["  no record was written, so this run says nothing about latency."]
        obs = Observation.parse(attempt.record)
        out.append("")
        out += read_run(obs, self.targets, self.gpu_of)
        out.append("")
        out += expectation_lines(expectation, targets_met(obs, self.targets))
        return out

    # --- cluster_status / submit ---------------------------------------------

    def tool_cluster_status(self, args: dict) -> str:
        """What is running. Operational, and read-only: it installs and destroys nothing.

        Here because a run can be interrupted between deploying and tearing down, and a reasoner
        that cannot see the node has to infer the cluster's state from its own transcript -- which
        is exactly the thing that is wrong after an interruption.
        """
        live = sorted(self.agent.deployer.live_tenant_releases())
        out = [f"gpu node under test {self.fleet.node.node_name}, "
               f"{len(self.fleet.node.gpu_ids)} GPU(s): gpu"
               + ", gpu".join(str(g) for g in self.fleet.node.gpu_ids)]
        out.append(f"  tenant releases installed right now: {', '.join(live) or 'none'}")
        out.append(f"  measurements this run: {self.spent} of {self.budget}"
                   + (f" -- {', '.join(a.run_id for a in self.attempts)}" if self.attempts else ""))
        if self.avoid:
            out.append(f"  your cuts in force: {_pairs(self.avoid)} (this run only; not stored)")
        return "\n".join(out)

    def tool_submit(self, args: dict) -> str:
        """End the run. Refuses a certification the records do not support.

        This is the one place a reasoner could quietly change the answer: declaring the fleet met
        its targets is the whole output of the search, and it is derived from a record here rather
        than accepted as a claim. The refusal names what the records actually show, so it is a
        correction rather than a wall.
        """
        verdict = args.get("verdict")
        if not isinstance(verdict, str) or verdict.strip().lower() not in (
                P.VERDICTS + (P.DERIVED_VERDICT,)):
            raise P.ToolError(
                f"'verdict' must be one of {P.DERIVED_VERDICT}, {', '.join(P.VERDICTS)}; got "
                f"{verdict!r}.")
        verdict = verdict.strip().lower()
        rationale = P.check_rationale(args.get("rationale"))
        met_run = self.met_run()
        if verdict == P.DERIVED_VERDICT and met_run is None:
            raise P.ToolError(
                "\"met\" is not yours to declare, and no record from this run supports it. It is "
                "derived from a measurement in which every tenant that stated a target held it. "
                f"What this run measured is: {self._met_summary() or 'nothing'}. Submit "
                f"{' or '.join(P.VERDICTS)} instead, and say in your rationale what the evidence "
                "does and does not settle.")
        choice = None
        if args.get("choice"):
            candidate, _ = self._choice(args["choice"])
            choice = {n: list(g) for n, g in candidate.placement.assignment.items()}
        deploy = bool(args.get("deploy"))
        self.submitted = {"verdict": verdict, "rationale": rationale, "choice": choice,
                          "leave_deployed": deploy,
                          "derived_met_run": met_run.run_id if met_run else None}
        return SUBMITTED

    def met_run(self):
        """The attempt in *this run* whose record holds every stated target, or None.

        Judged from the record at read time by `retrieval.targets_met`, so a tightened target
        un-certifies an old run rather than leaving a stale pass behind. A tenant with no verdict
        (None) does not count as a pass: a suppressed latency is not a met target, and at least
        one tenant must actually have a target for the question to mean anything.
        """
        for attempt in self.attempts:
            if not attempt.record:
                continue
            met = targets_met(Observation.parse(attempt.record), self.targets)
            if any(v is True for v in met.values()) and not any(v is False for v in met.values()):
                if all(v is not None for n, v in met.items() if self.targets[n] is not None):
                    return attempt
        return None

    def _met_summary(self) -> str:
        parts = []
        for attempt in self.attempts:
            if not attempt.record:
                parts.append(f"{attempt.run_id}: no record ({attempt.outcome})")
                continue
            met = targets_met(Observation.parse(attempt.record), self.targets)
            shown = ", ".join(
                f"{n} {'met' if v is True else 'MISSED' if v is False else 'no verdict'}"
                for n, v in sorted(met.items()))
            parts.append(f"{attempt.run_id}: {shown}")
        return "; ".join(parts)


def _pairs(cuts) -> list:
    """Cuts as the reasoner wrote them: lists, not Python tuples.

    Small, and worth doing: it is quoted back in the text the model reads next, and a reply that
    copies `[('r2', 't2')]` back into JSON is not valid JSON.
    """
    return [list(p) for p in cuts]


def expectation_lines(expectation: dict, met: dict) -> list:
    """What was expected against what happened. Derived, never stored.

    impl §11.2's reason for requiring the field: this comparison separates a wrong belief from a
    wrong choice. A placement that missed as expected is information about the fleet; one that
    missed unexpectedly is information about the reasoner's model of it, and the two call for
    different next moves. `unknown` is not scored, because declaring ignorance in advance is the
    honest move the design asks for and turning it into a miss would penalise it.
    """
    out = ["  what you expected, against what happened:"]
    for name in sorted(expectation):
        want, got = expectation[name], met.get(name)
        shown = {True: "met", False: "MISSED", None: "no verdict"}[got]
        if want == "unknown":
            note = "you said you did not know"
        elif got is None:
            note = f"you expected it to {want}, and this run cannot say"
        elif (want == "hold") == bool(got):
            note = f"as you expected ({want})"
        else:
            note = f"NOT as you expected: you said {want}"
        out.append(f"    {name:<8} expected {want:<8} measured {shown:<10} {note}")
    return out
