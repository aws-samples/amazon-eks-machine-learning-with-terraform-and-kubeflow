"""`SplitPackerAgent`: the reasoner chooses which split of a fleet to search, group by group.

    system  = SYSTEM
    turn 1  = problem(fleet, max_nodes, group budget)
    repeat:
        text        = reasoner.converse(system, turns)
        {tool,args} = prompt.parse_call(text, SPLIT_TOOLS)
        turns      += the tool's output as the next user turn
    until `submit`, or the turn cap

`packer.py` one level up. There the reasoner chooses placements on one node and `measure` costs a
GPU run; here it chooses which tenants share a node and `search_group` costs a whole single-node
search -- `packer.py` itself, run on the group's own fleet file, its own lease and its own budget.
What it sees in return is that search's summary, tenant by tenant. Why it is a reasoner and not the
ranking: which tenants interfere is learned only from a node they shared, and a group that missed
is a reason to ask `partition` for splits that keep two of its tenants apart.

**The loop enforces, it does not decide**, as in `packer.py`, and its bounds are the same three:

  - **the turn cap**, because `partition` and `recall` are free.
  - **the fallback.** An unreachable reasoner, or unusable replies after the retries, runs the
    group `splits.next_group` would -- the deterministic driver's next step -- and records that
    nothing chose it. The reasoner is told, and the conversation goes on.
  - **the derived verdict.** `met` only when every group of one listed split was searched and met,
    whatever the reasoner submitted; its own submission is kept beside it.

**Driven over `converse` only.** The SDK runtime matters inside a group, where the gate keeps
`measure` the only route to the GPUs, and every group search still uses it. This conversation has
no tool that reaches a cluster except through that search, and over `converse` it has no builtins
to deny. Offline, `ReplayReasoner` drives it, against a `run_group` that leases nothing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from . import prompt as P
from .reason import Reasoner, Transcript, Turn
from .splits import GroupRun, next_group

log = logging.getLogger(__name__)

SPLIT_TOOLS = ("partition", "search_group", "recall", "submit")
DEFAULT_MAX_TURNS = 40

STOP_SUBMITTED = "the reasoner submitted"
STOP_TURNS = "the turn cap was reached without a submission"
STOP_UNUSABLE = ("the reasoner produced nothing usable and the fallback had no group left to "
                 "search")

SYSTEM = """\
You decide how a fleet of LLM inference tenants is split over a few GPU nodes of one instance type.
Every tenant runs on exactly one node; a node's tenants are a group, and a group is searched as a
fleet of its own: a node is leased, placements of those tenants on its GPUs are measured, and the
search reports whether every tenant met its latency target. Tenants on different nodes share no
GPU, so a group's result is the same in every split that contains it, and it says nothing about a
tenant on another node.

You have four tools. Reply with exactly one JSON object and nothing else:
  {"tool": "<name>", "args": {...}, "note": "<one line: why this call>"}
""" + P.JSON_RULES + """

  partition    args {"apart": [[a, b], ...], "together": [[a, b], ...]}, both optional.
               The splits that fit at every tenant's KV floor, best first: fewest nodes, then
               fewest GPUs, then the largest smallest KV margin. `apart` pairs must be on
               different nodes, `together` pairs on the same one. Free. Splits and groups keep the
               ids they were first listed under.
  search_group args {"tenants": [names], "rationale": "<why this group, now>"}
               Search one group of a split `partition` listed. Costs one group of the budget,
               and a node for as long as the search runs. A group is never searched twice, and
               one an earlier run searched is listed as searched already, at no cost.
  recall       args {"tenants": [names]}, or {} for every group searched so far.
               What a group's search measured, tenant by tenant. Free.
  submit       args {"split": "S<n>" or null, "verdict": "met"|"infeasible"|"budget"|
               "unresolvable", "rationale": "..."}
               Ends the run. "met" needs a listed split every group of which was searched and
               met. Otherwise say which of the others it is, and why.

How to spend the budget: a split is met only if every one of its groups is, so search first the
group most likely to miss, and stop searching a split once one of its groups has missed. A group
that missed is evidence about who should not share a node: `recall` it, and ask `partition` for
splits that keep the tenants that missed apart from the ones that loaded them. The KV margin is
memory arithmetic, not latency; a group with a wide margin can still miss on interference.
"""


class SplitTools:
    """The four tools. Every invariant is here: the budget, no group searched twice, no group
    searched that is not part of a listed split, and no `met` without a met split."""

    def __init__(self, fleet, solve_splits, run_group, budget: int, recall=None):
        self.fleet = fleet
        self.names = [t.name for t in fleet.tenants]
        self.solve_splits = solve_splits      # (apart, together) -> partition.Splits
        self.run_group = run_group            # tenants -> GroupRun
        self.recall = recall                  # tenants -> an earlier run's GroupRun, or None
        self.budget = int(budget)
        self.splits: list = []                # every split listed, in the order first listed
        self.runs: dict = {}                  # frozenset(tenants) -> GroupRun
        self.order: list = []                 # (GroupRun, chosen_by) in the order searched
        self.submitted: dict | None = None

    @property
    def remaining(self) -> int:
        return self.budget - sum(not r.recalled for r in self.runs.values())

    def dispatch(self, tool: str, args: dict) -> str:
        return getattr(self, f"tool_{tool}")(args)

    # --- the tools -------------------------------------------------------------

    def tool_partition(self, args) -> str:
        apart = self._pairs(args.get("apart"), "apart")
        together = self._pairs(args.get("together"), "together")
        try:
            found = self.solve_splits(apart, together)
        except ValueError as exc:
            raise P.ToolError(str(exc)) from None
        listed = [self._register(s) for s in found.splits]
        if not listed:
            if found.unplaced:
                why = "; ".join(f"{n}: {w}" for n, w in found.unplaced.items())
                return f"no split: a tenant fits no node even alone ({why})."
            return ("no split over at most %d node(s) under those constraints; %s." % (
                found.max_nodes, "the weights alone rule one out" if found.proven_none
                else "none was found, which is not a proof that none exists"))
        out = [f"{len(listed)} split(s), best first"
               + ("" if found.exhausted else "; there may be more") + ":"]
        for i in listed:
            out += self._split_lines(i)
        return "\n".join(out)

    def tool_search_group(self, args, chosen_by: str = "reasoner") -> str:
        tenants = self._tenants(args.get("tenants"))
        key = frozenset(tenants)
        if key in self.runs:
            raise P.ToolError(f"that group was already searched ({self.runs[key].verdict}); "
                              "`recall` it. A group is never searched twice.")
        listed = self._listed_group(key)
        if listed is None:
            raise P.ToolError("that group is not a group of any split `partition` has listed. "
                              "Search only a listed group, so that it belongs to a whole split.")
        if self.remaining <= 0:
            raise P.ToolError(f"the group budget of {self.budget} is spent. Submit.")
        if chosen_by == "reasoner" and not isinstance(args.get("rationale"), str):
            raise P.ToolError("'rationale' is required: one or two sentences on why this group, "
                              "now.")
        run = self.run_group(listed)
        self.runs[key] = run
        self.order.append((run, chosen_by))
        return "\n".join(self._run_lines(run) + [""] + self._standing_lines()
                         + [f"{self.remaining} group search(es) left."])

    def tool_recall(self, args) -> str:
        if not args.get("tenants"):
            if not self.order:
                return "no group has been searched yet."
            return "\n".join(ln for run, _ in self.order for ln in self._run_lines(run))
        key = frozenset(self._tenants(args.get("tenants")))
        if key not in self.runs:
            raise P.ToolError("that group has not been searched.")
        return "\n".join(self._run_lines(self.runs[key]))

    def tool_submit(self, args) -> str:
        verdict = args.get("verdict")
        if verdict not in (P.DERIVED_VERDICT,) + P.VERDICTS:
            raise P.ToolError(f"'verdict' must be one of {P.DERIVED_VERDICT}, "
                              f"{', '.join(P.VERDICTS)}; got {verdict!r}.")
        sid = args.get("split")
        if sid is not None and self._split_index(sid) is None:
            raise P.ToolError(f"there is no listed split {sid!r}.")
        if verdict == P.DERIVED_VERDICT and (sid is None or not self.met(self._split_index(sid))):
            raise P.ToolError("'met' needs a listed split every group of which was searched and "
                              "met. " + "\n".join(self._standing_lines()))
        if not isinstance(args.get("rationale"), str) or not args["rationale"].strip():
            raise P.ToolError("'rationale' is required.")
        self.submitted = {"split": sid, "verdict": verdict, "rationale": args["rationale"]}
        return "submitted"

    # --- what the loop reads ---------------------------------------------------

    def met(self, i: int) -> bool:
        return all(frozenset(g.tenants) in self.runs and self.runs[frozenset(g.tenants)].met
                   for g in self.splits[i].groups)

    def met_split(self):
        """The submitted split if it is met, else the best-ranked listed split that is."""
        sid = (self.submitted or {}).get("split")
        if sid is not None and self.met(self._split_index(sid)):
            return self.splits[self._split_index(sid)]
        met = [s for i, s in enumerate(self.splits) if self.met(i)]
        return min(met, key=lambda s: s.rank_key) if met else None

    # --- helpers ---------------------------------------------------------------

    def _register(self, s) -> int:
        key = frozenset(frozenset(g.tenants) for g in s.groups)
        for i, known in enumerate(self.splits):
            if frozenset(frozenset(g.tenants) for g in known.groups) == key:
                return i
        self.splits.append(s)
        for g in s.groups:
            k = frozenset(g.tenants)
            run = None if k in self.runs or self.recall is None else self.recall(tuple(g.tenants))
            if run is not None:
                self.runs[k] = run
                self.order.append((run, "store"))
        return len(self.splits) - 1

    def _split_index(self, sid):
        if isinstance(sid, str) and sid.upper().startswith("S") and sid[1:].isdigit():
            i = int(sid[1:]) - 1
            return i if 0 <= i < len(self.splits) else None
        return None

    def _listed_group(self, key):
        for s in self.splits:
            for g in s.groups:
                if frozenset(g.tenants) == key:
                    return tuple(g.tenants)
        return None

    def _tenants(self, tenants) -> list:
        if not isinstance(tenants, list) or not tenants or not all(
                isinstance(n, str) for n in tenants):
            raise P.ToolError("'tenants' must be a non-empty list of tenant names.")
        unknown = sorted(set(tenants) - set(self.names))
        if unknown:
            raise P.ToolError(f"no tenant {', '.join(unknown)}; the tenants are "
                              f"{', '.join(self.names)}.")
        return tenants

    def _pairs(self, pairs, what) -> list:
        if pairs is None:
            return []
        if not isinstance(pairs, list) or not all(
                isinstance(p, list) and len(p) == 2 for p in pairs):
            raise P.ToolError(f"'{what}' must be a list of [name, name] pairs.")
        for p in pairs:
            self._tenants(p)
        return [tuple(p) for p in pairs]

    def _split_lines(self, i) -> list:
        s = self.splits[i]
        m = s.min_margin
        out = [f"  S{i + 1}: {s.nodes} node(s), {s.gpus} GPU(s), smallest KV margin "
               f"{'-' if m is None else f'{m:.2f}x'}"]
        for g in s.groups:
            run = self.runs.get(frozenset(g.tenants))
            state = "not searched" if run is None else (
                f"searched{' by an earlier run' if run.recalled else ''}: "
                f"{run.verdict or 'no verdict'}")
            gm = g.min_margin
            out.append(f"      {g.gpus} GPU(s), margin {'-' if gm is None else f'{gm:.2f}x'}, "
                       f"{state}: {', '.join(g.tenants)}")
        return out

    def _run_lines(self, run: GroupRun) -> list:
        s = run.summary or {}
        out = [f"group {', '.join(run.tenants)}: "
               f"{run.verdict or 'no verdict (its search wrote no summary)'}"]
        if s.get("stopped_because"):
            out.append(f"  its search stopped: {s['stopped_because']}")
        for a in s.get("attempts", []):
            met = a.get("targets_met") or {}
            itl = a.get("per_tenant_itl_ms") or {}
            cells = []
            for n in run.tenants:
                v = itl.get(n)
                verdict = {True: "met", False: "MISSED", None: "unjudged"}[met.get(n)]
                cells.append(f"{n} {'-' if v is None else f'{v:.1f}ms'} {verdict}")
            out.append(f"  {a.get('run_id')} ({a.get('outcome')}, {a.get('gpus_used')} GPU(s)): "
                       + "; ".join(cells))
        return out

    def _standing_lines(self) -> list:
        out = ["the listed splits now:"]
        for i in range(len(self.splits)):
            out += self._split_lines(i)
        return out


@dataclass
class SplitResult:
    """What the run did, in the shape `_search_splits` reports `splits.DriveResult` in."""

    split: object | None = None
    runs: list = field(default_factory=list)
    chosen_by: list = field(default_factory=list)   # "reasoner", "fallback" or "store", per run
    verdict: str | None = None
    submitted: dict | None = None
    stopped_because: str = ""
    turns: int = 0
    transcript: Transcript | None = None


class SplitPackerAgent:
    def __init__(self, tools: SplitTools, reasoner: Reasoner, *, max_nodes: int,
                 instance_type: str, max_turns: int = DEFAULT_MAX_TURNS, max_retries: int = 2):
        self.tools = tools
        self.reasoner = reasoner
        self.max_nodes = int(max_nodes)
        self.instance_type = instance_type
        self.max_turns = int(max_turns)
        self.max_retries = int(max_retries)

    def opening(self) -> str:
        fleet, tools = self.tools.fleet, self.tools
        node = fleet.node
        out = ["== THE PROBLEM ==",
               f"  up to {self.max_nodes} node(s) of {self.instance_type}, each with "
               f"{len(node.gpu_ids)} GPUs of {node.device_total_gib:.2f} GiB", "",
               f"  {'tenant':<12}{'model':<28}{'tp':>3}{'rate':>9}{'in':>7}{'out':>6}  target"]
        for t in fleet.tenants:
            slo = "--" if t.slo is None else ", ".join(
                f"{n}={getattr(t.slo, n):g}" for n in t.slo.stated)
            out.append(f"  {t.name:<12}{t.model_id:<28}{t.tp:>3}{t.rate_hz:>7.1f}Hz"
                       f"{t.isl_tokens:>7}{t.osl_tokens:>6}  {slo}")
        out += ["", "== BUDGET ==",
                f"  {tools.budget} group search(es). `partition` and `recall` are free.", "",
                P.JSON_CLOSING]
        return "\n".join(out)

    def run(self) -> SplitResult:
        tools = self.tools
        transcript = Transcript(reasoner=self.reasoner.name, system=SYSTEM,
                                turns=[Turn("user", self.opening())])
        result = SplitResult(transcript=transcript)
        t0, invalid = time.time(), 0
        while result.turns < self.max_turns:
            result.turns += 1
            try:
                text, usage = self.reasoner.converse(SYSTEM, transcript.turns)
            except Exception as exc:            # noqa: BLE001 - degrade, do not lose the run
                log.exception("the reasoner could not be reached")
                if self._fall_back(f"the reasoner could not be reached: {exc}", transcript):
                    continue
                result.stopped_because = STOP_UNUSABLE
                break
            transcript.turns.append(Turn("assistant", text))
            transcript.usage = dict(usage or {})
            transcript.latency_s = time.time() - t0
            try:
                call = P.parse_call(text, SPLIT_TOOLS)
            except P.InvalidReply as exc:
                log.warning("unusable reply (%s): %.600s", exc, text)
                invalid += 1
                transcript.retries = invalid
                if invalid <= self.max_retries:
                    transcript.turns.append(Turn("user", f"That reply cannot be used: {exc}"))
                    continue
                if self._fall_back(f"unusable replies: {exc}", transcript):
                    invalid = 0
                    continue
                result.stopped_because = STOP_UNUSABLE
                break
            invalid = 0
            log.info("turn %d: %s %s", result.turns, call["tool"], call["note"])
            try:
                out = tools.dispatch(call["tool"], call["args"])
            except P.ToolError as exc:
                transcript.turns.append(Turn("user", f"{call['tool']} refused: {exc}"))
                continue
            if call["tool"] == "submit":
                result.submitted = dict(tools.submitted)
                result.stopped_because = STOP_SUBMITTED
                break
            transcript.turns.append(Turn("user", out))
        else:
            result.stopped_because = STOP_TURNS

        result.runs = [run for run, _ in tools.order]
        result.chosen_by = [by for _, by in tools.order]
        result.split = tools.met_split()
        result.verdict = (P.DERIVED_VERDICT if result.split is not None
                          else (result.submitted or {}).get("verdict")
                          or ("budget" if tools.remaining <= 0 else "unresolvable"))
        return result

    def _fall_back(self, why: str, transcript: Transcript) -> bool:
        """The deterministic driver's next group, searched on the reasoner's behalf and marked."""
        tools = self.tools
        if not tools.splits:
            tools.tool_partition({})
        group = next_group(tools.splits, tools.runs)
        if group is None or tools.remaining <= 0:
            return False
        log.warning("falling back to the driver's next group %s: %s", group, why)
        out = tools.tool_search_group({"tenants": list(group)}, chosen_by="fallback")
        transcript.turns.append(Turn("user", (
            f"Your last replies could not be used ({why}), so the loop searched the group the "
            f"deterministic driver would have: {', '.join(group)}. It is recorded as a fallback, "
            f"not as your choice.\n{out}\n{P.JSON_CLOSING}")))
        return True
