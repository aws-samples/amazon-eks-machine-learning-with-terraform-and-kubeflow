"""`ModelPackerAgent`: the reasoner *is* the loop (impl §11).

    system  = prompt.SYSTEM
    turn 1  = prompt.problem(fleet, budget, ...)
    repeat:
        text        = reasoner.converse(system, turns)
        {tool,args} = prompt.parse_call(text)
        turns      += the tool's output as the next user turn
    until `submit`, or the budget and the turn cap are both spent

That is the whole of it, and the shape is the point. The previous design asked the model one
question per iteration -- here is the store, here are four candidates, pick one -- and everything
the model could not see in that one rendering was invisible to it forever. This one hands it the
problem and six tools, and what it looks at is its own decision: `solve` as often as it likes,
`recall` on a placement, `read_run` on the two records that disagree, then one `measure`.

**The loop enforces, it does not decide.** Every judgement is the reasoner's; every invariant is
mechanical and lives in `tools.py`, at the boundary where the action would happen. This module
adds exactly three things of its own, and each is a bound rather than a policy:

  - **the turn cap.** `measure` is behind the budget, but `solve` and `recall` are free, and a
    model that keeps asking free questions forever would never terminate. The cap bounds the
    conversation, not the search: hitting it stops the run with `verdict: budget` and says so.
    It costs tokens, not GPU time, so it is set loosely -- it exists to make non-termination
    impossible, not to hurry the reasoning.
  - **the malformed-reply fallback (impl §11.2).** An unusable reply is fed back with the reason,
    as `reason.ask` does, because being shown its own words is what actually fixes a format error.
    After the retries, the run does not die: it measures the solver's top-ranked candidate with
    every expectation `unknown` -- a packing choice, not a prediction -- and **records that this
    is what happened**. A silent fallback would leave a record claiming a reasoner chose a
    placement that it never saw.
  - **the derived verdict.** `tools.submit` refuses `met` unless a record supports it; here the
    same derivation runs independently of what the reasoner said, so a run that met its targets
    is reported as having met them even if the reasoner submitted something else. Certification
    comes from a measurement in both directions.

**This is no longer the shipped runtime; `sdkpacker.py` is.** impl §11 names the Claude Agent SDK,
and that driver now exists, emitting real `tool_use` blocks against the same `Tools` object. This
loop survives for the reason it was written: impl §12 step 6 requires the component exercised
offline against a stub reasoner and the deploy/harness fakes, and the SDK's own loop cannot be
driven by `ReplayReasoner` -- it needs a subprocess, credentials and a network. So the JSON-in-text
protocol is what the self-tests drive, and every invariant they check lives in `tools.py`, at the
boundary, where it holds for both drivers.

Kept as a test double, then, not as a control arm: it decides nothing the SDK driver does not, and
the two share `recorded_fallback` and `derive_verdict` below rather than each having their own. What
is genuinely untestable offline is narrow and stated where it lives -- see `sdkpacker.py`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..store.schema import Observation
from ..retrieval import targets_met
from . import prompt as P
from .policy import TopRanked
from .reason import Reasoner, Transcript, Turn
from .tools import SUBMITTED, Tools, expectation_lines

log = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 60

# The `policy_kind` written into a record nothing chose. Read by `__main__` to report it, and by
# anyone reading the store later who needs to know that this measurement answers no question.
PACKER_FALLBACK = "packer-fallback"

STOP_SUBMITTED = "the reasoner submitted"
STOP_TURNS = "the turn cap was reached without a submission"
STOP_UNUSABLE = "the reasoner produced nothing usable and there was no budget left to fall back on"


@dataclass
class PackerResult:
    """What the run did, and what it concluded. The verdict is derived, not accepted."""

    attempts: list = field(default_factory=list)
    verdict: str | None = None            # what the loop derived, which may not be the submitted one
    submitted: dict | None = None         # exactly what `submit` was called with, or None
    stopped_because: str = ""
    turns: int = 0
    fallbacks: list = field(default_factory=list)   # run ids the reasoner did not choose
    transcript: Transcript | None = None

    @property
    def measured(self) -> list:
        return [a for a in self.attempts if a.record]

    def met_run_id(self) -> str | None:
        return (self.submitted or {}).get("derived_met_run")


class ModelPackerAgent:
    def __init__(self, fleet, store, agent, reasoner: Reasoner, *, budget: int = 6,
                 top_k: int = 4, solver_seconds: float = 10.0,
                 max_turns: int = DEFAULT_MAX_TURNS, max_retries: int = 2,
                 include_prompt: bool = True):
        self.fleet = fleet
        self.store = store
        self.agent = agent
        self.reasoner = reasoner
        self.budget = int(budget)
        self.top_k = int(top_k)
        self.solver_seconds = float(solver_seconds)
        self.max_turns = int(max_turns)
        self.max_retries = int(max_retries)
        self.include_prompt = include_prompt

    def run(self, *, teardown_at_end: bool = False) -> PackerResult:
        tools = Tools(self.fleet, self.store, self.agent, budget=self.budget, top_k=self.top_k,
                      solver_seconds=self.solver_seconds, reasoner_name=self.reasoner.name)
        opening = P.problem(fleet=self.fleet, budget=self.budget,
                            store_runs=len(tools.records()), key_hash=self.store.key_hash,
                            calibration=self.store.read_calibration())
        transcript = Transcript(reasoner=self.reasoner.name, system=P.SYSTEM,
                                turns=[Turn("user", opening)])
        result = PackerResult(attempts=tools.attempts, transcript=transcript)
        t0 = time.time()
        invalid = 0

        while result.turns < self.max_turns:
            result.turns += 1
            try:
                text, usage = self.reasoner.converse(P.SYSTEM, transcript.turns)
            except Exception as exc:            # noqa: BLE001 - degrade, do not lose the run
                log.exception("the reasoner could not be reached")
                if self._fall_back(tools, result, f"the reasoner could not be reached: {exc}"):
                    continue
                result.stopped_because = STOP_UNUSABLE
                break
            transcript.turns.append(Turn("assistant", text))
            transcript.usage = dict(usage or {})
            transcript.latency_s = time.time() - t0

            try:
                call = P.parse_call(text)
            except P.InvalidReply as exc:
                invalid += 1
                transcript.retries = invalid
                if invalid <= self.max_retries:
                    log.warning("unusable reply (%s); asking again", exc)
                    transcript.turns.append(Turn("user", f"That reply cannot be used: {exc}"))
                    continue
                if self._fall_back(tools, result, f"unusable replies: {exc}"):
                    # The conversation continues, and it is *told* what was measured on its
                    # behalf. Dropping it here would leave a record no turn of this transcript
                    # accounts for.
                    invalid = 0
                    transcript.turns.append(Turn("user", self._fallback_note(result)))
                    continue
                result.stopped_because = STOP_UNUSABLE
                break

            # A well-formed call that is understood and refused is not a format failure, so it
            # does not count toward the fallback. The refusal is the answer.
            invalid = 0
            log.info("turn %d: %s %s", result.turns, call["tool"], call["note"])
            try:
                out = tools.dispatch(call["tool"], call["args"])
            except P.ToolError as exc:
                transcript.turns.append(Turn("user", f"{call['tool']} refused: {exc}"))
                continue
            if call["tool"] == "submit" and out == SUBMITTED:
                result.submitted = dict(tools.submitted)
                result.stopped_because = STOP_SUBMITTED
                break
            transcript.turns.append(Turn("user", out))
        else:
            result.stopped_because = STOP_TURNS

        result.verdict = self._verdict(tools, result)
        if teardown_at_end and not (result.submitted or {}).get("leave_deployed"):
            live = self.agent.deployer.live_tenant_releases()
            if live:
                log.info("tearing down %s", live)
                self.agent.deployer.destroy(live)
        log.info("finished after %d turn(s) and %d measurement(s): %s -> %s",
                 result.turns, tools.spent, result.stopped_because, result.verdict)
        return result

    # --- the three things the loop owns ---------------------------------------

    def _fall_back(self, tools: Tools, result: PackerResult, why: str) -> bool:
        return recorded_fallback(self.fleet, tools, result, why, top_k=self.top_k,
                                 solver_seconds=self.solver_seconds)

    def _fallback_note(self, result: PackerResult) -> str:
        return fallback_note(result) + " Reply with a single JSON tool call."

    def _verdict(self, tools: Tools, result: PackerResult) -> str | None:
        return derive_verdict(tools, result)


# --- the two things every driver owns, wherever the conversation comes from ----

def recorded_fallback(fleet, tools: Tools, result: PackerResult, why: str, *, top_k: int,
                      solver_seconds: float) -> bool:
    """impl §11.2's fallback: measure the solver's top-ranked candidate, and record that.

    False when there is nothing to fall back to -- no budget, or the solver offers nothing -- in
    which case the run ends rather than pretending. The expectations are all `unknown`, because
    nothing formed a belief: `TopRanked` takes what it was handed and has no opinion, and writing
    `hold` here would be inventing one on the reasoner's behalf.

    Module-level rather than a method because `sdkpacker.py` needs exactly this and there must not
    be a second implementation. A run that fell back has to be marked as such in the store whether
    the reasoner was driven over `converse` or by the SDK, and a fallback that differed between the
    two runtimes would make `policy_kind` mean two things.
    """
    if tools.remaining <= 0:
        return False
    proposal = TopRanked(top_k=top_k, max_iterations=None).next(
        _FallbackContext(fleet, tools, solver_seconds))
    if proposal is None:
        return False
    log.warning("falling back to the solver's top-ranked candidate: %s", why)
    args = {
        "choice": {"assignment": {n: list(g) for n, g
                                  in proposal.candidate.placement.assignment.items()}},
        "expectation": {t.name: "unknown" for t in fleet.tenants},
        "rationale": (f"NOT CHOSEN BY THE REASONER. {why}. Fell back to the solver's "
                      "top-ranked candidate, which is a packing choice and not a prediction."),
        "label": f"fb{len(result.attempts):02d}",
    }
    before = len(tools.attempts)
    # The marker goes in before the record is written, not patched on afterwards: the store is
    # where a fallback has to be visible, since that is what a later run reads.
    tools.tool_measure(args, provenance_extra={
        "policy_kind": PACKER_FALLBACK, "reasoner_failure": why,
        "fallback_policy": "TopRanked"})
    if len(tools.attempts) > before:
        # Surfaced on the result as well as in the record. A run in which nothing chose a
        # placement must not be something a reader has to go looking for.
        result.fallbacks.append(tools.attempts[-1].run_id)
    return True


def fallback_note(result: PackerResult) -> str:
    """What the reasoner is told was measured on its behalf. Never dropped silently.

    The transport-specific instruction ("reply with a JSON tool call") is appended by the driver:
    under the SDK there is no such envelope, and telling the model to produce one would be telling
    it to do the one thing that runtime cannot accept.
    """
    return ("Your last replies could not be used, so the loop measured the solver's top-ranked "
            f"candidate on your behalf: {result.fallbacks[-1]}. It was recorded as a fallback, not "
            "as your choice. `read_run` it if you want to see what it measured.")


def derive_verdict(tools: Tools, result: PackerResult) -> str | None:
    """What the run concluded, derived here and not taken from the reply.

    `met` only ever comes from a record. Everything else is the reasoner's reading, which is why
    its own verdict is kept verbatim in `submitted` beside this one: where the two differ that
    difference is a finding, and collapsing them would hide it.
    """
    met = tools.met_run()
    if met is not None:
        return P.DERIVED_VERDICT
    if result.submitted:
        return result.submitted["verdict"]
    if tools.remaining <= 0:
        return "budget"
    return "unresolvable"


class _FallbackContext:
    """The bare minimum `TopRanked` reads. Not a `Context`: it has no reasoner-facing state.

    Deliberately narrow. `TopRanked` is the fallback, so what it can see should be the solver and
    nothing that a broken reasoner turn might have left behind.
    """

    def __init__(self, fleet, tools: Tools, solver_seconds: float):
        self.fleet = fleet
        self.iteration = len(tools.attempts)
        self.history = list(tools.attempts)
        self.calibration = None
        self._tools = tools
        self._seconds = solver_seconds

    @property
    def tried(self) -> set:
        return {a.key for a in self.history}

    def solve(self, cap, avoid=()):
        return self._tools._solver().solve(cap=cap, max_seconds=self._seconds, avoid=avoid)


def expectation_report(result: PackerResult, targets: dict) -> list:
    """Belief against outcome for the whole run. Derived at read time, from stored fields.

    The expectations are read back out of each record's provenance rather than from the live
    `Tools` object, so this works on a run somebody else did -- and so there is one source for
    them, which is the store.

    The comparison itself is not written into any record, for the same reason a target verdict is
    not: the expectation is what the reasoner said, the metrics are what happened, and whether one
    matches the other is a function of the targets in force *now*. Storing it would freeze it
    against the targets of the day and quietly contradict `retrieval` the first time one changed.
    """
    out = []
    for attempt in result.attempts:
        if not attempt.record:
            continue
        exp = (attempt.record.get("provenance") or {}).get("reasoner_expectation")
        if not exp:
            continue
        out.append(f"{attempt.run_id}:")
        out += expectation_lines(exp, targets_met(Observation.parse(attempt.record), targets))
    return out
