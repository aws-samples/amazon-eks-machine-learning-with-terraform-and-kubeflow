"""The three places the harness asks CLM-8B a question.

  route     before the run: how hard and how risky is the task? Picks the first tier.
  gate      before each call to a risky tool: did the user ask for this action, and do its
            arguments match the tool results so far?
  escalate  after a tier's final answer: does it answer the request? If not, try the next tier.

Each hook is one /v1/systemone call. The thresholds come from the tiers file. They are
hand-set starting points, not calibrated values; calibrate them on your own traffic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .clm import CLM

# Every hook asks a choice between two described options. Asked as a 0..4 difficulty
# score or as a yes/no statement, CLM-8B gave nearly every task the same answer; a choice
# between two options separated them.
ROUTE_QUESTIONS = {
    "hard": {"type": "choice", "instructions": "How hard is this request?",
             "criteria": {"easy": "A simple request any assistant can answer",
                          "hard": "A difficult request that needs a strong model"}},
    "high_stakes": {"type": "choice", "instructions": "How risky is this request?",
                    "criteria": {"low": "Low risk: casual, informational or creative",
                                 "high": "High risk: production systems, money, health or law"}},
}
GATE_QUESTION = {"type": "choice", "instructions": "Should the assistant carry out the proposed action?",
                 "criteria": {"allow": "Yes: the user asked for exactly this action",
                              "block": "No: the user did not ask for this action, or it goes beyond what they asked"}}
# Intent alone misses a requested action with a wrong argument, such as an email to an address
# the order lookup did not return; the arguments question catches those.
ARGS_QUESTION = {"type": "choice", "instructions": "Do the proposed action's arguments match the tool results?",
                 "criteria": {"match": "Yes, every argument matches the tool results",
                              "mismatch": "No, an argument differs from the tool results"}}
ESCALATE_QUESTION = {"type": "choice", "instructions": "Does the assistant's final reply answer the user's request?",
                     "criteria": {"answered": "Yes, fully and correctly",
                                  "not_answered": "No: wrong, partial or refused"}}
MAX_CHARS = 6000   # stay inside CLM-8B's 2048-token state


@dataclass
class Decision:
    """One hook's verdict, with the CLM answers it was based on."""
    value: int | bool                 # route: the tier index; gate and escalate: True to proceed
    reasons: list[str] = field(default_factory=list)
    answers: dict = field(default_factory=dict)
    clm_ms: float = 0.0


def route(clm: CLM, task: str, policy: dict, n_tiers: int) -> Decision:
    """Pick the first tier from p(hard) and p(high stakes).

    policy["hard_limits"][i] is the highest p(hard) tier i takes; harder tasks go further
    up, and anything past the last limit goes to the last tier.
    """
    answers, ms = clm.ask(f"User: {task}"[:MAX_CHARS], ROUTE_QUESTIONS)
    hard = answers["hard"]["probabilities"]["hard"]
    stakes = answers["high_stakes"]["probabilities"]["high"]
    limits = policy["hard_limits"]
    tier = next((i for i, limit in enumerate(limits) if hard <= limit), len(limits))
    tier = min(tier, n_tiers - 1)
    reasons = [f"p(hard) {hard:.2f} -> tier {tier}"]
    floor = policy.get("high_stakes_min_tier")
    if floor is not None and stakes >= policy["high_stakes_floor"] and tier < floor:
        tier = min(floor, n_tiers - 1)
        reasons.append(f"p(high stakes) {stakes:.2f} >= {policy['high_stakes_floor']} -> at least tier {tier}")
    return Decision(tier, reasons, {"hard": hard, "high_stakes": stakes}, ms)


def _fit(state: str) -> str:
    """Keep the start (the task) and the end (the latest step) of a state that is too long."""
    if len(state) <= MAX_CHARS:
        return state
    return state[:MAX_CHARS // 3] + "\n...\n" + state[-2 * MAX_CHARS // 3:]


def gate(clm: CLM, task: str, transcript: str, name: str, arguments: dict | None, policy: dict) -> Decision:
    """Allow a risky tool call only if the user asked for it and its arguments match the tool results.

    The state includes the tool results so far, so CLM can compare the action with them,
    for example the email address an order lookup returned.
    """
    state = f"User: {task}\n\n{transcript}\n\nProposed action: {name}({json.dumps(arguments)})"
    answers, ms = clm.ask(_fit(state), {"allow": GATE_QUESTION, "args": ARGS_QUESTION})
    allow = answers["allow"]["probabilities"]["allow"]
    match = answers["args"]["probabilities"]["match"]
    checks = [("p(allow)", allow, policy["intent_below"]), ("p(args match)", match, policy["args_below"])]
    ok = all(p >= limit for _, p, limit in checks)
    reasons = [f"{label} {p:.2f} {'>=' if p >= limit else '<'} {limit}" for label, p, limit in checks]
    return Decision(ok, reasons, {"allow": allow, "args_match": match}, ms)


def escalate(clm: CLM, task: str, transcript: str, answer: str, threshold: float) -> Decision:
    """value is True when the answer should be escalated to the next tier."""
    state = f"User: {task}\n\n{transcript}\n\nAssistant (final reply): {answer}"
    answers, ms = clm.ask(_fit(state), {"answered": ESCALATE_QUESTION})
    p = answers["answered"]["probabilities"]["answered"]
    up = p < threshold
    return Decision(up, [f"p(answered) {p:.2f} {'<' if up else '>='} {threshold}"], {"answered": p}, ms)
