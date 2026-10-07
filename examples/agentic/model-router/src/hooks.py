"""The three places the harness asks CLM-8B a question.

  signals   before the run: the questions in the tiers file, for example how hard the task
            is. policy.py picks the first tier from the answers.
  gate      before each call to a risky tool: did the user ask for this action, and do its
            arguments match the tool results so far?
  escalate  after a tier's final answer: does it answer the request? If not, try the next tier.

Each hook is one /v1/systemone call. The limits come from the tiers file. They are
starting points; eval.py measures them on a task suite and --fit suggests new ones.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .clm import CLM

# Each question is a choice between described options. Asked as a 0..4 score or as a yes/no
# statement, CLM-8B gave nearly every task the same answer; a choice separated them.
# The route questions live in the tiers file (signals), so you can change them there.
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
    value: dict | bool                # signals: {option: p}; gate: True to run; escalate: True to move up
    reasons: list[str] = field(default_factory=list)
    answers: dict = field(default_factory=dict)
    clm_ms: float = 0.0


def signals(clm: CLM, task: str, questions: dict) -> Decision:
    """Ask the tiers file's questions about a task. value is {option: probability} over all of them."""
    answers, ms = clm.ask(_fit(f"User: {task}"), questions)
    probs = {}
    for name in questions:
        probs.update(answers[name]["probabilities"])
    return Decision(probs, [f"p({o}) {p:.2f}" for o, p in probs.items()], answers, ms)


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
