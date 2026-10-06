"""The agent loop, with the CLM hooks around it.

    route -> run the agent on that tier -> escalate? -> run it again on the next tier ...

The agent itself is the plain tool-calling loop: call the model, run the tools it asks
for (risky ones through the gate), send the results back, and stop when it answers
without a tool call or reaches max_steps. An escalated task starts again from the user's
message on the next tier, so each tier sees only its own conversation.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

import yaml

from . import hooks
from .backends import make_backend
from .clm import CLM
from .tools import TOOLS, run_tool

SYSTEM = ("You are a helpful general assistant. Answer any question, including ones no tool covers, "
          "from your own knowledge. You also have tools for an online store's orders and customer email; "
          "use them only when the request needs them. "
          "If a tool result says an action was blocked, do not retry it; tell the user what you did not do and why. "
          "Answer concisely.")
_VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def load_tiers(path: str) -> dict:
    """Read a tiers file. ${VAR} and ${VAR:-default} take values from the environment."""
    text = _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), open(path).read())
    cfg = yaml.safe_load(text)
    cfg["name"] = os.path.splitext(os.path.basename(path))[0]
    return cfg


@dataclass
class Attempt:
    tier: int
    tier_name: str
    model: str
    answer: str = ""
    finished: bool = False        # True when the model answered without a tool call
    error: str = ""
    steps: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    usage: list[dict] = field(default_factory=list)
    wall_ms: float = 0.0


@dataclass
class Trace:
    task: str
    tiers: str
    route: dict = field(default_factory=dict)
    attempts: list[Attempt] = field(default_factory=list)
    escalations: list[dict] = field(default_factory=list)
    answer: str = ""
    final_tier: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _transcript(attempt: Attempt) -> str:
    lines = []
    for c in attempt.tool_calls:
        lines.append(f"Tool call: {c['name']}({json.dumps(c['arguments'])}) -> {c['result']}")
    return "\n".join(lines)


def run_agent(tier_index: int, tier: dict, task: str, clm: CLM, cfg: dict) -> Attempt:
    backend = make_backend(tier)
    attempt = Attempt(tier_index, tier["name"], tier["model"])
    specs = [t.spec() for t in TOOLS.values()]
    messages = [{"role": "user", "content": task}]
    t0 = time.perf_counter()
    try:
        for attempt.steps in range(1, cfg.get("max_steps", 6) + 1):
            reply = backend.chat(SYSTEM, messages, specs, tier.get("max_tokens", 4096))
            attempt.usage.append(reply.usage)
            messages.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
            if not reply.tool_calls:
                attempt.answer, attempt.finished = reply.text.strip(), True
                break
            for call in reply.tool_calls:
                record = {"name": call.name, "arguments": call.arguments}
                tool = TOOLS.get(call.name)
                if tool is not None and tool.risky:
                    g = hooks.gate(clm, task, _transcript(attempt), call.name, call.arguments, cfg["gate"])
                    record["gate"] = {"allowed": g.value, "reasons": g.reasons, "clm_ms": g.clm_ms}
                    if not g.value:
                        result = json.dumps({"error": "blocked: this action needs the user's confirmation"})
                        record["result"] = result
                        attempt.tool_calls.append(record)
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                        continue
                result = run_tool(call.name, call.arguments)
                record["result"] = result
                attempt.tool_calls.append(record)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    except Exception as e:   # a backend failure is a reason to escalate, not to stop the harness
        attempt.error = f"{type(e).__name__}: {e}"[:300]
    attempt.wall_ms = (time.perf_counter() - t0) * 1000
    return attempt


def run_task(task: str, cfg: dict, clm: CLM, start_tier: str | None = None) -> Trace:
    """Route the task, run it, and escalate as needed. start_tier skips the route hook."""
    tiers = cfg["tiers"]
    trace = Trace(task, cfg["name"])
    if start_tier is None:
        r = hooks.route(clm, task, cfg["route"], len(tiers))
        trace.route = {"tier": tiers[r.value]["name"], "reasons": r.reasons, "answers": r.answers, "clm_ms": r.clm_ms}
        i = r.value
    else:
        i = [t["name"] for t in tiers].index(start_tier)
        trace.route = {"tier": start_tier, "reasons": ["set by the caller"]}
    while True:
        attempt = run_agent(i, tiers[i], task, clm, cfg)
        trace.attempts.append(attempt)
        if i == len(tiers) - 1:
            break
        if attempt.error:
            why = {"reasons": [f"error: {attempt.error}"]}
        elif not attempt.finished:
            why = {"reasons": [f"no final answer after {attempt.steps} steps"]}
        else:
            e = hooks.escalate(clm, task, _transcript(attempt), attempt.answer, cfg["escalate_below"])
            if not e.value:
                break
            why = {"reasons": e.reasons, "answers": e.answers, "clm_ms": e.clm_ms}
        trace.escalations.append({"from": tiers[i]["name"], "to": tiers[i + 1]["name"], **why})
        i += 1
    trace.answer, trace.final_tier = attempt.answer, attempt.tier_name
    return trace
