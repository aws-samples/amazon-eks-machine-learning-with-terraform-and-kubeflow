"""What the agent is told, and what it may say back (impl §11.2).

Pure. No network, no cluster, no clock: this module maps state to text and text to a validated
tool call. Everything that can be wrong about an LLM-driven search that is *not* the LLM lives
here, and all of it is testable offline for free.

This replaces `brief.py`, and the difference is the shape of the conversation rather than the
wording. The old brief was one shot: the whole store was rendered into a single prompt and the
reply had to be a decision. impl §11 asks for something else -- the reasoner *is* the loop, with
tools it calls as often as it likes -- so what it is handed at the start is only the problem, and
the evidence arrives when it asks for it, run by run, through `recall` and `read_run`. The
practical reason that matters: the store grows without bound, and a prompt that renders all of it
eventually cannot be sent at all. Handing over a truncated store would be impl §13's crude
similarity rule -- a cut by recency or count -- made silently.

Three choices carried over from the brief, because the reasons still hold:

**Structure, not summaries of structure.** Every placement is printed as the GPU set each tenant
holds. Printing "co-residency degree 1" and stopping is what the deleted interference model did,
and the store holds the case that refutes it: `r2`/`t2` sharing both their GPUs measured 5.76 ms
and sharing one of two measured 10.88 ms, and *every* summary of those two placements is
identical -- same degree, same co-resident pair set. A summary can only reproduce its own blind
spot, so the reasoner gets the assignment and has to do the seeing.

**Nothing here predicts a latency, because nothing in the architecture does (impl §9, §10).** A
candidate carries what the solver knows: GPUs used, KV cache per tenant, and the floor that
tenant's own target requires. A floor met is not a target met, and the text says so rather than
letting a satisfied constraint read as a forecast.

**The prompt never states the conclusion the reasoner is supposed to reach.** It does not group
records by "symmetric" and "asymmetric", does not print per-shard neighbour counts, does not point
at a residual. Naming those would turn a test of reasoning into a test of reading, and would
hard-code today's hypothesis into the prompt that has to find tomorrow's.

**A reply is invalid until proven otherwise, and an invalid reply is fed back rather than repaired.**
`parse_call` raises `InvalidReply` with text written to be read *by the model* on its next turn. A
loop that quietly fixed up a malformed tool call would make the search's behaviour depend on a
repair rule nobody reviewed.

**One conversation lives here.** A second one used to: the onboarding prompt (`ONBOARD_SYSTEM`,
`onboard_problem`, `ONBOARD_TOOL_SCHEMAS`) asked a reasoner which *experiment* settled a model's
memory law, and it is gone with the mode (impl §14.4). The prohibition it carried is worth keeping in
mind here, because it is why `tools.py` looks the way it does: the search chooses a placement to
measure and may not assert a latency, and that is enforced in the tool implementations rather than in
this text, for the reason impl §11.2 gives.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

# Goes into a run id, which is also a Job name and a record filename, so DNS-1123.
LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,18}[a-z0-9])?$")

TOOLS = ("solve", "recall", "read_run", "measure", "cluster_status", "submit")

# impl §11.2. `hold`/`miss` are claims about a target; `unknown` is how exploration is declared.
EXPECTATIONS = ("hold", "miss", "unknown")

# impl §11.2. `met` is absent on purpose: the loop derives it from a record, and the reasoner is
# not permitted to assert it. `submit` rejects it with that explanation.
VERDICTS = ("infeasible", "budget", "unresolvable")
DERIVED_VERDICT = "met"

_PREAMBLE = """\
You are searching for the best way to co-locate several inference servers (tenants) on the GPUs of
one node under NVIDIA MPS. You choose what to measure on real hardware, you say what the evidence
does and does not settle, and you decide when to stop.

Each measurement costs roughly six minutes of exclusive time on an 8-GPU node. The budget is a
small number of measurements. Spend them on placements that will change what is known.

Nothing in this system predicts a latency. The solver checks constraints -- memory arithmetic,
shard placement, and each tenant's KV cache floor -- and orders what survives by how few GPUs it
uses. That order is a packing order, not a forecast: candidate 1 is the densest feasible placement
and may well be the slowest. Which placement is fastest is what a measurement answers and nothing
else does.

A floor is a requirement, not a promise. It is Little's law run backwards from a tenant's own
stated ITL ceiling at its own stated rate: this much KV cache must be resident for that ceiling to
be reachable. A placement that clears every floor can still miss every target, because contention
for SMs is not memory and the solver does not model it. You may cap a tenant's share of its cards'
SMs on a measurement (`sm_caps`). A cap is a ceiling, not a reservation: it takes SMs from that
tenant and leaves them to the others on its cards. Uncapped, MPS gives idle SMs to whichever tenant
can use them, so a cap is worth a measurement only when the records give you a reason for one.

"""

# How a turn is delivered, and the only part of the prompt that differs between the two runtimes.
# Everything above and below is shared, so the substance of what the agent is told cannot drift
# between them -- only the transport can.
_JSON_ENVELOPE = """\
You have six tools. Call them one at a time. Every turn you make is a single JSON object and
nothing else -- no prose outside it, no markdown fence:

  {"tool": "<name>", "args": {...}, "note": "one line: why you are asking"}

{rules}
"""

# Said once and used by every JSON-over-converse loop. The failure it is written against was seen
# live (2026-10-09): `args` closed and the call itself not, three replies running. So the example
# has `args` nested, and the rule says where each of the two closing braces goes.
JSON_RULES = """\
Your whole reply is that one object: the first character is { and the last is }, with no text,
fence or second object around it. `args` is an object nested inside the call: close it with }
before "note", and close the call itself with one more } at the very end. For example:
  {"tool": "recall", "args": {"tenants": ["a", "b"]}, "note": "what a and b measured"}
Keep each string to one plain line."""

_JSON_ENVELOPE = _JSON_ENVELOPE.replace("{rules}", JSON_RULES)

_SDK_ENVELOPE = """\
You have six tools and nothing else. Call them one at a time, as tool calls. All six are already
loaded with their schemas: there is nothing to look up or discover before calling one, and no tool
for doing so. There is no other tool available to you either -- no shell, no file access, no way to
reach the cluster except `measure`. Attempts to use anything else are refused, and the refusal
costs you a turn.

"""

_CATALOGUE = """\
  solve            the feasible placements for this fleet on this node.
                   args: {"cap": <int, default 4>,
                          "avoid": [["<tenant>","<tenant>"], ...]   (optional; forbid these pairs
                                                                     from sharing a GPU),
                          "assignment": {"<tenant>": [<gpu>, ...]}  (optional; check just this one)}
                   Cheap, deterministic, and repeatable. Use it as a scratchpad.

  recall           prior runs bearing on a placement or on a tenant.
                   args: {"assignment": {...}} or {"candidate": <int>} or {"tenants": ["r2"]},
                         with "sm_caps": {...} beside a placement to ask about it capped
                   Returns the runs themselves, never an average, and says NO EVIDENCE when there
                   is none. Two runs that disagree are both shown to you: say they disagree rather
                   than picking one.

  read_run         one record in full: raw metrics, margins against the targets in force now, and
                   the signals that qualify them.  args: {"run_id": "<id>"}

  measure          deploy the placement, hold at the readiness gate, drive the load, record the
                   result, tear down. THE ONLY TOOL THAT COSTS ANYTHING.
                   args: {"choice": {"candidate": <int>} or {"assignment": {...}},
                          "expectation": {"<tenant>": "hold" | "miss" | "unknown", ...},
                          "rationale": "which run ids you are relying on, and where you had none",
                          "label": "short-kebab-case, <=20 chars",
                          "avoid": [["a","b"], ...]  (optional; applies to later solve calls in
                                                      this run only, and is never stored),
                          "sm_caps": {"<tenant>": <1-99>}  (optional; the percent of each of its
                                     cards' SMs that tenant may use. Only on a tenant sharing a
                                     card. A capped placement is a different placement: it is
                                     recorded with its caps and compared with them)}

  cluster_status   what is running on the node right now. Operational: use it if a run was
                   interrupted between deploying and tearing down.  args: {}

  submit           end the run.
                   args: {"verdict": "met" | "infeasible" | "budget" | "unresolvable",
                          "rationale": "what the evidence settles, citing run ids",
                          "choice": {"candidate": <int>} or {"assignment": {...}} (optional: the
                                    configuration you are recommending),
                          "deploy": false}

"""

_RULES = """\
`expectation` is required for every tenant on every measure call, and it is what makes your
judgement falsifiable. It is recorded before the measurement and compared with the result
afterwards, which separates a wrong belief from a wrong choice: a placement that missed as you
expected is information, and one that missed unexpectedly is a different problem. It constrains
nothing. Say "unknown" when you are exploring -- that is what exploring means here, and there is
no schedule above you deciding when to do it. A tenant that stated no target can only be
"unknown": there is nothing for it to hold or miss.

You cannot change the fleet, the targets, the offered rates or the request shapes. There is no tool
that would let you, and they are what the answer has to be true of.

You cannot certify. "met" as a verdict is derived by the loop from an actual record; submitting it
without one is refused, with the reason.

Stopping is a real answer, not a failure. So is "no evidence either way".
"""

# What `packer.py` sends: the tool call is JSON inside a text reply, parsed by `parse_call`.
SYSTEM = _PREAMBLE + _JSON_ENVELOPE + _CATALOGUE + _RULES

# What `sdkpacker.py` sends: the tool call is a real `tool_use` block, parsed by the SDK. The
# catalogue is kept verbatim rather than replaced by the schemas in `TOOL_SCHEMAS`, because what a
# tool *means* is not in its schema -- that `measure` is the only tool that costs anything, that
# `solve` is a scratchpad, that `recall` says NO EVIDENCE rather than inventing one. A schema can
# only carry the argument shape, and the argument shape is the part the model gets right anyway.
SYSTEM_SDK = _PREAMBLE + _SDK_ENVELOPE + _CATALOGUE + _RULES


# --- the same six tools, as schemas ------------------------------------------

# **Deliberately permissive.** These exist so the model learns the property names, not to validate
# anything: every real rule about an argument lives in `check_assignment`, `check_expectation`,
# `check_label`, `check_avoid`, `check_rationale` and the tool bodies, which raise `ToolError` with
# a message written to be read by the model. A schema that rejected a call first would replace that
# refusal with a generic transport-level validation error and lose the sentence explaining what to
# send instead -- and it would put a second, unreviewed copy of the invariants somewhere the tests
# in `tools.py` do not cover. So the types are loose, `required` names only what the tool genuinely
# cannot proceed without, and `additionalProperties` is left open.
_ASSIGNMENT = {"type": "object", "description": "tenant name -> list of GPU ordinals",
               "additionalProperties": {"type": "array", "items": {"type": "integer"}}}
_AVOID = {"type": "array", "description": "pairs forbidden from sharing a GPU, e.g. [[\"r2\",\"t2\"]]",
          "items": {"type": "array", "items": {"type": "string"}}}
_CHOICE = {"type": "object",
           "description": "{\"candidate\": <n>} from the last solve, or {\"assignment\": {...}}",
           "properties": {"candidate": {"type": "integer"},
                          "assignment": _ASSIGNMENT}}

_SM_CAPS = {"type": "object",
            "description": "tenant -> percent (1-99) of its cards' SMs it may use; optional",
            "additionalProperties": {"type": "integer"}}

TOOL_SCHEMAS = {
    "solve": {"type": "object", "properties": {
        "cap": {"type": "integer", "description": "how many candidates to return (default 4)"},
        "avoid": _AVOID,
        "assignment": dict(_ASSIGNMENT,
                           description="check just this one placement instead of listing the set")}},
    "recall": {"type": "object", "properties": {
        "assignment": _ASSIGNMENT,
        "candidate": {"type": "integer", "description": "a candidate of the last solve"},
        "tenants": {"type": "array", "items": {"type": "string"},
                    "description": "runs bearing on these tenants"},
        "sm_caps": _SM_CAPS}},
    "read_run": {"type": "object", "properties": {
        "run_id": {"type": "string", "description": "as `recall` printed it"}},
        "required": ["run_id"]},
    "measure": {"type": "object", "properties": {
        "choice": _CHOICE,
        "expectation": {"type": "object", "description":
                        "every tenant -> \"hold\" | \"miss\" | \"unknown\"",
                        "additionalProperties": {"type": "string"}},
        "rationale": {"type": "string", "description":
                      "which run ids you rely on, and where you had none"},
        "label": {"type": "string", "description": "short-kebab-case, <=20 chars"},
        "avoid": _AVOID,
        "sm_caps": _SM_CAPS},
        # Not `label`: it defaults to the measurement index, and a tool that refused to run for
        # want of a name would be enforcing tidiness rather than an invariant.
        "required": ["choice", "expectation", "rationale"]},
    "cluster_status": {"type": "object", "properties": {}},
    "submit": {"type": "object", "properties": {
        "verdict": {"type": "string", "description":
                    f"one of {DERIVED_VERDICT}, {', '.join(VERDICTS)}"},
        "rationale": {"type": "string", "description": "what the evidence settles, citing run ids"},
        "choice": dict(_CHOICE, description="optional: the configuration you recommend"),
        "deploy": {"type": "boolean", "description": "leave it running after the run ends"}},
        "required": ["verdict", "rationale"]},
}

# One line each, for the tool-registration surface. The prompt's catalogue is what actually teaches
# these; this is what a tool list shows, so it says the one thing a caller must not miss.
TOOL_SUMMARIES = {
    "solve": "The feasible placements for this fleet. Cheap, deterministic, a scratchpad.",
    "recall": "Prior runs bearing on a placement or a tenant. Runs, never an average.",
    "read_run": "One record in full, with margins against the targets in force now.",
    "measure": "Deploy, gate, drive the load, record, tear down. THE ONLY TOOL THAT COSTS "
               "ANYTHING, and the only one behind the budget.",
    "cluster_status": "What is installed on the node right now. Read-only.",
    "submit": "End the run with a verdict. Refused if the records do not support it.",
}


class InvalidReply(Exception):
    """A reply that cannot be acted on. The message is written to be shown to the model."""


class ToolError(Exception):
    """A well-formed call that must not be executed. Also written to be shown to the model.

    Separate from `InvalidReply` because they mean different things about the reasoner: a malformed
    reply may be a format failure that a retry fixes, and a refused call is a judgement the loop
    made about a call it understood. Only the first counts toward the retry budget that ends in the
    recorded fallback of impl §11.2.
    """


# --- the problem statement ---------------------------------------------------

# The last line of the problem statement, and the only transport-specific line in it. Passed in by
# the driver for exactly the reason `_JSON_ENVELOPE` is absent from `SYSTEM_SDK`: under the SDK a
# tool call is a structured block, so an instruction to reply with JSON text would be telling the
# model to do the one thing that runtime cannot accept -- and the model would obey it.
JSON_CLOSING = "Reply with a single JSON tool call, as described in your instructions."
SDK_CLOSING = "Call one of your six tools. Nothing else is available to you."


def problem(*, fleet, budget: int, store_runs: int, key_hash: str,
            calibration=None, gpu_names=None, closing: str = JSON_CLOSING) -> str:
    """The fleet, the node, the budget, and how much the store holds. Deterministic.

    Deliberately *not* the store's contents. The reasoner asks for those with `recall`, which is
    what lets a run against a store of five records and a run against a store of five hundred use
    the same prompt -- and what stops this function from having to choose which records to leave
    out (impl §13).
    """
    out = ["== THE PROBLEM =="]
    out += _fleet_lines(fleet, gpu_names or _ordinals(fleet))
    out.append("")
    out.append("== WHAT IS ALREADY MEASURED ==")
    out.append(f"  the store for validity key {key_hash} holds {store_runs} record(s) that may be "
               "cited as")
    out.append("  evidence about latency. `recall` is how you see them; nothing is rendered here,")
    out.append("  because which of them bear on a placement is your judgement and not this "
               "prompt's.")
    out.append("")
    out.append("== MEASURED RESOLUTION ==")
    out += _calibration_lines(calibration)
    out.append("")
    out.append("== BUDGET ==")
    out.append(f"  {budget} measurement(s). Nothing else you can call costs GPU time.")
    out.append("")
    out.append(closing)
    return "\n".join(ln.rstrip() for ln in "\n".join(out).split("\n"))


def _ordinals(fleet) -> dict:
    """UUID -> `gpu<ordinal>`. The agent knows the node's map; a bare record does not."""
    node = getattr(fleet, "node", None)
    if node is None or not getattr(node, "uuids", None):
        return {}
    return {u: f"gpu{g}" for g, u in zip(node.gpu_ids, node.uuids)}


def _fleet_lines(fleet, names) -> list:
    node = fleet.node
    # Named "gpu node under test" in the same words `read_run` uses, so a record whose load driver
    # ran somewhere else reconciles against this line instead of contradicting it.
    out = [f"  gpu node under test {node.node_name}",
           f"  GPUs {', '.join(f'gpu{g}' for g in node.gpu_ids)} "
           f"({node.device_total_gib:.2f} GiB each, as the driver reports it -- measured, not a "
           "nameplate)",
           "",
           f"  {'tenant':<8}{'model':<22}{'tp':>3}{'rate':>9}{'in':>7}{'out':>6}"
           f"  {'target'}"]
    for t in fleet.tenants:
        slo = "--" if t.slo is None else ", ".join(
            f"{n}={getattr(t.slo, n):g}" for n in t.slo.stated)
        out.append(f"  {t.name:<8}{t.model_id:<22}{t.tp:>3}{t.rate_hz:>7.1f}Hz"
                   f"{t.isl_tokens:>7}{t.osl_tokens:>6}  {slo}")
    out.append("")
    out.append("  Rate, request shape and target are each tenant's own. A tenant with no target")
    out.append("  can be measured but not certified: nothing invents one for it.")
    out.append("")
    out.append("  KV cache each tenant's own target requires, at its own rate:")
    for t in fleet.tenants:
        derived = t.kv_tokens_at_target()
        if derived is None:
            out.append(f"    {t.name:<8}no ITL ceiling stated, so no floor follows from a target")
        else:
            tokens, how = derived
            out.append(f"    {t.name:<8}{tokens:>7} tokens   ({how})")
        if t.min_kv_tokens:
            out.append(f"    {'':<8}{t.min_kv_tokens:>7} tokens   minimum viable pool, 2 x this "
                       f"tenant's max_model_len {t.max_model_len}; the larger of the two binds")
        out.append(f"    {'':<8}{t.max_num_seqs:>7} seqs     batch cap, derived from the slots the "
                   "same burst holds")
    out.append("")
    out.append("  A tighter ITL ceiling implies a *smaller* floor, not a larger one: a request")
    out.append("  that finishes sooner occupies the cache for less time. It is the rate that")
    out.append("  demands memory. Clearing a floor does not mean the target will be met.")
    out.append("")
    out.append("  Each tenant needs exactly `tp` GPUs, on distinct devices. Two tenants sharing a")
    out.append("  GPU contend for its SMs; tenants on different GPUs of the same node barely")
    out.append("  interact. Lower inter-token latency is better.")
    return out


def _calibration_lines(calibration) -> list:
    if calibration is None:
        # Not a zero. Without a floor nothing can be called indistinguishable, so the reasoner
        # must not be handed a number that would let it stop on a criterion nobody measured.
        return ["  UNCALIBRATED. No run-to-run spread has been measured, so there is no",
                "  threshold below which two measurements count as the same. Do not treat small",
                "  measured differences as real, and do not stop because they look small."]
    out = ["  run-to-run half-width of a repeated placement's ITL, by co-residency bucket:"]
    for bucket, hw in sorted(calibration.between_run_itl_halfwidth_ms.items()):
        out.append(f"    {bucket:<16}{hw:.4f} ms")
    out.append(f"  from {calibration.repeats} passes per placement. Differences smaller than "
               "this are not")
    out.append("  distinguishable by measurement.")
    return out


# --- rendering a feasible set (the `solve` tool's answer) --------------------

def candidate_lines(candidates, gpu_names=None, measured=()) -> list:
    """The feasible set, as constraints satisfied -- never as a predicted outcome.

    No latency appears here because none is available: what a candidate carries is the KV cache it
    gives each tenant and the floor that tenant's target requires. The ranking is stated as what it
    is, so an ordered list is not read as a ranked forecast.
    """
    measured = frozenset(measured)
    out = ["  Every placement below satisfies the memory arithmetic and every stated KV floor.",
           "  They are ordered by GPUs used, then by the tightest margin over a floor. That order",
           "  is a packing order and says nothing about which will measure fastest."]
    if not candidates:
        out.append("")
        out.append("  the solver returned nothing.")
        return out + completeness_lines(candidates)
    for i, c in enumerate(candidates, start=1):
        out.append(f"\n  candidate {i}   ({c.gpus_used} GPU(s) used)")
        if canonical(c.placement.assignment) in measured:
            out.append("    ALREADY MEASURED: choosing it repeats a placement, which measures "
                       "run-to-run spread and nothing else.")
        for name in sorted(c.placement.assignment):
            gpus = c.placement.assignment[name]
            floor = c.floor_tokens.get(name, 0)
            margin = c.margin.get(name)
            budget = (f"floor {floor:>7} ({margin:.2f}x)" if margin is not None
                      else "no floor stated")
            out.append(f"    {name:<8}gpus={','.join(f'gpu{g}' for g in sorted(gpus)):<14}"
                       f"envelope {c.placement.envelope_gib[name]:>6.2f} GiB   "
                       f"kv {c.placement.kv_tokens[name]:>7} tokens   {budget}")
        if c.min_margin is None:
            out.append("    -> no tenant stated a floor, so there is no margin to compare.")
        else:
            out.append(f"    -> tightest margin {c.min_margin:.2f}x, at {c.tightest_tenant}")
    return out + completeness_lines(candidates)


def completeness_lines(candidates) -> list:
    """Whether the list above is all of them. impl §9 requires this said, not implied.

    A reasoner told the set is complete may conclude the fleet is infeasible; one told it was
    truncated may not. Nothing here guesses: a plain list carries no such claim, and the honest
    thing to print for one is that it is unknown.
    """
    exhausted = getattr(candidates, "exhausted", None)
    if exhausted is None:
        return ["", "  Whether other feasible placements exist is not recorded for this list."]
    if exhausted:
        return ["", "  This is EVERY feasible placement: the solver proved there are no others.",
                "  If none of them will do, the fleet cannot be served on this node as specified."]
    return ["", f"  This list is TRUNCATED: {getattr(candidates, 'stopped_because', 'unknown')}.",
            "  Other feasible placements exist that are not shown. Do not conclude the fleet is",
            "  infeasible from this list, and name your own assignment if you want one of them."]


def canonical(assignment) -> tuple:
    """A hashable identity for an assignment. Same rule as `policy.canonical`."""
    return tuple(sorted((str(n), tuple(sorted(int(g) for g in gpus)))
                        for n, gpus in assignment.items()))


# --- parsing a turn ----------------------------------------------------------

def parse_call(text: str, tools=TOOLS) -> dict:
    """`{"tool": ..., "args": {...}, "note": ...}` from a reply. Raises `InvalidReply`.

    Only the envelope is checked here: that there is one JSON object, that it names a tool that
    exists, and that `args` is an object. What the arguments *mean* is checked by the tool that
    receives them, because that is where impl §11.2 puts the invariants -- in the implementations,
    not in the prompt, so they hold whichever runtime drives the loop.

    `tools` is which set exists, and it is a parameter rather than a constant read here: the message
    below is the only place the list of callable tools is written out for the model, so a caller with
    a different set gets a refusal that names *its* set. Nothing but the search passes one today.
    """
    obj = _json_object(text)
    tool = obj.get("tool")
    if not isinstance(tool, str) or tool.strip() not in tools:
        raise InvalidReply(
            f"'tool' must be one of {', '.join(tools)}; got {tool!r}. Reply with a single JSON "
            'object of the form {"tool": "<name>", "args": {...}, "note": "..."}.')
    args = obj.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise InvalidReply(f"'args' must be a JSON object (use {{}} if the tool takes none); got a "
                           f"{type(args).__name__}.")
    note = obj.get("note")
    return {"tool": tool.strip(), "args": args,
            "note": note if isinstance(note, str) else ""}


def _json_object(text: str) -> dict:
    """The reply's JSON object, tolerating a code fence but not tolerating prose instead."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s.rstrip())
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise InvalidReply("your reply contained no JSON object. Reply with a single JSON object "
                           "and nothing else.")
    try:
        obj = json.loads(s[start:end + 1])
    except json.JSONDecodeError as exc:
        missing = _unclosed(s[start:end + 1])
        if missing:
            # The one repair made, because it is the one with a single reading: every bracket that
            # was opened is closed, in order, at the end, and nothing else is changed. Anything
            # else -- a stray quote, a comma -- has more than one fix and goes back to the model.
            try:
                obj = json.loads(s[start:end + 1] + missing)
                log.info("closed a reply's JSON with %r, the closers it left off", missing)
            except json.JSONDecodeError:
                obj = None
            if obj is None:
                raise InvalidReply(
                    f"your JSON object is not closed: it needs {missing!r} at the end. Every {{ "
                    "and [ you open has to be closed -- `args` is nested inside the call, so it "
                    "needs its own } and the call one more. Reply with the whole call again.") from None
        else:
            raise InvalidReply(f"your reply is not valid JSON ({exc}). Reply with a single JSON "
                               "object and nothing else -- no trailing commas, no comments, and "
                               'every " inside a string escaped as \\".') from None
    if not isinstance(obj, dict):
        raise InvalidReply(f"the top level of your reply must be a JSON object, not a "
                           f"{type(obj).__name__}.")
    return obj


def _unclosed(s: str) -> str:
    """The closers `s` leaves off at its end, innermost first; "" when it closes everything, or
    when it closes something it never opened, which no suffix can repair."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return ""
    return "" if in_str else "".join(reversed(stack))


# --- checking the arguments the invariants live in ---------------------------

def check_assignment(assign, fleet) -> dict:
    """Every rule the deployer and the solver would otherwise discover the hard way.

    Structural only: it does not decide whether the placement fits in memory. `solve_pinned` and
    `Placement.verify()` do that, independently of each other, which is why a pinned assignment
    passes through all three.
    """
    if not isinstance(assign, dict):
        raise ToolError("'assignment' must be a JSON object mapping tenant name to a list of GPU "
                        f"ordinals; got a {type(assign).__name__}.")
    wanted = {t.name: t.tp for t in fleet.tenants}
    have = set(assign)
    if have != set(wanted):
        raise ToolError(
            f"'assignment' must name every tenant exactly once. This fleet has {sorted(wanted)}; "
            f"you sent {sorted(have)}"
            + (f" (missing {sorted(set(wanted) - have)})" if set(wanted) - have else "")
            + (f" (unknown {sorted(have - set(wanted))})" if have - set(wanted) else "") + ".")
    node_gpus = {int(g) for g in fleet.node.gpu_ids}
    out = {}
    for name, tp in sorted(wanted.items()):
        gpus = assign[name]
        if not isinstance(gpus, list) or not all(
                isinstance(g, int) and not isinstance(g, bool) for g in gpus):
            raise ToolError(f"{name}: GPUs must be a JSON list of integer ordinals; got {gpus!r}.")
        if len(set(gpus)) != len(gpus):
            raise ToolError(f"{name}: the same GPU appears twice ({gpus}). A tenant's tensor "
                            "parallel shards must be on distinct devices.")
        if len(gpus) != tp:
            raise ToolError(f"{name}: tensor parallel degree is {tp}, so it needs exactly {tp} "
                            f"GPU(s); you gave {len(gpus)} ({gpus}).")
        unknown = sorted(set(gpus) - node_gpus)
        if unknown:
            raise ToolError(f"{name}: this node has no GPU {unknown}. It has {sorted(node_gpus)}.")
        out[name] = tuple(sorted(gpus))
    return out


def check_expectation(exp, fleet) -> dict:
    """impl §11.2's required field, checked as required rather than defaulted.

    Two rules, and both are about what a claim can mean. Every tenant must appear: a placement is
    measured as a whole, and an expectation about three of four tenants cannot be compared with the
    result. And a tenant that stated no target may only be `unknown` -- `hold` and `miss` are both
    claims about a ceiling, so allowing either would let the run conclude something about a target
    nobody set, which is the invariant impl §11.2 states outright.
    """
    if not isinstance(exp, dict):
        raise ToolError(
            "'expectation' is required on every measure call and must be a JSON object mapping "
            f"every tenant to \"hold\", \"miss\" or \"unknown\"; got a {type(exp).__name__}. It is "
            "recorded before the measurement and compared with the result afterwards.")
    names = {t.name: t for t in fleet.tenants}
    missing = sorted(set(names) - set(exp))
    unknown_names = sorted(set(exp) - set(names))
    if missing or unknown_names:
        raise ToolError(
            "'expectation' must name every tenant exactly once. This fleet has "
            f"{sorted(names)}; you sent {sorted(exp)}"
            + (f" (missing {missing})" if missing else "")
            + (f" (unknown {unknown_names})" if unknown_names else "") + ".")
    out = {}
    for name, value in sorted(exp.items()):
        if not isinstance(value, str) or value.strip().lower() not in EXPECTATIONS:
            raise ToolError(f"expectation for {name} must be one of {', '.join(EXPECTATIONS)}; got "
                            f"{value!r}.")
        value = value.strip().lower()
        if names[name].slo is None and value != "unknown":
            raise ToolError(
                f"{name} states no latency target, so it cannot be expected to \"{value}\" one. "
                "Use \"unknown\" for it. Nothing invents a target for a tenant that did not state "
                "one, and no measurement can certify it.")
        out[name] = value
    return out


def check_label(label) -> str:
    if not isinstance(label, str) or not LABEL_RE.match(label):
        raise ToolError(
            f"'label' must be 1-20 characters of lowercase letters, digits and hyphens, starting "
            f"and ending with a letter or digit; got {label!r}. It becomes a Kubernetes Job name "
            "and a record filename.")
    return label


def check_avoid(avoid, fleet) -> tuple:
    """Pairs forbidden from sharing a GPU. Run-scoped, and never written to the store.

    The scope is the point (impl §11.2). A cut is a belief the reasoner formed from runs it read,
    and a mistaken belief must cost this run some iterations rather than being inherited by every
    later run as though it were a measured fact.
    """
    if avoid in (None, ()):
        return ()
    if not isinstance(avoid, list):
        raise ToolError("'avoid' must be a list of two-element lists of tenant names, e.g. "
                        f'[["r2","t2"]]; got a {type(avoid).__name__}.')
    names = {t.name for t in fleet.tenants}
    out = []
    for pair in avoid:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ToolError(f"each entry of 'avoid' must be a pair of tenant names; got {pair!r}.")
        a, b = str(pair[0]), str(pair[1])
        unknown = sorted({a, b} - names)
        if unknown:
            raise ToolError(f"'avoid' names {unknown}, which is not in this fleet ({sorted(names)}).")
        if a == b:
            raise ToolError(f"'avoid' pair [{a!r}, {b!r}] names one tenant twice, which forbids "
                            "nothing. A cut separates two different tenants.")
        out.append((a, b))
    return tuple(out)


def check_sm_caps(caps, fleet, assignment) -> dict:
    """impl §16's knob, checked. `{tenant: percent}`, empty for none.

    1 to 99: a cap of 100 is no cap, and allowing it would give one placement two identities. The
    tenant must share a card with another tenant in `assignment`, because the point of a ceiling is
    to leave SMs to a neighbour, and on a card of its own a cap only slows the tenant down -- a
    measurement that answers nothing the search asked.
    """
    if caps is None:
        return {}
    if not isinstance(caps, dict):
        raise ToolError("'sm_caps' must be a JSON object mapping tenant name to an integer percent "
                        f"from 1 to 99; got a {type(caps).__name__}.")
    names = {t.name for t in fleet.tenants}
    unknown = sorted(set(caps) - names)
    if unknown:
        raise ToolError(f"'sm_caps' names {unknown}, which is not in this fleet.")
    out = {}
    for name, cap in sorted(caps.items()):
        if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= 99:
            raise ToolError(f"{name}: an SM cap is an integer percent from 1 to 99; got {cap!r}. "
                            "Leave a tenant out of 'sm_caps' to leave it uncapped.")
        mine = set(assignment[name])
        if not any(mine & set(g) for n, g in assignment.items() if n != name):
            raise ToolError(f"{name} shares no GPU with another tenant in this placement, so a cap "
                            "would leave its SMs to nobody and only slow it down. Cap a tenant "
                            "that shares a card.")
        out[name] = cap
    return out


def check_rationale(rationale) -> str:
    """impl §11.2: the run ids relied on, and where there were none.

    Only non-emptiness is enforced. Requiring a run id to appear would make a run against an empty
    store impossible to justify, and "there is no evidence for this and I am exploring" is a
    rationale the design explicitly wants sayable.
    """
    if not isinstance(rationale, str) or not rationale.strip():
        raise ToolError("'rationale' must be a non-empty string saying which run ids you are "
                        "relying on, and where there were none. It is stored with the record.")
    return rationale.strip()
