"""`SdkPackerAgent`: the same search, driven by the Claude Agent SDK (impl §11, §11.4).

    system  = prompt.SYSTEM_SDK
    turn 1  = prompt.problem(fleet, budget, ...)
    repeat:
        the model emits a real `tool_use` block; the SDK routes it to `Tools`
        the tool's text goes back as a `tool_result`
    until `submit`, or the turn cap, or the model stops talking

This is `packer.py` with the transport replaced and nothing else. The six tools are the same six
`Tools` methods, the invariants are the same ones enforced inside them, and the fallback and the
derived verdict are imported from `packer.py` rather than reimplemented -- a run that fell back must
look the same in the store whichever driver produced it, or `policy_kind` means two things.

**What changes, and it is only this: the reply cannot be malformed.** `prompt.parse_call` existed
because a JSON tool call inside a text reply can be truncated, fenced, prosified or invented. The
SDK's tool calls are structured, so that entire failure class and its retry-then-fall-back ladder
are gone. What replaces them is narrower: the model can stop talking without submitting, and it can
call a tool that does not exist. The first falls back, once, and says so; the second is refused.

**The tool set is closed here, deliberately and by an allowlist.** This is the part that is not a
transport detail. The SDK's default posture is that the agent has the whole Claude Code toolset --
Bash, Write, Edit, subagents -- and `allowed_tools` does *not* change that: it is an auto-approval
list, not an existence list. A `can_use_tool` callback is worse than useless here, because an
`allowed_tools` entry auto-approves ahead of it and allow rules in a settings file can shadow it
invisibly; measured, it was never consulted at all while `Bash` ran. A `PreToolUse` hook is
consulted for every call, so `_gate` below denies by default and the six tools are the exception.

That is not hygiene. The measurement budget is the only thing standing between this agent and
arbitrary GPU spend, and it is enforced at the top of `tools.tool_measure`. A reachable shell makes
it decorative: the model could `helm install` a fleet directly, spend the node for an hour, and
write no record -- so nothing would stop a later run repeating it, which is the one failure the
store exists to prevent. `disallowed_tools` is set as well, but it is a blocklist and goes stale the
first time a new builtin ships; the hook is what actually holds.

**What is checked where.** Everything above the SDK boundary is in the self-tests, offline: the
schemas, every decision `_gate` makes, the transcript, the wrapper's translation of a `ToolError`
into text. The boundary itself -- the SDK spawning its CLI, authenticating, routing a `tool_use`
block to the right function -- needs a subprocess, credentials and a network, so it is not in the
suite. It has been exercised once against `FakeDeployer`/`FakeHarness`, which costs tokens and no
GPU, and that run is what settled two things stated above: that the six schemas load eagerly, so
`discovery_tools` needs no entry, and that a refused builtin comes back as something the model reads
and acts on rather than as an error it retries. What remains untested until the first live run is
only the half this module does not own: Helm, the readiness gate, and the harness Job.
"""

from __future__ import annotations

import logging
import time

from . import prompt as P
from .packer import PackerResult, derive_verdict, fallback_note, recorded_fallback
from .reason import DEFAULT_MODEL_ID, DEFAULT_REGION, Transcript, Turn
from .tools import SUBMITTED, Tools

log = logging.getLogger(__name__)

# The MCP server the six tools are registered under. It prefixes every tool name the model sees,
# and the gate's allowlist is written in those qualified names because that is what the hook is
# handed -- comparing bare names there would silently allow a builtin that shared one.
SERVER_NAME = "packer"

# Higher than `packer.DEFAULT_MAX_TURNS` for the same conversation, because the SDK counts turns
# its own way: a turn that is only text, and a turn spent discovering a tool's schema, both count.
# Measured on a two-tool-call probe, the SDK reported 4. It bounds the conversation, not the search
# -- `measure` is behind the budget, and everything else costs tokens.
DEFAULT_MAX_TURNS = 120

# Rounds of "ask, let it work, see whether it submitted". More than one only happens when the model
# stops without submitting and there is still budget to fall back on, so this caps how many times
# that can be forgiven before the run ends.
DEFAULT_MAX_ROUNDS = 3

STOP_SUBMITTED = "the reasoner submitted"
STOP_TURNS = "the turn cap was reached without a submission"
STOP_SILENT = "the reasoner stopped without submitting and there was nothing to fall back on"

# The CLI builtins named explicitly, so the common cases fail closed even if the hook is ever
# bypassed. Not the mechanism -- `_gate` is -- and deliberately not exhaustive: a list that claimed
# to be would be wrong by the next release, which is the argument for the hook in the first place.
BUILTINS_DENIED = ("Bash", "BashOutput", "Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep",
                   "WebFetch", "WebSearch", "Task", "Agent", "Skill", "Workflow", "SendMessage",
                   "ListAgents", "KillShell", "TodoWrite")


def qualified(tool: str) -> str:
    """The name the model calls, and the name the hook is given."""
    return f"mcp__{SERVER_NAME}__{tool}"


OUR_TOOLS = frozenset(qualified(t) for t in P.TOOLS)


def _gate(tool_name: str, *, submitted: bool, discovery: frozenset = frozenset()) -> str | None:
    """None to allow; otherwise the refusal, written to be read by the model.

    Deny by default. The allowlist is `OUR_TOOLS` plus, if the runtime needs it, whichever
    read-only tool the model uses to load a deferred schema -- see `discovery_tools`.

    Pure, so every decision it makes is a test rather than a live experiment. That matters more
    here than anywhere else in the agent: this function is what makes the measurement budget mean
    something, and a hole in it is not visible in any record it fails to prevent.
    """
    if tool_name in OUR_TOOLS or tool_name in discovery:
        if submitted and tool_name != qualified("cluster_status"):
            # After `submit` the run's conclusion is written. Letting `measure` run past it would
            # spend the node on a question the run has already answered, and letting anything else
            # run would produce a transcript whose last word is not the one that was submitted.
            # `cluster_status` is allowed through because it changes nothing and the honest reason
            # to call it late is to check the node was left as intended.
            return ("this run is over: you have submitted, and the verdict is recorded. No further "
                    "tool call will be accepted. Stop now.")
        return None
    return (f"{tool_name} is not available to this agent. You have exactly six tools: "
            f"{', '.join(P.TOOLS)}. There is no shell, no filesystem and no other route to the "
            "cluster -- `measure` is the only one, and it is what the measurement budget counts. "
            "Call one of the six instead.")


class SdkPackerAgent:
    """The search, with the SDK as the runtime. Same result type as `ModelPackerAgent`."""

    def __init__(self, fleet, store, agent, *, model_id: str = DEFAULT_MODEL_ID,
                 region: str = DEFAULT_REGION, budget: int = 6, top_k: int = 4,
                 solver_seconds: float = 10.0, max_turns: int = DEFAULT_MAX_TURNS,
                 max_rounds: int = DEFAULT_MAX_ROUNDS, include_prompt: bool = True,
                 discovery_tools: tuple = ()):
        self.fleet = fleet
        self.store = store
        self.agent = agent
        self.model_id = model_id
        self.region = region
        self.budget = int(budget)
        self.top_k = int(top_k)
        self.solver_seconds = float(solver_seconds)
        self.max_turns = int(max_turns)
        self.max_rounds = int(max_rounds)
        self.include_prompt = include_prompt
        # Empty, and measured to be safe to leave that way. An MCP tool's schema can arrive
        # deferred, in which case a model must load it before it can call anything at all -- so this
        # is the escape hatch for naming whichever read-only tool does that. It is not needed: in
        # three runs against the fakes the model opened with a schema lookup, was refused, and went
        # straight on to call all six by name. The schemas load eagerly. The lookup attempt survives
        # the prompt telling it there is nothing to look up, so it costs two turns and no GPU every
        # run -- which is a price worth paying to keep the deny-by-default set free of exceptions.
        self.discovery_tools = frozenset(discovery_tools)
        self.name = f"sdk:{model_id}"

    # --- the parts that need no SDK -------------------------------------------

    def tool_specs(self) -> list:
        """`(qualified_name, description, schema)` for each of the six. Pure, and asserted on."""
        return [(t, P.TOOL_SUMMARIES[t], P.TOOL_SCHEMAS[t]) for t in P.TOOLS]

    def gate(self, tool_name: str, *, submitted: bool) -> str | None:
        return _gate(tool_name, submitted=submitted, discovery=self.discovery_tools)

    def opening(self, tools: Tools) -> str:
        """The same problem statement, closing on the one instruction this runtime can honour."""
        return P.problem(fleet=self.fleet, budget=self.budget, store_runs=len(tools.records()),
                         key_hash=self.store.key_hash,
                         calibration=self.store.read_calibration(), closing=P.SDK_CLOSING)

    def call_tool(self, tools: Tools, name: str, args: dict, transcript: Transcript) -> str:
        """One tool call, and the text that goes back to the model.

        The translation `packer.py` does inline, done here so it is testable without the SDK: a
        `ToolError` becomes a refusal the model reads and can act on, and `submit`'s sentinel
        becomes a sentence, because under this runtime the model *sees* the result of submitting
        and "submitted" alone does not tell it to stop.

        Anything that is not a `ToolError` is re-raised. `Tools.records()` raises when the store is
        partially unreadable, and that must not become a message the model reasons around: a search
        against a history it cannot fully see is the failure the exception exists to cause.
        """
        try:
            out = tools.dispatch(name, args)
        except P.ToolError as exc:
            out = f"{name} refused: {exc}"
        if out == SUBMITTED:
            out = ("submitted. Your verdict and rationale are recorded and this run is over. "
                   "Stop now -- no further tool call will be accepted.")
        transcript.turns.append(Turn("user", out))
        return out

    # --- the run --------------------------------------------------------------

    def run(self, *, teardown_at_end: bool = False) -> PackerResult:
        """Drive the search. Synchronous, like every other entry point in this package."""
        import anyio
        return anyio.run(self._run, teardown_at_end)

    async def _run(self, teardown_at_end: bool) -> PackerResult:
        from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient,
                                      HookMatcher, ResultMessage, TextBlock, ToolUseBlock,
                                      create_sdk_mcp_server, tool)

        tools = Tools(self.fleet, self.store, self.agent, budget=self.budget, top_k=self.top_k,
                      solver_seconds=self.solver_seconds, reasoner_name=self.name)
        transcript = Transcript(reasoner=self.name, system=P.SYSTEM_SDK,
                                turns=[Turn("user", self.opening(tools))])
        result = PackerResult(attempts=tools.attempts, transcript=transcript)
        fatal: list = []
        t0 = time.time()

        def wrap(name: str):
            @tool(name, P.TOOL_SUMMARIES[name], P.TOOL_SCHEMAS[name])
            async def handler(args):
                try:
                    text = self.call_tool(tools, name, dict(args or {}), transcript)
                except Exception as exc:            # noqa: BLE001 - stashed and re-raised below
                    # Not handed to the model as a tool result. An exception here is a fault in the
                    # agent, not a refusal, and a model told about it would work around it.
                    log.exception("%s failed fatally", name)
                    fatal.append(exc)
                    text = "this run has failed and is ending. Stop."
                return {"content": [{"type": "text", "text": text}]}
            return handler

        server = create_sdk_mcp_server(name=SERVER_NAME, version="1.0.0",
                                       tools=[wrap(t) for t in P.TOOLS])

        async def pre_tool_use(input_data, tool_use_id, context):
            refusal = self.gate(input_data.get("tool_name", ""),
                                submitted=tools.submitted is not None)
            if refusal is None:
                return {}
            log.info("refused %s", input_data.get("tool_name"))
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                           "permissionDecision": "deny",
                                           "permissionDecisionReason": refusal}}

        options = ClaudeAgentOptions(
            model=self.model_id,
            system_prompt=P.SYSTEM_SDK,
            mcp_servers={SERVER_NAME: server},
            allowed_tools=sorted(OUR_TOOLS),
            disallowed_tools=list(BUILTINS_DENIED),
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[pre_tool_use])]},
            # Nothing from the filesystem. A settings file could add an allow rule that shadows a
            # permission decision, and this agent's tool set must be a property of this module.
            setting_sources=[],
            max_turns=self.max_turns,
            env={"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": self.region},
        )

        # The opening problem statement first; afterwards, whatever the fallback appended. Read off
        # the transcript rather than kept in a second variable, so what was sent and what is
        # recorded as having been sent cannot differ.
        async with ClaudeSDKClient(options=options) as client:
            for _ in range(self.max_rounds):
                await client.query(transcript.turns[-1].text)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        result.turns += 1
                        said = []
                        for block in message.content:
                            if isinstance(block, TextBlock) and block.text.strip():
                                said.append(block.text.strip())
                            elif isinstance(block, ToolUseBlock):
                                said.append(f"[calls {block.name} {block.input}]")
                        if said:
                            transcript.turns.append(Turn("assistant", "\n".join(said)))
                    elif isinstance(message, ResultMessage):
                        transcript.usage = dict(getattr(message, "usage", None) or {})
                        transcript.usage["total_cost_usd"] = getattr(
                            message, "total_cost_usd", None)
                        transcript.usage["sdk_stop"] = getattr(message, "subtype", None)
                transcript.latency_s = time.time() - t0
                if fatal:
                    raise fatal[0]
                if tools.submitted is not None:
                    result.submitted = dict(tools.submitted)
                    result.stopped_because = STOP_SUBMITTED
                    break
                if result.turns >= self.max_turns:
                    result.stopped_because = STOP_TURNS
                    break
                # It stopped talking without submitting. Same treatment as an unusable reply in
                # `packer.py`: measure the solver's top-ranked candidate, record that nothing chose
                # it, and tell the model what was done on its behalf before asking again.
                if not recorded_fallback(self.fleet, tools, result,
                                         "the reasoner stopped without submitting",
                                         top_k=self.top_k, solver_seconds=self.solver_seconds):
                    result.stopped_because = STOP_SILENT
                    break
                transcript.turns.append(Turn("user", fallback_note(result)))
            else:
                result.stopped_because = STOP_SILENT

        result.verdict = derive_verdict(tools, result)
        if teardown_at_end and not (result.submitted or {}).get("leave_deployed"):
            live = self.agent.deployer.live_tenant_releases()
            if live:
                log.info("tearing down %s", live)
                self.agent.deployer.destroy(live)
        log.info("finished after %d turn(s) and %d measurement(s): %s -> %s",
                 result.turns, tools.spent, result.stopped_because, result.verdict)
        return result
