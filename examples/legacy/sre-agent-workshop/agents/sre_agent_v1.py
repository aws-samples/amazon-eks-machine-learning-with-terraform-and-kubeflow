"""SRE Incident Response Agent — Module 1.

Read-only investigation agent that observes signals across Kubernetes,
Karpenter, and CloudWatch, correlates them, and produces a structured
Root Cause Report.

Design decisions (explained inline for participants):

1. System prompt is a module-level constant (SYSTEM_PROMPT below).
   Read and edit it in place — it's the single most important knob for
   changing agent behavior.

2. Two MCP servers are spawned as stdio subprocesses at agent start.
   The SDK owns their lifecycle.

3. The tool surface is DELIBERATELY curated. eks-mcp-server exposes
   dozens of tools; we allowlist the six that matter for a Pending-pod
   investigation. Unnecessary tools tempt the agent off-path.

4. describe_service_quota is a local Python tool, not an MCP call.
   For a 15-line helper, an in-process function is simpler than spinning
   up another MCP server.

5. Termination: 25 steps OR report emitted, whichever comes first.
   The step budget prevents runaway exploration on ambiguous incidents.

Usage:
    python sre_agent_v1.py --scenario baseline
    python sre_agent_v1.py --scenario gpu_incident \\
        --prompt "Ray inference pods stuck Pending. Investigate."
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Path setup so `from tools import ...` works regardless of cwd.
AGENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_DIR))

from claude_agent_sdk import (              # noqa: E402
    ClaudeAgentOptions,
    query,
    tool,
)
from claude_agent_sdk import create_sdk_mcp_server  # noqa: E402

from report_schema import RootCauseReport, validate_report  # noqa: E402
from tools.describe_service_quota import describe_service_quota  # noqa: E402


# ── Configuration ─────────────────────────────────────────────────────────

WORKSHOP_CLUSTER = os.environ.get("SRE_CLUSTER_NAME", "sre-agent-workshop")
REGION = os.environ.get("AWS_REGION", "us-east-1")
REPORTS_DIR = Path.home() / "sre-agent" / "reports"
MODEL_ID = os.environ.get(
    "SRE_AGENT_MODEL",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)


# ── System prompt ─────────────────────────────────────────────────────────
#
# THIS IS THE MOST IMPORTANT CODE IN THE FILE.
#
# The system prompt is the agent's operating manual. Every rule below
# changes agent behavior in observable ways. Read carefully — and when you
# want to explore how the agent responds differently, edit this constant
# in place and re-run.
#
# Design notes for participants:
# - Rule 1 (read-only) is a hard constraint. The agent has no write tools
#   registered in Module 1; the prompt reinforces this at the semantic layer.
# - Rule 2 (correlate before hypothesising) is what makes Module 1's GPU
#   scenario land correctly. Without it, the agent stops at the first
#   plausible signal.
# - Rule 3 (refuse to invent fixes) is arguably the most important rule in
#   the whole workshop. A confident-sounding wrong answer is worse than an
#   honest "insufficient evidence."
# - Rule 5 (classify remediation) uses a four-way enum. Free-text would let
#   the model equivocate; the enum forces a verdict.

SYSTEM_PROMPT = """\
# SRE Incident Response Agent — Module 1

You are an SRE agent for the AnyCompany Retail Platform team. Your job is to
investigate cluster incidents, correlate signals across sources, and produce
a structured Root Cause Report.

## Operating rules

1. **Read-only.** You may inspect the cluster, Karpenter, CloudWatch, and
   AWS Service Quotas. You may NOT modify any resource in this module.

2. **Correlate before hypothesising.** No single signal is conclusive. If
   you have only one source of evidence, gather at least one more before
   committing to a hypothesis.

3. **Refuse to invent fixes.** If evidence for a fix is not strong, mark
   `remediation_class` as `unknown` or `requires_account_change` and
   explain what you'd need to see. It is better to escalate honestly than
   to guess.

4. **Cite evidence.** Every hypothesis must reference specific observations
   (event message, NodeClaim condition text, CloudWatch log line, quota
   value). Vague citations are worthless in an incident.

5. **Classify remediation.** Every report must set `remediation_class` to
   one of:
   - `fixable_at_cluster_level` — a Kubernetes / Karpenter change fixes it
   - `requires_account_change` — needs quota, IAM, or SCP change outside cluster
   - `requires_code_change` — application code or deployment manifest change
   - `unknown` — insufficient evidence to classify

6. **Emit and stop.** When you have enough evidence to fill the report,
   emit the JSON and terminate. Do not keep exploring.

## Available tool categories

- **eks-mcp-server** (read-only): pods, events, deployments, Karpenter
  NodePools and NodeClaims
- **cloudwatch-mcp-server**: metrics, log filter, alarms
- **describe_service_quota** (local): current + default values of any AWS
  service quota

## Report contract

Emit JSON matching the RootCauseReport schema:

- `summary` (string): one-line human-readable summary
- `evidence` (list): `[{source, observation, timestamp?}]`
- `hypotheses` (list): `[{statement, confidence, supporting_evidence_ids}]`
   confidence is a float 0.0–1.0
- `remediation_class`: enum (see above)
- `mitigation_plan` (string OR structured): required if
  `remediation_class == fixable_at_cluster_level` (list of steps); for other
  classes, an explicit sentence stating why cluster-level fix is not applicable

## Termination

You have a budget of 25 tool-call turns. Stop when you emit the report or
exhaust the budget, whichever comes first.

## Style

- No small talk. No hedging language.
- Precise, technical, dense.
- Treat the user as an experienced SRE who wants the answer.
"""


# ── Curated tool surface ──────────────────────────────────────────────────
#
# eks-mcp-server exposes many tools. We allow only these six. Every other
# tool the MCP server offers is not registered on the agent's side, so the
# model cannot call it. This is a design decision, not a limitation.
EKS_MCP_ALLOWED_TOOLS: list[str] = [
    "list_pods",
    "get_pod",
    "list_events",
    "describe_deployment",
    "describe_nodepool",
    "list_nodeclaims",
    "describe_nodeclaim",
]

# cloudwatch-mcp-server tool surface — small on purpose.
CLOUDWATCH_MCP_ALLOWED_TOOLS: list[str] = [
    "filter_log_events",
    "get_metric_data",
    "describe_alarms",
]


# ── Local tool wrapper ────────────────────────────────────────────────────
#
# The @tool decorator turns a Python function into an Agent SDK tool. This
# is the simplest possible tool integration — no MCP server required for
# thin helpers. Module 4 shows how to promote a local tool into its own
# MCP server when you want other agents (or Claude Code) to reuse it.

@tool(
    "describe_service_quota",
    "Look up an AWS Service Quota by service code + quota code. Returns "
    "the current applied value and whether it can be raised. Use to check "
    "whether an EC2 provisioning failure is caused by an account quota.",
    {"service_code": str, "quota_code": str},
)
async def describe_service_quota_tool(args: dict) -> dict:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    describe_service_quota(args["service_code"], args["quota_code"]),
                    indent=2,
                ),
            }
        ]
    }


# ── MCP server config ─────────────────────────────────────────────────────

def build_mcp_servers() -> dict:
    """Return the MCP server dict passed to ClaudeAgentOptions.

    Participants: notice that we spawn each server as a subprocess via `uvx`.
    The SDK reads/writes their stdio; you don't manage the processes yourself.

    If either server fails to start (bad uvx install, missing kubeconfig,
    IAM issues), the agent errors immediately on the first tool call.
    """
    return {
        "eks": {
            "type": "stdio",
            "command": "uvx",
            "args": ["awslabs.eks-mcp-server@latest", "--readonly"],
            "env": {**os.environ},  # inherits AWS_PROFILE, KUBECONFIG, PATH, etc.
        },
        "cloudwatch": {
            "type": "stdio",
            "command": "uvx",
            "args": ["awslabs.cloudwatch-mcp-server@latest"],
            "env": {**os.environ},
        },
    }


# ── Prompt builder ────────────────────────────────────────────────────────

def build_user_prompt(scenario: str, extra_prompt: str | None) -> str:
    """Compose the initial user message from scenario + optional extra prompt.

    The scenario name tells the agent whether it's investigating a live
    incident or establishing baseline. The extra_prompt lets the operator
    inject specifics — a symptom, a suspicious service, a time window.
    """
    if scenario == "baseline":
        core = (
            f"You are running a BASELINE HEALTH CHECK against the "
            f"{WORKSHOP_CLUSTER} cluster (region {REGION}). "
            f"Investigate the cluster as if you were on-call. Confirm no "
            f"incidents are in progress. Emit a Root Cause Report with "
            f"remediation_class=unknown and empty hypotheses if the cluster "
            f"is healthy. Do NOT invent an incident."
        )
    else:
        core = (
            f"You are on-call for the {WORKSHOP_CLUSTER} cluster "
            f"(region {REGION}). Investigate the incident described below "
            f"and produce a Root Cause Report. Correlate at least two "
            f"signal sources before committing to a hypothesis."
        )

    if extra_prompt:
        core = f"{core}\n\nIncident context:\n{extra_prompt}"

    core += (
        "\n\nWhen you have sufficient evidence, emit a single JSON object "
        "matching the RootCauseReport schema in your final message. "
        "The JSON must be the entire message body, prefixed by the marker "
        "line: ---REPORT---"
    )
    return core


# ── Report extraction ─────────────────────────────────────────────────────

def extract_report(final_text: str) -> dict:
    """Pull the JSON report out of the agent's final message.

    The system prompt tells the model to prefix its report with ---REPORT---.
    We slice off everything before that marker, then parse the rest as JSON.
    """
    marker = "---REPORT---"
    if marker not in final_text:
        raise ValueError(
            "Agent did not emit a report marker (---REPORT---). "
            "Final message was:\n" + final_text[:1000]
        )
    json_str = final_text.split(marker, 1)[1].strip()
    # Strip markdown fences if the model added them
    if json_str.startswith("```"):
        json_str = json_str.split("```", 2)[1]
        if json_str.startswith("json"):
            json_str = json_str[len("json"):]
        json_str = json_str.strip()
    return json.loads(json_str)


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report["_meta"] = {
        "cluster": WORKSHOP_CLUSTER,
        "region": REGION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
    }
    path.write_text(json.dumps(report, indent=2))
    # Print the report to stdout so it's visible inline when Claude Code
    # (or a participant) runs the agent — no need to cat the file afterward.
    print("\n===== ROOT CAUSE REPORT =====")
    print(json.dumps(report, indent=2))
    print("=============================")
    print(f"[OK] report also written to {path}", file=sys.stderr)


# ── Main agentic loop ─────────────────────────────────────────────────────

async def run(scenario: str, extra_prompt: str | None, report_path: Path) -> None:
    # Local tools wrapped as an SDK MCP server so the agent can call them.
    local_tools_server = create_sdk_mcp_server(
        name="local-tools",
        version="0.1.0",
        tools=[describe_service_quota_tool],
    )

    options = ClaudeAgentOptions(
        model=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        max_turns=25,           # matches step budget in system prompt
        mcp_servers={
            **build_mcp_servers(),
            "local": local_tools_server,
        },
        allowed_tools=[
            # eks-mcp-server tools
            *[f"mcp__eks__{t}" for t in EKS_MCP_ALLOWED_TOOLS],
            # cloudwatch-mcp-server tools
            *[f"mcp__cloudwatch__{t}" for t in CLOUDWATCH_MCP_ALLOWED_TOOLS],
            # local tools
            "mcp__local__describe_service_quota",
        ],
    )

    user_prompt = build_user_prompt(scenario, extra_prompt)

    print(f"[info] scenario: {scenario}", file=sys.stderr)
    print(f"[info] cluster:  {WORKSHOP_CLUSTER}", file=sys.stderr)
    print(f"[info] model:    {MODEL_ID}", file=sys.stderr)
    print(f"[info] report:   {report_path}", file=sys.stderr)
    print("[info] streaming agent output …", file=sys.stderr)
    print("─" * 72, file=sys.stderr)

    final_text_parts: list[str] = []

    async for message in query(prompt=user_prompt, options=options):
        # Stream every message to stderr so participants can watch the agent's
        # reasoning + tool calls. Adjust verbosity here if it's too much.
        msg_type = type(message).__name__
        if msg_type == "AssistantMessage":
            for block in message.content:
                block_type = type(block).__name__
                if block_type == "TextBlock":
                    text = block.text
                    print(text, file=sys.stderr)
                    final_text_parts.append(text)
                elif block_type == "ToolUseBlock":
                    print(f"[tool-call] {block.name}({json.dumps(block.input)[:200]})",
                          file=sys.stderr)
        elif msg_type == "UserMessage":
            # Tool results — show a truncated preview
            for block in message.content:
                if type(block).__name__ == "ToolResultBlock":
                    preview = str(block.content)[:300]
                    print(f"[tool-result] {preview}", file=sys.stderr)
        elif msg_type == "ResultMessage":
            print(f"[done] turns={message.num_turns} cost=${message.total_cost_usd:.4f}",
                  file=sys.stderr)

    print("─" * 72, file=sys.stderr)

    final_text = "\n".join(final_text_parts)
    report = extract_report(final_text)
    validate_report(report)
    save_report(report, report_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SRE Incident Response Agent — Module 1 (read-only investigator)",
    )
    p.add_argument(
        "--scenario",
        required=True,
        help=(
            "Short scenario name. Used as the report filename. "
            "Use 'baseline' for healthy-cluster runs; other values indicate "
            "an incident investigation."
        ),
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Additional incident context to append to the initial prompt.",
    )
    p.add_argument(
        "--report-path",
        default=None,
        help=(
            "Explicit report output path. Defaults to "
            "~/sre-agent/reports/<scenario>.json."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_path) if args.report_path else (
        REPORTS_DIR / f"{args.scenario}.json"
    )
    asyncio.run(run(args.scenario, args.prompt, report_path))


if __name__ == "__main__":
    main()
