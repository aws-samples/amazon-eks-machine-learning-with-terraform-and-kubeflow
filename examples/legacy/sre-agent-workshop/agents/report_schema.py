"""RootCauseReport schema for the SRE agent.

Keep this file readable — participants inspect it in Part A to understand
what shape the agent's output takes and why enums (rather than free-text
strings) are used for critical fields.

The report is emitted as JSON to ~/sre-agent/reports/<scenario>.json and
printed to stdout so it's visible inline when Claude Code runs the agent.
"""
from __future__ import annotations

from typing import Literal, TypedDict


RemediationClass = Literal[
    "fixable_at_cluster_level",
    "requires_account_change",
    "requires_code_change",
    "unknown",
]


class Evidence(TypedDict, total=False):
    """One observation the agent made during investigation.

    `source` is a short label like "kubectl:list_events" or "cloudwatch:filter_log_events".
    `observation` is the raw fact — a quoted event message, a numeric value, a
    log line. Do not paraphrase.
    """
    id: str                # short evidence identifier (e.g. "e1", "e2")
    source: str            # short tool label
    observation: str       # raw fact
    timestamp: str         # ISO 8601 if available


class Hypothesis(TypedDict):
    """A candidate root cause with a confidence and its supporting evidence."""
    statement: str
    confidence: float                     # 0.0 – 1.0
    supporting_evidence_ids: list[str]    # references Evidence.id


class RootCauseReport(TypedDict):
    """Structured output of an investigation.

    The `remediation_class` enum is the most important field. Free-text
    would let the model equivocate; the enum forces an honest verdict.
    """
    summary: str
    evidence: list[Evidence]
    hypotheses: list[Hypothesis]
    remediation_class: RemediationClass
    mitigation_plan: str | list[str]      # str for narrative escalations; list for concrete steps


def validate_report(payload: dict) -> None:
    """Minimal validator run before writing the report to disk.

    Extend this as the workshop progresses — Module 2 adds mitigation_plan_execution;
    Module 3 adds correlated_alarms. Kept intentionally small in Module 1.
    """
    for field in ("summary", "evidence", "hypotheses", "remediation_class", "mitigation_plan"):
        if field not in payload:
            raise ValueError(f"report missing required field: {field}")

    valid_classes = {"fixable_at_cluster_level", "requires_account_change",
                     "requires_code_change", "unknown"}
    if payload["remediation_class"] not in valid_classes:
        raise ValueError(
            f"remediation_class must be one of {valid_classes}, got "
            f"{payload['remediation_class']!r}"
        )

    for i, ev in enumerate(payload["evidence"]):
        if "source" not in ev or "observation" not in ev:
            raise ValueError(f"evidence[{i}] missing source or observation")

    for i, hyp in enumerate(payload["hypotheses"]):
        for field in ("statement", "confidence", "supporting_evidence_ids"):
            if field not in hyp:
                raise ValueError(f"hypotheses[{i}] missing {field}")
        if not 0.0 <= hyp["confidence"] <= 1.0:
            raise ValueError(f"hypotheses[{i}].confidence out of range")
