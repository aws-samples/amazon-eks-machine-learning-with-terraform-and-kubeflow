"""AWS Service Quotas lookup tool.

Neither eks-mcp-server nor cloudwatch-mcp-server exposes AWS Service Quotas.
This tool is what turns Module 1's investigation from "plausible-sounding guess"
into "correct verdict backed by an authoritative source."

Design notes participants should read:

- Returns the *current* value, not the default. Current is what the account
  actually has right now; default is what AWS ships with.
- Region defaults to us-east-1 because that's the workshop region. Change if
  your agent runs in another region.
- Structured errors, not exceptions. The agent must be able to read a rate-limit
  or access-denied response and react.
"""
from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.exceptions import ClientError


REGION = os.environ.get("AWS_REGION", "us-east-1")

# Common quota codes participants may reach for. This is documentation, not
# an allowlist — the agent may call with any (service_code, quota_code) pair.
KNOWN_QUOTA_CODES = {
    "ec2:L-DB2E81BA": "Running On-Demand G and VT instances (vCPU count)",
    "ec2:L-1216C47A": "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances (vCPU count)",
    "ec2:L-7212CCBC": "Running On-Demand P instances (vCPU count)",
    "vpc:L-F678F1CE": "VPCs per Region",
}


def describe_service_quota(service_code: str, quota_code: str) -> dict[str, Any]:
    """Look up a specific AWS service quota.

    Returns:
        On success:
            {
                "service_code": str,
                "quota_code": str,
                "quota_name": str,          # human-readable name
                "value": float,             # current applied value
                "unit": str,                # None | "None" | "Count" etc.
                "adjustable": bool,         # can this be raised?
                "region": str,
            }
        On failure:
            {"error": {"code": str, "message": str}}
    """
    try:
        client = boto3.client("service-quotas", region_name=REGION)
        resp = client.get_service_quota(
            ServiceCode=service_code,
            QuotaCode=quota_code,
        )
        q = resp["Quota"]
        return {
            "service_code": service_code,
            "quota_code": quota_code,
            "quota_name": q.get("QuotaName", ""),
            "value": q["Value"],
            "unit": q.get("Unit", "None"),
            "adjustable": q["Adjustable"],
            "region": REGION,
        }
    except ClientError as e:
        return {
            "error": {
                "code": e.response.get("Error", {}).get("Code", "Unknown"),
                "message": str(e),
            }
        }


# Tool schema for the Claude Agent SDK.
TOOL_SCHEMA = {
    "name": "describe_service_quota",
    "description": (
        "Look up a specific AWS Service Quota by service code and quota code. "
        "Returns the current applied value, whether the quota is adjustable, "
        "and metadata. Use when investigating whether an EC2 RunInstances "
        "failure, service throttling, or capacity limit is due to an account "
        "quota constraint rather than a cluster-level configuration issue. "
        f"Common quota codes: {', '.join(f'{k} ({v})' for k, v in KNOWN_QUOTA_CODES.items())}. "
        "Returns a structured error object (not an exception) on failure."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "service_code": {
                "type": "string",
                "description": "The AWS service code (e.g. 'ec2', 'vpc', 'servicequotas').",
            },
            "quota_code": {
                "type": "string",
                "description": "The quota code (e.g. 'L-DB2E81BA' for GPU vCPU).",
            },
        },
        "required": ["service_code", "quota_code"],
    },
}
