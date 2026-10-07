"""Model backends. Each one turns the harness's messages into one API's format and back.

The harness keeps the conversation in one neutral format:

    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": "...", "content": "..."}

and every backend returns a Reply.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None      # None when the model's arguments were not valid JSON
    raw_arguments: str = ""


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)   # input_tokens, output_tokens


def make_backend(tier: dict):
    """Build the backend a tier in a tiers file names."""
    if tier["backend"] == "bedrock":
        from .bedrock import BedrockBackend
        return BedrockBackend(tier["model"], region=tier.get("region"))
    if tier["backend"] == "openai":
        from .openai_compat import OpenAIBackend
        return OpenAIBackend(tier["model"], base_url=tier["base_url"], api_key=tier.get("api_key"))
    raise ValueError(f"unknown backend {tier['backend']!r} in tier {tier['name']!r}")
