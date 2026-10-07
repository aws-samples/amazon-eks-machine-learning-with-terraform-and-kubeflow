"""Any OpenAI-compatible chat completions endpoint, for example the Dynamo frontend.

For tool calls to arrive in message.tool_calls, the server must parse them. For Qwen3 on
Dynamo, that is dgd-agg-tools.yaml in examples/inference/dynamo/vllm/qwen3-8b.
"""
from __future__ import annotations

import json

from openai import OpenAI

from . import Reply, ToolCall


def _to_openai(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name, "arguments": c.raw_arguments
                                                   or json.dumps(c.arguments or {})}}
                                     for c in m["tool_calls"]]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class OpenAIBackend:
    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key or "none")

    def chat(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> Reply:
        kwargs = {"tools": tools} if tools else {}
        r = self.client.chat.completions.create(model=self.model, messages=_to_openai(system, messages),
                                                max_tokens=max_tokens, **kwargs)
        choice = r.choices[0]
        calls = []
        for c in choice.message.tool_calls or []:
            try:
                arguments = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = None
            calls.append(ToolCall(c.id, c.function.name, arguments, c.function.arguments or ""))
        usage = {"input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens} if r.usage else {}
        return Reply(choice.message.content or "", calls, choice.finish_reason or "", usage)
