"""Amazon Bedrock through the Converse API.

Converse carries tool calls as toolUse content blocks and tool results as toolResult
blocks in a user message, so this backend translates both ways.
"""
from __future__ import annotations

import boto3

from . import Reply, ToolCall


def _to_converse(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "assistant":
            content = [{"text": m["content"]}] if m.get("content") else []
            content += [{"toolUse": {"toolUseId": c.id, "name": c.name, "input": c.arguments or {}}}
                        for c in m.get("tool_calls") or []]
            out.append({"role": "assistant", "content": content})
        elif m["role"] == "tool":
            block = {"toolResult": {"toolUseId": m["tool_call_id"], "content": [{"text": m["content"]}]}}
            # Converse wants all the results for one assistant turn in a single user message.
            if out and out[-1]["role"] == "user" and "toolResult" in out[-1]["content"][0]:
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": "user", "content": [{"text": m["content"]}]})
    return out


def _tool_config(tools: list[dict]) -> dict:
    return {"tools": [{"toolSpec": {"name": t["function"]["name"], "description": t["function"]["description"],
                                    "inputSchema": {"json": t["function"]["parameters"]}}} for t in tools]}


class BedrockBackend:
    def __init__(self, model: str, region: str | None = None):
        self.model = model
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def chat(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> Reply:
        kwargs = {"toolConfig": _tool_config(tools)} if tools else {}
        r = self.client.converse(modelId=self.model, system=[{"text": system}], messages=_to_converse(messages),
                                 inferenceConfig={"maxTokens": max_tokens}, **kwargs)
        text, calls = [], []
        for block in r["output"]["message"]["content"]:
            if "text" in block:
                text.append(block["text"])
            elif "toolUse" in block:
                u = block["toolUse"]
                calls.append(ToolCall(u["toolUseId"], u["name"], u.get("input") or {}))
        usage = {"input_tokens": r["usage"]["inputTokens"], "output_tokens": r["usage"]["outputTokens"]}
        return Reply("".join(text), calls, r.get("stopReason", ""), usage)
