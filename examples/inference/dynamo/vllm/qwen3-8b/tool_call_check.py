#!/usr/bin/env python3
"""Check tool calling end to end against a Dynamo deployment of dgd-agg-tools.yaml.

The calls an agent loop makes:

  1. Send a question with one tool. The model must answer with a tool call, in
     message.tool_calls, with arguments that parse as JSON and name the city asked about.
  2. Send the tool's result back as a "tool" message. The model must answer in plain
     content, with no further tool call.
  3. Send the same question with tool_choice="none". The model must not call the tool.
  4. Send the question again with stream=True. The streamed deltas must add up to the
     same tool call.

It also checks that the reasoning parser moved the <think> block out of content.

    kubectl port-forward -n kubeflow-user-example-com svc/dyn-qwen3-8b-frontend 8000:8000
    python tool_call_check.py --base-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import json
import sys

from openai import OpenAI

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, for example Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]
QUESTION = "What is the weather in Paris right now?"
TOOL_RESULT = {"city": "Paris", "temperature": 18, "unit": "celsius", "conditions": "light rain"}

failures = []


def check(ok: bool, what: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="room for the <think> block before the tool call or the answer")
    args = ap.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="dummy")

    # 1. The model calls the tool.
    messages = [{"role": "user", "content": QUESTION}]
    r = client.chat.completions.create(model=args.model, messages=messages, tools=TOOLS,
                                       tool_choice="auto", max_tokens=args.max_tokens)
    msg = r.choices[0].message
    calls = msg.tool_calls or []
    check(r.choices[0].finish_reason == "tool_calls",
          f"finish_reason is tool_calls (got {r.choices[0].finish_reason})")
    check(len(calls) == 1 and calls[0].function.name == "get_weather",
          f"one get_weather call (got {[c.function.name for c in calls]})")
    check("<tool_call>" not in (msg.content or ""), "no raw <tool_call> markup left in content")
    check("<think>" not in (msg.content or ""), "no <think> block left in content")
    check(bool(getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning_content")),
          "reasoning_content is set")
    if not calls:
        sys.exit(f"{len(failures)} check(s) failed; no tool call to continue with. content: {msg.content!r}")
    try:
        arguments = json.loads(calls[0].function.arguments)
    except json.JSONDecodeError:
        arguments = {}
    check(str(arguments.get("city", "")).lower() == "paris",
          f"arguments parse as JSON and name Paris (got {calls[0].function.arguments!r})")

    # 2. The model answers from the tool result.
    messages += [
        {"role": "assistant", "content": msg.content or "",
         "tool_calls": [{"id": c.id, "type": "function",
                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                        for c in calls]},
        {"role": "tool", "tool_call_id": calls[0].id, "content": json.dumps(TOOL_RESULT)},
    ]
    r = client.chat.completions.create(model=args.model, messages=messages, tools=TOOLS,
                                       max_tokens=args.max_tokens)
    msg = r.choices[0].message
    check(not msg.tool_calls, "no second tool call after the tool result")
    check("18" in (msg.content or ""), "the answer uses the tool result (mentions 18)")
    print(f"      answer: {(msg.content or '').strip()[:200]!r}")

    # 3. tool_choice="none" suppresses the call.
    r = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": QUESTION}],
                                       tools=TOOLS, tool_choice="none", max_tokens=args.max_tokens)
    check(not r.choices[0].message.tool_calls, 'no tool call with tool_choice="none"')

    # 4. Streaming returns the tool call in the deltas.
    stream = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": QUESTION}],
                                            tools=TOOLS, max_tokens=args.max_tokens, stream=True)
    name, arguments, finish = "", "", None
    for chunk in stream:
        if not chunk.choices:
            continue
        finish = chunk.choices[0].finish_reason or finish
        for c in chunk.choices[0].delta.tool_calls or []:
            name += c.function.name or ""
            arguments += c.function.arguments or ""
    try:
        city = str(json.loads(arguments).get("city", "")).lower()
    except json.JSONDecodeError:
        city = ""
    check(name == "get_weather" and city == "paris" and finish == "tool_calls",
          f"streamed get_weather call for Paris (got {name!r}, {arguments!r}, finish {finish})")

    if failures:
        sys.exit(f"{len(failures)} check(s) failed")
    print("all checks passed")


if __name__ == "__main__":
    main()
