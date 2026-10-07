"""Programmatic checks for eval.py: did a tier's attempt pass its task? No LLM judges it.

A task's "check" holds any of these; the attempt passes when all of them hold:

  contains       strings that must all appear in the answer; a list inside means any one of them
  equals         the whole answer, for "reply with ... only" tasks
  min_words, max_words, max_sentences, lines, bullets
  json           keys and values the answer, parsed as JSON, must have
  regex          the answer is a regular expression: it must match every string in match
                 and none in no_match
  calls          tool calls that must have run, with arguments that include args
  no_calls       tools that must not have run (a call the gate blocked did not run)

Text is compared in lower case, without Markdown emphasis or thousands separators.
"""
from __future__ import annotations

import json
import re


def _norm(text: str) -> str:
    text = re.sub(r"[*_`]", "", text.lower())
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip().strip("`").strip()


def _same(want, got) -> bool:
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(want - got) < 0.01
    return str(want).strip().lower() == str(got).strip().lower()


def _ran(attempt) -> list[dict]:
    """The tool calls that ran: not blocked by the gate and not rejected by the tool."""
    out = []
    for c in attempt.tool_calls:
        if c.get("gate") and not c["gate"]["allowed"]:
            continue
        if "error" in json.loads(c.get("result") or "{}"):
            continue
        out.append(c)
    return out


def check(spec: dict, attempt) -> tuple[bool, list[str]]:
    """Return (passed, the checks that failed)."""
    if attempt.error:
        return False, [f"error: {attempt.error}"]
    if not attempt.finished:
        return False, ["no final answer"]
    answer, text = attempt.answer.strip(), _norm(attempt.answer)
    fails = []
    for item in spec.get("contains", []):
        options = item if isinstance(item, list) else [item]
        if not any(_norm(o) in text for o in options):
            fails.append(f"missing {' or '.join(options)!r}")
    if "equals" in spec and text.strip().rstrip(".!").strip() != _norm(spec["equals"]):
        fails.append(f"not exactly {spec['equals']!r}")
    words = len(answer.split())
    if "min_words" in spec and words < spec["min_words"]:
        fails.append(f"{words} words < {spec['min_words']}")
    if "max_words" in spec and words > spec["max_words"]:
        fails.append(f"{words} words > {spec['max_words']}")
    if "max_sentences" in spec:
        n = len(re.findall(r"[.!?](\s|$)", answer))
        if n > spec["max_sentences"]:
            fails.append(f"{n} sentences > {spec['max_sentences']}")
    lines = [l for l in answer.splitlines() if l.strip()]
    if "lines" in spec and len(lines) != spec["lines"]:
        fails.append(f"{len(lines)} lines, not {spec['lines']}")
    if "bullets" in spec:
        n = sum(bool(re.match(r"\s*([-*•]|\d+[.)])\s", l)) for l in lines)
        if n != spec["bullets"] or n != len(lines):
            fails.append(f"{n} bullets in {len(lines)} lines, not {spec['bullets']}")
    if "json" in spec:
        try:
            got = json.loads(_strip_fences(answer))
            if not all(_same(v, got.get(k)) for k, v in spec["json"].items()):
                fails.append(f"JSON is {got}")
        except (json.JSONDecodeError, AttributeError):
            fails.append("not JSON")
    if "regex" in spec:
        try:
            pattern = re.compile(_strip_fences(answer).strip("/"))
            wrong = [s for s in spec["regex"]["match"] if not pattern.fullmatch(s)] + \
                    [s for s in spec["regex"]["no_match"] if pattern.fullmatch(s)]
            if wrong:
                fails.append(f"regex wrong on {wrong}")
        except re.error:
            fails.append("not a valid regex")
    ran = _ran(attempt)
    for want in spec.get("calls", []):
        if not any(c["name"] == want["name"] and
                   all(_same(v, (c["arguments"] or {}).get(k)) for k, v in want.get("args", {}).items())
                   for c in ran):
            fails.append(f"no {want['name']}({json.dumps(want.get('args', {}))}) ran")
    for name in spec.get("no_calls", []):
        if any(c["name"] == name for c in ran):
            fails.append(f"{name} ran")
    return not fails, fails
