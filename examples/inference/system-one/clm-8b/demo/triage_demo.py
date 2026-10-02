#!/usr/bin/env python3
"""Request triage with CLM-8B: one System One call per prompt, three typed answers.

For every prompt in prompts.jsonl the demo asks clm-serve three questions in one
/v1/systemone call:

  task_type    choice  what kind of work the prompt is
  tier         choice  which capability tier (tiers.yaml) the work needs
  high_stakes  noul    would a wrong answer cause real harm

and reports latency, input tokens, accuracy against the hand labels, and how a
confidence gate on the tier answer trades coverage for accuracy.

Latency is reported three ways, because clm-serve caches state embeddings:
  first call    the very first request after startup (cold option texts)
  steady state  new prompts, after a separate warm-up set has been sent
  repeat        the same prompts again; the state cache answers, zero encoder calls

The labels are one reviewer's judgement on a small set. This is an illustration
of the shape of the trade-off, not a benchmark.

    pip install -r requirements.txt
    kubectl port-forward svc/clm-serve 8700:8700 -n kubeflow-user-example-com &
    python triage_demo.py --url http://localhost:8700
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests
import yaml

HERE = Path(__file__).resolve().parent

TASK_TYPES = {
    "code": "Writing, fixing, reviewing or explaining code",
    "math_reasoning": "Calculation, quantitative analysis or proof",
    "extraction": "Pulling structured fields out of supplied text",
    "summarization": "Condensing supplied content",
    "tool_use": "Acting on external systems or looking up live data",
    "open_chat": "Conversation, advice or creative writing",
    "other": "Unclear or not enough information",
}

GATE_THRESHOLDS = [round(0.1 * i, 1) for i in range(10)]
ESCALATION_THRESHOLD = 0.5


def questions(tiers: dict[str, str]) -> dict:
    return {
        "task_type": {"type": "choice", "instructions": "What kind of task is this request?",
                      "criteria": TASK_TYPES},
        "tier": {"type": "choice", "instructions": "What capability does answering this request need?",
                 "criteria": tiers},
        "high_stakes": {"type": "noul",
                        "instructions": "A wrong answer to this request could cause financial, legal, "
                                        "medical, safety or production harm."},
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def latency_row(calls: list[dict]) -> dict:
    wall = [c["wall_ms"] for c in calls]
    server = [c["server_ms"] for c in calls if c["server_ms"] is not None]
    tokens = [c["input_tokens"] for c in calls if c["input_tokens"] is not None]
    return {"n": len(calls),
            "wall_p50_ms": pct(wall, 0.5), "wall_p95_ms": pct(wall, 0.95),
            "server_p50_ms": pct(server, 0.5), "server_p95_ms": pct(server, 0.95),
            "input_tokens_mean": statistics.mean(tokens) if tokens else None}


class CLM:
    def __init__(self, url: str, api_key: str | None, model: str):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        if api_key:
            self.s.headers["Authorization"] = f"Bearer {api_key}"
        self.model = model

    def health(self) -> dict:
        r = self.s.get(f"{self.url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def ask(self, prompt: str, qs: dict) -> dict:
        t0 = time.perf_counter()
        r = self.s.post(f"{self.url}/v1/systemone", timeout=60,
                        json={"state": prompt, "model": self.model, "questions": qs})
        wall_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        body = r.json()
        server_ms = r.headers.get("X-CLM-Latency-Ms")
        return {"answers": body["answers"], "wall_ms": wall_ms,
                "server_ms": float(server_ms) if server_ms else None,
                "input_tokens": body.get("usage", {}).get("input_tokens")}


def decision(answers: dict) -> dict:
    tier, hs = answers["tier"], answers["high_stakes"]["noul"]
    return {"task_type": answers["task_type"]["choice"],
            "task_type_confidence": answers["task_type"]["confidence"],
            "tier": tier["choice"], "tier_confidence": tier["confidence"],
            "tier_probabilities": tier["probabilities"],
            "high_stakes": hs >= 0.5, "high_stakes_p": hs,
            # noul answers carry no confidence field; distance from 0.5, rescaled to 0..1
            "high_stakes_confidence": abs(hs - 0.5) * 2}


def accuracy(rows: list[dict], key: str) -> float | None:
    return sum(r["pred"][key] == r[f"gold_{key}"] for r in rows) / len(rows) if rows else None


def gating(rows: list[dict]) -> list[dict]:
    """Accept the tier answer when its confidence clears the threshold, else escalate one tier."""
    out = []
    for t in GATE_THRESHOLDS:
        kept = [r for r in rows if r["pred"]["tier_confidence"] >= t]
        out.append({"threshold": t, "coverage": len(kept) / len(rows),
                    "accuracy": accuracy(kept, "tier"), "n_kept": len(kept)})
    return out


def plot_gating(gate: list[dict], path: Path, mock: bool) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the chart", file=sys.stderr)
        return False
    pts = [g for g in gate if g["accuracy"] is not None]
    xs, ys = [g["threshold"] for g in pts], [g["accuracy"] for g in pts]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.plot(xs, ys, color="#2a78d6", linewidth=2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=2)
    # label the ends and the escalation threshold only, with coverage alongside accuracy
    for g in pts:
        if g is pts[0] or g is pts[-1] or g["threshold"] == ESCALATION_THRESHOLD:
            first = g is pts[0]   # the first label sits right of its point, clear of the y-axis
            ax.annotate(f"{g['accuracy']:.0%} acc\n{g['coverage']:.0%} kept", (g["threshold"], g["accuracy"]),
                        textcoords="offset points", xytext=(10, -26) if first else (-14, 10),
                        ha="left" if first else "right", fontsize=8, color="#3d3d3d")
    title = "Tier accuracy on the requests CLM keeps, by confidence threshold"
    if mock:
        title += "\nMOCK ENCODER: pipeline test only, not CLM-8B"
    ax.set_title(title, fontsize=10, color="#1a1a1a", loc="left")
    ax.set_xlabel("confidence threshold (below it, escalate one tier)", fontsize=9, color="#5c5c5c")
    ax.set_ylabel("tier accuracy on kept requests", fontsize=9, color="#5c5c5c")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(GATE_THRESHOLDS)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbdbd")
    ax.tick_params(colors="#5c5c5c", labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


BASELINE_PROMPT = """Classify the user request below. Reply with JSON only, no prose:
{{"task_type": one of {task_types}, "tier": one of {tiers}, "high_stakes": true or false}}

tier meanings:
{tier_lines}
high_stakes: a wrong answer could cause financial, legal, medical, safety or production harm.

Request:
{prompt}"""


def baseline(url: str, model: str, api_key: str | None, rows: list[dict], tiers: dict[str, str]) -> dict:
    """The same three decisions from a generative model behind an OpenAI-compatible endpoint."""
    s = requests.Session()
    if api_key:
        s.headers["Authorization"] = f"Bearer {api_key}"
    results, lat, failures = [], [], 0
    for r in rows:
        content = BASELINE_PROMPT.format(task_types=list(TASK_TYPES), tiers=list(tiers),
                                         tier_lines="\n".join(f"- {k}: {v}" for k, v in tiers.items()),
                                         prompt=r["prompt"])
        t0 = time.perf_counter()
        resp = s.post(f"{url.rstrip('/')}/v1/chat/completions", timeout=120,
                      json={"model": model, "temperature": 0, "max_tokens": 100,
                            "messages": [{"role": "user", "content": content}]})
        lat.append((time.perf_counter() - t0) * 1000)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            parsed = json.loads(text[start:end])
            pred = {"task_type": parsed["task_type"], "tier": parsed["tier"], "high_stakes": bool(parsed["high_stakes"])}
            if pred["task_type"] not in TASK_TYPES or pred["tier"] not in tiers:
                raise ValueError("label outside the allowed set")
        except (ValueError, KeyError, TypeError):
            failures += 1
            pred = {"task_type": None, "tier": None, "high_stakes": None}
        results.append({**r, "pred": pred})
    return {"model": model, "parse_failures": failures, "n": len(rows),
            "wall_p50_ms": pct(lat, 0.5), "wall_p95_ms": pct(lat, 0.95),
            "accuracy": {k: accuracy(results, k) for k in ("task_type", "tier", "high_stakes")}}


def fmt(x, spec=".1f", suffix=""):
    return "n/a" if x is None else f"{x:{spec}}{suffix}"


def summary_md(m: dict) -> str:
    lines = []
    if m["mock"]:
        lines += ["> **MOCK ENCODER.** These numbers come from character n-grams, not CLM-8B. "
                  "They test the pipeline only. Do not quote them.", ""]
    lines += [f"# CLM-8B triage demo ({m['n_prompts']} prompts, {m['n_ambiguous']} ambiguous)", "",
              "Illustrative, not a benchmark: the labels are one reviewer's judgement.", "",
              "## Latency (one call answers all three questions)", "",
              "| pass | n | wall p50 | wall p95 | server p50 | server p95 | input tokens (mean) |",
              "|---|---|---|---|---|---|---|"]
    for name, row in m["latency"].items():
        lines.append(f"| {name} | {row['n']} | {fmt(row['wall_p50_ms'], suffix=' ms')} | "
                     f"{fmt(row['wall_p95_ms'], suffix=' ms')} | {fmt(row['server_p50_ms'], suffix=' ms')} | "
                     f"{fmt(row['server_p95_ms'], suffix=' ms')} | {fmt(row['input_tokens_mean'])} |")
    lines += ["", "## Accuracy against the labels", "", "| question | all | clear | ambiguous |", "|---|---|---|---|"]
    for k, v in m["accuracy"].items():
        lines.append(f"| {k} | {fmt(v['all'], '.0%')} | {fmt(v['clear'], '.0%')} | {fmt(v['ambiguous'], '.0%')} |")
    lines += ["", "## Confidence gating on the tier answer", "",
              "Below the threshold the router escalates one tier instead of trusting CLM.", "",
              "| threshold | kept | tier accuracy on kept |", "|---|---|---|"]
    for g in m["gating"]:
        lines.append(f"| {g['threshold']:.1f} | {g['coverage']:.0%} ({g['n_kept']}) | {fmt(g['accuracy'], '.0%')} |")
    lines += ["", f"## Escalations at threshold {ESCALATION_THRESHOLD}", "",
              "| id | CLM tier | confidence | label | high-stakes p |", "|---|---|---|---|---|"]
    for e in m["escalations"]:
        lines.append(f"| {e['id']} | {e['tier']} | {e['tier_confidence']:.2f} | {e['gold_tier']} | {e['high_stakes_p']:.2f} |")
    if m.get("baseline"):
        b = m["baseline"]
        lines += ["", f"## Generative baseline: {b['model']}", "",
                  f"Parse failures: {b['parse_failures']}/{b['n']}. "
                  f"Wall p50 {fmt(b['wall_p50_ms'], suffix=' ms')}, p95 {fmt(b['wall_p95_ms'], suffix=' ms')}.", "",
                  "| question | baseline accuracy | CLM accuracy |", "|---|---|---|"]
        for k, v in b["accuracy"].items():
            lines.append(f"| {k} | {fmt(v, '.0%')} | {fmt(m['accuracy'][k]['all'], '.0%')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("CLM_BASE_URL", "http://localhost:8700"))
    ap.add_argument("--model", default="clm-latest")
    ap.add_argument("--prompts", type=Path, default=HERE / "prompts.jsonl")
    ap.add_argument("--warmup", type=Path, default=HERE / "warmup.jsonl")
    ap.add_argument("--tiers", type=Path, default=HERE / "tiers.yaml")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--baseline-url", help="OpenAI-compatible endpoint for a generative-router baseline")
    ap.add_argument("--baseline-model")
    args = ap.parse_args()
    if bool(args.baseline_url) != bool(args.baseline_model):
        ap.error("--baseline-url and --baseline-model go together")

    clm = CLM(args.url, os.environ.get("CLM_API_KEY"), args.model)
    health = clm.health()
    if not health.get("embedder"):
        sys.exit(f"clm-serve is up but its encoder is not: {health}")
    mock = bool(health.get("mock"))
    if mock:
        print("WARNING: clm-serve reports mock=true. Numbers are meaningless; use them to test the pipeline only.",
              file=sys.stderr)

    tiers = yaml.safe_load(args.tiers.read_text())["tiers"]
    qs = questions(tiers)
    prompts, warmup = load_jsonl(args.prompts), load_jsonl(args.warmup)
    seen = {p["prompt"] for p in prompts}
    if any(w["prompt"] in seen for w in warmup):
        sys.exit("warm-up prompts must not overlap the evaluation prompts, or steady state reads the cache")

    first = clm.ask(warmup[0]["prompt"], qs)
    for w in warmup[1:]:
        clm.ask(w["prompt"], qs)
    rows, steady = [], []
    for p in prompts:
        call = clm.ask(p["prompt"], qs)
        steady.append(call)
        rows.append({**p, "pred": decision(call["answers"])})
    repeat = [clm.ask(p["prompt"], qs) for p in prompts]

    clear = [r for r in rows if not r["ambiguous"]]
    ambiguous = [r for r in rows if r["ambiguous"]]
    gate = gating(rows)
    escalations = [{"id": r["id"], "gold_tier": r["gold_tier"], **r["pred"]} for r in rows
                   if r["pred"]["tier_confidence"] < ESCALATION_THRESHOLD]
    metrics = {
        "mock": mock, "url": args.url, "model": args.model, "health": health,
        "n_prompts": len(rows), "n_ambiguous": len(ambiguous),
        "latency": {"first call": latency_row([first]), "steady state": latency_row(steady),
                    "repeat (cached)": latency_row(repeat)},
        "accuracy": {k: {"all": accuracy(rows, k), "clear": accuracy(clear, k), "ambiguous": accuracy(ambiguous, k)}
                     for k in ("task_type", "tier", "high_stakes")},
        "gating": gate,
        "escalations": [{k: e[k] for k in ("id", "tier", "tier_confidence", "gold_tier", "high_stakes_p")}
                        for e in escalations],
        "predictions": [{"id": r["id"], **r["pred"]} for r in rows],
    }
    if args.baseline_url:
        metrics["baseline"] = baseline(args.baseline_url, args.baseline_model, os.environ.get("BASELINE_API_KEY"),
                                       rows, tiers)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    summary = summary_md(metrics)
    (args.out / "summary.md").write_text(summary)
    plot_gating(gate, args.out / "confidence_gating.png", mock)
    print(summary)
    print(f"wrote {args.out}/metrics.json, summary.md, confidence_gating.png")


if __name__ == "__main__":
    main()
