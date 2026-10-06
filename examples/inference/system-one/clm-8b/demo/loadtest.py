#!/usr/bin/env python3
"""Load-test clm-serve: throughput, latency and errors at increasing concurrency.

Every request asks the triage demo's three questions (triage_demo.py) about one prompt
from prompts.jsonl, with a unique prefix, so no request is answered from the state
cache and each one costs three state embeddings, like a new production request.

Run it inside the cluster, next to the services (serve.ipynb Step 8 runs it as a Job).
Through kubectl port-forward, concurrent connections break the tunnel.

    python loadtest.py --url http://clm-serve:8700 --encoder-metrics-url http://clm-encoder:8000/metrics

A request that never completes shows up as an error (a client timeout). With
--encoder-metrics-url, the script reads the encoder's vllm:num_requests_running five
seconds after each level; it should be 0 on an idle server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml

from triage_demo import CLM, HERE, load_jsonl, pct, questions


def running_requests(metrics_url: str) -> float | None:
    """Sum of vllm:num_requests_running over all label sets, or None if unreadable."""
    try:
        text = requests.get(metrics_url, timeout=5).text
    except requests.RequestException:
        return None
    values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
              if line.startswith("vllm:num_requests_running")]
    return sum(values) if values else None


def level(clm: CLM, qs: dict, texts: list[str], concurrency: int) -> dict:
    errors = []

    def one(text):
        try:
            return clm.ask(f"[{uuid.uuid4().hex[:12]}] {text}", qs)
        except (requests.RequestException, KeyError, ValueError) as e:
            errors.append(f"{type(e).__name__}: {e}"[:200])
            return None

    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        calls = list(pool.map(one, texts))
    elapsed = time.perf_counter() - t0
    ok = [c for c in calls if c]
    wall = [c["wall_ms"] for c in ok]
    server = [c["server_ms"] for c in ok if c["server_ms"] is not None]
    return {"concurrency": concurrency, "requests": len(texts), "errors": len(texts) - len(ok),
            "error_samples": errors[:3], "throughput_rps": len(ok) / elapsed,
            "wall_p50_ms": pct(wall, 0.5), "wall_p95_ms": pct(wall, 0.95),
            "server_p50_ms": pct(server, 0.5), "server_p95_ms": pct(server, 0.95)}


def fmt(x: float | None, spec: str = ".0f") -> str:
    return "n/a" if x is None else format(x, spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("CLM_BASE_URL", "http://localhost:8700"))
    ap.add_argument("--model", default="clm-latest")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32")
    ap.add_argument("--requests", type=int, default=96, help="requests per concurrency level")
    ap.add_argument("--encoder-metrics-url", help="clm-encoder's /metrics, to check for requests left running")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    clm = CLM(args.url, os.environ.get("CLM_API_KEY"), args.model)
    health = clm.health()
    if not health.get("embedder"):
        sys.exit(f"clm-serve is up but its encoder is not: {health}")
    if health.get("mock"):
        print("WARNING: clm-serve reports mock=true. Numbers are meaningless; use them to test the pipeline only.",
              file=sys.stderr)

    qs = questions(yaml.safe_load((HERE / "tiers.yaml").read_text())["tiers"])
    prompts = [p["prompt"] for p in load_jsonl(HERE / "prompts.jsonl")]
    texts = [prompts[i % len(prompts)] for i in range(args.requests)]
    clm.ask(f"[{uuid.uuid4().hex[:12]}] {texts[0]}", qs)   # embeds the option texts once, outside the timings

    levels = []
    for c in (int(x) for x in args.concurrency.split(",")):
        row = level(clm, qs, texts, c)
        if args.encoder_metrics_url:
            time.sleep(5)   # let any finished request leave the scheduler before reading the gauge
            row["encoder_running_after"] = running_requests(args.encoder_metrics_url)
        levels.append(row)
        stuck = f"  encoder running after: {fmt(row.get('encoder_running_after'))}" if args.encoder_metrics_url else ""
        print(f"concurrency {c:3}: {row['throughput_rps']:6.1f} req/s  p50 {fmt(row['wall_p50_ms']):>6} ms  "
              f"p95 {fmt(row['wall_p95_ms']):>6} ms  errors {row['errors']}{stuck}")
    peak = max(levels, key=lambda r: r["throughput_rps"])

    lines = ["# clm-serve load test", "",
             "| concurrency | req/s | p50 ms | p95 ms | server p50 ms | server p95 ms | errors |"
             + (" encoder running after |" if args.encoder_metrics_url else ""),
             "|---|---|---|---|---|---|---|" + ("---|" if args.encoder_metrics_url else "")]
    for r in levels:
        lines.append(f"| {r['concurrency']} | {r['throughput_rps']:.1f} | {fmt(r['wall_p50_ms'])} | "
                     f"{fmt(r['wall_p95_ms'])} | {fmt(r['server_p50_ms'])} | {fmt(r['server_p95_ms'])} | {r['errors']} |"
                     + (f" {fmt(r.get('encoder_running_after'))} |" if args.encoder_metrics_url else ""))
    lines += ["", f"{args.requests} requests per level, three questions each (three state embeddings), "
                  "no state-cache hits. Wall times include the client; "
                  "server times are `X-CLM-Latency-Ms`.",
              f"Peak: {peak['throughput_rps']:.1f} requests/s at concurrency {peak['concurrency']}."]
    if health.get("mock"):
        lines.insert(1, "\n**MOCK ENCODER: pipeline test only, not CLM-8B.**")

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "loadtest.json"), "w") as f:
        json.dump({"levels": levels, "peak_rps": peak["throughput_rps"], "peak_concurrency": peak["concurrency"],
                   "requests_per_level": args.requests, "questions": list(qs), "clm_health": health}, f, indent=2)
    with open(os.path.join(out, "loadtest.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
