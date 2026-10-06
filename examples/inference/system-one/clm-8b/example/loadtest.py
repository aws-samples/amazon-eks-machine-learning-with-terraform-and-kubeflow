#!/usr/bin/env python3
"""Load-test clm-serve: check that every request completes at increasing concurrency.

Every request asks three typed questions (one of each type) about one state from
states.jsonl, with a unique prefix, so no request is answered from the state cache and
each one costs three state embeddings, like a new production request.

Run it inside the cluster, next to the services (serve.ipynb Step 7 runs it as a Job).
Through kubectl port-forward, concurrent connections break the tunnel.

    python loadtest.py --url http://clm-serve:8700 --encoder-metrics-url http://clm-encoder:8000/metrics

A request that fails or never completes counts as an error (the client times out after
60 seconds). With --encoder-metrics-url, the script reads the encoder's
vllm:num_requests_running five seconds after each level; it should be 0 on an idle server.

The results record request and error counts only, not latency or throughput.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent

QUESTIONS = {
    "topic": {"type": "choice", "instructions": "What is this request about?",
              "criteria": {"software": "Code, systems and data", "business": "Money, contracts and customers",
                           "general": "Anything else"}},
    "detail": {"type": "score", "instructions": "How much detail does a good answer need?",
               "criteria": ["A sentence", "A paragraph", "A full document"]},
    "factual": {"type": "noul", "instructions": "This request asks for facts that can be checked."},
}


def running_requests(metrics_url: str) -> float | None:
    """Sum of vllm:num_requests_running over all label sets, or None if unreadable."""
    try:
        text = requests.get(metrics_url, timeout=5).text
    except requests.RequestException:
        return None
    values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
              if line.startswith("vllm:num_requests_running")]
    return sum(values) if values else None


def level(session: requests.Session, url: str, model: str, states: list[str], concurrency: int) -> dict:
    errors = []

    def one(state):
        try:
            r = session.post(f"{url}/v1/systemone", timeout=60,
                             json={"state": f"[{uuid.uuid4().hex[:12]}] {state}", "model": model,
                                   "questions": QUESTIONS})
            r.raise_for_status()
            return set(r.json()["answers"]) == set(QUESTIONS)
        except (requests.RequestException, KeyError, ValueError) as e:
            errors.append(f"{type(e).__name__}: {e}"[:200])
            return False

    with ThreadPoolExecutor(concurrency) as pool:
        ok = sum(pool.map(one, states))
    return {"concurrency": concurrency, "requests": len(states), "errors": len(states) - ok,
            "error_samples": errors[:3]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("CLM_BASE_URL", "http://localhost:8700"))
    ap.add_argument("--model", default="clm-latest")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32")
    ap.add_argument("--requests", type=int, default=96, help="requests per concurrency level")
    ap.add_argument("--encoder-metrics-url", help="clm-encoder's /metrics, to check for requests left running")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    url = args.url.rstrip("/")
    session = requests.Session()
    if os.environ.get("CLM_API_KEY"):
        session.headers["Authorization"] = f"Bearer {os.environ['CLM_API_KEY']}"
    health = session.get(f"{url}/health", timeout=10).json()
    if not health.get("embedder"):
        sys.exit(f"clm-serve is up but its encoder is not: {health}")
    if health.get("mock"):
        print("WARNING: clm-serve reports mock=true; this tests the script, not CLM-8B.", file=sys.stderr)

    rows = [json.loads(line) for line in (HERE / "states.jsonl").read_text().splitlines() if line.strip()]
    states = [rows[i % len(rows)]["state"] for i in range(args.requests)]
    level(session, url, args.model, states[:1], 1)   # embeds the option texts once, before the first level

    levels = []
    for c in (int(x) for x in args.concurrency.split(",")):
        row = level(session, url, args.model, states, c)
        if args.encoder_metrics_url:
            time.sleep(5)   # let any finished request leave the scheduler before reading the gauge
            row["encoder_running_after"] = running_requests(args.encoder_metrics_url)
        levels.append(row)
        stuck = f"  encoder running after: {row['encoder_running_after']}" if args.encoder_metrics_url else ""
        print(f"concurrency {c:3}: {row['requests']} requests  errors {row['errors']}{stuck}")
        for sample in row["error_samples"]:
            print(f"    {sample}")

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "loadtest.json"), "w") as f:
        json.dump({"levels": levels, "requests_per_level": args.requests, "questions": list(QUESTIONS),
                   "mock": bool(health.get("mock"))}, f, indent=2)
    failed = sum(r["errors"] for r in levels)
    print(f"{failed} of {args.requests * len(levels)} requests failed")


if __name__ == "__main__":
    main()
