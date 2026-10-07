#!/usr/bin/env python3
"""Run tasks through the model-router harness with one tiers file.

    python run.py --tiers tiers/hybrid.yaml --task "When will order A1001 ship?"
    python run.py --tiers tiers/bedrock.yaml --tasks tasks/demo.jsonl
    python run.py --tiers tiers/bedrock.yaml --tasks tasks/demo.jsonl --route-only
    python run.py --tiers tiers/hybrid.yaml --tasks tasks/demo.jsonl --only refund-unrequested --tier small
    python run.py --tiers tiers/hybrid.yaml --tasks tasks/demo.jsonl --in-cluster-only

For each task it prints the route, each tier's tool calls with the gate's verdict, any
escalation, and the final answer. --route-only asks CLM-8B for the route and calls no
model. --tier skips the route and starts every task on one tier, for example to see the
gate in front of the small model. --prefer, --in-cluster-only and --latency-budget override
the tiers file's policy, and --profile merges a measured profile from eval.py into the
tiers. Each run's full traces, including token usage and timings, go to results/<tiers>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src import hooks, policy
from src.clm import CLM
from src.loop import load_tiers, run_task


def show(tid: str, trace) -> None:
    print(f"\n[{tid}] {trace.task}")
    print(f"  route: {trace.route['tier']}  ({'; '.join(trace.route['reasons'])})")
    for i, a in enumerate(trace.attempts):
        print(f"  {a.tier_name} ({a.model}): {a.steps} step(s)")
        for c in a.tool_calls:
            gate = c.get("gate")
            verdict = "" if gate is None else f"  gate: {'allowed' if gate['allowed'] else 'BLOCKED'} " \
                                              f"({'; '.join(gate['reasons'])})"
            print(f"    tool {c['name']}({json.dumps(c['arguments'])}){verdict}")
        if a.error:
            print(f"    error: {a.error}")
        if i < len(trace.escalations):
            e = trace.escalations[i]
            print(f"  escalate {e['from']} -> {e['to']}  ({'; '.join(e['reasons'])})")
    answer = trace.answer.replace("\n", " ")
    print(f"  answer ({trace.final_tier}): {answer[:300]}{'...' if len(answer) > 300 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiers", required=True, help="a tiers file, for example tiers/hybrid.yaml")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--task", help="one task")
    src.add_argument("--tasks", help="a JSONL file of {\"id\", \"task\"} lines")
    ap.add_argument("--only", help="comma-separated task IDs to run from --tasks")
    ap.add_argument("--route-only", action="store_true", help="print the route for each task and call no model")
    ap.add_argument("--tier", help="start every task on this tier instead of routing it, for example small")
    ap.add_argument("--prefer", choices=["order", "cost", "latency"], help="which allowed tier to start on")
    ap.add_argument("--in-cluster-only", action="store_true", help="use only tiers with profile.in_cluster")
    ap.add_argument("--latency-budget", type=float, metavar="MS", help="skip tiers whose profile.latency_ms is higher")
    ap.add_argument("--profile", help="measured profile to merge into the tiers, for example results/measured-hybrid.yaml")
    ap.add_argument("--clm-url", default=os.environ.get("CLM_BASE_URL", "http://localhost:8700"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()

    cfg = load_tiers(args.tiers, args.profile)
    if args.prefer:
        cfg.setdefault("policy", {})["prefer"] = args.prefer
    constraints = {}
    if args.in_cluster_only:
        constraints["in_cluster_only"] = True
    if args.latency_budget is not None:
        constraints["latency_budget_ms"] = args.latency_budget
    if all(policy.excluded(t, {**cfg.get("policy", {}), **constraints}) for t in cfg["tiers"]):
        sys.exit("no tier meets the constraints: " + "; ".join(
            f"{t['name']}: {policy.excluded(t, {**cfg.get('policy', {}), **constraints})}" for t in cfg["tiers"]))
    clm = CLM(args.clm_url, api_key=os.environ.get("CLM_API_KEY"))
    if not clm.healthy():
        sys.exit(f"clm-serve at {args.clm_url} is not ready (check that /health shows \"embedder\": true)")

    if args.task:
        tasks = [{"id": "task", "task": args.task}]
    else:
        tasks = [json.loads(line) for line in open(args.tasks) if line.strip()]
        if args.only:
            keep = set(args.only.split(","))
            tasks = [t for t in tasks if t["id"] in keep]

    if args.route_only:
        constraints = {**cfg.get("policy", {}), **constraints}
        for t in tasks:
            s = hooks.signals(clm, t["task"], cfg["signals"])
            i, reasons = policy.choose(cfg["tiers"], s.value, cfg.get("policy", {}), constraints)
            print(f"{t['id']:20} {cfg['tiers'][i]['name']:8} {'; '.join(s.reasons + reasons)}")
        return

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{cfg['name']}.jsonl")
    with open(path, "a") as f:
        for t in tasks:
            trace = run_task(t["task"], cfg, clm, start_tier=args.tier, constraints=constraints)
            show(t["id"], trace)
            f.write(json.dumps({"id": t["id"], **trace.to_dict()}) + "\n")
    print(f"\ntraces appended to {path}")


if __name__ == "__main__":
    main()
