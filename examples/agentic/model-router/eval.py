#!/usr/bin/env python3
"""Measure the router on a labelled task suite: run every task on every tier, check the answers
in code, and compare routing strategies.

    python eval.py --tiers tiers/bedrock.yaml
    python eval.py --tiers tiers/bedrock.yaml --reuse results/eval-bedrock.jsonl --fit hard

Each task in the suite (tasks/eval.jsonl) has a check that decides pass or fail with no LLM
judge; see src/checks.py. The run stores, for every task and tier, the result, the tokens, the
time and CLM-8B's escalate answer. Everything after that is computed from the stored rows,
so --reuse re-analyses them with the current tiers file and checks without calling any model.

The report shows:
  strategies  always one tier, the router, the router with escalation, and an oracle that knows
              the first tier that passes. Pass rate, where tasks start, tokens per task, and
              cost and time relative to always using the last tier (cost only with prices)
  by type     pass rate of each tier for each task type in the suite
  signals     how well each CLM-8B option predicts that the first tier fails (AUC: 0.5 is
              chance, 1.0 is perfect)
  --fit X     the per-tier limits on p(X) that keep the pass rate of always using the last
              tier at the lowest cost, with a 5-fold cross-validated result

Results go to results/eval-<tiers>.jsonl and results/measured-<tiers>.yaml (median time and
mean tokens per task for each tier, for run.py --profile). Neither belongs in version control.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import yaml

from src import hooks, policy
from src.checks import check
from src.clm import CLM, CLMError
from src.loop import Attempt, _transcript, load_tiers, run_agent

_local = threading.local()


def _clm(url: str) -> CLM:
    if not hasattr(_local, "clm"):
        _local.clm = CLM(url, api_key=os.environ.get("CLM_API_KEY"))
    return _local.clm


def _retry(fn, tries: int = 3):
    """Retry a step that failed to reach clm-serve: that is the harness failing, not the model."""
    for k in range(tries):
        try:
            out = fn()
            if not (isinstance(out, Attempt) and out.error.startswith("CLMError")):
                return out
        except CLMError:
            if k == tries - 1:
                raise
        time.sleep(5)
    return out


def run_one(t: dict, cfg: dict, url: str) -> dict:
    """Signals once, then the task on every tier with its check and CLM-8B's escalate answer."""
    clm = _clm(url)
    row = {"id": t["id"], "type": t.get("type", ""), "task": t["task"],
           "signals": _retry(lambda: hooks.signals(clm, t["task"], cfg["signals"])).value, "tiers": {}}
    for i, tier in enumerate(cfg["tiers"]):
        a = _retry(lambda: run_agent(i, tier, t["task"], clm, cfg))
        passed, fails = check(t["check"], a)
        p_answered = None
        if a.finished and not a.error:
            p_answered = _retry(lambda: hooks.escalate(clm, t["task"], _transcript(a), a.answer, 0)).answers["answered"]
        row["tiers"][tier["name"]] = {
            "pass": passed, "fails": fails, "finished": a.finished, "error": a.error,
            "input_tokens": sum(u.get("input_tokens", 0) for u in a.usage),
            "output_tokens": sum(u.get("output_tokens", 0) for u in a.usage),
            "wall_ms": a.wall_ms, "p_answered": p_answered, "answer": a.answer,
            "tool_calls": [{k: c.get(k) for k in ("name", "arguments", "gate", "result")} for c in a.tool_calls]}
    return row


# ---- strategies, computed from stored rows ----

def _escalates(r: dict, cfg: dict) -> bool:
    return not r["finished"] or bool(r["error"]) or r["p_answered"] < cfg["escalate_below"]


def outcome(row: dict, cfg: dict, constraints: dict, strategy: str) -> dict:
    """Where a task starts and ends under a strategy, and the tiers it ran on."""
    tiers, names = cfg["tiers"], [t["name"] for t in cfg["tiers"]]
    allowed = [i for i, t in enumerate(tiers) if not policy.excluded(t, constraints)]
    if strategy.startswith("always:"):
        visited = [names.index(strategy[7:])]
    elif strategy == "oracle":
        visited = [next((i for i in allowed if row["tiers"][names[i]]["pass"]), allowed[-1])]
    else:
        i, _ = policy.choose(tiers, row["signals"], cfg.get("policy", {}), constraints)
        visited = [i]
        while strategy == "router+escalate" and _escalates(row["tiers"][names[i]], cfg):
            j = policy.next_tier(tiers, i, constraints)
            if j is None:
                break
            visited.append(j)
            i = j
    runs = [row["tiers"][names[i]] for i in visited]
    cost = None
    if all(policy.est_cost(tiers[i]) is not None for i in visited):
        cost = sum((tiers[i]["profile"]["price_per_mtok"]["input"] * r["input_tokens"]
                    + tiers[i]["profile"]["price_per_mtok"]["output"] * r["output_tokens"]) / 1e6
                   for i, r in zip(visited, runs))
    return {"start": visited[0], "visited": visited, "pass": runs[-1]["pass"], "cost": cost,
            "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in runs),
            "wall_ms": sum(r["wall_ms"] for r in runs)}


def summarize(rows: list[dict], cfg: dict, constraints: dict, strategy: str) -> dict:
    outs = [outcome(r, cfg, constraints, strategy) for r in rows]
    costs = [o["cost"] for o in outs]
    return {"pass": sum(o["pass"] for o in outs), "n": len(outs),
            "starts": [sum(o["start"] == i for o in outs) for i in range(len(cfg["tiers"]))],
            "tokens": statistics.mean(o["tokens"] for o in outs),
            "cost": None if None in costs else sum(costs), "wall_ms": sum(o["wall_ms"] for o in outs),
            # without prices, each step up the list counts as one more unit of spend
            "spend": sum(costs) if None not in costs else sum(sum(i + 1 for i in o["visited"]) for o in outs)}


def report(rows: list[dict], cfg: dict, constraints: dict) -> None:
    names = [t["name"] for t in cfg["tiers"]]
    allowed = [n for t, n in zip(cfg["tiers"], names) if not policy.excluded(t, constraints)]
    strategies = [f"always:{n}" for n in allowed] + ["router", "router+escalate", "oracle"]
    base = summarize(rows, cfg, constraints, f"always:{allowed[-1]}")
    print(f"\n{'strategy':22} {'pass':>9}  {'starts (' + ' / '.join(names) + ')':28} {'tokens/task':>11} "
          f"{'cost':>6} {'time':>6}   (cost and time relative to always:{allowed[-1]})")
    for s in strategies:
        m = summarize(rows, cfg, constraints, s)
        cost = f"{m['cost'] / base['cost']:.2f}" if m["cost"] is not None and base["cost"] else "-"
        print(f"{s:22} {m['pass']:>3}/{m['n']:<3} {m['pass'] / m['n']:>4.0%}  {' / '.join(map(str, m['starts'])):28} "
              f"{m['tokens']:>11.0f} {cost:>6} {m['wall_ms'] / base['wall_ms']:>6.2f}")

    types = sorted({r["type"] for r in rows})
    print(f"\n{'pass rate by type':22} " + " ".join(f"{n:>9}" for n in names))
    for ty in types:
        sub = [r for r in rows if r["type"] == ty]
        print(f"{ty + f' ({len(sub)})':22} " + " ".join(f"{sum(r['tiers'][n]['pass'] for r in sub) / len(sub):>9.0%}"
                                                       for n in names))

    first = allowed[0]
    failed = [not r["tiers"][first]["pass"] for r in rows]
    print(f"\nsignals: AUC for predicting that {first} fails ({sum(failed)} of {len(rows)} tasks)")
    if 0 < sum(failed) < len(rows):
        for option in rows[0]["signals"]:
            pos = [r["signals"][option] for r, f in zip(rows, failed) if f]
            neg = [r["signals"][option] for r, f in zip(rows, failed) if not f]
            auc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
            print(f"  p({option}){'':<{max(0, 18 - len(option))}} {auc:.2f}")


# ---- fitting the limits on one option ----

def _with_limits(cfg: dict, option: str, limits: dict) -> dict:
    c = copy.deepcopy(cfg)
    for t in c["tiers"]:
        if t["name"] in limits:
            t.setdefault("profile", {}).setdefault("max", {})[option] = limits[t["name"]]
    return c


def fit(rows: list[dict], cfg: dict, constraints: dict, option: str, tolerance: float) -> dict:
    """The limits on p(option), rising with the tier, that keep the pass rate within tolerance
    of always using the last tier at the least spend."""
    allowed = [t["name"] for t in cfg["tiers"] if not policy.excluded(t, constraints)]
    target = summarize(rows, cfg, constraints, f"always:{allowed[-1]}")["pass"] / len(rows) - tolerance
    grid = [round(0.05 * k, 2) for k in range(0, 21)]
    best = None
    for combo in itertools.combinations_with_replacement(grid, len(allowed) - 1):
        limits = dict(zip(allowed[:-1], combo))
        m = summarize(rows, _with_limits(cfg, option, limits), constraints, "router+escalate")
        score = (m["pass"] / len(rows) >= target - 1e-9, m["pass"] if m["pass"] / len(rows) < target else 0, -m["spend"])
        if best is None or score > best[0]:
            best = (score, limits)
    return best[1]


def cross_validate(rows: list[dict], cfg: dict, constraints: dict, option: str, tolerance: float, k: int = 5) -> dict:
    rows = sorted(rows, key=lambda r: r["id"])
    held = []
    for f in range(k):
        train = [r for i, r in enumerate(rows) if i % k != f]
        test = [r for i, r in enumerate(rows) if i % k == f]
        c = _with_limits(cfg, option, fit(train, cfg, constraints, option, tolerance))
        held += [outcome(r, c, constraints, "router+escalate") for r in test]
    return {"pass": sum(o["pass"] for o in held), "n": len(held)}


def measured(rows: list[dict], cfg: dict) -> dict:
    out = {}
    for t in cfg["tiers"]:
        runs = [r["tiers"][t["name"]] for r in rows]
        out[t["name"]] = {"latency_ms": round(statistics.median(r["wall_ms"] for r in runs)),
                          "tokens": {"input": round(statistics.mean(r["input_tokens"] for r in runs)),
                                     "output": round(statistics.mean(r["output_tokens"] for r in runs))}}
    return out


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--tasks", default=os.path.join(here, "tasks", "eval.jsonl"))
    ap.add_argument("--only", help="comma-separated task IDs")
    ap.add_argument("--reuse", help="analyse stored rows from an earlier run instead of running the models")
    ap.add_argument("--fit", metavar="OPTION", help="suggest the tiers' limits on p(OPTION), for example hard")
    ap.add_argument("--tolerance", type=float, default=0.0, help="pass rate --fit may give up, for example 0.05")
    ap.add_argument("--in-cluster-only", action="store_true")
    ap.add_argument("--latency-budget", type=float, metavar="MS")
    ap.add_argument("--profile", help="measured profile to merge into the tiers, for example results/measured-bedrock.yaml")
    ap.add_argument("--workers", type=int, default=4, help="tasks run in parallel, which also affects the measured time")
    ap.add_argument("--clm-url", default=os.environ.get("CLM_BASE_URL", "http://localhost:8700"))
    ap.add_argument("--out", default=os.path.join(here, "results"))
    args = ap.parse_args()

    cfg = load_tiers(args.tiers, args.profile)
    constraints = {**cfg.get("policy", {})}
    if args.in_cluster_only:
        constraints["in_cluster_only"] = True
    if args.latency_budget is not None:
        constraints["latency_budget_ms"] = args.latency_budget

    if args.reuse:
        rows = [json.loads(line) for line in open(args.reuse) if line.strip()]
        specs = {t["id"]: t["check"] for t in map(json.loads, filter(str.strip, open(args.tasks)))}
        for row in rows:   # check the stored answers again, so a corrected check needs no new runs
            for r in row["tiers"].values():
                r["pass"], r["fails"] = check(specs[row["id"]], SimpleNamespace(**r))
    else:
        if not CLM(args.clm_url).healthy():
            sys.exit(f"clm-serve at {args.clm_url} is not ready")
        tasks = [json.loads(line) for line in open(args.tasks) if line.strip()]
        if args.only:
            keep = set(args.only.split(","))
            tasks = [t for t in tasks if t["id"] in keep]
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"eval-{cfg['name']}.jsonl")
        rows = []
        with open(path, "w") as f, ThreadPoolExecutor(args.workers) as pool:
            for row in pool.map(lambda t: run_one(t, cfg, args.clm_url), tasks):
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{row['id']:20} " + "  ".join(f"{n} {'pass' if r['pass'] else 'FAIL'}"
                                                     for n, r in row["tiers"].items()))
        with open(os.path.join(args.out, f"measured-{cfg['name']}.yaml"), "w") as f:
            yaml.safe_dump(measured(rows, cfg), f, sort_keys=False)
        print(f"\nrows written to {path}")

    report(rows, cfg, constraints)
    if args.fit:
        limits = fit(rows, cfg, constraints, args.fit, args.tolerance)
        c = _with_limits(cfg, args.fit, limits)
        m = summarize(rows, c, constraints, "router+escalate")
        cv = cross_validate(rows, cfg, constraints, args.fit, args.tolerance)
        print(f"\n--fit {args.fit}: set profile.max.{args.fit} to {limits}")
        print(f"  router+escalate on all tasks: {m['pass']}/{m['n']}; 5-fold cross-validated: {cv['pass']}/{cv['n']}")
        print("  the cross-validated figure is the one to trust; the other is fitted on the tasks it is scored on")


if __name__ == "__main__":
    main()
