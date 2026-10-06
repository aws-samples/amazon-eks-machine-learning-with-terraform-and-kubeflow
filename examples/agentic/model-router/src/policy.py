"""Pick a tier from what the task needs and what each tier offers.

  signals     what the task needs: CLM-8B's answers to the questions in the tiers file,
              flattened to {option: probability}, for example {"easy": 0.8, "hard": 0.2}
  profile     what a tier offers, in the tiers file: the highest probability of each option
              it takes (max), whether it runs in your cluster, and optional prices, tokens
              and latency
  policy      among the tiers that may take the task, which to prefer: the first in the
              file (order), the cheapest (cost) or the fastest (latency)
  constraints per request: only tiers in your cluster, or only tiers within a latency budget

Tiers are listed from smallest to largest. Escalation moves to the next tier in the list
that meets the request's constraints.
"""
from __future__ import annotations


def excluded(tier: dict, constraints: dict) -> str | None:
    """Why the request's constraints rule this tier out, or None."""
    prof = tier.get("profile", {})
    if constraints.get("in_cluster_only") and not prof.get("in_cluster"):
        return "not in the cluster"
    budget, latency = constraints.get("latency_budget_ms"), prof.get("latency_ms")
    if budget is not None and latency is not None and latency > budget:
        return f"latency {latency:.0f} ms > budget {budget:.0f} ms"
    return None


def over_limit(tier: dict, signals: dict) -> str | None:
    """Why this tier should not take the task, or None. A missing option or limit allows it."""
    for option, limit in tier.get("profile", {}).get("max", {}).items():
        p = signals.get(option)
        if p is not None and p > limit:
            return f"p({option}) {p:.2f} > {limit}"
    return None


def est_cost(tier: dict) -> float | None:
    """Price of a typical task on this tier, from price_per_mtok and tokens; None without prices.

    Without measured tokens it assumes 1,000 input and 500 output tokens per task.
    """
    prof = tier.get("profile", {})
    price, tokens = prof.get("price_per_mtok") or {}, prof.get("tokens") or {}
    if price.get("input") is None or price.get("output") is None:
        return None
    return (price["input"] * (tokens.get("input") or 1000) + price["output"] * (tokens.get("output") or 500)) / 1e6


def _key(tier: dict, prefer: str) -> float | None:
    if prefer == "cost":
        return est_cost(tier)
    if prefer == "latency":
        return tier.get("profile", {}).get("latency_ms")
    raise ValueError(f"unknown policy.prefer {prefer!r}; use order, cost or latency")


def choose(tiers: list[dict], signals: dict, policy: dict, constraints: dict) -> tuple[int, list[str]]:
    """Return the first tier for a task and the reasons."""
    reasons = []
    allowed = []
    for i, t in enumerate(tiers):
        why = excluded(t, constraints)
        if why:
            reasons.append(f"{t['name']}: {why}")
        else:
            allowed.append(i)
    if not allowed:
        raise ValueError("no tier meets the request's constraints")
    able = []
    for i in allowed:
        why = over_limit(tiers[i], signals)
        if why:
            reasons.append(f"{tiers[i]['name']}: {why}")
        else:
            able.append(i)
    if not able:
        reasons.append("no tier is rated for this task; using the largest allowed")
        return allowed[-1], reasons

    prefer = policy.get("prefer", "order")
    if prefer != "order":
        keys = {i: _key(tiers[i], prefer) for i in able}
        if any(k is None for k in keys.values()):
            reasons.append(f"prefer {prefer}: not set for every tier, so using the file order")
        else:
            best = min(able, key=lambda i: (keys[i], i))
            reasons.append(f"prefer {prefer}: {tiers[best]['name']}")
            return best, reasons
    return able[0], reasons


def next_tier(tiers: list[dict], i: int, constraints: dict) -> int | None:
    """The tier to escalate to from tier i, or None when there is none."""
    return next((j for j in range(i + 1, len(tiers)) if not excluded(tiers[j], constraints)), None)
