"""How far a KAI deployment's real traffic, and its latency under it, has moved from the fleet file.

The search was run on assumptions: each tenant's `demand` (rate, prompt and output lengths) and the
`targets` it had to hold under that demand. A deployment then meets whatever traffic arrives. This
records, hour by hour and per tenant, both kinds of drift side by side, because neither reads alone:

  - **demand drift**, the traffic against the fleet file's `demand`. Usually the cause, and the
    thing to change in the fleet file before running the pipeline again;
  - **latency drift**, ITL mean and TTFT p95 against the `targets` and against what the search
    certified. The effect, and whether the drift matters yet.

It decides nothing: no alert, no threshold, no rerun. The record is for a person to read.

**The figures are vLLM's own**, scraped by the cluster Prometheus from each tenant's `/metrics`
(the `kai-vllm` chart's ServiceMonitor). ITL is `vllm:request_time_per_output_token_seconds`, each
request's own mean gap between tokens, so its mean over requests is the harness's `itl_mean_ms`.
TTFT p95 is read off the `vllm:time_to_first_token_seconds` histogram, interpolated between bucket
edges, and the buckets are coarse above 250 ms (0.25, 0.5, 0.75, 1, 2.5 s): a trend, not the
harness's figure. Both are timed in the server rather than the client, which inside the cluster
differs by the network hop.

**An hour is one window**: long enough to hold enough requests, short enough that a daily peak is a
window of its own -- and the peak is what the fleet file's rate has to describe. A window with no
requests is `no traffic`, not zero drift; one with fewer than `MIN_REQUESTS` gets no p95, which a
handful of requests cannot give.

Pure: the queries are built and the answers turned into records here; `cluster.py` asks Prometheus.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

from . import plan as kp

WINDOW_S = 3600
# A p95 is the latency 95 of 100 requests beat; with fewer than 20 there is no 95th one to read.
MIN_REQUESTS = 20
# An hour is recorded only once it is this far in the past, so its last scrapes have landed.
SETTLE_S = 300
# Prometheus keeps 30 days; older hours are gone and are not asked for.
RETENTION_S = 30 * 24 * 3600

SCRAPE_INTERVAL = "15s"
TENANT = "colocation_kai_tenant"          # LABEL_TENANT as Prometheus spells a target label
DEPLOYMENT = "colocation_kai_deployment"  # LABEL_DEPLOYMENT likewise
RECORD_PREFIX = "colocation_kai"

MEASURED, THIN, IDLE = "measured", "too few requests", "no traffic"


def selector(namespace: str, name: str) -> str:
    return f'{{namespace="{namespace}",{DEPLOYMENT}="{kp.dns(name)}"}}'


def expressions(namespace: str, name: str, window: str = "1h") -> dict:
    """PromQL for one window per tenant, by figure. The one copy: the chart's recording rules and
    `drift` both use it, so a graph and a record cannot disagree on what a figure is."""
    sel, by = selector(namespace, name), f"sum by ({TENANT})"

    def mean(metric):
        return (f"{by} (increase({metric}_sum{sel}[{window}])) / "
                f"{by} (increase({metric}_count{sel}[{window}]))")

    return {
        "requests": f"{by} (increase(vllm:e2e_request_latency_seconds_count{sel}[{window}]))",
        "isl_tokens": mean("vllm:request_prompt_tokens"),
        "osl_tokens": mean("vllm:request_generation_tokens"),
        "itl_mean_s": mean("vllm:request_time_per_output_token_seconds"),
        "ttft_p95_s": (f"histogram_quantile(0.95, sum by ({TENANT}, le) "
                       f"(increase(vllm:time_to_first_token_seconds_bucket{sel}[{window}])))"),
    }


def recording_rules(namespace: str, name: str) -> list:
    """The same figures over a rolling hour, for graphing. `drift` does not read them."""
    return [{"record": f"{RECORD_PREFIX}:{figure}:1h", "expr": expr}
            for figure, expr in expressions(namespace, name).items()]


def window_ends(deployed_at: datetime, last_end: datetime | None, now: datetime) -> list:
    """The hour ends still to record: whole clock hours after the deploy, after the last one
    recorded, settled, and inside Prometheus's retention."""
    first = _ceil_hour(deployed_at) + timedelta(seconds=WINDOW_S)
    if last_end is not None:
        first = max(first, last_end + timedelta(seconds=WINDOW_S))
    first = max(first, _ceil_hour(now - timedelta(seconds=RETENTION_S)))
    last = _floor_hour(now - timedelta(seconds=SETTLE_S))
    out, t = [], first
    while t <= last:
        out.append(t)
        t += timedelta(seconds=WINDOW_S)
    return out


def references(doc: dict, result: str) -> dict:
    """Per tenant, what each figure is held against: the fleet file's demand and targets (copied
    into the export) and what the search certified for the result being served."""
    per = (doc.get(result) or {}).get("per_tenant", {})
    out = {}
    for n, t in doc["tenants"].items():
        cert = per.get(n, {})
        out[n] = {
            "demand": {k: t["demand"].get(k) for k in ("rate_hz", "isl_tokens", "osl_tokens")},
            "targets": {"itl_mean_ms": t["targets"].get("itl_mean_ms"),
                        "ttft_p95_ms": t["targets"].get("ttft_p95_ms"),
                        "tolerance_pct": t["targets"].get("tolerance_pct")},
            # A baseline holds only the solo ITL: its TTFT was never certified.
            "certified": {"itl_mean_ms": cert.get("itl_mean_ms", cert.get("solo_itl_ms")),
                          "ttft_p95_ms": cert.get("ttft_p95_ms")},
        }
    return out


def series(answer: dict) -> dict:
    """A query_range answer as {tenant: {unix end: value}}, NaN (0/0, an idle hour) as None."""
    out = {}
    for s in answer.get("data", {}).get("result", []):
        tenant = s["metric"].get(TENANT)
        for ts, v in s.get("values", []):
            x = float(v)
            out.setdefault(tenant, {})[int(float(ts))] = None if not math.isfinite(x) else x
    return out


def records(plan: kp.Plan, refs: dict, figures: dict, ends: list) -> list:
    """One record per tenant per window. `figures` is {figure: series(...)}, by tenant label."""
    out = []
    for end in ends:
        ts = int(end.timestamp())
        for n in plan.order:
            got = {f: figures.get(f, {}).get(kp.dns(n), {}).get(ts) for f in figures}
            requests = got.get("requests") or 0.0
            status = IDLE if requests < 0.5 else THIN if requests < MIN_REQUESTS else MEASURED
            ref = refs[n]
            idle = status == IDLE
            rate = None if idle else requests / WINDOW_S
            itl = None if idle or got.get("itl_mean_s") is None else got["itl_mean_s"] * 1000
            ttft = (None if status != MEASURED or got.get("ttft_p95_s") is None
                    else got["ttft_p95_s"] * 1000)
            out.append({
                "deployment": plan.name, "tenant": n, "served_name": plan.tenants[n].served_name,
                "window_start": _iso(end - timedelta(seconds=WINDOW_S)), "window_end": _iso(end),
                "export": plan.export_path, "export_sha256": plan.export_sha256,
                "result": plan.result, "status": status, "requests": round(requests),
                "demand": {
                    "rate_hz": _vs(rate, fleet=ref["demand"]["rate_hz"]),
                    "isl_tokens": _vs(None if idle else got.get("isl_tokens"),
                                      fleet=ref["demand"]["isl_tokens"]),
                    "osl_tokens": _vs(None if idle else got.get("osl_tokens"),
                                      fleet=ref["demand"]["osl_tokens"])},
                "latency": {
                    "itl_mean_ms": _vs(itl, target=ref["targets"]["itl_mean_ms"],
                                       certified=ref["certified"]["itl_mean_ms"]),
                    "ttft_p95_ms": _vs(ttft, target=ref["targets"]["ttft_p95_ms"],
                                       certified=ref["certified"]["ttft_p95_ms"]),
                    "tolerance_pct": ref["targets"]["tolerance_pct"]},
            })
    return out


def last_end(lines: list) -> datetime | None:
    ends = [json.loads(line)["window_end"] for line in lines if line.strip()]
    return max((datetime.fromisoformat(e) for e in ends), default=None)


def describe(recs: list) -> str:
    """One line per tenant per window, for a person."""
    out = []
    for r in recs:
        head = f"{r['window_start'][:16]}Z {r['tenant']}: {r['status']}"
        if r["status"] == IDLE:
            out.append(head)
            continue
        d, lat = r["demand"], r["latency"]
        parts = [f"{r['requests']} req, {_fig(d['rate_hz'], 'fleet', 'rate', '/s', 3)}",
                 _fig(d["isl_tokens"], "fleet", "in", " tok", 0),
                 _fig(d["osl_tokens"], "fleet", "out", " tok", 0),
                 _fig(lat["itl_mean_ms"], "target", "ITL", " ms", 1),
                 _fig(lat["ttft_p95_ms"], "target", "TTFT p95", " ms", 0)]
        out.append(head + "  " + "; ".join(p for p in parts if p))
    return "\n".join(out)


def _vs(observed, **refs) -> dict:
    out = {"observed": None if observed is None else round(observed, 4)}
    for k, ref in refs.items():
        out[k] = ref
        out[f"vs_{k}_pct"] = (None if observed is None or not ref
                              else round((observed - ref) / ref * 100, 1))
    return out


def _fig(v: dict, ref: str, label: str, unit: str, nd: int) -> str:
    if v["observed"] is None:
        return ""
    pct = v.get(f"vs_{ref}_pct")
    return (f"{label} {v['observed']:.{nd}f}{unit}"
            + ("" if pct is None else f" ({pct:+.0f}% vs {ref} {v[ref]})"))


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat()


def _floor_hour(t: datetime) -> datetime:
    return t.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _ceil_hour(t: datetime) -> datetime:
    f = _floor_hour(t)
    return f if f == t.astimezone(timezone.utc) else f + timedelta(hours=1)
