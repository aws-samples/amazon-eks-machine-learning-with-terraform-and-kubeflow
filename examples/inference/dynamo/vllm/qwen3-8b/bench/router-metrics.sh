#!/usr/bin/env bash
# Print the Dynamo frontend's router state: registered workers, routing decisions, and how much
# KV-cache overlap the router found.
#
# A latency table cannot tell you whether KV-aware routing did anything: --router-mode kv is
# accepted on a single-worker deployment, with prefix caching off, or on a workload that shares no
# prefixes, and behaves like round-robin while reporting nothing wrong. This is the check.
#
#   ./router-metrics.sh                 # snapshot
#   ./router-metrics.sh > before.txt    # ... run a benchmark ...
#   ./router-metrics.sh > after.txt ; diff before.txt after.txt
#
# Reads /metrics from inside the frontend pod, so it is safe to run mid-benchmark.
set -euo pipefail

NS="${NS:-kubeflow-user-example-com}"
FRONTEND="${FRONTEND:-dyn-qwen3-8b-frontend}"

POD="$(kubectl get pod -n "$NS" -l "app=$FRONTEND" -o name 2>/dev/null | head -1)"
if [ -z "$POD" ]; then
  # The operator's label set varies between releases; fall back to the name prefix.
  POD="$(kubectl get pod -n "$NS" -o name | grep -m1 "$FRONTEND" || true)"
fi
[ -n "$POD" ] || { echo "no frontend pod matching '$FRONTEND' in namespace '$NS'" >&2; exit 1; }

kubectl exec -n "$NS" "${POD#pod/}" -c main -- python3 -c '
import urllib.request, collections

raw = urllib.request.urlopen("http://localhost:8000/metrics").read().decode()

def samples(name):
    out = []
    for line in raw.splitlines():
        if line.startswith("#") or not line.startswith(name):
            continue
        head, _, value = line.rpartition(" ")
        if not head.startswith(name + "{") and head != name:
            continue
        labels = {}
        if "{" in head:
            for kv in head[head.index("{") + 1:head.rindex("}")].split("\","):
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                labels[k.strip()] = v.strip().strip("\"")
        out.append((labels, float(value)))
    return out

def one(name, default=0.0):
    s = samples(name)
    return s[0][1] if s else default

print("registered workers (router_worker_registered == 1):")
for labels, v in samples("dynamo_component_router_worker_registered"):
    if v == 1:
        print("  worker %-20s type=%s dp_rank=%s"
              % (labels.get("router_worker_id"), labels.get("worker_type"),
                 labels.get("dp_rank")))
print("  NOTE: this gauge is not cleared when a worker goes away. After a rolling update it")
print("        can list more workers than exist. Cross-check against `kubectl get pod`.")

decisions = one("dynamo_component_router_kv_hit_rate_count")
overlap = one("dynamo_component_router_kv_hit_rate_sum")
print()
print("block size            : %d tokens" % one("dynamo_frontend_model_kv_cache_block_size"))
print("KV blocks per worker  : %d" % one("dynamo_frontend_model_total_kv_blocks"))
print("routing decisions     : %d" % decisions)
print("mean prefix overlap   : %s"
      % ("%.4f" % (overlap / decisions) if decisions else "n/a (no requests yet)"))
print("KV event batches recvd: %d"
      % sum(v for _, v in samples("dynamo_component_router_kv_zmq_ingress_batches_total")))
applied = collections.Counter()
for labels, v in samples("dynamo_component_kv_cache_events_applied"):
    if v:
        applied[(labels.get("event_type"), labels.get("status"))] += v
print("KV events applied     : %s" % (dict(applied) or "none"))

# The histogram is the part worth reading. A working KV router splits requests into a near-zero
# bucket (full prefill) and a high bucket (prefix already resident); round-robin smears them
# across the middle.
print()
print("overlap distribution (fraction of prompt already cached on the chosen worker):")
prev = 0.0
buckets = sorted(
    ((float(l["le"]), v) for l, v in samples("dynamo_component_router_kv_hit_rate_bucket")
     if l.get("le") not in (None, "+Inf")),
)
for le, cum in buckets:
    n = cum - prev
    prev = cum
    if n:
        print("  <= %-5.2f  %6d  %s" % (le, n, "#" * min(60, int(n))))
'
