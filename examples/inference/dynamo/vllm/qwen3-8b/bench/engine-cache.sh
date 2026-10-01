#!/usr/bin/env bash
# Print each worker's ENGINE-side prefix cache hit rate, in tokens -- how many prompt tokens vLLM
# actually skipped prefilling. Reported in every routing mode, unlike the frontend metrics in
# router-metrics.sh, which exist only under --router-mode kv.
#
# That makes this the honest denominator for a KV-routing claim: round-robin can reach the same
# engine hit rate by accident when every worker ends up caching every prefix, in which case the
# routing is correct and buys nothing.
#
#   ./engine-cache.sh > before.txt   # ... run a benchmark ...
#   ./engine-cache.sh > after.txt ; diff before.txt after.txt
#
# Counters are cumulative since worker start, so take a delta.
set -euo pipefail

NS="${NS:-kubeflow-user-example-com}"
WORKER="${WORKER:-dyn-qwen3-8b-vllmworker}"
# The worker serves vLLM's Prometheus registry here, not on the frontend's 8000.
PORT="${PORT:-9090}"

mapfile -t PODS < <(kubectl get pod -n "$NS" -o name | grep "$WORKER" | sed 's|pod/||')
[ "${#PODS[@]}" -gt 0 ] || { echo "no worker pods matching '$WORKER' in '$NS'" >&2; exit 1; }

for pod in "${PODS[@]}"; do
  kubectl exec -n "$NS" "$pod" -c main -- python3 -c "
import urllib.request
raw = urllib.request.urlopen('http://localhost:$PORT/metrics').read().decode()
v = {}
for line in raw.splitlines():
    if line.startswith('#'):
        continue
    head, _, val = line.rpartition(' ')
    name = head.split('{', 1)[0]
    if name in ('vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total'):
        v[name] = v.get(name, 0.0) + float(val)
q = v.get('vllm:prefix_cache_queries_total', 0.0)
h = v.get('vllm:prefix_cache_hits_total', 0.0)
print('%-56s queried %12d  hit %12d  %s' % (
    '$pod', q, h, ('%.1f%%' % (100.0 * h / q)) if q else 'n/a'))
"
done
