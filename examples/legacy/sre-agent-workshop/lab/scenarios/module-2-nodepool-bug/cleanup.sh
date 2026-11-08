#!/usr/bin/env bash
# Restore the cluster to its Module 2 baseline: delete the Overpass workload
# and revert the NodePool to its pre-injection state.
#
# The snapshot is what inject.sh saved at /tmp/sre-m2-nodepool-original.yaml.
# If it's missing (agent already rolled back, or a different desktop),
# we re-add the 'workload' requirement manually.

set -uo pipefail

NAMESPACE="${SRE_M2_NAMESPACE:-data-services}"
NODEPOOL="${SRE_M2_NODEPOOL:-cpu-compute}"
SNAPSHOT=/tmp/sre-m2-nodepool-original.yaml

echo "──────────────────────────────────────────────────────────────"
echo "  Module 2 — cleanup"
echo "──────────────────────────────────────────────────────────────"

# ─── Delete workload ───────────────────────────────────────────────────────
if kubectl get ns "${NAMESPACE}" >/dev/null 2>&1; then
    echo "[info] deleting overpass-api deployment"
    kubectl delete deployment -n "${NAMESPACE}" overpass-api --ignore-not-found
    echo "[info] deleting namespace ${NAMESPACE}"
    kubectl delete namespace "${NAMESPACE}" --ignore-not-found
fi

# ─── Restore NodePool ──────────────────────────────────────────────────────
if [[ -f "${SNAPSHOT}" ]]; then
    echo "[info] restoring NodePool from ${SNAPSHOT}"
    kubectl apply -f "${SNAPSHOT}" >/dev/null
else
    echo "[warn] snapshot not found — re-adding 'workload' requirement manually"
    kubectl get nodepool.karpenter.sh "${NODEPOOL}" -o json 2>/dev/null | python3 -c "
import json, sys
np = json.load(sys.stdin)
reqs = np['spec']['template']['spec'].setdefault('requirements', [])
if not any(r.get('key') == 'workload' for r in reqs):
    reqs.append({
        'key': 'workload',
        'operator': 'In',
        'values': ['data-services'],
    })
print(json.dumps(np))
" | kubectl apply -f - >/dev/null
fi

rm -f /tmp/sre-m2-overpass.yaml
echo ""
echo "[OK] cluster restored to Module 2 baseline"
