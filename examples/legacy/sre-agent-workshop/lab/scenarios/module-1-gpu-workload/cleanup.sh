#!/usr/bin/env bash
# Restore the cluster to its Module 1 baseline (no Ray workload, no leftover
# NodeClaims for GPU nodes).

set -uo pipefail

NAMESPACE="${SRE_M1_NAMESPACE:-ray-inference}"

echo "──────────────────────────────────────────────────────────────"
echo "  Module 1 — cleanup"
echo "──────────────────────────────────────────────────────────────"

# Delete workload
if kubectl get ns "${NAMESPACE}" >/dev/null 2>&1; then
  echo "[info] deleting workload in namespace ${NAMESPACE}"
  kubectl delete deployment -n "${NAMESPACE}" ray-serve-head ray-serve-worker --ignore-not-found
  # Give Karpenter a moment to see the empty NodePool state; disruption policy
  # WhenEmpty on the cuda NodePool will consolidate away failed NodeClaims.
  echo "[info] waiting for Karpenter to reconcile (30s)…"
  sleep 30
  echo "[info] deleting namespace ${NAMESPACE}"
  kubectl delete namespace "${NAMESPACE}" --ignore-not-found
fi

# Drop stray failed NodeClaims that consolidation didn't clean up
stale=$(kubectl get nodeclaims.karpenter.sh -o json 2>/dev/null | \
  python3 -c "
import json, sys
data = json.load(sys.stdin)
for nc in data.get('items', []):
    conds = {c['type']: c['status'] for c in nc.get('status', {}).get('conditions', [])}
    if conds.get('Launched') == 'False':
        print(nc['metadata']['name'])
" || true)

if [[ -n "${stale}" ]]; then
  echo "[info] removing stale failed NodeClaims:"
  echo "${stale}" | sed 's/^/    /'
  echo "${stale}" | xargs -r kubectl delete nodeclaims.karpenter.sh --ignore-not-found
fi

# Remove the tmp manifest
rm -f /tmp/sre-m1-workload.yaml

echo ""
echo "[OK] cluster restored to Module 1 baseline"
