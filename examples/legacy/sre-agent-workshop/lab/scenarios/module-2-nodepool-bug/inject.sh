#!/usr/bin/env bash
# Module 2 fault: patch the data-services NodePool to strip a critical
# requirement, then deploy an Overpass API workload whose nodeSelector
# depends on that requirement. Result: Karpenter sees NO NodePool that
# matches, so it never creates a NodeClaim. Pod stays Pending forever.
#
# The symptom (pod Pending) is identical to Module 1's GPU-quota failure.
# The root cause is different — and this one IS fixable at the cluster level
# by patching the NodePool. The M2 agent's job is to correctly identify
# the fix, propose it, get approval, apply it, verify, and rollback if
# verification fails.

set -euo pipefail

NAMESPACE="${SRE_M2_NAMESPACE:-data-services}"
NODEPOOL="${SRE_M2_NODEPOOL:-cpu-compute}"

echo "──────────────────────────────────────────────────────────────"
echo "  Module 2 — NodePool selector-mismatch fault injection"
echo "──────────────────────────────────────────────────────────────"
echo "  Namespace: ${NAMESPACE}"
echo "  NodePool:  ${NODEPOOL}"
echo ""

# ─── Ensure namespace exists ───────────────────────────────────────────────
if ! kubectl get ns "${NAMESPACE}" >/dev/null 2>&1; then
    echo "[info] creating namespace ${NAMESPACE}"
    kubectl create namespace "${NAMESPACE}"
fi

# ─── Snapshot the NodePool before mutation ─────────────────────────────────
# The M2 agent will snapshot again when it decides to patch. We snapshot
# here too so the cleanup script has a reliable restore target.
SNAPSHOT=/tmp/sre-m2-nodepool-original.yaml
kubectl get nodepool.karpenter.sh "${NODEPOOL}" -o yaml \
  | grep -v -E "resourceVersion:|uid:|generation:|creationTimestamp:|selfLink:" \
  > "${SNAPSHOT}"
echo "[info] original NodePool snapshotted to ${SNAPSHOT}"

# ─── Patch the NodePool: remove the workload=data-services requirement ─────
# The Karpenter chart's default NodePool advertises capacity for several
# workload classes. Stripping the data-services requirement means no
# NodePool will match a pod that requests `workload: data-services` via
# nodeSelector — but the pod won't know why. It just says FailedScheduling.
kubectl get nodepool.karpenter.sh "${NODEPOOL}" -o json \
  | python3 -c "
import json, sys
np = json.load(sys.stdin)
reqs = np['spec']['template']['spec'].get('requirements', [])
filtered = [r for r in reqs if r.get('key') != 'workload']
np['spec']['template']['spec']['requirements'] = filtered
print(json.dumps(np))
" \
  | kubectl apply -f -

echo "[info] NodePool patched — 'workload' requirement removed"

# ─── Apply the Overpass API deployment ─────────────────────────────────────
# CPU-only workload — no GPU dependency. Its nodeSelector requests
# `workload: data-services`, which the (now-broken) NodePool no longer
# advertises.
cat > /tmp/sre-m2-overpass.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: overpass-api
  namespace: ${NAMESPACE}
  labels:
    app: overpass-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: overpass-api
  template:
    metadata:
      labels:
        app: overpass-api
    spec:
      containers:
        - name: overpass-api
          image: nginx:1.27-alpine
          command: ["/bin/sh", "-c", "sleep infinity"]
          resources:
            limits:
              cpu: "500m"
              memory: "512Mi"
            requests:
              cpu: "200m"
              memory: "256Mi"
      nodeSelector:
        workload: data-services
EOF
kubectl apply -f /tmp/sre-m2-overpass.yaml

echo ""
echo "[info] Overpass API deployment applied. Pod should stay Pending"
echo "       because no NodePool matches its nodeSelector."
echo ""
echo "[next] ./verify.sh    # confirm the failure state"
echo "[next] python3 ~/sre-agent/sre_agent_v2.py --scenario data_service_remediation \\"
echo "         --prompt \"Overpass API pods stuck Pending in ${NAMESPACE}. Investigate.\""
