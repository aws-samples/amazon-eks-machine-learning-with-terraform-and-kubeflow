#!/usr/bin/env bash
# Inject the Module 1 fault — GPU workload that cannot provision.
#
# Deploys Ray Serve pods requesting GPU nodes. The Karpenter 'cuda' NodePool
# will attempt to launch g5/g6 instances. In a Workshop Studio account,
# EC2 refuses RunInstances for GPU instance families, so Karpenter's
# NodeClaims stay Launched=False and Ray pods stay Pending indefinitely.
#
# The failure surfaces four correlated signals:
#   1. Pod FailedScheduling event (misleading — sounds like a label bug)
#   2. Karpenter NodeClaim Launched=False (with capacity/policy reason)
#   3. Karpenter controller logs in CloudWatch (RunInstances denial)
#   4. AWS Service Quotas — GPU vCPU quota == 0
#
# The agent must correlate all four to reach the correct verdict.

set -euo pipefail

NAMESPACE="${SRE_M1_NAMESPACE:-ray-inference}"
RELEASE="${SRE_M1_RELEASE:-ray-serve}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "──────────────────────────────────────────────────────────────"
echo "  Module 1 — GPU workload fault injection"
echo "──────────────────────────────────────────────────────────────"
echo "  Namespace: ${NAMESPACE}"
echo "  Release:   ${RELEASE}"
echo ""

# Ensure namespace exists
if ! kubectl get ns "${NAMESPACE}" >/dev/null 2>&1; then
  echo "[info] creating namespace ${NAMESPACE}"
  kubectl create namespace "${NAMESPACE}"
fi

# We install Ray via a lightweight Deployment manifest rather than helm — the
# real Ray chart carries a lot of extras (services, autoscaler, operator) that
# distract from the scheduling failure. What matters for the workshop is that
# some workload requests GPU nodes and stays Pending. A Deployment does that.
cat > /tmp/sre-m1-workload.yaml <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ray-serve-head
  namespace: NAMESPACE_PLACEHOLDER
  labels:
    app: ray-serve
    role: head
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ray-serve
      role: head
  template:
    metadata:
      labels:
        app: ray-serve
        role: head
    spec:
      containers:
        - name: ray-head
          image: rayproject/ray:2.9.0
          command: ["/bin/sh", "-c", "sleep infinity"]
          resources:
            limits:
              cpu: "2"
              memory: "4Gi"
              nvidia.com/gpu: 1
            requests:
              cpu: "1"
              memory: "2Gi"
              nvidia.com/gpu: 1
      nodeSelector:
        karpenter.sh/nodepool: cuda
      tolerations:
        - key: nvidia.com/gpu
          operator: Equal
          value: "true"
          effect: NoSchedule
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ray-serve-worker
  namespace: NAMESPACE_PLACEHOLDER
  labels:
    app: ray-serve
    role: worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ray-serve
      role: worker
  template:
    metadata:
      labels:
        app: ray-serve
        role: worker
    spec:
      containers:
        - name: ray-worker
          image: rayproject/ray:2.9.0
          command: ["/bin/sh", "-c", "sleep infinity"]
          resources:
            limits:
              cpu: "2"
              memory: "4Gi"
              nvidia.com/gpu: 1
            requests:
              cpu: "1"
              memory: "2Gi"
              nvidia.com/gpu: 1
      nodeSelector:
        karpenter.sh/nodepool: cuda
      tolerations:
        - key: nvidia.com/gpu
          operator: Equal
          value: "true"
          effect: NoSchedule
EOF

# Substitute the namespace and apply.
sed -i.bak "s/NAMESPACE_PLACEHOLDER/${NAMESPACE}/g" /tmp/sre-m1-workload.yaml
rm -f /tmp/sre-m1-workload.yaml.bak
kubectl apply -f /tmp/sre-m1-workload.yaml

echo ""
echo "[info] workload applied. Karpenter should now attempt to provision"
echo "       GPU nodes. In a Workshop Studio account, EC2 will refuse and"
echo "       the NodeClaims will stay Launched=False."
echo ""
echo "[info] give Karpenter ~30 seconds to create NodeClaims and reach the"
echo "       failed-launch state before running the agent."
echo ""
echo "[next] ./verify.sh    # confirm the failure state has surfaced"
echo "[next] python ~/sre-agent/sre_agent_v1.py --scenario gpu_incident \\"
echo "         --prompt \"Ray inference pods stuck Pending in ${NAMESPACE}. Investigate.\""
