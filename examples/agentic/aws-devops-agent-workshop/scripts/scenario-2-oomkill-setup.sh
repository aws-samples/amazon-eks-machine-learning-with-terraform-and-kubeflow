#!/usr/bin/env bash
# Scenario 2: OOMKill — 256Mi workload constrained to 128Mi limit.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "SCENARIO 2: OOMKill (memory misconfiguration)"

echo "[1/3] Ensuring namespace $DEMO_NAMESPACE"
ensure_namespace

echo "[2/3] Deploying leaky-service (256Mi workload, 128Mi limit)"
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: leaky-service
  namespace: $DEMO_NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: leaky-service
  template:
    metadata:
      labels:
        app: leaky-service
    spec:
      containers:
        - name: leaker
          image: public.ecr.aws/docker/library/python:3.12-slim
          command: ["python", "-c"]
          args:
            - |
              # Allocate ~256MB of RSS to trigger OOMKill against the 128Mi limit.
              data = bytearray(256 * 1024 * 1024)
              import time
              while True:
                  time.sleep(60)
          resources:
            requests:
              memory: "64Mi"
              cpu: "50m"
            limits:
              memory: "128Mi"
              cpu: "200m"
EOF

echo "[3/3] Waiting up to 60s for OOMKills to appear..."
for _ in $(seq 1 12); do
  if kubectl get pods -n "$DEMO_NAMESPACE" -l app=leaky-service \
       -o jsonpath='{.items[*].status.containerStatuses[*].lastState.terminated.reason}' \
       2>/dev/null | grep -q OOMKilled; then
    break
  fi
  sleep 5
done

echo
kubectl get pods -n "$DEMO_NAMESPACE" -l app=leaky-service

prompt "Pods in deployment leaky-service in namespace $DEMO_NAMESPACE (cluster $EKS_CLUSTER_NAME) are being OOMKilled repeatedly. Analyze the memory utilization from CloudWatch Container Insights, identify why the resource limits are insufficient, and recommend the correct resource configuration."
