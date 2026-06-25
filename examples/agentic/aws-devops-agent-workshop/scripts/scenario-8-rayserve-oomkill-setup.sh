#!/usr/bin/env bash
# Track B Scenario 8: Ray Serve worker OOMKill (Helm memory limit too low).
#
# Story: a Ray Serve worker pod (in production: a vLLM inference replica) is
# repeatedly OOMKilled because its actual memory usage exceeds the
# resources.limits.memory configured in the Helm values.
#
# This script uses a LIGHTWEIGHT MOCK: a plain Deployment labeled like a Ray
# Serve worker, with a Python bytearray allocation that exceeds its
# `resources.limits.memory`. The agent's investigation path (pod termination
# reason → memory time-series → Helm values fix) is identical to what it
# would do on a real Ray Serve vLLM deployment.
#
# Real-world inspiration:
#   examples/inference/rayserve/meta-llama3-8b-vllm/rayservice.yaml
#   chart: charts/machine-learning/serving/rayserve/
#   knob:  resources.limits.memory on the worker spec
#
# Prereq: CloudWatch Observability EKS add-on with metrics populated for >5
# minutes (Container Insights provides pod_memory_working_set the agent uses).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

NAMESPACE="ml-inference"

banner "SCENARIO 8: Ray Serve worker OOMKill (Helm memory limit too low)"
echo "Track B — ML Inference Ops"
echo

echo "[1/3] Ensuring namespace $NAMESPACE"
kubectl --context "$EKS_CLUSTER_NAME" create namespace "$NAMESPACE" 2>/dev/null || true

echo "[2/3] Deploying mock Ray Serve worker with UNDERSIZED memory limit"
echo "      Allocates 256MiB against resources.limits.memory = 128Mi"
kubectl --context "$EKS_CLUSTER_NAME" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nova-vision-model
  namespace: $NAMESPACE
  labels:
    ray.io/cluster: nova-vision-model
    ray.io/node-type: worker
    ray.io/group: vision-model-worker
    app: ray-serve-vision-model
    ml-platform/scenario: "8"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ray-serve-vision-model
  template:
    metadata:
      labels:
        ray.io/cluster: nova-vision-model
        ray.io/node-type: worker
        ray.io/group: vision-model-worker
        app: ray-serve-vision-model
    spec:
      containers:
        - name: ray-worker
          image: public.ecr.aws/docker/library/python:3.12-slim
          command: ["python","-c"]
          args:
            - |
              # Mock of a vLLM inference replica loading model weights into RAM.
              # In the real meta-llama3-8b-vllm example this would be the
              # vLLM engine init step. Here we allocate 256MiB against a 128Mi
              # limit so the kernel OOMKills predictably.
              data = bytearray(256 * 1024 * 1024)
              import time
              while True:
                  time.sleep(60)
          resources:
            requests:
              cpu: "200m"
              memory: "64Mi"
            limits:
              cpu: "500m"
              memory: "128Mi"   # <-- the bug we want the agent to find
EOF

echo "[3/3] Waiting up to 90s for OOMKills to appear..."
for _ in $(seq 1 18); do
  if kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods -l app=ray-serve-vision-model \
       -o jsonpath='{range .items[*]}{.status.containerStatuses[*].lastState.terminated.reason}{"\n"}{end}' \
       2>/dev/null | grep -q OOMKilled; then
    break
  fi
  sleep 5
done

echo
echo "Pods:"
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods -l app=ray-serve-vision-model -o wide

prompt "Ray Serve worker pods for our nova-vision-model deployment in namespace $NAMESPACE (cluster $EKS_CLUSTER_NAME) are repeatedly OOMKilled. This is a Ray Serve inference workload deployed via the charts/machine-learning/serving/rayserve/ Helm chart, pattern similar to examples/inference/rayserve/meta-llama3-8b-vllm. Analyze the memory utilization from CloudWatch Container Insights, identify why the resource limits are insufficient, and tell me the exact Helm values change to fix it."

echo
echo "Agent should correlate:"
echo "  • Pod termination reason: exit 137 / OOMKilled"
echo "  • pod_memory_working_set vs the 128Mi limit"
echo "  • Recommendation: raise resources.limits.memory in the Helm values"
echo "    on the worker spec (e.g. 320Mi for 256MiB workload + headroom)"
