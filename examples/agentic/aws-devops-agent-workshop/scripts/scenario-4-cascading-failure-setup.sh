#!/usr/bin/env bash
# Scenario 4: Cascading failure — multi-signal correlation demo.
# A CPU-hog batch-processor starves order-api + inventory-api on the same node,
# producing symptoms across CloudWatch metrics, K8s events, and node diagnostics.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "SCENARIO 4: Cascading failure (multi-signal)"
echo "Story: A runaway batch-processor consumes node CPU, starving order-api and"
echo "inventory-api. Symptoms span metrics, logs, events, and node diagnostics —"
echo "the agent must correlate across all of them."
echo

echo "[1/4] Deploying victim services (order-api, inventory-api)"
ensure_namespace
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-api
  namespace: $DEMO_NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: order-api
  template:
    metadata:
      labels:
        app: order-api
    spec:
      containers:
        - name: app
          image: public.ecr.aws/nginx/nginx:1.27
          ports:
            - containerPort: 80
          resources:
            requests: { cpu: "100m", memory: "64Mi" }
            limits:   { cpu: "200m", memory: "128Mi" }
          livenessProbe:
            httpGet: { path: /, port: 80 }
            initialDelaySeconds: 5
            periodSeconds: 3
            failureThreshold: 3
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inventory-api
  namespace: $DEMO_NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: inventory-api
  template:
    metadata:
      labels:
        app: inventory-api
    spec:
      containers:
        - name: app
          image: public.ecr.aws/nginx/nginx:1.27
          ports:
            - containerPort: 80
          resources:
            requests: { cpu: "100m", memory: "64Mi" }
            limits:   { cpu: "200m", memory: "128Mi" }
          livenessProbe:
            httpGet: { path: /, port: 80 }
            initialDelaySeconds: 5
            periodSeconds: 3
            failureThreshold: 3
EOF

echo "[2/4] Waiting for victim services to be Ready"
kubectl wait --for=condition=Ready pod -l app=order-api -n "$DEMO_NAMESPACE" --timeout=90s 2>/dev/null || true
kubectl wait --for=condition=Ready pod -l app=inventory-api -n "$DEMO_NAMESPACE" --timeout=90s 2>/dev/null || true

echo "[3/4] Deploying CPU-hungry batch-processor (the culprit)"
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: batch-processor
  namespace: $DEMO_NAMESPACE
spec:
  replicas: 3
  selector:
    matchLabels:
      app: batch-processor
  template:
    metadata:
      labels:
        app: batch-processor
    spec:
      containers:
        - name: cpu-stress
          image: public.ecr.aws/docker/library/python:3.12-slim
          command: ["python", "-c"]
          args:
            - |
              # Burn CPU on 2 threads to starve co-scheduled pods.
              import threading
              def burn():
                  while True:
                      pass
              for _ in range(2):
                  threading.Thread(target=burn, daemon=True).start()
              import time; time.sleep(600)
          resources:
            requests: { cpu: "500m" }
            limits:   { cpu: "1500m" }
EOF

echo "[4/4] Waiting 60s for cascade symptoms to develop..."
sleep 60

echo
echo "Current state:"
kubectl get pods -n "$DEMO_NAMESPACE" -o wide
echo
echo "Node CPU (metrics-server):"
kubectl top nodes 2>/dev/null || echo "(metrics-server not installed — agent will use CloudWatch instead)"

prompt "My order-api and inventory-api services in namespace $DEMO_NAMESPACE (cluster $EKS_CLUSTER_NAME) are experiencing high latency and intermittent failures. Multiple pods show restarts. I'm seeing degraded response times across multiple services simultaneously. Investigate across all available telemetry sources — CloudWatch metrics, container logs, pod events, and node diagnostics. Correlate the signals to identify the root cause and provide a remediation plan."

echo
echo "Agent should correlate:"
echo "  • CloudWatch CPU metrics (node-level spike)"
echo "  • Pod events (throttling, liveness failures, restarts)"
echo "  • MCP node diagnostics (cluster_health, quick_triage)"
echo "  • Container logs / event timeline"
echo "  → Root cause: batch-processor starving co-scheduled pods"
echo "  → Remediation: resource limits, PriorityClass, node affinity / taints"
