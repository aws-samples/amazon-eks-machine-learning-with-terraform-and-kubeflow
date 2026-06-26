#!/usr/bin/env bash
# Scenario 3: Failed deployment rollout (ImagePullBackOff on a bad tag).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "SCENARIO 3: Failed deployment rollout"

echo "[1/3] Deploying healthy payment-service"
ensure_namespace
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-service
  namespace: $DEMO_NAMESPACE
spec:
  replicas: 3
  selector:
    matchLabels:
      app: payment-service
  template:
    metadata:
      labels:
        app: payment-service
    spec:
      containers:
        - name: app
          image: public.ecr.aws/nginx/nginx:1.25
          ports:
            - containerPort: 80
          readinessProbe:
            httpGet: { path: /, port: 80 }
            initialDelaySeconds: 5
            periodSeconds: 5
EOF

echo "[2/3] Waiting for first rollout to be Available"
kubectl wait --for=condition=Available deployment/payment-service -n "$DEMO_NAMESPACE" --timeout=120s

echo "[3/3] Pushing bad image tag nginx:99.99.99-nonexistent"
kubectl set image deployment/payment-service app=public.ecr.aws/nginx/nginx:99.99.99-nonexistent -n "$DEMO_NAMESPACE"

echo "Waiting 20s for ImagePullBackOff..."
sleep 20

echo
kubectl get pods -n "$DEMO_NAMESPACE" -l app=payment-service
echo
kubectl rollout status deployment/payment-service -n "$DEMO_NAMESPACE" --timeout=5s 2>&1 || true

prompt "Deployment payment-service in namespace $DEMO_NAMESPACE (cluster $EKS_CLUSTER_NAME) is stuck during a rollout. New pods are not starting successfully. Investigate what went wrong and recommend whether to rollback or fix forward."
