#!/usr/bin/env bash
# Run loadgen.py against the Dynamo frontend from inside the cluster.
#
# In-cluster on purpose. Driving load through `kubectl port-forward` routes every request
# through a single userspace tunnel on the workstation, which saturates long before two GPUs
# do and would flatten exactly the throughput differences this is meant to measure.
#
#   ./run-bench.sh <tag> [extra loadgen args...]
#
# Example:
#   ./run-bench.sh disagg --concurrency 1,4,16,32 --requests 64
set -euo pipefail

NS="${NS:-kubeflow-user-example-com}"
FRONTEND="${FRONTEND:-dyn-qwen3-8b-frontend}"
MODEL="${MODEL:-qwen3-8b}"
# ECR Public rather than Docker Hub: no anonymous pull rate limit.
IMAGE="${IMAGE:-public.ecr.aws/docker/library/python:3.11-slim}"
TAG="${1:?usage: run-bench.sh <tag> [loadgen args...]}"
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="dyn-bench-${TAG}"

kubectl create configmap dyn-loadgen -n "$NS" \
  --from-file=loadgen.py="$HERE/loadgen.py" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

kubectl delete job "$JOB" -n "$NS" --ignore-not-found >/dev/null

# backoffLimit 0: a crashed run should surface as a failure, not silently retry and emit a
# second set of numbers into the same log.
kubectl apply -f - <<YAML >/dev/null
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB
  namespace: $NS
spec:
  backoffLimit: 0
  template:
    metadata:
      annotations:
        sidecar.istio.io/inject: 'false'
    spec:
      restartPolicy: Never
      containers:
        - name: loadgen
          image: $IMAGE
          command: ["python3", "/opt/bench/loadgen.py"]
          args:
            - --host
            - $FRONTEND
            - --port
            - "8000"
            - --model
            - $MODEL
            - --tag
            - "$TAG"
$(for a in "$@"; do printf '            - %s\n' "\"$a\""; done)
          volumeMounts:
            - name: bench
              mountPath: /opt/bench
          # Sized to fit a 2-vCPU CPU node. The generator only parses small SSE frames -- at
          # 32 concurrent streams that is on the order of a thousand json.loads per second,
          # which is nowhere near CPU bound, so it is not the thing under measurement.
          resources:
            requests:
              cpu: "1"
              memory: 1Gi
            limits:
              cpu: "2"
              memory: 2Gi
      volumes:
        - name: bench
          configMap:
            name: dyn-loadgen
YAML

echo "waiting for $JOB ..."
kubectl wait --for=condition=complete "job/$JOB" -n "$NS" --timeout=3600s &
wait_pid=$!
kubectl wait --for=condition=failed "job/$JOB" -n "$NS" --timeout=3600s && \
  { echo "JOB FAILED"; kubectl logs -n "$NS" "job/$JOB" --tail=50; exit 1; } &
fail_pid=$!
wait -n "$wait_pid" "$fail_pid" 2>/dev/null || true
kill "$wait_pid" "$fail_pid" 2>/dev/null || true

kubectl logs -n "$NS" "job/$JOB"
