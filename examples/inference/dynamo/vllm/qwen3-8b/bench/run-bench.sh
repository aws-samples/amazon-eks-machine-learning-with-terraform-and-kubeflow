#!/usr/bin/env bash
# Run loadgen.py against the Dynamo frontend from inside the cluster.
#
# In-cluster on purpose: `kubectl port-forward` funnels every request through one userspace
# tunnel, which saturates long before two GPUs do and flattens the differences being measured.
#
#   ./run-bench.sh <tag> [extra loadgen args...]
#
# Example:
#   ./run-bench.sh disagg --concurrency 1,4,16,32 --requests 64
#
# Set LOG=<path> to tee the results to a file. Worth doing: completed Job pods are garbage
# collected, and once the pod is gone the numbers are unrecoverable.
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

# backoffLimit 0: a crashed run should fail, not retry and emit a second set of numbers.
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
          # Sized to fit a 2-vCPU CPU node. The generator only parses small SSE frames, so it is
          # nowhere near CPU bound and is not the thing under measurement.
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
# Polled rather than two racing `kubectl wait` calls: the loser of that race survives as a
# grandchild and holds this script's stdout open, so a caller piping run-bench.sh anywhere never
# sees EOF and hangs on a benchmark that already finished.
deadline=$((SECONDS + 3600))
status=""
while [ "$SECONDS" -lt "$deadline" ]; do
  if [ "$(kubectl get job "$JOB" -n "$NS" \
            -o 'jsonpath={.status.conditions[?(@.type=="Complete")].status}')" = "True" ]; then
    status=complete
    break
  fi
  if [ "$(kubectl get job "$JOB" -n "$NS" \
            -o 'jsonpath={.status.conditions[?(@.type=="Failed")].status}')" = "True" ]; then
    status=failed
    break
  fi
  sleep 10
done

# Fetch the log before anything else: the Job outlives its pod, and once the pod is collected the
# results are gone for good.
LOG="${LOG:-}"
if [ -n "$LOG" ]; then
  kubectl logs -n "$NS" "job/$JOB" | tee "$LOG"
else
  kubectl logs -n "$NS" "job/$JOB"
fi

case "$status" in
  complete) ;;
  failed)   echo "JOB FAILED"; exit 1 ;;
  *)        echo "TIMED OUT waiting for $JOB"; exit 1 ;;
esac
