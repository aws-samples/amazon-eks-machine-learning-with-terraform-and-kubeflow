#!/usr/bin/env bash
# Track B Scenario 7: Ray Serve scheduling failure due to GPU toleration mismatch.
#
# Story: a RayCluster's worker spec specifies the wrong toleration key for the
# Karpenter GPU NodePool's taint. Karpenter provisions a GPU node with the
# correct taint. The Ray worker pod stays Pending forever because its
# toleration doesn't match the node's taint.
#
# Prereqs: KubeRay operator + Karpenter NodePool with GPU taint
#   nvidia.com/gpu=true:NoSchedule (matches mlops-* clusters and most repos
#   in this project).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

# Track B uses its own namespace.
NAMESPACE="ml-inference"

banner "SCENARIO 7: Ray Serve scheduling failure (GPU toleration mismatch)"
echo "Track B — ML Inference Ops"
echo
echo "Story: a RayCluster worker spec carries a deliberately wrong toleration"
echo "key. Karpenter provisions a GPU node with the correct taint, but the"
echo "worker pod stays Pending because its toleration doesn't match."
echo

# Prereq checks ------------------------------------------------------------
echo "[prereq] KubeRay operator"
if ! kubectl --context "$EKS_CLUSTER_NAME" get crd rayclusters.ray.io >/dev/null 2>&1; then
  echo "  ✗ RayCluster CRD missing. Install KubeRay first." >&2
  echo "  This repo provisions it via kubeflow_platform_enabled = true in TF." >&2
  exit 1
fi
echo "  ✓ RayCluster CRD present"

echo "[prereq] Karpenter GPU NodePool with nvidia.com/gpu=true taint"
GPU_NP=$(kubectl --context "$EKS_CLUSTER_NAME" get nodepools.karpenter.sh \
  -o jsonpath='{range .items[?(@.spec.template.spec.taints[*].key=="nvidia.com/gpu")]}{.metadata.name}{"\n"}{end}' \
  2>/dev/null | head -n1)
if [[ -z "$GPU_NP" ]]; then
  echo "  ✗ No Karpenter NodePool with nvidia.com/gpu taint found." >&2
  echo "  Adapt this scenario to your cluster's NodePool/managed-nodegroup taint." >&2
  exit 1
fi
echo "  ✓ GPU NodePool: $GPU_NP"

# Deploy ------------------------------------------------------------------
echo
echo "[1/3] Ensuring namespace $NAMESPACE"
kubectl --context "$EKS_CLUSTER_NAME" create namespace "$NAMESPACE" 2>/dev/null || true

echo "[2/3] Deploying minimal RayCluster with WRONG toleration key"
echo "      (toleration key: gpu=nvidia, NodePool taint: nvidia.com/gpu=true)"
kubectl --context "$EKS_CLUSTER_NAME" apply -f - <<EOF
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: nova-language-model
  namespace: $NAMESPACE
  labels:
    ml-platform/scenario: "7"
spec:
  rayVersion: "2.52.1"
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: public.ecr.aws/lts/ubuntu:22.04_stable
            command: ["/bin/sh","-c","echo head pod placeholder; sleep 3600"]
            resources:
              requests: { cpu: "200m", memory: "512Mi" }
              limits:   { cpu: "500m", memory: "1Gi" }
  workerGroupSpecs:
    - groupName: language-model-worker
      replicas: 1
      minReplicas: 1
      maxReplicas: 1
      rayStartParams: {}
      template:
        spec:
          # WRONG toleration on purpose:
          #   actual NodePool taint key is 'nvidia.com/gpu'
          #   we ask for 'gpu' — so this never matches.
          tolerations:
            - key: "gpu"
              operator: "Equal"
              value: "nvidia"
              effect: "NoSchedule"
          containers:
            - name: ray-worker
              image: public.ecr.aws/lts/ubuntu:22.04_stable
              command: ["/bin/sh","-c","echo worker pod placeholder; sleep 3600"]
              resources:
                requests:
                  cpu: "500m"
                  memory: "2Gi"
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "2"
                  memory: "8Gi"
                  nvidia.com/gpu: "1"
EOF

echo "[3/3] Waiting up to 60s for FailedScheduling event on the worker pod..."
for _ in $(seq 1 12); do
  if kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get events \
       --field-selector reason=FailedScheduling 2>/dev/null \
       | grep -q "nova-language-model"; then
    break
  fi
  sleep 5
done

echo
echo "Pods status:"
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get pods -l ray.io/cluster=nova-language-model

prompt "A Ray Serve worker for our nova-language-model RayCluster in namespace $NAMESPACE (cluster $EKS_CLUSTER_NAME) is stuck in Pending. The cluster has a Karpenter NodePool ready to provision GPU instances on demand. Investigate the scheduling failure: identify the root cause, locate which Helm chart values control the toleration block, and tell me the exact change to make."

echo
echo "Agent should correlate:"
echo "  • FailedScheduling event from Kubernetes events"
echo "  • Karpenter NodePool taint key (nvidia.com/gpu=true)"
echo "  • RayCluster worker tolerations (currently gpu=nvidia — wrong)"
echo "  • Recommendation: change toleration key to nvidia.com/gpu, value true"
