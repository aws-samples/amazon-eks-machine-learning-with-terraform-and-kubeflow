#!/bin/bash
# HyperPod Auto-Recovery Demo Script
#
# This script demonstrates HyperPod's automatic node recovery and job auto-restart
# features by:
# 1. Deploying a training job on HyperPod nodes
# 2. Monitoring node health and job status
# 3. Simulating a node issue (manual step)
# 4. Observing automatic recovery
#
# Prerequisites:
# - HyperPod cluster deployed with hyperpod_enabled = true
# - kubectl configured to access the EKS cluster
# - Helm 3.x installed
#
# Usage:
#   ./demo-auto-recovery.sh [namespace]

set -e

NAMESPACE=${1:-"kubeflow"}
INSTANCE_TYPE=${2:-"ml.g5.12xlarge"}  # Override with: ./demo-auto-recovery.sh kubeflow ml.g5.48xlarge
RELEASE_NAME="hyperpod-demo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Extract base instance type (without ml. prefix) for node_type value
BASE_INSTANCE_TYPE=$(echo $INSTANCE_TYPE | sed 's/^ml\.//')

# Determine GPU count based on instance type
case $INSTANCE_TYPE in
    ml.g5.xlarge|ml.g5.2xlarge) GPU_COUNT=1 ;;
    ml.g5.4xlarge|ml.g5.8xlarge) GPU_COUNT=1 ;;
    ml.g5.12xlarge|ml.g5.16xlarge) GPU_COUNT=4 ;;
    ml.g5.24xlarge) GPU_COUNT=4 ;;
    ml.g5.48xlarge) GPU_COUNT=8 ;;
    ml.p4d.24xlarge|ml.p4de.24xlarge) GPU_COUNT=8 ;;
    ml.p5.48xlarge) GPU_COUNT=8 ;;
    *) GPU_COUNT=1 ;;
esac

echo "Using instance type: $INSTANCE_TYPE (${GPU_COUNT} GPUs)"

echo "=============================================="
echo "  HyperPod Auto-Recovery Demo"
echo "=============================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."
command -v kubectl >/dev/null 2>&1 || { echo "kubectl not found"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "helm not found"; exit 1; }

# Check for HyperPod nodes
echo ""
echo "Step 1: Checking for HyperPod nodes..."
HYPERPOD_NODES=$(kubectl get nodes -l node.kubernetes.io/instance-type=${INSTANCE_TYPE} -o name 2>/dev/null || true)
if [ -z "$HYPERPOD_NODES" ]; then
    echo "No HyperPod nodes found with instance-type ${INSTANCE_TYPE}"
    echo "Checking for other ml.* instance types..."
    kubectl get nodes -o custom-columns='NAME:.metadata.name,INSTANCE-TYPE:.metadata.labels.node\.kubernetes\.io/instance-type' | grep -E "ml\." || echo "No HyperPod nodes found"
    echo ""
    echo "Please ensure HyperPod is deployed with GPU instances before running this demo."
    echo "Or specify the correct instance type: ./demo-auto-recovery.sh kubeflow ml.g5.12xlarge"
    exit 1
fi
echo "Found HyperPod nodes:"
echo "$HYPERPOD_NODES"

# Show HyperPod node health
echo ""
echo "Step 2: Checking HyperPod node health status..."
kubectl get nodes -l node.kubernetes.io/instance-type=${INSTANCE_TYPE} \
  -o custom-columns='NAME:.metadata.name,HEALTH:.metadata.labels.sagemaker\.amazonaws\.com/node-health-status,DEEP-HEALTH:.metadata.labels.sagemaker\.amazonaws\.com/deep-health-check-status'

# Deploy a test training job
echo ""
echo "Step 3: Deploying test training job with HyperPod auto-restart enabled..."

# Check if chart exists
CHART_PATH="$REPO_ROOT/charts/machine-learning/training/pytorchjob-elastic"
if [ ! -d "$CHART_PATH" ]; then
    echo "Chart not found at $CHART_PATH"
    exit 1
fi

# Create a simple test values file
cat > /tmp/hyperpod-demo-values.yaml <<EOF
image: 'nvcr.io/nvidia/pytorch:24.01-py3'
backoff_limit: 100
resources:
  requests:
    "nvidia.com/gpu": ${GPU_COUNT}
  limits:
    "nvidia.com/gpu": ${GPU_COUNT}
  nnodes: 1
  nproc_per_node: ${GPU_COUNT}
  node_type: '${BASE_INSTANCE_TYPE}'
tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
elastic_policy:
  rdzv_backend: c10d
  rdzv_port: 44000
  min_replicas: 1
  max_replicas: 1
# Skip PVCs for simple demo (no EFS dependency)
pvc: []
pre_script:
  - echo "Starting HyperPod demo training job..."
  - echo "This job will run for 5 minutes to demonstrate auto-recovery"
inline_script:
  - |
    python3 << 'PYEOF'
    import time
    import os
    import torch
    import torch.distributed as dist

    # Initialize distributed if available
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
    else:
        rank = 0

    hostname = os.environ.get('HOSTNAME', 'unknown')
    print(f"[Rank {rank}] Starting training on {hostname}")
    print(f"[Rank {rank}] GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Rank {rank}] GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[Rank {rank}] GPU {i}: {torch.cuda.get_device_name(i)}")

    # Simple training simulation
    for step in range(300):
        if rank == 0 and step % 30 == 0:
            print(f"Step {step}/300 - Training in progress...")
        time.sleep(1)

    if rank == 0:
        print("Training complete!")

    if 'WORLD_SIZE' in os.environ:
        dist.destroy_process_group()
    PYEOF
train:
  env: []
  command: []
  args: []
hyperpod:
  enabled: true
  auto_restart: true
EOF

# Uninstall if exists
helm uninstall $RELEASE_NAME -n $NAMESPACE 2>/dev/null || true

# Install the demo job
helm install $RELEASE_NAME "$CHART_PATH" \
  -f /tmp/hyperpod-demo-values.yaml \
  -n $NAMESPACE

echo ""
echo "Demo job deployed!"
echo ""
echo "Step 4: Monitoring job and node status..."
echo ""
echo "Commands to monitor the demo:"
echo ""
echo "  # Watch job pods:"
echo "  kubectl get pods -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME -w"
echo ""
echo "  # View training logs:"
echo "  kubectl logs -f pytorchjob-$RELEASE_NAME-worker-0 -n $NAMESPACE"
echo ""
echo "  # Check HyperPod node health:"
echo "  kubectl get nodes -l node.kubernetes.io/instance-type=${INSTANCE_TYPE} -o custom-columns='NAME:.metadata.name,HEALTH:.metadata.labels.sagemaker\.amazonaws\.com/node-health-status'"
echo ""
echo "=============================================="
echo "  TO DEMONSTRATE AUTO-RECOVERY:"
echo "=============================================="
echo ""
echo "  1. Watch the training logs in one terminal"
echo "  2. In another terminal, cordon the node to simulate failure:"
echo "     kubectl cordon <node-name>"
echo ""
echo "  3. Delete the training pod to trigger rescheduling:"
echo "     kubectl delete pod pytorchjob-$RELEASE_NAME-worker-0 -n $NAMESPACE"
echo ""
echo "  4. Observe HyperPod detecting the issue and job restarting"
echo "     The job should resume from the last checkpoint!"
echo ""
echo "  5. Uncordon the node when done:"
echo "     kubectl uncordon <node-name>"
echo ""
echo "  6. Clean up:"
echo "     helm uninstall $RELEASE_NAME -n $NAMESPACE"
echo ""
