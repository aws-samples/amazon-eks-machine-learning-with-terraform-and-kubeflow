# Running Training Examples on SageMaker HyperPod

This guide explains how to run the existing training examples on SageMaker HyperPod nodes with automatic node recovery and job auto-restart.

## Overview

SageMaker HyperPod on EKS provides resilient ML infrastructure with:
- **Automatic Node Recovery**: Failed nodes are automatically replaced
- **Deep Health Checks**: Hardware-level GPU/accelerator monitoring
- **Job Auto-Restart**: Training jobs resume automatically after node failures
- **Checkpointless Training**: Jobs resume from last checkpoint without manual intervention

## Quick Start

The training charts (`pytorchjob-distributed`, `pytorchjob-elastic`) now support a `hyperpod` configuration section. Simply add `hyperpod.yaml` overlay to enable HyperPod features.

### Example: Fine-tune Qwen3-14B on HyperPod

```bash
# Standard EKS (without HyperPod)
helm install qwen-sft \
  charts/machine-learning/training/pytorchjob-distributed/ \
  -f examples/training/accelerate/qwen3-14b-sft/fine-tune.yaml \
  -n kubeflow-user-example-com

# With HyperPod (just add the overlay!)
helm install qwen-sft-hp \
  charts/machine-learning/training/pytorchjob-distributed/ \
  -f examples/training/accelerate/qwen3-14b-sft/fine-tune.yaml \
  -f examples/training/accelerate/qwen3-14b-sft/hyperpod.yaml \
  -n kubeflow-user-example-com
```

### Example: BERT Training on HyperPod

```bash
# For g5.12xlarge (4 GPUs)
cd examples/training/accelerate/bert-glue-mrpc
helm install bert-hp \
  ../../../../charts/machine-learning/training/pytorchjob-elastic/ \
  -f pretrain.yaml \
  -f hyperpod-g5-12xlarge.yaml \
  -n kubeflow

# Watch training progress
kubectl logs -f pytorchjob-bert-hp-worker-0 -n kubeflow
```

For different instance types, create an overlay (e.g., `hyperpod-g5-48xlarge.yaml`):
```yaml
resources:
  requests:
    "nvidia.com/gpu": 8
  limits:
    "nvidia.com/gpu": 8
  nnodes: 1
  nproc_per_node: 8
  node_type: 'g5.48xlarge'

hyperpod:
  enabled: true
  auto_restart: true
```

## How It Works

The `hyperpod.yaml` overlay enables these features:

```yaml
hyperpod:
  enabled: true      # Enable HyperPod mode
  auto_restart: true # Enable job auto-restart on node failures
```

When `hyperpod.enabled: true`:
1. **Node Type Prefix**: Automatically adds `ml.` prefix to `node_type` (e.g., `p4d.24xlarge` becomes `ml.p4d.24xlarge`)
2. **HyperPod Tolerations**: Adds tolerations for HyperPod health-aware scheduling
3. **Auto-Restart Annotation**: Adds `sagemaker.amazonaws.com/enable-job-auto-restart: "true"` annotation

## Creating HyperPod Overlays for Other Examples

To enable HyperPod for any existing training example, create a `hyperpod.yaml` file:

```yaml
# hyperpod.yaml
hyperpod:
  enabled: true
  auto_restart: true
```

That's it! The chart handles everything else automatically.

## HyperPod Node Types

HyperPod uses `ml.*` instance types that map to EC2 types:

| EC2 Type | HyperPod Type | GPUs | Memory |
|----------|---------------|------|--------|
| g5.xlarge | ml.g5.xlarge | 1 | 24 GB |
| g5.48xlarge | ml.g5.48xlarge | 8 | 192 GB |
| g6.48xlarge | ml.g6.48xlarge | 8 | 192 GB |
| p4d.24xlarge | ml.p4d.24xlarge | 8 | 320 GB (A100) |
| p5.48xlarge | ml.p5.48xlarge | 8 | 640 GB (H100) |
| trn1.32xlarge | ml.trn1.32xlarge | 16 | Trainium |

## Demo: Auto-Recovery

Run the auto-recovery demo to see HyperPod's resilience features in action:

```bash
cd examples/hyperpod/scripts

# Usage: ./demo-auto-recovery.sh [namespace] [instance-type]
./demo-auto-recovery.sh kubeflow ml.g5.12xlarge
```

The demo:
1. Checks for HyperPod nodes and their health status
2. Deploys a test training job with auto-restart enabled
3. Provides commands to simulate node failure and observe recovery

### Manual Auto-Restart Test

```bash
# 1. Deploy a training job
./demo-auto-recovery.sh kubeflow ml.g5.12xlarge

# 2. Watch training logs in one terminal
kubectl logs -f pytorchjob-hyperpod-demo-worker-0 -n kubeflow

# 3. In another terminal, delete the pod to simulate failure
kubectl delete pod pytorchjob-hyperpod-demo-worker-0 -n kubeflow

# 4. Watch the pod restart automatically
kubectl get pods -n kubeflow -w

# 5. Check logs again - training should resume
kubectl logs -f pytorchjob-hyperpod-demo-worker-0 -n kubeflow
```

## Verifying HyperPod Components

After deploying the infrastructure, verify all HyperPod components are running:

### 1. Check HyperPod Nodes
```bash
# List HyperPod nodes (ml.* instance types)
kubectl get nodes -o custom-columns=\
'NAME:.metadata.name,'\
'TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,'\
'HEALTH:.metadata.labels.sagemaker\.amazonaws\.com/node-health-status'

# Verify GPUs are available
kubectl describe node <hyperpod-node-name> | grep nvidia.com/gpu
```

### 2. Check Health Monitoring Agent
```bash
# Should see health-monitoring-agent daemonset running
kubectl get pods -n aws-hyperpod
kubectl get daemonset -n aws-hyperpod

# Expected output:
# NAME                          READY   STATUS    RESTARTS   AGE
# health-monitoring-agent-xxx   1/1     Running   0          1h
```

### 3. Check Training Operator
```bash
# HyperPod training operator should be running in kubeflow namespace
kubectl get pods -n kubeflow | grep training-operator

# Expected output:
# hyperpod-training-operators-xxx   1/1     Running   0          1h
```

### 4. Check Helm Releases
```bash
# Verify HyperPod helm release
helm list -A | grep hyperpod

# Check installed components
helm get values hyperpod -n kube-system
```

### 5. Verify NVIDIA Device Plugin
```bash
# Should be running on HyperPod nodes
kubectl get pods -n kube-system | grep nvidia
kubectl get pods -n kube-system -o wide | grep nvidia
```

## Troubleshooting

### Job not scheduling on HyperPod nodes

Check if nodes exist with correct labels:
```bash
kubectl get nodes -l node.kubernetes.io/instance-type=ml.p4d.24xlarge
```

### Job not auto-restarting

Verify the annotation is present:
```bash
kubectl get pytorchjob <job-name> -o yaml | grep auto-restart
```

Ensure job-auto-restart controller is running:
```bash
kubectl get pods -n aws-hyperpod -l app=job-auto-restart
```

## Directory Structure

```
examples/hyperpod/
├── README.md                    # This file
└── scripts/
    └── demo-auto-recovery.sh    # Auto-recovery demonstration

examples/training/accelerate/
├── bert-glue-mrpc/
│   ├── pretrain.yaml            # Base training config
│   ├── hyperpod.yaml            # HyperPod overlay (for g6.48xlarge)
│   └── hyperpod-g5-12xlarge.yaml # HyperPod overlay for g5.12xlarge
└── qwen3-14b-sft/
    ├── fine-tune.yaml           # Base training config
    └── hyperpod.yaml            # HyperPod overlay
```
