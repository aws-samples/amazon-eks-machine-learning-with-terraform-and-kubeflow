# SageMaker HyperPod EKS Terraform Module

This module integrates Amazon SageMaker HyperPod with your EKS cluster, providing resilient, self-healing infrastructure for ML workloads.

## Features

- **Automatic Node Recovery**: Failed nodes are automatically replaced by SageMaker
- **Deep Health Checks**: Hardware-level monitoring for GPUs and accelerators
- **Job Auto-Restart**: Training jobs automatically resume after node failures
- **Health-Aware Scheduling**: Kubernetes scheduling respects node health status
- **EKS Integration**: Native Kubernetes orchestration with SageMaker-managed nodes

## Quick Start

### 1. Enable HyperPod in terraform.tfvars

```hcl
# Enable HyperPod
hyperpod_enabled = true
hyperpod_cluster_name = "my-hyperpod-cluster"

# Configure instance groups
hyperpod_instance_groups = [
  {
    name           = "gpu-workers"
    instance_type  = "ml.g5.48xlarge"
    instance_count = 2
    ebs_volume_gb  = 500
    deep_health_checks = ["InstanceStress", "InstanceConnectivity"]
  }
]

# Enable automatic node recovery
hyperpod_node_recovery = "Automatic"
```

### 2. Deploy Infrastructure

```bash
terraform init
terraform plan
terraform apply
```

### 3. Verify HyperPod Nodes

```bash
# Check nodes joined the cluster
kubectl get nodes -l node.kubernetes.io/instance-type=ml.g5.48xlarge

# Check HyperPod health components
kubectl get pods -n aws-hyperpod

# Check node health status
./modules/hyperpod/scripts/check-hyperpod-status.sh <cluster-name> <region>
```

### 4. Run Training on HyperPod

Existing training examples work on HyperPod with a simple overlay:

```bash
# Deploy training job on HyperPod
helm install qwen-sft-hp \
  charts/machine-learning/training/pytorchjob-distributed/ \
  -f examples/training/accelerate/qwen3-14b-sft/fine-tune.yaml \
  -f examples/training/accelerate/qwen3-14b-sft/hyperpod.yaml \
  -n kubeflow-user-example-com
```

See [examples/hyperpod/README.md](../../../../../examples/hyperpod/README.md) for more details.

## Configuration

### Instance Types

#### GPU Instances (NVIDIA)

| Instance Type | GPUs | Memory | Deep Health Checks |
|--------------|------|--------|-------------------|
| ml.g5.xlarge | 1 | 24 GB | Yes |
| ml.g5.48xlarge | 8 | 192 GB | Yes |
| ml.g6.48xlarge | 8 | 192 GB | Yes |
| ml.p4d.24xlarge | 8 | 320 GB (A100) | Yes |
| ml.p5.48xlarge | 8 | 640 GB (H100) | Yes |

#### Trainium Instances

| Instance Type | Accelerators | Deep Health Checks |
|--------------|--------------|-------------------|
| ml.trn1.32xlarge | 16 | Yes |
| ml.trn2.48xlarge | 16 | Yes |

#### CPU Instances (no deep health checks)

| Instance Type | vCPUs | Memory |
|--------------|-------|--------|
| ml.m5.xlarge | 4 | 16 GB |
| ml.m5.24xlarge | 96 | 384 GB |

### Deep Health Checks

For GPU/Trainium instances, enable hardware-level monitoring:

```hcl
deep_health_checks = ["InstanceStress", "InstanceConnectivity"]
```

- **InstanceStress**: GPU stress tests to detect hardware issues
- **InstanceConnectivity**: Network connectivity tests for distributed training

### Node Recovery Modes

```hcl
hyperpod_node_recovery = "Automatic"  # Recommended: auto-replace failed nodes
hyperpod_node_recovery = "None"       # Manual intervention required
```

## Multi-Node Production Setup

For production distributed training workloads:

```hcl
hyperpod_enabled = true
hyperpod_cluster_name = "production-hyperpod"

hyperpod_instance_groups = [
  {
    name           = "training-gpu"
    instance_type  = "ml.p4d.24xlarge"
    instance_count = 4
    ebs_volume_gb  = 1000
    deep_health_checks = ["InstanceStress", "InstanceConnectivity"]
  },
  {
    name           = "inference-gpu"
    instance_type  = "ml.g5.48xlarge"
    instance_count = 2
    ebs_volume_gb  = 500
    deep_health_checks = ["InstanceStress", "InstanceConnectivity"]
  }
]

hyperpod_node_recovery = "Automatic"
hyperpod_install_helm_dependencies = true
hyperpod_enable_training_operator = false  # Already installed in base
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         EKS Cluster                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  System Nodes   │  │ Karpenter Nodes │  │  HyperPod Nodes │  │
│  │  (EKS Managed)  │  │  (On-Demand)    │  │ (SageMaker Mgd) │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                   │              │
│                                    ┌──────────────┴──────────┐  │
│                                    │   HyperPod Components   │  │
│                                    │  ├─ Health Monitor      │  │
│                                    │  ├─ Deep Health Check   │  │
│                                    │  ├─ Job Auto-Restart    │  │
│                                    │  └─ Node Recovery       │  │
│                                    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │    SageMaker    │
                    │  HyperPod API   │
                    │  ├─ Cluster Mgmt│
                    │  ├─ Node Mgmt   │
                    │  └─ Health Mgmt │
                    └─────────────────┘
```

## Module Inputs

| Name | Description | Type | Default |
|------|-------------|------|---------|
| `hyperpod_enabled` | Enable HyperPod integration | bool | `false` |
| `hyperpod_cluster_name` | HyperPod cluster name | string | `""` |
| `hyperpod_instance_groups` | Instance group configurations | list(object) | See below |
| `hyperpod_node_recovery` | Recovery mode: Automatic or None | string | `"Automatic"` |
| `hyperpod_install_helm_dependencies` | Install HyperPod helm charts | bool | `true` |
| `hyperpod_enable_training_operator` | Enable Training Operator via HyperPod | bool | `false` |

## Module Outputs

| Name | Description |
|------|-------------|
| `hyperpod_cluster_arn` | ARN of the HyperPod cluster |
| `hyperpod_cluster_name` | Name of the HyperPod cluster |
| `hyperpod_cluster_status` | Status of the HyperPod cluster |
| `hyperpod_execution_role_arn` | ARN of the execution IAM role |
| `lifecycle_scripts_bucket` | S3 bucket for lifecycle scripts |

## Training Charts Integration

The training charts (`pytorchjob-distributed`, `pytorchjob-elastic`) support HyperPod natively:

```yaml
# In your training values file
hyperpod:
  enabled: true      # Enable HyperPod mode
  auto_restart: true # Enable job auto-restart

resources:
  node_type: 'p4d.24xlarge'  # Automatically becomes ml.p4d.24xlarge
```

## Monitoring

### Check Cluster Status

```bash
./scripts/check-hyperpod-status.sh <cluster-name> <region>
```

### AWS CLI Commands

```bash
# Cluster status
aws sagemaker describe-cluster --cluster-name <cluster-name>

# List nodes
aws sagemaker list-cluster-nodes --cluster-name <cluster-name>
```

### Kubernetes Commands

```bash
# HyperPod nodes
kubectl get nodes -l node.kubernetes.io/instance-type=ml.p4d.24xlarge

# Health status
kubectl get nodes -o custom-columns='NAME:.metadata.name,HEALTH:.metadata.labels.sagemaker\.amazonaws\.com/node-health-status'

# HyperPod components
kubectl get pods -n aws-hyperpod
```

## Troubleshooting

### Nodes Not Joining

1. Check EKS access entries are created (HyperPod handles this automatically)
2. Verify VPC/subnet configuration matches EKS cluster
3. Check SageMaker cluster status: `aws sagemaker describe-cluster --cluster-name <name>`

### Deep Health Check Failures

1. Check node labels: `kubectl get node <name> -o yaml | grep deep-health`
2. View deep health controller logs: `kubectl logs -n aws-hyperpod -l app=deep-health-check`

### Job Not Auto-Restarting

1. Verify annotation: `kubectl get pytorchjob <name> -o yaml | grep auto-restart`
2. Check controller: `kubectl get pods -n aws-hyperpod -l app=job-auto-restart`

## Important Notes

1. **EKS Access Entries**: HyperPod automatically creates access entries for its execution role. Do NOT create them separately in Terraform.

2. **Training Operator**: If your base EKS cluster already has Kubeflow Training Operator installed, set `hyperpod_enable_training_operator = false`.

3. **S3 Bucket Naming**: The lifecycle scripts bucket is created with `sagemaker-` prefix (required for managed IAM policy).

4. **Instance Type Format**: HyperPod uses `ml.*` prefix (e.g., `ml.p4d.24xlarge`). The training charts handle this automatically when `hyperpod.enabled = true`.
