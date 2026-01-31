# MLOps on Amazon EKS - Comprehensive Guide

## Overview

This project provides a **prototypical MLOps solution** on [Amazon EKS](https://docs.aws.amazon.com/whitepapers/latest/overview-deployment-options/amazon-elastic-kubernetes-service.html) using a modular, Helm-based approach. It supports comprehensive ML workflows including distributed training, and inference serving. Comprehensive support for agentic applications is under development.

The solution deploys a complete MLOps platform using Terraform on Amazon EKS with VPC networking, shared file systems (EFS and FSx for Lustre), auto-scaling infrastructure (Karpenter and Cluster Autoscaler), service mesh (Istio), authentication (Dex/OAuth2), and optional ML platform components (Kubeflow, MLFlow, Airflow, KServe, etc.).

### Key Features

- **Modular Architecture**: Enable/disable MLOps components as needed (Airflow, Kubeflow, KServe, Kueue, MLFlow, Slinky Slurm)
- **Helm-Based Pipelines**: Execute ML pipeline steps using Helm charts with YAML recipes
- **Multi-Accelerator Support**: Nvidia GPUs, AWS Trainium, AWS Inferentia with EFA
- **Distributed Training & Inference**: Multi-node training and inference with popular frameworks
- **Shared Storage**: EFS and FSx for Lustre with automatic S3 backup
- **Auto-Scaling**: Karpenter for GPU/AI chips, Cluster Autoscaler for CPU
- **Service Mesh & Security**: Istio ingress gateway, cert-manager, OAuth2 authentication

## Prerequisites

- AWS Account with [Administrator access](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html)
- AWS Region selection (examples use `us-west-2`)
- [EC2 Service Limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) increased for:
  - p4d.24xlarge, g6.xlarge, g6.2xlarge, g6.48xlarge (8+ each)
  - inf2.xlarge, inf2.48xlarge, trn1.32xlarge (8+ each)
  - Increase EC2 Service Limits for additional EC2 instance types you plan to use
- [EC2 Key Pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html)
- **S3 Bucket** for Terraform state and FSx Lustre data (see below)
- Your public IP address (from [AWS check ip](http://checkip.amazonaws.com/))

### Create S3 Bucket

Create an S3 bucket before proceeding. This bucket stores Terraform state and FSx Lustre data. The bucket name must be globally unique.

```bash
# Replace <YOUR_S3_BUCKET> with a unique name (e.g., my-mlops-bucket-12345)
aws s3 mb s3://<YOUR_S3_BUCKET> --region us-west-2
```

## Getting Started

### 1. Launch Build Machine

Use CloudFormation template [ml-ops-desktop.yaml](./ml-ops-desktop.yaml) to create build machine:

```bash
# Via AWS Console or CLI with --capabilities CAPABILITY_NAMED_IAM
```

* Once the stack status in CloudFormation console is ```CREATE_COMPLETE```, find the ML Ops desktop instance launched in your stack in the Amazon EC2 console, and [connect to the instance using SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html) as user ```ubuntu```, using your SSH key pair.
* When you connect using SSH, and you see the message ```"Cloud init in progress! Logs: /var/log/cloud-init-output.log"```, disconnect and try later after about 15 minutes. The desktop installs the Amazon DCV server on first-time startup, and reboots after the install is complete.
* If you see the message ```ML Ops desktop is enabled!```, run the command ```sudo passwd ubuntu``` to set a new password for user ```ubuntu```. Now you are ready to connect to the desktop using the [Amazon DCV client](https://docs.aws.amazon.com/dcv/latest/userguide/client.html)
* The build machine desktop uses EC2 [user-data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html) to initialize the desktop. Most *transient* failures in the desktop initialization can be fixed by rebooting the desktop.


### 2. Clone Repository

```bash
cd ~
git clone https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow.git
cd amazon-eks-machine-learning-with-terraform-and-kubeflow
```

### 3. Install kubectl

```bash
./eks-cluster/utils/install-kubectl-linux.sh
```

### 4. Configure Terraform Backend

Configure S3 backend for Terraform state storage. Replace `<YOUR_S3_BUCKET>` with the S3 bucket you created in Prerequisites. The `<S3_PREFIX>` is a folder path in your bucket (e.g., `eks-ml-platform`) - it will be created automatically.

```bash
# Example: ./eks-cluster/utils/s3-backend.sh my-mlops-bucket-12345 eks-ml-platform
./eks-cluster/utils/s3-backend.sh <YOUR_S3_BUCKET> <S3_PREFIX>
```

### 5. Initialize and Apply Terraform

* Logout from AWS Public ECR as otherwise `terraform apply` command may fail.
* Specify at least three AWS Availability Zones from your AWS Region in the `azs` terraform variable.
* Replace `<YOUR_S3_BUCKET>` with the S3 bucket you created in Prerequisites.
* To use Amazon EC2 [P4](https://aws.amazon.com/ec2/instance-types/p4/), and [P5](https://aws.amazon.com/ec2/instance-types/p5/) instances, set `cuda_efa_az` terrraform variable to a zone in `azs` list that supports P-family instances.
* To use Amazon EC2 AWS [Inferentia](https://aws.amazon.com/ai/machine-learning/inferentia/) and [Trainium](https://aws.amazon.com/ai/machine-learning/trainium/) instance types instances, set `neuron_az` terraform variable to an Availability Zone in your `azs` list that supports these instance types. 

The command below is just an example: You will need to adapt it specify the `azs` that support the P-family, and AWS Inferentia and Trainium instances within your AWS Region.

```bash
docker logout public.ecr.aws
cd eks-cluster/terraform/aws-eks-cluster-and-nodegroup
terraform init

# Replace <YOUR_S3_BUCKET> with your actual S3 bucket name
terraform apply -var="profile=default" -var="region=us-west-2" \
  -var="cluster_name=my-eks-cluster" \
  -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' \
  -var="import_path=s3://<YOUR_S3_BUCKET>/eks-ml-platform/" \
  -var="cuda_efa_az=us-west-2c" \
  -var="neuron_az=us-west-2c"
```

Alternatively:

```bash
docker logout public.ecr.aws
cd eks-cluster/terraform/aws-eks-cluster-and-nodegroup
terraform init

cp terraform.tfvars.example terraform.tfvars # Add/modify variables as needed
./create_eks.sh
```

#### Using On-Demand Capacity Reservations (ODCR)

To use [ODCR](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-reservations.html) with Karpenter for guaranteed capacity on cudaefa NodePool (p4d, p5 instances), first create capacity reservations in the AWS Console or CLI, then add the ODCR variables:

```bash
terraform apply -var="profile=default" -var="region=us-west-2" \
  -var="cluster_name=my-eks-cluster" \
  -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' \
  -var="import_path=s3://<YOUR_S3_BUCKET>/eks-ml-platform/" \
  -var="cuda_efa_az=us-west-2c" \
  -var="neuron_az=us-west-2c" \
  -var="karpenter_odcr_enabled=true" \
  -var='karpenter_odcr_capacity_types=["reserved","on-demand"]' \
  -var='karpenter_odcr_cudaefa_ids=["cr-xxxxx","cr-yyyyy"]'
```

Verify nodes launch with reserved capacity:
```bash
kubectl get nodes -l karpenter.sh/nodepool=cudaefa -o jsonpath='{range .items[*]}{.metadata.name}: {.metadata.labels.karpenter\.sh/capacity-type}{"\n"}{end}'
```

### 6. Create Home Folders on Shared Storage

```bash
cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/
kubectl apply -f eks-cluster/utils/attach-pvc.yaml -n kubeflow
# wait for the `attach-pvc` Pod to be in "ready" Status. 
# check status of the Pod by running
kubectl get pods -n kubeflow -w
# Once the attach-pvc Pod is Ready, run the following commands
kubectl exec -it -n kubeflow attach-pvc -- /bin/bash

# Inside pod
cd /efs && mkdir home && chown 1000:100 home
cd /fsx && mkdir home && chown 1000:100 home
exit
```

## System Architecture

The solution uses Terraform to deploy a comprehensive MLOps platform on Amazon EKS with the following architecture:

### Architecture Schematic

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud (Region)                             │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          VPC (192.168.0.0/16)                          │ │
│  │                                                                        │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │ │
│  │  │  Public Subnet   │  │  Public Subnet   │  │  Public Subnet   │      │ │
│  │  │   (AZ-1)         │  │   (AZ-2)         │  │   (AZ-3)         │      │ │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘      │ │
│  │           │                     │                     │                │ │
│  │  ┌────────▼─────────────────────▼──────────────────---▼────────┐       │ │
│  │  │            Internet Gateway + NAT Gateway                   │       │ │
│  │  └─────────────────────────────────────────────────────────────┘       │ │
│  │                                                                        │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │ │
│  │  │ Private Subnet   │  │ Private Subnet   │  │ Private Subnet   │      │ │
│  │  │   (AZ-1)         │  │   (AZ-2)         │  │   (AZ-3)         │      │ │
│  │  │                  │  │                  │  │                  │      │ │
│  │  │ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │      │ │
│  │  │ │ EKS Cluster  │ │  │ │ EKS Cluster  │ │  │ │ EKS Cluster  │ │      │ │
│  │  │ │              │ │  │ │              │ │  │ │              │ │      │ │
│  │  │ │ System Nodes │ │  │ │ GPU Nodes    │ │  │ │ Neuron Nodes │ │      │ │
│  │  │ │ (Cluster AS) │ │  │ │ (Karpenter)  │ │  │ │ (Karpenter)  │ │      │ │
│  │  │ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │      │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘      │ │
│  │                                                                        │ │
│  └──────────────────────────────────────────────────────────────────────--┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        Shared Storage Layer                            │ │
│  │                                                                        │ │
│  │  ┌─────────────────────────┐      ┌──────────────────────────┐         │ │
│  │  │  Amazon EFS             │      │  FSx for Lustre          │         │ │
│  │  │  (Code, Logs, Configs)  │      │  (Data, Models)          │         │ │
│  │  │  /efs mount             │      │  /fsx mount              │         │ │
│  │  └─────────────────────────┘      └──────────-┬──────────────┘         │ │
│  │                                               │                        │ │
│  │                                               │ Auto Import/Export     │ │
│  │                                               ▼                        │ │
│  │                                    ┌──────────────────────┐            │ │
│  │                                    │   Amazon S3 Bucket   │            │ │
│  │                                    │   (ml-platform/)     │            │ │
│  │                                    └──────────────────────┘            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        EKS Cluster Components                               │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Core Infrastructure                               │ │
│  │  • Istio Service Mesh (ingress gateway, mTLS)                          │ │
│  │  • Cert Manager (TLS certificates)                                     │ │
│  │  • AWS Load Balancer Controller                                        │ │
│  │  • EBS/EFS/FSx CSI Drivers                                             │ │
│  │  • Cluster Autoscaler (CPU nodes)                                      │ │
│  │  • Karpenter (GPU/Neuron nodes)                                        │ │
│  │  • AWS EFA Device Plugin                                               │ │
│  │  • Nvidia Device Plugin / Neuron Device Plugin                         │ │
│  │  • Neuron Scheduler                                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    Authentication & Security                           │ │
│  │  • Dex (OIDC provider)                                                 │ │
│  │  • OAuth2 Proxy (authentication proxy)                                 │ │
│  │  • IRSA (IAM Roles for Service Accounts)                               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    ML Platform Components                              │ │
│  │  • MPI Operator (distributed training)                                 │ │
│  │  • Kubeflow Training Operator (PyTorchJob, TFJob, etc.)                │ │
│  │  • KubeRay Operator (Ray clusters)                                     │ │
│  │  • LeaderWorkerSet (LWS) for multi-node inference                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │              Optional Modular Components (Helm)                        │ │
│  │  • Kubeflow Platform (Notebooks, Pipelines, Katib, Dashboard)          │ │
│  │  • MLFlow (experiment tracking)                                        │ │
│  │  • Airflow (workflow orchestration)                                    │ │
│  │  • KServe (model serving)                                              │ │
│  │  • Kueue (job queueing)                                                │ │
│  │  • Slurm (HPC workload manager)                                        │ │
│  │  • Prometheus Stack (monitoring)                                       │ │
│  │  • DCGM Exporter (GPU metrics)                                         │ │
│  │  • ACK SageMaker Controller                                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Architecture Components

#### Networking Layer
- **VPC**: Custom VPC with public and private subnets across 3 availability zones
- **Internet Gateway**: Outbound internet access for public subnets
- **NAT Gateway**: Outbound internet access for private subnets
- **Security Groups**: EFS, FSx for Lustre, and EKS cluster security groups

#### Compute Layer
- **EKS Control Plane**: Managed Kubernetes control plane (v1.33)
- **System Node Group**: CPU-only nodes for system workloads (Cluster Autoscaler managed)
- **GPU Node Groups**: Nvidia GPU nodes with EFA support (Karpenter managed)
- **Neuron Node Groups**: AWS Trainium/Inferentia nodes with EFA (Karpenter managed)
- **Launch Templates**: Custom AMIs with EFA network interfaces and EBS volumes

#### Storage Layer
- **Amazon EFS**: Shared file system for code, configurations, logs, and checkpoints
- **FSx for Lustre**: High-performance file system for datasets and models
- **S3 Integration**: FSx auto-imports/exports to S3 bucket (eventual consistency)
- **EBS Volumes**: Optional per-pod EBS volumes via CSI driver

#### Auto-Scaling
- **Cluster Autoscaler**: Scales CPU-only managed node groups (0 to max)
- **Karpenter**: Scales GPU and Neuron nodes dynamically based on pod requirements
- **Node Provisioners**: Separate provisioners for CUDA, CUDA+EFA, and Neuron workloads

#### Service Mesh & Ingress
- **Istio**: Service mesh with mTLS, traffic management, and observability
- **Ingress Gateway**: ClusterIP service with HTTP/HTTPS/TCP ports
- **Virtual Services**: Route traffic to ML platform components

#### Authentication & Authorization
- **Dex**: OpenID Connect (OIDC) identity provider
- **OAuth2 Proxy**: Authentication proxy for web applications
- **IRSA**: IAM roles for Kubernetes service accounts (S3, ECR, SageMaker access)
- **Kubeflow Profiles**: Multi-tenant user namespaces with RBAC

#### ML Operators
- **MPI Operator**: Distributed training with MPI (Horovod, etc.)
- **Training Operator**: PyTorchJob, TFJob, MPIJob, etc.
- **KubeRay Operator**: Ray cluster management for distributed compute
- **LeaderWorkerSet**: Multi-node inference with leader election

#### Device Plugins
- **Nvidia Device Plugin**: GPU resource management and scheduling
- **Neuron Device Plugin**: Neuron core/device resource management
- **EFA Device Plugin**: EFA network interface management
- **Neuron Scheduler**: Custom scheduler for Neuron workloads

### Infrastructure as Code

All infrastructure is defined in Terraform with modular components:
- **Main Module**: VPC, EKS cluster, node groups, storage, IAM roles
- **Istio Module**: Service mesh configuration
- **Kubeflow Module**: ML platform components (optional)
- **MLFlow Module**: Experiment tracking (optional)
- **Slurm Module**: HPC workload manager (optional)

### Deployment Flow

1. **Terraform Init**: Initialize S3 backend for state management
2. **Terraform Apply**: Deploy VPC, EKS, storage, and core components
3. **Helm Releases**: Deploy ML operators, device plugins, and optional modules
4. **User Namespaces**: Create user profiles with IRSA and PVCs
5. **ML Workloads**: Deploy training/inference jobs via Helm charts

## Modular Components

Enable optional components via Terraform variables:

| Component | Variable | Default |
|-----------|----------|---------|
| [Airflow](https://airflow.apache.org/) | airflow_enabled | false |
| [kagent](https://github.com/kagent-dev/kagent) | kagent_enabled | false |
| [Karpenter ODCR](https://karpenter.sh/docs/concepts/nodeclasses/#speccapacityreservationselectorterms) (cudaefa only) | karpenter_odcr_enabled | false |
| [Kubeflow](https://www.kubeflow.org/) | kubeflow_platform_enabled | false |
| [KServe](https://kserve.github.io/website/latest/) | kserve_enabled | false |
| [Kueue](https://kueue.sigs.k8s.io/) | kueue_enabled | false |
| [MLFlow](https://mlflow.org/) | mlflow_enabled | false |
| [Nvidia DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) | dcgm_exporter_enabled | false |
| [SageMaker Controller](https://github.com/aws-controllers-k8s/sagemaker-controller) | ack_sagemaker_enabled | false |
| [SageMaker HyperPod](https://aws.amazon.com/sagemaker/ai/hyperpod/) | hyperpod_enabled | false |
| [Slinky Slurm](https://github.com/slinkyproject) | slurm_enabled | false |

## Inference Examples

For comprehensive inference examples across multiple serving platforms and accelerators, see [Inference Examples README](./examples/inference/README.md).

Supported serving platforms:
- Ray Serve with vLLM
- Triton Inference Server (vLLM, TensorRT-LLM, Python, Ray vLLM backends)
- DJL Serving

## Training Examples

For comprehensive training examples across multiple frameworks and accelerators, see [Training Examples README](./examples/training/README.md).

Supported frameworks:
- Hugging Face Accelerate
- Nemo Megatron
- Neuronx Distributed
- Neuronx Distributed Training
- Megatron-DeepSpeed
- RayTrain

### MCP Gateway Registry

Deploy [Model Context Protocol (MCP) Gateway Registry](https://github.com/agentic-community/mcp-gateway-registry) for agentic AI applications:

- **[Single Service Deployment](./examples/agentic/mcp-gateway-registry/)**: Monolithic deployment with self-signed SSL
- **[Microservices Deployment](./examples/agentic/mcp-gateway-microservices/)**: 6-service architecture with authentication, registry, and multiple MCP servers

Features:
- OAuth authentication (GitHub, AWS Cognito)
- MCP server registry and gateway
- Financial information, time, and custom tool servers
- Shared EFS storage for persistent state

### kagent - Kubernetes Native AI Agents

[kagent](https://github.com/kagent-dev/kagent) is a Kubernetes-native framework for building AI agents with tool capabilities and LLM integration.

**Enable kagent:**
```bash
terraform apply \
  -var="kagent_enabled=true"
```

**Configuration Options:**
- `kagent_version`: Helm chart version (default: `"0.7.11"`, pinned for stability - override to upgrade)
- `kagent_database_type`: Choose `"sqlite"` (default, single replica) or `"postgresql"` (HA, multi-replica)
- `kagent_enable_ui`: Enable web UI (default: `true`)
- `kagent_enable_istio_ingress`: Expose UI via Istio ingress (default: `false`)
- `kagent_enable_bedrock_access`: Enable IRSA for Amazon Bedrock access (default: `false`)

**Access kagent UI:**
```bash
# Port-forward (default)
kubectl port-forward -n kagent svc/kagent-ui 8080:8080

# Or via Terraform output
$(terraform output -raw kagent_ui_access_command)
```

**LLM Integration Options:**

kagent supports multiple LLM providers. You can use self-hosted models in EKS or cloud-based services.

**Option 1: Self-Hosted Models in EKS (Recommended)**

Deploy LLM serving solutions within the same EKS cluster:

```yaml
# Example: Using vLLM for self-hosted models
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: llama-3-8b
  namespace: kagent
spec:
  provider: OpenAI  # vLLM provides OpenAI-compatible API
  model: meta-llama3-8b-instruct
  apiKeySecret: kagent-openai
  apiKeySecretKey: OPENAI_API_KEY
  openAI:
    baseUrl: http://vllm-service.inference.svc.cluster.local:8000/v1
```

See the `examples/inference/` directory for deploying vLLM, Ray Serve, or Triton in EKS.

**Option 2: OpenAI or Compatible APIs**

A placeholder `kagent-openai` secret is automatically created. Update it with your OpenAI API key:

```bash
kubectl create secret generic kagent-openai \
  --from-literal=OPENAI_API_KEY=<your-openai-api-key> \
  -n kagent \
  --dry-run=client -o yaml | kubectl apply -f -
```

Then create a ModelConfig:

```yaml
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: gpt-4
  namespace: kagent
spec:
  provider: OpenAI
  model: gpt-4
  apiKeySecret: kagent-openai
  apiKeySecretKey: OPENAI_API_KEY
  openAI:
    baseUrl: https://api.openai.com/v1
```

**Option 3: Amazon Bedrock (Optional)**

For AWS Bedrock integration, enable IRSA:

```bash
terraform apply \
  -var="kagent_enabled=true" \
  -var="kagent_enable_bedrock_access=true"
```

When enabled, an IAM role with Bedrock permissions is automatically created and attached to the kagent controller via IRSA.

```yaml
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: claude-sonnet
  namespace: kagent
spec:
  provider: Bedrock
  model: anthropic.claude-3-5-sonnet-20241022-v2:0
  region: us-west-2
```

**Note**: The module automatically configures `controller.serviceAccount.name=kagent-sa` and `controller.serviceAccount.create=false` in the Helm values when Bedrock access is enabled.

**High Availability:**

For production deployments with multiple controller replicas:
```bash
terraform apply \
  -var="kagent_enabled=true" \
  -var="kagent_database_type=postgresql" \
  -var="kagent_controller_replicas=3"
```

## Legacy Examples

See [README.md](./examples/legacy/README.md)

## Architecture Concepts

### Helm-Based Pipeline Execution

The solution uses Helm charts to execute discrete MLOps pipeline steps:

1. Each pipeline step is a Helm chart installation with a YAML recipe (Helm values file)
2. Steps execute within a single Helm Release for atomicity
3. Charts are installed, executed, then uninstalled before the next step
4. Can be orchestrated manually (Helm CLI), via Kubeflow Pipelines, or Apache Airflow

### YAML Recipes

Common fields in Helm values files:

- `image`: Docker container image
- `resources`: Infrastructure requirements (CPU, memory, GPUs)
- `git`: Code repository to clone (available in `$GIT_CLONE_DIR`)
- `inline_script`: Arbitrary script definitions
- `pre_script`: Executed after git clone, before job launch
- `train`/`process`: Launch commands and arguments
- `post_script`: Optional post-execution script
- `pvc`: Persistent volume mounts (EFS at `/efs`, FSx at `/fsx`)
- `ebs`: Optional EBS volume configuration

### Storage Architecture

- **EFS**: Code, configs, logs, training checkpoints
- **FSx for Lustre**: Data, pre-trained models (auto-syncs with S3)
- **S3**: Automatic backup of FSx content under `ml-platform/` prefix
- **Eventual Consistency**: FSx maintains eventual consistency with S3

### Infrastructure Management

- **Karpenter**: Auto-scales GPU and AI chip nodes (Nvidia, Trainium, Inferentia)
- **Cluster Autoscaler**: Auto-scales CPU-only nodes
- **EKS Managed Node Groups**: Automatic scaling from zero to required capacity

## Kubeflow Platform (Optional)

When enabled, includes:
- Kubeflow Notebooks
- Kubeflow Tensorboard
- Kubeflow Pipelines
- Kubeflow Katib
- Kubeflow Central Dashboard (v1.9.2)

Access dashboard via port-forwarding:
```bash
sudo kubectl port-forward svc/istio-ingressgateway -n ingress 443:443
```

Add to `/etc/hosts`:
```
127.0.0.1 istio-ingressgateway.ingress.svc.cluster.local
```

Login: `user@example.com` with static password from Terraform output

## Cleanup

### Before Destroying Infrastructure

1. Verify S3 backup of important data
2. Uninstall all Helm releases:
```bash
for x in $(helm list -q -n kubeflow-user-example-com); do 
  helm uninstall $x -n kubeflow-user-example-com
done
```

3. Wait 5 minutes, then delete remaining pods:
```bash
kubectl delete --all pods -n kubeflow-user-example-com
```

4. Delete attach-pvc pod:
```bash
kubectl delete -f eks-cluster/utils/attach-pvc.yaml -n kubeflow
```

5. Wait 15 minutes for auto-scaling to zero

### Destroy Infrastructure

This command should mirror your [Apply Terraform](#5-initialize-and-apply-terraform) command:

```bash
cd eks-cluster/terraform/aws-eks-cluster-and-nodegroup
# Replace <YOUR_S3_BUCKET> with your actual S3 bucket name
terraform destroy -var="profile=default" -var="region=us-west-2" \
  -var="cluster_name=my-eks-cluster" \
  -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' \
  -var="import_path=s3://<YOUR_S3_BUCKET>/ml-platform/"
```

## Supported Technologies

### ML Frameworks
- Hugging Face Accelerate
- PyTorch Lightning
- Nemo Megatron-LM
- Megatron-DeepSpeed
- Neuronx Distributed
- Ray Train
- TensorFlow

### Inference Engines
- vLLM (GPU & Neuron)
- Ray Serve
- Triton Inference Server
- TensorRT-LLM
- DJL Serving

### Accelerators
- Nvidia GPUs (P4d, G6, etc.)
- AWS Trainium (Trn1)
- AWS Inferentia (Inf2)
- AWS EFA for distributed training

## Utilities

### Helper Scripts

```bash
# Install kubectl
./eks-cluster/utils/install-kubectl-linux.sh

# Configure S3 backend (replace <YOUR_S3_BUCKET> and <S3_PREFIX> with your values)
./eks-cluster/utils/s3-backend.sh <YOUR_S3_BUCKET> <S3_PREFIX>

# Prepare S3 bucket with COCO dataset (replace <YOUR_S3_BUCKET> with your value)
./eks-cluster/utils/prepare-s3-bucket.sh <YOUR_S3_BUCKET>

# Create EKS cluster with logging
./eks-cluster/terraform/aws-eks-cluster-and-nodegroup/create_eks.sh

# Destroy with retries
./eks-cluster/terraform/aws-eks-cluster-and-nodegroup/terraform_destroy_retry.sh
```

## Resources

- **Source Repository**: https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow
- **Mask R-CNN Blog**: https://aws.amazon.com/blogs/opensource/distributed-tensorflow-training-using-kubeflow-on-amazon-eks/
- **AWS EKS Documentation**: https://docs.aws.amazon.com/eks/
- **Kubeflow Documentation**: https://www.kubeflow.org/docs/
- **Helm Documentation**: https://helm.sh/docs/

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [CONTRIBUTING.md](CONTRIBUTING.md#security-issue-notifications) for security issue notifications.

## License

See [LICENSE](./LICENSE) file.
