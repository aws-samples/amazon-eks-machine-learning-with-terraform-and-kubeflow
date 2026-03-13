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

### In This Guide

- [System Architecture](#system-architecture) — Detailed infrastructure components and deployment flow
- [Training Examples](./examples/training/README.md) — Multi-framework distributed training
- [Inference Examples](./examples/inference/README.md) — Model serving across platforms
- [Agentic AI Examples](./examples/agentic/README.md) — Kubernetes-native AI agents with kagent

**Deployment Options:**
* [Quick Start (Basic)](#quick-start-basic) — Minimal configuration. Automatically uses the default VPC and public subnets, creates S3 bucket, and deploys a basic EKS cluster. No EC2 key pair required. This repository is automatically cloned at `/home/ubuntu/amazon-eks-machine-learning-with-terraform-and-kubeflow`. Recommended for most users.

* [Advanced Setup](#advanced-setup) — Full control over VPC, subnets, security groups, EKS cluster configuration with EFA support, and SSH key pair. Recommended for advanced users.

## Getting Started

### Prerequisites

**Requirements:**
* [AWS Account](https://aws.amazon.com/account/) with [Administrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html) access

**Supported AWS Regions:**
us-east-1, us-east-2, us-west-2, eu-west-1, eu-central-1, ap-southeast-1, ap-southeast-2, ap-northeast-1, ap-northeast-2, ap-south-1

## Quick Start (Basic)



The basic template automatically discovers the default VPC and its public subnets, creates an S3 bucket for Terraform state and FSx data, deploys a complete EKS cluster, and uses AWS Systems Manager (SSM) Session Manager instead of SSH key pairs.

### Setup Steps

1. **Select your [AWS Region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html)** from the supported regions above

2. **Get Your Public IP:** Use [AWS check ip](http://checkip.amazonaws.com/) to find your public IP address (needed for `DesktopAccessCIDR` parameter, append `/32` to your IP)

3. **Clone Repository:** Clone this repository to your laptop:
   ```bash
   git clone https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow.git
   cd amazon-eks-machine-learning-with-terraform-and-kubeflow
   ```

### Launch the Stack

Create a CloudFormation stack using the [ml-ops-desktop-basic.yaml](./ml-ops-desktop-basic.yaml) template (see [Basic Template Parameters](#basic-template-parameters)) using
[AWS Management Console](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-create-stack.html), or if you prefer CLI, run following commands in a terminal window:

   ```bash
   cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
   bash quick-start.sh
   ```
   
**Important:** The template creates [IAM](https://aws.amazon.com/iam/) resources:
* **Console:** Check "I acknowledge that AWS CloudFormation might create IAM resources" during review
* **CLI:** Use `--capabilities CAPABILITY_NAMED_IAM` flag

**Note:** The stack waits for UserData to complete, including EFS cluster, Fsx Cluster and EKS cluster deployment via Terraform. This takes approximately 45 minutes. **Do not proceed until the stack status shows `CREATE_COMPLETE`.**

### What Gets Created

The stack automatically creates:
* ML Ops desktop with DCV server
* S3 bucket for Terraform state and FSx data
* Basic EKS cluster with:
  * CPU nodes (Cluster Autoscaler managed)
  * GPU nodes via Karpenter (Nvidia CUDA)
  * Neuron nodes via Karpenter (Trainium/Inferentia)
  * EFS and FSx for Lustre shared storage
  * Note: EFA-based multi-node inference or training is not available in Quick Start setup.

### Connect via SSM Session Manager

1. Wait for stack status to show `CREATE_COMPLETE` in CloudFormation console
2. Find your desktop instance in EC2 console (tagged with your stack name)
3. Select the instance and click **Connect** → **Session Manager** → **Start session**
4. Set a password for the `ubuntu` user:
   ```bash
   sudo passwd ubuntu
   ```

### Connect via Amazon DCV Client

1. Download and install the [Amazon DCV client](https://docs.aws.amazon.com/dcv/latest/userguide/client.html) on your laptop
2. Find the public IP of your instance in the EC2 console
3. Connect to `https://<public-ip>:8443` using the DCV client
4. Login as user `ubuntu` with the password you set via SSM

### Basic Template Parameters

| Parameter Name | Description |
| --- | ----------- |
| AWSUbuntuAMIType | **Required**. Selects the AMI type (default: UbuntuPro2404LTS). |
| DesktopAccessCIDR | **Required**. Public IP CIDR range for DCV desktop access. Use [AWS check ip](http://checkip.amazonaws.com/) to find your public IP address, append `/32`. |
| DesktopInstanceType | **Required**. Amazon EC2 instance type (default: m7i.xlarge). |
| EBSOptimized | **Required**. Enable [network optimization for EBS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-optimized.html) (default: true). |
| EbsVolumeSize | **Required**. Size of EBS volume in GB (default: 1000 GB). |
| EbsVolumeType | **Required**. [EBS volume type](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-volume-types.html) (default: gp3). |

### Basic Template Stack Outputs

| Output | Description |
| --- | ----------- |
| VpcId | Default VPC ID discovered by the template |
| SubnetIds | Public subnet IDs used by the Auto Scaling Group |
| SecurityGroupId | Desktop security group ID |
| AutoScalingGroupName | Auto Scaling Group name |

### Next Steps After Deployment

After the stack shows `CREATE_COMPLETE`:

```bash
# Verify cluster access
kubectl get nodes

# Create home folders on shared storage
cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
kubectl apply -f eks-cluster/utils/attach-pvc.yaml -n kubeflow
kubectl wait --for=condition=ready pod/attach-pvc -n kubeflow --timeout=300s
kubectl exec -it -n kubeflow attach-pvc -- bash -c "cd /efs && mkdir -p home && chown 1000:100 home && cd /fsx && mkdir -p home && chown 1000:100 home"
```

## Advanced Setup

For advanced users requiring full control over networking, GPU/Neuron configurations, and EFA-based distributed inference and training, use the advanced template [ml-ops-desktop.yaml](./ml-ops-desktop.yaml). This template does not automatically create an EKS cluster—you must manually run Terraform commands after the desktop is ready.

### Prerequisites (Advanced Setup)

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

To use [ODCR](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-reservations.html) with Karpenter for guaranteed capacity on cudaefa NodePool (p4d, p5 instances), first create capacity reservations in the AWS Console or CLI, then add the capacity reservation variables:

```bash
terraform apply -var="profile=default" -var="region=us-west-2" \
  -var="cluster_name=my-eks-cluster" \
  -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' \
  -var="import_path=s3://<YOUR_S3_BUCKET>/eks-ml-platform/" \
  -var="cuda_efa_az=us-west-2c" \
  -var="neuron_az=us-west-2c" \
  -var="karpenter_cr_enabled=true" \
  -var='karpenter_cr_capacity_types=["reserved","on-demand"]' \
  -var='karpenter_cr_cudaefa_ids=["cr-xxxxx","cr-yyyyy"]'
```

Verify nodes launch with reserved capacity:
```bash
kubectl get nodes -l karpenter.sh/nodepool=cudaefa -o jsonpath='{range .items[*]}{.metadata.name}: {.metadata.labels.karpenter\.sh/capacity-type}{"\n"}{end}'
```

#### Using Capacity Blocks for ML

[Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html) let you reserve GPU instances (p4d, p5) for a defined period. First purchase a Capacity Block in the AWS Console or CLI, then configure.

**Important:** The `cuda_efa_az` variable must match the Availability Zone of your Capacity Block reservation. Karpenter can only provision nodes in the AZ where the cudaefa subnet is tagged. You can verify your Capacity Block's AZ with:
```bash
aws ec2 describe-capacity-reservations --capacity-reservation-ids cr-xxxxx \
  --query 'CapacityReservations[].AvailabilityZone' --output text
```

```bash
terraform apply -var="profile=default" -var="region=us-west-2" \
  -var="cluster_name=my-eks-cluster" \
  -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' \
  -var="import_path=s3://<YOUR_S3_BUCKET>/eks-ml-platform/" \
  -var="cuda_efa_az=us-west-2c" \
  -var="neuron_az=us-west-2c" \
  -var="karpenter_cr_enabled=true" \
  -var='karpenter_cr_capacity_types=["reserved"]' \
  -var='karpenter_cr_cudaefa_ids=["cr-xxxxx"]'
```

You can also select Capacity Blocks by tags:
```bash
terraform apply -var="profile=default" -var="region=us-west-2" \
  -var="cluster_name=my-eks-cluster" \
  -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' \
  -var="import_path=s3://<YOUR_S3_BUCKET>/eks-ml-platform/" \
  -var="cuda_efa_az=us-west-2c" \
  -var="neuron_az=us-west-2c" \
  -var="karpenter_cr_enabled=true" \
  -var='karpenter_cr_capacity_types=["reserved"]' \
  -var='karpenter_cr_cudaefa_tags={"purpose":"ml-training"}'
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

## Modular Components

User can enable or disable optional modular components via Terraform variables:

| Component | Variable | Default |
|-----------|----------|---------|
| [Airflow](https://airflow.apache.org/) | airflow_enabled | false |
| [kagent](https://github.com/kagent-dev/kagent) | kagent_enabled | false |
| [kmcp](https://github.com/kagent-dev/kmcp) | kmcp_enabled | false |
| [Capacity Reservations](https://karpenter.sh/docs/tasks/odcrs/) (ODCR / Capacity Blocks, cudaefa only) | karpenter_cr_enabled | false |
| [Kubeflow](https://www.kubeflow.org/) | kubeflow_platform_enabled | false |
| [KServe](https://kserve.github.io/website/latest/) | kserve_enabled | false |
| [Kueue](https://kueue.sigs.k8s.io/) | kueue_enabled | false |
| [MLFlow](https://mlflow.org/) | mlflow_enabled | false |
| [Nvidia DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) | dcgm_exporter_enabled | false |
| [SageMaker Controller](https://github.com/aws-controllers-k8s/sagemaker-controller) | ack_sagemaker_enabled | false |
| [SageMaker HyperPod](https://aws.amazon.com/sagemaker/ai/hyperpod/) | hyperpod_enabled | false |
| [Slinky Slurm](https://github.com/slinkyproject) | slurm_enabled | false |

## Preinstalled Development Tools

The ML Ops desktop in this project is pre-configured with several development tools:

* **AWS CLI** - Pre-configured with IAM role credentials
* **Claude Code CLI** - Command-line interface for Claude AI
* **Docker** - Container runtime for inference and training workloads
* **JupyterLab** - Interactive notebook environment
* **Kiro** - AI-powered IDE for assisted development
* **Miniconda3** - Python environment manager at `/home/ubuntu/miniconda3`
* **Visual Studio Code** - Full-featured code editor with extensions

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

## Agentic AI Examples

For Kubernetes-native AI agents with kagent, see [Agentic AI Examples README](./examples/agentic/README.md).

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
2. Run cleanup script:
```bash
./eks-cluster/utils/cleanup-before-destroy.sh
```

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

# Cleanup before destroy
./eks-cluster/utils/cleanup-before-destroy.sh

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
