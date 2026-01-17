# BEGIN variables

variable "credentials" {
 description = "path to the aws credentials file"
 default = "~/.aws/credentials"
 type    = string
}

variable "profile" {
 description = "name of the aws config profile"
 default = "default"
 type    = string
}

variable "cluster_name" {
  description = "unique name of the eks cluster"
  type    = string
}

variable "eks_admin_role_arn" {
  description = "IAM role ARN to grant EKS cluster admin access. If empty, auto-detects from current caller."
  type        = string
  default     = ""
}

variable "k8s_version" {
  description = "kubernetes version"
  default = "1.33"
  type    = string
}

variable "region" {
 description = "name of aws region to use"
 type    = string
}

variable "azs" {
 description = "list of aws availability zones in aws region"
 type = list
}

variable "neuron_az" {
 description = "single aws availability zone in aws region for neuron"
 type = string
 default = "none"
}

variable "cuda_efa_az" {
 description = "single aws availability zone in aws region for cuda with efa"
 type = string
 default = "none"
}


variable "cidr_vpc" {
 description = "RFC 1918 CIDR range for EKS cluster VPC"
 default = "192.168.0.0/16"
 type    = string
}

variable "cidr_private" {
 description = "RFC 1918 CIDR range list for EKS cluster VPC subnets"
 default = ["192.168.64.0/18", "192.168.128.0/18", "192.168.192.0/18"]
 type    = list 
}

variable "cidr_public" {
 description = "RFC 1918 CIDR range list for EKS cluster VPC subnets"
 default = ["192.168.0.0/24", "192.168.1.0/24", "192.168.2.0/24"]
 type    = list 
}

variable "efs_performance_mode" {
   default = "generalPurpose"
   type = string
}

variable "efs_throughput_mode" {
   description = "EFS performance mode"
   default = "bursting"
   type = string
}

variable "import_path" {
  description = "fsx for lustre s3 import path"
  type = string
  default = ""
}

variable "key_pair" {
  description = "Name of EC2 key pair used to launch EKS cluster worker node EC2 instances"
  type = string
  default = ""
}

variable "node_volume_size" {
  description = "Node disk size in GB"
  type = number
  default = 200
}

variable "node_group_desired" {
    description = "Node group desired size"
    default = 0
    type = number
}

variable "node_group_max" {
    description = "Node group maximum size"
    default = 32
    type = number
}

variable "node_group_min" {
    description = "Node group minimum size"
    default = 0
    type = number
}

variable "capacity_type" {
  description = "Work node group capacity type: ON_DEMAND or SPOT capacity"
  default = "ON_DEMAND"
  type = string
}

variable "system_capacity_type" {
  description = "System node group capacity type: ON_DEMAND or SPOT capacity"
  default = "ON_DEMAND"
  type = string
}

variable "auth_namespace" {
  description = "Auth namespace"
  default = "auth"
  type = string
}

variable "ingress_namespace" {
  description = "Ingress namespace"
  default = "ingress"
  type = string
}

variable "kubeflow_namespace" {
  description = "Kubeflow namespace"
  default = "kubeflow"
  type = string
}

variable "static_email" {
  description = "Default user email"
  type = string
  default = "user@example.com"
}

variable "static_username" {
  description = "Default username"
  type = string
  default = "user"
}

variable "efa_enabled" {
  description = "Map of EFA enabled instance type to number of network interfaces"
  type = map(number)
  default = {
    "p4d.24xlarge" = 4
    "p4de.24xlarge" = 4
    "p5.48xlarge" = 32
    "p5e.48xlarge" = 16
    "p5en.48xlarge" = 16
    "trn1.32xlarge" = 8
    "trn1n.32xlarge" = 16
    "trn2.48xlarge" = 16
  }
}

variable "nvidia_instances" {
  description = "Nvidia instances. Ignored if karpenter_enabled=true."
  type = list(string)
  default = [
    "g6.2xlarge",
    "g6.48xlarge",
    "p4d.24xlarge"
  ]
}

variable "system_instances" {
  description = "List of instance types for system nodes."
  type = list(string)
  default = [
    "t3a.large",
    "t3a.xlarge",
    "t3a.2xlarge",
    "m5.large", 
    "m5.xlarge", 
    "m5.2xlarge", 
    "m5.4xlarge", 
    "m5a.large", 
    "m5a.xlarge", 
    "m5a.2xlarge", 
    "m5a.4xlarge", 
    "m7a.large", 
    "m7a.xlarge", 
    "m7a.2xlarge",
    "m7a.4xlarge"
  ]
}

variable "system_volume_size" {
  description = "System node volume size in GB"
  type = number
  default = 200
}

variable "neuron_instances" {
  description = "Neuron instances. Ignored if karpenter_enabled=true."
  type = list(string)
  default = [
    "inf2.xlarge",
    "inf2.48xlarge",
    "trn1.32xlarge",
    "trn2.48xlarge"
  ]
}

variable "custom_taints" {
  description = "List of custom taints applied to node groups.  Ignored if karpenter_enabled=true"
  type = list(object({
    key = string
    value = string
    effect = string
  }))
  default = []
}


variable "kueue_enabled" {
  description = "Kueue enabled"
  type = bool
  default = false
}

variable "kueue_namespace" {
  description = "Kueue name space"
  type = string
  default = "kueue-system"
}

variable "kueue_version" {
  description = "Kueue version"
  type = string
  default = "0.11.4"
}

variable "karpenter_enabled" {
  description = "Karpenter enabled"
  type = bool
  default = true
}

variable "karpenter_namespace" {
  description = "Karpenter name space"
  type = string
  default = "kube-system"
}

variable "karpenter_version" {
  description = "Karpenter version"
  type = string
  default = "1.5.0"
}

variable "karpenter_capacity_type" {
  description = "Karpenter capacity type: 'on-demand' or 'spot'"
  type = string
  default = "on-demand"
}

variable "karpenter_consolidate_after" {
  description = "Karpenter consolidate-after delay"
  type = string
  default = "600s"
}

variable "karpenter_max_pods" {
  description = "Karpenter kubelet maxPods"
  type = number
  default = 20
}

variable "prometheus_enabled" {
  description = "Prometheus kube stack enabled"
  type = bool
  default = true
}

variable "prometheus_namespace" {
  description = "Prometheus name space"
  type = string
  default = "kube-system"
}

variable "prometheus_version" {
  description = "Prometheus community kube-prometheus-stack chart version"
  type = string
  default = "60.3.0"
}

variable "nvidia_plugin_version" {
  description = "NVIDIA Device Plugin Version"
  type = string
  default = "v0.14.3"
}

variable "local_helm_repo" {
  description = "Local Helm charts path"
  type        = string
  default     = "../../../charts"
}

variable "tags" {
  description = "Tags"
  type        = map
  default     = {}
}

variable "ingress_scheme" {
  description = "ingress scheme: 'internal' or 'internet-facing'"
  type = string
  default = "internal"
}

variable "ingress_cidrs" {
  description = "ingress source cidrs comma separated list"
  type = string
  default = "0.0.0.0/0"
}

variable "ingress_gateway" {
  description = "Ingress Gateway name"
  type = string
  default = "ingress-gateway"
}


variable "cluster_issuer" {
  description = "Cluster issuer name"
  type        = string
  default = "ca-self-signing-issuer"
}

variable "kubeflow_platform_enabled" {
  description = "Install Kubeflow Components, if enabled"
  type        = bool
  default = false
}

variable "ack_sagemaker_enabled" {
  description = "Install ACK for SageMaker"
  type        = bool
  default = false
}

variable "kserve_enabled" {
  description = "Install Kserve, if enabled"
  type        = bool
  default = false
}

variable "kserve_namespace" {
  description = "KServe namespace"
  type        = string
  default = "kserve"
}

variable "kserve_version" {
  description = "KServe version"
  type        = string
  default = "v0.15.1"
}

variable "airflow_enabled" {
  description = "Install Airflow, if enabled"
  type        = bool
  default = false
}

variable "airflow_namespace" {
  description = "Airflow namespace"
  type        = string
  default = "airflow"
}

variable "airflow_version" {
  description = "Airflow version"
  type        = string
  default = "1.16.0"
}

variable "system_group_desired" {
    description = "System group desired size"
    default = 8
    type = number
}

variable "system_group_max" {
    description = "System group maximum size"
    default = 32
    type = number
}

variable "system_group_min" {
    description = "System group minimum size"
    default = 8
    type = number
}

variable fsx_storage_capacity {
  description = "FSx Lustre storage capacity in multiples of 1200"
  default = 1200
  type = number
}

variable "dcgm_exporter_enabled" {
  description = "Install DCGM Exporter"
  type        = bool
  default = false
}

variable "slurm_enabled" {
  description = "Install Slurm, if enabled"
  type        = bool
  default = false
}

variable "slurm_namespace" {
  description = "Slurm namespace"
  type        = string
  default = "slurm"
}

variable "slurm_root_ssh_authorized_keys" {
  description = "Slurm Root SSH public keys"
  type        = list
  default = []
}

variable "slurm_login_enabled" {
  description = "Slurm login enabled"
  type        = bool
  default = false
}

variable "slurm_storage_type" {
  description = "Slurm shared storage type: efs or fsx"
  type        = string
  validation {
    condition     = contains(["efs", "fsx"], var.slurm_storage_type)
    error_message = "The slurm_storage_type must be either 'efs' or 'fsx'."
  }
  default = "efs"
}

variable "slurm_storage_capacity" {
  description = "Slurm shared storage capacity"
  type        = string
  default = "1200Gi"
}

variable "slurm_db_max_capacity" {
  description = "Slurm DB Max Capacity"
  type        = number
  default = 16.0
}


variable "mlflow_enabled" {
  description = "Install MLFlow, if enabled"
  type        = bool
  default = false
}

variable "mlflow_namespace" {
  description = "MLFlow namespace"
  type        = string
  default = "mlflow"
}

variable "mlflow_version" {
  description = "MLFlow chart version"
  type        = string
  default = "0.17.2"
}

variable "mlflow_force_destroy_bucket" {
  description = "MLFlow force destroy bucket"
  type        = bool
  default = false
}

variable "mlflow_admin_username" {
  description = "MLFlow admin username"
  type        = string
  default = "admin"
}

variable "mlflow_db_max_capacity" {
  description = "MLFlow DB Max Capacity"
  type        = number
  default = 16.0
}

variable neuron_capacity_reservation_id {
  description = "targeted odcr id for neuron type devices"
  type = string
  default = ""
}

variable nvidia_capacity_reservation_id {
  description = "targeted odcr id for nvidia devices"
  type = string
  default = ""
}

#################################################
# SageMaker HyperPod Variables
#################################################

variable "hyperpod_enabled" {
  description = "Enable SageMaker HyperPod EKS integration for resilient ML workloads"
  type        = bool
  default     = false
}

# Karpenter ODCR Configuration (cudaefa NodePool only)
variable "karpenter_odcr_enabled" {
  description = "Enable ODCR support for Karpenter cudaefa NodePool"
  type        = bool
  default     = false
}

variable "hyperpod_cluster_name" {
  description = "Name for the SageMaker HyperPod cluster. If empty, uses cluster_name-cluster"
  type        = string
  default     = ""
}

variable "hyperpod_instance_groups" {
  description = <<-EOT
    Configuration for HyperPod instance groups. Each group can have different instance types.

    Supported instance types include:
    - CPU: ml.m5.xlarge, ml.m5.2xlarge, ml.m5.4xlarge, ml.m5.12xlarge, ml.m5.24xlarge
    - GPU: ml.g5.xlarge, ml.g5.2xlarge, ml.g5.4xlarge, ml.g5.8xlarge, ml.g5.12xlarge, ml.g5.24xlarge, ml.g5.48xlarge
    - GPU High-end: ml.p4d.24xlarge, ml.p4de.24xlarge, ml.p5.48xlarge
    - Trainium: ml.trn1.2xlarge, ml.trn1.32xlarge, ml.trn1n.32xlarge
    - Trainium2: ml.trn2.48xlarge

    deep_health_checks options: ["InstanceStress", "InstanceConnectivity"]
  EOT
  type = list(object({
    name               = string
    instance_type      = string
    instance_count     = number
    ebs_volume_gb      = optional(number, 500)
    deep_health_checks = optional(list(string), ["InstanceStress", "InstanceConnectivity"])
  }))
  default = [
    {
      name           = "worker-group-gpu"
      instance_type  = "ml.g5.2xlarge"
      instance_count = 1
      ebs_volume_gb  = 500
    }
  ]
}


variable "hyperpod_node_recovery" {
  description = "HyperPod node recovery mode. 'Automatic' enables self-healing for failed nodes."
  type        = string
  default     = "Automatic"

  validation {
    condition     = contains(["Automatic", "None"], var.hyperpod_node_recovery)
    error_message = "hyperpod_node_recovery must be either 'Automatic' or 'None'."
  }
}

variable "karpenter_odcr_cudaefa_ids" {
  description = "List of ODCR IDs for CUDA EFA instances (e.g., ['cr-xxx', 'cr-yyy'])"
  type        = list(string)
  default     = []
}

variable "karpenter_odcr_cudaefa_tags" {
  description = "Tags to select ODCRs for CUDA EFA instances (e.g., {purpose = 'distributed-training'})"
  type        = map(string)
  default     = {}
}

variable "karpenter_odcr_capacity_types" {
  description = "Karpenter ODCR capacity types list: 'on-demand', 'spot', 'reserved'"
  type        = list(string)
  default     = ["on-demand"]
}

# END variables
