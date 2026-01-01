#################################################
# HyperPod Module Variables
#################################################

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "hyperpod_cluster_name" {
  description = "Name for the SageMaker HyperPod cluster"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "eks_cluster_arn" {
  description = "ARN of the EKS cluster to orchestrate HyperPod"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for HyperPod cluster"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for HyperPod instances"
  type        = list(string)
}

variable "security_group_ids" {
  description = "Security group IDs for HyperPod cluster"
  type        = list(string)
}

variable "instance_groups" {
  description = "Configuration for HyperPod instance groups"
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
      instance_type  = "ml.g5.xlarge"
      instance_count = 1
      ebs_volume_gb  = 500
    }
  ]
}

variable "node_recovery" {
  description = "Node recovery mode: Automatic or None"
  type        = string
  default     = "Automatic"
}

variable "lifecycle_scripts_s3_bucket" {
  description = "S3 bucket for lifecycle scripts (will be created if not provided)"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags to apply to HyperPod resources"
  type        = map(string)
  default     = {}
}

#################################################
# Helm Configuration Variables
#################################################

variable "install_helm_dependencies" {
  description = "Install HyperPod Helm chart dependencies"
  type        = bool
  default     = true
}

variable "enable_training_operator" {
  description = "Enable HyperPod Training Operator with job auto-restart support"
  type        = bool
  default     = true
}

variable "enable_health_monitoring" {
  description = "Enable HyperPod health monitoring agent for node health tracking"
  type        = bool
  default     = true
}

variable "enable_deep_health_check" {
  description = "Enable HyperPod deep health check for GPU/accelerator health monitoring"
  type        = bool
  default     = true
}

variable "enable_task_governance" {
  description = "Enable HyperPod task governance with RBAC roles and bindings. Requires compute quota configuration."
  type        = bool
  default     = false
}
