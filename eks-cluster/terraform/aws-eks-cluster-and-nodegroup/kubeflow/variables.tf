variable "kubeflow_namespace" {
  description = "kubeflow namespace"
  type = string
}

variable "local_helm_repo" {
  description = "Local Helm charts path"
  type        = string
  default     = "../../../charts"
}

variable "ingress_gateway" {
  description = "Ingress gateway name"
  type = string
}

variable "ingress_namespace" {
  description = "Ingress namespace"
  type = string
}

variable "ingress_sa" {
  description = "Ingress service account"
  type = string
}

variable "static_email" {
  description = "Kubeflow default user email"
  type = string
}

variable "user_profile_role_arn" {
  description = "AWS IAM role ARN for user profile"
  type = string
}

variable "profile_controller_role_arn" {
  description = "AWS IAM role ARN for profile controller"
  type = string
}

variable "kubeflow_user_profile" {
  description = "Kubeflow default user profile name"
  type = string
}

variable "fsx" {
  description = "FSx for Lustre file system"

  type = object({
    fs_id = string
    dns_name      = string
    mount_name  = string
  })
}

variable "efs_fs_id" {
  description = "EFS file-system id"
  type = string
}

