variable "kubeflow_namespace" {
  description = "kubeflow namespace"
  type = string
}

variable "local_helm_repo" {
  description = "Local Helm charts path"
  type        = string
  default     = "../../../charts"
}
