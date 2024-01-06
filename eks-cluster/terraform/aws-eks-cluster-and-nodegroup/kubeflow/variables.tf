variable "namespace" {
  description = "Kubeflow namespace"
  default = "kubeflow"
  type = string
}


variable "local_helm_repo" {
  description = "Local Helm charts path"
  type        = string
  default     = "../../../charts"
}