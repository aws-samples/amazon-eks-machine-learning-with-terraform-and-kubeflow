variable "mlflow_namespace" {
  description = "MFlow namespace"
  type        = string
}

variable "mlflow_version" {
  description = "MFlow chart version"
  type        = string
}


variable "force_destroy_bucket" {
  description = "MFlow force destroy bucket"
  type        = bool
}

variable "eks_cluster_id" {
  description = "EKS cluster id"
  type        = string
}

variable "eks_oidc_provider_arn" {
  description = "EKS OIDC provider ARN"
  type        = string
}

variable "eks_oidc_issuer" {
  description = "EKS OIDC issuer"
  type        = string
}

variable "admin_username" {
  description = "MLFlow admin username"
  type        = string
}

variable "admin_password" {
  description = "MLFlow admin password"
  type        = string
}

variable "db_max_capacity" {
  description = "MLFlow DB max capacity"
  type        = number
}

variable "db_subnet_ids" {
  description = "MLFlow DB subnet ids"
  type        = list
}

variable "db_vpc_id" {
  description = "MLFlow DB VPC id"
  type        = string
}

variable "db_port" {
  description = "MLFlow DB port"
  type        = number
}