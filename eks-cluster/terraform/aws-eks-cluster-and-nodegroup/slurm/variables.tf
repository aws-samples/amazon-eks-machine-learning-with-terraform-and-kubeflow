variable "eks_cluster_id" {
  description = "EKS cluster id"
  type        = string
}

variable "local_helm_repo" {
  description = "Local Helm charts path"
  type        = string
}

variable "slurm_namespace" {
  description = "Slurm namespace"
  type        = string
}

variable "efs_fs_id" {
  description = "EFS file-system id"
  type = string
}

variable "root_ssh_authorized_keys" {
  description = "Slurm Root SSH public keys"
  type        = list
}


variable "login_enabled" {
  description = "Slurm login enabled"
  type        = bool
}

variable "storage_capacity" {
  description = "Shared storage capacity"
  type        = string
}

variable "storage_type" {
  description = "Shared storage type"
  type        = string
}

variable "fsx" {
  description = "FSx for Lustre file system"

  type = object({
    fs_id = string
    dns_name      = string
    mount_name  = string
  })
}

variable "db_max_capacity" {
  description = "DB max capacity"
  type        = number
}

variable "db_subnet_ids" {
  description = "DB subnet ids"
  type        = list
}

variable "db_vpc_id" {
  description = "DB VPC id"
  type        = string
}

variable "db_port" {
  description = "DB port"
  type        = number
}