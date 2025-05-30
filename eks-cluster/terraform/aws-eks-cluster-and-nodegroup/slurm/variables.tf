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

variable "ssh_public_key" {
  description = "Slurm SSH public key for node login"
  type        = string
}

variable "storage_capacity" {
  description = "Shared storage capacity"
  type        = string
}

variable "password" {
  description = "Slurm password for user rocky"
  type        = string
}