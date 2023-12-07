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

variable "region" {
 description = "name of aws region to use"
 type    = string
}

variable "subnet_ids" {
  description = "Node group subnet ids"
  type    = string
}

variable "cluster_name" {
  description = "unique name of the eks cluster"
  type    = string
}

variable "nodegroup_name" {
  description = "Node group name in cluster"
  type    = string
}

variable "node_role_arn" {
  description = "Node role arn"
  type    = string
}


variable "node_volume_size" {
  description = "EKS cluster worker node EBS volume size in GBs"
  default="200"
  type=string
}

variable "node_instance_type" {
  description = "GPU enabled instance types for training."
  default = "p3dn.24xlarge"
  type = string
}

variable "key_pair" {
  description = "Name of EC2 key pair used to launch EKS cluster worker node EC2 instances"
  type = string
  default = ""
}

variable "node_group_desired" {
    description = "EKS worker node auto-scaling group desired size"
    default = "0"
    type = string
}

variable "node_group_max" {
    description = "EKS worker node auto-scaling group maximum"
    default = "8"
    type = string
}

variable "node_group_min" {
    description = "EKS worker node auto-scaling group minimum"
    default = "0" 
    type = string
}

variable "capacity_type" {
  description = "ON_DEMAND or SPOT capacity"
  default = "ON_DEMAND"
  type = string
}

# END variables

provider "aws" {
  region                  = var.region
  shared_credentials_files = [var.credentials]
  profile                 = var.profile
}

resource "aws_eks_node_group" "training_ng" {
  cluster_name    = var.cluster_name 
  node_group_name = var.nodegroup_name 
  node_role_arn   = var.node_role_arn 
  subnet_ids      = split(",", var.subnet_ids) 
  instance_types  = split(",", var.node_instance_type)
  disk_size       = var.node_volume_size 
  ami_type        = "AL2_x86_64_GPU"
  capacity_type = var.capacity_type

  scaling_config {
    desired_size = var.node_group_desired 
    max_size     = var.node_group_max 
    min_size     = var.node_group_min 
  }

  remote_access {
     ec2_ssh_key = var.key_pair != "" ? var.key_pair : null
  }

  taint {
    key = "nvidia.com/gpu"
    value = "true"
    effect = "NO_SCHEDULE"
  }
  
}


locals {
  summary = <<SUMMARY

  EKS NodeGroup Summary: 
  	nodegroup: ${aws_eks_node_group.training_ng.arn} 
 
SUMMARY
}

output "summary" {
  value = local.summary
}
