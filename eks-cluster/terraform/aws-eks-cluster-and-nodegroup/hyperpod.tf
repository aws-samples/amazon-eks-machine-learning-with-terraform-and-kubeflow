#################################################
# SageMaker HyperPod Module Invocation
#################################################

module "hyperpod" {
  count  = var.hyperpod_enabled ? 1 : 0
  source = "./hyperpod"

  # Cluster naming
  cluster_name          = var.cluster_name
  hyperpod_cluster_name = var.hyperpod_cluster_name != "" ? var.hyperpod_cluster_name : "${var.cluster_name}-cluster"
  region                = var.region

  # EKS cluster configuration
  eks_cluster_arn = aws_eks_cluster.eks_cluster.arn

  # Network configuration
  vpc_id             = aws_vpc.vpc.id
  private_subnet_ids = aws_subnet.private[*].id
  security_group_ids = [aws_eks_cluster.eks_cluster.vpc_config[0].cluster_security_group_id]

  # HyperPod configuration
  instance_groups = var.hyperpod_instance_groups
  node_recovery   = var.hyperpod_node_recovery

  # Component toggles use module defaults (defined in hyperpod/variables.tf)
  # Override in terraform.tfvars if needed

  tags = var.tags

  depends_on = [
    aws_eks_cluster.eks_cluster,
    aws_eks_node_group.system_ng,
    kubernetes_namespace.kubeflow
  ]
}

#################################################
# Outputs
#################################################

output "hyperpod_cluster_arn" {
  description = "ARN of the SageMaker HyperPod cluster"
  value       = var.hyperpod_enabled ? module.hyperpod[0].hyperpod_cluster_arn : null
}

output "hyperpod_cluster_name" {
  description = "Name of the SageMaker HyperPod cluster"
  value       = var.hyperpod_enabled ? module.hyperpod[0].hyperpod_cluster_name : null
}