#################################################
# SageMaker HyperPod Cluster
#
# Creates a HyperPod cluster with EKS orchestration.
# HyperPod automatically manages:
# - EKS access entries for the execution role
# - Node registration with the EKS cluster
# - Health monitoring and node recovery
#################################################

resource "awscc_sagemaker_cluster" "hyperpod" {
  cluster_name = var.hyperpod_cluster_name

  # EKS Orchestrator Configuration
  orchestrator = {
    eks = {
      cluster_arn = var.eks_cluster_arn
    }
  }

  # VPC Configuration - must match EKS cluster VPC
  vpc_config = {
    security_group_ids = var.security_group_ids
    subnets            = var.private_subnet_ids
  }

  # Node Recovery - Automatic enables self-healing
  node_recovery = var.node_recovery

  # Instance Groups Configuration
  instance_groups = [
    for ig in var.instance_groups : {
      instance_group_name = ig.name
      instance_type       = ig.instance_type
      instance_count      = ig.instance_count
      execution_role      = aws_iam_role.hyperpod_execution.arn

      # Lifecycle configuration for node setup
      life_cycle_config = {
        source_s3_uri = local.lifecycle_s3_uri
        on_create     = "on_create.sh"
      }

      # Additional EBS storage per instance
      instance_storage_configs = [
        {
          ebs_volume_config = {
            volume_size_in_gb = ig.ebs_volume_gb
          }
        }
      ]

      # Deep health checks - only for GPU/Trainium instances
      # Supported: G5, P4, P5, Trn instance families
      on_start_deep_health_checks = length(ig.deep_health_checks) > 0 ? ig.deep_health_checks : null
    }
  ]

  # Tags for resource management
  tags = [
    for k, v in var.tags : {
      key   = k
      value = v
    }
  ]

  depends_on = [
    helm_release.hyperpod,
    aws_iam_role_policy_attachment.hyperpod_managed,
    aws_iam_role_policy_attachment.hyperpod_custom,
    aws_s3_object.on_create_script
  ]
}

# NOTE: Do NOT create aws_eks_access_entry here!
# HyperPod automatically creates the access entry for the execution role.
# Creating it in Terraform causes conflicts.
