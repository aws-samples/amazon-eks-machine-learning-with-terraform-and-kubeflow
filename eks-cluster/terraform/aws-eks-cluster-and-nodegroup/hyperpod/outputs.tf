#################################################
# HyperPod Module Outputs
#################################################

output "hyperpod_cluster_arn" {
  description = "ARN of the SageMaker HyperPod cluster"
  value       = awscc_sagemaker_cluster.hyperpod.cluster_arn
}

output "hyperpod_cluster_name" {
  description = "Name of the SageMaker HyperPod cluster"
  value       = awscc_sagemaker_cluster.hyperpod.cluster_name
}

output "hyperpod_cluster_status" {
  description = "Status of the SageMaker HyperPod cluster"
  value       = awscc_sagemaker_cluster.hyperpod.cluster_status
}

output "hyperpod_execution_role_arn" {
  description = "ARN of the HyperPod execution IAM role"
  value       = aws_iam_role.hyperpod_execution.arn
}

output "lifecycle_scripts_bucket" {
  description = "S3 bucket containing lifecycle scripts"
  value       = local.lifecycle_bucket_name
}

output "instance_groups" {
  description = "Configured HyperPod instance groups"
  value = [
    for ig in var.instance_groups : {
      name          = ig.name
      instance_type = ig.instance_type
      count         = ig.instance_count
    }
  ]
}
