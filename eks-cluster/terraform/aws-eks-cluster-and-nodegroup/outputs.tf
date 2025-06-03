output "cluster_vpc" {
  description = "Cluster VPC ID"
  value = aws_vpc.vpc.id
}

output "cluster_subnets" {
  description = "Cluster Subnet Ids"
  value = aws_subnet.private.*.id
}

output "cluster_id" {
  description = "Cluster Id"
  value = aws_eks_cluster.eks_cluster.id
}

output "cluster_version" {
  description = "Cluster version"
  value = aws_eks_cluster.eks_cluster.version
}

output "cluster_endpoint" {
  description = "Cluster Endpoint"
  value = aws_eks_cluster.eks_cluster.endpoint
}

output "cluster_oidc_arn" {
  description = "Cluster OIDC ARN"
  value = aws_iam_openid_connect_provider.eks_oidc_provider.arn
}

output "node_role_arn" {
  description = "Managed node group IAM role ARN"
  value = aws_iam_role.node_role.arn
}

output "efs_id" {
  description = "EFS file-system id"
  value = aws_efs_file_system.fs.id
}

output "efs_dns" {
  description = "EFS file-system DNS"
  value = "${aws_efs_file_system.fs.id}.efs.${var.region}.amazonaws.com"
}

output "fsx_id" {
  description = "FSx for Lustre file-system id"
  value = aws_fsx_lustre_file_system.fs.id
}

output "fsx_mount_name" {
  description = "FSx for Lustre file-system mount name"
  value = aws_fsx_lustre_file_system.fs.mount_name
}

output "static_email" {
  description = "kubeflow email"
  value = var.static_email
}

output "static_username" {
  description = "kubeflow username"
  value = var.static_username
}


output "static_password" {
  description = "kubeflow password"
  sensitive = true
  value = random_password.static_password.result
}

output "mlflow_db_secret_arn" {
  description = "MLFlow DB secret ARN"
  value = module.mlflow.*.db_secret_arn
}