#---------------------------------------------------------------
# kagent Module Outputs
#---------------------------------------------------------------

output "namespace" {
  description = "kagent Kubernetes namespace"
  value       = kubernetes_namespace.kagent.metadata[0].name
}

output "kagent_version" {
  description = "kagent Helm chart version deployed"
  value       = var.kagent_version
}

output "database_type" {
  description = "Database type used by kagent (sqlite or postgresql)"
  value       = var.database_type
}

output "database_endpoint" {
  description = "PostgreSQL database endpoint (if using postgresql)"
  value       = var.database_type == "postgresql" ? aws_rds_cluster.kagent[0].endpoint : null
}

output "database_secret_name" {
  description = "Kubernetes secret name containing database credentials (if using postgresql)"
  value       = var.database_type == "postgresql" ? kubernetes_secret.kagent_db[0].metadata[0].name : null
}

output "ui_enabled" {
  description = "Whether kagent UI is enabled"
  value       = var.enable_ui
}

output "ui_service_name" {
  description = "kagent UI service name"
  value       = var.enable_ui ? "kagent-ui" : null
}

output "ui_port" {
  description = "kagent UI service port"
  value       = var.enable_ui ? 8080 : null
}

output "ui_access_command" {
  description = "kubectl port-forward command to access kagent UI locally"
  value       = var.enable_ui ? "kubectl port-forward -n ${kubernetes_namespace.kagent.metadata[0].name} svc/kagent-ui 8080:8080" : null
}

output "ingress_enabled" {
  description = "Whether Istio ingress is enabled for kagent UI"
  value       = var.enable_ui && var.enable_istio_ingress
}

output "ingress_url" {
  description = "Ingress URL for kagent UI (if ingress is enabled)"
  value       = var.enable_ui && var.enable_istio_ingress ? (
    var.ingress_hosts[0] == "*" ?
      "<configured-dns-host>${var.ingress_path_prefix}" :
      "https://${var.ingress_hosts[0]}${var.ingress_path_prefix}"
  ) : null
}

output "bedrock_access_enabled" {
  description = "Whether Amazon Bedrock access via IRSA is enabled"
  value       = var.enable_bedrock_access
}

output "bedrock_iam_role_arn" {
  description = "IAM role ARN for Bedrock access (if enabled)"
  value       = var.enable_bedrock_access ? aws_iam_role.kagent_bedrock[0].arn : null
}

output "service_account_name" {
  description = "Kubernetes service account name with IRSA (if Bedrock access is enabled)"
  value       = var.enable_bedrock_access ? kubernetes_service_account.kagent[0].metadata[0].name : null
}

output "controller_replicas" {
  description = "Number of kagent controller replicas"
  value       = var.database_type == "postgresql" ? var.controller_replicas : 1
}

output "istio_injection_enabled" {
  description = "Whether Istio sidecar injection is enabled for kagent namespace"
  value       = var.enable_istio_injection
}
