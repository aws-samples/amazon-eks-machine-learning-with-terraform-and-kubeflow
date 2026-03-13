#---------------------------------------------------------------
# kmcp Module Outputs
#---------------------------------------------------------------

output "namespace" {
  description = "kmcp Kubernetes namespace"
  value       = kubernetes_namespace.kmcp.metadata[0].name
}

output "kmcp_version" {
  description = "kmcp Helm chart version deployed"
  value       = var.kmcp_version
}

output "controller_replicas" {
  description = "Number of kmcp controller replicas"
  value       = var.controller_replicas
}

output "metrics_service_name" {
  description = "kmcp controller metrics service name"
  value       = "kmcp-controller-metrics"
}

output "metrics_port" {
  description = "kmcp controller metrics port"
  value       = 8443
}

output "metrics_access_command" {
  description = "kubectl port-forward command to access kmcp metrics locally"
  value       = "kubectl port-forward -n ${kubernetes_namespace.kmcp.metadata[0].name} svc/kmcp-controller-metrics 8443:8443"
}

output "metrics_service_dns" {
  description = "In-cluster DNS for kmcp metrics service (for Prometheus scrape configs)"
  value       = "kmcp-controller-metrics.${kubernetes_namespace.kmcp.metadata[0].name}.svc.cluster.local:8443"
}

output "istio_injection_enabled" {
  description = "Whether Istio sidecar injection is enabled for kmcp namespace"
  value       = var.enable_istio_injection
}
