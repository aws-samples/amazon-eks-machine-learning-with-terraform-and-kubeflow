#---------------------------------------------------------------
# kmcp - Kubernetes MCP Server Controller
# https://github.com/kagent-dev/kmcp
#---------------------------------------------------------------

module "kmcp" {
  count  = var.kmcp_enabled ? 1 : 0
  source = "./kmcp"

  # Basic configuration
  kmcp_namespace = var.kmcp_namespace
  kmcp_version   = var.kmcp_version

  # Controller configuration
  controller_replicas = var.kmcp_controller_replicas

  # Istio sidecar injection
  enable_istio_injection = var.kmcp_enable_istio_injection

  depends_on = [
    module.eks_blueprints_addons
  ]
}
