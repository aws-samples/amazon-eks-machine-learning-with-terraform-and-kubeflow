# kmcp - Kubernetes MCP Server Controller
# https://github.com/kagent-dev/kmcp

#---------------------------------------------------------------
# Namespace
#---------------------------------------------------------------

resource "kubernetes_namespace" "kmcp" {
  metadata {
    name = var.kmcp_namespace
    labels = {
      name            = var.kmcp_namespace
      istio-injection = var.enable_istio_injection ? "enabled" : "disabled"
    }
  }
}

#---------------------------------------------------------------
# kmcp CRDs Helm Release (install first)
#---------------------------------------------------------------

resource "helm_release" "kmcp_crds" {
  name       = "kmcp-crds"
  chart      = "kmcp-crds"
  repository = "oci://ghcr.io/kagent-dev/kmcp/helm"
  version    = var.kmcp_version
  namespace  = kubernetes_namespace.kmcp.metadata[0].name

  skip_crds = false
  wait      = true

  depends_on = [
    kubernetes_namespace.kmcp
  ]
}

#---------------------------------------------------------------
# kmcp Main Helm Release
#---------------------------------------------------------------

resource "helm_release" "kmcp" {
  name       = "kmcp"
  chart      = "kmcp"
  repository = "oci://ghcr.io/kagent-dev/kmcp/helm"
  version    = var.kmcp_version
  namespace  = kubernetes_namespace.kmcp.metadata[0].name

  # Controller replicas
  set {
    name  = "controller.replicas"
    value = var.controller_replicas
  }

  # Resource requests
  set {
    name  = "controller.resources.requests.cpu"
    value = var.controller_cpu_request
  }

  set {
    name  = "controller.resources.requests.memory"
    value = var.controller_memory_request
  }

  # Resource limits
  set {
    name  = "controller.resources.limits.cpu"
    value = var.controller_cpu_limit
  }

  set {
    name  = "controller.resources.limits.memory"
    value = var.controller_memory_limit
  }

  # Leader election
  set {
    name  = "controller.leaderElection.enabled"
    value = var.enable_leader_election
  }

  # Metrics
  set {
    name  = "controller.metrics.enabled"
    value = var.enable_metrics
  }

  dynamic "set" {
    for_each = var.enable_metrics ? [1] : []
    content {
      name  = "controller.metrics.secure"
      value = var.metrics_secure_serving
    }
  }

  # Health probes
  set {
    name  = "controller.health.enabled"
    value = "true"
  }

  # Additional Helm values
  values = var.additional_helm_values != "" ? [var.additional_helm_values] : []

  depends_on = [
    helm_release.kmcp_crds
  ]
}

#---------------------------------------------------------------
# Prometheus ServiceMonitor (optional)
#---------------------------------------------------------------

resource "kubernetes_manifest" "kmcp_service_monitor" {
  count = var.enable_service_monitor ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "kmcp-controller-metrics"
      namespace = kubernetes_namespace.kmcp.metadata[0].name
      labels = {
        app = "kmcp"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          "app.kubernetes.io/name" = "kmcp"
        }
      }
      endpoints = [
        {
          port     = "metrics"
          interval = "30s"
          path     = "/metrics"
        }
      ]
    }
  }

  depends_on = [
    helm_release.kmcp
  ]
}
