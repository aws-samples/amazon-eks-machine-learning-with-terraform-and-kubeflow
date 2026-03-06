# kmcp - Kubernetes MCP Server Controller
# https://github.com/kagent-dev/kmcp

#---------------------------------------------------------------
# Namespace
#---------------------------------------------------------------

resource "kubernetes_namespace" "kmcp" {
  metadata {
    name = var.kmcp_namespace
    labels = merge(
      { name = var.kmcp_namespace },
      var.enable_istio_injection ? { "istio-injection" = "enabled" } : {}
    )
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
  timeout   = 300
  atomic    = true

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
  timeout    = 600

  # Controller replicas
  set {
    name  = "controller.replicas"
    value = tostring(var.controller_replicas)
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
    value = tostring(var.enable_leader_election)
  }

  # Metrics
  set {
    name  = "controller.metrics.enabled"
    value = tostring(var.enable_metrics)
  }

  dynamic "set" {
    for_each = var.enable_metrics ? [1] : []
    content {
      name  = "controller.metrics.secure"
      value = tostring(var.metrics_secure_serving)
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
        merge(
          {
            port     = "metrics"
            interval = "30s"
            path     = "/metrics"
          },
          var.metrics_secure_serving ? {
            scheme = "https"
            tlsConfig = {
              insecureSkipVerify = true
            }
          } : {
            scheme    = null
            tlsConfig = null
          }
        )
      ]
    }
  }

  depends_on = [
    helm_release.kmcp
  ]
}
