locals {
  istio_repo_url = "https://istio-release.storage.googleapis.com/charts"
  istio_repo_version = "1.26.0"
}

module "istio_base" {
  source = "aws-ia/eks-blueprints-addon/aws"
  version = "1.1.1"

  chart         = "base"
  chart_version = local.istio_repo_version
  repository    = local.istio_repo_url
  name          = "istio-base"
  namespace     = var.istio_system_namespace
  description   = "Istio base"
  wait          = true
}

module "istio_istiod" {
  source = "aws-ia/eks-blueprints-addon/aws"
  version = "1.1.1"

  chart         = "istiod"
  chart_version = local.istio_repo_version
  repository    = local.istio_repo_url
  name          = "istio-istiod"
  namespace     = var.istio_system_namespace
  description   = "Istio istiod"
  wait          = true

  values = [
    <<-EOT
      meshConfig:
        accessLogFile: /dev/stdout
        defaultConfig:
          proxyMetadata: {}
          tracing: {}
        enablePrometheusMerge: true
        rootNamespace: ${var.istio_system_namespace}
        tcpKeepalive:
          interval: 5s
          probes: 3
          time: 10s
        trustDomain: cluster.local
        extensionProviders:
          - name: oauth2-proxy
            envoyExtAuthzHttp:
              service: oauth2-proxy.${var.auth_namespace}.svc.cluster.local
              port: 80
              includeRequestHeadersInCheck: ["authorization", "cookie"]
              headersToUpstreamOnAllow: ["authorization", "path", "x-auth-request-user", "x-auth-request-email", "x-auth-request-access-token"]
              headersToDownstreamOnDeny: ["content-type", "set-cookie"]
      pilot:
        env:
          ENABLE_DEBUG_ON_HTTP: false
          CLOUD_PLATFORM: aws
      istio_cni:
        enabled: true
        chained: true
      global:
        istioNamespace:  ${var.istio_system_namespace}
    EOT
  ]

  depends_on = [ module.istio_base ]
}

module "istio_cni" {
  source = "aws-ia/eks-blueprints-addon/aws"
  version = "1.1.1"

  chart         = "cni"
  chart_version = local.istio_repo_version
  repository    = local.istio_repo_url
  name          = "istio-cni"
  namespace     = var.istio_system_namespace
  description   = "Istio cni"
  wait          = true

  values = [
    <<-EOT
      cni:
        excludeNamespaces:
          - ${var.istio_system_namespace}
          - kube-system
    EOT
  ]

  depends_on = [ module.istio_istiod ]
}
