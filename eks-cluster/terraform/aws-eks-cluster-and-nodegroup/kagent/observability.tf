#---------------------------------------------------------------
# F9: Observability Pipeline — OTEL Collector + Phoenix
#
# Architecture:
#   Agent pods → OTEL Collector → CloudWatch (spans as logs)
#                               → Phoenix (trace UI)
#                               → Prometheus (metrics)
#---------------------------------------------------------------

#---------------------------------------------------------------
# Variables
#---------------------------------------------------------------

variable "enable_observability" {
  description = "Deploy OTEL Collector and Phoenix for memory observability"
  type        = bool
  default     = true
}

variable "otel_collector_version" {
  description = "OpenTelemetry Collector Helm chart version"
  type        = string
  default     = "0.150.1"
}

variable "phoenix_image" {
  description = "Arize Phoenix container image"
  type        = string
  default     = "arizephoenix/phoenix:latest"
}

variable "cloudwatch_log_group" {
  description = "CloudWatch log group for OTEL traces"
  type        = string
  default     = "/aws/otel/memledger"
}

# Uses var.region from variables.tf

#---------------------------------------------------------------
# CloudWatch Log Group for OTEL traces
#---------------------------------------------------------------

resource "aws_cloudwatch_log_group" "otel_traces" {
  count             = var.enable_observability ? 1 : 0
  name              = var.cloudwatch_log_group
  retention_in_days = 14

  tags = {
    Service = "memledger"
    Purpose = "otel-traces"
  }
}

#---------------------------------------------------------------
# IRSA for OTEL Collector → CloudWatch
#---------------------------------------------------------------

data "aws_iam_policy_document" "otel_collector_assume" {
  count = var.enable_observability ? 1 : 0

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [var.eks_oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${var.eks_oidc_issuer}:sub"
      values   = ["system:serviceaccount:${var.kagent_namespace}:otel-collector"]
    }
    condition {
      test     = "StringEquals"
      variable = "${var.eks_oidc_issuer}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "otel_collector" {
  count              = var.enable_observability ? 1 : 0
  name               = "${var.cluster_name}-otel-collector"
  assume_role_policy = data.aws_iam_policy_document.otel_collector_assume[0].json
}

resource "aws_iam_role_policy" "otel_cloudwatch" {
  count = var.enable_observability ? 1 : 0
  name  = "otel-cloudwatch-write"
  role  = aws_iam_role.otel_collector[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
          "xray:GetSamplingRules",
          "xray:GetSamplingTargets",
        ]
        Resource = "*"
      }
    ]
  })
}

resource "kubernetes_service_account" "otel_collector" {
  count = var.enable_observability ? 1 : 0

  metadata {
    name      = "otel-collector"
    namespace = var.kagent_namespace
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.otel_collector[0].arn
    }
  }

  depends_on = [kubernetes_namespace.kagent]
}

#---------------------------------------------------------------
# memledger-ui ServiceAccount — reuses Bedrock IRSA role
#---------------------------------------------------------------

resource "kubernetes_service_account" "memledger_ui" {
  count = var.enable_observability ? 1 : 0

  metadata {
    name      = "memledger-ui-agent"
    namespace = var.kagent_namespace
    annotations = {
      "eks.amazonaws.com/role-arn" = var.enable_bedrock_access ? aws_iam_role.kagent_bedrock[0].arn : ""
    }
  }

  depends_on = [kubernetes_namespace.kagent]
}

#---------------------------------------------------------------
# OpenTelemetry Collector — Deployment mode
#---------------------------------------------------------------

resource "helm_release" "otel_collector" {
  count = var.enable_observability ? 1 : 0

  name             = "otel-collector"
  chart            = "opentelemetry-collector"
  repository       = "https://open-telemetry.github.io/opentelemetry-helm-charts"
  version          = var.otel_collector_version
  namespace        = var.kagent_namespace
  create_namespace = false
  timeout          = 300
  wait             = true

  values = [
    <<-EOT
mode: deployment
replicaCount: 1

serviceAccount:
  create: false
  name: otel-collector

image:
  repository: otel/opentelemetry-collector-contrib

resources:
  requests:
    cpu: 100m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi

ports:
  otlp:
    enabled: true
    containerPort: 4317
    servicePort: 4317
    protocol: TCP
  otlp-http:
    enabled: true
    containerPort: 4318
    servicePort: 4318
    protocol: TCP
  prometheus:
    enabled: true
    containerPort: 8889
    servicePort: 8889
    protocol: TCP

service:
  type: ClusterIP

config:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317
        http:
          endpoint: 0.0.0.0:4318

  processors:
    batch:
      timeout: 5s
      send_batch_size: 256
    memory_limiter:
      check_interval: 5s
      limit_mib: 400
      spike_limit_mib: 100

  exporters:
    # Phoenix — OTLP gRPC for trace visualization
    otlp/phoenix:
      endpoint: phoenix.${var.kagent_namespace}.svc.cluster.local:4317
      tls:
        insecure: true

    # AWS X-Ray — traces for CloudWatch integration
    awsxray:
      region: "${var.region}"

    # Prometheus — expose metrics endpoint for scraping
    prometheus:
      endpoint: "0.0.0.0:8889"

    # Debug logging
    debug:
      verbosity: basic

  service:
    pipelines:
      traces:
        receivers: [otlp]
        processors: [memory_limiter, batch]
        exporters: [otlp/phoenix, awsxray, debug]
      metrics:
        receivers: [otlp]
        processors: [memory_limiter, batch]
        exporters: [prometheus, debug]
    EOT
  ]

  depends_on = [
    kubernetes_service_account.otel_collector,
    aws_cloudwatch_log_group.otel_traces,
  ]
}

#---------------------------------------------------------------
# Arize Phoenix — Trace Visualization UI
#---------------------------------------------------------------

resource "kubernetes_deployment" "phoenix" {
  count = var.enable_observability ? 1 : 0

  metadata {
    name      = "phoenix"
    namespace = var.kagent_namespace
    labels = {
      app = "phoenix"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "phoenix"
      }
    }

    template {
      metadata {
        labels = {
          app = "phoenix"
        }
      }

      spec {
        container {
          name  = "phoenix"
          image = var.phoenix_image

          port {
            container_port = 6006
            name           = "http"
          }

          port {
            container_port = 4317
            name           = "otlp-grpc"
          }

          env {
            name  = "PHOENIX_WORKING_DIR"
            value = "/data"
          }

          env {
            name  = "PHOENIX_PORT"
            value = "6006"
          }

          env {
            name  = "PHOENIX_HOST"
            value = "0.0.0.0"
          }

          resources {
            requests = {
              cpu    = "100m"
              memory = "256Mi"
            }
            limits = {
              cpu    = "500m"
              memory = "512Mi"
            }
          }

          volume_mount {
            name       = "phoenix-data"
            mount_path = "/data"
          }
        }

        volume {
          name = "phoenix-data"
          empty_dir {}
        }
      }
    }
  }

  depends_on = [kubernetes_namespace.kagent]
}

resource "kubernetes_service" "phoenix" {
  count = var.enable_observability ? 1 : 0

  metadata {
    name      = "phoenix"
    namespace = var.kagent_namespace
  }

  spec {
    selector = {
      app = "phoenix"
    }

    port {
      name        = "http"
      port        = 6006
      target_port = 6006
    }

    port {
      name        = "otlp-grpc"
      port        = 4317
      target_port = 4317
    }

    type = "ClusterIP"
  }
}

#---------------------------------------------------------------
# ServiceMonitor for OTEL Collector → Prometheus scraping
#---------------------------------------------------------------

resource "kubernetes_manifest" "otel_servicemonitor" {
  count = var.enable_observability ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "otel-collector"
      namespace = var.kagent_namespace
      labels = {
        release = "prometheus"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          "app.kubernetes.io/name" = "opentelemetry-collector"
        }
      }
      endpoints = [
        {
          port     = "prometheus"
          interval = "15s"
        }
      ]
    }
  }

  depends_on = [helm_release.otel_collector]
}

#---------------------------------------------------------------
# Outputs
#---------------------------------------------------------------

output "otel_collector_endpoint" {
  description = "OTEL Collector OTLP gRPC endpoint (for agent pods)"
  value       = var.enable_observability ? "otel-collector-opentelemetry-collector.${var.kagent_namespace}.svc.cluster.local:4317" : ""
}

output "phoenix_url" {
  description = "Phoenix UI URL (port-forward to access)"
  value       = var.enable_observability ? "http://phoenix.${var.kagent_namespace}.svc.cluster.local:6006" : ""
}

output "otel_collector_role_arn" {
  description = "OTEL Collector IAM role ARN"
  value       = var.enable_observability ? aws_iam_role.otel_collector[0].arn : ""
}
