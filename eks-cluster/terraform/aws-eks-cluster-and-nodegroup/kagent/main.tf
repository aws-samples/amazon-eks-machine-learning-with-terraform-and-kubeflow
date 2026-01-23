# kagent - Kubernetes native AI agent framework
# https://github.com/kagent-dev/kagent

#---------------------------------------------------------------
# Namespace
#---------------------------------------------------------------

resource "kubernetes_namespace" "kagent" {
  metadata {
    name = var.kagent_namespace
    labels = {
      name              = var.kagent_namespace
      istio-injection   = var.enable_istio_injection ? "enabled" : "disabled"
    }
  }
}

#---------------------------------------------------------------
# Service Account with IRSA (for Bedrock access)
#---------------------------------------------------------------

# IMPORTANT: This ServiceAccount is created externally and configured in the Helm chart
# The ServiceAccount name MUST match the IRSA trust policy in iam.tf
# By default: "kagent-sa" (configurable via bedrock_service_account_name variable)
# The Helm chart is configured to use this ServiceAccount via:
#   - controller.serviceAccount.create = false
#   - controller.serviceAccount.name = var.bedrock_service_account_name

resource "kubernetes_service_account" "kagent" {
  count = var.enable_bedrock_access ? 1 : 0

  metadata {
    name      = var.bedrock_service_account_name
    namespace = kubernetes_namespace.kagent.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.kagent_bedrock[0].arn
    }
  }
}

#---------------------------------------------------------------
# PostgreSQL Database (optional, for HA)
#---------------------------------------------------------------

resource "random_password" "kagent_db_password" {
  count   = var.database_type == "postgresql" ? 1 : 0
  length  = 16
  special = false
}

resource "aws_db_subnet_group" "kagent" {
  count      = var.database_type == "postgresql" ? 1 : 0
  name       = "${var.cluster_name}-kagent-db-subnet"
  subnet_ids = var.db_subnet_ids

  tags = {
    Name = "${var.cluster_name}-kagent-db-subnet"
  }
}

resource "aws_security_group" "kagent_db" {
  count       = var.database_type == "postgresql" ? 1 : 0
  name        = "${var.cluster_name}-kagent-db-sg"
  description = "Security group for kagent PostgreSQL database"
  vpc_id      = var.vpc_id

  ingress {
    description = "PostgreSQL from EKS cluster"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-kagent-db-sg"
  }
}

resource "aws_rds_cluster" "kagent" {
  count                   = var.database_type == "postgresql" ? 1 : 0
  cluster_identifier      = "${var.cluster_name}-kagent-db"
  engine                  = "aurora-postgresql"
  engine_mode             = "provisioned"
  engine_version          = var.db_engine_version
  database_name           = var.db_name
  master_username         = var.db_username
  master_password         = random_password.kagent_db_password[0].result
  db_subnet_group_name    = aws_db_subnet_group.kagent[0].name
  vpc_security_group_ids  = [aws_security_group.kagent_db[0].id]
  skip_final_snapshot     = true
  apply_immediately       = true
  backup_retention_period = 7

  serverlessv2_scaling_configuration {
    min_capacity = var.db_min_capacity
    max_capacity = var.db_max_capacity
  }

  tags = {
    Name = "${var.cluster_name}-kagent-db"
  }
}

resource "aws_rds_cluster_instance" "kagent" {
  count              = var.database_type == "postgresql" ? 1 : 0
  identifier         = "${var.cluster_name}-kagent-db-instance"
  cluster_identifier = aws_rds_cluster.kagent[0].id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.kagent[0].engine
  engine_version     = aws_rds_cluster.kagent[0].engine_version

  tags = {
    Name = "${var.cluster_name}-kagent-db-instance"
  }
}

resource "kubernetes_secret" "kagent_db" {
  count = var.database_type == "postgresql" ? 1 : 0

  metadata {
    name      = "kagent-db-credentials"
    namespace = kubernetes_namespace.kagent.metadata[0].name
  }

  data = {
    host     = aws_rds_cluster.kagent[0].endpoint
    port     = "5432"
    database = var.db_name
    username = var.db_username
    password = random_password.kagent_db_password[0].result
    # Connection string format for kagent
    connection_string = "postgresql://${var.db_username}:${random_password.kagent_db_password[0].result}@${aws_rds_cluster.kagent[0].endpoint}:5432/${var.db_name}"
  }

  type = "Opaque"
}

#---------------------------------------------------------------
# kagent CRDs Helm Release (install first)
#---------------------------------------------------------------

resource "helm_release" "kagent_crds" {
  name       = "kagent-crds"
  chart      = "kagent-crds"
  repository = "oci://ghcr.io/kagent-dev/kagent/helm"
  version    = var.kagent_version
  namespace  = kubernetes_namespace.kagent.metadata[0].name

  # CRDs should be installed first and kept even on uninstall
  skip_crds = false

  depends_on = [
    kubernetes_namespace.kagent
  ]
}

#---------------------------------------------------------------
# kagent Main Helm Release
#---------------------------------------------------------------

resource "helm_release" "kagent" {
  name       = "kagent"
  chart      = "kagent"
  repository = "oci://ghcr.io/kagent-dev/kagent/helm"
  version    = var.kagent_version
  namespace  = kubernetes_namespace.kagent.metadata[0].name

  # Database configuration
  dynamic "set" {
    for_each = var.database_type == "postgresql" ? [1] : []
    content {
      name  = "database.type"
      value = "postgres"
    }
  }

  dynamic "set" {
    for_each = var.database_type == "postgresql" ? [1] : []
    content {
      name  = "database.postgres.secretName"
      value = kubernetes_secret.kagent_db[0].metadata[0].name
    }
  }

  dynamic "set" {
    for_each = var.database_type == "sqlite" ? [1] : []
    content {
      name  = "database.type"
      value = "sqlite"
    }
  }

  dynamic "set" {
    for_each = var.database_type == "sqlite" ? [1] : []
    content {
      name  = "database.sqlite.databaseName"
      value = var.sqlite_database_name
    }
  }

  # Service account (IRSA for Bedrock)
  dynamic "set" {
    for_each = var.enable_bedrock_access ? [1] : []
    content {
      name  = "controller.serviceAccount.create"
      value = "false"
    }
  }

  dynamic "set" {
    for_each = var.enable_bedrock_access ? [1] : []
    content {
      name  = "controller.serviceAccount.name"
      value = kubernetes_service_account.kagent[0].metadata[0].name
    }
  }

  # Controller replicas (only >1 if using PostgreSQL)
  set {
    name  = "controller.replicas"
    value = var.database_type == "postgresql" ? var.controller_replicas : 1
  }

  # Resource limits
  set {
    name  = "controller.resources.requests.cpu"
    value = var.controller_cpu_request
  }

  set {
    name  = "controller.resources.requests.memory"
    value = var.controller_memory_request
  }

  set {
    name  = "controller.resources.limits.cpu"
    value = var.controller_cpu_limit
  }

  set {
    name  = "controller.resources.limits.memory"
    value = var.controller_memory_limit
  }

  # UI configuration
  set {
    name  = "ui.enabled"
    value = var.enable_ui
  }

  dynamic "set" {
    for_each = var.enable_ui ? [1] : []
    content {
      name  = "ui.resources.requests.cpu"
      value = var.ui_cpu_request
    }
  }

  dynamic "set" {
    for_each = var.enable_ui ? [1] : []
    content {
      name  = "ui.resources.requests.memory"
      value = var.ui_memory_request
    }
  }

  dynamic "set" {
    for_each = var.enable_ui ? [1] : []
    content {
      name  = "ui.resources.limits.cpu"
      value = var.ui_cpu_limit
    }
  }

  dynamic "set" {
    for_each = var.enable_ui ? [1] : []
    content {
      name  = "ui.resources.limits.memory"
      value = var.ui_memory_limit
    }
  }

  # Additional Helm values
  values = var.additional_helm_values != "" ? [var.additional_helm_values] : []

  depends_on = [
    helm_release.kagent_crds,
    kubernetes_secret.kagent_db
  ]
}

#---------------------------------------------------------------
# Istio VirtualService (optional, for UI access via ingress)
#---------------------------------------------------------------

resource "kubernetes_manifest" "kagent_virtualservice" {
  count = var.enable_ui && var.enable_istio_ingress ? 1 : 0

  manifest = {
    apiVersion = "networking.istio.io/v1beta1"
    kind       = "VirtualService"
    metadata = {
      name      = "kagent-ui"
      namespace = kubernetes_namespace.kagent.metadata[0].name
    }
    spec = {
      hosts = [
        var.ingress_host
      ]
      gateways = [
        var.istio_gateway
      ]
      http = [
        {
          match = [
            {
              uri = {
                prefix = var.ingress_path_prefix
              }
            }
          ]
          rewrite = {
            uri = "/"
          }
          route = [
            {
              destination = {
                host = "kagent-ui.${kubernetes_namespace.kagent.metadata[0].name}.svc.cluster.local"
                port = {
                  number = 8080
                }
              }
            }
          ]
        }
      ]
    }
  }

  depends_on = [
    helm_release.kagent
  ]
}
