resource "kubernetes_namespace" "mlflow" {
  metadata {
    labels = {
      istio-injection = "disabled"
    }

    name = "${var.mlflow_namespace}"
  }
}

data "aws_vpc" "db_vpc" {
  id = var.db_vpc_id
}

resource "aws_security_group" "db_sg" {
  name = "${var.eks_cluster_id}-mlflow-db-sg"
  description = "Security group for MLFlow DB in vpc"
  vpc_id      = var.db_vpc_id

  ingress {
    from_port   = var.db_port 
    to_port     = var.db_port 
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.db_vpc.cidr_block] 
  }
}

resource "aws_rds_cluster" "db" {
  engine             = "aurora-postgresql"
  engine_mode        = "provisioned"
  database_name      = "mlflow"
  master_username = "mlflowuser"
  manage_master_user_password = true
  storage_encrypted  = true
  skip_final_snapshot = true
  db_subnet_group_name = aws_db_subnet_group.this.name
  enable_http_endpoint = true
  port = var.db_port
  vpc_security_group_ids = [aws_security_group.db_sg.id]

  serverlessv2_scaling_configuration {
    max_capacity             = var.db_max_capacity
    min_capacity             = 0.5
  }
}

resource "aws_rds_cluster_instance" "db" {
  cluster_identifier = aws_rds_cluster.db.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.db.engine
  engine_version     = aws_rds_cluster.db.engine_version
  db_subnet_group_name = aws_db_subnet_group.this.name
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket_prefix = "mlflow-artifacts"
  force_destroy = var.force_destroy_bucket
}

resource "aws_iam_role" "mlflow" {
  name = "${var.eks_cluster_id}-mlflow-role"

  assume_role_policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
            "Federated": "${var.eks_oidc_provider_arn}"
        },
        "Action": "sts:AssumeRoleWithWebIdentity",
        "Condition": {
            "StringEquals": {
            "${var.eks_oidc_issuer}:aud": "sts.amazonaws.com"
            }
        }
      }
  ]
}
POLICY
}

resource "aws_iam_role_policy" "mlflow" {
   name = "${var.eks_cluster_id}-mlflow-policy"
   role = aws_iam_role.mlflow.id

    policy = jsonencode({
      "Version": "2012-10-17",
      "Statement": [
        {
            "Action": [
                "s3:Get*",
                "s3:List*",
                "s3:PutObject*",
                "s3:DeleteObject*"
            ],
            "Resource": [
                "${aws_s3_bucket.mlflow_artifacts.arn}",
                "${aws_s3_bucket.mlflow_artifacts.arn}/*",
            ],
            "Effect": "Allow"
        }
      ]
    })

  depends_on = [ 
    aws_s3_bucket.mlflow_artifacts,
    aws_iam_role.mlflow
  ]
}

resource "aws_db_subnet_group" "this" {
  name_prefix       = "mlflow"
  subnet_ids = var.db_subnet_ids
}

data "aws_secretsmanager_secret_version" "db" {
  secret_id = aws_rds_cluster.db.master_user_secret[0].secret_arn

  depends_on = [ 
    aws_rds_cluster.db,
    aws_rds_cluster_instance.db
  ]
}

resource "null_resource" "add_repo" {
  triggers = {
    always_run = "${timestamp()}"
  }

  provisioner "local-exec" {
    command = <<-EOT
      helm repo add community-charts https://community-charts.github.io/helm-charts
      helm repo update community-charts
    EOT
  }
}

resource "helm_release" "mlflow" {
  name       = "mlflow"
  chart      = "community-charts/mlflow"
  cleanup_on_fail = true
  create_namespace = true
  version    = var.mlflow_version
  namespace  = var.mlflow_namespace
  timeout = 300
  wait = true

  values = [
    <<-EOT
      serviceAccount:
        create: true
        annotations:
          eks.amazonaws.com/role-arn: "${aws_iam_role.mlflow.arn}"
        name: "mlflow"
      backendStore:
        databaseConnectionCheck: true
        postgres:
          enabled: true
          host: "${aws_rds_cluster.db.endpoint}"
          port: ${aws_rds_cluster.db.port}
          database: "${aws_rds_cluster.db.database_name}"
          user: ${jsondecode(data.aws_secretsmanager_secret_version.db.secret_string)["username"]}
          password: ${jsondecode(data.aws_secretsmanager_secret_version.db.secret_string)["password"]}
      artifactRoot:
        s3:
          enabled: true
          bucket: "${aws_s3_bucket.mlflow_artifacts.id}"
      auth:
        enabled: true
        appName: "basic-auth"
        adminUsername: "${var.admin_username}"
        adminPassword: "${var.admin_password}"
        
    EOT
  ]

  depends_on = [ 
    null_resource.add_repo,
    aws_rds_cluster_instance.db, 
    aws_iam_role_policy.mlflow 
  ]
}