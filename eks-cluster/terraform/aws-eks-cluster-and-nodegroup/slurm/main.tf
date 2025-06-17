resource "kubernetes_namespace" "slurm" {
  metadata {
    labels = {
      istio-injection = "disabled"
    }

    name = "${var.slurm_namespace}"
  }
}

resource "random_password" "db" {
  length           = 16
  special          = false
}

data "aws_vpc" "db_vpc" {
  id = var.db_vpc_id
}

resource "helm_release" "slurm_ebs_sc" {
  name       = "slurm-ebs-sc"
  chart      = "${var.local_helm_repo}/ebs-sc"
  version    = "1.0.2"
  wait       = "false"

  values = [
    <<-EOT
      class_name: slurm-ebs-sc
      class_name_wait:  slurm-ebs-sc-wait
    EOT
  ]
  
}

resource "aws_security_group" "db_sg" {
  name = "${var.eks_cluster_id}-slurm-db-sg"
  description = "Security group for Slurm DB in vpc"
  vpc_id      = var.db_vpc_id

  ingress {
    from_port   = var.db_port 
    to_port     = var.db_port 
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.db_vpc.cidr_block] 
  }
}

resource "aws_rds_cluster" "db" {
  engine             = "aurora-mysql"
  engine_mode        = "provisioned"
  database_name      = "slurm"
  master_username = "slurm"
  master_password = "${random_password.db.result}"
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

resource "aws_db_subnet_group" "this" {
  name_prefix       = "slurm"
  subnet_ids = var.db_subnet_ids
}

resource "helm_release" "pv_efs" {
  count = var.storage_type == "efs" ? 1 : 0

  chart = "${var.local_helm_repo}/pv-efs"
  name = "pv-efs"
  version = "1.0.0"
  namespace = kubernetes_namespace.slurm.metadata[0].name
  
  values = [
    <<-EOT
      namespace: ${kubernetes_namespace.slurm.metadata[0].name}
      efs:
        volume_name: slurm-efs-pv
        claim_name: slurm-efs-pvc
        class_name: slurm-efs-sc
        fs_id: ${var.efs_fs_id}
        storage: ${var.storage_capacity}
    EOT
  ]
}

resource "helm_release" "pv_fsx" {
  count = var.storage_type == "fsx" ? 1 : 0

  chart = "${var.local_helm_repo}/pv-fsx"
  name = "pv-fsx"
  version = "1.1.0"
  namespace = kubernetes_namespace.slurm.metadata[0].name
  
  values = [
    <<-EOT
      namespace: ${kubernetes_namespace.slurm.metadata[0].name}
      fsx:
        volume_name: slurm-fsx-pv
        claim_name: slurm-fsx-pvc
        class_name: slurm-fsx-sc
        fs_id: ${var.fsx.fs_id}
        mount_name: ${var.fsx.mount_name}
        dns_name: ${var.fsx.dns_name}
        storage: ${var.storage_capacity}
    EOT
  ]
}

locals {
  pvc_release = var.storage_type == "efs" ? helm_release.pv_efs : helm_release.pv_fsx
  pvc_name = var.storage_type == "efs" ? "slurm-efs-pvc" : "slurm-fsx-pvc"
  pvc_path = var.storage_type == "efs" ? "/efs/home" : "/fsx/home"
}

resource "helm_release" "slurm" {
  chart = "oci://ghcr.io/slinkyproject/charts/slurm"
  name = "slurm"
  namespace = kubernetes_namespace.slurm.metadata[0].name
  version = "0.3.0"
  wait = true
  timeout = 300
  
  values = [
    <<-EOT
      login:
        enabled: ${var.login_enabled}
        service:
          type: ClusterIP
        rootSshAuthorizedKeys: var.root_ssh_authorized_keys
      accounting:
        enabled: true
        external:
          enabled: true
          host: "${aws_rds_cluster.db.endpoint}"
          port: ${aws_rds_cluster.db.port}
          database: "${aws_rds_cluster.db.database_name}"
          user: "slurm"
          password: "${random_password.db.result}"
      slurm-exporter:
        enabled: false
      restapi:
        service:
          type: ClusterIP
      controller:
        persistence:
          storageClass: "slurm-ebs-sc-wait"
          size: 100Gi
        service:
          type: ClusterIP 
      mariadb:
        enabled: false
      compute:
        nodesets: {}
    EOT
  ]

  depends_on = [
    local.pvc_release,
    aws_rds_cluster.db,
    aws_rds_cluster_instance.db, 
    helm_release.slurm_operator,
    helm_release.slurm_ebs_sc
  ]
}


resource "helm_release" "slurm_operator" {
  chart = "oci://ghcr.io/slinkyproject/charts/slurm-operator"
  name = "slurm-operator"
  namespace = kubernetes_namespace.slurm.metadata[0].name
  version = "0.3.0"
}