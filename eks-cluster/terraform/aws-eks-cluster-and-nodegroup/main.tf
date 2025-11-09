provider "aws" {
  region                   = var.region
  shared_credentials_files = [var.credentials]
  profile                  = var.profile
}

provider "aws" {
  region                   = "us-east-1"
  alias                    = "virginia"
  shared_credentials_files = [var.credentials]
  profile                  = var.profile
}

# AWS EKS cluster authentication data source
data "aws_eks_cluster_auth" "cluster" {
  name = var.cluster_name
}

provider "kubectl" {
  host                   = aws_eks_cluster.eks_cluster.endpoint
  cluster_ca_certificate = base64decode(aws_eks_cluster.eks_cluster.certificate_authority[0].data)
  load_config_file       = false
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = aws_eks_cluster.eks_cluster.endpoint
    cluster_ca_certificate = base64decode(aws_eks_cluster.eks_cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

provider "kubernetes" {
  host                   = aws_eks_cluster.eks_cluster.endpoint
  cluster_ca_certificate = base64decode(aws_eks_cluster.eks_cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

data "aws_caller_identity" "current" {}

resource "aws_vpc" "vpc" {
  cidr_block           = var.cidr_vpc
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.cluster_name}-vpc",
  }

}

resource "aws_subnet" "private" {
  count = length(var.azs)

  availability_zone = var.azs[count.index]
  cidr_block        = var.cidr_private[count.index]
  vpc_id            = aws_vpc.vpc.id

  tags = {
    Name                                        = "${var.cluster_name}-subnet-${count.index}",
    "kubernetes.io/cluster/${var.cluster_name}" = "shared",
    "kubernetes.io/role/internal-elb" : "1",
    "karpenter.sh/discovery"         = "${var.cluster_name}"
    "karpenter.sh/discovery/neuron"  = var.azs[count.index] == var.neuron_az ? "${var.cluster_name}" : "nil"
    "karpenter.sh/discovery/cudaefa" = var.azs[count.index] == var.cuda_efa_az ? "${var.cluster_name}" : "nil"
  }

}

resource "aws_subnet" "public" {
  count = length(var.azs)

  availability_zone = var.azs[count.index]
  cidr_block        = var.cidr_public[count.index]
  vpc_id            = aws_vpc.vpc.id

  tags = {
    Name                                        = "${var.cluster_name}-subnet-${count.index}",
    "kubernetes.io/cluster/${var.cluster_name}" = "shared",
    "kubernetes.io/role/elb"                    = "1"
  }

}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.vpc.id

  tags = {
    Name = "${var.cluster_name}-igw"
  }
}

resource "aws_eip" "ip" {
}

resource "aws_nat_gateway" "ngw" {
  allocation_id = aws_eip.ip.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name = "${var.cluster_name}-ngw"
  }

}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.ngw.id
  }

  tags = {
    Name = "${var.cluster_name}-private"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "${var.cluster_name}-public"
  }
}

resource "aws_route_table_association" "private" {
  count = length(var.azs)

  subnet_id      = aws_subnet.private.*.id[count.index]
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "public" {
  count = length(var.azs)

  subnet_id      = aws_subnet.public.*.id[count.index]
  route_table_id = aws_route_table.public.id
}


resource "aws_iam_role" "cluster_role" {
  name = "${var.cluster_name}-control-role"

  assume_role_policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "eks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
POLICY
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.cluster_role.name
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSServicePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
  role       = aws_iam_role.cluster_role.name
}

resource "aws_efs_file_system" "fs" {

  performance_mode = var.efs_performance_mode
  throughput_mode  = var.efs_throughput_mode
  encrypted        = true


  tags = {
    Name = var.cluster_name
  }
}


resource "aws_security_group" "efs_sg" {
  name        = "${var.cluster_name}-efs-sg"
  description = "Security group for efs clients in vpc"
  vpc_id      = aws_vpc.vpc.id

  egress {
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.vpc.cidr_block]
  }

  tags = {
    Name = "${var.cluster_name}-efs-sg"
  }
}

resource "aws_efs_mount_target" "target" {
  count          = length(var.azs)
  file_system_id = aws_efs_file_system.fs.id

  subnet_id       = aws_subnet.private.*.id[count.index]
  security_groups = [aws_security_group.efs_sg.id]

  depends_on = [aws_subnet.private, aws_security_group.efs_sg]
}

resource "aws_security_group" "fsx_lustre_sg" {
  name        = "${var.cluster_name}-fsx-lustre-sg"
  description = "Security group for fsx lustre clients in vpc"
  vpc_id      = aws_vpc.vpc.id

  egress {
    from_port   = 988
    to_port     = 988
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 988
    to_port     = 988
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.vpc.cidr_block]
  }

  tags = {
    Name = "${var.cluster_name}-fsx-lustre-sg"
  }
}

resource "aws_fsx_lustre_file_system" "fs" {
  file_system_type_version = "2.15"

  storage_capacity = var.fsx_storage_capacity
  subnet_ids       = [aws_subnet.private[0].id]
  deployment_type  = "SCRATCH_2"

  security_group_ids = [aws_security_group.fsx_lustre_sg.id]

  log_configuration {
    level = "WARN_ERROR"
  }

  tags = {
    Name = var.cluster_name
  }
}

resource "aws_fsx_data_repository_association" "this" {
  file_system_id                   = aws_fsx_lustre_file_system.fs.id
  data_repository_path             = var.import_path
  file_system_path                 = "/"
  batch_import_meta_data_on_create = true

  s3 {
    auto_export_policy {
      events = ["NEW", "CHANGED", "DELETED"]
    }

    auto_import_policy {
      events = ["NEW", "CHANGED", "DELETED"]
    }
  }

  timeouts {
    create = "30m"  # Increase from default 10m to 30m
    delete = "30m"  # Optional: also increase delete timeout
  }
}

locals {
  use_k8s_version = substr(var.k8s_version, 0, 3) == "1.1" ? "1.33" : var.k8s_version
  s3_bucket       = split("/", substr(var.import_path, 5, -1))[0]
}

resource "aws_eks_cluster" "eks_cluster" {
  name                      = var.cluster_name
  role_arn                  = aws_iam_role.cluster_role.arn
  version                   = local.use_k8s_version
  enabled_cluster_log_types = ["api", "audit"]

  vpc_config {
    subnet_ids = flatten([aws_subnet.private.*.id])
  }

  provisioner "local-exec" {
    when    = destroy
    command = "kubectl config unset current-context"
  }

  provisioner "local-exec" {
    command = "aws --region ${var.region} eks update-kubeconfig --name ${aws_eks_cluster.eks_cluster.name}"
  }

}

resource "aws_security_group_rule" "eks_cluster_ingress" {
  type              = "ingress"
  from_port         = 0
  to_port           = 65535
  protocol          = "all"
  cidr_blocks       = [aws_vpc.vpc.cidr_block]
  security_group_id = aws_eks_cluster.eks_cluster.vpc_config[0].cluster_security_group_id
}

module "ebs_csi_driver_irsa" {
  source  = "aws-ia/eks-blueprints-addon/aws"
  version = "1.1.1"

  # Disable helm release
  create_release = false

  # IAM role for service account (IRSA)
  create_role   = true
  create_policy = false
  role_name     = substr("${aws_eks_cluster.eks_cluster.id}-ebs-csi-driver", 0, 38)
  role_policies = {
    AmazonEBSCSIDriverPolicy = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
  }

  oidc_providers = {
    this = {
      provider_arn    = aws_iam_openid_connect_provider.eks_oidc_provider.arn
      namespace       = "kube-system"
      service_account = "ebs-csi-controller-sa"
    }
  }

  tags = var.tags

}

module "eks_blueprints_addons" {

  source  = "aws-ia/eks-blueprints-addons/aws"
  version = "1.21.0"

  cluster_name      = aws_eks_cluster.eks_cluster.id
  cluster_endpoint  = aws_eks_cluster.eks_cluster.endpoint
  cluster_version   = aws_eks_cluster.eks_cluster.version
  oidc_provider_arn = aws_iam_openid_connect_provider.eks_oidc_provider.arn

  enable_aws_load_balancer_controller = true
  enable_metrics_server               = true
  enable_aws_efs_csi_driver           = true
  enable_aws_fsx_csi_driver           = true

  aws_load_balancer_controller = {
    namespace     = "kube-system"
    chart_version = "v1.13.2"
    wait          = true

    set = [
      {
        name  = "vpcId"
        value = aws_vpc.vpc.id
      }
    ]
  }

  aws_efs_csi_driver = {
    namespace     = "kube-system"
    chart_version = "3.1.7"
  }

  aws_fsx_csi_driver = {
    namespace     = "kube-system"
    chart_version = "1.10.0"
  }

  eks_addons = {
    aws-ebs-csi-driver = {
      addon_version            = "v1.33.0-eksbuild.1"
      service_account_role_arn = module.ebs_csi_driver_irsa.iam_role_arn
    }
  }

  depends_on = [helm_release.cluster-autoscaler]
}

data "aws_iam_policy_document" "cert_manager" {
  source_policy_documents   = []
  override_policy_documents = []

  statement {
    actions   = ["route53:GetChange", ]
    resources = ["arn:aws:route53:::change/*"]
  }

  statement {
    actions = [
      "route53:ChangeResourceRecordSets",
      "route53:ListResourceRecordSets",
    ]
    resources = ["arn:aws:route53:::hostedzone/*"]
  }

  statement {
    actions   = ["route53:ListHostedZonesByName"]
    resources = ["*"]
  }
}


module "cert_manager" {
  source  = "aws-ia/eks-blueprints-addon/aws"
  version = "1.1.1"

  create         = true
  create_release = true

  name             = "cert-manager"
  description      = "A Helm chart to deploy cert-manager"
  namespace        = "cert-manager"
  create_namespace = true
  chart            = "cert-manager"
  chart_version    = "1.13.3"
  repository       = "https://charts.jetstack.io"

  wait = true

  set = [
    {
      name  = "installCRDs"
      value = true
    },
    {
      name  = "serviceAccount.name"
      value = "cert-manager"
    }
  ]

  # IAM role for service account (IRSA)
  set_irsa_names                = ["serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"]
  create_role                   = true
  role_name                     = "cert-manager"
  role_name_use_prefix          = true
  role_path                     = "/"
  role_permissions_boundary_arn = null
  role_description              = "IRSA for cert-manger"
  role_policies                 = {}

  allow_self_assume_role  = true
  source_policy_documents = data.aws_iam_policy_document.cert_manager[*].json
  policy_statements       = []
  policy_name             = "cert-manager"
  policy_name_use_prefix  = true
  policy_path             = null
  policy_description      = "IAM Policy for cert-manager"


  oidc_providers = {
    this = {
      provider_arn = aws_iam_openid_connect_provider.eks_oidc_provider.arn
      # namespace is inherited from chart
      service_account = "cert-manager"
    }
  }

  depends_on = [module.eks_blueprints_addons]
}

resource "aws_iam_role" "cluster_autoscaler" {
  name = "${var.cluster_name}-cluster-autoscaler-role"

  assume_role_policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
      {
      "Effect": "Allow",
      "Principal": {
          "Federated": "${aws_iam_openid_connect_provider.eks_oidc_provider.arn}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
          "StringEquals": {
          "${substr(aws_eks_cluster.eks_cluster.identity[0].oidc[0].issuer, 8, -1)}:aud": "sts.amazonaws.com"
          }
      }
      }
  ]
}
POLICY
}

resource "aws_iam_role_policy" "cluster_autoscaler" {
  name = "cluster-autoscaler-policy"
  role = aws_iam_role.cluster_autoscaler.id

  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Action" : [
          "autoscaling:SetDesiredCapacity",
          "autoscaling:TerminateInstanceInAutoScalingGroup"
        ],
        "Resource" : "*",
        "Condition" : {
          "StringEquals" : {
            "aws:ResourceTag/k8s.io/cluster-autoscaler/enabled" : "true",
            "aws:ResourceTag/k8s.io/cluster-autoscaler/${aws_eks_cluster.eks_cluster.id}" : "owned"
          }
        }
      },
      {
        "Action" : [
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:DescribeAutoScalingInstances",
          "autoscaling:DescribeLaunchConfigurations",
          "autoscaling:DescribeScalingActivities",
          "autoscaling:DescribeTags",
          "ec2:DescribeImages",
          "ec2:DescribeInstanceTypes",
          "ec2:DescribeLaunchTemplateVersions",
          "ec2:GetInstanceTypesFromInstanceRequirements",
          "eks:DescribeNodegroup"
        ],
        "Resource" : "*",
        "Effect" : "Allow"
      }
    ]
  })
}

resource "helm_release" "cluster-autoscaler" {
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  version    = "9.46.6"
  namespace  = "kube-system"
  timeout    = 300
  wait       = true

  values = [
    <<-EOT
      cloudProvider: "aws"
      awsRegion: "${var.region}"
      autoDiscovery:
        clusterName: "${aws_eks_cluster.eks_cluster.id}"
      extraArgs:
        skip-nodes-with-system-pods: "false"
        skip-nodes-with-local-storage: "false"
        expander: "least-waste"
        balance-similar-node-groups: "true"
      podAnnotations:
        cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
      rbac:
        serviceAccount:
          annotations:
            eks.amazonaws.com/role-arn: "${aws_iam_role.cluster_autoscaler.arn}" 
    EOT
  ]

  depends_on = [aws_eks_node_group.system_ng]

}

resource "helm_release" "aws-efa-k8s-device-plugin" {
  name       = "aws-efa-k8s-device-plugin"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-efa-k8s-device-plugin"
  version    = "v0.4.4"
  namespace  = "kube-system"


  values = [
    <<-EOT
      supportedInstanceLabels:
        keys: 
          - "node.kubernetes.io/instance-type"
        values:
          - "trn1.32xlarge"
          - "trn1n.32xlarge"
          - "trn2.48xlarge"
          - "p4d.24xlarge"
          - "p4de.24xlarge"
          - "p5.48xlarge"
          - "p5e.48xlarge"
          - "p5en.48xlarge"
          - "g6e.48xlarge"
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
        - key: "aws.amazon.com/neuron"
          operator: "Exists"
          effect: "NoSchedule"
        - key: "aws.amazon.com/neuroncore"
          operator: "Exists"
          effect: "NoSchedule"
        - key: "aws.amazon.com/neurondevice"
          operator: "Exists"
          effect: "NoSchedule"
        - key: "aws.amazon.com/efa"
          operator: "Exists"
          effect: "NoSchedule"
    EOT
  ]

}

resource "kubernetes_namespace" "auth" {
  metadata {
    labels = {
      istio-injection = "enabled"
    }

    name = var.auth_namespace
  }

  depends_on = [helm_release.cluster-autoscaler]
}

resource "kubernetes_namespace" "kubeflow" {
  metadata {
    labels = {
      istio-injection = "enabled"
    }

    name = var.kubeflow_namespace
  }

  depends_on = [helm_release.cluster-autoscaler]
}

resource "kubernetes_namespace" "lws-system" {
  metadata {
    labels = {
      istio-injection = "disabled"
    }

    name = "lws-system"
  }

  depends_on = [helm_release.cluster-autoscaler]
}

data "tls_certificate" "this" {
  url = aws_eks_cluster.eks_cluster.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks_oidc_provider" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.this.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.eks_cluster.identity[0].oidc[0].issuer

}

resource "aws_iam_role" "node_role" {
  name = "${aws_eks_cluster.eks_cluster.id}-node-role"

  assume_role_policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Principal" : {
          "Service" : "ec2.amazonaws.com"
        },
        "Action" : "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "node_AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.node_role.name
}

resource "aws_iam_role_policy_attachment" "node_AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.node_role.name
}

resource "aws_iam_role_policy_attachment" "node_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.node_role.name
}

resource "aws_iam_role_policy_attachment" "node_AmazonS3ReadOnlyPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
  role       = aws_iam_role.node_role.name
}

resource "aws_eks_node_group" "system_ng" {
  cluster_name    = aws_eks_cluster.eks_cluster.id
  node_group_name = "system"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.private.*.id
  instance_types  = var.system_instances
  disk_size       = var.system_volume_size
  ami_type        = "AL2023_x86_64_STANDARD"
  capacity_type   = var.system_capacity_type

  scaling_config {
    desired_size = var.system_group_desired
    max_size     = var.system_group_max
    min_size     = var.system_group_min
  }

  dynamic "remote_access" {
    for_each = var.key_pair != "" ? [1] : []
    content {
      ec2_ssh_key = var.key_pair
    }
  }

  depends_on = [
    aws_subnet.private,
    aws_subnet.public,
    aws_route_table_association.private,
    aws_route_table_association.public,
    kubectl_manifest.aws_auth
  ]
}

resource "aws_launch_template" "nvidia" {

  count = var.karpenter_enabled ? 0 : length(var.nvidia_instances) > 0 ? length(var.nvidia_instances) : 0

  instance_type = var.nvidia_instances[count.index]

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = var.node_volume_size
      volume_type           = "gp3"
      iops                  = 3000
      encrypted             = true
      delete_on_termination = true
      throughput            = 125
    }
  }

  dynamic "network_interfaces" {
    for_each = range(0, lookup(var.efa_enabled, var.nvidia_instances[count.index], 0), 1)
    iterator = nic
    content {
      device_index                = nic.value != 0 ? 1 : nic.value
      delete_on_termination       = true
      associate_public_ip_address = false
      interface_type              = "efa"
      network_card_index          = nic.value
    }
  }

  dynamic "capacity_reservation_specification" {
    for_each = var.nvidia_capacity_reservation_id != "" ? [1] : []
    content {
      capacity_reservation_target {
        capacity_reservation_id = var.nvidia_capacity_reservation_id
      }
    }
  }

  key_name = var.key_pair != "" ? var.key_pair : null

  user_data = filebase64("../../user-data.txt")
}

resource "aws_eks_node_group" "nvidia" {
  count = var.karpenter_enabled ? 0 : length(var.nvidia_instances) > 0 ? length(var.nvidia_instances) : 0

  cluster_name    = var.cluster_name
  node_group_name = "nvidia-${count.index}"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.private.*.id
  ami_type        = "AL2023_x86_64_NVIDIA"
  capacity_type   = var.capacity_type

  launch_template {
    id      = aws_launch_template.nvidia[count.index].id
    version = aws_launch_template.nvidia[count.index].latest_version
  }

  scaling_config {
    desired_size = var.node_group_desired
    max_size     = var.node_group_max
    min_size     = var.node_group_min
  }

  taint {
    key    = "fsx.csi.aws.com/agent-not-ready"
    effect = "NO_EXECUTE"
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  dynamic "taint" {
    for_each = var.custom_taints

    content {
      key    = taint.value.key
      value  = taint.value.value
      effect = taint.value.effect
    }
  }

  depends_on = [helm_release.cluster-autoscaler]
}

resource "aws_launch_template" "neuron" {

  count = var.karpenter_enabled ? 0 : length(var.neuron_instances) > 0 ? length(var.neuron_instances) : 0

  instance_type = var.neuron_instances[count.index]

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = var.node_volume_size
      volume_type           = "gp3"
      iops                  = 3000
      encrypted             = true
      delete_on_termination = true
      throughput            = 125
    }
  }

  dynamic "network_interfaces" {
    for_each = range(0, lookup(var.efa_enabled, var.neuron_instances[count.index], 0), 1)
    iterator = nic
    content {
      device_index                = nic.value != 0 ? 1 : nic.value
      delete_on_termination       = true
      associate_public_ip_address = false
      interface_type              = "efa"
      network_card_index          = nic.value
    }
  }

  # Capacity reservation configuration - only if reservation ID is provided
  dynamic "capacity_reservation_specification" {
    for_each = var.neuron_capacity_reservation_id != "" ? [1] : []
    content {
      capacity_reservation_target {
        capacity_reservation_id = var.neuron_capacity_reservation_id
      }
    }
  }

  key_name = var.key_pair != "" ? var.key_pair : null

  user_data = filebase64("../../user-data.txt")
}

resource "aws_eks_node_group" "neuron" {
  count = var.karpenter_enabled ? 0 : length(var.neuron_instances) > 0 ? length(var.neuron_instances) : 0

  cluster_name    = var.cluster_name
  node_group_name = "neuron-${count.index}"
  node_role_arn   = aws_iam_role.node_role.arn
  subnet_ids      = aws_subnet.private.*.id
  ami_type        = "AL2023_x86_64_NEURON"
  capacity_type   = var.capacity_type

  launch_template {
    id      = aws_launch_template.neuron[count.index].id
    version = aws_launch_template.neuron[count.index].latest_version
  }

  scaling_config {
    desired_size = var.node_group_desired
    max_size     = var.node_group_max
    min_size     = var.node_group_min
  }

  taint {
    key    = "fsx.csi.aws.com/agent-not-ready"
    effect = "NO_EXECUTE"
  }

  taint {
    key    = "aws.amazon.com/neuron"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  dynamic "taint" {
    for_each = var.custom_taints

    content {
      key    = taint.value.key
      value  = taint.value.value
      effect = taint.value.effect
    }
  }

  depends_on = [helm_release.cluster-autoscaler]
}

module "karpenter" {
  count = var.karpenter_enabled ? 1 : 0

  source  = "terraform-aws-modules/eks/aws//modules/karpenter"
  version = "20.37.0"

  cluster_name = aws_eks_cluster.eks_cluster.id

  irsa_oidc_provider_arn          = aws_iam_openid_connect_provider.eks_oidc_provider.arn
  irsa_namespace_service_accounts = ["${var.karpenter_namespace}:karpenter"]

  create_iam_role      = true
  create_node_iam_role = true
  enable_irsa          = true
  create_access_entry  = false
  iam_role_name        = "${aws_eks_cluster.eks_cluster.id}-karpenter-controller"
  node_iam_role_name   = "${aws_eks_cluster.eks_cluster.id}-karpenter-node"

  node_iam_role_attach_cni_policy = true
  node_iam_role_additional_policies = {
    s3_policy = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
  }

}

resource "kubectl_manifest" "aws_auth" {

  count = var.karpenter_enabled ? 1 : 0

  yaml_body = <<YAML
  apiVersion: v1
  kind: ConfigMap
  metadata:
    name: aws-auth
    namespace: kube-system
  data:
    mapRoles: |
      - rolearn: "${aws_iam_role.node_role.arn}"
        username: system:node:{{EC2PrivateDNSName}}
        groups:
          - system:bootstrappers
          - system:nodes
      - rolearn: "${module.karpenter[0].node_iam_role_arn}"
        username: system:node:{{EC2PrivateDNSName}}
        groups:
          - system:bootstrappers
          - system:nodes
  
  YAML
}


resource "helm_release" "prometheus" {
  count = var.prometheus_enabled ? 1 : 0

  name             = "prometheus"
  chart            = "kube-prometheus-stack"
  cleanup_on_fail  = true
  create_namespace = true
  repository       = "https://prometheus-community.github.io/helm-charts"
  version          = var.prometheus_version
  namespace        = var.prometheus_namespace
  timeout          = 300
  wait             = true

  set {
    name  = "prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues"
    value = false
  }

  depends_on = [helm_release.cluster-autoscaler]

}

resource "helm_release" "kueue" {
  count = var.kueue_enabled ? 1 : 0

  name             = "kueue"
  chart            = "kueue"
  cleanup_on_fail  = true
  create_namespace = true
  repository       = "oci://registry.k8s.io/kueue/charts/"
  version          = var.kueue_version
  namespace        = var.kueue_namespace
  timeout          = 300
  wait             = true

  values = [
    <<-EOT
      enableCertManager: true
      enablePrometheus: true
    EOT
  ]

  depends_on = [helm_release.cluster-autoscaler]

}


resource "helm_release" "karpenter-crd" {
  count = var.karpenter_enabled ? 1 : 0

  name       = "karpenter-crd"
  chart      = "karpenter-crd"
  repository = "oci://public.ecr.aws/karpenter/"
  version    = var.karpenter_version
  namespace  = var.karpenter_namespace
}

resource "helm_release" "karpenter" {
  count = var.karpenter_enabled ? 1 : 0

  name             = "karpenter"
  chart            = "karpenter"
  cleanup_on_fail  = true
  create_namespace = true
  repository       = "oci://public.ecr.aws/karpenter/"
  version          = var.karpenter_version
  namespace        = var.karpenter_namespace
  timeout          = 300
  wait             = true

  values = [
    <<-EOT
      controller:
        resources:
          limits:
            cpu: 1
            memory: 2Gi
          requests:
            cpu: 1
            memory: 2Gi
      settings:
        clusterName: "${aws_eks_cluster.eks_cluster.id}"
        clusterEndpoint: "${aws_eks_cluster.eks_cluster.endpoint}"
        interruptionQueue: "${module.karpenter[0].queue_name}"
      serviceAccount:
        annotations:
          eks.amazonaws.com/role-arn: "${module.karpenter[0].iam_role_arn}"
      webhook:
        enabled: false
    EOT
  ]

  depends_on = [helm_release.karpenter-crd, helm_release.cluster-autoscaler]

}

resource "helm_release" "karpenter_components" {
  count = var.karpenter_enabled ? 1 : 0

  chart     = "${var.local_helm_repo}/karpenter-components"
  name      = "karpenter-components"
  version   = "1.0.7"
  namespace = var.karpenter_namespace

  values = [
    <<-EOT
      namespace: "${var.karpenter_namespace}"
      role_name: "${module.karpenter[0].node_iam_role_name}"
      cluster_id: "${aws_eks_cluster.eks_cluster.id}"
      consolidate_after: "${var.karpenter_consolidate_after}"
      capacity_type: "${var.karpenter_capacity_type}"
      max_pods: "${var.karpenter_max_pods}"
    EOT
  ]

  depends_on = [helm_release.karpenter]
}

resource "helm_release" "neuron_helm_chart" {
  chart     = "oci://public.ecr.aws/neuron/neuron-helm-chart"
  name      = "neuron-helm-chart"
  version   = "1.1.1"
  namespace = "kube-system"

  values = [
    <<-EOT
      scheduler:
        enabled: true
        customScheduler:
          fullnameOverride: "neuron-scheduler"
      npd:
        enabled: false
    EOT
  ]

  depends_on = [helm_release.cluster-autoscaler]
}

resource "helm_release" "nvidia_device_plugin" {
  chart     = "${var.local_helm_repo}/nvidia-device-plugin"
  name      = "nvidia-device-plugin"
  version   = "1.0.0"
  namespace = "kube-system"

  set {
    name  = "namespace"
    value = "kube-system"
  }
}

resource "kubernetes_namespace" "istio_system" {
  metadata {
    labels = {
      istio-operator-managed = "Reconcile"
      istio-injection        = "disabled"
    }

    name = "istio-system"
  }

  depends_on = [helm_release.cluster-autoscaler]
}

module "istio" {
  source = "./istio"

  istio_system_namespace = kubernetes_namespace.istio_system.metadata[0].name
  auth_namespace         = kubernetes_namespace.auth.metadata[0].name

  depends_on = [module.cert_manager]
}

locals {
  istio_repo_url     = "https://istio-release.storage.googleapis.com/charts"
  istio_repo_version = "1.26.0"
}

resource "kubernetes_namespace" "ingress" {
  metadata {
    labels = {
      istio-injection = "enabled"
    }

    name = var.ingress_namespace
  }

  depends_on = [helm_release.cluster-autoscaler]
}

resource "helm_release" "istio-ingressgateway" {
  name        = "istio-ingressgateway"
  chart       = "gateway"
  version     = local.istio_repo_version
  repository  = local.istio_repo_url
  namespace   = kubernetes_namespace.ingress.metadata[0].name
  description = "Istio ingressgateway"
  wait        = true

  values = [
    <<-EOT
      service:
        type: ClusterIP
        ports:
        - name: status-port
          port: 15021
          protocol: TCP
          targetPort: 15021
        - name: http2
          port: 80
          protocol: TCP
          targetPort: 8080
        - name: https
          port: 443
          protocol: TCP
          targetPort: 8443
        - name: tcp
          port: 31400
          protocol: TCP
          targetPort: 31400
        - name: tls
          port: 15443
          protocol: TCP
          targetPort: 15443
    EOT
  ]

  depends_on = [module.istio]
}

resource "helm_release" "cluster-issuer" {
  name      = "cluster-issuer"
  chart     = "${var.local_helm_repo}/cluster-issuer"
  version   = "1.0.0"
  namespace = "cert-manager"

  values = [
    <<-EOT
      cluster_issuer:
        name: ${var.cluster_issuer}
    EOT
  ]

  depends_on = [module.cert_manager]
}

resource "helm_release" "istio-ingress" {
  name      = "istio-ingress"
  chart     = "${var.local_helm_repo}/istio-ingress"
  version   = "1.0.0"
  namespace = kubernetes_namespace.ingress.metadata[0].name

  values = [
    <<-EOT
      ingress:
        namespace: "${var.ingress_namespace}"
        gateway: "${var.ingress_gateway}"
      healthcheck:
        port: 8080
        path: "/healthcheck"
      cluster_issuer:
        name: "${var.cluster_issuer}"
    EOT
  ]

  depends_on = [helm_release.cluster-issuer, helm_release.istio-ingressgateway]
}

resource "aws_iam_role" "user_profile_role" {
  name = "${aws_eks_cluster.eks_cluster.id}-user-profile"

  assume_role_policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
      {
      "Effect": "Allow",
      "Principal": {
          "Federated": "${aws_iam_openid_connect_provider.eks_oidc_provider.arn}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
          "StringEquals": {
          "${substr(aws_eks_cluster.eks_cluster.identity[0].oidc[0].issuer, 8, -1)}:aud": "sts.amazonaws.com"
          }
      }
      }
  ]
}
POLICY
}

resource "aws_iam_role_policy" "user_profile_policy" {
  name = "user-profile-policy"
  role = aws_iam_role.user_profile_role.id

  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Action" : [
          "s3:Get*",
          "s3:List*",
          "s3:PutObject*",
          "s3:DeleteObject*"
        ],
        "Resource" : [
          "arn:aws:s3:::${local.s3_bucket}",
          "arn:aws:s3:::${local.s3_bucket}/*",
          "arn:aws:s3:::sagemaker-${var.region}-${data.aws_caller_identity.current.account_id}",
          "arn:aws:s3:::sagemaker-${var.region}-${data.aws_caller_identity.current.account_id}/*"
        ],
        "Effect" : "Allow"
      },
      {
        "Action" : [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:GetRepositoryPolicy",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages",
          "ecr:BatchGetImage"
        ],
        "Resource" : ["*"],
        "Effect" : "Allow"
      }
    ]
  })
}

module "profiles-controller-irsa" {
  source  = "aws-ia/eks-blueprints-addon/aws"
  version = "1.1.1"

  # Disable helm release
  create_release = false

  # IAM role for service account (IRSA)
  create_role   = true
  create_policy = true
  role_name     = substr("${aws_eks_cluster.eks_cluster.id}-profiles-controller", 0, 38)
  policy_name   = substr("${aws_eks_cluster.eks_cluster.id}-profiles-controller", 0, 38)
  policy_statements = [
    {
      sid       = "statement0"
      effect    = "Allow"
      actions   = ["iam:GetRole", "iam:UpdateAssumeRolePolicy"],
      resources = ["*"]
    }
  ]

  oidc_providers = {
    this = {
      provider_arn    = aws_iam_openid_connect_provider.eks_oidc_provider.arn
      namespace       = kubernetes_namespace.kubeflow.metadata[0].name
      service_account = "profiles-controller-service-account"
    }
  }

  tags = var.tags
}

resource "helm_release" "ebs-sc" {
  name    = "ebs-sc"
  chart   = "${var.local_helm_repo}/ebs-sc"
  version = "1.0.2"
  wait    = "false"

  depends_on = [module.eks_blueprints_addons]
}

resource "helm_release" "mpi-operator" {
  name      = "mpi-operator"
  chart     = "${var.local_helm_repo}/mpi-operator"
  version   = "2.1.0"
  namespace = kubernetes_namespace.kubeflow.metadata[0].name

  set {
    name  = "namespace"
    value = kubernetes_namespace.kubeflow.metadata[0].name
  }

  depends_on = [helm_release.cluster-autoscaler]
}

resource "null_resource" "git_clone" {
  triggers = {
    always_run = "${timestamp()}"
  }

  provisioner "local-exec" {
    command = <<-EOT
      rm -rf /tmp/lws || true
      git clone https://github.com/kubernetes-sigs/lws.git /tmp/lws
      cd /tmp/lws
      git fetch origin cd3a73a3f5e00cfa9cd2c1c92f00f6189d88073e
      git reset --hard cd3a73a3f5e00cfa9cd2c1c92f00f6189d88073e
    EOT
  }
}

resource "helm_release" "lws" {
  name      = "lws"
  chart     = "/tmp/lws/charts/lws"
  version   = "0.1.0"
  namespace = kubernetes_namespace.lws-system.metadata[0].name

  set {
    name  = "image.manager.tag"
    value = "v0.7.0"
  }
  depends_on = [null_resource.git_clone]
}

module "kubeflow-components" {
  source = "./kubeflow"

  kubeflow_namespace          = kubernetes_namespace.kubeflow.metadata[0].name
  ingress_gateway             = var.ingress_gateway
  ingress_namespace           = kubernetes_namespace.ingress.metadata[0].name
  ingress_sa                  = "istio-ingressgateway"
  static_email                = var.static_email
  user_profile_role_arn       = aws_iam_role.user_profile_role.arn
  profile_controller_role_arn = module.profiles-controller-irsa.iam_role_arn
  kubeflow_user_profile       = "kubeflow-user-example-com"
  kubeflow_platform_enabled   = var.kubeflow_platform_enabled

  efs_fs_id = aws_efs_file_system.fs.id
  fsx = {
    fs_id      = aws_fsx_lustre_file_system.fs.id
    mount_name = aws_fsx_lustre_file_system.fs.mount_name
    dns_name   = "${aws_fsx_lustre_file_system.fs.id}.fsx.${var.region}.amazonaws.com"
  }

  local_helm_repo = var.local_helm_repo

  depends_on = [
    helm_release.istio-ingress,
    aws_fsx_data_repository_association.this,
    aws_efs_file_system.fs
  ]
}

resource "random_password" "static_password" {
  length  = 16
  special = true
}
resource "random_string" "static_user_id" {
  length  = 16
  special = false
}

resource "random_password" "oidc_client_secret" {
  length  = 32
  special = false
}

locals {
  oidc_client_id = "oauth2-proxy"
}

resource "helm_release" "dex" {
  chart     = "${var.local_helm_repo}/dex"
  name      = "dex"
  version   = "1.0.0"
  namespace = kubernetes_namespace.auth.metadata[0].name

  values = [
    <<-EOT
      dex:
        namespace: ${kubernetes_namespace.auth.metadata[0].name}
        user:
          email: "${var.static_email}"
          username: "${var.static_username}"
          userid: "${random_string.static_user_id.result}"
          bcrypt_hash: "${random_password.static_password.bcrypt_hash}"
        oidc:
          client_id: "${local.oidc_client_id}"
          client_secret: "${random_password.oidc_client_secret.result}"
      ingress:
        namespace: ${kubernetes_namespace.ingress.metadata[0].name}
        gateway: ${var.ingress_gateway}
    EOT
  ]

  depends_on = [helm_release.istio-ingress]

}

resource "helm_release" "oauth2-proxy" {
  name        = "oauth2-proxy"
  chart       = "oauth2-proxy"
  version     = "6.23.1"
  repository  = "https://oauth2-proxy.github.io/manifests"
  namespace   = kubernetes_namespace.auth.metadata[0].name
  description = "Oauth2 proxy"
  wait        = true

  values = [
    <<-EOT
      config:
        clientID: "${local.oidc_client_id}"
        clientSecret: "${random_password.oidc_client_secret.result}"
        configFile: |-
          provider = "oidc"
          oidc_issuer_url = "https://istio-ingressgateway.${kubernetes_namespace.ingress.metadata[0].name}.svc.cluster.local/dex"

          request_logging = true
          auth_logging = true
      
          ssl_insecure_skip_verify = true
          set_authorization_header = true
          set_xauthrequest = true
          cookie_samesite = "lax"

          email_domains = ["*"]
          skip_provider_button = true
          upstreams = [ "static://200" ]
    EOT
  ]
  depends_on = [helm_release.dex]
}

resource "helm_release" "oauth2-proxy-route" {
  chart     = "${var.local_helm_repo}/oauth2-proxy-route"
  name      = "oauth2-proxy-route"
  version   = "1.0.0"
  namespace = kubernetes_namespace.auth.metadata[0].name

  values = [
    <<-EOT
      oauth2_proxy:
        namespace: ${kubernetes_namespace.auth.metadata[0].name}
      ingress:
        namespace: ${kubernetes_namespace.ingress.metadata[0].name}
        gateway: ${var.ingress_gateway}
    EOT
  ]

  depends_on = [helm_release.oauth2-proxy]
}

resource "aws_iam_role" "ack_sagemaker_role" {
  count = var.ack_sagemaker_enabled ? 1 : 0
  name  = "${var.cluster_name}-ack-sagemaker-role"

  assume_role_policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
      {
      "Effect": "Allow",
      "Principal": {
          "Federated": "${aws_iam_openid_connect_provider.eks_oidc_provider.arn}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
          "StringEquals": {
          "${substr(aws_eks_cluster.eks_cluster.identity[0].oidc[0].issuer, 8, -1)}:aud": "sts.amazonaws.com"
          }
      }
      }
  ]
}
POLICY
}

resource "aws_iam_role_policy_attachment" "ack_sagemaker_AmazonSageMakerFullAccess" {
  count      = var.ack_sagemaker_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  role       = aws_iam_role.ack_sagemaker_role[count.index].name
}


resource "helm_release" "ack_sagemaker_controller" {
  count = var.ack_sagemaker_enabled ? 1 : 0

  name             = "sagemaker-controller"
  chart            = "sagemaker-chart"
  cleanup_on_fail  = true
  create_namespace = true
  repository       = "oci://public.ecr.aws/aws-controllers-k8s/"
  version          = "1.2.15"
  namespace        = "kube-system"
  timeout          = 300
  wait             = true

  values = [
    <<-EOT
      aws:
        region: ${var.region}
      serviceAccount:
        annotations:
          eks.amazonaws.com/role-arn: ${aws_iam_role.ack_sagemaker_role[count.index].arn}
    EOT
  ]

  depends_on = [helm_release.cluster-autoscaler]

}

resource "helm_release" "kserve-crd" {
  count = var.kserve_enabled ? 1 : 0

  name             = "kserve-crd"
  chart            = "kserve-crd"
  cleanup_on_fail  = true
  create_namespace = true
  repository       = "oci://ghcr.io/kserve/charts/"
  version          = var.kserve_version
  namespace        = var.kserve_namespace
  timeout          = 300
  wait             = true

  depends_on = [helm_release.cluster-autoscaler]

}

resource "helm_release" "kserve" {
  count = var.kserve_enabled ? 1 : 0

  name             = "kserve"
  chart            = "kserve"
  cleanup_on_fail  = true
  create_namespace = true
  repository       = "oci://ghcr.io/kserve/charts/"
  version          = var.kserve_version
  namespace        = var.kserve_namespace
  timeout          = 300
  wait             = true

  values = [
    <<-EOT
      kserve:
        controller:
          deploymentMode: "RawDeployment"
          gateway:
            ingressGateway:
              className:  "istio"
              createGateway: "false"
              kserveGateway: "ingress/istio-ingressgateway"
    EOT
  ]

  depends_on = [helm_release.kserve-crd]

}

resource "helm_release" "airflow" {
  count = var.airflow_enabled ? 1 : 0

  name             = "airflow"
  chart            = "airflow"
  cleanup_on_fail  = true
  create_namespace = true
  repository       = "https://airflow.apache.org/"
  version          = var.airflow_version
  namespace        = var.airflow_namespace
  timeout          = 300
  wait             = true

  values = [
    <<-EOT
      createUserJob:
        useHelmHooks: false
        applyCustomEnv: false
      migrateDatabaseJob:
        useHelmHooks: false
        applyCustomEnv: false
    EOT
  ]

  depends_on = [helm_release.cluster-autoscaler]

}

resource "helm_release" "dcgm_exporter" {
  count = var.dcgm_exporter_enabled ? 1 : 0

  name             = "dcgm-exporter"
  chart            = "dcgm-exporter"
  cleanup_on_fail  = true
  create_namespace = true
  repository       = "https://nvidia.github.io/dcgm-exporter/helm-charts/"
  version          = "4.0.4"
  namespace        = "kube-system"
  timeout          = 300
  wait             = true

  values = [
    <<-EOT
      nodeSelector:
	      karpenter.k8s.aws/instance-gpu-manufacturer: nvidia
      tolerations:
 	      - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
    EOT
  ]

  depends_on = [helm_release.cluster-autoscaler]

}

module "slurm" {
  count  = var.slurm_enabled ? 1 : 0
  source = "./slurm"

  slurm_namespace          = var.slurm_namespace
  efs_fs_id                = aws_efs_file_system.fs.id
  root_ssh_authorized_keys = var.slurm_root_ssh_authorized_keys
  storage_capacity         = var.slurm_storage_capacity
  storage_type             = var.slurm_storage_type
  local_helm_repo          = var.local_helm_repo
  login_enabled            = var.slurm_login_enabled
  eks_cluster_id           = aws_eks_cluster.eks_cluster.id

  db_max_capacity = var.slurm_db_max_capacity
  db_subnet_ids   = aws_subnet.private.*.id
  db_vpc_id       = aws_vpc.vpc.id
  db_port         = 3306

  fsx = {
    fs_id      = aws_fsx_lustre_file_system.fs.id
    mount_name = aws_fsx_lustre_file_system.fs.mount_name
    dns_name   = "${aws_fsx_lustre_file_system.fs.id}.fsx.${var.region}.amazonaws.com"
  }

  depends_on = [
    aws_efs_file_system.fs,
    aws_fsx_lustre_file_system.fs,
    module.eks_blueprints_addons
  ]
}

module "mlflow" {
  count  = var.mlflow_enabled ? 1 : 0
  source = "./mlflow"

  mlflow_namespace      = var.mlflow_namespace
  mlflow_version        = var.mlflow_version
  force_destroy_bucket  = var.mlflow_force_destroy_bucket
  eks_cluster_id        = aws_eks_cluster.eks_cluster.id
  eks_oidc_provider_arn = aws_iam_openid_connect_provider.eks_oidc_provider.arn
  eks_oidc_issuer       = substr(aws_eks_cluster.eks_cluster.identity[0].oidc[0].issuer, 8, -1)
  admin_username        = var.mlflow_admin_username
  admin_password        = random_password.static_password.result
  db_max_capacity       = var.mlflow_db_max_capacity
  db_subnet_ids         = aws_subnet.private.*.id
  db_vpc_id             = aws_vpc.vpc.id
  db_port               = 5432

  depends_on = [
    module.eks_blueprints_addons
  ]
}
