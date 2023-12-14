# BEGIN variables

variable "credentials" {
 description = "path to the aws credentials file"
 default = "~/.aws/credentials"
 type    = string
}

variable "profile" {
 description = "name of the aws config profile"
 default = "default"
 type    = string
}

variable "cluster_name" {
  description = "unique name of the eks cluster"
  type    = string
}

variable "k8s_version" {
  description = "kubernetes version"
  default = "1.28"
  type    = string
}

variable "region" {
 description = "name of aws region to use"
 type    = string
}

variable "azs" {
 description = "list of aws availabilty zones in aws region"
 type = list
}


variable "cidr_vpc" {
 description = "RFC 1918 CIDR range for EKS cluster VPC"
 default = "192.168.0.0/16"
 type    = string
}

variable "cidr_private" {
 description = "RFC 1918 CIDR range list for EKS cluster VPC subnets"
 default = ["192.168.64.0/18", "192.168.128.0/18", "192.168.192.0/18"]
 type    = list 
}

variable "cidr_public" {
 description = "RFC 1918 CIDR range list for EKS cluster VPC subnets"
 default = ["192.168.0.0/24", "192.168.1.0/24", "192.168.2.0/24"]
 type    = list 
}

variable "efs_performance_mode" {
   default = "generalPurpose"
   type = string
}

variable "efs_throughput_mode" {
   description = "EFS performance mode"
   default = "bursting"
   type = string
}

variable "import_path" {
  description = "fsx for lustre s3 import path"
  type = string
  default = ""
}

variable "inference_max" {
  description = "Maximum inference nodes"
  type = string
  default = "2"
}

variable "inference_instance_type" {
  description = "GPU enabled instance types for inference."
  default = "g5.xlarge"
  type = string
}

variable "kubeflow_namespace" {
  description = "Kubeflow namespace"
  default = "kubeflow"
  type = string
}


# END variables

terraform {
  required_version = ">= 1.5.1"

  required_providers {
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.14.0"
    }
  }
}

provider "aws" {
  region                  = var.region
  shared_credentials_files = [var.credentials]
  profile                 = var.profile
}

provider "kubectl" {
  host                   = aws_eks_cluster.eks_cluster.endpoint
  cluster_ca_certificate = base64decode(aws_eks_cluster.eks_cluster.certificate_authority[0].data)
  load_config_file       = false

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws-iam-authenticator"
    args = [
      "token",
      "-i",
      aws_eks_cluster.eks_cluster.id,
    ]
  }
}

provider "helm" {
  kubernetes {
    host                   = aws_eks_cluster.eks_cluster.endpoint
    cluster_ca_certificate = base64decode(aws_eks_cluster.eks_cluster.certificate_authority[0].data)
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      args        = ["eks", "get-token", "--cluster-name", aws_eks_cluster.eks_cluster.id]
      command     = "aws"
    }
  }
}

provider "kubernetes" {
  host                   = aws_eks_cluster.eks_cluster.endpoint
  cluster_ca_certificate = base64decode(aws_eks_cluster.eks_cluster.certificate_authority[0].data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    args        = ["eks", "get-token", "--cluster-name", aws_eks_cluster.eks_cluster.id]
    command     = "aws"
  }
}


data "aws_caller_identity" "current" {}

resource "aws_vpc" "vpc" {
  cidr_block = var.cidr_vpc
  enable_dns_support = true
  enable_dns_hostnames  = true

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
    Name = "${var.cluster_name}-subnet-${count.index}",
    "kubernetes.io/cluster/${var.cluster_name}" = "shared",
    "kubernetes.io/role/internal-elb": "1"
  }

}

resource "aws_subnet" "public" {
  count = length(var.azs) 

  availability_zone = var.azs[count.index]
  cidr_block        = var.cidr_public[count.index]
  vpc_id            = aws_vpc.vpc.id

  tags = {
    Name = "${var.cluster_name}-subnet-${count.index}",
    "kubernetes.io/cluster/${var.cluster_name}" = "shared",
    "kubernetes.io/role/elb" = "1"
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

  depends_on = [aws_internet_gateway.igw, aws_subnet.public]
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_nat_gateway.ngw.id
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
 throughput_mode = var.efs_throughput_mode
 encrypted = true


  tags = {
    Name = var.cluster_name
  }
}


resource "aws_security_group" "efs_sg" {
  name = "${var.cluster_name}-efs-sg"
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
  count = length(var.azs) 
  file_system_id = aws_efs_file_system.fs.id

  subnet_id      = aws_subnet.private.*.id[count.index] 
  security_groups = [aws_security_group.efs_sg.id] 
}

resource "aws_security_group" "fsx_lustre_sg" {
  name = "${var.cluster_name}-fsx-lustre-sg"
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
  import_path      = var.import_path != "" ? var.import_path: null 
  storage_capacity = 1200
  subnet_ids       = [aws_subnet.private[0].id]
  deployment_type = "SCRATCH_2"
  security_group_ids = [aws_security_group.fsx_lustre_sg.id] 

  tags = {
    Name = var.cluster_name
  }
}

locals {
  use_k8s_version = substr(var.k8s_version, 0, 3) == "1.1" ? "1.28": var.k8s_version
  cluster_autoscaler_version=substr(local.use_k8s_version, 0, 4)
}

resource "aws_eks_cluster" "eks_cluster" {
  name            = var.cluster_name
  role_arn        = aws_iam_role.cluster_role.arn
  version	        = local.use_k8s_version


  vpc_config {
    subnet_ids         = flatten([aws_subnet.private.*.id])
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
    aws_iam_role_policy_attachment.cluster_AmazonEKSServicePolicy,
  ]

  provisioner "local-exec" {
    when    = destroy
    command = "kubectl config unset current-context"
  }

  provisioner "local-exec" {
    command = "aws --region ${var.region} eks update-kubeconfig --name ${aws_eks_cluster.eks_cluster.name}"
  }

  provisioner "local-exec" {
    command = "kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -k \"github.com/kubernetes-sigs/aws-fsx-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.1\""
  }

  provisioner "local-exec" {
    command = "kubectl apply -k \"github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.7\""
  }

  provisioner "local-exec" {
    command = "kubectl create namespace ${var.kubeflow_namespace}"
  }

}

data "tls_certificate" "this" {
  url = aws_eks_cluster.eks_cluster.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks_oidc_provider" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.this.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.eks_cluster.identity[0].oidc[0].issuer

  depends_on = [
    aws_eks_cluster.eks_cluster
  ]
  
}


locals {
  cluster_oidc_path = format("oidc.eks.%s.amazonaws.com/id/%s", "${var.region}", join("", regex("https://([^.]+).+", "${aws_eks_cluster.eks_cluster.endpoint}"))) 
}

resource "aws_iam_role" "cluster_autoscaler_role" {
  name = "${aws_eks_cluster.eks_cluster.id}-cluster-autoscaler-role"

  assume_role_policy = jsonencode({
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Federated": "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/${local.cluster_oidc_path}"
        },
        "Action": "sts:AssumeRoleWithWebIdentity",
        "Condition": {
          "StringEquals": {
            "${local.cluster_oidc_path}:aud": "sts.amazonaws.com"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "cluster_autoscaler_policy" {
   name = "cluster-autoscaler-policy"
   role = aws_iam_role.cluster_autoscaler_role.id

    policy = jsonencode({
      "Version": "2012-10-17",
      "Statement": [
        {
            "Action": [
                "autoscaling:DescribeAutoScalingGroups",
                "autoscaling:DescribeAutoScalingInstances",
                "autoscaling:DescribeLaunchConfigurations",
                "autoscaling:DescribeTags",
                "autoscaling:SetDesiredCapacity",
                "autoscaling:TerminateInstanceInAutoScalingGroup",
                "ec2:DescribeLaunchTemplateVersions"
            ],
            "Resource": "*",
            "Effect": "Allow"
        }
      ]
    })
}

resource "null_resource" "eks_cluster_autoscaler_role" {
  
  triggers = {
    cluster_autoscaler_role = "${aws_iam_role.cluster_autoscaler_role.arn}"
  }

  provisioner "local-exec" {
    command = "kubectl apply -f  https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml"
  }

  provisioner "local-exec" {
    command = <<-EOT
      kubectl -n kube-system patch deployment cluster-autoscaler --patch \
      '{"spec": { "template": { "metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"}}, "spec": { "containers": [{ "image": "registry.k8s.io/autoscaling/cluster-autoscaler:v${local.cluster_autoscaler_version}.0", "name": "cluster-autoscaler", "resources": { "requests": {"cpu": "100m", "memory": "300Mi"}}, "command": [ "./cluster-autoscaler", "--v=4", "--stderrthreshold=info", "--cloud-provider=aws", "--skip-nodes-with-local-storage=false", "--expander=least-waste", "--node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/${var.cluster_name}", "--balance-similar-node-groups", "--skip-nodes-with-system-pods=false" ]}]}}}}'
    EOT
  }

  provisioner "local-exec" {
    command = <<-EOT
      kubectl patch ServiceAccount cluster-autoscaler -n kube-system --patch \
      '{"metadata":{"annotations":{"eks.amazonaws.com/role-arn": "${aws_iam_role.cluster_autoscaler_role.arn}"}}}'
    EOT
  }

  depends_on = [
    aws_iam_openid_connect_provider.eks_oidc_provider
  ]
  
}

resource "kubectl_manifest" "fsx_sc" {

  depends_on = [
    aws_eks_cluster.eks_cluster
  ]

  yaml_body = <<YAML
  kind: StorageClass
  apiVersion: storage.k8s.io/v1
  metadata:
    name: fsx-sc
  provisioner: fsx.csi.aws.com
  YAML
}

resource "kubectl_manifest" "pv_fsx" {

  depends_on = [
    kubectl_manifest.fsx_sc
  ]

  yaml_body = <<YAML
  apiVersion: v1
  kind: PersistentVolume
  metadata:
    name: pv-fsx
  spec:
    capacity:
      storage: 1200Gi 
    volumeMode: Filesystem
    accessModes:
      - ReadWriteMany
    mountOptions:
      - noatime
      - flock
    persistentVolumeReclaimPolicy: Retain
    csi:
      driver: fsx.csi.aws.com
      volumeHandle: "${aws_fsx_lustre_file_system.fs.id}"
      volumeAttributes:
        dnsname: "${aws_fsx_lustre_file_system.fs.id}.fsx.${var.region}.amazonaws.com"
        mountname: "${aws_fsx_lustre_file_system.fs.mount_name}"
  YAML
}

resource "kubectl_manifest" "pvc_fsx" {

  depends_on = [
    kubectl_manifest.pv_fsx
  ]

  yaml_body = <<YAML
  apiVersion: v1
  kind: PersistentVolumeClaim
  metadata:
    name: pv-fsx
    namespace: "${var.kubeflow_namespace}"
  spec:
    accessModes:
      - ReadWriteMany
    storageClassName: "" 
    resources:
      requests:
        storage: 1200Gi
    volumeName: pv-fsx
  YAML
}


resource "kubectl_manifest" "efs_sc" {

  depends_on = [
    aws_eks_cluster.eks_cluster
  ]

  yaml_body = <<YAML
  kind: StorageClass
  apiVersion: storage.k8s.io/v1
  metadata:
    name: efs-sc
  provisioner: efs.csi.aws.com
  YAML
}

resource "kubectl_manifest" "pv_efs" {

  depends_on = [
    kubectl_manifest.efs_sc
  ]

  yaml_body = <<YAML
  apiVersion: v1
  kind: PersistentVolume
  metadata:
    name: pv-efs
  spec:
    capacity:
      storage: 1000Gi
    volumeMode: Filesystem
    accessModes:
      - ReadWriteMany
    persistentVolumeReclaimPolicy: Retain
    storageClassName: efs-sc
    csi:
      driver: efs.csi.aws.com
      volumeHandle: "${aws_efs_file_system.fs.id}"
  YAML
}

resource "kubectl_manifest" "pvc_efs" {

  depends_on = [
    kubectl_manifest.pv_efs
  ]

  yaml_body = <<YAML
  apiVersion: v1
  kind: PersistentVolumeClaim
  metadata:
    name: pv-efs
    namespace: "${var.kubeflow_namespace}"
  spec:
    accessModes:
      - ReadWriteMany
    storageClassName: efs-sc 
    resources:
      requests:
        storage: 100Gi
    YAML
}


resource "aws_iam_role" "node_role" {
  name = "${aws_eks_cluster.eks_cluster.id}-node-role"

  assume_role_policy = jsonencode({
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Service": "ec2.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "node_autoscaler_policy" {
   name = "node-autoscaler-policy"
   role = aws_iam_role.node_role.id

    policy = jsonencode({
      "Version": "2012-10-17",
      "Statement": [
        {
            "Action": [
                "autoscaling:DescribeAutoScalingGroups",
                "autoscaling:DescribeAutoScalingInstances",
                "autoscaling:DescribeLaunchConfigurations",
                "autoscaling:DescribeTags",
                "autoscaling:SetDesiredCapacity",
                "autoscaling:TerminateInstanceInAutoScalingGroup",
                "ec2:DescribeLaunchTemplateVersions",
                "eks:DescribeNodegroup"
            ],
            "Resource": "*",
            "Effect": "Allow"
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
  cluster_name    = var.cluster_name 
  node_group_name = "system" 
  node_role_arn   = aws_iam_role.node_role.arn 
  subnet_ids      = aws_subnet.private.*.id 
  instance_types  = ["m7a.large", "c7a.large"]
  disk_size       = 40 
  ami_type        = "AL2_x86_64"

  scaling_config {
    desired_size = 2 
    max_size     = 8 
    min_size     = 2 
  }

  depends_on = [
    aws_eks_cluster.eks_cluster,
    aws_iam_role.node_role,
    aws_iam_role_policy_attachment.node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_AmazonEC2ContainerRegistryReadOnly,
    aws_iam_role_policy_attachment.node_AmazonS3ReadOnlyPolicy
  ]

}

resource "aws_eks_node_group" "inference_ng" {
  cluster_name    = var.cluster_name 
  node_group_name = "inference" 
  node_role_arn   = aws_iam_role.node_role.arn 
  subnet_ids      = aws_subnet.private.*.id 
  instance_types  = split(",", var.inference_instance_type)
  disk_size       = 100
  ami_type        = "AL2_x86_64_GPU"

  scaling_config {
    desired_size = 0 
    max_size     = var.inference_max
    min_size     = 0 
  }

  taint {
    key = "nvidia.com/gpu"
    value = "true"
    effect = "NO_SCHEDULE"
  }
  
  depends_on = [
    aws_eks_cluster.eks_cluster,
    aws_iam_role.node_role,
    aws_iam_role_policy_attachment.node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_AmazonEC2ContainerRegistryReadOnly,
    aws_iam_role_policy_attachment.node_AmazonS3ReadOnlyPolicy
  ]

}

module "eks_blueprints_addons" {
  depends_on = [
    aws_eks_node_group.system_ng
  ]

  source = "aws-ia/eks-blueprints-addons/aws"
  version = "~> 1.12" #ensure to update this to the latest/desired version

  cluster_name      = aws_eks_cluster.eks_cluster.id
  cluster_endpoint  = aws_eks_cluster.eks_cluster.endpoint
  cluster_version   = aws_eks_cluster.eks_cluster.version
  oidc_provider_arn = aws_iam_openid_connect_provider.eks_oidc_provider.arn

  enable_aws_load_balancer_controller    = true
}


locals {
  summary = <<SUMMARY

  EKS Cluster Summary: 
  	vpc:    ${aws_vpc.vpc.id}
  	subnets: ${join(",", aws_subnet.private.*.id)}
  	cluster security group: ${aws_eks_cluster.eks_cluster.vpc_config[0].cluster_security_group_id}
  	endpoint: ${aws_eks_cluster.eks_cluster.endpoint}
  EKS NodeGroup Summary:
    node role: ${aws_iam_role.node_role.arn}
    system: ${aws_eks_node_group.system_ng.arn} 
    inference: ${aws_eks_node_group.inference_ng.arn}
  EFS Summary:
  	file system id: ${aws_efs_file_system.fs.id}
  	dns: ${aws_efs_file_system.fs.id}.efs.${var.region}.amazonaws.com
  FSx for Lustre Summary:
  	file system id: ${aws_fsx_lustre_file_system.fs.id}
        mount_name: ${aws_fsx_lustre_file_system.fs.mount_name}
SUMMARY
}

output "summary" {
  value = local.summary
}
