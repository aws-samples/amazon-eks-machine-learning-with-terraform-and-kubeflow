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
  default = "1.24"
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
  description = "GPU enabled instance types for inference. Must have 1 GPU."
  default = "g4dn.xlarge,g5.xlarge"
  type = string
}


# END variables

provider "aws" {
  region                  = var.region
  shared_credentials_files = [var.credentials]
  profile                 = var.profile
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
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }

}

resource "aws_subnet" "public" {
  count = length(var.azs) 

  availability_zone = var.azs[count.index]
  cidr_block        = var.cidr_public[count.index]
  vpc_id            = aws_vpc.vpc.id

  tags = {
    Name = "${var.cluster_name}-subnet-${count.index}",
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
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

resource "aws_security_group" "cluster_sg" {
  name = "${var.cluster_name}-cluster-sg"
  description = "Cluster communication with worker nodes"
  vpc_id      = aws_vpc.vpc.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-cluster-sg"
  }
}

resource "aws_security_group_rule" "cluster_ingress_self" {
  description              = "Allow cluster control plabe to communicate with each other"
  from_port                = 0
  protocol                 = "-1"
  security_group_id        = aws_security_group.cluster_sg.id
  source_security_group_id = aws_security_group.cluster_sg.id
  to_port                  = 65535
  type                     = "ingress"
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
  use_k8s_version = substr(var.k8s_version, 0, 3) == "1.1" ? "1.24": var.k8s_version
  cluster_autoscaler_version=substr(local.use_k8s_version, 0, 4)
}

resource "aws_eks_cluster" "eks_cluster" {
  name            = var.cluster_name
  role_arn        = aws_iam_role.cluster_role.arn
  version	        = local.use_k8s_version


  vpc_config {
    security_group_ids = [aws_security_group.cluster_sg.id]
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
    command = "kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -f  https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml"
  }

  provisioner "local-exec" {
    command = <<-EOT
      kubectl -n kube-system patch deployment cluster-autoscaler --patch \
      '{"spec": { "template": { "metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"}}, "spec": { "containers": [{ "image": "k8s.gcr.io/autoscaling/cluster-autoscaler:v${local.cluster_autoscaler_version}.0", "name": "cluster-autoscaler", "resources": { "requests": {"cpu": "100m", "memory": "300Mi"}}, "command": [ "./cluster-autoscaler", "--v=4", "--stderrthreshold=info", "--cloud-provider=aws", "--skip-nodes-with-local-storage=false", "--expander=least-waste", "--node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/${var.cluster_name}", "--balance-similar-node-groups", "--skip-nodes-with-system-pods=false" ]}]}}}}'
    EOT
  }

  provisioner "local-exec" {
    command = "kubectl create namespace kubeflow"
  }

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
    command = <<-EOT
      kubectl patch ServiceAccount cluster-autoscaler -n kube-system --patch \
      '{"metadata":{"annotations":{"eks.amazonaws.com/role-arn": "${aws_iam_role.cluster_autoscaler_role.arn}"}}}'
    EOT
  }

  depends_on = [
    aws_eks_cluster.eks_cluster
  ]
  
}

resource "null_resource" "fsx_id" {
  triggers = {
    fsx_id = "${aws_fsx_lustre_file_system.fs.id}"
  }

  provisioner "local-exec" {
    command = "sed -i -e \"s/volumeHandle: .*/volumeHandle: ${aws_fsx_lustre_file_system.fs.id}/g\" -e \"s/dnsname: .*/dnsname: ${aws_fsx_lustre_file_system.fs.id}.fsx.${var.region}.amazonaws.com/g\" -e \"s/mountname: .*/mountname: ${aws_fsx_lustre_file_system.fs.mount_name}/g\" ../../pv-kubeflow-fsx.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -k \"github.com/kubernetes-sigs/aws-fsx-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-0.8\""
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../fsx-sc.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../pv-kubeflow-fsx.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../pvc-kubeflow-fsx.yaml"
  }

  depends_on = [
    aws_eks_cluster.eks_cluster
  ]
}

resource "null_resource" "efs_id" {
  triggers = {
    efs_id = "${aws_efs_file_system.fs.id}"
  }

  provisioner "local-exec" {
    command = "sed -i -e \"s/volumeHandle: .*/volumeHandle: ${aws_efs_file_system.fs.id}/g\" ../../pv-kubeflow-efs-gp-bursting.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -k \"github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.4\""
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../efs-sc.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../pv-kubeflow-efs-gp-bursting.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../pvc-kubeflow-efs-gp-bursting.yaml"
  }

  depends_on = [
    aws_eks_cluster.eks_cluster
  ]
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
                "ec2:DescribeLaunchTemplateVersions"
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
  instance_types  = ["m5a.large"]
  disk_size       = 40 
  ami_type        = "AL2_x86_64"

  scaling_config {
    desired_size = 2 
    max_size     = 2 
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

  depends_on = [
    aws_eks_cluster.eks_cluster,
    aws_iam_role.node_role,
    aws_iam_role_policy_attachment.node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.node_AmazonEC2ContainerRegistryReadOnly,
    aws_iam_role_policy_attachment.node_AmazonS3ReadOnlyPolicy
  ]

}

locals {
  summary = <<SUMMARY

  EKS Cluster Summary: 
  	vpc:    ${aws_vpc.vpc.id}
  	subnets: ${join(",", aws_subnet.private.*.id)}
  	cluster security group: ${aws_security_group.cluster_sg.id}
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
