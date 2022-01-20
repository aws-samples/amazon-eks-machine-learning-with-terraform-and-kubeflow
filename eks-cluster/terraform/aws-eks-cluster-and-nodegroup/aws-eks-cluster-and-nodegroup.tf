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
  default = "1.21"
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

variable "nodegroup_name" {
  description = "Node group name in cluster"
  type    = string
  default = "ng1"
}


variable "node_volume_size" {
  description = "EKS cluster worker node EBS volume size in GBs"
  default="200"
  type=string
}

variable "node_instance_type" {
  description = "EC2 GPU enabled instance types for EKS cluster worker nodes"
  default = "p3.16xlarge"
  type = string
}

variable "key_pair" {
  description = "Name of EC2 key pair used to launch EKS cluster worker node EC2 instances"
  type = string
}

variable "node_group_desired" {
    description = "EKS worker node auto-scaling group desired size"
    default = "2"
    type = string
}

variable "node_group_max" {
    description = "EKS worker node auto-scaling group maximum"
    default = "2"
    type = string
}

variable "node_group_min" {
    description = "EKS worker node auto-scaling group minimum"
    default = "0" 
    type = string
}

# END variables

provider "aws" {
  region                  = var.region
  shared_credentials_file = var.credentials
  profile                 = var.profile
}

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

resource "aws_eks_cluster" "eks_cluster" {
  name            = var.cluster_name
  role_arn        = aws_iam_role.cluster_role.arn
  version	  = var.k8s_version

  vpc_config {
    security_group_ids = [aws_security_group.cluster_sg.id]
    subnet_ids         = flatten([aws_subnet.private.*.id])
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy,
    aws_iam_role_policy_attachment.cluster_AmazonEKSServicePolicy,
  ]

  provisioner "local-exec" {
    command = "aws --region ${var.region} eks update-kubeconfig --name ${aws_eks_cluster.eks_cluster.name}"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "kubectl config unset current-context"
  }

  provisioner "local-exec" {
    command = "sed -i -e \"s/volumeHandle: .*/volumeHandle: ${aws_efs_file_system.fs.id}/g\" ../../pv-kubeflow-efs-gp-bursting.yaml"
  }

  provisioner "local-exec" {
    command = "sed -i -e \"s/volumeHandle: .*/volumeHandle: ${aws_fsx_lustre_file_system.fs.id}/g\" -e \"s/dnsname: .*/dnsname: ${aws_fsx_lustre_file_system.fs.id}.fsx.${var.region}.amazonaws.com/g\" -e \"s/mountname: .*/mountname: ${aws_fsx_lustre_file_system.fs.mount_name}/g\" ../../pv-kubeflow-fsx.yaml"
  }


  provisioner "local-exec" {
    command = "kubectl create namespace kubeflow"
  }

  provisioner "local-exec" {
    command = "kubectl apply -k \"github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.3\""
  }

  provisioner "local-exec" {
    command = "kubectl apply -k \"github.com/kubernetes-sigs/aws-fsx-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-0.4\""
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../efs-sc.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../fsx-sc.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../pv-kubeflow-efs-gp-bursting.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../pvc-kubeflow-efs-gp-bursting.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../pv-kubeflow-fsx.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -n kubeflow -f ../../pvc-kubeflow-fsx.yaml"
  }

  provisioner "local-exec" {
    command = "kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml"
  }
}

# Nodegroup resources

resource "aws_iam_role" "node_role" {
  name = "${aws_eks_cluster.eks_cluster.id}-node-role"

  assume_role_policy = <<POLICY
{
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
}
POLICY
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

resource "aws_eks_node_group" "ng" {
  cluster_name    = var.cluster_name 
  node_group_name = var.nodegroup_name 
  node_role_arn   = aws_iam_role.node_role.arn 
  subnet_ids      = aws_subnet.private.*.id 
  instance_types  = [var.node_instance_type]
  disk_size       = var.node_volume_size 
  ami_type        = "AL2_x86_64_GPU"

  scaling_config {
    desired_size = var.node_group_desired 
    max_size     = var.node_group_max 
    min_size     = var.node_group_min 
  }

  remote_access {
    ec2_ssh_key = var.key_pair
  }

}

locals {
  summary = <<SUMMARY

  EKS Cluster Summary: 
  	vpc:    ${aws_vpc.vpc.id}
  	subnets: ${join(",", aws_subnet.private.*.id)}
  	cluster security group: ${aws_security_group.cluster_sg.id}
  	endpoint: ${aws_eks_cluster.eks_cluster.endpoint}
  EKS NodeGroup Summary: 
  	arn: ${aws_eks_node_group.ng.arn} 
  EFS Summary:
  	file system id: ${aws_efs_file_system.fs.id}
  	dns: ${aws_efs_file_system.fs.id}.efs.${var.region}.amazonaws.com
  FSx Lustre Summary:
  	file system id: ${aws_fsx_lustre_file_system.fs.id}
        mount_name: ${aws_fsx_lustre_file_system.fs.mount_name}
SUMMARY
}

output "summary" {
  value = local.summary
}
