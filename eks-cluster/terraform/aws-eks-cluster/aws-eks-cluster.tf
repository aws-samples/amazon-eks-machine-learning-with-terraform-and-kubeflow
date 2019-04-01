# BEGIN variables

variable "credentials" {
 description = "path to the aws credentials file"
 default = "~/.aws/credentials"
 type    = "string"
}

variable "profile" {
 description = "name of the aws config profile"
 default = "default"
 type    = "string"
}

variable "cluster_name" {
  description = "unique name of the eks cluster"
  type    = "string"
}

variable "k8s_version" {
  description = "kubernetes version"
  default = "1.11"
  type    = "string"
}

variable "region" {
 description = "name of aws region to use"
 type    = "string"
}

variable "azs" {
 description = "list of aws availabilty zones in aws region"
 type = "list"
}


variable "cidr_vpc" {
 description = "RFC 1918 CIDR range for EKS cluster VPC"
 default = "192.168.0.0/16"
 type    = "string"
}

variable "cidr_subnet" {
 description = "RFC 1918 CIDR range list for EKS cluster VPC subnets"
 default = ["192.168.64.0/18", "192.168.128.0/18", "192.168.192.0/18"]
 type    = "list"
}

variable "efs_performance_mode" {
   default = "generalPurpose"
   type = "string"
}

variable "efs_throughput_mode" {
   description = "EFS performance mode"
   default = "bursting"
   type = "string"
}

# END variables

provider "aws" {
  region                  = "${var.region}"
  shared_credentials_file = "${var.credentials}"
  profile                 = "${var.profile}"
}

resource "aws_vpc" "vpc" {
  cidr_block = "${var.cidr_vpc}"

  tags = {
    Name = "${var.cluster_name}-vpc",
  }
}

resource "aws_subnet" "subnet" {
  count = "${length(var.azs)}" 

  availability_zone = "${var.azs[count.index]}"
  cidr_block        = "${var.cidr_subnet[count.index]}"
  vpc_id            = "${aws_vpc.vpc.id}"

  tags = {
    Name = "${var.cluster_name}-subnet-${count.index}",
  }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = "${aws_vpc.vpc.id}"

  tags = {
    Name = "${var.cluster_name}-igw"
  }
}

resource "aws_route_table" "rt" {
  vpc_id = "${aws_vpc.vpc.id}"

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = "${aws_internet_gateway.igw.id}"
  }

  tags = {
    Name = "${var.cluster_name}-rt"
  }
}

resource "aws_route_table_association" "rta" {
  count = "${length(var.azs)}" 

  subnet_id      = "${aws_subnet.subnet.*.id[count.index]}"
  route_table_id = "${aws_route_table.rt.id}"
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
  role       = "${aws_iam_role.cluster_role.name}"
}

resource "aws_iam_role_policy_attachment" "cluster_AmazonEKSServicePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
  role       = "${aws_iam_role.cluster_role.name}"
}

resource "aws_security_group" "cluster_sg" {
  name = "${var.cluster_name}-cluster-sg"
  description = "Cluster communication with worker nodes"
  vpc_id      = "${aws_vpc.vpc.id}"

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
  security_group_id        = "${aws_security_group.cluster_sg.id}"
  source_security_group_id = "${aws_security_group.cluster_sg.id}"
  to_port                  = 65535
  type                     = "ingress"
}

resource "aws_efs_file_system" "fs" {

 performance_mode = "${var.efs_performance_mode}"
 
 throughput_mode = "${var.efs_throughput_mode}"


  tags = {
    Name = "${var.cluster_name}"
  }
}

resource "aws_eks_cluster" "eks_cluster" {
  name            = "${var.cluster_name}"
  role_arn        = "${aws_iam_role.cluster_role.arn}"
  version	  = "${var.k8s_version}"

  vpc_config {
    security_group_ids = ["${aws_security_group.cluster_sg.id}"]
    subnet_ids         = ["${aws_subnet.subnet.*.id}"]
  }

  depends_on = [
    "aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy",
    "aws_iam_role_policy_attachment.cluster_AmazonEKSServicePolicy",
  ]
}

locals {
  kubeconfig = <<KUBECONFIG


apiVersion: v1
clusters:
- cluster:
    server: ${aws_eks_cluster.eks_cluster.endpoint}
    certificate-authority-data: ${aws_eks_cluster.eks_cluster.certificate_authority.0.data}
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: aws
  name: aws
current-context: aws
kind: Config
preferences: {}
users:
- name: aws
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1alpha1
      command: aws-iam-authenticator
      args:
        - "token"
        - "-i"
        - "${var.cluster_name}"
KUBECONFIG
}

output "kubeconfig" {
  value = "${local.kubeconfig}"
}

locals {
  summary = <<SUMMARY

  EKS Cluster Summary: 
  vpc:    ${aws_vpc.vpc.id}
  subnets: ${join(",", aws_subnet.subnet.*.id)}
  cluster security group: ${aws_security_group.cluster_sg.id}
  endpoint: ${aws_eks_cluster.eks_cluster.endpoint}
  EFS file system id: ${aws_efs_file_system.fs.id}

SUMMARY
}

output "summary" {
  value = "${local.summary}"
}

