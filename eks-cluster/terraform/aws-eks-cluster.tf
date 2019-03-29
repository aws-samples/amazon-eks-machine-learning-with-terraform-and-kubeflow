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
  default = "my-eks-cluster"
  type    = "string"
}

variable "region" {
 description = "name of aws region to use"
 default = "us-east-1"
 type    = "string"
}

variable "azs" {
 description = "list of aws availabilty zones in aws region"
 default = [ "us-east-1c", "us-east-1d", "us-east-1f" ]
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

variable "node_volume_size" {
  description = "EKS cluster worker node EBS volume size in GBs"
  default="200"
  type="string"
}

variable "node_instance_type" {
  description = "EC2 GPU enabled instance type for EKS cluster worker nodes"
  default = "p3.16xlarge"
  type = "string"
}

variable "key_pair" {
  description = "Name of EC2 key pair used to launch EKS cluster worker node EC2 instances"
  default = "saga"
  type = "string"
}

variable "node_group_max" {
    description = "EKS worker node auto-scaling group maximum"
    default = "2"
    type = "string"
}

variable "node_group_min" {
    description = "EKS worker node auto-scaling group minimum"
    default = "0" 
    type = "string"
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

variable "efs_pv_name" {
  description = "K8s persistent volume name for EFS"
  default = "efs-gp-bursting"
  type = "string"
}

variable "eks_gpu_ami" {
    description = "GPU enabled EKS AMI: https://docs.aws.amazon.com/eks/latest/userguide/gpu-ami.html"
    type = "map"
    default = {
        "us-east-1"  = "ami-06ec2ea207616c078"
        "us-east-2"  = "ami-0e6993a35aae3407b"
        "us-west-2"  = "ami-08377056d89909b2a"
    }
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

resource "aws_eks_cluster" "eks_cluster" {
  name            = "${var.cluster_name}"
  role_arn        = "${aws_iam_role.cluster_role.arn}"

  vpc_config {
    security_group_ids = ["${aws_security_group.cluster_sg.id}"]
    subnet_ids         = ["${aws_subnet.subnet.*.id}"]
  }

  depends_on = [
    "aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy",
    "aws_iam_role_policy_attachment.cluster_AmazonEKSServicePolicy",
  ]
}

resource "aws_iam_role" "node_role" {
  name = "${var.cluster_name}-node-role"

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
  role       = "${aws_iam_role.node_role.name}"
}

resource "aws_iam_role_policy_attachment" "node_AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = "${aws_iam_role.node_role.name}"
}

resource "aws_iam_role_policy_attachment" "node_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = "${aws_iam_role.node_role.name}"
}

resource "aws_iam_instance_profile" "node_profile" {
  name = "${var.cluster_name}-node-profile"
  role = "${aws_iam_role.node_role.name}"
}

resource "aws_security_group" "node_sg" {
  name = "${var.cluster_name}-nodes-sg"
  description = "Security group for all nodes in the cluster"
  vpc_id      = "${aws_vpc.vpc.id}"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-nodes-sg"
  }
}

resource "aws_security_group_rule" "node_ingress_self" {
  description              = "Allow nodes to communicate with each other"
  from_port                = 0
  protocol                 = "-1"
  security_group_id        = "${aws_security_group.node_sg.id}"
  source_security_group_id = "${aws_security_group.node_sg.id}"
  to_port                  = 65535
  type                     = "ingress"
}

resource "aws_security_group_rule" "node_ingress_control" {
  description              = "Allow worker Kubelets and pods to receive communication from the cluster control plane"
  from_port                = 1025
  protocol                 = "tcp"
  security_group_id        = "${aws_security_group.node_sg.id}"
  source_security_group_id = "${aws_security_group.cluster_sg.id}"
  to_port                  = 65535
  type                     = "ingress"
}

locals {
  node-userdata = <<USERDATA
#!/bin/bash
set -o xtrace
/etc/eks/bootstrap.sh '${var.cluster_name}'
USERDATA
}

resource "aws_launch_configuration" "eks_gpu" {
  name                        = "${var.cluster_name}-node-config"
  associate_public_ip_address = true 
  iam_instance_profile        = "${aws_iam_instance_profile.node_profile.name}"
  image_id                    = "${lookup(var.eks_gpu_ami, var.region, "us-east-1")}"
  instance_type               = "${var.node_instance_type}"
  security_groups             = ["${aws_security_group.node_sg.id}"]
  user_data                   = "${base64encode(local.node-userdata)}"
  
  key_name = "${var.key_pair}"
  enable_monitoring = true
  ebs_optimized = true
  
  root_block_device {
  	delete_on_termination = true
  	volume_size = "${var.node_volume_size}"
  }

  depends_on = [
    "aws_eks_cluster.eks_cluster"
  ]

  lifecycle {
    create_before_destroy = true
  }

}

resource "aws_autoscaling_group" "node_group" {
  count = "${length(var.azs)}" 

  name = "${var.cluster_name}-node-asg-${count.index}"
  vpc_zone_identifier   = ["${aws_subnet.subnet.*.id[count.index]}"] 

  health_check_grace_period = "0"
  desired_capacity   = "0"
  max_size           = "${var.node_group_max}" 
  min_size           = "${var.node_group_min}" 

  launch_configuration = "${aws_launch_configuration.eks_gpu.id}"

  depends_on = [
    "aws_eks_cluster.eks_cluster"
  ]
}

resource "aws_efs_file_system" "fs" {

 performance_mode = "${var.efs_performance_mode}"
 
 throughput_mode = "${var.efs_throughput_mode}"


  tags = {
    Name = "${var.cluster_name}-fs"
  }
}

resource "aws_efs_mount_target" "target" {
  count = "${length(var.azs)}" 

  file_system_id = "${aws_efs_file_system.fs.id}"

  subnet_id      = "${aws_subnet.subnet.*.id[count.index]}"
  security_groups = ["${aws_security_group.node_sg.id}"] 
}


locals {
  config_map_aws_auth = <<CONFIGMAPAWSAUTH


apiVersion: v1
kind: ConfigMap
metadata:
  name: aws-auth
  namespace: kube-system
data:
  mapRoles: |
    - rolearn: ${aws_iam_role.node_role.arn}
      username: system:node:{{EC2PrivateDNSName}}
      groups:
        - system:bootstrappers
        - system:nodes
CONFIGMAPAWSAUTH
}

output "config_map_aws_auth" {
  value = "${local.config_map_aws_auth}"
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
  efspv = <<EFSPV

apiVersion: v1
kind: PersistentVolume
metadata:
  name: ${var.efs_pv_name}
spec:
  capacity:
    storage: 1Pi
  accessModes:
    - ReadWriteMany
  mountOptions:
    - nfsvers=4.1
    - rsize=1048576
    - wsize=1048576
    - hard
    - timeo=600
    - retrans=2
    - timeo=600
    - noresvport
  nfs:
    server: ${aws_efs_file_system.fs.id}.${var.region}.amazonaws.com
    path: "/"

EFSPV
}

output "efspv" {
  value = "${local.efspv}"
}

locals {
  summary = <<SUMMARY

  network summary: 
  vpc:    ${aws_vpc.vpc.id}
  subnet: ${aws_subnet.subnet.*.id[0]}
  subnet: ${aws_subnet.subnet.*.id[1]}
  subnet: ${aws_subnet.subnet.*.id[2]}
  control plane security group: ${aws_security_group.cluster_sg.id}
  node security group: ${aws_security_group.node_sg.id} 

SUMMARY
}

output "summary" {
  value = "${local.summary}"
}

