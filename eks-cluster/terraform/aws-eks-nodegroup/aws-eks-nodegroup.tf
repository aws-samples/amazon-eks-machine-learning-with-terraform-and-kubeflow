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

variable "region" {
 description = "name of aws region to use"
 type    = "string"
}

variable "cluster_name" {
  description = "unique name of the eks cluster"
  type    = "string"
}

variable "nodegroup_name" {
  description = "Node group name in cluster"
  type    = "string"
}

variable "cluster_sg" {
  description = "cluster security group"
  type = "string"
}

variable "subnet_id" {
  description = "subnet id for ndoe group"
  type = "string"
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
  type = "string"
}

variable "node_group_desired" {
    description = "EKS worker node auto-scaling group desired size"
    default = "2"
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

variable "efs_id" {
  description = "EFS file-system id"
  type = "string"
}

variable "eks_gpu_ami" {
    description = "See https://docs.aws.amazon.com/eks/latest/userguide/gpu-ami.html. Must match k8s version."
    type = "map"
    default = {
	"us-east-1"  = "ami-0730212bffaa1732a",
 	"us-east-2"  = "ami-095e1b9737cfe76bc",
	"us-west-2"  = "ami-0ad9a8dc09680cfc2",
	"ap-south-1"  = "ami-08a65b6849b5d0131",
	"ap-northeast-1" = "ami-0e4846cf3cb2440e5",
	"ap-northeast-2" = "ami-0bfd27f77f9bf86b3",
	"ap-southeast-1" = "ami-024a58dc3e6f0d78b",
	"ap-southeast-2" = "ami-0b24ff00f0fb4b71f",
	"eu-central-1" = "ami-08b45c59715f94f5f",
 	"eu-west-1" =  "ami-03b8e736123fd2b9b",
	"eu-west-2" = "ami-03a9c642e26334ac0",
 	"eu-west-3" = "ami-05e2c706ec723b102",
	"eu-north-1"  = "ami-06f065fbe98f1b3a5"
    }
}

variable "associate_public_ip" {
   description = "associate public IP with node instance"
   type = "string"
   default = "true"
}

# END variables

provider "aws" {
  region                  = "${var.region}"
  shared_credentials_file = "${var.credentials}"
  profile                 = "${var.profile}"
}

data "aws_eks_cluster" "eks_cluster" {
   name = "${var.cluster_name}"
}

resource "aws_iam_role" "node_role" {
  name = "${data.aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}-role"

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

resource "aws_iam_role_policy_attachment" "node_AmazonS3ReadOnlyPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
  role       = "${aws_iam_role.node_role.name}"
}

resource "aws_iam_instance_profile" "node_profile" {
  name = "${data.aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}-profile"
  role = "${aws_iam_role.node_role.name}"
}

resource "aws_security_group" "node_sg" {
  name = "${data.aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}-sg"
  description = "Security group for all nodes in the cluster"
  vpc_id      = "${lookup(data.aws_eks_cluster.eks_cluster.vpc_config[0], "vpc_id")}"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${data.aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}-sg"
  }
}

resource "aws_security_group_rule" "cluster_ingress_workers" {
  description              = "Allow worker Kubelets and pods to communicate to control plane"
  from_port                = 0 
  protocol                 = "-1"
  security_group_id        = "${var.cluster_sg}"
  source_security_group_id = "${aws_security_group.node_sg.id}"
  to_port                  = 65535
  type                     = "ingress"
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
  from_port                = 0 
  protocol                 = "-1"
  security_group_id        = "${aws_security_group.node_sg.id}"
  source_security_group_id = "${var.cluster_sg}"
  to_port                  = 65535
  type                     = "ingress"
}

locals {
  node-userdata = <<USERDATA
#!/bin/bash
set -o xtrace
/etc/eks/bootstrap.sh --apiserver-endpoint '${data.aws_eks_cluster.eks_cluster.endpoint}' --b64-cluster-ca '${data.aws_eks_cluster.eks_cluster.certificate_authority.0.data}' '${data.aws_eks_cluster.eks_cluster.id}'
USERDATA
}

resource "aws_launch_configuration" "eks_gpu" {
  name                        = "${data.aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}"
  associate_public_ip_address = "${var.associate_public_ip}" 
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

  lifecycle {
    create_before_destroy = true
  }

}

resource "aws_autoscaling_group" "node_group" {
  name                  = "${data.aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}"
  vpc_zone_identifier   = ["${var.subnet_id}"]

  health_check_grace_period = "0"
  desired_capacity   = "${var.node_group_desired}"
  max_size           = "${var.node_group_max}" 
  min_size           = "${var.node_group_min}" 

  launch_configuration = "${aws_launch_configuration.eks_gpu.id}"
  tag {
    key                 = "Name"
    value               = "${var.cluster_name}-${var.nodegroup_name}-node"
    propagate_at_launch = true
  }

  tag {
    key                 = "kubernetes.io/cluster/${var.cluster_name}"
    value               = "owned"
    propagate_at_launch = true
  }
}

resource "aws_efs_mount_target" "target" {
  file_system_id = "${var.efs_id}"

  subnet_id      = "${var.subnet_id}"
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
  efspv = <<EFSPV

apiVersion: v1
kind: PersistentVolume
metadata:
  name: ${var.cluster_name}-${var.nodegroup_name}
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
    server: ${var.efs_id}.efs.${var.region}.amazonaws.com
    path: "/"

EFSPV
}

output "efspv" {
  value = "${local.efspv}"
}

locals {
  efspvc = <<EFSPVC

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${var.cluster_name}-${var.nodegroup_name}
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ""
  resources:
    requests:
      storage: 100Gi

EFSPVC
}

output "efspvc" {
  value = "${local.efspvc}"
}

locals {
  summary = <<SUMMARY

  EKS Cluster NodeGroup Summary: 
  Node security group: ${aws_security_group.node_sg.id} 
  Node instance role arn: ${aws_iam_role.node_role.arn}
  Efs DNS: ${var.efs_id}.efs.${var.region}.amazonaws.com

SUMMARY
}

output "summary" {
  value = "${local.summary}"
}

