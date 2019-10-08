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
  default = "1.13"
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


# Nodegroup variables

variable "nodegroup_name" {
  description = "Node group name in cluster"
  type    = "string"
  default = "ng1"
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

variable "eks_gpu_ami" {
    description = "See https://docs.aws.amazon.com/eks/latest/userguide/gpu-ami.html. Must match k8s version."
    type = "map"
    default = {
        "us-east-1"  = "ami-0b2f191b2ddb13526"
        "us-east-2"  = "ami-080eaa48408e4cce8"
        "us-west-2"  = "ami-0842679a8337cad05"
        "ap-east-1" = "ami-0a63a7853310f3111"
        "ap-south-1" = "ami-0e80f1d31941f6010"
        "ap-northeast-1" = "ami-0a443fa8decd2df67"
        "ap-northeast-2" = "ami-019966f3e72baba33"
	"ap-southeast-1" = "ami-07f34d09a8d7fe5a4"
	"ap-southeast-2" = "ami-03563e56d0af30647"
        "eu-central-1" = "ami-0e262bd06b1cbc1d6"
        "eu-west-1" = "ami-05cded29b25d50f91"
        "eu-west-2" = "ami-0163f3b1681613866"
        "eu-west-3" = "ami-0bc7f58961988d81e"
        "eu-north-1" = "ami-0b219ca1c6aa2f7fb"
        "me-south-1" = "ami-0dd405f6294e0414d"
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

# Cluster resources

resource "aws_vpc" "vpc" {
  cidr_block = "${var.cidr_vpc}"
  enable_dns_support = true
  enable_dns_hostnames  = true

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
    subnet_ids         = flatten(["${aws_subnet.subnet.*.id}"])
  }

  depends_on = [
    "aws_iam_role_policy_attachment.cluster_AmazonEKSClusterPolicy",
    "aws_iam_role_policy_attachment.cluster_AmazonEKSServicePolicy",
  ]

  provisioner "local-exec" {
    command = "aws --region ${var.region} eks update-kubeconfig --name ${aws_eks_cluster.eks_cluster.name}"
  }

  provisioner "local-exec" {
    when    = "destroy"
    command = "kubectl config unset users.${aws_eks_cluster.eks_cluster.arn} ; kubectl config unset clusters.${aws_eks_cluster.eks_cluster.arn} ; kubectl config unset contexts.${aws_eks_cluster.eks_cluster.arn} ; kubectl config unset current-context"
  }

}

# Nodegroup resources

resource "aws_iam_role" "node_role" {
  name = "${aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}-role"

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
  name = "${aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}-profile"
  role = "${aws_iam_role.node_role.name}"
}

resource "aws_security_group" "node_sg" {
  name = "${aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}-sg"
  description = "Security group for all nodes in the cluster"
  vpc_id      = "${aws_vpc.vpc.id}"

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}-sg"
  }
}

resource "aws_security_group_rule" "cluster_ingress_workers" {
  description              = "Allow worker Kubelets and pods to communicate to control plane"
  from_port                = 0 
  protocol                 = "-1"
  security_group_id        = "${aws_security_group.cluster_sg.id}"
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
  source_security_group_id = "${aws_security_group.cluster_sg.id}"
  to_port                  = 65535
  type                     = "ingress"
}

locals {
  node-userdata = <<USERDATA
#!/bin/bash
set -o xtrace
/etc/eks/bootstrap.sh --apiserver-endpoint '${aws_eks_cluster.eks_cluster.endpoint}' --b64-cluster-ca '${aws_eks_cluster.eks_cluster.certificate_authority.0.data}' '${aws_eks_cluster.eks_cluster.id}'
USERDATA
}

resource "aws_launch_configuration" "eks_gpu" {
  name                        = "${aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}"
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


resource "aws_autoscaling_group" "node_group" {
  name                  = "${aws_eks_cluster.eks_cluster.id}-${var.nodegroup_name}"
  vpc_zone_identifier   = flatten(["${aws_subnet.subnet.*.id}"]) 

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
  count = "${length(var.azs)}" 
  file_system_id = "${aws_efs_file_system.fs.id}"

  subnet_id      = "${aws_subnet.subnet.*.id[count.index]}" 
  security_groups = ["${aws_security_group.node_sg.id}"] 
}

resource "local_file" "aws_auth" {
  depends_on = [
    "aws_autoscaling_group.node_group"
 ]

  content  = "${local.config_map_aws_auth}"
  filename = "/tmp/aws-auth.yaml"

  provisioner "local-exec" {
    command = "kubectl apply -f /tmp/aws-auth.yaml ; kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta2/nvidia-device-plugin.yml"
  }

}

locals {
  summary = <<SUMMARY

  EKS Cluster Summary: 
  	vpc:    ${aws_vpc.vpc.id}
  	subnets: ${join(",", aws_subnet.subnet.*.id)}
  	cluster security group: ${aws_security_group.cluster_sg.id}
  	endpoint: ${aws_eks_cluster.eks_cluster.endpoint}

  EKS Cluster NodeGroup Summary: 
  	node security group: ${aws_security_group.node_sg.id} 
  	node instance role arn: ${aws_iam_role.node_role.arn}
  EFS Summary:
  	file system id: ${aws_efs_file_system.fs.id}
  	dns: ${aws_efs_file_system.fs.id}.efs.${var.region}.amazonaws.com
SUMMARY
}

output "summary" {
  value = "${local.summary}"
}
