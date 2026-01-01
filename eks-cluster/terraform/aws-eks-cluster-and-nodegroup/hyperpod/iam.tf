#################################################
# IAM Role for SageMaker HyperPod Cluster
#################################################

resource "aws_iam_role" "hyperpod_execution" {
  name = "${var.cluster_name}-hyperpod-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = var.tags
}

# Attach managed policy for SageMaker HyperPod
resource "aws_iam_role_policy_attachment" "hyperpod_managed" {
  role       = aws_iam_role.hyperpod_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerClusterInstanceRolePolicy"
}

# Custom policy for VPC, EKS, and S3 access
resource "aws_iam_policy" "hyperpod_custom" {
  name        = "${var.cluster_name}-hyperpod-custom-policy"
  description = "Custom policy for HyperPod VPC and EKS access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "VPCPermissions"
        Effect = "Allow"
        Action = [
          "ec2:AssignPrivateIpAddresses",
          "ec2:CreateNetworkInterface",
          "ec2:CreateNetworkInterfacePermission",
          "ec2:DeleteNetworkInterface",
          "ec2:DeleteNetworkInterfacePermission",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DescribeVpcs",
          "ec2:DescribeDhcpOptions",
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups",
          "ec2:UnassignPrivateIpAddresses",
          "ec2:DescribeInstances"
        ]
        Resource = "*"
      },
      {
        Sid    = "EKSAccess"
        Effect = "Allow"
        Action = [
          "eks:DescribeCluster",
          "eks:ListClusters",
          "eks-auth:AssumeRoleForPodIdentity"
        ]
        Resource = "*"
      },
      {
        Sid    = "S3LifecycleScripts"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::sagemaker-*",
          "arn:aws:s3:::sagemaker-*/*",
          local.lifecycle_bucket_arn,
          "${local.lifecycle_bucket_arn}/*"
        ]
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "hyperpod_custom" {
  role       = aws_iam_role.hyperpod_execution.name
  policy_arn = aws_iam_policy.hyperpod_custom.arn
}

# EKS CNI Policy - required for VPC networking on HyperPod nodes
resource "aws_iam_role_policy_attachment" "hyperpod_cni" {
  role       = aws_iam_role.hyperpod_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

# ECR Access - required for pulling container images
resource "aws_iam_role_policy_attachment" "hyperpod_ecr" {
  role       = aws_iam_role.hyperpod_execution.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}