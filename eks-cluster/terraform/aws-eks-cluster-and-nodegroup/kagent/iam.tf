#---------------------------------------------------------------
# IAM Role for Service Account (IRSA) - Bedrock Access
#---------------------------------------------------------------

# IAM role for kagent controller to access Amazon Bedrock
resource "aws_iam_role" "kagent_bedrock" {
  count = var.enable_bedrock_access ? 1 : 0

  name        = "${var.cluster_name}-kagent-bedrock-role"
  description = "IAM role for kagent controller to access Amazon Bedrock"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = var.eks_oidc_provider_arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "${var.eks_oidc_issuer}:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            # Allow ServiceAccounts matching *-agent pattern in kagent namespace
            # This enables BYO agents to access Bedrock while following naming convention
            "${var.eks_oidc_issuer}:sub" = "system:serviceaccount:${var.kagent_namespace}:*-agent"
          }
        }
      }
    ]
  })

  tags = {
    Name    = "${var.cluster_name}-kagent-bedrock-role"
    Cluster = var.cluster_name
  }
}

# IAM policy for Bedrock access
resource "aws_iam_policy" "kagent_bedrock" {
  count = var.enable_bedrock_access ? 1 : 0

  name        = "${var.cluster_name}-kagent-bedrock-policy"
  description = "IAM policy for kagent to invoke Amazon Bedrock models"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BedrockFoundationModels"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:GetFoundationModel",
          "bedrock:ListFoundationModels"
        ]
        Resource = var.bedrock_model_arns
      },
      {
        Sid    = "BedrockInferenceProfiles"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        # Allow cross-region inference profiles (required for Claude 4.x models)
        Resource = "arn:aws:bedrock:*:*:inference-profile/*"
      },
      # Allow invoking EKS MCP Server tools for cluster operations
      {
        Sid    = "EKSMCPServer"
        Effect = "Allow"
        Action = [
          "eks-mcp:InvokeMcp",
          "eks-mcp:CallReadOnlyTool",
          "eks-mcp:CallPrivilegedTool"
        ]
        Resource = "*"
      },
      # Allow EKS MCP Server to access EKS clusters and Kubernetes API
      {
        Sid    = "EKSClusterAccess"
        Effect = "Allow"
        Action = [
          "eks:AccessKubernetesApi",
          "eks:Describe*",
          "eks:List*"
        ]
        Resource = "*"
      },
      # Allow CloudWatch access for metrics and logs tools
      {
        Sid    = "CloudWatchAccess"
        Effect = "Allow"
        Action = [
          "cloudwatch:GetMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics",
          "logs:GetLogEvents",
          "logs:FilterLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams",
          # CloudWatch Logs Insights (for get_cloudwatch_logs tool)
          "logs:StartQuery",
          "logs:StopQuery",
          "logs:GetQueryResults"
        ]
        Resource = "*"
      },
      # Allow EC2 describe for VPC config tools
      {
        Sid    = "EC2DescribeAccess"
        Effect = "Allow"
        Action = [
          "ec2:DescribeVpcs",
          "ec2:DescribeSubnets",
          "ec2:DescribeSecurityGroups",
          "ec2:DescribeRouteTables",
          "ec2:DescribeNatGateways",
          "ec2:DescribeInternetGateways"
        ]
        Resource = "*"
      },
      # Allow IAM read access for policy inspection tools
      {
        Sid    = "IAMReadAccess"
        Effect = "Allow"
        Action = [
          "iam:GetRole",
          "iam:GetRolePolicy",
          "iam:ListRolePolicies",
          "iam:ListAttachedRolePolicies",
          "iam:GetPolicy",
          "iam:GetPolicyVersion"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Name    = "${var.cluster_name}-kagent-bedrock-policy"
    Cluster = var.cluster_name
  }
}

# Attach Bedrock policy to IAM role
resource "aws_iam_role_policy_attachment" "kagent_bedrock" {
  count = var.enable_bedrock_access ? 1 : 0

  role       = aws_iam_role.kagent_bedrock[0].name
  policy_arn = aws_iam_policy.kagent_bedrock[0].arn
}

#---------------------------------------------------------------
# EKS Access Entry - Kubernetes RBAC for IRSA role
#---------------------------------------------------------------

# Grant the IRSA role access to Kubernetes API via EKS Access Entries
# This maps the IAM role to Kubernetes RBAC permissions
resource "aws_eks_access_entry" "kagent_bedrock" {
  count = var.enable_bedrock_access ? 1 : 0

  cluster_name  = var.cluster_name
  principal_arn = aws_iam_role.kagent_bedrock[0].arn
  type          = "STANDARD"

  tags = {
    Name    = "${var.cluster_name}-kagent-bedrock-access"
    Cluster = var.cluster_name
  }
}

# Associate cluster admin policy for EKS MCP Server operations
resource "aws_eks_access_policy_association" "kagent_bedrock" {
  count = var.enable_bedrock_access ? 1 : 0

  cluster_name  = var.cluster_name
  principal_arn = aws_iam_role.kagent_bedrock[0].arn
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"

  access_scope {
    type = "cluster"
  }

  depends_on = [aws_eks_access_entry.kagent_bedrock]
}
