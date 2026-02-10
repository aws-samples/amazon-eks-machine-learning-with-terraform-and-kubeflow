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
      # Allow EKS MCP Server to access Kubernetes API on behalf of the agent
      {
        Sid    = "EKSClusterAccess"
        Effect = "Allow"
        Action = [
          "eks:AccessKubernetesApi",
          "eks:DescribeCluster",
          "eks:ListClusters"
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
