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
            "${replace(var.eks_oidc_provider_arn, "/^(.*provider/)/", "")}:sub" = "system:serviceaccount:${var.kagent_namespace}:kagent-controller"
            "${replace(var.eks_oidc_provider_arn, "/^(.*provider/)/", "")}:aud" = "sts.amazonaws.com"
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
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:GetFoundationModel",
          "bedrock:ListFoundationModels"
        ]
        Resource = var.bedrock_model_arns
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
