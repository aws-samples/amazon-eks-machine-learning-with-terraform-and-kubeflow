# Codifies the AWS DevOps Agent ↔ EKS wiring (workshop Module 1).
#
# Replaces the manual console workflow:
#   1. EKS console → Access tab → Create access entry with the Agent Space role ARN
#   2. Attach AmazonAIOpsAssistantPolicy with cluster scope
#
# Usage:
#   terraform init
#   terraform apply \
#     -var="cluster_name=<your-eks-cluster>" \
#     -var="agent_space_role_arn=<arn from Agent Space → Settings → IAM role>"
#
# After apply, the agent's IAM role can read the cluster (pods, events, logs)
# via the EKS API. No write/mutate access by design — match what the workshop
# does. To grant scoped write later, use a separate access policy
# (e.g. AmazonEKSEditPolicy) on a different access entry.

resource "aws_eks_access_entry" "agent" {
  cluster_name  = var.cluster_name
  principal_arn = var.agent_space_role_arn
  type          = "STANDARD"

  tags = merge(var.tags, {
    Purpose = "aws-devops-agent-workshop"
  })
}

resource "aws_eks_access_policy_association" "agent_aiops" {
  cluster_name  = var.cluster_name
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonAIOpsAssistantPolicy"
  principal_arn = var.agent_space_role_arn

  access_scope {
    type = "cluster"
  }

  depends_on = [aws_eks_access_entry.agent]
}
