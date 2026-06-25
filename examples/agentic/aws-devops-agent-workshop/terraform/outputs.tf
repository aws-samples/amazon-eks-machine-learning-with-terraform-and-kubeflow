output "access_entry_arn" {
  description = "ARN of the EKS access entry created for the Agent Space role."
  value       = aws_eks_access_entry.agent.access_entry_arn
}

output "policy_association_id" {
  description = "Composite ID of the AmazonAIOpsAssistantPolicy association."
  value       = aws_eks_access_policy_association.agent_aiops.id
}

output "verification_command" {
  description = "Run this after apply to confirm the agent can talk to the cluster."
  value       = <<-EOT
    aws eks list-access-entries --cluster-name ${var.cluster_name} \
      --query 'accessEntries[?contains(@, `${var.agent_space_role_arn}`)]'

    aws eks list-associated-access-policies \
      --cluster-name ${var.cluster_name} \
      --principal-arn ${var.agent_space_role_arn}
  EOT
}
