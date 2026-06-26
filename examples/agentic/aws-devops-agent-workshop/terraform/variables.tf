variable "cluster_name" {
  description = "Name of the EKS cluster to wire AWS DevOps Agent into."
  type        = string
}

variable "agent_space_role_arn" {
  description = <<-EOT
    IAM role ARN from your AWS DevOps Agent Space.
    Find it in: AWS console → DevOps Agent → Agent Space → Settings → IAM role.
    The role typically looks like:
      arn:aws:iam::<account>:role/AWSServiceRoleForBedrockAgentCore_<id>
    or a custom role you've configured for the Agent Space.
  EOT
  type        = string

  validation {
    condition     = can(regex("^arn:aws:iam::[0-9]{12}:role/.+$", var.agent_space_role_arn))
    error_message = "agent_space_role_arn must be a valid IAM role ARN (arn:aws:iam::<account>:role/<name>)."
  }
}

variable "tags" {
  description = "Optional tags applied to created EKS access entries."
  type        = map(string)
  default     = {}
}
