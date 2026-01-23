#---------------------------------------------------------------
# Required Variables
#---------------------------------------------------------------

variable "kagent_namespace" {
  description = "Kubernetes namespace for kagent"
  type        = string
}

variable "kagent_version" {
  description = "kagent Helm chart version"
  type        = string
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "eks_cluster_id" {
  description = "EKS cluster ID"
  type        = string
}

variable "eks_oidc_provider_arn" {
  description = "EKS OIDC provider ARN for IRSA"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

#---------------------------------------------------------------
# Database Configuration
#---------------------------------------------------------------

variable "database_type" {
  description = "Database type: 'sqlite' (default, single replica) or 'postgresql' (HA, multi-replica)"
  type        = string
  default     = "sqlite"
  validation {
    condition     = contains(["sqlite", "postgresql"], var.database_type)
    error_message = "database_type must be either 'sqlite' or 'postgresql'"
  }
}

variable "sqlite_database_name" {
  description = "SQLite database file name (used when database_type is sqlite)"
  type        = string
  default     = "kagent.db"
}

# PostgreSQL configuration (only used when database_type is postgresql)

variable "vpc_id" {
  description = "VPC ID for PostgreSQL security group (required if database_type is postgresql)"
  type        = string
  default     = ""
}

variable "vpc_cidr" {
  description = "VPC CIDR block for PostgreSQL security group (required if database_type is postgresql)"
  type        = string
  default     = ""
}

variable "db_subnet_ids" {
  description = "Subnet IDs for PostgreSQL database (required if database_type is postgresql)"
  type        = list(string)
  default     = []
}

variable "db_engine_version" {
  description = "Aurora PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "kagent"
}

variable "db_username" {
  description = "PostgreSQL master username"
  type        = string
  default     = "kagent"
}

variable "db_min_capacity" {
  description = "PostgreSQL Aurora Serverless v2 minimum capacity (ACU)"
  type        = number
  default     = 0.5
}

variable "db_max_capacity" {
  description = "PostgreSQL Aurora Serverless v2 maximum capacity (ACU)"
  type        = number
  default     = 2
}

#---------------------------------------------------------------
# Controller Configuration
#---------------------------------------------------------------

variable "controller_replicas" {
  description = "Number of kagent controller replicas (only >1 if using PostgreSQL)"
  type        = number
  default     = 1
  validation {
    condition     = var.controller_replicas >= 1
    error_message = "controller_replicas must be at least 1"
  }
}

variable "controller_cpu_request" {
  description = "CPU request for kagent controller"
  type        = string
  default     = "100m"
}

variable "controller_memory_request" {
  description = "Memory request for kagent controller"
  type        = string
  default     = "128Mi"
}

variable "controller_cpu_limit" {
  description = "CPU limit for kagent controller"
  type        = string
  default     = "1000m"
}

variable "controller_memory_limit" {
  description = "Memory limit for kagent controller"
  type        = string
  default     = "512Mi"
}

#---------------------------------------------------------------
# UI Configuration
#---------------------------------------------------------------

variable "enable_ui" {
  description = "Enable kagent UI deployment"
  type        = bool
  default     = true
}

variable "ui_cpu_request" {
  description = "CPU request for kagent UI"
  type        = string
  default     = "100m"
}

variable "ui_memory_request" {
  description = "Memory request for kagent UI"
  type        = string
  default     = "128Mi"
}

variable "ui_cpu_limit" {
  description = "CPU limit for kagent UI"
  type        = string
  default     = "500m"
}

variable "ui_memory_limit" {
  description = "Memory limit for kagent UI"
  type        = string
  default     = "256Mi"
}

#---------------------------------------------------------------
# Istio Ingress Configuration
#---------------------------------------------------------------

variable "enable_istio_ingress" {
  description = "Enable Istio VirtualService for kagent UI"
  type        = bool
  default     = false
}

variable "istio_gateway" {
  description = "Istio Gateway to use for kagent UI (e.g., 'ingress/istio-ingressgateway')"
  type        = string
  default     = "ingress/istio-ingressgateway"
}

variable "ingress_host" {
  description = "Host/domain for kagent UI ingress"
  type        = string
  default     = "kagent.example.com"
}

variable "ingress_path_prefix" {
  description = "Path prefix for kagent UI ingress (e.g., '/kagent')"
  type        = string
  default     = "/kagent"
}

#---------------------------------------------------------------
# Istio Sidecar Injection
#---------------------------------------------------------------

variable "enable_istio_injection" {
  description = "Enable Istio sidecar injection for kagent namespace"
  type        = bool
  default     = false
}

#---------------------------------------------------------------
# AWS IAM / Bedrock Configuration
#---------------------------------------------------------------

variable "enable_bedrock_access" {
  description = "Enable IRSA for Amazon Bedrock access"
  type        = bool
  default     = false
}

variable "bedrock_model_arns" {
  description = "List of Bedrock model ARNs to grant access to (e.g., ['arn:aws:bedrock:*::foundation-model/*'])"
  type        = list(string)
  default     = ["arn:aws:bedrock:*::foundation-model/*"]
}

#---------------------------------------------------------------
# Additional Helm Values
#---------------------------------------------------------------

variable "additional_helm_values" {
  description = "Additional Helm values as YAML string to pass to kagent chart"
  type        = string
  default     = ""
}
