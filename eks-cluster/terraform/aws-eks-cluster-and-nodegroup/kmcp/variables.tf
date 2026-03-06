#---------------------------------------------------------------
# Required Variables
#---------------------------------------------------------------

variable "kmcp_namespace" {
  description = "Kubernetes namespace for kmcp"
  type        = string
}

variable "kmcp_version" {
  description = "kmcp Helm chart version"
  type        = string
}

#---------------------------------------------------------------
# Controller Configuration
#---------------------------------------------------------------

variable "controller_replicas" {
  description = "Number of kmcp controller replicas"
  type        = number
  default     = 1
  validation {
    condition     = var.controller_replicas >= 1
    error_message = "controller_replicas must be at least 1"
  }
}

variable "controller_cpu_request" {
  description = "CPU request for kmcp controller"
  type        = string
  default     = "100m"
}

variable "controller_memory_request" {
  description = "Memory request for kmcp controller"
  type        = string
  default     = "128Mi"
}

variable "controller_cpu_limit" {
  description = "CPU limit for kmcp controller"
  type        = string
  default     = "1000m"
}

variable "controller_memory_limit" {
  description = "Memory limit for kmcp controller"
  type        = string
  default     = "512Mi"
}

#---------------------------------------------------------------
# Leader Election
#---------------------------------------------------------------

variable "enable_leader_election" {
  description = "Enable leader election for controller HA"
  type        = bool
  default     = true
}

#---------------------------------------------------------------
# Metrics Configuration
#---------------------------------------------------------------

variable "enable_metrics" {
  description = "Enable metrics endpoint on the controller"
  type        = bool
  default     = true
}

variable "metrics_secure_serving" {
  description = "Enable secure (HTTPS) metrics serving"
  type        = bool
  default     = false
}

#---------------------------------------------------------------
# Istio Sidecar Injection
#---------------------------------------------------------------

variable "enable_istio_injection" {
  description = "Enable Istio sidecar injection for kmcp namespace"
  type        = bool
  default     = false
}

#---------------------------------------------------------------
# Prometheus ServiceMonitor
#---------------------------------------------------------------

variable "enable_service_monitor" {
  description = "Create a Prometheus ServiceMonitor for kmcp metrics (requires Prometheus Operator)"
  type        = bool
  default     = false
}

#---------------------------------------------------------------
# Additional Helm Values
#---------------------------------------------------------------

variable "additional_helm_values" {
  description = "Additional Helm values as YAML string to pass to kmcp chart"
  type        = string
  default     = ""
}
