#################################################
# HyperPod Helm Dependencies
#################################################

# Create aws-hyperpod namespace for HyperPod components
resource "kubernetes_namespace" "aws_hyperpod" {
  count = var.install_helm_dependencies ? 1 : 0

  metadata {
    name = "aws-hyperpod"
    labels = {
      "app.kubernetes.io/managed-by" = "Helm"
    }
    annotations = {
      "meta.helm.sh/release-name"      = "hyperpod"
      "meta.helm.sh/release-namespace" = "kube-system"
    }
  }
}

# Clone the HyperPod helm chart from sagemaker-hyperpod-cli repository
# Uses shallow clone to minimize download size
resource "null_resource" "clone_hyperpod_cli" {
  count = var.install_helm_dependencies ? 1 : 0

  provisioner "local-exec" {
    command = <<-EOT
      set -e
      CHART_DIR="${path.module}/sagemaker-hyperpod-cli"
      if [ ! -d "$CHART_DIR/helm_chart/HyperPodHelmChart" ]; then
        rm -rf "$CHART_DIR"
        git clone --depth 1 https://github.com/aws/sagemaker-hyperpod-cli.git "$CHART_DIR"
      fi
      cd "$CHART_DIR/helm_chart/HyperPodHelmChart" && helm dependency update
    EOT
  }

  triggers = {
    chart_dir = "${path.module}/sagemaker-hyperpod-cli"
  }
}

resource "helm_release" "hyperpod" {
  count = var.install_helm_dependencies ? 1 : 0

  name             = "hyperpod"
  chart            = "${path.module}/sagemaker-hyperpod-cli/helm_chart/HyperPodHelmChart"
  namespace        = "kube-system"
  create_namespace = false
  wait             = false
  timeout          = 600
  dependency_update = true

  #################################################
  # HyperPod Core Components
  #################################################

  # Training Operator with HyperPod job auto-restart support
  set {
    name  = "trainingOperators.enabled"
    value = tostring(var.enable_training_operator)
  }

  # Skip namespace creation - kubeflow namespace is managed by Terraform
  set {
    name  = "trainingOperators.kubeflow-training-operator.namespace.create"
    value = "false"
  }

  # Skip aws-hyperpod namespace creation if it already exists
  set {
    name  = "namespace.create"
    value = "false"
  }

  # Health Monitoring Agent
  set {
    name  = "health-monitoring-agent.enabled"
    value = tostring(var.enable_health_monitoring)
  }

  set {
    name  = "health-monitoring-agent.region"
    value = var.region
  }

  # Deep Health Check
  set {
    name  = "deep-health-check.enabled"
    value = tostring(var.enable_deep_health_check)
  }

  # Job Auto Restart (for PyTorchJob resilience)
  set {
    name  = "job-auto-restart.enabled"
    value = "true"
  }

  #################################################
  # Task Governance (RBAC)
  #################################################

  set {
    name  = "cluster-role-and-bindings.enabled"
    value = tostring(var.enable_task_governance)
  }

  set {
    name  = "namespaced-role-and-bindings.enable"
    value = tostring(var.enable_task_governance)
  }

  set {
    name  = "team-role-and-bindings.enabled"
    value = tostring(var.enable_task_governance)
  }

  #################################################
  # Disable components already in base project
  #################################################

  set {
    name  = "nvidia-device-plugin.devicePlugin.enabled"
    value = "false"
  }

  set {
    name  = "aws-efa-k8s-device-plugin.devicePlugin.enabled"
    value = "false"
  }

  set {
    name  = "neuron-device-plugin.devicePlugin.enabled"
    value = "false"
  }

  set {
    name  = "cert-manager.enabled"
    value = "false"
  }

  set {
    name  = "mpi-operator.enabled"
    value = "false"
  }

  depends_on = [
    null_resource.clone_hyperpod_cli,
    kubernetes_namespace.aws_hyperpod
  ]
}