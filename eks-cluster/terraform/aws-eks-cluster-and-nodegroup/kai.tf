#---------------------------------------------------------------
# KAI Scheduler with GPU fractions (NvFractions)
# https://github.com/kai-scheduler/KAI-Scheduler
#
# Off by default. When enabled, adds a separate kai-gpu Karpenter pool (Ubuntu 24.04) where the
# GPU Operator installs driver r615+, which NvFractions needs and the AL2023 cuda pool cannot
# have. The pool is tainted kai.scheduler/gpu, and everything here tolerates only that taint, so
# the default GPU stack (cuda pool, standalone device plugin) is unchanged and the two never meet.
#
# KAI workloads: schedulerName kai-scheduler, label kai.scheduler/queue, nodeSelector
# kai.scheduler/node: "true", and a toleration for kai.scheduler/gpu.
#---------------------------------------------------------------

data "aws_ssm_parameter" "kai_ubuntu_ami" {
  count = var.kai_enabled ? 1 : 0
  name  = "/aws/service/canonical/ubuntu/eks/24.04/${local.use_k8s_version}/stable/${var.kai_ubuntu_ami_release}/amd64/hvm/ebs-gp3/ami-id"
}

locals {
  # The kai block of the karpenter-components values; with kai_enabled=false nothing renders.
  kai_karpenter_values = {
    enabled          = var.kai_enabled
    ami_id           = var.kai_enabled ? nonsensitive(data.aws_ssm_parameter.kai_ubuntu_ami[0].value) : ""
    cluster_endpoint = var.kai_enabled ? aws_eks_cluster.eks_cluster.endpoint : ""
    cluster_ca       = var.kai_enabled ? aws_eks_cluster.eks_cluster.certificate_authority[0].data : ""
    cluster_dns_ip   = var.kai_enabled ? cidrhost(aws_eks_cluster.eks_cluster.kubernetes_network_config[0].service_ipv4_cidr, 10) : ""
    instance_types   = var.kai_instance_types
  }
}

resource "helm_release" "gpu_operator" {
  count = var.kai_enabled ? 1 : 0

  name             = "gpu-operator"
  repository       = "https://helm.ngc.nvidia.com/nvidia"
  chart            = "gpu-operator"
  version          = var.kai_gpu_operator_version
  namespace        = "gpu-operator"
  create_namespace = true

  values = [
    <<-EOT
      daemonsets:
        # Replaces the chart default (nvidia.com/gpu), which would admit the operands to the
        # cuda/cudaefa nodes; the standalone device plugin owns those.
        tolerations:
          - key: kai.scheduler/gpu
            operator: Exists
            effect: NoSchedule
      driver:
        # gpu-fractioning needs r615+. Not precompiled: NVIDIA publishes precompiled images only
        # for 580/595 on 6.8 kernels; the Ubuntu EKS AMI runs 7.0.
        version: "${var.kai_gpu_driver_version}"
        kernelModuleType: auto
      toolkit:
        version: v1.20.1          # gpu-fractioning needs >= v1.20.1
      devicePlugin:
        version: v0.20.1          # gpu-fractioning needs >= v0.20.1
      node-feature-discovery:
        worker:
          # only kai-gpu nodes need the PCI labels that tell the operator a GPU is present
          nodeSelector:
            kai.scheduler/node: "true"
          tolerations:
            - key: kai.scheduler/gpu
              operator: Exists
              effect: NoSchedule
    EOT
  ]

  lifecycle {
    precondition {
      condition     = var.karpenter_enabled
      error_message = "kai_enabled=true needs karpenter_enabled=true: the kai-gpu nodes come from a Karpenter pool."
    }
  }

  depends_on = [helm_release.karpenter_components]
}

resource "helm_release" "kai_scheduler" {
  count = var.kai_enabled ? 1 : 0

  name             = "kai-scheduler"
  repository       = "oci://ghcr.io/kai-scheduler/kai-scheduler"
  chart            = "kai-scheduler"
  version          = var.kai_scheduler_version
  namespace        = "kai-scheduler"
  create_namespace = true

  values = [
    <<-EOT
      global:
        nvFractions:
          set: true
      scheduler:
        placementStrategy: binpack
    EOT
  ]

  depends_on = [helm_release.gpu_operator]
}

# gpu-fractioning's operator builds its two DaemonSets with no tolerations, and its
# GpuFractioningConfig has no field for one, so on the tainted pool they would never run. Two
# parts: the patch below adds the toleration, and this policy rejects any later update that takes
# it away -- the operator rewrites the DaemonSets on every reconcile. The operator itself must keep
# running: it is what sets the node condition gpu-fractioning.nvidia.com/Ready that KAI schedules
# on. Its DaemonSet updates now fail (quietly, retried every 30s), which also means an upgrade of
# gpu-fractioning does not reach the nodes: delete the binding, upgrade, re-run the patch, restore.
resource "kubectl_manifest" "kai_fractioning_toleration_policy" {
  count = var.kai_enabled ? 1 : 0

  yaml_body = <<-EOT
    apiVersion: admissionregistration.k8s.io/v1
    kind: ValidatingAdmissionPolicy
    metadata:
      name: kai-fractioning-keep-tolerations
    spec:
      failurePolicy: Fail
      matchConstraints:
        resourceRules:
          - apiGroups: ["apps"]
            apiVersions: ["v1"]
            operations: ["UPDATE"]
            resources: ["daemonsets"]
      matchConditions:
        - name: fractioning-daemonsets
          expression: "object.metadata.namespace == 'kai-scheduler' && object.metadata.name in ['gpu-fractioning-fractiond', 'gpu-fractioning-mpsd']"
      validations:
        # Creation is allowed, so the operator can (re)create the DaemonSets; only an update that
        # removes a toleration the DaemonSet has is refused.
        - expression: >-
            !(has(oldObject.spec.template.spec.tolerations) &&
              oldObject.spec.template.spec.tolerations.exists(t, t.key == 'kai.scheduler/gpu')) ||
            (has(object.spec.template.spec.tolerations) &&
              object.spec.template.spec.tolerations.exists(t, t.key == 'kai.scheduler/gpu'))
          message: "gpu-fractioning DaemonSets must keep tolerating kai.scheduler/gpu (the kai-gpu pool is tainted); see kai.tf"
  EOT

  depends_on = [helm_release.kai_scheduler]
}

resource "kubectl_manifest" "kai_fractioning_toleration_policy_binding" {
  count = var.kai_enabled ? 1 : 0

  yaml_body = <<-EOT
    apiVersion: admissionregistration.k8s.io/v1
    kind: ValidatingAdmissionPolicyBinding
    metadata:
      name: kai-fractioning-keep-tolerations
    spec:
      policyName: kai-fractioning-keep-tolerations
      validationActions: ["Deny"]
  EOT

  depends_on = [kubectl_manifest.kai_fractioning_toleration_policy]
}

resource "null_resource" "kai_fractioning_tolerations" {
  count = var.kai_enabled ? 1 : 0

  triggers = {
    kai_scheduler_version = var.kai_scheduler_version
  }

  # The operator creates the DaemonSets shortly after the release installs; wait for them, then add
  # the toleration. A merge patch sets the whole list, so re-running is harmless.
  provisioner "local-exec" {
    command = <<-EOT
      set -e
      for ds in gpu-fractioning-fractiond gpu-fractioning-mpsd; do
        for i in $(seq 1 60); do
          kubectl get ds -n kai-scheduler "$ds" >/dev/null 2>&1 && break
          sleep 5
        done
        kubectl patch ds -n kai-scheduler "$ds" --type merge \
          -p '{"spec":{"template":{"spec":{"tolerations":[{"key":"kai.scheduler/gpu","operator":"Exists","effect":"NoSchedule"}]}}}}'
      done
    EOT
  }

  depends_on = [kubectl_manifest.kai_fractioning_toleration_policy_binding]
}
