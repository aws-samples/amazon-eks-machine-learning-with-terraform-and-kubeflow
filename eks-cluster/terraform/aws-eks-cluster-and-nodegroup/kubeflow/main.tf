resource "random_string" "minio_access_key" {
  length           = 16
  special          = false
}

resource "random_password" "minio_secret_key" {
  length           = 32
  special          = false
}

resource "kubernetes_namespace" "kubeflow_user_profile" {
  metadata {
    labels = {
      istio-injection = "enabled"
    }
    annotations = {
      "owner" = "user@example.com"
    }

    name = "${var.kubeflow_user_profile}"
  }
}

resource "helm_release" "pv_efs" {
  chart = "${var.local_helm_repo}/pv-efs"
  name = "pv-efs"
  version = "1.0.0"
  namespace = var.kubeflow_namespace
  
  set {
    name  = "namespace"
    value = var.kubeflow_namespace
  }

  set {
    name  = "efs.fs_id"
    value = var.efs_fs_id
  }
}

resource "helm_release" "pv_fsx" {
  chart = "${var.local_helm_repo}/pv-fsx"
  name = "pv-fsx"
  version = "1.1.0"
  namespace = var.kubeflow_namespace
  
  set {
    name  = "namespace"
    value = var.kubeflow_namespace
  }

  set {
    name  = "fsx.fs_id"
    value = var.fsx.fs_id
  }

  set {
    name  = "fsx.mount_name"
    value = var.fsx.mount_name
  }

  set {
    name  = "fsx.dns_name"
    value = var.fsx.dns_name
  }

}

resource "helm_release" "user_profile_pv_efs" {
  chart = "${var.local_helm_repo}/pv-efs"
  name = "user-profile-pv-efs"
  version = "1.0.0"
  namespace = var.kubeflow_namespace
  
  set {
    name  = "efs.volume_name"
    value = "user-profile-pv-efs"
  }

  set {
    name  = "efs.claim_name"
    value = "pv-efs"
  }

  set {
    name  = "efs.class_name"
    value = "user-profile-efs-sc"
  }

  set {
    name  = "namespace"
    value = kubernetes_namespace.kubeflow_user_profile.metadata[0].name
  }

  set {
    name  = "efs.fs_id"
    value = var.efs_fs_id
  }
}

resource "helm_release" "user_profile_pv_fsx" {
  chart = "${var.local_helm_repo}/pv-fsx"
  name = "user-profile-pv-fsx"
  version = "1.1.0"
  namespace = var.kubeflow_namespace
  
   set {
    name  = "fsx.volume_name"
    value = "user-profile-pv-fsx"
  }

  set {
    name  = "fsx.claim_name"
    value = "pv-fsx"
  }

  set {
    name  = "fsx.class_name"
    value = "user-profile-fsx-sc"
  }

  set {
    name  = "namespace"
    value = kubernetes_namespace.kubeflow_user_profile.metadata[0].name
  }

  set {
    name  = "fsx.fs_id"
    value = var.fsx.fs_id
  }

  set {
    name  = "fsx.mount_name"
    value = var.fsx.mount_name
  }

  set {
    name  = "fsx.dns_name"
    value = var.fsx.dns_name
  }

}

resource "helm_release" "kubeflow-training-operator" {
  count = var.enable_training_operator ? 1 : 0

  name       = "kubeflow-training-operator"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-training-operator"
  version  = "1.0.0"
  namespace = var.kubeflow_namespace

  set {
    name  = "namespace"
    value = var.kubeflow_namespace
  }
}

resource "helm_release" "kuberay-operator" {
  name       = "kuberay-operator"
  chart      = "kuberay-operator"
  repository  = "https://ray-project.github.io/kuberay-helm/"
  version    = "1.4.2"
  namespace  = var.kubeflow_namespace
}

resource "helm_release" "kubeflow-roles" {
  name       = "kubeflow-roles"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-roles"
  version  = "1.0.0"
  namespace = var.kubeflow_namespace
}


resource "helm_release" "kubeflow-admission-webhook" {
  count = var.kubeflow_platform_enabled ? 1 : 0

  name       = "kubeflow-admission-webhook"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-admission-webhook"
  version  = "1.0.0"
  namespace = var.kubeflow_namespace

  values = [
    <<-EOT
      kubeflow:
        namespace: "${var.kubeflow_namespace}"
    EOT
  ]

  depends_on = [ helm_release.kubeflow-roles ]
}

resource "helm_release" "kubeflow-profiles-and-kfam" {
  count = var.kubeflow_platform_enabled ? 1 : 0

  name       = "kubeflow-profiles-and-kfam"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-profiles-and-kfam"
  version  = "1.0.0"
  namespace = var.kubeflow_namespace

  values = [
    <<-EOT
      profile_controller:
        role_arn: "${var.profile_controller_role_arn}"
      kubeflow:
        namespace: "${var.kubeflow_namespace}"
      ingress:
        namespace: "${var.ingress_namespace}"
        gateway: "${var.ingress_gateway}"
        sa: "${var.ingress_sa}"
      notebook_controller:
        sa: "notebook-controller-service-account"
      pipeline_ui:
        sa: "ml-pipeline-ui"
    EOT
  ]

  depends_on = [ helm_release.kubeflow-admission-webhook ]
}

resource "helm_release" "kubeflow-notebooks" {
  count = var.kubeflow_platform_enabled ? 1 : 0

  name       = "kubeflow-notebooks"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-notebooks"
  version  = "1.0.0"
  namespace = var.kubeflow_namespace

  values = [
    <<-EOT
      kubeflow:
        namespace: "${var.kubeflow_namespace}"
      ingress:
        namespace: "${var.ingress_namespace}"
        gateway: "${var.ingress_gateway}"
        sa: "${var.ingress_sa}"
    EOT
  ]

  depends_on = [ helm_release.kubeflow-profiles-and-kfam ]
}

resource "helm_release" "kubeflow-tensorboards" {
  count = var.kubeflow_platform_enabled ? 1 : 0

  name       = "kubeflow-tensorboards"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-tensorboards"
  version  = "1.0.0"
  namespace = var.kubeflow_namespace

  values = [
    <<-EOT
      kubeflow:
        namespace: "${var.kubeflow_namespace}"
      ingress:
        namespace: "${var.ingress_namespace}"
        gateway: "${var.ingress_gateway}"
        sa: "${var.ingress_sa}"
    EOT
  ]

  depends_on = [ helm_release.kubeflow-profiles-and-kfam ]
}

resource "helm_release" "kubeflow-pipelines" {
  count = var.kubeflow_platform_enabled ? 1 : 0

  name       = "kubeflow-pipelines"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-pipelines"
  version  = "1.0.0"
  namespace = var.kubeflow_namespace

  values = [
    <<-EOT
      kubeflow:
        namespace: "${var.kubeflow_namespace}"
      ingress:
        namespace: "${var.ingress_namespace}"
        gateway: "${var.ingress_gateway}"
        sa: "${var.ingress_sa}"
      minio:
        access_key: "${random_string.minio_access_key.result}"
        secret_key: "${random_password.minio_secret_key.result}"
      
    EOT
  ]

  depends_on = [ helm_release.kubeflow-profiles-and-kfam ]
}

resource "helm_release" "kubeflow-volumes" {
  count = var.kubeflow_platform_enabled ? 1 : 0

  name       = "kubeflow-volumes"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-volumes"
  version  = "1.0.0"
  namespace = var.kubeflow_namespace

  values = [
    <<-EOT
      kubeflow:
        namespace: "${var.kubeflow_namespace}"
      ingress:
        namespace: "${var.ingress_namespace}"
        gateway: "${var.ingress_gateway}"
        sa: "${var.ingress_sa}"
      
    EOT
  ]

  depends_on = [ helm_release.kubeflow-profiles-and-kfam ]
}

resource "helm_release" "kubeflow-user-profile" {
  count = var.kubeflow_platform_enabled ? 1 : 0

  name       = "kubeflow-user-profile"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-user-profile"
  version  = "1.0.0"
  namespace = var.kubeflow_namespace

 values = [
    <<-EOT
      user: 
        email: ${var.static_email}
        profile: ${var.kubeflow_user_profile}
      awsIamForServiceAccount:
        awsIamRole: ${var.user_profile_role_arn}
    EOT
  ]

  depends_on = [ helm_release.kubeflow-profiles-and-kfam ]
}

resource "helm_release" "kubeflow-katib" {
  count = var.kubeflow_platform_enabled ? 1 : 0

  name       = "kubeflow-katib"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-katib"
  version  = "1.0.1"
  namespace = var.kubeflow_namespace

  values = [
    <<-EOT
      kubeflow:
        namespace: "${var.kubeflow_namespace}"
      ingress:
        namespace: "${var.ingress_namespace}"
        gateway: "${var.ingress_gateway}"
        sa: "${var.ingress_sa}"
      
    EOT
  ]

  depends_on = [ helm_release.kubeflow-pipelines ]
}

resource "helm_release" "kubeflow-user-profile-defaults" {
  count = var.kubeflow_platform_enabled ? 1 : 0

  name       = "kubeflow-user-profile-defaults"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-user-profile-defaults"
  version  = "1.0.7"
  namespace = var.kubeflow_namespace

 values = [
    <<-EOT
      user: 
        profile: ${var.kubeflow_user_profile}
    EOT
  ]

  depends_on = [ helm_release.kubeflow-user-profile ]
}

resource "helm_release" "kubeflow-central-dashboard" {
  count = var.kubeflow_platform_enabled ? 1 : 0

  name       = "kubeflow-central-dashboard"
  chart      = "${var.local_helm_repo}/ml-platform/kubeflow-central-dashboard"
  version  = "1.0.0"
  namespace = var.kubeflow_namespace

  values = [
    <<-EOT
      kubeflow:
        namespace: "${var.kubeflow_namespace}"
      user: 
        profile: ${var.kubeflow_user_profile}
      ingress:
        namespace: "${var.ingress_namespace}"
        gateway: "${var.ingress_gateway}"
        sa: "${var.ingress_sa}"
    EOT
  ]

  depends_on = [
    helm_release.pv_efs,
    helm_release.pv_fsx,
    helm_release.user_profile_pv_efs,
    helm_release.user_profile_pv_fsx,
    helm_release.kubeflow-user-profile-defaults,
    helm_release.kubeflow-tensorboards,
    helm_release.kubeflow-volumes,
    helm_release.kubeflow-pipelines,
    helm_release.kubeflow-katib
  ]
}