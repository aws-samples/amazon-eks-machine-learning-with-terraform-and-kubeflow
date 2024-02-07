resource "random_string" "minio_access_key" {
  length           = 16
  special          = false
}

resource "random_password" "minio_secret_key" {
  length           = 32
  special          = false
}

resource "helm_release" "kubeflow-training-operator" {
  name       = "kubeflow-training-operator"
  chart      = "${var.local_helm_repo}/kubeflow-training-operator"
  version  = "1.0.0"

  set {
    name  = "namespace"
    value = var.kubeflow_namespace
  }
}

resource "helm_release" "kubeflow-roles" {
   name       = "kubeflow-roles"
  chart      = "${var.local_helm_repo}/kubeflow-roles"
  version  = "1.0.0"
}


resource "helm_release" "kubeflow-admission-webhook" {
  name       = "kubeflow-admission-webhook"
  chart      = "${var.local_helm_repo}/kubeflow-admission-webhook"
  version  = "1.0.0"

  values = [
    <<-EOT
      kubeflow:
        namespace: "${var.kubeflow_namespace}"
    EOT
  ]

  depends_on = [ helm_release.kubeflow-roles ]
}

resource "helm_release" "kubeflow-profiles-and-kfam" {
  name       = "kubeflow-profiles-and-kfam"
  chart      = "${var.local_helm_repo}/kubeflow-profiles-and-kfam"
  version  = "1.0.0"

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
  name       = "kubeflow-notebooks"
  chart      = "${var.local_helm_repo}/kubeflow-notebooks"
  version  = "1.0.0"

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
  name       = "kubeflow-tensorboards"
  chart      = "${var.local_helm_repo}/kubeflow-tensorboards"
  version  = "1.0.0"

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
  name       = "kubeflow-pipelines"
  chart      = "${var.local_helm_repo}/kubeflow-pipelines"
  version  = "1.0.0"

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
  name       = "kubeflow-volumes"
  chart      = "${var.local_helm_repo}/kubeflow-volumes"
  version  = "1.0.0"

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
  name       = "kubeflow-user-profile"
  chart      = "${var.local_helm_repo}/kubeflow-user-profile"
  version  = "1.0.0"

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
  name       = "kubeflow-katib"
  chart      = "${var.local_helm_repo}/kubeflow-katib"
  version  = "1.0.0"

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

resource "helm_release" "kubeflow-user-defaults" {
  name       = "kubeflow-user-defaults"
  chart      = "${var.local_helm_repo}/kubeflow-user-defaults"
  version  = "1.0.0"

 values = [
    <<-EOT
      user: 
        profile: ${var.kubeflow_user_profile}
      efs:
        fs_id: ${var.efs_fs_id}
      fsx:
        fs_id: ${var.fsx.fs_id}
        dns_name: ${var.fsx.dns_name}
        mount_name: ${var.fsx.mount_name}
    EOT
  ]

  depends_on = [ helm_release.kubeflow-user-profile, helm_release.kubeflow-pipelines ]
}

resource "helm_release" "kubeflow-central-dashboard" {
  name       = "kubeflow-central-dashboard"
  chart      = "${var.local_helm_repo}/kubeflow-central-dashboard"
  version  = "1.0.0"

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
    helm_release.kubeflow-user-defaults,
    helm_release.kubeflow-tensorboards,
    helm_release.kubeflow-volumes,
    helm_release.kubeflow-pipelines,
    helm_release.kubeflow-katib
  ]
}