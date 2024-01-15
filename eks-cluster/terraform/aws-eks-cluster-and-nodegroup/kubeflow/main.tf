resource "helm_release" "mpijob" {
  name       = "mpijob"
  chart      = "${var.local_helm_repo}/mpijob"
  version    = "2.0.0"
}

resource "helm_release" "mpi-operator" {
  name       = "mpi-operator"
  chart      = "${var.local_helm_repo}/mpi-operator"
  version    = "2.0.0"

  namespace = var.kubeflow_namespace

  set {
    name  = "namespace"
    value = var.kubeflow_namespace
  }
}
