locals {
  slurm_git_url = "https://github.com/stackhpc/slurm-k8s-cluster"
  slurm_git_commit = "a0a23230f8eac3b4ec168d9e76181e95c42ffcd5"
}

resource "kubernetes_namespace" "slurm" {
  metadata {
    labels = {
      istio-injection = "disabled"
    }

    name = "${var.slurm_namespace}"
  }
}


resource "null_resource" "slurm_git_clone" {
  triggers = {
    always_run = "${timestamp()}"
  }

  provisioner "local-exec" {
    command = <<-EOT
      rm -rf /tmp/slurm || true
      git clone ${local.slurm_git_url} /tmp/slurm
      pushd /tmp/slurm
      git fetch origin ${local.slurm_git_commit}
      git reset --hard ${local.slurm_git_commit}
      mkdir /tmp/slurm/slurm-cluster-chart/charts
      popd
      cp -r ${var.local_helm_repo}/pv-efs /tmp/slurm/slurm-cluster-chart/charts/
      rm /tmp/slurm/slurm-cluster-chart/templates/pvc.yaml
      sed -i '/dependencies:/,$d' /tmp/slurm/slurm-cluster-chart/Chart.yaml
      sed -i 's/LoadBalancer/ClusterIP/1' /tmp/slurm/slurm-cluster-chart/templates/login-service.yaml
    EOT
  }

  depends_on = [kubernetes_namespace.slurm]

}

resource "helm_release" "slurm" {
  chart = "/tmp/slurm/slurm-cluster-chart"
  name = "slurm"
  namespace = var.slurm_namespace
  
  values = [
    <<-EOT
      rooknfs:
        enabled: false
      database:
        image: mariadb:10.10
        storage: 10Gi
      storage:
        mountPath: /home
        storageClassName: slurm-efs-sc
        claimName: pv-efs
        capacity: ${var.storage_capacity}
      pv-efs:
        namespace: ${kubernetes_namespace.slurm.metadata[0].name}
        efs:
          volume_name: slurm-pv-efs
          claim_name: pv-efs
          class_name: slurm-efs-sc
          fs_id: ${var.efs_fs_id}
          storage: ${var.storage_capacity}
      openOnDemand:
        password: "${var.password}"
      sshPublicKey: "${var.ssh_public_key}"
    EOT
  ]

  depends_on = [null_resource.slurm_git_clone]
}