#---------------------------------------------------------------
# kagent - Kubernetes Native AI Agent Framework
# https://github.com/kagent-dev/kagent
#---------------------------------------------------------------

module "kagent" {
  count  = var.kagent_enabled ? 1 : 0
  source = "./kagent"

  # Basic configuration
  kagent_namespace = var.kagent_namespace
  kagent_version   = var.kagent_version
  cluster_name     = var.cluster_name
  region           = var.region

  # EKS cluster information
  eks_cluster_id        = aws_eks_cluster.eks_cluster.id
  eks_oidc_provider_arn = aws_iam_openid_connect_provider.eks_oidc_provider.arn

  # Database configuration
  database_type        = var.kagent_database_type
  controller_replicas  = var.kagent_controller_replicas
  db_max_capacity      = var.kagent_db_max_capacity

  # PostgreSQL infrastructure (only needed if database_type is postgresql)
  vpc_id         = var.kagent_database_type == "postgresql" ? aws_vpc.vpc.id : ""
  vpc_cidr       = var.kagent_database_type == "postgresql" ? var.cidr_vpc : ""
  db_subnet_ids  = var.kagent_database_type == "postgresql" ? aws_subnet.private.*.id : []

  # UI configuration
  enable_ui = var.kagent_enable_ui

  # Istio ingress configuration
  enable_istio_ingress = var.kagent_enable_istio_ingress
  istio_gateway        = "ingress/istio-ingressgateway"
  ingress_host         = "istio-ingressgateway.ingress.svc.cluster.local"
  ingress_path_prefix  = "/kagent"

  # Istio sidecar injection
  enable_istio_injection = var.kagent_enable_istio_injection

  # AWS Bedrock access via IRSA
  enable_bedrock_access = var.kagent_enable_bedrock_access

  depends_on = [
    module.eks_blueprints_addons,
    helm_release.cluster-autoscaler,
    helm_release.istio-ingress
  ]
}
