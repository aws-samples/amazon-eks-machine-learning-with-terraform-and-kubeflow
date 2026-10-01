# Volcano, installed here for exactly one reason: it is what unlocks the Dynamo operator's
# multi-node code path. It is therefore tied to dynamo_enabled rather than carrying a flag of
# its own -- a cluster that runs Dynamo needs it, and a cluster that does not has no use for it.
#
# Why it is not optional. Dynamo resolves a LeaderWorkerSet feature gate by probing the API
# server, in internal/features/gates.go at v1.5.0:
#
#     lwsAvailable, _     := detectAPIAvailability(ctx, cfg, "leaderworkerset.x-k8s.io", "", "")
#     volcanoAvailable, _ := detectAPIAvailability(ctx, cfg, "scheduling.volcano.sh", "", "")
#     if ptr.Deref(config.Orchestrators.LWS.Enabled, lwsAvailable && volcanoAvailable) {
#         if !lwsAvailable     { return Gates{}, fmt.Errorf(...) }
#         if !volcanoAvailable { return Gates{}, fmt.Errorf(...) }
#         gates.LWS = true
#     }
#
# LWS alone is not enough, and setting the operator's own orchestrators.lws.enabled does not
# relax the Volcano half -- because the field is left unset the deref falls through to the
# conjunction, so the gate simply resolves false and the operator starts normally. Explicitly
# enabling it would instead fail operator startup outright. Either way there is no configuration
# that gets the multi-node path without this API group present.
#
# The consequence of the gate being off is quiet: a DynamoGraphDeployment carrying
# multinode.nodeCount fails reconciliation with `no_multinode_orchestrator_available` and creates
# no LeaderWorkerSet, no DynamoComponentDeployment and no pods, logging nothing at default
# verbosity. Single-node deployments are entirely unaffected, which is why this is easy to hit
# late. See examples/inference/dynamo/vllm/qwen3-32b/README.md.
#
# Note this is separate from Dynamo's own Volcano *integration*, which is a third gate keyed on
# config.Orchestrators.VolcanoScheduler.Enabled and defaults to false. The LWS gate wants the API
# group to exist; it does not want workloads routed through the Volcano scheduler.
#
# The alternative upstream offers is Grove + KAI Scheduler. Volcano is chosen because this
# cluster already installs LWS unconditionally, so Volcano completes a stack that is otherwise
# only half present, while Grove would add two more controllers and a second scheduler.
#
# What this does NOT do: route any existing workload through Volcano. Nothing in this repository
# sets schedulerName: volcano, and the LWS release in main.tf is not configured with
# gangSchedulingManagement, so pods continue to be placed by kube-scheduler. Real gang scheduling
# -- every pod of a multi-node group admitted together or none -- is a follow-on change. It
# matters at scale, and it is not what unblocks a two-node deployment, because the operator
# injects a wait-for-leader init container whose poll loop has no timeout.
resource "helm_release" "volcano" {
  count = var.dynamo_enabled ? 1 : 0

  name = "volcano"

  # Installed from the published repository rather than copied into charts/, like every other
  # third-party release in this configuration: the chart is used exactly as shipped, with no
  # value overrides below, so a local copy would be 45k lines of upstream CRD YAML that nobody
  # here maintains. The provider resolves `repository` itself, so no `helm repo add` is needed.
  #
  # Pinned, because the CRDs this installs are what the Dynamo operator's startup probe looks
  # for, and a silent major-version bump would change that API surface under it.
  repository = "https://volcano-sh.github.io/helm-charts"
  chart      = "volcano"
  version    = "1.15.2"

  namespace        = "volcano-system"
  create_namespace = true
  cleanup_on_fail  = true
  timeout          = 600
  wait             = true

  # Chart defaults are left alone. They already install only the admission, controller and
  # scheduler components with metrics_enable = false and vap_enable = false, and
  # custom.enabled_admissions does not register the jobflow webhook, so that subchart's CRDs are
  # present but nothing admits against them.

  depends_on = [helm_release.cluster-autoscaler]
}
