#!/usr/bin/env bash
# Everything the pipeline needs to be true before a measurement round, checked offline or with
# read-only cluster calls. Nothing here installs, deletes or modifies anything.
#
# Exits 1 if it found something that would spoil a run, 0 if it found nothing.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
v2=$(cd "$here/../../.." && pwd)
ns=${COLOCATION_NAMESPACE:-kubeflow-user-example-com}
problems=0

say()  { printf '%s\n' "$*"; }
bad()  { printf 'PROBLEM  %s\n' "$*"; problems=$((problems + 1)); }

say "== cluster =="
ctx=$(kubectl config current-context 2>/dev/null)
if [ -z "$ctx" ]; then
  bad "no kubectl context selected; every cluster check below is skipped"
else
  say "context   $ctx"
  say "region    $(printf '%s' "$ctx" | cut -d: -f4)"
  say "namespace $ns"
fi

say
say "== one DCGM exporter, and it must be ours =="
# A GPU allows one DCGM host engine to own its profiling fields. The second to ask crash-loops,
# and every record of the run it was instrumenting comes back with no per_gpu block: the run
# reports success and has lost every cause signal. A DaemonSet counts even at zero pods,
# because it follows the lease onto the node.
if [ -n "$ctx" ]; then
  foreign_ds=$(kubectl get ds -A -o custom-columns=NS:.metadata.namespace,NAME:.metadata.name \
                 --no-headers 2>/dev/null | grep -i dcgm | grep -v "^$ns " || true)
  foreign_po=$(kubectl get pods -A -o custom-columns=NS:.metadata.namespace,NAME:.metadata.name \
                 --no-headers 2>/dev/null | grep -i dcgm | grep -v "^$ns " || true)
  foreign_cr=$(kubectl get dcgmexporters -A --no-headers 2>/dev/null | grep -v "^$ns " || true)
  if [ -n "$foreign_ds" ] || [ -n "$foreign_po" ] || [ -n "$foreign_cr" ]; then
    bad "another DCGM exporter is on this cluster:"
    [ -n "$foreign_ds" ] && printf '%s\n' "$foreign_ds"
    [ -n "$foreign_po" ] && printf '%s\n' "$foreign_po"
    [ -n "$foreign_cr" ] && printf '%s\n' "$foreign_cr"
    say "  On EKS this is normally the amazon-cloudwatch-observability addon, which ships it on"
    say "  by default. Deleting the DaemonSet or the CR is undone by its operator within seconds;"
    say "  turn it off at the addon:"
    say "    aws eks update-addon --cluster-name <cluster> --region <region> \\"
    say "      --addon-name amazon-cloudwatch-observability \\"
    say "      --configuration-values '{\"dcgmExporter\":{\"enabled\":false}}' --resolve-conflicts OVERWRITE"
    say "  Cost: Container Insights loses its GPU dashboards. Node, pod and log metrics are unaffected."
  else
    say "no foreign exporter found"
  fi
fi

say
say "== the GPU node is held =="
# An unheld GPU node is reclaimed in about 600 s, and the replacement's GPU UUIDs make every
# stored placement un-reusable. The holder is a bare Pod at priority -1000, so a run preempts it
# instead of ordering a second machine, and preemption deletes it: it does not survive a run.
if [ -n "$ctx" ]; then
  holder=$(kubectl get pod gpu-node-holder -n "$ns" \
             -o jsonpath='{.status.phase}' 2>/dev/null || true)
  daemon=$(kubectl get pods -n "$ns" -l app.kubernetes.io/name=mps-daemon \
             -o jsonpath='{range .items[*]}{.metadata.name}={.status.phase} {end}' 2>/dev/null || true)
  if [ "$holder" = "Running" ]; then
    say "gpu-node-holder Running"
  elif [ -n "$daemon" ]; then
    say "no holder, but an mps-daemon is up: $daemon"
    say "  A live daemon holds the node. It stops holding the moment it is uninstalled."
  elif [ -n "$holder" ]; then
    bad "gpu-node-holder is $holder, not Running — a Pending holder holds nothing, and carries"
    say "  none of its own do-not-disrupt protection either. It needs every GPU on the node, so"
    say "  any tenant pod still up leaves it Pending."
  else
    bad "nothing is holding the GPU node: no gpu-node-holder pod and no mps-daemon"
    say "  kubectl apply -f $v2/manifests/gpu-node-holder.yaml"
  fi
fi

say
say "== the harness image =="
tag=$(bash "$here/harness-image.sh" --tag-only 2>/dev/null)
say "tag from source  $tag"
if [ -n "$ctx" ]; then
  region=$(printf '%s' "$ctx" | cut -d: -f4)
  if aws ecr describe-images --repository-name colocation-harness --region "$region" \
       --image-ids imageTag="$tag" >/dev/null 2>&1; then
    say "in ECR           yes, so profile and search cost nothing to start"
  else
    say "in ECR           no, so the first profile or search builds, self-tests and pushes it"
  fi
fi

say
say "== the harness Job template's cluster-specific fields =="
# Each of these fails as a Job that sits Pending with no event explaining why. They are typed in
# by hand, so they are wrong on a cluster the file was not written for, and being wrong has cost
# a run: the template pinned a zone from another region, the driver could not be scheduled, and
# measure polled a Job that could never start while an engine held two GPUs.
job=$v2/containers/colocation-harness/deploy/job.yaml
if [ -f "$job" ]; then
  grep -nE 'namespace:|karpenter\.sh/nodepool|compute-tier|topology\.kubernetes\.io/zone' "$job" \
    | sed 's/^/  /'
  if grep -q "namespace: *$ns" "$job"; then
    say "namespace matches $ns"
  else
    bad "$job does not name namespace $ns"
  fi
  # An actual key, not the comment saying the key is deliberately absent.
  if grep -qE '^[[:space:]]*topology\.kubernetes\.io/zone:' "$job"; then
    bad "the template pins a zone; measure.py reads the zone off the node under test, and a"
    say "  pinned one from another region is unschedulable"
  fi
else
  bad "no $job"
fi

say
say "== stores =="
say "local     ${COLOCATION_STORE:-$HOME/.colocation/store}"
say "canonical ${COLOCATION_CANONICAL_STORE:-/efs/colocation}  (readable only from inside a pod)"

say
if [ "$problems" -gt 0 ]; then
  say "$problems problem(s) above. Fix them before measuring; do not work around them."
  exit 1
fi
say "nothing found."
