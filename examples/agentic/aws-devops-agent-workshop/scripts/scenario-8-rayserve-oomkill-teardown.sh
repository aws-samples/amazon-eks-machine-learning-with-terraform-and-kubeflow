#!/usr/bin/env bash
# Teardown for Scenario 8. Deletes the RayCluster + namespace.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

NAMESPACE="ml-inference"

banner "TEARDOWN: Scenario 8 (Ray worker OOMKill)"
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete raycluster nova-vision-model --ignore-not-found
# Don't delete the namespace if Scenario 7 RayCluster is still around.
if ! kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" get raycluster nova-language-model >/dev/null 2>&1; then
  kubectl --context "$EKS_CLUSTER_NAME" delete namespace "$NAMESPACE" --ignore-not-found
fi
echo "✓ Done."
