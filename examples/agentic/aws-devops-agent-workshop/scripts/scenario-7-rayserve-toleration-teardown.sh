#!/usr/bin/env bash
# Teardown for Scenario 7. Deletes the RayCluster + namespace.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

NAMESPACE="ml-inference"

banner "TEARDOWN: Scenario 7 (Ray toleration mismatch)"
kubectl --context "$EKS_CLUSTER_NAME" -n "$NAMESPACE" delete raycluster nova-language-model --ignore-not-found
kubectl --context "$EKS_CLUSTER_NAME" delete namespace "$NAMESPACE" --ignore-not-found
echo "✓ Done."
