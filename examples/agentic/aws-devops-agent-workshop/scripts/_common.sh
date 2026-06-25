#!/usr/bin/env bash
# Shared helpers for demo scenario scripts. Source this file at the top of each.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"

# Pull pinned defaults if env.sh is present, but let kubeconfig context win.
if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/env.sh" >/dev/null
fi

KCTX="$(kubectl config current-context 2>/dev/null || true)"
if [[ -n "$KCTX" && "$KCTX" =~ ^arn:aws:eks:([^:]+):[0-9]+:cluster/(.+)$ ]]; then
  AWS_REGION="${BASH_REMATCH[1]}"
  EKS_CLUSTER_NAME="${BASH_REMATCH[2]}"
elif [[ -n "$KCTX" ]]; then
  EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-$KCTX}"
fi

export AWS_REGION="${AWS_REGION:-us-west-2}"
export EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-unknown}"
export DEMO_NAMESPACE="${DEMO_NAMESPACE:-demo-app}"

banner() {
  echo "============================================"
  echo "$*"
  echo "============================================"
}

prompt() {
  echo
  echo "============================================"
  echo "INVESTIGATION PROMPT — paste into Agent Space:"
  echo "============================================"
  echo
  echo "$*"
  echo
  echo "============================================"
}

ensure_namespace() {
  kubectl create namespace "$DEMO_NAMESPACE" 2>/dev/null || true
}
