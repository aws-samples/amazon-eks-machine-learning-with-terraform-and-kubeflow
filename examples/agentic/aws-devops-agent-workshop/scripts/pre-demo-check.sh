#!/usr/bin/env bash
# Pre-demo sanity check: cluster reachable, container insights present, plugin installed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "PRE-DEMO CHECK"
echo "Context: $(kubectl config current-context 2>/dev/null || echo '<none>')"
echo "Cluster: $EKS_CLUSTER_NAME    Region: $AWS_REGION    Namespace: $DEMO_NAMESPACE"
echo

ok=1

step() { printf "  [%s] %s\n" "$1" "$2"; }

if kubectl get nodes >/dev/null 2>&1; then
  step "OK" "kubectl can reach the cluster"
else
  step "FAIL" "kubectl cannot reach the cluster"
  ok=0
fi

if kubectl get pods -n amazon-cloudwatch 2>/dev/null | grep -q Running; then
  step "OK" "Container Insights running in amazon-cloudwatch namespace"
else
  step "WARN" "Container Insights not detected (OOMKill scene will lack metrics)"
fi

if command -v session-manager-plugin >/dev/null 2>&1; then
  step "OK" "session-manager-plugin installed"
else
  step "WARN" "session-manager-plugin not installed (not required for new DNS scenario)"
fi

if kubectl get deploy -n kube-system coredns >/dev/null 2>&1; then
  replicas=$(kubectl get deploy coredns -n kube-system -o jsonpath='{.spec.replicas}')
  step "OK" "CoreDNS deployment found (replicas=$replicas)"
else
  step "FAIL" "CoreDNS deployment not found in kube-system"
  ok=0
fi

if kubectl get ns "$DEMO_NAMESPACE" >/dev/null 2>&1; then
  step "INFO" "Namespace $DEMO_NAMESPACE already exists — run cleanup.sh if from a prior run"
else
  step "OK" "Namespace $DEMO_NAMESPACE not yet created"
fi

echo
if [[ $ok -eq 1 ]]; then
  echo "Ready to demo."
else
  echo "FAIL — fix the issues above before starting."
  exit 1
fi
