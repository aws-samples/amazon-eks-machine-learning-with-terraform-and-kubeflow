#!/usr/bin/env bash
# Scenario 1: Morning Health Check. No setup needed — prints prompts only.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "SCENARIO 1: Morning Health Check ($EKS_CLUSTER_NAME / $AWS_REGION)"

prompt "Run a health check on my EKS cluster $EKS_CLUSTER_NAME in $AWS_REGION. Check all nodes for pressure conditions, identify any pods in error states, and flag any resource concerns."

echo
echo "FOLLOW-UP PROMPT (after results):"
echo "  Which namespaces are consuming the most resources? Are any workloads at"
echo "  risk of being OOMKilled based on current utilization vs. limits?"
