#!/usr/bin/env bash
# Scenario 4B: Container Insights follow-up. Run AFTER scenario-4 setup is live
# AND after the agent's first investigation. Forces it to use CW metrics
# (cfs_throttled_seconds, node_cpu_utilization) instead of inferring from K8s state.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "SCENARIO 4B: Container Insights follow-up"
echo "Prereqs:"
echo "  • scenario-4-cascading-failure-setup.sh has been run (batch-processor live)"
echo "  • CloudWatch Observability EKS add-on installed (it is — verified 2026-06-22)"
echo

if ! kubectl get deploy batch-processor -n "$DEMO_NAMESPACE" >/dev/null 2>&1; then
  echo "⚠️  batch-processor not found in $DEMO_NAMESPACE — run scenario-4-cascading-failure-setup.sh first."
  exit 1
fi

echo "batch-processor is live. Sample of metrics the agent should query:"
echo
aws cloudwatch list-metrics --namespace ContainerInsights --region "$AWS_REGION" \
  --metric-name node_cpu_utilization \
  --dimensions Name=ClusterName,Value="$EKS_CLUSTER_NAME" \
  --query 'Metrics[0:3].Dimensions[?Name==`NodeName`].Value' --output text 2>/dev/null \
  | head -5
echo

prompt "Now that CloudWatch Container Insights is enabled on cluster $EKS_CLUSTER_NAME, re-investigate the order-api / inventory-api degradation using the time-series metrics — not just current Kubernetes state. Specifically:

1. Quote container_cpu_cfs_throttled_seconds for order-api and inventory-api pods over the last 15 minutes — what percentage of their CPU time is being stolen?
2. Plot node_cpu_utilization for each of the 8 nodes. Identify which nodes are hot vs cold and why.
3. Give me the exact UTC timestamp when node CPU on the affected nodes crossed 50% — correlate it with the CloudTrail event for the batch-processor apply.
4. For pod_memory_working_set on the order-api pods, is memory pressure also contributing or is this purely CPU?
5. Compare affected nodes to a baseline node — what's the delta in cpu_utilization, pod density, and throttling?

Use the metrics to quantify the cascade, not infer it."

echo
echo "What to watch for in the agent's answer (and on the CW dashboard side-by-side):"
echo "  ✓ Numeric CFS throttling percentages, not 'pods are being throttled'"
echo "  ✓ Specific UTC timestamps for the cascade onset"
echo "  ✓ Per-node comparison (hot vs cold) with deltas"
echo "  ✓ Distinguishes CPU pressure from memory pressure"
echo "  ✓ Cites CloudWatch namespace/dimensions in tool calls (visible in MCP gateway log)"
