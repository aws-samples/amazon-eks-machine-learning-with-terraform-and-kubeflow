#!/bin/bash
# HyperPod Status Check Script
# Usage: ./check-hyperpod-status.sh <cluster-name> <region>

CLUSTER_NAME=${1:-"hyperpod-test-cluster"}
REGION=${2:-"us-west-2"}

echo "=========================================="
echo "      HyperPod Status Check"
echo "      Cluster: $CLUSTER_NAME"
echo "=========================================="

echo -e "\n=== Helm Release ==="
helm list -n kube-system | grep -E "NAME|hyperpod"

echo -e "\n=== HyperPod Pods ==="
kubectl get pods -n aws-hyperpod

echo -e "\n=== HyperPod DaemonSets ==="
kubectl get daemonsets -n aws-hyperpod

echo -e "\n=== HyperPod Deployments ==="
kubectl get deployments -n aws-hyperpod

echo -e "\n=== HyperPod Nodes ==="
kubectl get nodes -l sagemaker.amazonaws.com/compute-type=hyperpod \
  -o custom-columns=\
'NAME:.metadata.name,'\
'INSTANCE-TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,'\
'HEALTH:.metadata.labels.sagemaker\.amazonaws\.com/node-health-status,'\
'DEEP-HEALTH:.metadata.labels.sagemaker\.amazonaws\.com/deep-health-check-status'

echo -e "\n=== SageMaker Cluster Status ==="
aws sagemaker describe-cluster --cluster-name "$CLUSTER_NAME" --region "$REGION" \
  --query '{Status:ClusterStatus,FailureMessage:FailureMessage}' --output table 2>/dev/null

echo -e "\n=== SageMaker Cluster Nodes ==="
aws sagemaker list-cluster-nodes --cluster-name "$CLUSTER_NAME" --region "$REGION" \
  --query 'ClusterNodeSummaries[*].{Instance:InstanceId,Type:InstanceType,Group:InstanceGroupName,Status:InstanceStatus.Status}' \
  --output table 2>/dev/null
