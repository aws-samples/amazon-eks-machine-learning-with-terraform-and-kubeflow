#!/bin/bash

set -e

echo "Starting cleanup before infrastructure destroy..."
echo ""

# Step 1: Uninstall all Helm releases
echo "Step 1/4: Uninstalling all Helm releases in kubeflow-user-example-com namespace..."
for x in $(helm list -q -n kubeflow-user-example-com 2>/dev/null || true); do 
  echo "  Uninstalling $x..."
  helm uninstall $x -n kubeflow-user-example-com
done
echo "  Done."
echo ""

# Step 2: Wait 5 minutes
echo "Step 2/4: Waiting 5 minutes for resources to terminate..."
sleep 300
echo "  Done."
echo ""

# Step 3: Delete remaining pods
echo "Step 3/4: Deleting remaining pods in kubeflow-user-example-com namespace..."
kubectl delete --all pods -n kubeflow-user-example-com --ignore-not-found=true
echo "  Done."
echo ""

# Step 4: Delete attach-pvc pod
echo "Step 4/4: Deleting attach-pvc pod in kubeflow namespace..."
kubectl delete -f eks-cluster/utils/attach-pvc.yaml -n kubeflow --ignore-not-found=true
echo "  Done."
echo ""

# Final wait
echo "Waiting 15 minutes for auto-scaling GPU and Neuron nodes to zero..."
sleep 900
echo ""

echo "Cleanup before terraform destroy complete! You can now run terraform destroy."
