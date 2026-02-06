#!/bin/bash
# EKS Ops Agent - Build and Deploy
#
# This script builds the container, pushes to ECR, and deploys to kagent with IRSA.
#
# Prerequisites:
#   1. Docker installed and running
#   2. AWS CLI configured with ECR and EKS access
#   3. kubectl configured for your EKS cluster
#   4. kagent installed with kagent_enable_bedrock_access=true
#
# Usage:
#   ./build-and-deploy.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="kagent"
AGENT_NAME="eks-ops-agent"

# Configuration
IMAGE_NAME="eks-ops-agent"
VERSION="${VERSION:-0.1.0}"
AWS_REGION="${AWS_REGION:-us-west-2}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "EKS Ops Agent - Build and Deploy"
echo "========================================"

#
# PHASE 1: Build and Push Container
#
echo ""
echo -e "${YELLOW}Phase 1: Building container...${NC}"
echo "========================================"

echo "Image:   ${IMAGE_NAME}"
echo "Version: ${VERSION}"
echo "Region:  ${AWS_REGION}"

# Build the image
echo ""
echo "Building Docker image..."
docker build \
    --platform linux/amd64 \
    -t "${IMAGE_NAME}:${VERSION}" \
    -t "${IMAGE_NAME}:latest" \
    "${SCRIPT_DIR}"

echo "Local build complete: ${IMAGE_NAME}:${VERSION}"

# Push to ECR
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO="${ECR_REGISTRY}/${IMAGE_NAME}"

# Create ECR repository if it doesn't exist
echo ""
echo "Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "${IMAGE_NAME}" --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository \
        --repository-name "${IMAGE_NAME}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Tag and push
echo "Pushing to ECR..."
docker tag "${IMAGE_NAME}:${VERSION}" "${ECR_REPO}:${VERSION}"
docker tag "${IMAGE_NAME}:latest" "${ECR_REPO}:latest"
docker push "${ECR_REPO}:${VERSION}"
docker push "${ECR_REPO}:latest"

IMAGE_URI="${ECR_REPO}:${VERSION}"
echo ""
echo -e "${GREEN}Container pushed: ${IMAGE_URI}${NC}"

#
# PHASE 2: Deploy to kagent
#
echo ""
echo -e "${YELLOW}Phase 2: Deploying to kagent...${NC}"
echo "========================================"

# Construct the Bedrock IAM role ARN
CLUSTER_NAME=$(aws eks list-clusters --region "$AWS_REGION" --query 'clusters[0]' --output text 2>/dev/null || echo "eks-1")
BEDROCK_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${CLUSTER_NAME}-kagent-bedrock-role"

echo "Cluster:  ${CLUSTER_NAME}"
echo "IAM Role: ${BEDROCK_ROLE_ARN}"

# Verify the role exists
if ! aws iam get-role --role-name "${CLUSTER_NAME}-kagent-bedrock-role" >/dev/null 2>&1; then
    echo -e "${RED}Error: Bedrock IAM role not found: ${BEDROCK_ROLE_ARN}${NC}"
    echo "Make sure terraform was run with kagent_enable_bedrock_access=true"
    exit 1
fi

# Create a temporary manifest with the actual image URI
TEMP_MANIFEST=$(mktemp)
sed "s|image: .*|image: ${IMAGE_URI}|" "${SCRIPT_DIR}/manifests/eks-ops-agent.yaml" > "$TEMP_MANIFEST"

echo ""
echo "Applying agent manifest..."
kubectl apply -f "$TEMP_MANIFEST"
rm "$TEMP_MANIFEST"

# Wait for ServiceAccount to be created by kagent controller
echo ""
echo "Waiting for ServiceAccount to be created..."
for i in {1..30}; do
    if kubectl get serviceaccount "$AGENT_NAME" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo "ServiceAccount created."
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Error: ServiceAccount not created after 30 seconds${NC}"
        exit 1
    fi
    sleep 1
done

echo ""
echo "Annotating ServiceAccount for IRSA..."
kubectl annotate serviceaccount "$AGENT_NAME" -n "$NAMESPACE" \
    "eks.amazonaws.com/role-arn=${BEDROCK_ROLE_ARN}" \
    --overwrite

echo ""
echo "Restarting deployment to pick up IRSA credentials..."
kubectl rollout restart deployment "$AGENT_NAME" -n "$NAMESPACE"

echo ""
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment "$AGENT_NAME" -n "$NAMESPACE" --timeout=120s

#
# DONE
#
echo ""
echo -e "${GREEN}========================================"
echo "Build and deploy complete!"
echo "========================================${NC}"
echo ""
echo "Image: ${IMAGE_URI}"
echo ""
echo "Verify the agent:"
echo "  kubectl get agents -n ${NAMESPACE}"
echo "  kubectl get pods -n ${NAMESPACE} -l kagent.dev/agent=${AGENT_NAME}"
echo ""
echo "View logs:"
echo "  kubectl logs -n ${NAMESPACE} -l kagent.dev/agent=${AGENT_NAME} -f"
echo ""
echo "Access kagent UI:"
echo "  kubectl port-forward -n ${NAMESPACE} svc/kagent-ui 8080:8080"
echo "  Then open http://localhost:8080"
