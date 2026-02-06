#!/bin/bash
# Build and push EKS Ops Agent container to ECR
#
# Usage:
#   ./build.sh              # Build and push to ECR (default)
#   ./build.sh --local      # Build locally only, no push
#
# Environment variables:
#   AWS_REGION    - AWS region (default: us-west-2)
#   VERSION       - Image version tag (default: 0.1.0)

set -e

# Configuration
IMAGE_NAME="eks-ops-agent"
VERSION="${VERSION:-0.1.0}"
AWS_REGION="${AWS_REGION:-us-west-2}"

# Parse arguments
LOCAL_ONLY=false

if [[ "$1" == "--local" ]]; then
    LOCAL_ONLY=true
fi

echo "========================================"
echo "EKS Ops Agent - Container Build"
echo "========================================"
echo "Image:   ${IMAGE_NAME}"
echo "Version: ${VERSION}"
echo "Mode:    $([ "$LOCAL_ONLY" = true ] && echo "Local only" || echo "Build + Push to ECR")"
echo "========================================"

# Build the image
echo "Building Docker image..."
docker build \
    --platform linux/amd64 \
    -t "${IMAGE_NAME}:${VERSION}" \
    -t "${IMAGE_NAME}:latest" \
    .

echo "Local build complete: ${IMAGE_NAME}:${VERSION}"

# Exit if local only
if [ "$LOCAL_ONLY" = true ]; then
    echo "========================================"
    echo "Local build complete!"
    echo "Image: ${IMAGE_NAME}:${VERSION}"
    echo "========================================"
    exit 0
fi

# Push to ECR
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO="${ECR_REGISTRY}/${IMAGE_NAME}"

# Create ECR repository if it doesn't exist
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

echo "========================================"
echo "Build and push complete!"
echo ""
echo "IMAGE_URI=${ECR_REPO}:${VERSION}"
echo "========================================"
