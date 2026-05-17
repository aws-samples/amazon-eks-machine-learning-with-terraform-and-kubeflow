#!/bin/bash
# EKS Ops Agent - Build, Push, and Deploy
#
# Builds the Docker image, pushes it to ECR, and deploys via Helm.
#
# Prerequisites:
#   1. Docker installed and running
#   2. AWS CLI configured with ECR access
#   3. Helm installed
#   4. TF_DIR and REPO_DIR environment variables set (from notebook)
#
# Environment variable overrides (for module progression):
#   ENABLE_MCP_TOOLS=true   Enable EKS MCP Server tools (Module 2)
#   ENABLE_MEMORY=true      Enable Redis memory (Module 3)
#
# Usage:
#   ./build-and-deploy.sh                                    # Module 1: barebone
#   ENABLE_MCP_TOOLS=true ./build-and-deploy.sh              # Module 2: MCP tools
#   ENABLE_MCP_TOOLS=true ENABLE_MEMORY=true ./build-and-deploy.sh  # Module 3: memory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
IMAGE_NAME="eks-ops-agent"
VERSION="${VERSION:-0.1.1}"
AWS_REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || echo us-west-2)}"
ENABLE_MCP_TOOLS="${ENABLE_MCP_TOOLS:-false}"
ENABLE_MEMORY="${ENABLE_MEMORY:-false}"
# v2 (deferred): composition mode (DynamoDB primary + OpenSearch search index).
# AWS SDK integration lands in memledger v2. v1 supports pgvector only — leave
# ENABLE_COMPOSITION unset/false. See ./v2-preview/ for the reserved config.
ENABLE_COMPOSITION="${ENABLE_COMPOSITION:-false}"
OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-}"
MEMLEDGER_DDB_TABLE="${MEMLEDGER_DDB_TABLE:-memledger-memory}"

# Validate required env vars
if [ -z "$TF_DIR" ]; then
    echo "ERROR: TF_DIR environment variable is not set" >&2
    exit 1
fi
if [ -z "$REPO_DIR" ]; then
    echo "ERROR: REPO_DIR environment variable is not set" >&2
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "EKS Ops Agent - Build, Push, and Deploy"
echo "========================================"
echo "Image:           ${IMAGE_NAME}"
echo "Version:         ${VERSION}"
echo "Region:          ${AWS_REGION}"
echo "ENABLE_MCP_TOOLS:   ${ENABLE_MCP_TOOLS}"
echo "ENABLE_MEMORY:      ${ENABLE_MEMORY}"
echo "ENABLE_COMPOSITION: ${ENABLE_COMPOSITION}"
if [ "$ENABLE_COMPOSITION" = "true" ]; then
    echo "OPENSEARCH_ENDPOINT: ${OPENSEARCH_ENDPOINT}"
    echo "MEMLEDGER_DDB_TABLE:   ${MEMLEDGER_DDB_TABLE}"
fi

# Build the image
echo ""
echo -e "${YELLOW}Building Docker image...${NC}"
docker build \
    --no-cache \
    --platform linux/amd64 \
    -t "${IMAGE_NAME}:${VERSION}" \
    -t "${IMAGE_NAME}:latest" \
    "${SCRIPT_DIR}"

echo "Local build complete: ${IMAGE_NAME}:${VERSION}"

# Push to ECR
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO="${ECR_REGISTRY}/${IMAGE_NAME}"

echo ""
echo "Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names "${IMAGE_NAME}" --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository \
        --repository-name "${IMAGE_NAME}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true

echo "Logging into ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "Pushing to ECR..."
docker tag "${IMAGE_NAME}:${VERSION}" "${ECR_REPO}:${VERSION}"
docker tag "${IMAGE_NAME}:latest" "${ECR_REPO}:latest"
docker push "${ECR_REPO}:${VERSION}"
docker push "${ECR_REPO}:latest"

echo ""
echo -e "${GREEN}Push complete: ${ECR_REPO}:${VERSION}${NC}"

# Get Bedrock role ARN from Terraform output
echo ""
echo "Reading Bedrock role ARN from Terraform output..."
BEDROCK_ROLE_ARN=$(cd "$TF_DIR" && terraform output -raw kagent_bedrock_iam_role_arn)
echo "Bedrock Role ARN: ${BEDROCK_ROLE_ARN}"

# Deploy with Helm
echo ""
echo -e "${YELLOW}Deploying with Helm...${NC}"
cd "$REPO_DIR"

HELM_ARGS=""
if [ "$ENABLE_MEMORY" = "true" ]; then
    if [ -n "$MEMLEDGER_PG_DSN" ]; then
        echo "Using MEMLEDGER_PG_DSN from environment: ${MEMLEDGER_PG_DSN##*@}"
    else
        MEMLEDGER_PG_DSN=$(kubectl get secret kagent-db-credentials -n kagent -o jsonpath='{.data.connection_string}' 2>/dev/null | base64 -d 2>/dev/null || echo "")
        if [ -z "$MEMLEDGER_PG_DSN" ]; then
            echo -e "${YELLOW}Warning: kagent-db-credentials secret not found. Memory will not work without MEMLEDGER_PG_DSN.${NC}"
        else
            echo "memledger PG URL (from kagent-db-credentials): ${MEMLEDGER_PG_DSN##*@}"
        fi
    fi
fi

COMPOSITION_HELM_ARGS=""
if [ "$ENABLE_COMPOSITION" = "true" ]; then
    # v2 path — composition.yaml is not baked into the v1 image. Enabling this
    # in v1 will deploy with the env wired up but no config file present.
    if [ -z "$OPENSEARCH_ENDPOINT" ]; then
        echo -e "${YELLOW}ERROR: ENABLE_COMPOSITION=true but OPENSEARCH_ENDPOINT is not set.${NC}"
        exit 1
    fi
    COMPOSITION_HELM_ARGS="\
        --set env[5].name=MEMLEDGER_CONFIG_PATH --set env[5].value=/app/memledger-composition.yaml \
        --set env[6].name=OPENSEARCH_ENDPOINT --set env[6].value=${OPENSEARCH_ENDPOINT} \
        --set env[7].name=MEMLEDGER_DDB_TABLE --set env[7].value=${MEMLEDGER_DDB_TABLE}"
fi

helm upgrade --install eks-ops-agent -n kagent \
  charts/machine-learning/agentic/kagent-agent \
  -f examples/agentic/eks-ops-agent/eks-ops-agent.yaml \
  --set image.repository="${ECR_REPO}" \
  --set image.tag="${VERSION}" \
  --set "env[0].name=AWS_REGION" --set "env[0].value=${AWS_REGION}" \
  --set "env[1].name=BEDROCK_MODEL_ID" --set "env[1].value=us.anthropic.claude-sonnet-4-20250514-v1:0" \
  --set "env[2].name=ENABLE_MCP_TOOLS" --set "env[2].value=${ENABLE_MCP_TOOLS}" \
  --set "env[3].name=ENABLE_MEMORY" --set "env[3].value=${ENABLE_MEMORY}" \
  --set "env[4].name=MEMLEDGER_PG_DSN" --set "env[4].value=${MEMLEDGER_PG_DSN}" \
  --set "env[5].name=OTEL_EXPORTER_OTLP_LOGS_HEADERS" --set "env[5].value=x-aws-log-group=memledger-agent-spans,x-aws-log-stream=otel-spans,x-aws-metric-namespace=memledger" \
  --set "env[6].name=OTEL_EXPORTER_OTLP_LOGS_ENDPOINT" --set "env[6].value=https://logs.${AWS_REGION}.amazonaws.com/v1/logs" \
  --set "env[7].name=OTEL_RESOURCE_ATTRIBUTES" --set "env[7].value=service.name=eks-ops-agent" \
  $COMPOSITION_HELM_ARGS \
  $HELM_ARGS

# Wait for kagent controller to create the ServiceAccount
echo ""
echo "Waiting for ServiceAccount to be created..."
for i in $(seq 1 30); do
    if kubectl get serviceaccount "${IMAGE_NAME}" -n kagent >/dev/null 2>&1; then
        echo "ServiceAccount created."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo -e "${YELLOW}Warning: ServiceAccount not created after 30 seconds${NC}"
    fi
    sleep 1
done

# Annotate ServiceAccount with IRSA role for Bedrock access
echo ""
echo "Annotating ServiceAccount for IRSA..."
kubectl annotate serviceaccount "${IMAGE_NAME}" -n kagent \
    "eks.amazonaws.com/role-arn=${BEDROCK_ROLE_ARN}" \
    --overwrite

# Restart deployment to pick up IRSA credentials
echo ""
echo "Restarting deployment to pick up IRSA credentials..."
kubectl rollout restart deployment "${IMAGE_NAME}" -n kagent
kubectl rollout status deployment "${IMAGE_NAME}" -n kagent --timeout=120s

echo ""
echo -e "${GREEN}========================================"
echo "Deploy complete!"
echo "========================================${NC}"
