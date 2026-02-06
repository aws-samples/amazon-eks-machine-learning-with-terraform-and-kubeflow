#!/bin/bash
# Create Kubernetes secret for Bedrock credentials
#
# Usage:
#   ./create-bedrock-secret.sh
#
# Prerequisites:
#   - AWS CLI configured with valid credentials
#   - kubectl configured to access your EKS cluster

set -e

NAMESPACE="${NAMESPACE:-kagent}"
SECRET_NAME="kagent-bedrock"

echo "Creating Bedrock credentials secret in namespace: ${NAMESPACE}"

# Get current AWS credentials
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-$(aws configure get aws_access_key_id)}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-$(aws configure get aws_secret_access_key)}"
AWS_SESSION_TOKEN="${AWS_SESSION_TOKEN:-$(aws configure get aws_session_token)}"

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Error: AWS credentials not found. Please configure AWS CLI or set environment variables."
    exit 1
fi

# Create or update the secret
kubectl create secret generic "${SECRET_NAME}" \
    --namespace "${NAMESPACE}" \
    --from-literal=AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    --from-literal=AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    --from-literal=AWS_SESSION_TOKEN="${AWS_SESSION_TOKEN:-}" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "Secret '${SECRET_NAME}' created/updated in namespace '${NAMESPACE}'"
