#!/bin/bash
set -e

STACK_NAME="eks-${RANDOM}"
TEMPLATE_URL="https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow/raw/refs/heads/master/ml-ops-desktop-basic.yaml"
TEMPLATE_FILE="/tmp/ml-ops-desktop-basic.yaml"

usage() {
  echo "Usage: $0 [--instance-type <type>] [--cidr <x.x.x.x/32>] [--stack-name <name>] [--region <region>]"
  echo ""
  echo "Options:"
  echo "  --cidr           Desktop access CIDR (default: auto-detected public IP using http://checkip.amazonaws.com/)"
  echo "  --stack-name     CloudFormation stack name (default: ${STACK_NAME})"
  echo "  --region         AWS region (default: auto detected using aws configure get region)"
  echo ""
  echo "Examples:"
  echo "  $0"
  echo "  $0 --cidr 203.0.113.10/32"
  exit 1
}

INSTANCE_TYPE="${DEFAULT_INSTANCE_TYPE}"
CIDR=""
REGION=$(aws configure get region)

while [[ $# -gt 0 ]]; do
  case $1 in
    --cidr) CIDR="$2"; shift 2 ;;
    --stack-name) STACK_NAME="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --help|-h) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# Check AWS CLI is installed and configured
if ! command -v aws &> /dev/null; then
  echo "Error: AWS CLI is not installed. Please install it first."
  echo "  https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
  exit 1
fi

if ! aws sts get-caller-identity &> /dev/null; then
  echo "Error: AWS CLI is not configured or credentials are invalid."
  echo "  Run 'aws configure' to set up your credentials."
  exit 1
fi

# Auto-detect public IP if CIDR not provided
if [ -z "${CIDR}" ]; then
  echo "Detecting your public IP address..."
  PUBLIC_IP=$(curl -s http://checkip.amazonaws.com/ | tr -d '[:space:]')
  if [ -z "${PUBLIC_IP}" ]; then
    echo "Error: Could not detect public IP. Please provide --cidr manually."
    exit 1
  fi
  CIDR="${PUBLIC_IP}/32"
  echo "Detected public IP: ${PUBLIC_IP}"
fi

echo ""
echo "=== Workshop Setup ==="
echo "  Stack name:    ${STACK_NAME}"
echo "  Region:        ${REGION}"
echo "  Access CIDR:   ${CIDR}"
echo ""

# Download the CloudFormation template
echo "Downloading CloudFormation template..."
curl -sL "${TEMPLATE_URL}" -o "${TEMPLATE_FILE}"
if [ ! -f "${TEMPLATE_FILE}" ]; then
  echo "Error: Failed to download CloudFormation template."
  exit 1
fi
echo "Template downloaded successfully."

# Create the CloudFormation stack
echo ""
echo "Creating CloudFormation stack '${STACK_NAME}'..."
aws cloudformation create-stack \
  --stack-name "${STACK_NAME}" \
  --template-body "file://${TEMPLATE_FILE}" \
  --parameters \
    ParameterKey=DesktopAccessCIDR,ParameterValue="${CIDR}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "${REGION}" > /dev/null

echo "Stack creation initiated."
echo ""
echo "Waiting for stack to complete (this typically takes 45-60 minutes)..."
echo "You can also monitor progress in the AWS CloudFormation console."
echo ""

# Monitor stack status
START_TIME=$(date +%s)
while true; do
  STATUS=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query 'Stacks[0].StackStatus' \
    --output text 2>/dev/null)

  TIMESTAMP=$(date '+%H:%M:%S')
  ELAPSED_SECONDS=$(( $(date +%s) - START_TIME ))
  ELAPSED_MINUTES=$(( ELAPSED_SECONDS / 60 ))

  case "${STATUS}" in
    CREATE_COMPLETE)
      echo "[${TIMESTAMP}] (${ELAPSED_MINUTES}m elapsed) Stack status: ${STATUS}"
      echo ""
      echo "=== Stack created successfully ==="
      break
      ;;
    CREATE_IN_PROGRESS)
      echo "[${TIMESTAMP}] (${ELAPSED_MINUTES}m elapsed) Stack status: ${STATUS} ..."
      sleep 60
      ;;
    CREATE_FAILED|ROLLBACK_IN_PROGRESS|ROLLBACK_COMPLETE|ROLLBACK_FAILED)
      echo "[${TIMESTAMP}] (${ELAPSED_MINUTES}m elapsed) Stack status: ${STATUS}"
      echo ""
      echo "Error: Stack creation failed. Check the CloudFormation console for details:"
      echo "  https://${REGION}.console.aws.amazon.com/cloudformation/home?region=${REGION}#/stacks"
      exit 1
      ;;
    *)
      echo "[${TIMESTAMP}] (${ELAPSED_MINUTES}m elapsed) Stack status: ${STATUS}"
      sleep 30
      ;;
  esac
done

# Get instance info
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=${STACK_NAME}-ml-ops-desktop" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "${REGION}" 2>/dev/null)

PUBLIC_IP_INSTANCE=$(aws ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region "${REGION}" 2>/dev/null)

echo ""
echo "=== Next Steps ==="
echo ""
echo "1. Connect via SSM Session Manager:"
echo "   aws ssm start-session --target ${INSTANCE_ID} --region ${REGION}"
echo ""
echo "2. Set password for ubuntu user:"
echo "   sudo passwd ubuntu"
echo ""
echo "3. Connect via DCV client:"
echo "   Download: https://docs.aws.amazon.com/dcv/latest/userguide/client.html"
echo "   Connect to: https://${PUBLIC_IP_INSTANCE}:8443"
echo "   Login as: ubuntu"
echo ""
echo "=== Setup Complete ==="

# Cleanup downloaded template
rm -f "${TEMPLATE_FILE}"
