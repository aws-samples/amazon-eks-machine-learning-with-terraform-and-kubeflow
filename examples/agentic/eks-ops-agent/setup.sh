#!/bin/bash
# EKS Ops Agent - Workshop Setup
#
# This script prepares the environment for the workshop by:
# 1. Adding IAM permissions to the instance role (if running on EC2)
# 2. Verifying AWS credentials and Bedrock access
#
# Usage:
#   ./setup.sh
#
# Run this BEFORE terraform apply if running from an EC2 instance.

set -e

echo "========================================"
echo "EKS Ops Agent - Workshop Setup"
echo "========================================"

# Get AWS region from environment, CLI config, or default
AWS_REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null)}"
AWS_REGION="${AWS_REGION:-us-west-2}"

# Get AWS account info
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "AWS Account: ${AWS_ACCOUNT_ID}"
echo "AWS Region:  ${AWS_REGION}"

# Detect if running on EC2 (works on Linux, safely fails on Mac)
IS_EC2=false
if curl -s --connect-timeout 2 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" >/dev/null 2>&1; then
    IS_EC2=true
fi

if [ "$IS_EC2" = true ]; then
    echo "Detected: Running on EC2 instance"

    # Get metadata token
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

    # Get instance profile ARN (using simple text extraction for portability)
    IAM_INFO=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/info)
    INSTANCE_ROLE_ARN=$(echo "$IAM_INFO" | grep "InstanceProfileArn" | sed 's/.*"InstanceProfileArn" *: *"\([^"]*\)".*/\1/')

    if [ -n "$INSTANCE_ROLE_ARN" ]; then
        # Extract instance profile name from ARN
        INSTANCE_PROFILE_NAME=$(echo "$INSTANCE_ROLE_ARN" | awk -F'/' '{print $NF}')

        # Get the role name from the instance profile
        # Try the API first, fall back to deriving from profile name (replace InstanceProfile with InstanceRole)
        INSTANCE_ROLE_NAME=$(aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" --query 'InstanceProfile.Roles[0].RoleName' --output text 2>/dev/null || true)

        # If API call failed or returned empty, try to derive role name from profile name
        if [ -z "$INSTANCE_ROLE_NAME" ] || [ "$INSTANCE_ROLE_NAME" = "None" ]; then
            # Common pattern: replace "InstanceProfile" with "InstanceRole" in the name
            INSTANCE_ROLE_NAME=$(echo "$INSTANCE_PROFILE_NAME" | sed 's/InstanceProfile/InstanceRole/')
            echo "Derived Instance Role: ${INSTANCE_ROLE_NAME}"
        fi

        if [ -n "$INSTANCE_ROLE_NAME" ] && [ "$INSTANCE_ROLE_NAME" != "None" ]; then
            echo "Instance Role: ${INSTANCE_ROLE_NAME}"

            # Check if policy already exists
            POLICY_EXISTS=$(aws iam list-role-policies --role-name "$INSTANCE_ROLE_NAME" --query "PolicyNames[?contains(@, 'kagent-workshop-permissions')]" --output text 2>/dev/null)

            if [ -z "$POLICY_EXISTS" ]; then
                echo "Adding IAM permissions for kagent..."

                aws iam put-role-policy \
                    --role-name "$INSTANCE_ROLE_NAME" \
                    --policy-name "kagent-workshop-permissions" \
                    --policy-document "{
                        \"Version\": \"2012-10-17\",
                        \"Statement\": [
                            {
                                \"Sid\": \"KagentIAMPermissions\",
                                \"Effect\": \"Allow\",
                                \"Action\": [
                                    \"iam:CreateRole\",
                                    \"iam:DeleteRole\",
                                    \"iam:GetRole\",
                                    \"iam:TagRole\",
                                    \"iam:UntagRole\",
                                    \"iam:UpdateRole\",
                                    \"iam:UpdateRoleDescription\",
                                    \"iam:CreatePolicy\",
                                    \"iam:DeletePolicy\",
                                    \"iam:GetPolicy\",
                                    \"iam:TagPolicy\",
                                    \"iam:UntagPolicy\",
                                    \"iam:CreatePolicyVersion\",
                                    \"iam:DeletePolicyVersion\",
                                    \"iam:GetPolicyVersion\",
                                    \"iam:ListPolicyVersions\",
                                    \"iam:AttachRolePolicy\",
                                    \"iam:DetachRolePolicy\",
                                    \"iam:ListAttachedRolePolicies\",
                                    \"iam:ListRolePolicies\"
                                ],
                                \"Resource\": [
                                    \"arn:aws:iam::${AWS_ACCOUNT_ID}:role/*-kagent-*\",
                                    \"arn:aws:iam::${AWS_ACCOUNT_ID}:policy/*-kagent-*\"
                                ]
                            }
                        ]
                    }"

                echo "IAM permissions added successfully"
            else
                echo "IAM permissions already configured"
            fi
        else
            echo "Warning: Could not determine instance role name"
        fi
    else
        echo "Warning: Could not determine instance profile ARN"
    fi
else
    echo "Detected: Not running on EC2 (local machine)"
    echo "Assuming AWS credentials have sufficient IAM permissions"
fi

# Verify Bedrock access
echo ""
echo "Verifying Bedrock model access in ${AWS_REGION}..."
BEDROCK_MODELS=$(aws bedrock list-foundation-models --region "$AWS_REGION" --query "modelSummaries[?contains(modelId, 'claude')].modelId" --output text 2>/dev/null | head -3)

if [ -n "$BEDROCK_MODELS" ]; then
    echo "Bedrock Claude models available:"
    echo "$BEDROCK_MODELS" | tr '\t' '\n' | head -3 | sed 's/^/  - /'
else
    echo "Warning: Could not list Bedrock models. Ensure model access is enabled in the AWS console."
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. cd eks-cluster/terraform/aws-eks-cluster-and-nodegroup"
echo "  2. Add to terraform.tfvars:"
echo "       kagent_enabled = true"
echo "       kagent_enable_bedrock_access = true"
echo "       kagent_enable_ui = true"
echo "  3. terraform apply"
echo "========================================"
