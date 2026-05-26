#!/usr/bin/env bash
# Set up IAM and OpenSearch access for memledger v2 composition mode.
#
# Idempotent — safe to re-run. Provisions:
#   - Inline IAM policy on the agent's IRSA role (DynamoDB + OpenSearch)
#   - OpenSearch domain access policy update (allows the IRSA role)
#
# If you've already provisioned via memledger-core's
# examples/aws/provision-opensearch.sh, you can skip this script.
#
# Pre-requisites:
#   - DynamoDB table already created (or set CREATE_TABLE=true)
#   - OpenSearch domain already provisioned and accessible
#   - aws CLI configured with permissions to update IAM + OpenSearch
#
# Usage:
#   bash setup-memledger-composition.sh
#
# Override defaults via env vars:
#   AGENT_ROLE      (default: kagent-on-eks-kagent-bedrock-role)
#   ACCOUNT_ID      (default: derived from `aws sts get-caller-identity`)
#   AWS_REGION      (default: us-west-2)
#   DDB_TABLE       (default: memledger-memory)
#   OS_DOMAIN       (default: memledger-dev)
#   ADMIN_USER      (default: admin)  — kept in OpenSearch access policy
#   POLICY_NAME     (default: memledger-composition-policy)

set -euo pipefail

AGENT_ROLE="${AGENT_ROLE:-kagent-on-eks-kagent-bedrock-role}"
AWS_REGION="${AWS_REGION:-us-west-2}"
DDB_TABLE="${DDB_TABLE:-memledger-memory}"
OS_DOMAIN="${OS_DOMAIN:-memledger-dev}"
ADMIN_USER="${ADMIN_USER:-admin}"
POLICY_NAME="${POLICY_NAME:-memledger-composition-policy}"
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"

echo "==> Configuring memledger v2 composition IAM + OpenSearch access"
echo "    Account:      ${ACCOUNT_ID}"
echo "    Region:       ${AWS_REGION}"
echo "    Agent role:   ${AGENT_ROLE}"
echo "    DDB table:    ${DDB_TABLE}"
echo "    OS domain:    ${OS_DOMAIN}"

# 1. Inline IAM policy on the agent role
echo ""
echo "==> Attaching IAM policy ${POLICY_NAME} to ${AGENT_ROLE}"

POLICY_DOC=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DynamoDBList",
      "Effect": "Allow",
      "Action": ["dynamodb:ListTables"],
      "Resource": "*"
    },
    {
      "Sid": "DynamoDBMemledger",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:BatchGetItem",
        "dynamodb:BatchWriteItem",
        "dynamodb:Query",
        "dynamodb:Scan",
        "dynamodb:DescribeTable",
        "dynamodb:CreateTable",
        "dynamodb:UpdateTimeToLive",
        "dynamodb:DescribeTimeToLive"
      ],
      "Resource": [
        "arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${DDB_TABLE}",
        "arn:aws:dynamodb:${AWS_REGION}:${ACCOUNT_ID}:table/${DDB_TABLE}/index/*"
      ]
    },
    {
      "Sid": "OpenSearchMemledger",
      "Effect": "Allow",
      "Action": [
        "es:ESHttpGet",
        "es:ESHttpPost",
        "es:ESHttpPut",
        "es:ESHttpDelete",
        "es:ESHttpHead"
      ],
      "Resource": "arn:aws:es:${AWS_REGION}:${ACCOUNT_ID}:domain/${OS_DOMAIN}/*"
    }
  ]
}
EOF
)

aws iam put-role-policy \
    --role-name "${AGENT_ROLE}" \
    --policy-name "${POLICY_NAME}" \
    --policy-document "${POLICY_DOC}"

echo "    OK"

# 2. OpenSearch domain access policy
echo ""
echo "==> Updating OpenSearch domain ${OS_DOMAIN} access policy"

OS_ACCESS_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::${ACCOUNT_ID}:user/${ADMIN_USER}",
          "arn:aws:iam::${ACCOUNT_ID}:role/${AGENT_ROLE}"
        ]
      },
      "Action": "es:*",
      "Resource": "arn:aws:es:${AWS_REGION}:${ACCOUNT_ID}:domain/${OS_DOMAIN}/*"
    }
  ]
}
EOF
)

# Compact to single-line — OpenSearch CLI rejects multi-line strings for --access-policies
OS_ACCESS_POLICY_COMPACT=$(echo "${OS_ACCESS_POLICY}" | tr -d '\n' | tr -s ' ')

aws opensearch update-domain-config \
    --domain-name "${OS_DOMAIN}" \
    --region "${AWS_REGION}" \
    --access-policies "${OS_ACCESS_POLICY_COMPACT}" \
    > /dev/null

echo "    OK (domain may take a few minutes to finish reprocessing)"

echo ""
echo "==> Done. Wait for the OpenSearch domain to finish processing:"
echo "    aws opensearch describe-domain --domain-name ${OS_DOMAIN} --region ${AWS_REGION} --query 'DomainStatus.Processing'"
