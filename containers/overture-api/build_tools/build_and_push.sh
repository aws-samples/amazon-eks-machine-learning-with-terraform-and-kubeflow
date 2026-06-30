#!/usr/bin/env bash

# Builds the overture-api Docker image and pushes it to the per-account ECR
# in the given region. Mirrors the v1 build pattern (e.g., containers/ray-pytorch).
#
# On success, prints the full ECR URI on the last line as:
#   Amazon ECR URI: <account>.dkr.ecr.<region>.amazonaws.com/<image>:<tag>
#
# Consumers (notably the chart's install.sh) parse that line to set the
# image override.

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/set_env.sh

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <aws-region>" >&2
    exit 1
fi
region=$1

image=$IMAGE_NAME
tag=$IMAGE_TAG

account=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$account" ]; then
    echo "ERROR: could not resolve AWS account from current credentials" >&2
    exit 255
fi

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${tag}"

# Create the ECR repository if it doesn't exist.
aws ecr describe-repositories --region "${region}" --repository-names "${image}" > /dev/null 2>&1 \
    || aws ecr create-repository --region "${region}" --repository-name "${image}" > /dev/null

# Login to ECR public for the python base image pull.
aws ecr-public get-login-password --region us-east-1 \
    | docker login --username AWS --password-stdin public.ecr.aws

# Build from the container directory (parent of build_tools/).
docker build -t "${image}:${tag}" "$DIR/.."
docker tag "${image}:${tag}" "${fullname}"

# Login to the per-account ECR and push.
aws ecr get-login-password --region "${region}" \
    | docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"

docker push "${fullname}"

echo "Amazon ECR URI: ${fullname}"
