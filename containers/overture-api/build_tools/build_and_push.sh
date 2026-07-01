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

# Login to the per-account ECR before buildx --push.
aws ecr get-login-password --region "${region}" \
    | docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"

# Build+push explicitly targeting linux/amd64. EKS nodes on standard/CPU
# NodePools run amd64 regardless of the developer machine architecture,
# so building without --platform on an ARM64 workstation (e.g. Apple Silicon)
# produces an unpullable image. `docker buildx --push` is the modern
# invocation that handles this correctly; we ensure a buildx builder exists
# and target linux/amd64 unambiguously.
docker buildx inspect overture-api-builder >/dev/null 2>&1 \
    || docker buildx create --name overture-api-builder --driver docker-container >/dev/null

docker buildx build \
    --builder overture-api-builder \
    --platform linux/amd64 \
    --tag "${fullname}" \
    --push \
    "$DIR/.."

echo "Amazon ECR URI: ${fullname}"
