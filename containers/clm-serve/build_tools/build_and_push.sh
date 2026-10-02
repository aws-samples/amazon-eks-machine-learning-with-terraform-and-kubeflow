#!/usr/bin/env bash

# Builds the clm-serve Docker image and pushes it to the per-account ECR in the
# given region.
#
# On success, prints the full ECR URI on the last line as:
#   Amazon ECR URI: <account>.dkr.ecr.<region>.amazonaws.com/<image>:<tag>
#
# The URI is not written into any values file: pass it at install time with
# --set image.name=<uri>, as examples/inference/system-one/clm-8b/serve.ipynb does.

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

# EKS nodes are amd64; target it explicitly so a build on an ARM64 workstation
# (e.g. Apple Silicon) still produces a pullable image.
docker buildx inspect clm-serve-builder >/dev/null 2>&1 \
    || docker buildx create --name clm-serve-builder --driver docker-container >/dev/null

docker buildx build \
    --builder clm-serve-builder \
    --platform linux/amd64 \
    --tag "${fullname}" \
    --push \
    "$DIR/.."

echo "Amazon ECR URI: ${fullname}"
