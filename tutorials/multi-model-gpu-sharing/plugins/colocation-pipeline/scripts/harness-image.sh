#!/usr/bin/env bash
# The harness image identity, derived from the build context rather than chosen by hand.
#
#   harness-image.sh --tag-only   print just the tag. Offline: no aws, no kubectl, no docker.
#   harness-image.sh              print <repo>:<tag>. Needs aws and the cluster context.
#   harness-image.sh --ensure     print <repo>:<tag>, building, self-testing and pushing first
#                                 if that tag is not already in ECR.
#
# Only the image reference goes to stdout, so callers can use it as $(...). Everything else,
# including every line docker prints, goes to stderr.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
v2=$(cd "$here/../../.." && pwd)
context=$v2/containers/colocation-harness
mode=${1:-repo}

cd "$context"

# Verbatim from the recipe the skill has always used. A different set of files, or a different
# order, is a different tag, and every tag already in ECR was computed by this expression.
# __pycache__ is excluded because there is no .dockerignore: running the harness locally leaves
# .pyc files in the build context, and hashing them would move the tag with no source change.
tag=src-$(find Dockerfile requirements.txt harness tests -type f -name '*.py' \
            -o -type f -name 'Dockerfile' -o -type f -name 'requirements.txt' \
          | grep -v __pycache__ | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-12)

if [ "$mode" = "--tag-only" ]; then
  echo "$tag"
  exit 0
fi

account=$(aws sts get-caller-identity --query Account --output text)
# The region comes out of the EKS context ARN (arn:aws:eks:<region>:<account>:cluster/<name>),
# never written down. A written-down region goes stale silently: the push succeeds against the
# wrong region's ECR and it surfaces much later as a pod that cannot pull.
region=$(kubectl config current-context | cut -d: -f4)
if [ -z "$region" ]; then
  echo "harness-image: no region in the kubectl context; is a cluster context selected?" >&2
  exit 1
fi
repo=$account.dkr.ecr.$region.amazonaws.com/colocation-harness

if [ "$mode" != "--ensure" ]; then
  echo "$repo:$tag"
  exit 0
fi

if aws ecr describe-images --repository-name colocation-harness --region "$region" \
     --image-ids imageTag="$tag" >/dev/null 2>&1; then
  echo "harness-image: $tag already in ECR, nothing to build" >&2
  echo "$repo:$tag"
  exit 0
fi

# Build, verify, then push, in that order. The verification is offline, and doing it before the
# push is the whole point: a harness image that measures wrongly is worse than no image, and a
# measuring defect in this repo has voided complete search results before.
{
  aws ecr create-repository --repository-name colocation-harness --region "$region" || true
  aws ecr get-login-password --region "$region" \
    | docker login --username AWS --password-stdin "$account.dkr.ecr.$region.amazonaws.com"
  docker build -t "colocation-harness:$tag" .
  docker run --rm --entrypoint python "colocation-harness:$tag" -m tests.selftest
  docker tag "colocation-harness:$tag" "$repo:$tag"
  docker push "$repo:$tag"
} >&2

echo "$repo:$tag"
