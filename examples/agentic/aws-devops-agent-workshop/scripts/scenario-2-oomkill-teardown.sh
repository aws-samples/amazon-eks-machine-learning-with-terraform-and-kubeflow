#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "TEARDOWN: leaky-service"
kubectl delete deployment leaky-service -n "$DEMO_NAMESPACE" --ignore-not-found
echo "Done."
