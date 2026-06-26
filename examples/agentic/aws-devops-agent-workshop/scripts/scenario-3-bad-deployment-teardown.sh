#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "TEARDOWN: payment-service"
kubectl rollout undo deployment/payment-service -n "$DEMO_NAMESPACE" 2>/dev/null || true
sleep 3
kubectl delete deployment payment-service -n "$DEMO_NAMESPACE" --ignore-not-found
echo "Done."
