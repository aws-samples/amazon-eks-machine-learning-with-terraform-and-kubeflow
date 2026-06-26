#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "TEARDOWN: cascading failure"
for d in batch-processor order-api inventory-api; do
  kubectl delete deployment "$d" -n "$DEMO_NAMESPACE" --ignore-not-found
done
echo "✓ Done. Node CPU should recover within 30s."
