#!/usr/bin/env bash
# Full cleanup. Safe to run repeatedly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "FULL CLEANUP"

# Belt-and-suspenders: if CoreDNS was scaled to 0 by an older scenario, restore it.
CURRENT=$(kubectl get deploy coredns -n kube-system -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "")
if [[ "$CURRENT" == "0" ]]; then
  echo "CoreDNS at 0 — restoring to 2."
  kubectl scale deployment coredns -n kube-system --replicas=2
fi

for d in leaky-service payment-service web-frontend crashing-app \
         order-api inventory-api batch-processor; do
  kubectl delete deployment "$d" -n "$DEMO_NAMESPACE" --ignore-not-found
done

kubectl delete namespace "$DEMO_NAMESPACE" --ignore-not-found
rm -f /tmp/coredns-replicas.txt

echo "✓ Cleanup complete."
