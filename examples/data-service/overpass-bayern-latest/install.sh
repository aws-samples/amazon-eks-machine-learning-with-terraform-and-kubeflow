#!/usr/bin/env bash
# =============================================================================
# install.sh — overpass-bayern-latest
# =============================================================================
# Installs the Overpass API instance loaded with the bayern OSM extract
# dated March 9, 2026.
#
# Prerequisites:
#   - kubectl configured to point at the target EKS cluster
#   - helm >= 3.x installed
#   - PVCs pv-fsx and pv-efs bound in kubeflow-user-example-com namespace
#   - Outbound HTTPS access from pods (verified with Istio sidecar disabled)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# To target a different namespace:
#   NAMESPACE=my-namespace ./install.sh
# =============================================================================

set -euo pipefail

RELEASE_NAME="overpass-bayern-latest"
NAMESPACE="${NAMESPACE:-kubeflow-user-example-com}"
VALUES_FILE="./values.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="${SCRIPT_DIR}/../../../charts/data-service/overpass-api"

# ── Preflight checks ──────────────────────────────────────────────────────────

echo "============================================"
echo "Overpass API — Install"
echo "Release    : ${RELEASE_NAME}"
echo "Namespace  : ${NAMESPACE}"
echo "Chart      : ${CHART_DIR}"
echo "============================================"
echo ""

# Verify kubectl context
echo "kubectl context: $(kubectl config current-context)"
echo ""

# Verify PVCs are bound
echo "Checking PVCs in namespace ${NAMESPACE}..."
for PVC in pv-fsx pv-efs; do
  STATUS=$(kubectl get pvc "${PVC}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.phase}' 2>/dev/null || echo "NOT_FOUND")
  if [ "${STATUS}" != "Bound" ]; then
    echo "ERROR: PVC ${PVC} is not Bound in namespace ${NAMESPACE} (status: ${STATUS})"
    echo "Ensure the PVCs are provisioned before running this script."
    exit 1
  fi
  echo "  ✓ ${PVC} is Bound"
done
echo ""

# ── Install ───────────────────────────────────────────────────────────────────

echo "Running helm install..."
helm install "${RELEASE_NAME}" \
    "${CHART_DIR}" \
  --namespace "${NAMESPACE}" \
  --values "${SCRIPT_DIR}/${VALUES_FILE}" \
  --timeout 120m

echo ""
echo "Helm install submitted."
echo ""

# ── Post-install guidance ─────────────────────────────────────────────────────

cat <<EOF
============================================
Next steps
============================================

1. Monitor the download job (downloads ~600MB bayern PBF):

   kubectl logs -n ${NAMESPACE} job/${RELEASE_NAME}-download -f

2. Monitor the init job (initializes Overpass database, ~20-40 min):

   kubectl logs -n ${NAMESPACE} job/${RELEASE_NAME}-init -c overpass-init -f

3. Check overall status:

   kubectl get jobs,pods -n ${NAMESPACE} -l app.kubernetes.io/instance=${RELEASE_NAME}

4. Once the pod is Running and Ready (1/1), smoke-test the instance:

   kubectl run smoke-test --rm -it --restart=Never \\
     --image=curlimages/curl:8.5.0 \\
     --annotations="sidecar.istio.io/inject=false" \\
     -n ${NAMESPACE} \\
     -- curl -s \\
       "http://${RELEASE_NAME}.${NAMESPACE}.svc.cluster.local/api/interpreter" \\
       --data '[out:json][timeout:25];way["highway"](48.74,11.40,48.76,11.44);out count;'

   Expected: JSON response with a "total" count > 0 (roads in Ingolstadt).

5. The endpoint URL for agent configuration:

   http://${RELEASE_NAME}.${NAMESPACE}.svc.cluster.local/api/interpreter

   This is also available as OVERPASS_API_URL in the ConfigMap:
   kubectl get configmap ${RELEASE_NAME}-endpoints -n ${NAMESPACE} -o yaml

EOF
