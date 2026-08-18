#!/usr/bin/env bash
# =============================================================================
# install.sh — overture-transportation-ingolstadt
# =============================================================================
# Installs the Overture Maps API instance filtered to the Ingolstadt urban
# core. Automatically resolves the latest Overture release from the STAC
# catalog at deploy time, falling back to the pinned tag in values.yaml.
#
# Steps:
#   1) Pre-flight: verify kubectl context + PVC bindings.
#   2) Build the overture-api container image and push to the per-account ECR
#      (via containers/overture-api/build_tools/build_and_push.sh).
#   3) helm install the chart, overriding image.repository to the ECR URI
#      resolved in step 2.
#
# Prerequisites:
#   - kubectl configured to point at the target EKS cluster
#   - helm >= 3.x installed
#   - PVCs pv-fsx and pv-efs bound in kubeflow-user-example-com namespace
#   - Outbound HTTPS access from pods (Istio sidecar disabled)
#   - aws CLI configured with credentials that can push to ECR
#   - docker installed
#
# Usage:
#   chmod +x install.sh
#   ./install.sh                       # uses AWS_REGION from env or defaults to us-east-1
#
# To target a different namespace:
#   NAMESPACE=my-namespace ./install.sh
# =============================================================================

set -euo pipefail

RELEASE_NAME="overture-transportation-ingolstadt"
NAMESPACE="${NAMESPACE:-kubeflow-user-example-com}"
REGION="${AWS_REGION:-$(kubectl config current-context | awk -F: '{print $4}')}"
VALUES_FILE="./values.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="${SCRIPT_DIR}/../../../charts/data-service/overture-api"
CONTAINER_BUILD_DIR="${SCRIPT_DIR}/../../../containers/overture-api/build_tools"

# ── Preflight checks ──────────────────────────────────────────────────────────

echo "============================================"
echo "Overture API — Install"
echo "Release    : ${RELEASE_NAME}"
echo "Namespace  : ${NAMESPACE}"
echo "Region     : ${REGION}"
echo "Chart      : ${CHART_DIR}"
echo "============================================"
echo ""

echo "kubectl context: $(kubectl config current-context)"
echo ""

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

# ── Build + push container image ──────────────────────────────────────────────

if [ ! -x "${CONTAINER_BUILD_DIR}/build_and_push.sh" ]; then
    echo "ERROR: ${CONTAINER_BUILD_DIR}/build_and_push.sh not found or not executable"
    exit 1
fi

echo "Building and pushing overture-api container image to ECR (region: ${REGION})..."
BUILD_OUTPUT=$("${CONTAINER_BUILD_DIR}/build_and_push.sh" "${REGION}")
echo "${BUILD_OUTPUT}"

# Parse the ECR URI from the build_and_push.sh stdout (last "Amazon ECR URI: ..." line).
ECR_URI=$(echo "${BUILD_OUTPUT}" | grep '^Amazon ECR URI: ' | tail -1 | sed -e 's|^Amazon ECR URI: ||')
if [ -z "${ECR_URI}" ]; then
    echo "ERROR: failed to parse ECR URI from build_and_push.sh output"
    exit 1
fi
# Strip tag — chart values use repository and tag separately.
IMAGE_REPOSITORY="${ECR_URI%:*}"
IMAGE_TAG="${ECR_URI##*:}"

echo "Resolved image: ${IMAGE_REPOSITORY}:${IMAGE_TAG}"
echo ""

# ── Resolve latest Overture release ────────────────────────────────────────────
# Overture Maps applies a 60-day S3 lifecycle expiration to release data, so
# any pinned tag in values.yaml may expire. Query the STAC catalog for the
# latest available release and override via --set. Falls back to the pinned
# value in values.yaml if the lookup fails.

STAC_URL="https://stac.overturemaps.org/catalog.json"
echo "Resolving latest Overture release from STAC catalog..."
OVERTURE_RELEASE=$(curl -sf "${STAC_URL}" | jq -r '.latest // empty' 2>/dev/null || true)

OVERTURE_RELEASE_SET=()
if [ -n "${OVERTURE_RELEASE}" ]; then
  echo "Using latest Overture release: ${OVERTURE_RELEASE}"
  OVERTURE_RELEASE_SET=(--set "overture.release=${OVERTURE_RELEASE}")
else
  echo "WARNING: Could not resolve latest Overture release from STAC catalog."
  echo "         Falling back to pinned release in values.yaml."
fi
echo ""

# ── Install ───────────────────────────────────────────────────────────────────

echo "Running helm install..."
helm install "${RELEASE_NAME}" \
    "${CHART_DIR}" \
  --namespace "${NAMESPACE}" \
  --values "${SCRIPT_DIR}/${VALUES_FILE}" \
  --set image.repository="${IMAGE_REPOSITORY}" \
  --set image.tag="${IMAGE_TAG}" \
  "${OVERTURE_RELEASE_SET[@]}" \
  --timeout 15m

echo ""
echo "Helm install submitted."
echo ""

# ── Post-install guidance ─────────────────────────────────────────────────────

cat <<EOF
============================================
Next steps
============================================

1. Monitor the init job (DuckDB reads Overture from S3, filters to bbox,
   writes filtered parquet to FSx — typically ~30-90s):

   kubectl logs -n ${NAMESPACE} job/${RELEASE_NAME}-init -c filter -f

2. Check overall status:

   kubectl get jobs,pods -n ${NAMESPACE} -l app.kubernetes.io/instance=${RELEASE_NAME}

3. Once the pod is Running and Ready (1/1), smoke-test the instance:

   kubectl run smoke-test --rm -it --restart=Never \\
     --image=curlimages/curl:8.5.0 \\
     --annotations="sidecar.istio.io/inject=false" \\
     -n ${NAMESPACE} \\
     -- curl -s \\
       "http://${RELEASE_NAME}.${NAMESPACE}.svc.cluster.local/api/status"

   Expected: JSON response with "ready": true and "segment_count" > 0.

4. The endpoint URL for agent configuration:

   http://${RELEASE_NAME}.${NAMESPACE}.svc.cluster.local

   Also available as OVERTURE_API_URL in the ConfigMap:
   kubectl get configmap ${RELEASE_NAME}-endpoints -n ${NAMESPACE} -o yaml

EOF
