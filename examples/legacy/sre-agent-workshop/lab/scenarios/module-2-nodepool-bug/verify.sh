#!/usr/bin/env bash
# Confirm the Module 2 fault state has surfaced before running the agent.
#
# Checks:
#   1. Overpass deployment exists in the expected namespace
#   2. Overpass pod is Pending
#   3. NO NodeClaim exists for the data-services NodePool (proof that
#      Karpenter never even attempted to launch — different from M1
#      where Karpenter tried and EC2 refused)
#   4. NodePool is missing the 'workload' requirement (the actual bug)
#
# Exits non-zero if any check fails.

set -uo pipefail

NAMESPACE="${SRE_M2_NAMESPACE:-data-services}"
NODEPOOL="${SRE_M2_NODEPOOL:-cpu-compute}"

green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }

pass_count=0
fail_count=0

echo "──────────────────────────────────────────────────────────────"
echo "  Module 2 — fault state verification"
echo "──────────────────────────────────────────────────────────────"

# 1. Deployment exists
if kubectl get deploy -n "${NAMESPACE}" overpass-api >/dev/null 2>&1; then
    green "  ✅ overpass-api deployment applied"
    pass_count=$((pass_count + 1))
else
    red "  ❌ overpass-api deployment missing"
    fail_count=$((fail_count + 1))
fi

# 2. Pod is Pending
sleep 3
pending_count=$(kubectl get pods -n "${NAMESPACE}" -l app=overpass-api --field-selector=status.phase=Pending 2>/dev/null | grep -c "^overpass-api" || true)
if [[ "${pending_count}" -ge 1 ]]; then
    green "  ✅ overpass-api pod in Pending state"
    pass_count=$((pass_count + 1))
else
    yellow "  ⏳ pod not yet Pending — waiting 10s and retrying…"
    sleep 10
    pending_count=$(kubectl get pods -n "${NAMESPACE}" -l app=overpass-api --field-selector=status.phase=Pending 2>/dev/null | grep -c "^overpass-api" || true)
    if [[ "${pending_count}" -ge 1 ]]; then
        green "  ✅ overpass-api pod in Pending state"
        pass_count=$((pass_count + 1))
    else
        red "  ❌ overpass-api pod is not Pending"
        fail_count=$((fail_count + 1))
    fi
fi

# 3. NO NodeClaim was created for this workload
# The critical difference from M1: Karpenter never attempted provisioning
# because no NodePool matched. We can't tell "for THIS workload" perfectly,
# but we can check that the count of Launched=False NodeClaims did not go up.
launched_false=$(kubectl get nodeclaims.karpenter.sh -o json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
n = 0
for nc in data.get('items', []):
    for c in nc.get('status', {}).get('conditions', []):
        if c.get('type') == 'Launched' and c.get('status') == 'False':
            n += 1
            break
print(n)
")
if [[ "${launched_false}" == "0" ]]; then
    green "  ✅ No failed-launch NodeClaims (Karpenter never attempted — as expected)"
    pass_count=$((pass_count + 1))
else
    yellow "  ⚠️  ${launched_false} NodeClaim(s) with Launched=False — may be from a prior scenario"
fi

# 4. NodePool has no 'workload' requirement
has_workload_req=$(kubectl get nodepool.karpenter.sh "${NODEPOOL}" -o json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
reqs = data.get('spec', {}).get('template', {}).get('spec', {}).get('requirements', [])
print('yes' if any(r.get('key') == 'workload' for r in reqs) else 'no')
")
if [[ "${has_workload_req}" == "no" ]]; then
    green "  ✅ NodePool '${NODEPOOL}' missing the 'workload' requirement (the injected bug)"
    pass_count=$((pass_count + 1))
else
    red "  ❌ NodePool '${NODEPOOL}' still has the 'workload' requirement — inject.sh may not have run correctly"
    fail_count=$((fail_count + 1))
fi

echo ""
echo "──────────────────────────────────────────────────────────────"
if [[ "${fail_count}" -eq 0 ]]; then
    green "  ⚠️  Cluster is in the incident state — ready for agent investigation."
    echo ""
    echo "     Next step:"
    echo "       python3 ~/sre-agent/sre_agent_v2.py --scenario data_service_remediation \\"
    echo "         --prompt \"Overpass API pods stuck Pending in ${NAMESPACE}. Investigate.\""
    exit 0
else
    red "  ${fail_count} check(s) failed. Fix the environment before running the agent."
    exit 1
fi
