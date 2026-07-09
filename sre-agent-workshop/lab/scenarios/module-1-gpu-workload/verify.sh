#!/usr/bin/env bash
# Verify that Module 1's fault state has fully surfaced before running the agent.
#
# Checks:
#   1. Ray workload deployments exist and are Pending
#   2. Karpenter NodeClaims exist with Launched=False (or none if Karpenter
#      hasn't reconciled yet — retry once)
#   3. GPU quota is 0
#
# Exits non-zero if any check fails so participants know the environment is
# not ready.

set -uo pipefail

NAMESPACE="${SRE_M1_NAMESPACE:-ray-inference}"
REGION="${AWS_REGION:-us-east-1}"

green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }

pass_count=0
fail_count=0

check() {
  local label="$1"; shift
  if "$@"; then
    green "  ✅ ${label}"
    pass_count=$((pass_count + 1))
  else
    red "  ❌ ${label}"
    fail_count=$((fail_count + 1))
  fi
}

echo "──────────────────────────────────────────────────────────────"
echo "  Module 1 — fault state verification"
echo "──────────────────────────────────────────────────────────────"

# 1. Workload deployments exist
check "ray-serve deployments applied" \
  kubectl get deploy -n "${NAMESPACE}" ray-serve-head ray-serve-worker >/dev/null 2>&1

# 2. Pods are Pending (allow up to ~10 seconds for pods to appear)
sleep 3
pending_count=$(kubectl get pods -n "${NAMESPACE}" -l app=ray-serve --field-selector=status.phase=Pending 2>/dev/null | grep -c "^ray-serve" || true)
if [[ "${pending_count}" -ge 1 ]]; then
  green "  ✅ ${pending_count} Ray pod(s) in Pending state"
  pass_count=$((pass_count + 1))
else
  yellow "  ⏳ no Pending pods yet — Karpenter may not have reconciled. Retrying in 10s…"
  sleep 10
  pending_count=$(kubectl get pods -n "${NAMESPACE}" -l app=ray-serve --field-selector=status.phase=Pending 2>/dev/null | grep -c "^ray-serve" || true)
  if [[ "${pending_count}" -ge 1 ]]; then
    green "  ✅ ${pending_count} Ray pod(s) in Pending state"
    pass_count=$((pass_count + 1))
  else
    red "  ❌ Ray pods should be Pending but are not"
    fail_count=$((fail_count + 1))
  fi
fi

# 3. Karpenter has attempted to provision — NodeClaims exist
nodeclaim_count=$(kubectl get nodeclaims.karpenter.sh 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')
if [[ "${nodeclaim_count}" -ge 1 ]]; then
  green "  ✅ ${nodeclaim_count} Karpenter NodeClaim(s) created"
  pass_count=$((pass_count + 1))
  # Show their Launched status for context
  echo ""
  yellow "  NodeClaim status (agent's most-important signal):"
  kubectl get nodeclaims.karpenter.sh -o json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
for nc in data.get('items', []):
    name = nc['metadata']['name']
    launched = 'Unknown'
    reason = ''
    for c in nc.get('status', {}).get('conditions', []):
        if c.get('type') == 'Launched':
            launched = c.get('status', 'Unknown')
            reason = c.get('reason', '')
    print(f'    {name:50s} Launched={launched:8s} reason={reason}')
" 2>/dev/null
  echo ""
else
  yellow "  ⏳ no NodeClaims yet — Karpenter may need more time. Waiting 20s…"
  sleep 20
  nodeclaim_count=$(kubectl get nodeclaims.karpenter.sh 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')
  if [[ "${nodeclaim_count}" -ge 1 ]]; then
    green "  ✅ ${nodeclaim_count} Karpenter NodeClaim(s) created"
    pass_count=$((pass_count + 1))
  else
    yellow "  ⚠️  no NodeClaims yet — this may still be OK if Karpenter is slow."
    yellow "     Re-run this verify.sh after a minute. The agent may catch the"
    yellow "     scheduling failure via the pod event alone even without NodeClaims."
  fi
fi

# 4. GPU quota check via AWS CLI (the same source the agent will consult)
gpu_quota=$(aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --region "${REGION}" \
  --query 'Quota.Value' \
  --output text 2>/dev/null || echo "unknown")

if [[ "${gpu_quota}" == "0.0" || "${gpu_quota}" == "0" ]]; then
  green "  ✅ GPU vCPU quota is 0 — this is the account-level truth the agent must find"
  pass_count=$((pass_count + 1))
elif [[ "${gpu_quota}" == "unknown" ]]; then
  yellow "  ⚠️  could not query Service Quotas — check IAM permissions on the desktop role"
else
  red "  ❌ GPU vCPU quota is ${gpu_quota} (expected 0). This scenario needs a"
  red "     Workshop Studio account with GPU quota locked at 0. If you increased it"
  red "     yourself, the scenario will not reproduce."
  fail_count=$((fail_count + 1))
fi

echo ""
echo "──────────────────────────────────────────────────────────────"
if [[ "${fail_count}" -eq 0 ]]; then
  green "  ⚠️  Cluster is in the incident state — ready for agent investigation."
  echo ""
  echo "     Next step:"
  echo "       python ~/sre-agent/sre_agent_v1.py --scenario gpu_incident \\"
  echo "         --prompt \"Ray inference pods stuck Pending in ${NAMESPACE}. Investigate.\""
  exit 0
else
  red "  ${fail_count} check(s) failed. Fix the environment before running the agent."
  exit 1
fi
