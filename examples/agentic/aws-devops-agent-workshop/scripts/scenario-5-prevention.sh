#!/usr/bin/env bash
# Scenario 5: Proactive prevention. Run after the other scenarios so the agent
# has investigation history to reason over. Prompt-only — no cluster changes.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

banner "SCENARIO 5: Proactive prevention"
echo "Option A: open Agent Space → Improvements → 'Run Now'"
echo "Option B: paste the prompt below."

prompt "Review all the issues you've investigated in my cluster $EKS_CLUSTER_NAME today — the OOMKills, the failed deployments, the CPU starvation cascade. What patterns do you see? Recommend improvements across:
1. Observability (what monitoring should I add?)
2. Infrastructure (what resource configs should change?)
3. Deployment pipelines (what gates should I add?)
4. Application resilience (what patterns should I implement?)"
