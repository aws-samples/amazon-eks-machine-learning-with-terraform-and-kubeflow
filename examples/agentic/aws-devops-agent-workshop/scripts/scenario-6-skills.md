# Scenario 6: Agent Skills — "Your Playbooks, Automated"

Console-only walkthrough. No shell script.

## Pre-demo setup (do once, before the demo)

1. Open Agent Space → Web App → **Skills** tab.
2. Click **Create** (or **+**) under **Custom Skills**.
3. Create this skill:

**Name:** `eks-pod-failure-investigation`

**Content:**

```
When investigating pod failures in an EKS cluster:
1. First check the pod status, events, and recent restarts
2. Pull container logs for the last 5 minutes
3. Check node resource pressure (CPU, memory, disk) using CloudWatch or MCP tools
4. If OOMKilled: compare resource requests/limits vs actual usage
5. If CrashLoopBackOff: check exit codes and startup logs
6. If Pending: check scheduling constraints, resource quotas, and node capacity
7. Compare the affected node with a healthy node if node-level issues suspected
8. Always provide:
   - Root cause summary
   - Severity rating (Critical/High/Medium/Low)
   - Immediate fix command
   - Long-term prevention recommendation
```

4. Toggle the skill **ON**.
5. Verify `understanding-agent-space` managed skill is also **ON**.

## During the demo

1. Show the **Skills** tab (~10 sec) — point out Custom + Managed skills side by side.
2. Say:
   > "This encodes your team's investigation playbook. The agent follows it automatically — no need to say 'use this skill'."
3. Reference earlier scenarios: "Notice how the OOMKill and cascading-failure investigations followed this exact structure — root cause, severity, fix command, prevention."

## Key message

> New team members get senior-level investigation quality from day one. Managed Skills auto-learn from past investigations; Custom Skills codify what your team already does well.
