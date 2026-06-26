# Setup & Run Guide

End-to-end guide for setting up and running the AWS DevOps Agent EKS workshop. Walk through the steps in order; later sections give you narration cues, troubleshooting tips, and teardown commands.

For scenario-level detail (what each one demonstrates, the key messages, what the agent should surface), see [scenario-playbook.md](scenario-playbook.md).

---

## Step 1 — Provision an EKS cluster

If you don't already have one, follow this repo's [Quick Start (Basic)](../../../../README.md#quick-start-basic). Confirm before continuing:

```bash
kubectl get nodes
# Expect: at least one node Ready
```

> Track B (Scenarios 7–8) additionally needs Karpenter with a GPU NodePool tainted `nvidia.com/gpu=true:NoSchedule`, plus the KubeRay operator. The repo's Terraform stack provisions both when `kubeflow_platform_enabled = true`.

## Step 2 — Install the CloudWatch Observability EKS add-on

Required for Scenarios 2, 4, 4b, and 8 — provides `pod_memory_working_set`, `node_cpu_utilization`, and CFS-throttling metrics.

```bash
aws eks create-addon \
  --cluster-name <your-cluster> \
  --region <your-region> \
  --addon-name amazon-cloudwatch-observability
```

Wait until the add-on is `ACTIVE` and metrics have populated (5+ minutes):

```bash
aws eks describe-addon \
  --cluster-name <your-cluster> --region <your-region> \
  --addon-name amazon-cloudwatch-observability \
  --query 'addon.status'
# Expect: "ACTIVE"

aws cloudwatch list-metrics --namespace ContainerInsights --region <your-region> \
  --metric-name node_cpu_utilization \
  --query 'Metrics[0].Dimensions'
# Expect: dimensions including your cluster name
```

## Step 3 — Deploy the EKS Node Diagnostics MCP server

This is the custom MCP server the AWS DevOps Agent uses to reach node-OS-level data (iptables, dmesg, kubelet logs, etc.) via SSM Automation. It runs in your account and is required for Scenario 4.

Deploy it from [aws-samples/sample-eks-node-diagnostics-mcp](https://github.com/aws-samples/sample-eks-node-diagnostics-mcp) following its README. When the CDK deploy completes, save the four values it prints:

- **MCP gateway URL**
- **OAuth Client ID**
- **OAuth Client Secret**
- **Token URL**

You'll paste them into the AWS DevOps Agent console in Step 5.

## Step 4 — Create an AWS DevOps Agent Space

In the AWS console:

1. Open **DevOps Agent** → **Create Agent Space**.
2. Name it (e.g., `eks-workshop`) and choose your region.
3. Find the Agent Space's **runtime execution role** ARN — you'll need it for Step 6.

> **Important:** an Agent Space has two associated IAM roles. The "Operator access" role shown under **Settings** is the *administrator* role used to set up and configure the Space — **not** the role the agent assumes when it investigates your cluster. The Terraform module in Step 6 needs the **runtime execution role**, otherwise the agent will report "kubectl access not configured" even after `terraform apply` succeeds.
>
> The most reliable way to find the runtime role is via CloudTrail after the agent makes its first call:
>
> 1. From the Agent Space chat, run any prompt that touches the cluster (e.g., "list pods in the default namespace").
> 2. The agent will fail with an access error the first time — that's expected.
> 3. Run:
>    ```bash
>    aws cloudtrail lookup-events \
>      --lookup-attributes AttributeKey=EventName,AttributeValue=ListAccessEntries \
>      --max-results 5
>    ```
>    Or look for any `eks:Describe*` / `eks:List*` event your Agent Space role attempted. The `userIdentity.arn` on those events is the runtime role.
> 4. Use that ARN in Step 6.

## Step 5 — Register the MCP server as a Capability Provider

In the same Agent Space:

1. Open **Capability Providers** → **Register**.
2. Fill in the values from Step 3:
   - Server name: `eks-node-diagnostics`
   - Endpoint URL: the MCP gateway URL
   - Auth: OAuth 2.0 → paste Client ID, Client Secret, Token URL
3. Save. The console validates the connection and discovers the available tools.

## Step 6 — Wire the Agent Space to your EKS cluster (Terraform)

This codifies the workshop's manual access-entry steps. The module creates an EKS access entry for the Agent Space role and attaches `AmazonAIOpsAssistantPolicy` with cluster scope.

```bash
cd examples/agentic/aws-devops-agent-workshop/terraform
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars:
#   cluster_name         = "<your cluster>"
#   agent_space_role_arn = "<the ARN from Step 4>"
$EDITOR terraform.tfvars

terraform init
terraform apply
```

Verify:

```bash
aws eks list-associated-access-policies \
  --cluster-name <your-cluster> \
  --principal-arn <agent-space-role-arn> \
  --query 'associatedAccessPolicies[].policyArn'
# Expect: ["arn:aws:eks::aws:cluster-access-policy/AmazonAIOpsAssistantPolicy"]
```

## Step 7 — Pre-demo check

```bash
cd examples/agentic/aws-devops-agent-workshop
bash scripts/pre-demo-check.sh
```

Confirms kubectl reachability, Container Insights, CoreDNS, and namespace state. If any line says `FAIL`, fix it before running the scenarios.

## Step 8 — Open the windows you'll need during the demo

Have these ready before pasting the first prompt:

- **Browser tab 1:** AWS DevOps Agent → your Agent Space (chat window).
- **Browser tab 2:** CloudWatch → Container Insights → Performance monitoring → EKS Pods (filter to your cluster, namespace `demo-app` for Track A and `ml-inference` for Track B).
- **Browser tab 3:** AWS DevOps Agent → Investigations (for the audit trail / screenshots).
- **Terminal 1:** `kubectl` context set to the target cluster.
- **Terminal 2:** `aws logs tail /aws/bedrock-agentcore/EksNodeLogMcpStack-gateway --region <your-region> --follow` (so you can see the MCP gateway calls live while the agent investigates).

---

## Step 9 — Run the scenarios

Recommended order matches the playbook. Time estimates include the agent's reply window.

### Track A — General SRE (≈ 50 min)

| Step | Command | What to do when prompt prints | Narration cue |
|---|---|---|---|
| 1 | `bash scripts/scenario-1-healthcheck.sh` | Paste the printed prompt into the Agent Space chat. Wait for the report. | Look at what it surfaced *without* anyone reporting an incident — that's the demo's opening hook. Point out any pre-existing findings (e.g., long-stuck pending pods) as proof of "agent already knows your environment." |
| 2 | `bash scripts/scenario-2-oomkill-setup.sh` | Paste prompt. | The agent should cite exit code 137 and the time-to-OOMKill ("~1 second"). With Container Insights, it also plots `pod_memory_working_set` vs limit. Run teardown after. |
| 3 | `bash scripts/scenario-2-oomkill-teardown.sh` | — | — |
| 4 | `bash scripts/scenario-3-bad-deployment-setup.sh` | Paste prompt. | Listen for "previous revision is healthy, no production impact yet" before recommending action. That nuance is the SRE-judgment moment. |
| 5 | `bash scripts/scenario-3-bad-deployment-teardown.sh` | — | — |
| 6 | `bash scripts/scenario-4-cascading-failure-setup.sh` | Paste prompt. | Headline scene. The agent correlates K8s events + CloudTrail (who deployed the culprit + when) + node metrics. Often cites credit exhaustion timing for t3a nodes. |
| 7 | `bash scripts/scenario-4b-insights-followup.sh` | Paste prompt (after the first scenario-4 response). | The agent re-investigates using time-series metrics. Watch for it to *correct itself* if its first answer was inference-based. Highlight the honesty about Container Insights data gaps. |
| 8 | `bash scripts/scenario-4-cascading-failure-teardown.sh` | — | — |
| 9 | `bash scripts/scenario-5-prevention.sh` | Paste prompt. | Agent reviews everything investigated today. Expect a P0–P3 sprint plan grouped by observability / infrastructure / pipelines / resilience. |
| 10 | Open `scripts/scenario-6-skills.md` and walk through in the Agent Space console UI. | — | This step is doc-driven (no script). Demonstrates the **Skills tab** — how proven investigation procedures get encoded as custom Skills the agent applies automatically. |

### Track B — ML Inference Ops (≈ 25 min)

| Step | Command | What to do when prompt prints | Narration cue |
|---|---|---|---|
| 11 | `bash scripts/scenario-7-rayserve-toleration-setup.sh` | Paste prompt. Allow ~60s after setup for the FailedScheduling event to populate. | The fault is *layered*: surface error says "Insufficient GPU"; real cause is a toleration mismatch making Karpenter decline to provision. A good agent response digs past the surface message and recommends the exact Helm-values toleration fix. |
| 12 | `bash scripts/scenario-7-rayserve-toleration-teardown.sh` | — | — |
| 13 | `bash scripts/scenario-8-rayserve-oomkill-setup.sh` | Paste prompt. OOMKills appear within 6s. | A capable agent will extend the mock-workload recommendation to a *production-tier* recommendation for the real `meta-llama3-8b-vllm` workload — citing model size, Ray overhead, KV cache. That's the "ML-awareness without being told" moment. |
| 14 | `bash scripts/scenario-8-rayserve-oomkill-teardown.sh` | — | — |

### Post-flight (1 min)

```bash
bash scripts/cleanup.sh
```

Removes any lingering demo workloads. Cluster is back to a clean baseline.

---

## Narration patterns

Three reusable phrases that work in almost every scenario response:

- **"Notice it didn't just describe what happened — it explained *why*."** Use whenever the agent moves from observation → causal reasoning.
- **"And it gave me the exact remediation, not a generic recommendation."** Use whenever the agent produces a `kubectl ...` or `helm upgrade ...` command targeted at the specific resource at fault.
- **"It also surfaced something I didn't ask about."** Use whenever the agent volunteers a finding outside the scope of the question. This is the moment that distinguishes an autonomous agent from a tool-use loop.

---

## Interpreting the agent's output

A few patterns worth knowing before you read the agent's responses out loud to an audience:

- **The agent surfaces evidence + reasoning + remediation.** Look for all three. If a response is missing the evidence layer (e.g., a memory recommendation with no `pod_memory_working_set` reference), pull on that thread with a follow-up question.
- **Verify specific evidence claims against the actual resource.** Like any LLM-driven tool, the agent can occasionally over-specify a detail. The direction of the finding is usually right; specific details can drift. Confirm before acting.
- **Bonus findings are a feature.** The agent will often surface cross-cutting issues outside the scope of your question — a misconfigured probe noticed during an OOMKill investigation, a dual-autoscaler conflict noticed during a scheduling failure. Treat these as signal, validate, and follow up.
- **Custom Skills shape behavior.** If a scenario produces a response too generic for your needs, the right next step is to encode the investigation lens you want as a Custom Skill (see Scenario 6) so the agent applies it automatically.

---

## Troubleshooting

| Symptom | Most likely cause | Fix |
|---|---|---|
| Agent says "doesn't have kubectl access to this cluster" | Step 6 was never run, or wrong ARN | Re-run terraform with the correct Agent Space role ARN |
| Agent gives a generic "raise the memory limit" answer in Scenario 2 or 8 (not specific values) | Container Insights metrics not yet flowing | Wait 5+ minutes after add-on installation; verify with `aws cloudwatch list-metrics --namespace ContainerInsights` |
| Scenario 7 worker pod is `Running` instead of `Pending` | A worker node already has the wrong-key toleration applied (unlikely) | Check `kubectl describe pod` for the scheduling event; adjust the script's wrong toleration key to something more unique |
| `pre-demo-check.sh` warns about `session-manager-plugin` | Not required for these scenarios | Ignore the warning |
| Scenario 4 agent doesn't correlate the CloudTrail user who deployed batch-processor | Your IAM session's events are out of the visible CloudTrail window | Re-run scenario 4 setup with a fresh IAM session |

---

## Teardown

When the demo is fully complete and you want to remove the wiring:

```bash
cd examples/agentic/aws-devops-agent-workshop/terraform
terraform destroy
```

That removes the EKS access entry + policy association. The cluster itself, the Agent Space, the MCP server, and the CloudWatch Observability add-on remain (they're outside this module's scope). Delete those separately when you're done with the cluster.
