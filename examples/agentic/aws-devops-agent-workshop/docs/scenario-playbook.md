# AWS DevOps Agent — EKS SRE Scenario Playbook

## Objective

Show how **AWS DevOps Agent** acts as an autonomous SRE teammate for EKS — replacing dashboard-hopping and manual investigation with an intelligent agent that orchestrates across your entire observability stack. Two tracks:

- **Track A — General SRE:** patterns that apply to any EKS workload (web services, microservices, batch). 6 scenarios.
- **Track B — ML Inference Ops:** scenarios specific to multi-node ML inference on Ray Serve / KubeRay / GPU nodepools. 2 scenarios.

> **Theme: "What if your on-call engineer woke up to answers instead of alerts?"**

---

## Target Audience

- **Track A:** Operations / SRE teams managing EKS clusters at scale.
- **Track B:** ML platform engineers running multi-node LLM inference on EKS (Ray Serve, KubeRay, GPU/EFA workloads).

Either track can be run independently. Track A first is the recommended path; Track B builds on the same agent + cluster setup.

---

## Capabilities Covered

### 🤖 Agent-Driven Architecture

DevOps agents redefine the single-pane-of-glass — not as a unified dashboard, but as an intelligent agent that orchestrates across all your observability tools.

### 🔍 Autonomous Investigation

AI agents automatically navigate across your monitoring stack to identify issues without human intervention.

### ⚡ Automatic Root Cause Analysis

Agents correlate signals across logs, metrics, and traces to pinpoint problems in seconds, not hours.

### 💡 Actionable Insights

Agents deliver context-aware recommendations and automated remediation paths.

### 🛡️ Proactive Prevention

The agent analyzes incident patterns to recommend improvements before issues recur.

### 📋 Codified SRE Expertise

Encode your team's proven runbooks as Skills — making senior-level investigation quality available to every team member from day one.

### 🤖 ML Inference Awareness *(Track B)*

The agent investigates Ray Serve, KubeRay, and GPU-nodepool failures the same way it investigates regular pods — but the evidence it surfaces (GPU tolerations, Karpenter NodeClaim status, Ray head/worker logs, model resource limits) is the difference between "looks broken" and "here's the exact Helm values change you need."

---

## Track B

Both Track B scripts use lightweight stand-ins for the real Ray Serve inference workloads in `examples/inference/rayserve/`:

- Scenario 7 deploys a minimal RayCluster with a deliberately wrong toleration key.
- Scenario 8 deploys a plain Deployment labeled like a Ray worker, allocating memory beyond its Helm-configured limit.

The agent's investigation path on these stand-ins is identical to what it would do on a real vLLM / Llama-3 / Qwen Ray Serve deployment — the `pod_memory_working_set` vs `resources.limits.memory` story is the same; only the absolute numbers change. This lets the scenarios run in seconds against minimal CPU pods rather than spinning up GPU nodes, pulling multi-GB containers, and downloading model weights. The playbook always points to the real example as the production reference.

---

## Setup (one-time)

### Track A prereqs

- A live EKS cluster you can reach with `kubectl`.
- An AWS DevOps Agent Space created in the AWS console. Note its IAM role ARN — you'll need it below.
- The custom MCP server for node-level diagnostics deployed into your account: see [aws-samples/sample-eks-node-diagnostics-mcp](https://github.com/aws-samples/sample-eks-node-diagnostics-mcp). Register the resulting MCP gateway URL as a Capability Provider in your Agent Space.
- The CloudWatch Observability EKS add-on installed (provides `pod_memory_working_set`, `node_cpu_utilization`, etc.). Required for Scenarios 2, 4, and 8.

### Track B additional prereqs

- **Karpenter** with at least one GPU NodePool tainted `nvidia.com/gpu=true:NoSchedule`. Clusters using managed node groups instead can adapt Scenario 7 to use a node-group taint.
- **KubeRay operator** installed in the cluster.

Each scenario script deploys its own workload on demand into the `demo-app` namespace (Track A) or `ml-inference` namespace (Track B). The corresponding teardown script removes it. No long-running RayService or vLLM workload is required ahead of time.

### Connect AWS DevOps Agent to your EKS cluster

The Agent Space is created in the AWS DevOps Agent console — that step and the IAM role for the Agent Space are out of scope for the Terraform module here. What this module does:

- Creates an EKS access entry for the Agent Space's IAM role on your cluster.
- Attaches the AWS-managed `AmazonAIOpsAssistantPolicy` (an EKS cluster access policy) with cluster scope.

That gives the agent read access to the cluster's Kubernetes API. The agent's own IAM permissions (CloudWatch, EKS describe, etc.) are managed by AWS as part of the Agent Space service.

```bash
cd examples/agentic/aws-devops-agent-workshop/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with:
#   cluster_name         = "<your cluster>"
#   agent_space_role_arn = "<from Agent Space → Settings → IAM role>"
terraform init
terraform apply
```

This is the codified version of the workshop's manual "EKS console → Access tab → Create access entry → Attach AmazonAIOpsAssistantPolicy" steps.

---

# Track A — General SRE Scenarios

Run in order; later scenarios build narrative continuity for Scenario 5 (Proactive Prevention).

### Scenario 1: Morning Health Check — "The Agent Knows Your Environment"

| Aspect | Detail |
| --- | --- |
| **What** | Agent performs a proactive cluster health assessment |
| **Shows** | Environment awareness, resource analysis, proactive detection |
| **Outcome** | Agent identifies CPU over-commitment, stale workloads, and resource risks — without any incident being reported |
| **Key Message** | The agent already knows your environment and catches things humans miss |

---

### Scenario 2: Resource Misconfiguration — "Telemetry Correlation"

| Aspect | Detail |
| --- | --- |
| **What** | A workload is OOMKilled due to incorrect resource limits |
| **Shows** | CloudWatch Container Insights integration, metric-to-event correlation, resource recommendation |
| **Outcome** | Agent correlates memory utilization trends with OOMKill events, identifies the exact misconfiguration, and provides correct resource values |
| **Key Message** | The agent doesn't just tell you *what* happened — it correlates telemetry to explain *why*, and gives exact values to fix it |

---

### Scenario 3: Failed Deployment — "When Releases Go Wrong"

| Aspect | Detail |
| --- | --- |
| **What** | A bad deployment (broken image tag) leaves a rollout stuck |
| **Shows** | Deployment investigation, image pull failure detection, rollback recommendation |
| **Outcome** | Agent identifies the bad image reference, traces the rollout history, and recommends whether to rollback or fix forward with exact commands |
| **Key Message** | With GitHub/GitLab connected, the agent correlates which commit introduced the issue. Release Management (preview) can catch these *before* they hit production |

---

### Scenario 4: Cascading Failure — "Intelligent Observability Orchestration"

| Aspect | Detail |
| --- | --- |
| **What** | A resource-hungry workload starves other services on the same node, causing cascading failures across multiple services |
| **Shows** | Multi-signal correlation across CloudWatch metrics, Kubernetes events, container logs, and node-level diagnostics |
| **Outcome** | Agent autonomously queries 4+ data sources, correlates CPU spike → pod throttling → liveness failures → restarts, and identifies the culprit workload |
| **Key Message** | This isn't a dashboard — it's an intelligent agent that decides which tools to query, in what order, and correlates signals that would take a human 30+ minutes to connect |

**What the agent correlates:**

- 📊 CloudWatch: Node CPU utilization spike
- 📋 Kubernetes Events: Pod restarts, throttling
- 📝 Container Logs: Timeout errors, connection refused
- 🔧 Node Diagnostics (MCP): Resource contention, CPU over-commitment

---

### Scenario 5: Proactive Prevention — "Getting Smarter Over Time"

| Aspect | Detail |
| --- | --- |
| **What** | After resolving incidents, the agent analyzes patterns and recommends improvements |
| **Shows** | Pattern recognition, observability gap detection, structured improvement recommendations |
| **Outcome** | Agent provides targeted recommendations across 4 areas: observability, infrastructure optimization, deployment pipelines, and application resilience |
| **Key Message** | Shifts from reactive firefighting to proactive engineering — the agent learns from every incident |

---

### Scenario 6: Agent Skills — "Your Playbooks, Automated"

| Aspect | Detail |
| --- | --- |
| **What** | Show how proven investigation procedures are encoded as Skills |
| **Shows** | Custom Skills (user-defined runbooks), Managed Skills (auto-learned), Knowledge/Memories |
| **Outcome** | Agent follows codified procedures automatically — no need to tell it "use this skill" |
| **Key Message** | New team members get senior-level investigation quality from day one. The agent also auto-learns from past investigations |

---

# Track B — ML Inference Ops Scenarios

Specific to multi-node ML inference on Ray Serve / KubeRay / GPU nodepools. Builds on the same Agent Space + cluster wiring as Track A. Prereq: a `RayService` deployed via the repo's Ray Serve Helm chart, GPU nodes available via Karpenter.

### Scenario 7: Ray Serve Scheduling Failure — "GPU Toleration Mismatch in Helm Values"

| Aspect | Detail |
| --- | --- |
| **What** | A `RayService` worker pod is stuck in `Pending` because the Helm chart values specify a GPU toleration key that doesn't match the Karpenter NodePool's taint |
| **Shows** | Cross-layer correlation: scheduling event → NodePool taint → Helm values → toleration mismatch |
| **Prereq** | This scenario assumes Karpenter is provisioning GPU nodes via a tainted NodePool. For clusters using managed node groups, the same fault can be reproduced by replacing the Karpenter step with a node-group taint mismatch — script comments call this out. |
| **Outcome** | Agent surfaces `0/N nodes matched the toleration key …` from scheduling events, identifies the mismatch with the actual NodePool/node-group taint, and recommends the corrected `tolerations:` block to set in the Helm values |
| **Key Message** | The agent reads scheduling events the way an SRE would. The recommendation is "change this YAML key in the Helm values" — actionable, not just descriptive. |

**What the agent should correlate:**

- 📋 Kubernetes Events: `FailedScheduling` with toleration message
- 🏷️ Karpenter NodePool / node-group spec: actual taint key/value
- 📦 Helm release values: the `tolerations:` array on the worker spec
- 📍 Worker pod spec: rendered tolerations vs taint

---

### Scenario 8: Ray Worker OOMKill — "Inference Worker Exceeds Helm-Configured Memory Limit"

| Aspect | Detail |
| --- | --- |
| **What** | A Ray Serve worker is repeatedly OOMKilled because its actual memory usage exceeds the `resources.limits.memory` configured in the Helm values |
| **Shows** | Container Insights memory time-series, pod termination reason, the gap between observed peak memory and the configured limit, Helm-values-targeted recommendation |
| **Prereq** | CloudWatch Observability EKS add-on must be active and have populated metrics for at least 5 minutes (Container Insights is required for `pod_memory_working_set`). |
| **Outcome** | Agent retrieves `lastState.terminated.reason: OOMKilled` (exit 137), plots `pod_memory_working_set` against the configured `resources.limits.memory` from the Helm release, identifies the gap, and recommends a higher limit value to set in the Helm values file with structured rationale (observed peak + headroom). |
| **Key Message** | The agent's OOMKill investigation on a Ray Serve worker mirrors the Track-A OOMKill investigation, but the **recommended fix is targeted at the Helm chart values** — "set `resources.limits.memory: <value>` in the rayserve chart's worker spec" — not a generic `kubectl patch`. |

**What the agent should correlate:**

- 📊 CloudWatch Container Insights: `pod_memory_working_set` time series for the Ray worker pod
- 🔄 Pod termination reason: exit code 137 / `OOMKilled`
- 🎯 Helm release values: current `resources.limits.memory` for the affected worker

A capable agent will extend the mock-workload recommendation to a production-tier recommendation for the real `meta-llama3-8b-vllm` workload — citing model size, Ray/Python overhead, and shared-memory needs. Cross-cutting findings (e.g., probe misconfigurations adjacent to the OOMKill) are the kind of bonus observation an autonomous investigator surfaces.

---

## Demo Flow Summary

Estimated time = setup + agent investigation + narration. Add ~5 min for cleanup if you tear down between scenarios.

| Track | Scenario | Est. time | Setup script | Teardown |
|---|---|---|---|---|
| Setup (one-time) | Codified EKS wiring | 3 min | `terraform apply` in `terraform/` | `terraform destroy` |
| A | 1. Morning Health Check | 5 min | `scripts/scenario-1-healthcheck.sh` (prompt only) | n/a |
| A | 2. OOMKill | 8 min | `scripts/scenario-2-oomkill-setup.sh` | `scripts/scenario-2-oomkill-teardown.sh` |
| A | 3. Failed Deployment | 8 min | `scripts/scenario-3-bad-deployment-setup.sh` | `scripts/scenario-3-bad-deployment-teardown.sh` |
| A | 4. Cascading Failure | 12 min | `scripts/scenario-4-cascading-failure-setup.sh` | `scripts/scenario-4-cascading-failure-teardown.sh` |
| A | 5. Proactive Prevention | 8 min | `scripts/scenario-5-prevention.sh` (prompt only) | n/a |
| A | 6. Agent Skills | 6 min | `scripts/scenario-6-skills.md` (console walkthrough) | n/a |
| B | 7. Ray Serve scheduling failure | 10 min | `scripts/scenario-7-rayserve-toleration-setup.sh` | `scripts/scenario-7-rayserve-toleration-teardown.sh` |
| B | 8. Ray worker OOMKill | 12 min | `scripts/scenario-8-rayserve-oomkill-setup.sh` | `scripts/scenario-8-rayserve-oomkill-teardown.sh` |

**Track A end-to-end:** ~50 min. **Track B end-to-end:** ~25 min (after a one-time RayService deploy). **Both tracks:** ~75 min, fits a 90-minute customer demo with Q&A.

---

## Integrations

| Category | Integrations |
| --- | --- |
| **Observability** | CloudWatch Container Insights, Amazon Managed Prometheus, Grafana |
| **EKS Native** | Kubernetes API (pods, events, logs), EKS access entries, Karpenter NodePools |
| **Node-Level Diagnostics** | Custom MCP server via SSM Automation (20+ node-level log sources). Deployed once via [aws-samples/sample-eks-node-diagnostics-mcp](https://github.com/aws-samples/sample-eks-node-diagnostics-mcp). |
| **ML Platform** *(Track B)* | KubeRay operator, Ray Serve Helm chart from this repo's `charts/machine-learning/serving/rayserve/` |

