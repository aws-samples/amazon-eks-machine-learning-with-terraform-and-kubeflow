# AWS DevOps Agent — EKS SRE Workshop

Demos AWS DevOps Agent (the managed service) as an autonomous SRE teammate for EKS, with two tracks:

- **Track A — General SRE:** 6 scenarios (health check, OOMKill, bad deploy, cascading failure, proactive prevention, agent skills).
- **Track B — ML Inference Ops:** 2 ML-specific scenarios (Ray Serve scheduling failure, Ray worker OOMKill) inspired by `examples/inference/rayserve/`.

Each scenario provides expected agent behavior and narration cues so you can drive the demo confidently. Track A focuses on cluster-level investigation; Track B focuses on Helm-values-targeted recommendations on ML workloads.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │       AWS DevOps Agent (managed)        │
                        │  Agent Space ▸ Investigations ▸ Skills  │
                        └──┬──────────────┬──────────────────┬────┘
                           │              │                  │
            MCP / HTTPS + OAuth2          │ EKS API           │ AWS SDK
                           │              │ (read-only via    │ (GetMetricData,
                           ▼              │  AmazonAIOps-     │  StartQuery,
                  ┌──────────────────┐    │  AssistantPolicy) │  …)
                  │  MCP Gateway     │    │                   │
                  │  (AgentCore)     │    │                   │
                  └────────┬─────────┘    │                   │
                           │              │                   ▼
                           ▼              │           ┌─────────────────────┐
                  ┌──────────────────┐    │           │  Amazon CloudWatch  │
                  │  Lambda (MCP     │    │           │  • Metrics          │
                  │  server, 20+     │    │           │    (Container       │
                  │  diagnostic      │    │           │     Insights:       │
                  │  tools)          │    │           │     pod_memory_*,   │
                  └────────┬─────────┘    │           │     node_cpu_*,     │
                           │              │           │     CFS throttling) │
                       SSM Automation     │           │  • Logs             │
                           │              │           │    (Logs Insights   │
                           ▼              ▼           │     queries)        │
                  ┌──────────────────┐  ┌──────────────────────────────────┐│
                  │   EKS worker     │  │       Amazon EKS cluster         ││
                  │   nodes          │  │                                  ││
                  │   (kubelet,      │  │  ┌────────────┐  ┌────────────┐  ││
                  │    iptables,     │  │  │  demo-app  │  │ml-inference│  ││
                  │    dmesg,        │  │  │ (Track A)  │  │ (Track B)  │  ││
                  │    CNI, etc.)    │◀─┤  └────────────┘  └────────────┘  ││
                  └──────────────────┘  │                                  ││
                                        │   Karpenter NodePools            ││
                                        │   CloudWatch Observability       ││
                                        │   add-on (publishes              ├┘
                                        │   Container Insights metrics)    │
                                        └─────────────────┬────────────────┘
                                                          │
                                                          │ publishes
                                                          │ metrics + logs
                                                          ▼
                                              (back up to CloudWatch above)

           ┌──────────────────────────────────┐
           │  Terraform module (this dir)     │
           │  • aws_eks_access_entry          │ wires the agent's IAM role
           │  • AmazonAIOpsAssistantPolicy    │ into the cluster
           │    association (cluster scope)   │
           └──────────────────────────────────┘
```

**Three paths the agent uses to investigate:**

- **MCP path** — for node-OS-level data (iptables, dmesg, kubelet, CNI configs) that the K8s API can't see. Routed via the MCP Gateway and SSM Automation into worker nodes.
- **EKS API path** — for cluster-state queries (pods, events, deployments, scheduling). Authorized by the EKS access entry + `AmazonAIOpsAssistantPolicy` that this directory's Terraform module creates.
- **CloudWatch path** — for time-series metrics (`pod_memory_working_set`, `node_cpu_utilization`, CFS throttling) and log queries. The CloudWatch Observability EKS add-on publishes these from the cluster; the agent reads them directly via the AWS SDK using its Agent Space role.

What this workshop's Terraform module provides: the access entry + policy association that lets the agent read the cluster. Everything else (Agent Space, MCP server, CloudWatch add-on) is set up separately as described in [docs/setup.md](docs/setup.md).

## Quick start

Full step-by-step in [docs/setup.md](docs/setup.md). High level:

1. Provision an EKS cluster and install the CloudWatch Observability add-on.
2. Deploy the [EKS Node Diagnostics MCP server](https://github.com/aws-samples/sample-eks-node-diagnostics-mcp) into your account.
3. Create an AWS DevOps Agent Space in the console and register the MCP server as a Capability Provider.
4. Run the Terraform module in `terraform/` to wire the Agent Space's role into the cluster.
5. Run `scripts/pre-demo-check.sh`, then walk through the scenarios in `scripts/`.

## See also

- [docs/setup.md](docs/setup.md) — step-by-step setup + run guide with narration cues and troubleshooting.
- [docs/scenario-playbook.md](docs/scenario-playbook.md) — scenario reference with key messages, expected agent behavior, and demo flow timings (Track A ~50 min, Track B ~25 min, both ~75 min).
- `examples/inference/rayserve/meta-llama3-8b-vllm/` — the real Ray Serve / vLLM example Track B's scenarios are inspired by.
- `charts/machine-learning/serving/rayserve/` — the Helm chart Track B scenarios point the agent at for fixes.
