# AWS DevOps Agent — EKS SRE Workshop

Demos AWS DevOps Agent (the managed service) as an autonomous SRE teammate for EKS, with two tracks:

- **Track A — Foundation:** 6 general SRE scenarios (health check, OOMKill, bad deploy, cascading failure, proactive prevention, agent skills).
- **Track B — ML Inference Ops:** 2 ML-specific scenarios (Ray Serve scheduling failure, Ray worker OOMKill) inspired by `examples/inference/rayserve/`.

Both tracks validated live against AWS DevOps Agent on a K8s 1.35 cluster. The agent produced Tier-1 reasoning on every scenario — no custom Skills needed.

## Layout

```
aws-devops-agent-workshop/
├── docs/
│   └── scenario-playbook.md            # full narrative + scenario tables + status & maturity
├── terraform/                          # codified EKS access entry + AmazonAIOpsAssistantPolicy
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── versions.tf
│   └── terraform.tfvars.example
├── scripts/
│   ├── _common.sh                      # shared helpers (auto-detects cluster from kubeconfig)
│   ├── env.sh.example                  # template for runtime env (gitignored env.sh)
│   ├── pre-demo-check.sh
│   ├── cleanup.sh
│   ├── scenario-1-healthcheck.sh
│   ├── scenario-2-oomkill-{setup,teardown}.sh
│   ├── scenario-3-bad-deployment-{setup,teardown}.sh
│   ├── scenario-4-cascading-failure-{setup,teardown}.sh
│   ├── scenario-4b-insights-followup.sh
│   ├── scenario-5-prevention.sh
│   ├── scenario-6-skills.md            # console walkthrough — no shell script
│   ├── scenario-7-rayserve-toleration-{setup,teardown}.sh
│   └── scenario-8-rayserve-oomkill-{setup,teardown}.sh
└── manifests/
    └── demo-workload.yaml
```

## Quick start

1. **Wire AWS DevOps Agent to your EKS cluster** (codifies the workshop's Module 1 manual steps):
   ```bash
   cd examples/agentic/aws-devops-agent-workshop/terraform
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars with your cluster name + Agent Space IAM role ARN
   terraform init
   terraform apply
   ```

2. **Deploy the EKS Node Diagnostics MCP server** (one-time, lives in your account):
   See [aws-samples/sample-eks-node-diagnostics-mcp](https://github.com/aws-samples/sample-eks-node-diagnostics-mcp). Register the resulting MCP gateway URL as a Capability Provider in your Agent Space.

3. **Install the CloudWatch Observability EKS add-on** (required for Scenarios 2, 4, and 8).

4. **Run the demo:**
   ```bash
   cd examples/agentic/aws-devops-agent-workshop
   bash scripts/pre-demo-check.sh
   # Track A:
   bash scripts/scenario-1-healthcheck.sh
   bash scripts/scenario-2-oomkill-setup.sh    # ...etc
   # Track B (requires Karpenter + KubeRay + Container Insights):
   bash scripts/scenario-7-rayserve-toleration-setup.sh
   bash scripts/scenario-8-rayserve-oomkill-setup.sh
   ```

## See also

- [docs/scenario-playbook.md](docs/scenario-playbook.md) — full scenario narrative with key messages, expected agent behavior, and demo flow timings (Track A ~50 min, Track B ~25 min, both ~75 min)
- `examples/inference/rayserve/meta-llama3-8b-vllm/` — the real Ray Serve / vLLM example that Track B's scenarios mock
- `charts/machine-learning/serving/rayserve/` — the Helm chart Track B scenarios point the agent at for fixes
