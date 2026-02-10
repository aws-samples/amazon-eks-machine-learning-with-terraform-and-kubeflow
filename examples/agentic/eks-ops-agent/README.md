# EKS Ops Agent

A LangGraph-based AI agent for managing and troubleshooting Amazon EKS clusters, deployed via [kagent](https://kagent.dev).

## Overview

EKS Ops Agent demonstrates building effective AI agents with:
- **LangGraph** for agent orchestration
- **Amazon Bedrock** (Claude) as the LLM
- **kagent** for Kubernetes-native deployment and lifecycle management
- **EKS MCP Server** for cluster operations (Module 2)
- **Memory** for context persistence (Module 3)
- **Langfuse** for observability (Module 4)

## Prerequisites

- **AWS account with Bedrock access**
  - Enable model access in [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/) → Model access
  - Default model: `us.anthropic.claude-sonnet-4-20250514-v1:0` (Claude Sonnet 4)
  - You can use any Bedrock model by setting `BEDROCK_MODEL_ID` in `manifests/eks-ops-agent.yaml`
  - Note: Claude 4.x models require cross-region inference profiles (prefix with `us.`)
- **EKS cluster with kagent installed**
- **Docker** for container builds
- **AWS CLI and kubectl** configured for your cluster

## Quick Start

### Step 1: Enable kagent in Terraform

Add to your `terraform.tfvars`:
```hcl
kagent_enabled              = true
kagent_enable_bedrock_access = true
kagent_enable_ui            = true
```

### Step 2: Setup IAM Permissions (EC2 only)

If running from an EC2 instance, run the setup script to add required IAM permissions:
```bash
./setup.sh
```

### Step 3: Apply Terraform

```bash
cd eks-cluster/terraform/aws-eks-cluster-and-nodegroup
terraform apply
```

### Step 4: Build and Deploy the Agent

```bash
cd examples/agentic/eks-ops-agent
./build-and-deploy.sh
```

This script will:
- Build the container image
- Push to Amazon ECR
- Deploy the agent to kagent
- Configure IRSA for Bedrock access

### Step 5: Access the Agent

```bash
# Port-forward the kagent UI
kubectl port-forward -n kagent svc/kagent-ui 8080:8080
```

Open http://localhost:8080 and select "eks-ops-agent" to start chatting.

## Configuration

### Changing the LLM Model

Edit `manifests/eks-ops-agent.yaml`:
```yaml
env:
  - name: BEDROCK_MODEL_ID
    value: "us.anthropic.claude-sonnet-4-20250514-v1:0"  # Change this
```

Supported models (examples):
| Model | BEDROCK_MODEL_ID |
|-------|------------------|
| Claude Sonnet 4 | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
| Claude 3.5 Sonnet | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| Meta Llama 3 70B | `meta.llama3-70b-instruct-v1:0` |
| Mistral Large | `mistral.mistral-large-2402-v1:0` |

### Agent Naming Convention

Agents with Bedrock access must have names ending in `-agent` (e.g., `eks-ops-agent`, `my-cool-agent`). This is enforced by the IAM trust policy for security.

## Project Structure

```
eks-ops-agent/
├── build-and-deploy.sh    # Build container and deploy to kagent
├── setup.sh               # EC2 IAM setup (run before terraform)
├── Dockerfile
├── pyproject.toml
├── manifests/
│   └── eks-ops-agent.yaml # Agent CRD (BYO agent)
└── src/
    ├── agent.py           # LangGraph agent definition
    ├── app.py             # KAgentApp wrapper (A2A protocol)
    ├── config.py          # Configuration
    ├── tools.py           # Module 2: EKS MCP Server tools
    └── agent-card.json    # Agent metadata for kagent
```

## Modules

### Module 1: Barebone Agent
Simple LangGraph agent with Bedrock Claude that can answer Kubernetes/EKS questions.

### Module 2: EKS MCP Server Integration (Current)
Adds tools for cluster operations via [EKS MCP Server](https://docs.aws.amazon.com/eks/latest/userguide/eks-mcp.html):

**Available Tools (16 total):**
| Category | Tools |
|----------|-------|
| Cluster Management | `manage_eks_stacks` |
| Kubernetes Resources | `manage_k8s_resource`, `apply_yaml`, `list_k8s_resources`, `list_api_versions` |
| Application Support | `generate_app_manifest`, `get_pod_logs`, `get_k8s_events`, `get_eks_vpc_config` |
| CloudWatch | `get_cloudwatch_logs`, `get_cloudwatch_metrics`, `get_eks_metrics_guidance` |
| IAM | `get_policies_for_role`, `add_inline_policy` |
| Troubleshooting | `search_eks_troubleshoot_guide`, `get_eks_insights` |

**Configuration:**
```yaml
# manifests/eks-ops-agent.yaml
env:
  - name: ENABLE_MCP_TOOLS
    value: "true"  # Set to "false" for Q&A-only mode
```

**How it works:**
1. Agent loads tools from EKS MCP Server at startup via `mcp-proxy-for-aws`
2. Uses SigV4 authentication with IRSA credentials (no static keys)
3. LangGraph ReAct pattern decides when to call tools based on user queries
4. Tools execute cluster operations and return results to the LLM

**Example prompts:**
- "List all pods in the default namespace"
- "Get the logs from pod xyz in namespace abc"
- "What events happened in the cluster in the last hour?"
- "Check the health of my EKS cluster"
- "Generate a deployment manifest for an nginx application"

### Module 3: Memory
*TODO* - Add context persistence:
- Short-term: KAgentCheckpointer (PostgreSQL)
- Long-term: Redis key-value store

### Module 4: Observability
*TODO* - Add Langfuse integration for:
- Trace visualization
- Cost tracking
- Performance monitoring

## Troubleshooting

### Check agent logs
```bash
kubectl logs -n kagent -l kagent.dev/agent=eks-ops-agent -f
```

### Check agent status
```bash
kubectl get agents -n kagent
kubectl get pods -n kagent -l kagent.dev/agent=eks-ops-agent
```

### Common errors

**AccessDenied on InvokeModel**
- Ensure model access is enabled in Bedrock console
- Check the IAM role has correct permissions
- For Claude 4.x, use `us.` prefix for cross-region inference

**IRSA not working**
- Verify ServiceAccount annotation: `kubectl get sa eks-ops-agent -n kagent -o yaml`
- Check IAM role trust policy allows `*-agent` pattern
