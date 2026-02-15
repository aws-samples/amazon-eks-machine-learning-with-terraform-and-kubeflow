# EKS Ops Agent Workshop

Build an AI agent that manages and troubleshoots Amazon EKS clusters using LangGraph, MCP Server and kagent.

## What You'll Build

| Module | Description |
|--------|-------------|
| **Module 1** | Barebone agent - Build and deploy BYO agent with Amazon Bedrock as model provider using kagent |
| **Module 2** | EKS MCP Server integration - Connect the agent to EKS MCP Server and access tools for cluster operations |
| **Module 3** | Memory - Short-term and long-term context persistence |
| **Module 4** | Observability - Langfuse tracing and monitoring |

## Prerequisites

- **AWS account with Bedrock access**
  - Enable model access in [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/) → Model access
  - Default model: `us.anthropic.claude-sonnet-4-20250514-v1:0` (Claude Sonnet 4)
  - You can use any Bedrock model by setting `BEDROCK_MODEL_ID` in `manifests/eks-ops-agent.yaml`
  - Note: Claude 4.x models require cross-region inference profiles (prefix with `us.`)
- **EKS cluster with kagent installed**
- **Docker** for container builds
- **AWS CLI and kubectl** configured for your cluster

> **Note:** Before starting this workshop, complete the [main repository setup](../../../README.md) to provision your cloud desktop (EC2 instance) and EKS cluster via Terraform.

---

## Module 1: Barebone Agent

In this module, you'll deploy a simple Q&A agent that can answer Kubernetes and EKS questions using Amazon Bedrock Claude.

### Step 1.1: Configure Terraform for kagent

Edit your `terraform.tfvars` file:

```bash
cd eks-cluster/terraform/aws-eks-cluster-and-nodegroup
```

Add these variables:
```hcl
kagent_enabled               = true
kagent_enable_bedrock_access = true
kagent_enable_ui             = true
```

### Step 1.2: Run Setup Script (EC2 only)

If running from an EC2 instance, the setup script adds required IAM permissions for Terraform to create the Bedrock access role:

```bash
cd examples/agentic/eks-ops-agent
./setup.sh
```

### Step 1.3: Apply Terraform

```bash
cd eks-cluster/terraform/aws-eks-cluster-and-nodegroup
terraform apply
```

This creates:
- kagent controller and UI
- IAM role for Bedrock access (with `*-agent` ServiceAccount pattern)
- EKS Access Entry for Kubernetes API access

### Step 1.4: Build and Deploy the Agent

```bash
cd examples/agentic/eks-ops-agent
./build-and-deploy.sh
```

The script will:
1. Build the container image
2. Push to Amazon ECR
3. Deploy the agent CRD to kagent
4. Configure IRSA (IAM Roles for Service Accounts)
5. Restart the deployment to pick up credentials

### Step 1.5: Access the Agent

Port-forward the kagent UI:
```bash
kubectl port-forward -n kagent svc/kagent-ui 8080:8080
```

Open http://localhost:8080 and select **eks-ops-agent** to start chatting.

### Step 1.6: Test Module 1

Try these prompts to verify the agent works:
- "What is a Kubernetes Pod?"
- "How do I troubleshoot a CrashLoopBackOff error?"
- "Explain the difference between a Deployment and a StatefulSet"

The agent should respond with helpful Kubernetes/EKS guidance (but cannot access your actual cluster yet - that comes in Module 2).

### Verify: Check Agent Logs

```bash
kubectl logs -n kagent -l kagent=eks-ops-agent -f
```

You should see:
```
INFO - Creating agent without tools (Q&A mode)
INFO - Starting EKS Ops Agent on 0.0.0.0:8080
```

---

## Module 2: EKS MCP Server Integration

In this module, you'll add 20 tools from the [AWS managed EKS MCP Server](https://docs.aws.amazon.com/eks/latest/userguide/eks-mcp.html) that allow the agent to query and manage your actual EKS cluster.


### Step 2.1: Code Snippet

The MCP integration is in `src/tools.py`. Here's how it works:

```python
# src/tools.py - Key code snippet

from langchain_mcp_adapters.client import MultiServerMCPClient

def get_mcp_server_config() -> dict:
    """Configure connection to EKS MCP Server via mcp-proxy-for-aws."""
    eks_mcp_endpoint = f"https://eks-mcp.{config.AWS_REGION}.api.aws/mcp"

    return {
        "eks-mcp": {
            "transport": "stdio",  # Spawn mcp-proxy-for-aws as subprocess
            "command": "uvx",
            "args": [
                "mcp-proxy-for-aws@latest",
                eks_mcp_endpoint,
                "--service", "eks-mcp",
                "--region", config.AWS_REGION,
            ],
            "env": {
                "AWS_REGION": config.AWS_REGION,
                # Pass through IRSA credentials automatically
                **{k: v for k, v in os.environ.items() if k.startswith("AWS_")},
            },
        }
    }

async def load_eks_tools() -> list:
    """Load tools from EKS MCP Server."""
    global _mcp_client

    mcp_config = get_mcp_server_config()
    _mcp_client = MultiServerMCPClient(mcp_config)
    tools = await _mcp_client.get_tools()

    return tools
```

The agent uses `ChatBedrockConverse` (Converse API) for better tool result handling:

```python
# src/agent.py - Key code snippet

from langchain_aws import ChatBedrockConverse

def get_llm(tools: list = None) -> ChatBedrockConverse:
    """Create Bedrock Claude LLM using Converse API."""
    llm = ChatBedrockConverse(
        model=config.BEDROCK_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0,
        max_tokens=4096,
    )
    if tools:
        return llm.bind_tools(tools)
    return llm
```

### Step 2.2: Enable MCP Tools

The manifest already has MCP tools enabled. Verify in `manifests/eks-ops-agent.yaml`:

```yaml
env:
  - name: ENABLE_MCP_TOOLS
    value: "true"
```

### Step 2.3: Rebuild and Deploy

```bash
cd examples/agentic/eks-ops-agent
./build-and-deploy.sh
```

### Step 2.4: Verify Tools Loaded

Check the agent logs:

```bash
kubectl logs -n kagent -l kagent=eks-ops-agent -f
```

You should see tools getting loaded:

```text
2026-02-10 17:04:49,277 - root - INFO - Logging configured with level: INFO
2026-02-10 17:04:59,513 - __main__ - INFO - Loading EKS MCP Server tools...
2026-02-10 17:04:59,514 - tools - INFO - Connecting to EKS MCP Server in us-west-2...
Downloading beartype (1.3MiB)
Downloading lupa (2.0MiB)
Downloading pygments (1.2MiB)
Downloading awscrt (3.9MiB)
 Downloaded lupa
 Downloaded awscrt
 Downloaded pygments
 Downloaded beartype
Installed 88 packages in 10.11s
/usr/local/lib/python3.13/contextlib.py:109: DeprecationWarning: Use `streamable_http_client` instead.
  self.gen = func(*args, **kwds)
2026-02-10 17:05:30,581 - tools - INFO - Loaded 20 tools from EKS MCP Server:
2026-02-10 17:05:30,581 - tools - INFO -   - manage_k8s_resource
2026-02-10 17:05:30,581 - tools - INFO -   - generate_app_manifest
2026-02-10 17:05:30,582 - tools - INFO -   - read_k8s_resource
2026-02-10 17:05:30,582 - tools - INFO -   - get_eks_insights
2026-02-10 17:05:30,582 - tools - INFO -   - get_eks_metrics_guidance
2026-02-10 17:05:30,582 - tools - INFO -   - list_api_versions
2026-02-10 17:05:30,582 - tools - INFO -   - get_policies_for_role
2026-02-10 17:05:30,582 - tools - INFO -   - get_pod_logs
2026-02-10 17:05:30,583 - tools - INFO -   - search_eks_documentation
2026-02-10 17:05:30,583 - tools - INFO -   - list_k8s_resources
2026-02-10 17:05:30,583 - tools - INFO -   - add_inline_policy
2026-02-10 17:05:30,583 - tools - INFO -   - get_cloudwatch_metrics
2026-02-10 17:05:30,583 - tools - INFO -   - manage_eks_stacks
2026-02-10 17:05:30,583 - tools - INFO -   - describe_eks_resource
2026-02-10 17:05:30,583 - tools - INFO -   - search_eks_troubleshooting_guide
2026-02-10 17:05:30,583 - tools - INFO -   - get_cloudwatch_logs
2026-02-10 17:05:30,584 - tools - INFO -   - list_eks_resources
2026-02-10 17:05:30,584 - tools - INFO -   - get_eks_vpc_config
2026-02-10 17:05:30,584 - tools - INFO -   - get_k8s_events
2026-02-10 17:05:30,584 - tools - INFO -   - apply_yaml
2026-02-10 17:05:30,584 - __main__ - INFO - Loaded 20 EKS MCP tools
```

### Step 2.5: Test MCP Tools

Open the kagent UI and try these prompts (replace `<cluster-name>` with your actual cluster name, e.g., `<CLUSTER_NAME>`):

1. **List resources:**
   ```
   List all pods in the default namespace on cluster <cluster-name>
   ```

2. **Get pod logs:**
   ```
   Get the logs from pod <pod-name> in namespace <namespace> on cluster <cluster-name>
   ```

3. **Check cluster events:**
   ```
   Show me recent events in the kube-system namespace on cluster <cluster-name>
   ```

4. **Generate manifests:**
   ```
   Generate a deployment manifest for a Redis application with 2 replicas.
   Deploy the manifest, ensure that Pods are in Running state.
   ```

5. **Get cluster insights:**
   ```
   Get insights and recommendations for cluster <cluster-name>
   ```

6. **Multi-step deployment task:**
   ```
   On cluster <CLUSTER_NAME>, deploy a new nginx application called "test-nginx" with 3 replicas in the default namespace. After deployment, verify all pods are running. Then scale it down to 1 replica and confirm the change.
   ```

7. **Cluster discovery** (agent discovers clusters automatically):
   ```
   List all pods in the default namespace
   ```

8. **NodeGroup inspection:**
   ```
   How many NodeGroups are in cluster <CLUSTER_NAME>?
   ```

9. **NodeGroup details:**
   ```
   Give me details about the NodeGroup <nodegroup-name> in cluster <CLUSTER_NAME>
   ```

---

## Project Structure

```
eks-ops-agent/
├── build-and-deploy.sh    # Build container and deploy to kagent
├── setup.sh               # EC2 IAM setup (run before terraform)
├── Dockerfile
├── pyproject.toml         # Python dependencies
├── manifests/
│   └── eks-ops-agent.yaml # Agent CRD (BYO agent)
└── src/
    ├── agent.py           # LangGraph agent with ReAct pattern
    ├── app.py             # KAgentApp wrapper (A2A protocol)
    ├── config.py          # Environment-based configuration
    ├── tools.py           # Module 2: EKS MCP Server tools
    └── agent-card.json    # Agent metadata for kagent
```

---

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for Bedrock and EKS MCP | `us-west-2` |
| `BEDROCK_MODEL_ID` | Bedrock model to use | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
| `ENABLE_MCP_TOOLS` | Enable EKS MCP Server tools | `true` (in manifest) |

### Tested Bedrock Models

| Model | BEDROCK_MODEL_ID |
|-------|------------------|
| Claude Sonnet 4 | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
| Claude 3.5 Sonnet | `anthropic.claude-3-5-sonnet-20241022-v2:0` |

> **Note:** Other Bedrock models that support tool calling should also work. Claude 4.x models require cross-region inference profiles (prefix with `us.`).

### Agent Naming Convention

Agents must have names ending in `-agent` (e.g., `eks-ops-agent`, `my-custom-agent`). This is enforced by the IAM trust policy for security.

---

## Troubleshooting

### Check agent status
```bash
kubectl get agents -n kagent
kubectl get pods -n kagent -l kagent=eks-ops-agent
```

### View agent logs
```bash
kubectl logs -n kagent -l kagent=eks-ops-agent -f
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `AccessDenied on InvokeModel` | Bedrock model not enabled | Enable model access in [Bedrock console](https://console.aws.amazon.com/bedrock/) |
| `AccessDenied on InvokeModel` (Claude 4.x) | Missing inference profile prefix | Use `us.` prefix (e.g., `us.anthropic.claude-sonnet-4-...`) |
| `MCP tools not loading` | Environment variable not set | Verify `ENABLE_MCP_TOOLS=true` in manifest |
| `MCP tools not loading` | Missing IAM permissions | Check IAM role has `eks-mcp:*` permissions |
| `Tool calls failing with Unauthorized` | Missing EKS Access Entry | Run `aws eks list-access-entries --cluster-name <cluster>` to verify |
| `IRSA not working` | ServiceAccount not annotated | Check: `kubectl get sa eks-ops-agent -n kagent -o yaml \| grep eks.amazonaws.com` |

---

## Module 3: Memory - Add conversation persistence with Redis


## Module 4: Observability** - Add Langfuse for tracing and monitoring
