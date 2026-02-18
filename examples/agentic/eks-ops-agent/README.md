# EKS Ops Agent Workshop

Build an AI agent that manages and troubleshoots Amazon EKS clusters using LangGraph, MCP Server and kagent.

## What You'll Build

| Module | Description |
|--------|-------------|
| **Module 1** | Barebone agent - Build and deploy BYO agent with Amazon Bedrock as model provider using kagent |
| **Module 2** | EKS MCP Server integration - Connect the agent to EKS MCP Server and access tools for cluster operations |
| **Module 3** | Memory - Long-term memory with Redis for user defaults |

## Prerequisites

> **⚠️ Before You Begin:** This workshop assumes you have completed the [main repository setup](../../../README.md) through **Step 4** to provision your EC2 cloud desktop with Docker, AWS CLI, and kubectl installed.

- **AWS account with Bedrock access**
  - Enable model access in [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/) → Model access
  - Default model: `us.anthropic.claude-sonnet-4-20250514-v1:0` (Claude Sonnet 4)
  - You can use any Bedrock model by setting `BEDROCK_MODEL_ID` in `manifests/eks-ops-agent.yaml`
  - Note: Claude 4.x models require cross-region inference profiles (prefix with `us.`)
- **EC2 cloud desktop** from main repository setup (has Docker, AWS CLI, kubectl, terraform, VSCode and Kiro pre-installed)
- **EKS cluster** - will be created in Module 1 with kagent enabled

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              EKS Cluster                                     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         kagent namespace                               │  │
│  │                                                                        │  │
│  │   ┌─────────────┐     ┌───────────────────┐     ┌───────────────────┐  │  │
│  │   │  kagent-ui  │◄───►│ kagent-controller │◄───►│   eks-ops-agent   │  │  │
│  │   │    (Pod)    │     │       (Pod)       │     │       (Pod)       │  │  │
│  │   └─────────────┘     └─────────┬─────────┘     └─────────┬─────────┘  │  │
│  │                                 │                         │            │  │
│  │                       ┌─────────▼─────────┐               │            │  │
│  │                       │   PostgreSQL /    │               │            │  │
│  │                       │   SQLite (DB)     │               │            │  │
│  │                       └───────────────────┘               │            │  │
│  │                                                           │            │  │
│  │                       ┌───────────────────┐               │            │  │
│  │                       │  Redis (Module 3) │◄──────────────┘            │  │
│  │                       │  (User Defaults)  │                            │  │
│  │                       └───────────────────┘                            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
 ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
 │  Amazon Bedrock │        │   EKS MCP       │        │  Kubernetes     │
 │    (Claude)     │        │   Server        │        │     API         │
 └─────────────────┘        └─────────────────┘        └─────────────────┘
```

### Components

| Component | Role |
|-----------|------|
| **kagent-ui** | Web interface for chatting with agents |
| **kagent-controller** | Manages agent lifecycle, routes messages, stores sessions |
| **eks-ops-agent** | Your LangGraph agent (BYO agent pattern) |
| **PostgreSQL/SQLite** | Stores chat sessions, agent state (short-term memory) |
| **Redis** | Stores user defaults across sessions (long-term memory - Module 3) |
| **Amazon Bedrock** | LLM provider (Claude) for agent reasoning |
| **EKS MCP Server** | AWS-managed API providing 20+ EKS tools |

### Request Flow

```
User Question
     │
     ▼
┌─────────────┐    A2A Protocol     ┌───────────────────┐
│  kagent-ui  │ ───────────────────►│ kagent-controller │
└─────────────┘     (JSON-RPC)      └─────────┬─────────┘
                                              │
                                              │ Routes to agent
                                              ▼
                                    ┌───────────────────┐
                                    │   eks-ops-agent   │
                                    │    (LangGraph)    │
                                    └─────────┬─────────┘
                                              │
                  ┌───────────────────────────┼───────────────────────────┐
                  │                           │                           │
                  ▼                           ▼                           ▼
         ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
         │    Bedrock    │          │   EKS MCP     │          │     Redis     │
         │    (LLM)      │          │   (Tools)     │          │  (Defaults)   │
         └───────────────┘          └───────────────┘          └───────────────┘
```

---

## Module 1: Barebone Agent

In this module, you'll deploy a simple Q&A agent that can answer Kubernetes and EKS questions using Amazon Bedrock Claude.

### Step 1.1: Configure Terraform for kagent

Edit your `terraform.tfvars` file:

```bash
cd eks-cluster/terraform/aws-eks-cluster-and-nodegroup
```

Your `terraform.tfvars` should include these kagent variables:

```hcl
# Required variables (update with your values)
profile      = "default"
region       = "us-west-2"
cluster_name = "my-cluster"    # Choose your cluster name
azs          = ["us-west-2a", "us-west-2b", "us-west-2c"]
import_path  = "s3://<your_bucket>"

# kagent - AI Agent Framework
kagent_enabled               = true
kagent_database_type         = "sqlite"
kagent_enable_ui             = true
kagent_enable_bedrock_access = true
```

### Step 1.2: Run Setup Script (EC2 only)

**Run this BEFORE terraform apply.** The setup script adds IAM permissions to your EC2 instance role that Terraform needs to create the kagent Bedrock access role:

```bash
cd examples/agentic/eks-ops-agent
./setup.sh
```

You should see output like:
```
EKS Ops Agent - Workshop Setup
AWS Account: 123456789012
AWS Region:  us-west-2
Detected: Running on EC2 instance
Instance Role: eks-dcv-stack-InstanceRole-XXXX
Adding/updating IAM permissions for kagent...
IAM permissions added/updated successfully
```

### Step 1.3: Apply Terraform

```bash
cd eks-cluster/terraform/aws-eks-cluster-and-nodegroup
terraform apply
```

This creates:
- EKS cluster with kagent controller and UI
- IAM role for Bedrock access (with `*-agent` ServiceAccount pattern)
- EKS Access Entry for Kubernetes API access

**Verify kagent is running:**

```bash
# Update kubeconfig
aws eks update-kubeconfig --name <cluster-name> --region us-west-2

# Check kagent pods
kubectl get pods -n kagent
```

Expected output (you'll see 15+ pods):
```
NAME                                  READY   STATUS    RESTARTS   AGE
kagent-controller-xxxxxxxxxx-xxxxx    1/1     Running   0          2m
kagent-ui-xxxxxxxxxx-xxxxx            1/1     Running   0          2m
k8s-agent-xxxxxxxxxx-xxxxx            1/1     Running   0          2m
helm-agent-xxxxxxxxxx-xxxxx           1/1     Running   0          2m
... (and more built-in agents)
```

> **Note:** kagent comes with several built-in agents (k8s-agent, helm-agent, istio-agent, etc.). These are pre-configured agents for common operations. In this workshop, you'll build and deploy your own **eks-ops-agent** alongside them.

### Step 1.4: Build and Deploy the Agent

```bash
cd examples/agentic/eks-ops-agent
./build-and-deploy.sh
```

The script will:
1. Build the container image (`docker build`)
2. Create ECR repository and push the image
3. Apply the Agent CRD manifest to kagent
4. Annotate ServiceAccount for IRSA (IAM Roles for Service Accounts)
5. Wait until agent pods are ready

You should see output ending with:
```
deployment "eks-ops-agent" successfully rolled out

========================================
Build and deploy complete!
========================================
```

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

In this module, you'll enable 20+ tools from the [AWS managed EKS MCP Server](https://docs.aws.amazon.com/eks/latest/userguide/eks-mcp.html) that allow the agent to query and manage your actual EKS cluster.

**How this works:** The MCP integration code is already in `src/tools.py`. Setting `ENABLE_MCP_TOOLS=true` tells the agent to load these tools at startup. This pattern keeps the codebase modular - features are added incrementally and enabled via configuration.

### Step 2.1: Understand the Code

The MCP integration is in `src/tools.py`. Here's how it connects to the EKS MCP Server:

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

Edit `manifests/eks-ops-agent.yaml` and change `ENABLE_MCP_TOOLS` from `"false"` to `"true"`:

```yaml
env:
  - name: ENABLE_MCP_TOOLS
    value: "true"    # Changed from "false"
```

### Step 2.3: Redeploy the Agent

```bash
cd examples/agentic/eks-ops-agent
./build-and-deploy.sh
```

### Step 2.4: Verify Tools Loaded

Check the agent logs:

```bash
kubectl logs -n kagent -l kagent=eks-ops-agent
```

You should see 20 tools loaded:

```text
INFO - Memory disabled (set ENABLE_MEMORY=true to enable)
INFO - Creating agent with 20 tools
INFO - Starting EKS Ops Agent on 0.0.0.0:8080
```

The key confirmation is **"Creating agent with 20 tools"** - this means EKS MCP Server tools are loaded.

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

### Additional Prompts to Explore

These prompts demonstrate advanced EKS MCP Server capabilities:

**Cluster Health & Insights:**
```
Get EKS insights for cluster <CLUSTER_NAME>. Are there any misconfigurations?
```
```
Is my cluster ready for upgrade? Check upgrade readiness.
```

**Troubleshooting:**
```
Search the troubleshooting guide for how to fix ImagePullBackOff errors
```
```
Get recent events in the kagent namespace and summarize any warnings or errors
```

**IAM Analysis:**
```
What policies are attached to the node role for this cluster?
```

**CloudWatch Metrics:**
```
What CloudWatch metrics are available for monitoring pods?
```

**VPC & Networking:**
```
Show me the VPC configuration for cluster <CLUSTER_NAME> including subnets and security groups
```

**Application Deployment:**
```
Generate a deployment manifest for an nginx web app called "hello-eks" with 3 replicas and expose it via a LoadBalancer service on port 80. Deploy it to the default namespace on cluster <CLUSTER_NAME> and verify the pods are running.
```

### Sample Troubleshooting

This scenario demonstrates the agent's ability to diagnose issues in real-time.

**Step 1: Deploy a broken application**
```
Deploy an nginx app called "broken-app" using image "nginx:doesnotexist" with 2 replicas to the default namespace on cluster <CLUSTER_NAME>
```

This creates pods that will fail with `ImagePullBackOff` error.

**Step 2: Ask the agent to diagnose**
```
The broken-app pods are not running. Investigate why and tell me how to fix it.
```

The agent will use multiple tools to diagnose:
- `list_k8s_resources` - check pod status
- `get_k8s_events` - find error events
- `search_eks_troubleshooting_guide` - look up the error
- Provide actionable fix recommendations

> **Note:** Replace `<CLUSTER_NAME>` with your actual cluster name when testing these prompts.

---

## Module 3: Memory with Redis

In this module, you'll enable memory that persists user defaults (cluster, namespace) across chat sessions. When a user sets their defaults, they're stored in Redis and automatically retrieved in future sessions.

**How this works:** The memory code is already in `src/memory.py`. Setting `ENABLE_MEMORY=true` and deploying Redis enables this feature. Like Module 2, this follows the modular pattern - features are enabled via configuration.

> **Note:** This module uses an in-cluster Redis deployment for simplicity. For production use cases requiring durable long-term memory, consider managed services like Amazon ElastiCache, Amazon MemoryDB, Amazon OpenSearch (with vector search for semantic memory), or Amazon DynamoDB.

### How It Works

```
Session 1:
  User: "Set my default cluster to <CLUSTER_NAME> and namespace to default"
  Agent: ✓ Saved defaults

Session 2 (new session):
  User: "List all pods"
  Agent: (retrieves defaults from Redis, uses <CLUSTER_NAME>/default)
```

### Step 3.1: Understand the Code

The memory implementation is in `src/memory.py`. Here's the key pattern:

```python
# src/memory.py - Key code snippet

class MemoryService:
    """Redis-backed memory service for user defaults."""

    async def get_defaults(self, user_id: str) -> UserDefaults:
        """Retrieve user's saved cluster and namespace."""
        client = await self._get_client()
        data = await client.get(f"user:{user_id}:defaults")
        return UserDefaults.from_dict(json.loads(data)) if data else UserDefaults()

    async def set_defaults(self, user_id: str, cluster: str, namespace: str) -> UserDefaults:
        """Save user's default cluster and namespace."""
        client = await self._get_client()
        defaults = UserDefaults(cluster=cluster, namespace=namespace)
        await client.set(f"user:{user_id}:defaults", json.dumps(defaults.to_dict()))
        return defaults
```

Memory tools are exposed to the agent so it can get/set defaults:

```python
# src/memory.py - Memory tools

@tool
async def set_user_defaults(cluster: str = None, namespace: str = None) -> str:
    """Save the user's default cluster and/or namespace for future requests."""
    defaults = await _memory_service.set_defaults(user_id, cluster, namespace)
    return f"Saved defaults: {defaults}"

@tool
async def get_user_defaults() -> str:
    """Retrieve the user's saved default cluster and namespace."""
    defaults = await _memory_service.get_defaults(user_id)
    return f"User defaults: {defaults}"
```

### Step 3.2: Deploy Redis

Deploy Redis to your cluster:

```bash
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: kagent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: kagent
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
EOF
```

Expected output:
```
deployment.apps/redis created
service/redis created
```

Verify Redis is running:
```bash
kubectl get pods -n kagent -l app=redis
```
```
NAME                     READY   STATUS    RESTARTS   AGE
redis-xxxxxxxxxx-xxxxx   1/1     Running   0          10s
```

### Step 3.3: Enable Memory

Edit `manifests/eks-ops-agent.yaml` and change `ENABLE_MEMORY` from `"false"` to `"true"`:

```yaml
        # Module 3: Set to "true" to enable Redis memory for user defaults
        - name: ENABLE_MEMORY
          value: "true"    # Changed from "false"
```

> **Note:** `REDIS_URL` is already set to `redis://redis.kagent.svc.cluster.local:6379` in the manifest.
>
> **Tip:** To find any service URL, run `kubectl get svc <name> -n <namespace>`. The URL pattern is: `redis://<service>.<namespace>.svc.cluster.local:<port>`

### Step 3.4: Rebuild and Deploy

```bash
cd examples/agentic/eks-ops-agent
./build-and-deploy.sh
```

### Step 3.5: Verify Memory Loaded

Check the agent logs:

```bash
kubectl logs -n kagent -l kagent=eks-ops-agent -f
```

You should see:

```text
INFO - Loaded 20 EKS MCP tools
INFO - Memory enabled (Redis: redis://redis.kagent.svc.cluster.local:6379)
INFO - Loaded 3 memory tools
INFO - Creating agent with 23 tools
```

The agent now has 23 tools (20 MCP + 3 memory).

### Step 3.6: Test Memory

Open the kagent UI and try these prompts (replace `<CLUSTER_NAME>` with your actual cluster name):

1. **Set defaults:**
   ```
   Set my default cluster to <CLUSTER_NAME> and namespace to default
   ```

2. **Verify defaults saved:**
   ```
   What are my defaults?
   ```

3. **Start a new chat session** (click "New Chat" in kagent UI)

4. **Use defaults implicitly:**
   ```
   List all pods
   ```
   The agent should use your saved cluster and namespace without asking.

5. **Clear defaults:**
   ```
   Clear my defaults
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
    ├── memory.py          # Module 3: Redis memory for user defaults
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
| `ENABLE_MEMORY` | Enable Redis memory for user defaults | `false` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |

### Tested Bedrock Models

| Model | BEDROCK_MODEL_ID |
|-------|------------------|
| Claude Sonnet 4 (default) | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
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