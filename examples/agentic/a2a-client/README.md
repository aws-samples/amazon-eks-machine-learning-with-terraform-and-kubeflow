# A2A Client Demo

Simple demonstration of Agent-to-Agent (A2A) communication. This client acts as a customer care agent (Agent1) that hands off EKS infrastructure issues to the eks-ops-agent (Agent2).

## How It Works

1. Agent1 processes sample support tickets
2. Simple tickets (password reset, billing) are resolved locally
3. EKS infrastructure tickets are handed off to Agent2 via A2A protocol
4. Agent2 investigates the cluster and returns findings

## Usage

### 1. Port-forward eks-ops-agent

```bash
kubectl port-forward -n kagent svc/eks-ops-agent 8081:8080
```

### 2. Run the client

```bash
cd examples/agentic/a2a-client
uv run python main.py
```

