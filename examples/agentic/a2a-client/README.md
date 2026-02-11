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

## Sample Output

```
┌─ Agent1 ─────────────────────────────────────────────┐
│ Customer Care Agent (Agent1)                         │
│                                                      │
│ Processing support tickets...                        │
│ EKS infrastructure issues will be handed off to      │
│ eks-ops-agent.                                       │
└──────────────────────────────────────────────────────┘

Processing TICKET-001: Customer forgot their password
✓ Resolved: Sent password reset link to registered email.

Processing TICKET-002: Customer charged twice for subscription
✓ Resolved: Refund processed. Duplicate charge was a payment gateway error.

Processing TICKET-003: Customer reports their ML training jobs are failing
→ EKS issue detected, initiating A2A handoff...

┌─ A2A Handoff ────────────────────────────────────────┐
│ Handing off to EKS Ops Agent...                      │
│                                                      │
│ Ticket: TICKET-003                                   │
│ Issue: ML training jobs failing with OOMKilled       │
│ Namespace: kubeflow-user-example-com                 │
└──────────────────────────────────────────────────────┘

┌─ EKS Ops Agent Response ─────────────────────────────┐
│ I investigated the pods in kubeflow-user-example-com │
│ namespace and found...                               │
└──────────────────────────────────────────────────────┘
```

## Customization

Edit `main.py` to:
- Change the target agent URL (`EKS_OPS_AGENT_URL`)
- Add more sample tickets
- Modify the handoff message format
