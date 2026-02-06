# EKS Ops Agent

A LangGraph-based AI agent for managing and troubleshooting Amazon EKS clusters, deployed via kagent.

## Overview

EKS Ops Agent demonstrates building effective AI agents with:
- **LangGraph** for agent orchestration
- **Amazon Bedrock** (Claude) as the LLM
- **kagent** for Kubernetes-native deployment and lifecycle management
- **EKS MCP Server** for cluster operations (Phase 2)
- **Memory** for context persistence (Phase 3)
- **Langfuse** for observability (Phase 4)

## Quick Start

```bash
# One-time setup (if running on EC2)
./setup.sh

# Build and deploy
./build-and-deploy.sh
```

## Modules

### Phase 1: Barebone Agent
Simple LangGraph agent with Bedrock Claude that can answer Kubernetes/EKS questions.

### Phase 2: EKS MCP Server Integration
*TODO* - Add tools for cluster operations:
- Upgrade readiness checking
- Deployment debugging
- Inference endpoint health monitoring

### Phase 3: Memory
*TODO* - Add context persistence:
- Short-term: KAgentCheckpointer (PostgreSQL)
- Long-term: Redis key-value store

### Phase 4: Observability
*TODO* - Add Langfuse integration for:
- Trace visualization
- Cost tracking
- Performance monitoring

## Architecture



## Prerequisites

- **AWS account needs to have Bedrock access**
  - Enable model access in [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/) → Model access
  - Default model: `us.anthropic.claude-sonnet-4-20250514-v1:0` (Claude Sonnet 4)
  - You can use any Bedrock model by setting `BEDROCK_MODEL_ID` in `manifests/eks-ops-agent.yaml`
  - Note: Newer models (Claude 4.x) require cross-region inference profiles (prefix with `us.`)
- **EKS cluster with kagent installed** (`kagent_enabled = true` in terraform)
- **Docker** for container builds
- **AWS CLI and kubectl** configured for your cluster

## Project Structure

```
eks-ops-agent/
├── build-and-deploy.sh    # Build container and deploy to kagent
├── setup.sh               # EC2 IAM setup (run before terraform)
├── Dockerfile
├── pyproject.toml
├── manifests/
│   └── eks-ops-agent.yaml # Agent CRD
└── src/
    ├── agent.py           # LangGraph agent definition
    ├── app.py             # KAgentApp wrapper (A2A protocol)
    ├── config.py          # Configuration
    └── agent-card.json    # Agent metadata for kagent
```

