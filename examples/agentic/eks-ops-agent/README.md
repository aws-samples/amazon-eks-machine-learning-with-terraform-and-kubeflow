# EKS Ops Agent

An AI agent that manages and troubleshoots Amazon EKS clusters using [LangGraph](https://github.com/langchain-ai/langgraph), the [AWS EKS MCP Server](https://docs.aws.amazon.com/eks/latest/userguide/eks-mcp.html), and [kagent](https://github.com/kagent-dev/kagent).

The workshop is structured as three incremental modules:

| Module | What You Build |
|--------|---------------|
| 0 | Enable kagent on an existing EKS cluster |
| 1 | Barebone Q&A agent with Amazon Bedrock |
| 2 | Add 20+ cluster operations tools via EKS MCP Server |
| 3 | Persistent user defaults with Redis memory |

## Prerequisites

- EKS cluster provisioned via the [quick start setup](../../../README.md#quick-start-basic) (or adapted for [advanced setup](../../../README.md#advanced-setup))
- AWS account with Bedrock model access enabled (default model: Claude Sonnet 4)

## Getting Started

The full walkthrough lives in the Jupyter notebook:

👉 **[eks-ops-agent.ipynb](eks-ops-agent.ipynb)**

Open it on the ML Ops desktop and follow the modules in order.
