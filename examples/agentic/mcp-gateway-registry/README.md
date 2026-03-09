# MCP Gateway & Registry

Deploy the [MCP Gateway & Registry](https://github.com/agentic-community/mcp-gateway-registry) on your EKS cluster. This platform provides centralized access to MCP servers and AI agents with OAuth authentication, dynamic tool discovery, and unified governance.

The workshop walks through deploying the full stack (MongoDB, Keycloak, Registry, Auth Server, MCPGW) and accessing the Registry UI via port-forwarding.

| Module | What You Build |
|--------|---------------|
| 0 | Clone the upstream Helm chart and build dependencies |
| 1 | Deploy the full stack into the `mcp-gateway` namespace |
| 2 | Access the Registry UI and retrieve login credentials |

## Prerequisites

- EKS cluster provisioned via the [quick start setup](../../../README.md#quick-start-basic) (or adapted for [advanced setup](../../../README.md#advanced-setup))
- `helm` CLI v3.0+ and `kubectl` configured to access your cluster

## Getting Started

The full walkthrough lives in the Jupyter notebook:

👉 **[mcp-gateway-registry.ipynb](mcp-gateway-registry.ipynb)**

Open it on the ML Ops desktop and follow the modules in order.
