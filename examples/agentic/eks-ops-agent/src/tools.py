"""
Module 2: EKS MCP Server Tools

This module connects to the AWS managed EKS MCP Server and loads
tools for the LangGraph agent to perform cluster operations.

EKS MCP Server provides 16 tools across 6 categories:
- Cluster Management: manage_eks_stacks
- Kubernetes Resources: manage_k8s_resource, apply_yaml, list_k8s_resources, list_api_versions
- Application Support: generate_app_manifest, get_pod_logs, get_k8s_events, get_eks_vpc_config
- CloudWatch: get_cloudwatch_logs, get_cloudwatch_metrics, get_eks_metrics_guidance
- IAM: get_policies_for_role, add_inline_policy
- Troubleshooting: search_eks_troubleshoot_guide, get_eks_insights
"""

import logging
import os

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import config

logger = logging.getLogger(__name__)


def get_mcp_server_config() -> dict:
    """
    Build MCP server configuration for EKS MCP Server.

    Uses mcp-proxy-for-aws to handle SigV4 authentication.
    IRSA credentials are automatically available in the container.
    """
    eks_mcp_endpoint = f"https://eks-mcp.{config.AWS_REGION}.api.aws/mcp"

    return {
        "eks-mcp": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "mcp-proxy-for-aws@latest",
                eks_mcp_endpoint,
                "--service", "eks-mcp",
                "--region", config.AWS_REGION,
            ],
            "env": {
                "AWS_REGION": config.AWS_REGION,
                # Pass through IRSA credentials
                **{k: v for k, v in os.environ.items() if k.startswith("AWS_")},
            },
        }
    }


async def load_eks_tools() -> list:
    """
    Load tools from EKS MCP Server.

    Returns:
        List of LangChain-compatible tools for EKS operations.

    Example:
        tools = await load_eks_tools()
        llm_with_tools = llm.bind_tools(tools)
    """
    try:
        mcp_config = get_mcp_server_config()
        logger.info(f"Connecting to EKS MCP Server in {config.AWS_REGION}...")

        # langchain-mcp-adapters 0.2.x API - no context manager
        client = MultiServerMCPClient(mcp_config)
        tools = await client.get_tools()

        logger.info(f"Loaded {len(tools)} tools from EKS MCP Server:")
        for tool in tools:
            logger.info(f"  - {tool.name}")
        return tools

    except Exception as e:
        logger.error(f"Failed to load EKS MCP tools: {e}")
        logger.warning("Agent will run without EKS tools (Q&A only mode)")
        return []
