"""
Module 2: EKS MCP Server Tools

This module connects to the AWS managed EKS MCP Server and loads
tools for the LangGraph agent to perform cluster operations.

Uses the recommended integration pattern from mcp-proxy-for-aws:
- aws_iam_streamablehttp_client for SigV4 authentication
- load_mcp_tools for LangChain-compatible tool loading

EKS MCP Server provides 20 tools across 6 categories:
- Cluster Management: manage_eks_stacks
- Kubernetes Resources: manage_k8s_resource, apply_yaml, list_k8s_resources, etc.
- Application Support: generate_app_manifest, get_pod_logs, get_k8s_events, etc.
- CloudWatch: get_cloudwatch_logs, get_cloudwatch_metrics, get_eks_metrics_guidance
- IAM: get_policies_for_role, add_inline_policy
- Troubleshooting: search_eks_troubleshooting_guide, get_eks_insights
"""

import logging

from mcp import ClientSession
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

from config import config

logger = logging.getLogger(__name__)


def get_eks_mcp_endpoint() -> str:
    """Get the EKS MCP Server endpoint URL."""
    return f"https://eks-mcp.{config.AWS_REGION}.api.aws/mcp"


async def load_eks_tools() -> list:
    """
    Load tools from EKS MCP Server.

    Uses aws_iam_streamablehttp_client for SigV4 authentication
    and load_mcp_tools for LangChain-compatible tool conversion.

    Returns:
        List of LangChain-compatible tools for EKS operations.

    Example:
        tools = await load_eks_tools()
        llm_with_tools = llm.bind_tools(tools)
    """
    endpoint = get_eks_mcp_endpoint()
    logger.info(f"Connecting to EKS MCP Server: {endpoint}")

    try:
        # Create authenticated client using mcp-proxy-for-aws
        mcp_client = aws_iam_streamablehttp_client(
            endpoint=endpoint,
            aws_region=config.AWS_REGION,
            aws_service="eks-mcp",
        )

        # Connect and load tools
        async with mcp_client as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Load tools using langchain-mcp-adapters
                tools = await load_mcp_tools(session)

                logger.info(f"Loaded {len(tools)} tools from EKS MCP Server:")
                for tool in tools:
                    logger.info(f"  - {tool.name}")

                return tools

    except Exception as e:
        logger.error(f"Failed to load EKS MCP tools: {e}")
        logger.warning("Agent will run without EKS tools (Q&A only mode)")
        return []
