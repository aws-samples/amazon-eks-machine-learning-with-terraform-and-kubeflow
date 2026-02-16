"""
KAgentApp wrapper for EKS Ops Agent.

This module wraps the LangGraph agent with kagent's KAgentApp to:
1. Expose the A2A (Agent-to-Agent) protocol endpoint
2. Register with kagent controller
3. Enable session persistence via KAgentCheckpointer
4. Appear in kagent UI

Module 2 adds:
5. Load EKS MCP Server tools for cluster operations

Module 3 adds:
6. Long-term memory via Redis for user defaults (cluster, namespace)
"""

import asyncio
import json
import logging
import os

import httpx
import uvicorn
from kagent.core import KAgentConfig
from kagent.langgraph import KAgentApp, KAgentCheckpointer

from agent import create_agent_graph
from config import config as app_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_checkpointer() -> KAgentCheckpointer:
    """Create KAgentCheckpointer for LangGraph state persistence."""
    config = KAgentConfig()
    return KAgentCheckpointer(
        client=httpx.AsyncClient(base_url=config.url),
        app_name=config.app_name,
    )


def load_agent_card() -> dict:
    """Load the agent card configuration."""
    card_path = os.path.join(os.path.dirname(__file__), "agent-card.json")
    with open(card_path, "r") as f:
        return json.load(f)


async def load_mcp_tools() -> list:
    """
    Load tools from EKS MCP Server if enabled.

    Returns:
        List of tools, or empty list if disabled or failed.
    """
    if not app_config.ENABLE_MCP_TOOLS:
        logger.info("MCP tools disabled (set ENABLE_MCP_TOOLS=true to enable)")
        return []

    try:
        # Import here to avoid loading MCP dependencies if not needed
        from tools import load_eks_tools

        logger.info("Loading EKS MCP Server tools...")
        return await load_eks_tools()

    except ImportError as e:
        logger.warning(f"MCP dependencies not installed: {e}")
        logger.info("Install with: pip install mcp langchain-mcp-adapters")
        return []
    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}")
        logger.info("Agent will run in Q&A mode (no cluster tools)")
        return []


def load_memory_tools() -> list:
    """
    Load memory tools if enabled.

    Returns:
        List of memory tools, or empty list if disabled.
    """
    if not app_config.ENABLE_MEMORY:
        logger.info("Memory disabled (set ENABLE_MEMORY=true to enable)")
        return []

    try:
        from memory import MemoryService, get_memory_tools, set_memory_service

        # Initialize memory service
        memory_service = MemoryService(redis_url=app_config.REDIS_URL)
        set_memory_service(memory_service)

        logger.info(f"Memory enabled (Redis: {app_config.REDIS_URL})")
        return get_memory_tools()

    except ImportError as e:
        logger.warning(f"Memory dependencies not installed: {e}")
        logger.info("Install with: pip install 'eks-ops-agent[memory]'")
        return []
    except Exception as e:
        logger.error(f"Failed to initialize memory: {e}")
        return []


def main():
    """Main entry point - starts the KAgentApp server."""
    # Load MCP tools (Module 2)
    mcp_tools = asyncio.run(load_mcp_tools())
    if mcp_tools:
        logger.info(f"Loaded {len(mcp_tools)} EKS MCP tools")

    # Load memory tools (Module 3)
    memory_tools = load_memory_tools()
    if memory_tools:
        logger.info(f"Loaded {len(memory_tools)} memory tools")

    # Combine all tools
    tools = mcp_tools + memory_tools
    if not tools:
        logger.info("Running in Q&A mode (no tools)")

    # Create checkpointer for session persistence
    checkpointer = create_checkpointer()

    # Create the LangGraph agent with checkpointer and tools
    graph = create_agent_graph(checkpointer=checkpointer, tools=tools if tools else None)

    # Load agent card
    agent_card = load_agent_card()

    # Create KAgentApp
    config = KAgentConfig()
    app = KAgentApp(
        graph=graph,
        agent_card=agent_card,
        config=config,
        tracing=False,  # Disable until Module 4 (Langfuse)
    )

    # Start the server
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting EKS Ops Agent on {host}:{port}")

    uvicorn.run(
        app.build(),
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
