"""
KAgentApp wrapper for Compliance Agent.

This module wraps the LangGraph agent with kagent's KAgentApp to:
1. Expose the A2A (Agent-to-Agent) protocol endpoint
2. Register with kagent controller
3. Enable session persistence via KAgentCheckpointer
4. Appear in kagent UI
5. Long-term memory via memledger (pgvector) for governance and audit
"""

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


def load_memory_tools() -> list:
    """
    Load memory tools for the compliance agent.

    Uses memledger with pgvector for persistent, searchable memory.
    Memory is the core capability of the compliance agent.

    Returns:
        List of memory tools, or empty list if disabled.
    """
    if not app_config.ENABLE_MEMORY:
        logger.info("Memory disabled (set ENABLE_MEMORY=true to enable)")
        return []

    if not app_config.MEMLEDGER_PG_DSN and not app_config.MEMLEDGER_CONFIG_PATH:
        logger.warning("Memory enabled but neither MEMLEDGER_PG_DSN nor MEMLEDGER_CONFIG_PATH set. Memory disabled.")
        return []

    try:
        from memory import MemoryService, get_memory_tools, set_memory_service

        memory_service = MemoryService(
            pg_connection_string=app_config.MEMLEDGER_PG_DSN,
            embedding_provider=app_config.MEMLEDGER_EMBEDDING_PROVIDER,
            embedding_model=app_config.MEMLEDGER_EMBEDDING_MODEL,
            embedding_dimensions=app_config.MEMLEDGER_EMBEDDING_DIMENSIONS,
            config_path=app_config.MEMLEDGER_CONFIG_PATH,
        )
        set_memory_service(memory_service)

        if app_config.MEMLEDGER_CONFIG_PATH:
            logger.info(f"Memory enabled (memledger config: {app_config.MEMLEDGER_CONFIG_PATH})")
        else:
            dsn_display = app_config.MEMLEDGER_PG_DSN.split('@')[-1] if '@' in app_config.MEMLEDGER_PG_DSN else 'configured'
            logger.info(f"Memory enabled (memledger pgvector: {dsn_display})")
        return get_memory_tools()

    except ImportError as e:
        logger.warning(f"Memory dependencies not installed: {e}")
        logger.info("Install with: pip install memledger[pgvector,bedrock]")
        return []
    except Exception as e:
        logger.error(f"Failed to initialize memory: {e}")
        return []


def main():
    """Main entry point - starts the KAgentApp server."""
    # Load memory tools (core capability for compliance agent)
    memory_tools = load_memory_tools()
    if memory_tools:
        logger.info(f"Loaded {len(memory_tools)} memory/compliance tools")

    tools = memory_tools
    if not tools:
        logger.warning("Running in Q&A mode (no tools) — compliance agent needs memory to be useful")

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
        tracing=False,
    )

    # Start the server
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting Compliance Agent on {host}:{port}")

    uvicorn.run(
        app.build(),
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
