"""
KAgentApp wrapper for EKS Ops Agent.

This module wraps the LangGraph agent with kagent's KAgentApp to:
1. Expose the A2A (Agent-to-Agent) protocol endpoint
2. Register with kagent controller
3. Enable session persistence via KAgentCheckpointer
4. Appear in kagent UI
"""

import json
import logging
import os

import httpx
import uvicorn
from kagent.core import KAgentConfig
from kagent.langgraph import KAgentApp, KAgentCheckpointer

from agent import create_agent_graph

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


def main():
    """Main entry point - starts the KAgentApp server."""
    # Create checkpointer for session persistence
    checkpointer = create_checkpointer()

    # Create the LangGraph agent with checkpointer
    graph = create_agent_graph(checkpointer=checkpointer)

    # Load agent card
    agent_card = load_agent_card()

    # Create KAgentApp
    config = KAgentConfig()
    app = KAgentApp(
        graph=graph,
        agent_card=agent_card,
        config=config,
        tracing=True,  # Enable tracing for observability
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
