"""
EKS Ops Agent - LangGraph agent with Bedrock Claude.

Module 1: Simple agent that can answer questions.
Module 2: Add EKS MCP Server tools for cluster operations.
Module 3: Add memory (short-term via checkpointer, long-term via Redis).
Module 4: Add Langfuse observability.
"""

from typing import Annotated, Optional, TypedDict

from langchain_aws import ChatBedrock
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from config import config


# --- State Definition ---


class AgentState(TypedDict):
    """State passed between nodes in the graph."""

    messages: Annotated[list, add_messages]


# --- LLM Setup ---


def get_llm() -> ChatBedrock:
    """Create Bedrock Claude LLM instance."""
    return ChatBedrock(
        model_id=config.BEDROCK_MODEL_ID,
        region_name=config.AWS_REGION,
        model_kwargs={
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    )


# --- System Prompt ---

SYSTEM_PROMPT = """You are an EKS Operations Agent - an AI assistant specialized in
managing and troubleshooting Amazon EKS Kubernetes clusters.

Your capabilities include:
- Answering questions about Kubernetes and EKS
- Helping diagnose cluster issues
- Providing guidance on best practices

In future phases, you will be able to:
- Query cluster resources using EKS MCP Server tools
- Check cluster upgrade readiness
- Debug deployment issues
- Monitor inference endpoint health
- Remember context from previous conversations

Always be helpful, accurate, and concise in your responses.
"""


# --- Graph Nodes ---


def agent_node(state: AgentState) -> dict:
    """
    Main agent node - invokes the LLM with current messages.

    In Phase 2, this will be expanded to a multi-node workflow:
    - check_memory: Query long-term memory for context
    - gather_info: Call EKS MCP tools to collect cluster data
    - analyze: Process gathered information
    - diagnose: Identify root cause
    - recommend_fix: Suggest remediation
    - apply_fix: Execute the fix (with human approval)
    - update_memory: Store learnings for future sessions
    """
    llm = get_llm()

    # Prepend system prompt to messages
    messages = [("system", SYSTEM_PROMPT)] + state["messages"]

    response = llm.invoke(messages)
    return {"messages": [response]}


# --- Graph Construction ---


def create_agent_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None
) -> StateGraph:
    """
    Create the LangGraph agent graph.

    Args:
        checkpointer: Optional checkpointer for session persistence.
                     When running via KAgentApp, this is KAgentCheckpointer.
                     For local testing, can be MemorySaver or None.

    Returns:
        Compiled LangGraph graph.
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("agent", agent_node)

    # Add edges
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    # Compile with optional checkpointer
    return builder.compile(checkpointer=checkpointer)


# --- Convenience Functions for Local Testing ---


def invoke(message: str, thread_id: str = "default") -> str:
    """
    Invoke the agent with a user message (for local testing).

    Args:
        message: User's input message
        thread_id: Conversation thread ID for state persistence

    Returns:
        Agent's response as a string
    """
    graph = create_agent_graph()
    result = graph.invoke(
        {"messages": [("user", message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


# --- CLI for local testing ---

if __name__ == "__main__":
    print("EKS Ops Agent - Phase 1 (Barebone)")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = invoke(user_input)
        print(f"Agent: {response}\n")
