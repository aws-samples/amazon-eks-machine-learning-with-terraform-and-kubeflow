"""
EKS Ops Agent - LangGraph agent with Bedrock Claude.

Module 1: Simple agent that can answer questions.
Module 2: Add EKS MCP Server tools for cluster operations.
Module 3: Add memory (short-term via checkpointer, long-term via Redis).
Module 4: Add Langfuse observability.
"""

import logging
from typing import Annotated, Literal, Optional, TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from config import config

logger = logging.getLogger(__name__)


# --- State Definition ---


class AgentState(TypedDict):
    """State passed between nodes in the graph."""

    messages: Annotated[list, add_messages]


# --- LLM Setup ---


def get_llm(tools: list = None) -> ChatBedrockConverse:
    """
    Create Bedrock Claude LLM instance using Converse API.

    Uses ChatBedrockConverse which handles tool results differently
    than the legacy InvokeModel API used by ChatBedrock.

    Args:
        tools: Optional list of tools to bind to the LLM.

    Returns:
        ChatBedrockConverse instance, optionally with tools bound.
    """
    llm = ChatBedrockConverse(
        model=config.BEDROCK_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0,
        max_tokens=4096,
    )

    if tools:
        return llm.bind_tools(tools)
    return llm


# --- System Prompts ---

SYSTEM_PROMPT_BASE = """You are an EKS Operations Agent - an AI assistant specialized in
managing and troubleshooting Amazon EKS Kubernetes clusters.

Your capabilities include:
- Answering questions about Kubernetes and EKS
- Helping diagnose cluster issues
- Providing guidance on best practices

Always be helpful, accurate, and concise in your responses."""

# Original system prompt (kept for reference)
# SYSTEM_PROMPT_WITH_TOOLS = """You are an EKS Operations Agent - an AI assistant specialized in
# managing and troubleshooting Amazon EKS Kubernetes clusters.
#
# You have access to EKS MCP Server tools that allow you to:
# - Query and manage Kubernetes resources (pods, deployments, services, etc.)
# - Get pod logs and cluster events
# - Apply YAML manifests
# - Retrieve CloudWatch logs and metrics
# - Search troubleshooting guides
# - Get EKS cluster insights and recommendations
#
# When a user asks about their cluster, USE THE TOOLS to get real data.
# Don't just give generic advice - investigate the actual cluster state.
#
# Guidelines:
# 1. For troubleshooting: First get relevant logs/events, then diagnose
# 2. For status checks: Use list_k8s_resources to get current state
# 3. For debugging pods: Use get_pod_logs and get_k8s_events
# 4. For cluster health: Use get_eks_insights for recommendations
#
# Always be helpful, accurate, and concise in your responses."""

SYSTEM_PROMPT_WITH_TOOLS = """You are an EKS Operations Agent - an AI assistant specialized in
managing and troubleshooting Amazon EKS Kubernetes clusters.

You have access to EKS MCP Server tools that allow you to:
- Query and manage Kubernetes resources (pods, deployments, services, etc.)
- Get pod logs and cluster events
- Apply YAML manifests
- Retrieve CloudWatch logs and metrics
- Search troubleshooting guides
- Get EKS cluster insights and recommendations

## ReAct Reasoning Pattern

For each step, follow this pattern:

**Thought**: Explain your reasoning about what you need to do next and why.
**Action**: Call the appropriate tool to gather information or take action.
**Observation**: Analyze the tool result and determine next steps.

Always think out loud before taking action. This helps users understand your reasoning.

## Guidelines

1. **Troubleshooting**: First explain what you're investigating, then get logs/events, then diagnose
2. **Status checks**: Explain what resources you're checking, then use list_k8s_resources
3. **Debugging pods**: Describe your debugging approach, then use get_pod_logs and get_k8s_events
4. **Cluster health**: Explain what insights you're looking for, then use get_eks_insights

When a user asks about their cluster, USE THE TOOLS to get real data.
Don't just give generic advice - investigate the actual cluster state.

Always be helpful, accurate, and concise in your responses."""


# --- Graph Nodes ---


def create_agent_node(llm):
    """Create the agent node function with the given LLM."""

    def agent_node(state: AgentState) -> dict:
        """
        Main agent node - invokes the LLM with current messages.

        The LLM may respond directly or request tool calls.
        """
        system_prompt = SYSTEM_PROMPT_WITH_TOOLS if hasattr(llm, 'bound_tools') else SYSTEM_PROMPT_BASE
        messages = [("system", system_prompt)] + state["messages"]

        response = llm.invoke(messages)
        return {"messages": [response]}

    return agent_node


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Determine if the agent should continue to tools or end.

    Returns:
        "tools" if the last message has tool calls, "end" otherwise.
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"


# --- Graph Construction ---


def create_agent_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    tools: list = None,
) -> StateGraph:
    """
    Create the LangGraph agent graph.

    Args:
        checkpointer: Optional checkpointer for session persistence.
                     When running via KAgentApp, this is KAgentCheckpointer.
                     For local testing, can be MemorySaver or None.
        tools: Optional list of tools for the agent to use.
               When provided, creates a ReAct-style tool-using agent.

    Returns:
        Compiled LangGraph graph.
    """
    # Create LLM with optional tools
    llm = get_llm(tools=tools)

    # Build graph
    builder = StateGraph(AgentState)

    # Add agent node
    builder.add_node("agent", create_agent_node(llm))

    if tools:
        # Module 2: ReAct pattern with tools
        logger.info(f"Creating agent with {len(tools)} tools")

        # Add tool node
        builder.add_node("tools", ToolNode(tools))

        # Add edges with conditional routing
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END,
            }
        )
        builder.add_edge("tools", "agent")  # Loop back after tool execution
    else:
        # Module 1: Simple Q&A agent
        logger.info("Creating agent without tools (Q&A mode)")
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)

    # Compile with optional checkpointer
    return builder.compile(checkpointer=checkpointer)


# --- Convenience Functions for Local Testing ---


def invoke(message: str, thread_id: str = "default", tools: list = None) -> str:
    """
    Invoke the agent with a user message (for local testing).

    Args:
        message: User's input message
        thread_id: Conversation thread ID for state persistence
        tools: Optional list of tools

    Returns:
        Agent's response as a string
    """
    graph = create_agent_graph(tools=tools)
    result = graph.invoke(
        {"messages": [("user", message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


# --- CLI for local testing ---

if __name__ == "__main__":
    print("EKS Ops Agent - Module 1 (Barebone)")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = invoke(user_input)
        print(f"Agent: {response}\n")
