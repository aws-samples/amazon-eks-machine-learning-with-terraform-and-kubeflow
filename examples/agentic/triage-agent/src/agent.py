"""
Triage Agent - LangGraph ReAct agent with Bedrock Claude.

First-responder agent for incident triage: ingests alerts, correlates with
past incidents in shared memory, writes investigation findings with honest
confidence levels.
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

SYSTEM_PROMPT_BASE = """You are a Triage Agent responsible for initial incident assessment and alert correlation.
When you receive an alert, always search shared memory first for similar past incidents.
Write your findings with honest confidence levels — if unsure, set importance below 0.5.
Escalate to the ops team when correlation is unclear or severity is high.
You have access to your own namespace /triage/* and can read /ops/* and /shared/*."""

SYSTEM_PROMPT_WITH_TOOLS = """You are a Triage Agent responsible for initial incident assessment and alert correlation.
When you receive an alert, always search shared memory first for similar past incidents.
Write your findings with honest confidence levels — if unsure, set importance below 0.5.
Escalate to the ops team when correlation is unclear or severity is high.
You have access to your own namespace /triage/* and can read /ops/* and /shared/*.

IMPORTANT — Memory-first workflow:
- When you receive an alert, ALWAYS call ingest_alert first to search for correlations.
- Before investigating, call recall_context to check for similar past incidents and proven procedures.
- Use correlate_incidents to search the ops namespace for known incident patterns.
- When correlation is unclear or severity is critical/high, call escalate_to_ops.
- When recording findings, use remember_knowledge with honest importance (0.3-0.6 range).
- When the user reports a procedure worked or failed, call mark_memory_outcome to track the result."""


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
        for tool_call in last_message.tool_calls:
            logger.info(f"TOOL_CALL: {tool_call['name']} | Args: {tool_call['args']}")
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
        tools: Optional list of tools for the agent to use.

    Returns:
        Compiled LangGraph graph.
    """
    llm = get_llm(tools=tools)

    builder = StateGraph(AgentState)
    builder.add_node("agent", create_agent_node(llm))

    if tools:
        logger.info(f"Creating triage agent with {len(tools)} tools")
        builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END,
            }
        )
        builder.add_edge("tools", "agent")
    else:
        logger.info("Creating triage agent without tools (Q&A mode)")
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)

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
    print("Triage Agent - Local Testing")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = invoke(user_input)
        print(f"Agent: {response}\n")
