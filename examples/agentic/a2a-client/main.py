"""
A2A Client - Simple agent that hands off EKS issues to eks-ops-agent.

This demonstrates Agent-to-Agent (A2A) communication where a customer care
agent detects infrastructure issues and hands off to a specialized EKS agent.

Usage:
    # First, port-forward the eks-ops-agent service
    kubectl port-forward -n kagent svc/eks-ops-agent 8081:8080

    # Run the client
    uv run python main.py
"""

import httpx
import json
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

# Default configuration
EKS_OPS_AGENT_URL = "http://localhost:8081"

# Sample tickets that Agent1 handles
SAMPLE_TICKETS = [
    {
        "id": "TICKET-001",
        "type": "password_reset",
        "description": "Customer forgot their password",
        "resolution": "Sent password reset link to registered email."
    },
    {
        "id": "TICKET-002",
        "type": "billing",
        "description": "Customer charged twice for subscription",
        "resolution": "Refund processed. Duplicate charge was a payment gateway error."
    },
    {
        "id": "TICKET-003",
        "type": "eks_infrastructure",
        "description": "Customer reports their ML training jobs are failing with OOMKilled errors",
        "namespace": "kubeflow-user-example-com",
        "symptoms": [
            "Pods restarting frequently",
            "Jobs failing after 10-15 minutes",
            "Error: OOMKilled in pod events"
        ]
    }
]


def handle_simple_ticket(ticket: dict) -> str:
    """Handle simple tickets that don't need EKS investigation."""
    return ticket.get("resolution", "Ticket resolved.")


async def handoff_to_eks_ops_agent(ticket: dict) -> str:
    """
    Hand off EKS-related issues to eks-ops-agent via A2A protocol.

    Passes context in the message so Agent2 can investigate.
    """
    # Build context message for Agent2
    namespace = ticket.get("namespace", "default")
    symptoms = ticket.get("symptoms", [])
    symptoms_text = "\n".join(f"- {s}" for s in symptoms)

    handoff_message = f"""I'm a customer care agent handing off an infrastructure issue.

**Ticket:** {ticket['id']}
**Issue:** {ticket['description']}

**Symptoms reported:**
{symptoms_text}

**Investigation needed:**
Please check the pods in namespace `{namespace}` for issues.
Look at pod status, events, logs, and resource limits.
Provide a summary of what you find and recommended actions.

Note: Discover the cluster name using your tools if needed."""

    console.print(Panel(
        f"[yellow]Handing off to EKS Ops Agent...[/yellow]\n\n{handoff_message}",
        title="A2A Handoff",
        border_style="yellow"
    ))

    # A2A protocol request (using message/send method)
    request_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    a2a_request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "message/send",
        "params": {
            "message": {
                "messageId": message_id,
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": handoff_message
                    }
                ]
            }
        }
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                EKS_OPS_AGENT_URL,
                json=a2a_request,
                headers={"Content-Type": "application/json"}
            )

            # Fire and forget - just confirm the request was sent
            if response.status_code == 200:
                return "Handoff sent successfully. Agent2 is now investigating."
            else:
                return f"Handoff sent (status: {response.status_code})"

    except httpx.ConnectError:
        return (
            "Could not connect to eks-ops-agent. Make sure you have port-forwarded:\n"
            "kubectl port-forward -n kagent svc/eks-ops-agent 8081:8080"
        )
    except httpx.TimeoutException:
        # Timeout is fine - agent2 is processing, we don't wait
        return "Handoff sent. Agent2 is processing (response may take time)."
    except Exception as e:
        return f"Error during A2A handoff: {str(e)}"


async def process_tickets():
    """Process sample tickets, handing off EKS issues to Agent2."""

    console.print(Panel(
        "[bold]Customer Care Agent (Agent1)[/bold]\n\n"
        "Processing support tickets...\n"
        "EKS infrastructure issues will be handed off to eks-ops-agent.",
        title="Agent1",
        border_style="blue"
    ))

    for ticket in SAMPLE_TICKETS:
        console.print(f"\n[bold cyan]Processing {ticket['id']}:[/bold cyan] {ticket['description']}")

        if ticket["type"] == "eks_infrastructure":
            # Hand off to Agent2
            console.print("[yellow]→ EKS issue detected, initiating A2A handoff...[/yellow]")
            response = await handoff_to_eks_ops_agent(ticket)

            console.print(Panel(
                Markdown(response),
                title="EKS Ops Agent Response",
                border_style="green"
            ))
        else:
            # Handle simple ticket locally
            resolution = handle_simple_ticket(ticket)
            console.print(f"[green]✓ Resolved:[/green] {resolution}")


def main():
    """Entry point."""
    import asyncio
    asyncio.run(process_tickets())


if __name__ == "__main__":
    main()
