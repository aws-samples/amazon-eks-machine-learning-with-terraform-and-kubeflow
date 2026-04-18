"""
Triage Agent Memory - Memledger-backed memory for incident triage.

Provides:
- Alert ingestion with automatic correlation against past incidents
- Cross-agent search in /ops/* namespace
- Escalation writing to /shared/escalations/
- Standard memory tools (remember, recall, audit, lineage)

Namespaces:
    /triage/alerts      — ingested alerts with correlation data
    /triage/findings    — investigation findings
    /ops/incidents      — read-only search of ops incident history
    /shared/escalations — cross-agent escalation records
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.tools import tool

from memledger import Memledger, RecordType
from memledger.models import EmbeddingConfig

logger = logging.getLogger(__name__)

# Global memory service instance (set by app.py)
_memory_service: Optional["MemoryService"] = None

AGENT_ID = "triage-agent"


def set_memory_service(service: "MemoryService") -> None:
    """Set the global memory service instance."""
    global _memory_service
    _memory_service = service


class MemoryService:
    """
    Memledger-backed memory service for the Triage Agent.

    Supports two initialization modes:
    - Direct: pg_connection_string -> pgvector backend
    - Config file: config_path -> any backend or composition
    """

    def __init__(
        self,
        pg_connection_string: str = "",
        embedding_provider: str = "bedrock",
        embedding_model: str = "amazon.titan-embed-text-v2:0",
        embedding_dimensions: int = 1024,
        config_path: str = "",
    ):
        self._pg_connection_string = pg_connection_string
        self._config_path = config_path
        self._embedding_config = EmbeddingConfig(
            provider=embedding_provider,
            model=embedding_model,
            dimensions=embedding_dimensions,
        )
        self._ml: Optional[Memledger] = None

    async def _get_ml(self) -> Memledger:
        """Lazy-initialize memledger connection on first use."""
        if self._ml is None:
            if self._config_path:
                self._ml = await Memledger.from_config(self._config_path)
                logger.info("MemoryService initialized with memledger (config: %s)", self._config_path)
            else:
                self._ml = await Memledger.create(
                    backend_name="pgvector",
                    connection_string=self._pg_connection_string,
                    embedding_config=self._embedding_config,
                )
                logger.info("MemoryService initialized with memledger (pgvector)")
        return self._ml

    async def remember(
        self,
        content: str,
        namespace: str = "/triage/alerts",
        record_type: RecordType = RecordType.EPISODIC,
        metadata: Optional[dict[str, Any]] = None,
        **typed_fields: Any,
    ) -> str:
        """Store a memory for future recall. Returns the record ID."""
        ml = await self._get_ml()

        return await ml.add(
            content=content,
            record_type=record_type,
            namespace=namespace,
            metadata=metadata or {},
            **typed_fields,
        )

    async def recall(
        self,
        query: str,
        namespace: Optional[str] = None,
        top_k: int = 5,
        record_type: Optional[RecordType] = None,
    ):
        """Search memories by semantic similarity."""
        ml = await self._get_ml()

        return await ml.search(
            query=query,
            namespace=namespace,
            top_k=top_k,
            record_type=record_type,
        )

    async def close(self) -> None:
        """Close memledger connection."""
        if self._ml:
            await self._ml.close()
            self._ml = None


# --- Triage-Specific Memory Tools ---


@tool
async def ingest_alert(
    source: str,
    severity: str,
    description: str,
) -> str:
    """
    Ingest an incoming alert, correlate it with past incidents, and store it.

    This is the primary entry point for alert processing. It:
    1. Searches /ops/incidents/* for similar past incidents
    2. Calculates importance based on severity and correlation strength
    3. Writes an episodic record to /triage/alerts/

    Args:
        source: Alert source (e.g. "CloudWatch", "PagerDuty", "Prometheus", "manual")
        severity: Alert severity — "critical", "high", "medium", "low"
        description: Full alert description including any error messages, affected resources

    Returns:
        Ingestion summary with correlation results and assigned importance
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_ml()
        alert_id = f"alert-{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step 1: Correlate with past incidents in ops namespace
        correlations = await ml.search(
            query=description,
            namespace="/ops/incidents",
            top_k=3,
        )

        correlation_summary = []
        max_correlation_score = 0.0
        if correlations.records:
            for rec in correlations.records:
                score = rec.score if rec.score else 0.0
                max_correlation_score = max(max_correlation_score, score)
                correlation_summary.append(
                    f"  - [{rec.id[:8]}] (score: {score:.3f}): {rec.content[:100]}"
                )

        # Also search /triage/alerts for recent similar alerts
        recent_alerts = await ml.search(
            query=description,
            namespace="/triage/alerts",
            top_k=2,
        )
        if recent_alerts.records:
            for rec in recent_alerts.records:
                score = rec.score if rec.score else 0.0
                correlation_summary.append(
                    f"  - [triage/{rec.id[:8]}] (score: {score:.3f}): {rec.content[:80]}"
                )

        # Step 2: Calculate importance based on severity and correlation
        severity_scores = {
            "critical": 0.6,
            "high": 0.5,
            "medium": 0.4,
            "low": 0.3,
        }
        base_importance = severity_scores.get(severity.lower(), 0.4)

        # Boost importance if we found strong correlations (known pattern)
        if max_correlation_score > 0.8:
            importance = min(base_importance + 0.1, 0.6)
        elif max_correlation_score > 0.6:
            importance = base_importance
        else:
            # No strong correlation — novel alert, keep importance conservative
            importance = max(base_importance - 0.05, 0.3)

        # Step 3: Store the alert record
        metadata = {
            "alert_id": alert_id,
            "source": source,
            "severity": severity.lower(),
            "ingested_at": timestamp,
            "created_by": AGENT_ID,
            "correlation_score": round(max_correlation_score, 3),
            "importance": round(importance, 2),
        }

        content = f"[{severity.upper()}] Alert from {source}: {description}"

        record_id = await _memory_service.remember(
            content=content,
            namespace="/triage/alerts",
            record_type=RecordType.EPISODIC,
            metadata=metadata,
            importance=importance,
            created_by=AGENT_ID,
        )

        # Build response
        lines = [
            f"Alert ingested: {alert_id}",
            f"  Record ID: {record_id[:8]}",
            f"  Severity: {severity.upper()}",
            f"  Source: {source}",
            f"  Importance: {importance:.2f}",
            f"  Correlation score: {max_correlation_score:.3f}",
        ]

        if correlation_summary:
            lines.append(f"  Correlated incidents ({len(correlation_summary)}):")
            lines.extend(correlation_summary)
        else:
            lines.append("  No correlated incidents found — this may be a novel alert.")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to ingest alert: {e}")
        return f"Failed to ingest alert: {str(e)}"


@tool
async def correlate_incidents(
    query: str,
    namespace: str = "/ops/incidents",
    top_k: int = 5,
) -> str:
    """
    Search the ops namespace for incidents matching a query.

    Cross-agent search specifically targeting the ops team's incident records.
    Use this to find known patterns, past resolutions, and related incidents.

    Args:
        query: What to search for (natural language description of the incident pattern)
        namespace: Namespace to search in (default: /ops/incidents, can also search /ops/*, /shared/*)
        top_k: Number of results to return

    Returns:
        Matching incidents with confidence signals, or "No matches found"
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_ml()

        results = await ml.search_hybrid(
            query=query,
            namespace=namespace,
            top_k=top_k,
        )

        if not results.records:
            return f"No matching incidents found in {namespace}."

        lines = [f"Found {len(results.records)} incidents in {namespace} (search took {results.search_time_ms}ms):"]
        for i, rec in enumerate(results.records, 1):
            score_str = f" (score: {rec.score:.3f})" if rec.score else ""
            type_str = f" [{rec.record_type.value}]"
            meta_str = ""
            if rec.metadata:
                severity = rec.metadata.get("severity", "")
                created_by = rec.metadata.get("created_by", "")
                if severity:
                    meta_str += f" | severity: {severity}"
                if created_by:
                    meta_str += f" | by: {created_by}"
            lines.append(
                f"  {i}. [{rec.namespace}]{type_str}{score_str}: {rec.content[:120]}{meta_str}"
            )

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to correlate incidents: {e}")
        return f"Failed to search incidents: {str(e)}"


@tool
async def escalate_to_ops(
    summary: str,
    alert_id: str,
    severity: str,
) -> str:
    """
    Escalate an alert to the ops team by writing to the shared escalations namespace.

    Use this when:
    - Severity is critical or high and no clear correlation was found
    - Correlation is ambiguous and human judgment is needed
    - The alert pattern is novel and requires investigation

    Args:
        summary: Triage summary — what was found, what's unclear, recommended next steps
        alert_id: The alert ID from ingest_alert (e.g. "alert-abc123def456")
        severity: Escalation severity — "critical", "high", "medium"

    Returns:
        Confirmation with escalation record ID
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        escalation_id = f"esc-{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        metadata = {
            "escalation_id": escalation_id,
            "triggered_by": alert_id,
            "severity": severity.lower(),
            "created_by": AGENT_ID,
            "escalated_at": timestamp,
            "status": "open",
        }

        content = f"[ESCALATION][{severity.upper()}] {summary} (triggered by {alert_id})"

        record_id = await _memory_service.remember(
            content=content,
            namespace="/shared/escalations",
            record_type=RecordType.EPISODIC,
            metadata=metadata,
            importance=0.7,  # escalations are high importance by definition
            created_by=AGENT_ID,
        )

        return (
            f"Escalated to ops team:\n"
            f"  Escalation ID: {escalation_id}\n"
            f"  Record ID: {record_id[:8]}\n"
            f"  Severity: {severity.upper()}\n"
            f"  Triggered by: {alert_id}\n"
            f"  Namespace: /shared/escalations\n"
            f"  Status: OPEN — awaiting ops review"
        )

    except Exception as e:
        logger.error(f"Failed to escalate: {e}")
        return f"Failed to escalate: {str(e)}"


# --- Standard Memory Tools (adapted for triage agent) ---


@tool
async def remember_knowledge(
    content: str,
    namespace: str = "/triage/alerts",
    record_type: str = "episodic",
    metadata: Optional[dict] = None,
    importance: float = 0.45,
    replaces: Optional[str] = None,
) -> str:
    """
    Store triage findings in long-term memory for future recall.

    Use this after completing alert analysis, discovering a pattern, or
    documenting investigation findings. Default importance is 0.45 (conservative)
    — adjust based on confidence in the finding.

    Args:
        content: What to remember. Be specific and include context.
            Good: "OOM alerts from payment-service correlate with HikariCP pool leak. 3 incidents in past week."
            Bad: "Alert was about memory"
        namespace: Category path for organizing memories.
            /triage/alerts — alert records (default)
            /triage/findings — investigation conclusions
            /shared/knowledge — cross-agent knowledge
        record_type: Type of memory - 'episodic' (events, default), 'semantic' (facts), 'procedural' (how-to)
        metadata: Optional key-value pairs (e.g. {"severity": "high", "cluster": "eks-prod"})
        importance: Confidence/importance score (0.0-1.0). Default 0.45. Use lower values when unsure.
        replaces: Description of the old memory this one supersedes.

    Returns:
        Confirmation with the stored memory ID
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_ml()
        rt = RecordType(record_type)

        # Resolve supersedes ID from description
        supersedes_id = None
        if replaces:
            results = await ml.search(query=replaces, top_k=1)
            if results.records:
                supersedes_id = results.records[0].id

        record_id = await _memory_service.remember(
            content=content,
            namespace=namespace,
            record_type=rt,
            metadata=metadata or {},
            importance=importance,
            created_by=AGENT_ID,
            supersedes=supersedes_id,
        )

        msg = f"Stored {record_type} memory [{record_id[:8]}] (importance: {importance:.2f}): {content[:100]}"
        if supersedes_id:
            msg += f"\n(Superseded old memory [{supersedes_id[:8]}] — marked as deprecated)"

        # Check for conflict detection
        conflict_entries = ml.audit.by_record(record_id)
        for entry in conflict_entries:
            if entry.operation == "conflict_detected":
                msg += (
                    f"\nWarning: Potential conflict with existing memory "
                    f"[{entry.details.get('existing_id', '?')[:8]}] "
                    f"(similarity: {entry.details.get('similarity', '?')})"
                )
                break

        return msg

    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        return f"Failed to store memory: {str(e)}"


@tool
async def recall_knowledge(
    query: str,
    namespace: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """
    Search long-term memory for relevant past knowledge.

    Use this BEFORE starting any investigation to check if similar alerts
    or incidents have been seen before. Searches across all accessible namespaces.

    Args:
        query: What to search for (natural language description)
        namespace: Optional scope to narrow search
            /triage/alerts — search triage history
            /ops/incidents — search ops incident history
            /shared/* — search shared knowledge
            None — search everything
        top_k: Number of results to return

    Returns:
        Matching memories with relevance scores, or "No relevant memories found"
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_ml()
        results = await ml.search_hybrid(
            query=query,
            namespace=namespace,
            top_k=top_k,
        )

        if not results.records:
            return "No relevant memories found."

        lines = []
        for i, rec in enumerate(results.records, 1):
            score_str = f" (score: {rec.score:.3f})" if rec.score else ""
            type_str = f" [{rec.record_type.value}]"
            meta_str = f" | metadata: {rec.metadata}" if rec.metadata else ""
            lines.append(
                f"{i}. [{rec.namespace}]{type_str}{score_str}: {rec.content}{meta_str}"
            )

        return (
            f"Found {len(results.records)} memories "
            f"(search took {results.search_time_ms}ms):\n"
            + "\n".join(lines)
        )

    except Exception as e:
        logger.error(f"Failed to recall memories: {e}")
        return f"Failed to search memory: {str(e)}"


@tool
async def recall_context(
    query: str,
    namespace: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """
    Comprehensive memory search: find similar episodes, suggest procedures, and flag known failures.

    Use this for complex situations where you need the full picture. More thorough
    than recall_knowledge — groups results by type.

    Args:
        query: What situation you're dealing with (natural language)
        namespace: Optional scope to narrow search
        top_k: Number of results per category

    Returns:
        Grouped results: similar episodes, suggested procedures, known failures
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_ml()
        context = await ml.recall_context(
            query=query,
            namespace=namespace,
            top_k=top_k,
        )
        return context.summary()

    except Exception as e:
        logger.error(f"Failed contextual recall: {e}")
        return f"Failed to search memory: {str(e)}"


@tool
async def mark_memory_outcome(
    description: str,
    success: bool,
    record_id: Optional[str] = None,
) -> str:
    """
    Record whether a recalled memory or procedure was helpful or not.

    Use this when the user or ops team reports that a previously recalled
    memory was accurate (success) or misleading (failure).

    Args:
        description: What the memory was about (e.g. "OOM correlation with pool leak")
        success: True if it was helpful, False if it was misleading
        record_id: Optional exact ID if known

    Returns:
        Confirmation message
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_ml()

        result = await ml.record_outcome(
            record_id=record_id,
            success=success,
            description=description,
        )

        if not result:
            return f"Could not find a memory matching '{description}' to record outcome."

        outcome = "successful" if success else "unsuccessful"
        return f"Recorded outcome as {outcome} for memory matching: {description[:80]}"

    except Exception as e:
        logger.error(f"Failed to record outcome: {e}")
        return f"Failed to record outcome: {str(e)}"


@tool
async def memory_stats(
    namespace: Optional[str] = None,
) -> str:
    """
    Get statistics about what's stored in memory.

    Use this to understand what knowledge domains are populated before searching.

    Args:
        namespace: Optional scope (e.g. "/triage/alerts" to see alert stats only)

    Returns:
        Summary of stored memories: counts by type, namespaces, success rates
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_ml()
        stats = await ml.stats(namespace=namespace)

        if not stats or stats.get("total_count", 0) == 0:
            scope = f" in namespace '{namespace}'" if namespace else ""
            return f"No memories stored{scope}."

        lines = [f"Memory statistics{' for ' + namespace if namespace else ''}:"]
        lines.append(f"  Total memories: {stats['total_count']}")

        if stats.get("by_record_type"):
            type_parts = [f"{t}: {c}" for t, c in stats["by_record_type"].items()]
            lines.append(f"  By type: {', '.join(type_parts)}")

        if stats.get("namespaces"):
            lines.append(f"  Namespaces: {', '.join(stats['namespaces'][:10])}")

        if stats.get("avg_success_rate") is not None:
            lines.append(f"  Average success rate: {stats['avg_success_rate']:.1%}")

        if stats.get("oldest_memory"):
            lines.append(f"  Oldest: {stats['oldest_memory']}")
        if stats.get("newest_memory"):
            lines.append(f"  Newest: {stats['newest_memory']}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        return f"Failed to get stats: {str(e)}"


@tool
async def memory_audit(
    record_id: Optional[str] = None,
    last_n: int = 10,
) -> str:
    """
    Show the audit trail of memory operations.

    Use this to understand how knowledge evolved or for compliance trail.

    Args:
        record_id: Optional — show audit trail for a specific memory only
        last_n: Number of recent entries to show (default 10)

    Returns:
        Formatted audit log with timestamps and operations
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_ml()

        if record_id:
            entries = ml.audit.by_record(record_id)
            if not entries:
                return f"No audit entries found for memory {record_id[:8]}."
            lines = [f"Audit trail for memory {record_id[:8]} ({len(entries)} entries):"]
        else:
            entries = ml.audit.recent(last_n)
            if not entries:
                return "No audit entries yet."
            lines = [f"Memory audit log (last {len(entries)} operations):"]

        for i, entry in enumerate(entries, 1):
            lines.append(f"  {i}. {entry.summary()}")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to get audit log: {e}")
        return f"Failed to get audit log: {str(e)}"


@tool
async def memory_lineage(
    description: str,
    record_id: Optional[str] = None,
) -> str:
    """
    Trace the provenance and lineage of a memory — who created it, who used it,
    what it superseded, and what confidence level it has.

    Args:
        description: What the memory is about (e.g. "OOM alert correlation")
        record_id: Optional exact ID if known

    Returns:
        Provenance chain showing creation, usage, supersession, and confidence
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_ml()
        lineage = await ml.get_lineage(
            record_id=record_id,
            description=description,
        )

        if "error" in lineage:
            return f"Could not find memory: {lineage['error']}"

        rec = lineage["record"]
        lines = [f"Memory Lineage for [{rec['id'][:8]}]:"]
        lines.append(f"  Content: {rec['content']}")
        lines.append(f"  Type: {rec['record_type']} | Status: {rec['status']}")
        lines.append(f"  Created: {rec.get('created_at', 'unknown')}")
        lines.append(f"  Confidence: {lineage.get('confidence', 'N/A')}")

        if lineage.get("created_by"):
            lines.append(f"  Created by: {lineage['created_by']}")
        if lineage.get("workflow_id"):
            lines.append(f"  Workflow: {lineage['workflow_id']}")
        if lineage.get("triggered_by"):
            lines.append(f"  Triggered by: {lineage['triggered_by']}")
        if lineage.get("accessed_by"):
            lines.append(f"  Accessed by: {', '.join(lineage['accessed_by'])}")

        if lineage.get("supersedes_chain"):
            lines.append(f"  Supersedes chain ({len(lineage['supersedes_chain'])} predecessors):")
            for pred in lineage["supersedes_chain"]:
                lines.append(f"    <- [{pred['id'][:8]}] {pred['content'][:60]} ({pred['status']})")

        if lineage.get("derived_records"):
            lines.append(f"  Derived records ({len(lineage['derived_records'])}):")
            for derived in lineage["derived_records"]:
                lines.append(f"    -> [{derived['id'][:8]}] {derived['content'][:60]}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to get lineage: {e}")
        return f"Failed to get lineage: {str(e)}"


def get_memory_tools() -> list:
    """Get the list of memory tools for the triage agent."""
    return [
        # Triage-specific tools
        ingest_alert,
        correlate_incidents,
        escalate_to_ops,
        # Standard memory tools
        remember_knowledge,
        recall_knowledge,
        recall_context,
        mark_memory_outcome,
        memory_stats,
        memory_audit,
        memory_lineage,
    ]
