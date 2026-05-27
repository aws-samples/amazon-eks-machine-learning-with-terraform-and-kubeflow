"""
Module 3: Long-term Memory with Memledger

This module provides persistent, searchable memory for the EKS Ops Agent
using memledger backed by PostgreSQL+pgvector (Aurora or in-cluster Postgres).

Two categories of memory:
1. User defaults — key-value storage for cluster/namespace preferences
   (backward-compatible with the original Redis implementation)
2. Semantic memory — vector-indexed knowledge that the agent can search
   (incidents, runbooks, operational learnings)

Memledger schema:
    Namespace: /users/{user_id}/defaults  — user preferences
    Namespace: /incidents/{cluster}       — incident history
    Namespace: /runbooks                  — operational procedures
    Namespace: /learnings                 — agent's accumulated knowledge

Usage:
    memory = MemoryService(pg_connection_string="postgresql://...")
    await memory.initialize()
    await memory.set_defaults(user_id, cluster="eks-1", namespace="default")
    defaults = await memory.get_defaults(user_id)
    await memory.remember("OOM kills caused by pool leak", namespace="/incidents/eks-1")
    results = await memory.recall("memory issues", namespace="/incidents")
"""

import logging
from typing import Any, Optional

from langchain_core.tools import tool

from memledger import Memledger, RecordType
from memledger.models import EmbeddingConfig

logger = logging.getLogger(__name__)

# Global memory service instance (set by app.py)
_memory_service: Optional["MemoryService"] = None

# Default user ID when not available from session
DEFAULT_USER_ID = "default"


def set_memory_service(service: "MemoryService") -> None:
    """Set the global memory service instance."""
    global _memory_service
    _memory_service = service


class UserDefaults:
    """User's default settings for cluster operations."""

    def __init__(self, cluster: Optional[str] = None, namespace: Optional[str] = None):
        self.cluster = cluster
        self.namespace = namespace

    def to_dict(self) -> dict:
        return {"cluster": self.cluster, "namespace": self.namespace}

    @classmethod
    def from_dict(cls, data: dict) -> "UserDefaults":
        return cls(cluster=data.get("cluster"), namespace=data.get("namespace"))

    def __str__(self) -> str:
        parts = []
        if self.cluster:
            parts.append(f"cluster={self.cluster}")
        if self.namespace:
            parts.append(f"namespace={self.namespace}")
        return ", ".join(parts) if parts else "no defaults set"


class MemoryService:
    """
    Memledger-backed memory service for the EKS Ops Agent.

    Provides:
    - User defaults (cluster/namespace preferences) via memledger records
    - Semantic memory (searchable operational knowledge) via vector search

    Supports two initialization modes:
    - Direct: pg_connection_string → pgvector backend (default)
    - Config file: config_path → any backend or composition (for multi-backend)
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

    async def _get_memledger(self) -> Memledger:
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
            try:
                from memledger.telemetry import instrument
                instrument(self._ml)
                logger.info("memledger OTEL instrumentation enabled")
            except Exception as e:
                logger.warning(f"memledger OTEL instrumentation not available: {e}")
        return self._ml

    def _defaults_ns(self, user_id: str) -> str:
        """Namespace for user defaults."""
        return f"/users/{user_id}/defaults"

    async def get_defaults(self, user_id: str) -> UserDefaults:
        """Retrieve user's default settings from memledger."""
        memledger = await self._get_memledger()

        try:
            results = await memledger.search(
                query="user default cluster namespace",
                namespace=self._defaults_ns(user_id),
                top_k=1,
            )

            if results.records:
                metadata = results.records[0].metadata
                defaults = UserDefaults.from_dict(metadata)
                logger.info(f"Retrieved defaults for user {user_id}: {defaults}")
                return defaults

            logger.debug(f"No defaults found for user {user_id}")
            return UserDefaults()

        except Exception as e:
            logger.warning(f"Failed to get defaults for user {user_id}: {e}")
            return UserDefaults()

    async def set_defaults(
        self,
        user_id: str,
        cluster: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> UserDefaults:
        """Save user's default settings to memledger."""
        memledger = await self._get_memledger()

        try:
            # Get existing defaults and merge
            existing = await self.get_defaults(user_id)

            if cluster is not None:
                existing.cluster = cluster
            if namespace is not None:
                existing.namespace = namespace

            # Use a stable record ID so we overwrite rather than duplicate
            record_id = f"defaults-{user_id}"
            content = f"User defaults: {existing}"

            await memledger.add(
                content=content,
                record_type=RecordType.SEMANTIC,
                namespace=self._defaults_ns(user_id),
                metadata=existing.to_dict(),
                record_id=record_id,
                user_id=user_id,
            )

            logger.info(f"Saved defaults for user {user_id}: {existing}")
            return existing

        except Exception as e:
            logger.error(f"Failed to save defaults for user {user_id}: {e}")
            raise

    async def clear_defaults(self, user_id: str) -> None:
        """Clear user's default settings."""
        memledger = await self._get_memledger()

        try:
            record_id = f"defaults-{user_id}"
            await memledger.delete(record_id)
            logger.info(f"Cleared defaults for user {user_id}")

        except Exception as e:
            logger.warning(f"Failed to clear defaults for user {user_id}: {e}")

    async def remember(
        self,
        content: str,
        namespace: str = "/learnings",
        record_type: RecordType = RecordType.SEMANTIC,
        metadata: Optional[dict[str, Any]] = None,
        **typed_fields: Any,
    ) -> str:
        """Store a memory for future recall. Returns the record ID."""
        memledger = await self._get_memledger()

        return await memledger.add(
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
        memledger = await self._get_memledger()

        return await memledger.search(
            query=query,
            namespace=namespace,
            top_k=top_k,
            record_type=record_type,
        )

    async def close(self) -> None:
        """Close memledger connection."""
        if self._ml:
            await memledger.close()
            self._ml = None


# --- Memory Tools for the Agent ---


@tool
async def set_user_defaults(
    cluster: Optional[str] = None,
    namespace: Optional[str] = None,
) -> str:
    """
    Save the user's default cluster and/or namespace for future requests.

    Use this when the user asks to set, save, or remember their defaults.
    Examples: "Set my default cluster to eks-1", "Remember namespace as kubeflow"

    Args:
        cluster: The default EKS cluster name to use
        namespace: The default Kubernetes namespace to use

    Returns:
        Confirmation message with the saved defaults
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    if cluster is None and namespace is None:
        return "Please specify at least one default to save (cluster or namespace)."

    try:
        defaults = await _memory_service.set_defaults(
            user_id=DEFAULT_USER_ID,
            cluster=cluster,
            namespace=namespace,
        )
        return f"Saved defaults: {defaults}"

    except Exception as e:
        logger.error(f"Failed to set defaults: {e}")
        return f"Failed to save defaults: {str(e)}"


@tool
async def get_user_defaults() -> str:
    """
    Retrieve the user's saved default cluster and namespace.

    Use this when you need defaults for a request that doesn't specify cluster/namespace.

    Returns:
        The user's saved defaults, or a message if none are set
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        defaults = await _memory_service.get_defaults(user_id=DEFAULT_USER_ID)

        if defaults.cluster is None and defaults.namespace is None:
            return "No defaults set. Ask the user to set defaults."

        return f"User defaults: {defaults}"

    except Exception as e:
        logger.error(f"Failed to get defaults: {e}")
        return f"Failed to retrieve defaults: {str(e)}"


@tool
async def clear_user_defaults() -> str:
    """
    Clear the user's saved defaults.

    Use this when the user asks to clear, remove, or forget their defaults.

    Returns:
        Confirmation message
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        await _memory_service.clear_defaults(user_id=DEFAULT_USER_ID)
        return "Defaults cleared successfully."

    except Exception as e:
        logger.error(f"Failed to clear defaults: {e}")
        return f"Failed to clear defaults: {str(e)}"


EKS_OPS_AGENT_ID = "eks-ops-agent"


@tool
async def remember_knowledge(
    content: str,
    namespace: str = "/learnings",
    record_type: str = "semantic",
    metadata: Optional[dict] = None,
    confidence: float = 0.5,
    hedged: bool = False,
    derived_from: Optional[list[str]] = None,
    supersedes: Optional[str] = None,
    replaces: Optional[str] = None,
    workflow_id: Optional[str] = None,
    triggered_by: Optional[str] = None,
) -> str:
    """
    Store operational knowledge with full v1 trust attribution.

    Args:
        content: What to remember. Be specific and include context.
        namespace: Category path (/incidents/{cluster}, /runbooks, /learnings, /eks-ops/remediations).
        record_type: 'semantic' (facts), 'episodic' (events), 'procedural' (how-to).
        metadata: Optional key-value extras.
        confidence: Trust signal (0.0-1.0). How sure are you? Lower for speculation.
        hedged: True if the claim is speculative or unverified.
        derived_from: List of memory IDs this record derives from (provenance chain).
            Use when the new knowledge is a remediation derived from a recalled incident.
        supersedes: Memory ID this record replaces.
        replaces: Natural-language description; resolved to a supersedes ID via search.
        workflow_id: Workflow/run identifier for cross-step correlation.
        triggered_by: Upstream alert/event ID that caused this record.

    Returns:
        Confirmation with the stored memory ID.
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        memledger = await _memory_service._get_memledger()
        rt = RecordType(record_type)

        supersedes_id = supersedes
        if not supersedes_id and replaces:
            results = await memledger.search(query=replaces, top_k=1)
            if results.records:
                supersedes_id = results.records[0].id

        record_id = await _memory_service.remember(
            content=content,
            namespace=namespace,
            record_type=rt,
            metadata=metadata or {},
            confidence=confidence,
            hedged=hedged,
            derived_from=derived_from,
            supersedes=supersedes_id,
            agent_id=EKS_OPS_AGENT_ID,
            created_by=EKS_OPS_AGENT_ID,
            workflow_id=workflow_id,
            triggered_by=triggered_by,
        )

        msg = f"Stored {record_type} memory id={record_id} (confidence={confidence:.2f} hedged={hedged}): {content[:100]}"
        if supersedes_id:
            msg += f"\n(Superseded old memory [{supersedes_id[:8]}] — marked as deprecated)"

        # Check if conflict was detected during add()
        conflict_entries = memledger.audit.by_record(record_id)
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

    Use this BEFORE starting any diagnostic or troubleshooting task to check
    if similar issues have been seen before. Also use when the user asks
    about past incidents or operational history.

    Args:
        query: What to search for (natural language description)
            Examples: "OOM kills in payment service", "how to upgrade EKS version"
        namespace: Optional scope to narrow search
            /incidents — search all incident history
            /incidents/eks-prod — search specific cluster incidents
            /runbooks — search procedures
            None — search everything
        top_k: Number of results to return

    Returns:
        Matching memories with relevance scores, or "No relevant memories found"
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        memledger = await _memory_service._get_memledger()
        # Use hybrid search (vector + BM25) when backend supports it,
        # falls back to vector-only transparently
        results = await memledger.search_hybrid(
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
                f"{i}. id={rec.id} [{rec.namespace}]{type_str}{score_str} "
                f"confidence={rec.confidence:.2f} hedged={rec.hedged}: "
                f"{rec.content}{meta_str}"
            )

        return (
            f"Found {len(results.records)} memories "
            f"(search took {results.search_time_ms}ms):\n"
            + "\n".join(lines)
            + "\n\nNote: Use the full `id=...` UUIDs above (not 8-char prefixes) "
            "when passing to derived_from or supersedes."
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

    Use this for complex situations where you need the full picture — what happened before,
    what procedures exist, and what approaches failed. More thorough than recall_knowledge.

    Args:
        query: What situation you're dealing with (natural language)
            Examples: "pod OOM kills in payment service", "EKS upgrade from 1.28 to 1.29"
        namespace: Optional scope to narrow search
        top_k: Number of results per category

    Returns:
        Grouped results: similar episodes, suggested procedures, known failures
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        memledger = await _memory_service._get_memledger()
        context = await memledger.recall_context(
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
    Record whether a recalled memory or procedure was helpful (success) or not (failure).

    Use this when the user reports that a previously recalled memory, procedure,
    or recommendation worked or didn't work. Memledger will find the most relevant
    memory matching the description and record the outcome.

    Successful memories will rank higher in future searches.

    Args:
        description: What the memory was about (e.g. "connection pool fix procedure")
        success: True if it was helpful, False if it was misleading
        record_id: Optional exact ID if known (from recall results). If not provided,
                   memledger will search for the best matching memory.

    Returns:
        Confirmation message
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        memledger = await _memory_service._get_memledger()

        result = await memledger.record_outcome(
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
    Helps decide whether to search memory or investigate from scratch.

    Args:
        namespace: Optional scope (e.g. "/incidents" to see incident stats only)

    Returns:
        Summary of stored memories: counts by type, namespaces, success rates
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        memledger = await _memory_service._get_memledger()
        stats = await memledger.stats(namespace=namespace)

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
async def review_memories(
    view: str = "active",
    namespace: Optional[str] = None,
    days: int = 30,
) -> str:
    """
    Review memories by lifecycle status or staleness — the ops lead's memory audit tool.

    Use this when the user wants to understand what's in memory, find outdated knowledge,
    or identify memories that need cleanup. This is the "memory hygiene" tool.

    Args:
        view: What to show:
            'active' — all active memories (default)
            'stale' — memories not accessed in the last N days (use 'days' to control)
            'deprecated' — memories that have been superseded by newer knowledge
            'expired' — memories marked for archival
            'archived' — soft-deleted memories (still recoverable)
        namespace: Optional scope (e.g. "/incidents" to audit only incident memories)
        days: For 'stale' view — how many days without access counts as stale (default 30)

    Returns:
        Formatted list of memories matching the criteria
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        memledger = await _memory_service._get_memledger()

        if view == "stale":
            records = await memledger.get_stale(days=days, namespace=namespace)
            label = f"stale (not accessed in {days} days)"
        elif view in ("deprecated", "expired", "archived", "active"):
            records = await memledger.get_by_status(status=view, namespace=namespace)
            label = view
        else:
            return f"Unknown view '{view}'. Use: active, stale, deprecated, expired, archived."

        if not records:
            return f"No {label} memories found{' in ' + namespace if namespace else ''}."

        lines = [f"Found {len(records)} {label} memories:"]
        for i, rec in enumerate(records, 1):
            age = ""
            if rec.last_accessed_at:
                age = f" | last accessed: {rec.last_accessed_at.strftime('%Y-%m-%d')}"
            elif rec.created_at:
                age = f" | created: {rec.created_at.strftime('%Y-%m-%d')}"
            success = ""
            if rec.success_count or rec.failure_count:
                success = f" | outcomes: {rec.success_count}✓ {rec.failure_count}✗"
            lines.append(
                f"  {i}. [{rec.record_type.value}] {rec.content[:80]}"
                f" (id: {rec.id[:8]}){age}{success}"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to review memories: {e}")
        return f"Failed to review memories: {str(e)}"


@tool
async def manage_memory_lifecycle(
    action: str,
    record_ids: Optional[list[str]] = None,
    scope: Optional[str] = None,
    namespace: Optional[str] = None,
    days: int = 30,
) -> str:
    """
    Manage memory lifecycle — expire, archive, or clean up memories.

    Use this when the user wants to retire outdated knowledge, archive old memories,
    or clean up after a memory audit. Always use review_memories first to see what
    will be affected.

    Args:
        action: What to do:
            'expire' — mark memories as expired (grace period before archive)
            'archive' — soft-delete memories (recoverable but hidden from search)
            'deprecate' — mark as superseded (use when newer knowledge exists)
        record_ids: Specific memory IDs to act on (from review_memories output)
        scope: Bulk scope instead of individual IDs:
            'all_stale' — act on all memories not accessed in N days
            'all_deprecated' — act on all deprecated memories
            'all_expired' — act on all expired memories
        namespace: Optional scope for bulk operations
        days: For 'all_stale' scope — staleness threshold (default 30)

    Returns:
        Summary of what was changed
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    if action not in ("expire", "archive", "deprecate"):
        return f"Unknown action '{action}'. Use: expire, archive, deprecate."

    try:
        memledger = await _memory_service._get_memledger()

        # Resolve record IDs from scope if not provided directly
        ids = record_ids or []
        if not ids and scope:
            if scope == "all_stale":
                records = await memledger.get_stale(days=days, namespace=namespace)
            elif scope == "all_deprecated":
                records = await memledger.get_by_status("deprecated", namespace=namespace)
            elif scope == "all_expired":
                records = await memledger.get_by_status("expired", namespace=namespace)
            else:
                return f"Unknown scope '{scope}'. Use: all_stale, all_deprecated, all_expired."
            ids = [r.id for r in records]

        if not ids:
            return f"No memories to {action}. Use review_memories to find candidates."

        # Map action to status
        status_map = {"expire": "expired", "archive": "archived", "deprecate": "deprecated"}
        target_status = status_map[action]

        if action == "archive":
            count = await memledger.bulk_archive(ids)
        else:
            count = await memledger.bulk_update_status(ids, target_status)

        return f"Done: {action}d {count} memories (target status: {target_status})."

    except Exception as e:
        logger.error(f"Failed to manage lifecycle: {e}")
        return f"Failed to manage lifecycle: {str(e)}"


@tool
async def session_history(
    session_id: str,
) -> str:
    """
    Retrieve all memories from a specific session — see what was learned in a conversation.

    Use this when the user asks about what happened in a previous session or wants
    to review what knowledge was captured during a specific interaction.

    Args:
        session_id: The session identifier to look up

    Returns:
        All memories stored during that session
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        memledger = await _memory_service._get_memledger()
        records = await memledger.get_by_session(session_id)

        if not records:
            return f"No memories found for session '{session_id}'."

        lines = [f"Session '{session_id}' — {len(records)} memories:"]
        for i, rec in enumerate(records, 1):
            lines.append(
                f"  {i}. [{rec.record_type.value}] {rec.content[:100]}"
            )
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        return f"Failed to get session history: {str(e)}"


@tool
async def memory_audit(
    record_id: Optional[str] = None,
    last_n: int = 10,
) -> str:
    """
    Show the audit trail of memory operations — what was stored, searched, updated, and when.

    Use this when the user asks about memory history, wants to understand how knowledge
    evolved, or needs a compliance trail of what the agent learned and when.

    Args:
        record_id: Optional — show audit trail for a specific memory only
        last_n: Number of recent entries to show (default 10)

    Returns:
        Formatted audit log with timestamps and operations
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        memledger = await _memory_service._get_memledger()

        if record_id:
            entries = memledger.audit.by_record(record_id)
            if not entries:
                return f"No audit entries found for memory {record_id[:8]}."
            lines = [f"Audit trail for memory {record_id[:8]} ({len(entries)} entries):"]
        else:
            entries = memledger.audit.recent(last_n)
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

    Use this when the user asks about the origin of knowledge, wants to understand
    how information propagated, or needs to verify the reliability of a memory.

    Args:
        description: What the memory is about (e.g. "connection pool fix")
        record_id: Optional exact ID if known

    Returns:
        Provenance chain showing creation, usage, supersession, and confidence
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        memledger = await _memory_service._get_memledger()
        lineage = await memledger.get_lineage(
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
                lines.append(f"    ← [{pred['id'][:8]}] {pred['content'][:60]} ({pred['status']})")

        if lineage.get("derived_records"):
            lines.append(f"  Derived records ({len(lineage['derived_records'])}):")
            for derived in lineage["derived_records"]:
                lines.append(f"    → [{derived['id'][:8]}] {derived['content'][:60]}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to get lineage: {e}")
        return f"Failed to get lineage: {str(e)}"


def get_memory_tools() -> list:
    """Get the list of memory tools for the agent."""
    return [
        set_user_defaults,
        get_user_defaults,
        clear_user_defaults,
        remember_knowledge,
        recall_knowledge,
        recall_context,
        mark_memory_outcome,
        memory_stats,
        review_memories,
        manage_memory_lifecycle,
        session_history,
        memory_audit,
        memory_lineage,
    ]
