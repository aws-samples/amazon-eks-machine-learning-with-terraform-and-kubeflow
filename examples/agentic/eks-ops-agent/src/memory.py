"""
Module 3: Long-term Memory with Engram

This module provides persistent, searchable memory for the EKS Ops Agent
using engram backed by PostgreSQL+pgvector (Aurora or in-cluster Postgres).

Two categories of memory:
1. User defaults — key-value storage for cluster/namespace preferences
   (backward-compatible with the original Redis implementation)
2. Semantic memory — vector-indexed knowledge that the agent can search
   (incidents, runbooks, operational learnings)

Engram schema:
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

from engram import Engram, RecordType
from engram.models import EmbeddingConfig

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
    Engram-backed memory service for the EKS Ops Agent.

    Provides:
    - User defaults (cluster/namespace preferences) via engram records
    - Semantic memory (searchable operational knowledge) via vector search
    """

    def __init__(
        self,
        pg_connection_string: str,
        embedding_provider: str = "bedrock",
        embedding_model: str = "amazon.titan-embed-text-v2:0",
        embedding_dimensions: int = 1024,
    ):
        self._pg_connection_string = pg_connection_string
        self._embedding_config = EmbeddingConfig(
            provider=embedding_provider,
            model=embedding_model,
            dimensions=embedding_dimensions,
        )
        self._engram: Optional[Engram] = None

    async def _get_engram(self) -> Engram:
        """Lazy-initialize engram connection on first use."""
        if self._engram is None:
            self._engram = await Engram.create(
                backend_name="pgvector",
                connection_string=self._pg_connection_string,
                embedding_config=self._embedding_config,
            )
            logger.info("MemoryService initialized with engram (pgvector)")
        return self._engram

    def _defaults_ns(self, user_id: str) -> str:
        """Namespace for user defaults."""
        return f"/users/{user_id}/defaults"

    async def get_defaults(self, user_id: str) -> UserDefaults:
        """Retrieve user's default settings from engram."""
        engram = await self._get_engram()

        try:
            results = await engram.search(
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
        """Save user's default settings to engram."""
        engram = await self._get_engram()

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

            await engram.add(
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
        engram = await self._get_engram()

        try:
            record_id = f"defaults-{user_id}"
            await engram.delete(record_id)
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
        engram = await self._get_engram()

        return await engram.add(
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
        engram = await self._get_engram()

        return await engram.search(
            query=query,
            namespace=namespace,
            top_k=top_k,
            record_type=record_type,
        )

    async def close(self) -> None:
        """Close engram connection."""
        if self._engram:
            await engram.close()
            self._engram = None


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


@tool
async def remember_knowledge(
    content: str,
    namespace: str = "/learnings",
    record_type: str = "semantic",
    metadata: Optional[dict] = None,
) -> str:
    """
    Store operational knowledge in long-term memory for future recall.

    Use this after resolving an incident, discovering a useful pattern, or
    learning something about the cluster that should be remembered.

    Args:
        content: What to remember. Be specific and include context.
            Good: "EKS cluster eks-prod OOM kills caused by HikariCP pool leak. Fix: set maxPoolSize=50"
            Bad: "Fixed a bug"
        namespace: Category path for organizing memories.
            /incidents/{cluster} — incident resolutions
            /runbooks — operational procedures
            /learnings — general knowledge
        record_type: Type of memory - 'semantic' (facts), 'episodic' (events), 'procedural' (how-to)
        metadata: Optional key-value pairs (e.g. {"severity": "P1", "cluster": "eks-prod"})

    Returns:
        Confirmation with the stored memory ID
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        rt = RecordType(record_type)
        record_id = await _memory_service.remember(
            content=content,
            namespace=namespace,
            record_type=rt,
            metadata=metadata or {},
        )
        return f"Stored {record_type} memory [{record_id[:8]}]: {content[:100]}"

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
        results = await _memory_service.recall(
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
        engram = await _memory_service._get_engram()
        context = await engram.recall_context(
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
    or recommendation worked or didn't work. The tool will find the most relevant
    memory matching the description and record the outcome.

    Successful memories will rank higher in future searches.

    Args:
        description: What the memory was about (e.g. "connection pool fix procedure")
        success: True if it was helpful, False if it was misleading
        record_id: Optional exact ID if known (from recall results). If not provided,
                   the tool will search for the best matching memory.

    Returns:
        Confirmation message
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        engram = await _memory_service._get_engram()

        # If no record_id provided, search for the best match
        if not record_id:
            results = await engram.search(query=description, top_k=1)
            if not results.records:
                return f"Could not find a memory matching '{description}' to record outcome."
            record_id = results.records[0].id
            matched_content = results.records[0].content[:80]
        else:
            matched_content = record_id[:8]

        result = await engram.record_outcome(record_id, success=success)
        outcome = "successful" if success else "unsuccessful"
        if result:
            return f"Recorded outcome as {outcome} for memory: {matched_content}"
        return f"Memory not found."

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
        engram = await _memory_service._get_engram()
        stats = await engram.stats(namespace=namespace)

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
    ]
