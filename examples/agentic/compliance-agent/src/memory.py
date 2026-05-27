"""
Compliance Agent: Memory Governance Tools

This module provides memory tools for the compliance agent — focused on
auditing, RBAC enforcement, lifecycle management, and trust reporting.

The compliance agent can read from all namespaces but only writes to
/compliance/*. It uses memledger's policy infrastructure (ConfidencePolicy,
NamespaceRBAC) to enforce governance rules.

Tools:
    recall_knowledge      — cross-agent search across all namespaces
    memory_stats          — statistics about stored memories
    memory_audit          — audit trail of memory operations
    memory_lineage        — provenance chain for a specific memory
    review_memories       — list memories by lifecycle status
    manage_memory_lifecycle — bulk lifecycle operations
    audit_cross_agent_access — full 5-hop attribution chain
    check_namespace_compliance — RBAC access check for all agents
    run_staleness_scan    — temporal decay scoring across namespaces
    enforce_lifecycle     — bulk expire/archive/deprecate operations
    generate_trust_report — full trust attestation for a memory
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from langchain_core.tools import tool

from memledger import Memledger, RecordType
from memledger.models import EmbeddingConfig
from memledger.policies.confidence_policy import ConfidencePolicy
from memledger.policies.namespace_rbac import NamespaceRBAC

logger = logging.getLogger(__name__)

# Global memory service instance (set by app.py)
_memory_service: Optional["MemoryService"] = None

# Agent identifiers for cross-agent RBAC checks
AGENT_IDS = ["eks-ops-agent", "compliance-agent", "planning-agent"]

# Compliance agent identity
COMPLIANCE_AGENT_ID = "compliance-agent"


def set_memory_service(service: "MemoryService") -> None:
    """Set the global memory service instance."""
    global _memory_service
    _memory_service = service


class MemoryService:
    """
    Memledger-backed memory service for the Compliance Agent.

    Provides cross-agent memory governance: auditing, RBAC enforcement,
    lifecycle management, and trust reporting.

    Supports two initialization modes:
    - Direct: pg_connection_string -> pgvector backend (default)
    - Config file: config_path -> any backend or composition (for multi-backend)
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

    async def recall(
        self,
        query: str,
        namespace: Optional[str] = None,
        top_k: int = 5,
        record_type: Optional[RecordType] = None,
    ):
        """Search memories by semantic similarity."""
        ml = await self._get_memledger()
        return await ml.search(
            query=query,
            namespace=namespace,
            top_k=top_k,
            record_type=record_type,
        )

    async def remember(
        self,
        content: str,
        namespace: str = "/compliance/reports",
        record_type: RecordType = RecordType.SEMANTIC,
        metadata: Optional[dict[str, Any]] = None,
        **typed_fields: Any,
    ) -> str:
        """Store a compliance report in memory. Returns the record ID."""
        ml = await self._get_memledger()
        return await ml.add(
            content=content,
            record_type=record_type,
            namespace=namespace,
            metadata=metadata or {},
            agent_id=COMPLIANCE_AGENT_ID,
            **typed_fields,
        )

    async def close(self) -> None:
        """Close memledger connection."""
        if self._ml:
            await self._ml.close()
            self._ml = None


# --- Memory Tools for the Agent ---


@tool
async def recall_knowledge(
    query: str,
    namespace: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """
    Search long-term memory across all agent namespaces.

    The compliance agent has read access to all namespaces for auditing.
    Use this to find memories created by any agent.

    Args:
        query: What to search for (natural language description)
        namespace: Optional scope to narrow search (e.g. "/incidents", "/runbooks")
            None searches everything.
        top_k: Number of results to return

    Returns:
        Matching memories with relevance scores, or "No relevant memories found"
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_memledger()
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
            agent_str = f" | agent: {rec.agent_id}" if hasattr(rec, 'agent_id') and rec.agent_id else ""
            lines.append(
                f"{i}. [{rec.namespace}]{type_str}{score_str}: {rec.content}{meta_str}{agent_str}"
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
async def memory_stats(
    namespace: Optional[str] = None,
) -> str:
    """
    Get statistics about what's stored in memory across all agent namespaces.

    Use this to understand the overall health and distribution of memories
    before running audits or compliance checks.

    Args:
        namespace: Optional scope (e.g. "/incidents" to see incident stats only)

    Returns:
        Summary of stored memories: counts by type, namespaces, success rates
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_memledger()
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
    Show the audit trail of memory operations — what was stored, searched, updated, and when.

    Use this to understand how knowledge evolved, verify compliance, or investigate
    who changed what and when across all agents.

    Args:
        record_id: Optional — show audit trail for a specific memory only
        last_n: Number of recent entries to show (default 10)

    Returns:
        Formatted audit log with timestamps and operations
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_memledger()

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
        description: What the memory is about (e.g. "connection pool fix")
        record_id: Optional exact ID if known

    Returns:
        Provenance chain showing creation, usage, supersession, and confidence
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_memledger()
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


@tool
async def review_memories(
    view: str = "active",
    namespace: Optional[str] = None,
    days: int = 30,
) -> str:
    """
    Review memories by lifecycle status or staleness — the compliance audit tool.

    Use this to find outdated knowledge, identify memories needing cleanup,
    or generate inventory reports across all agent namespaces.

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
        ml = await _memory_service._get_memledger()

        if view == "stale":
            records = await ml.get_stale(days=days, namespace=namespace)
            label = f"stale (not accessed in {days} days)"
        elif view in ("deprecated", "expired", "archived", "active"):
            records = await ml.get_by_status(status=view, namespace=namespace)
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
                success = f" | outcomes: {rec.success_count}ok {rec.failure_count}fail"
            agent = ""
            if hasattr(rec, 'agent_id') and rec.agent_id:
                agent = f" | agent: {rec.agent_id}"
            lines.append(
                f"  {i}. [{rec.record_type.value}] {rec.content[:80]}"
                f" (id: {rec.id[:8]}){age}{success}{agent}"
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

    Use this after reviewing memories to retire outdated knowledge, archive
    old memories, or clean up. Always use review_memories first.

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
        ml = await _memory_service._get_memledger()

        ids = record_ids or []
        if not ids and scope:
            if scope == "all_stale":
                records = await ml.get_stale(days=days, namespace=namespace)
            elif scope == "all_deprecated":
                records = await ml.get_by_status("deprecated", namespace=namespace)
            elif scope == "all_expired":
                records = await ml.get_by_status("expired", namespace=namespace)
            else:
                return f"Unknown scope '{scope}'. Use: all_stale, all_deprecated, all_expired."
            ids = [r.id for r in records]

        if not ids:
            return f"No memories to {action}. Use review_memories to find candidates."

        status_map = {"expire": "expired", "archive": "archived", "deprecate": "deprecated"}
        target_status = status_map[action]

        if action == "archive":
            count = await ml.bulk_archive(ids)
        else:
            count = await ml.bulk_update_status(ids, target_status)

        return f"Done: {action}d {count} memories (target status: {target_status})."

    except Exception as e:
        logger.error(f"Failed to manage lifecycle: {e}")
        return f"Failed to manage lifecycle: {str(e)}"


# --- Compliance-Specific Tools ---


@tool
async def audit_cross_agent_access(
    description: str,
    record_id: Optional[str] = None,
) -> str:
    """
    Audit the full provenance chain for a memory — show who created it,
    who accessed it, confidence score, and supersession history.

    This is the compliance agent's core audit tool. It calls memledger's
    get_lineage() with a 5-hop depth to trace the full attribution chain.

    Args:
        description: What the memory is about (e.g. "OOM fix for payment service")
        record_id: Optional exact memory ID if known

    Returns:
        Formatted compliance audit report with full provenance chain
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_memledger()
        lineage = await ml.get_lineage(
            record_id=record_id,
            description=description,
            max_hops=5,
        )

        if "error" in lineage:
            return f"Audit failed — could not find memory: {lineage['error']}"

        rec = lineage["record"]
        confidence = lineage.get("confidence", "N/A")

        lines = []
        lines.append("=" * 60)
        lines.append("CROSS-AGENT ACCESS AUDIT REPORT")
        lines.append("=" * 60)
        lines.append(f"Record ID:    {rec['id']}")
        lines.append(f"Content:      {rec['content'][:120]}")
        lines.append(f"Type:         {rec['record_type']} | Status: {rec['status']}")
        lines.append(f"Namespace:    {rec.get('namespace', 'N/A')}")
        lines.append(f"Created:      {rec.get('created_at', 'unknown')}")
        lines.append("")

        # Creator attribution
        creator = lineage.get("created_by", "UNATTRIBUTED")
        lines.append(f"CREATOR:      {creator}")
        if creator == "UNATTRIBUTED":
            lines.append("  ** COMPLIANCE GAP: Memory has no creator attribution **")
        lines.append("")

        # Confidence score
        lines.append(f"CONFIDENCE:   {confidence}")
        if isinstance(confidence, (int, float)) and confidence < 0.5:
            lines.append("  ** LOW CONFIDENCE: Below 0.5 threshold **")
        lines.append("")

        # Access history
        accessors = lineage.get("accessed_by", [])
        lines.append(f"ACCESSORS ({len(accessors)}):")
        if accessors:
            for accessor in accessors:
                lines.append(f"  - {accessor}")
        else:
            lines.append("  (no access records)")
        lines.append("")

        # Supersession history
        chain = lineage.get("supersedes_chain", [])
        lines.append(f"SUPERSESSION CHAIN ({len(chain)} predecessors):")
        if chain:
            for i, pred in enumerate(chain, 1):
                lines.append(
                    f"  {i}. [{pred['id'][:8]}] {pred['content'][:60]} "
                    f"(status: {pred['status']})"
                )
        else:
            lines.append("  (original record — no predecessors)")
        lines.append("")

        # Derived records
        derived = lineage.get("derived_records", [])
        if derived:
            lines.append(f"DERIVED RECORDS ({len(derived)}):")
            for d in derived:
                lines.append(f"  -> [{d['id'][:8]}] {d['content'][:60]}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to audit cross-agent access: {e}")
        return f"Audit failed: {str(e)}"


@tool
async def check_namespace_compliance(
    namespace: Optional[str] = None,
) -> str:
    """
    Check RBAC compliance for a namespace — verify which agents have read, write,
    or denied access.

    Uses memledger's NamespaceRBAC policy to check access for all known agents
    (eks-ops-agent, compliance-agent, planning-agent) against the specified namespace.

    Args:
        namespace: The namespace to check (e.g. "/incidents/eks-prod", "/compliance/reports").
            If None, checks all known namespaces from memory stats.

    Returns:
        Formatted RBAC compliance report showing access decisions for each agent
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_memledger()
        rbac = NamespaceRBAC()

        # If no namespace specified, discover namespaces from stats
        namespaces_to_check = []
        if namespace:
            namespaces_to_check = [namespace]
        else:
            stats = await ml.stats()
            namespaces_to_check = stats.get("namespaces", [])
            if not namespaces_to_check:
                return "No namespaces found in memory store."

        lines = []
        lines.append("=" * 60)
        lines.append("NAMESPACE RBAC COMPLIANCE REPORT")
        lines.append("=" * 60)
        lines.append(f"Checked at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"Agents checked: {', '.join(AGENT_IDS)}")
        lines.append("")

        compliance_gaps = 0
        for ns in namespaces_to_check:
            lines.append(f"Namespace: {ns}")
            lines.append("-" * 40)

            for agent_id in AGENT_IDS:
                read_access = rbac.check_access(agent_id=agent_id, namespace=ns, operation="read")
                write_access = rbac.check_access(agent_id=agent_id, namespace=ns, operation="write")

                read_str = "ALLOW" if read_access else "DENY"
                write_str = "ALLOW" if write_access else "DENY"

                lines.append(f"  {agent_id:20s} | read: {read_str:5s} | write: {write_str:5s}")

                # Flag unexpected access patterns
                if write_access and ns.startswith("/compliance") and agent_id != "compliance-agent":
                    lines.append(f"    ** COMPLIANCE GAP: Non-compliance agent has write access to {ns} **")
                    compliance_gaps += 1

            lines.append("")

        lines.append(f"TOTAL COMPLIANCE GAPS: {compliance_gaps}")
        lines.append("=" * 60)
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to check namespace compliance: {e}")
        return f"RBAC check failed: {str(e)}"


@tool
async def run_staleness_scan(
    days: int = 7,
    namespace: Optional[str] = None,
) -> str:
    """
    Scan for stale memories across namespaces and apply temporal decay scoring.

    Finds all memories not accessed within the specified days, calculates a
    decay score based on age, and writes a compliance report to /compliance/reports/.

    Args:
        days: Number of days without access to consider stale (default 7)
        namespace: Optional namespace to limit the scan. None scans all namespaces.

    Returns:
        Staleness scan report with decay scores and recommended actions
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_memledger()
        stale_records = await ml.get_stale(days=days, namespace=namespace)

        now = datetime.now(timezone.utc)
        scan_time = now.strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = []
        lines.append("=" * 60)
        lines.append("STALENESS SCAN REPORT")
        lines.append("=" * 60)
        lines.append(f"Scan time:     {scan_time}")
        lines.append(f"Threshold:     {days} days")
        lines.append(f"Scope:         {namespace or 'all namespaces'}")
        lines.append(f"Stale records: {len(stale_records)}")
        lines.append("")

        if not stale_records:
            lines.append("No stale memories found. Memory store is healthy.")
            lines.append("=" * 60)
            return "\n".join(lines)

        # Group by namespace and calculate decay scores
        by_namespace: dict[str, list] = {}
        for rec in stale_records:
            ns = getattr(rec, 'namespace', '/unknown')
            by_namespace.setdefault(ns, []).append(rec)

        for ns, records in sorted(by_namespace.items()):
            lines.append(f"Namespace: {ns} ({len(records)} stale)")
            lines.append("-" * 40)

            for rec in records:
                # Calculate temporal decay score (0.0 = completely decayed, 1.0 = fresh)
                last_access = getattr(rec, 'last_accessed_at', None) or getattr(rec, 'created_at', None)
                if last_access:
                    age_days = (now - last_access).days
                    # Exponential decay: half-life of 30 days
                    decay_score = 0.5 ** (age_days / 30.0)
                else:
                    age_days = -1
                    decay_score = 0.0

                # Recommend action based on decay score
                if decay_score < 0.1:
                    action = "ARCHIVE"
                elif decay_score < 0.3:
                    action = "EXPIRE"
                else:
                    action = "REVIEW"

                lines.append(
                    f"  [{rec.id[:8]}] {rec.content[:50]}"
                    f" | age: {age_days}d | decay: {decay_score:.2f} | rec: {action}"
                )

            lines.append("")

        # Write compliance report to memory
        report_content = (
            f"Staleness scan at {scan_time}: {len(stale_records)} stale records "
            f"found across {len(by_namespace)} namespaces (threshold: {days}d)"
        )
        report_id = await _memory_service.remember(
            content=report_content,
            namespace="/compliance/reports",
            record_type=RecordType.SEMANTIC,
            metadata={
                "report_type": "staleness_scan",
                "threshold_days": days,
                "stale_count": len(stale_records),
                "scan_time": scan_time,
            },
            confidence=0.95,
            agent_id=COMPLIANCE_AGENT_ID,
            created_by=COMPLIANCE_AGENT_ID,
        )
        lines.append(f"Report saved to /compliance/reports/ (id: {report_id[:8]})")
        lines.append("=" * 60)
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to run staleness scan: {e}")
        return f"Staleness scan failed: {str(e)}"


@tool
async def enforce_lifecycle(
    action: str,
    scope: str,
    namespace: Optional[str] = None,
    days: int = 7,
) -> str:
    """
    Bulk lifecycle enforcement — expire stale, archive expired, or deprecate conflicting memories.

    This is the compliance agent's enforcement tool. It wraps manage_memory_lifecycle
    for bulk operations across namespaces.

    Args:
        action: What to enforce:
            'expire_stale' — mark all stale memories as expired
            'archive_expired' — archive all expired memories (remove from search)
            'deprecate_conflicting' — deprecate memories that have been superseded
        scope: Scope of enforcement:
            'all' — enforce across all namespaces
            'namespace' — enforce only within the specified namespace
        namespace: Required when scope='namespace'. The namespace to enforce within.
        days: For 'expire_stale' — staleness threshold in days (default 7)

    Returns:
        Enforcement summary with counts of affected memories
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    valid_actions = ("expire_stale", "archive_expired", "deprecate_conflicting")
    if action not in valid_actions:
        return f"Unknown action '{action}'. Use: {', '.join(valid_actions)}."

    if scope == "namespace" and not namespace:
        return "namespace parameter is required when scope='namespace'."

    try:
        ml = await _memory_service._get_memledger()
        ns = namespace if scope == "namespace" else None

        if action == "expire_stale":
            records = await ml.get_stale(days=days, namespace=ns)
            if not records:
                return f"No stale memories found (threshold: {days} days)."
            ids = [r.id for r in records]
            count = await ml.bulk_update_status(ids, "expired")
            summary = f"Expired {count} stale memories (not accessed in {days}+ days)"

        elif action == "archive_expired":
            records = await ml.get_by_status("expired", namespace=ns)
            if not records:
                return "No expired memories to archive."
            ids = [r.id for r in records]
            count = await ml.bulk_archive(ids)
            summary = f"Archived {count} expired memories"

        elif action == "deprecate_conflicting":
            records = await ml.get_by_status("active", namespace=ns)
            if not records:
                return "No active memories to check for conflicts."
            # Find records that have been superseded but not yet deprecated
            deprecated_count = 0
            for rec in records:
                lineage = await ml.get_lineage(record_id=rec.id)
                if lineage.get("superseded_by"):
                    await ml.bulk_update_status([rec.id], "deprecated")
                    deprecated_count += 1
            summary = f"Deprecated {deprecated_count} conflicting/superseded memories"

        # Write enforcement report
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        report_content = f"Lifecycle enforcement at {now}: {summary}"
        report_id = await _memory_service.remember(
            content=report_content,
            namespace="/compliance/reports",
            record_type=RecordType.SEMANTIC,
            metadata={
                "report_type": "lifecycle_enforcement",
                "action": action,
                "scope": scope,
                "namespace": namespace,
                "enforcement_time": now,
            },
            confidence=0.95,
            agent_id=COMPLIANCE_AGENT_ID,
            created_by=COMPLIANCE_AGENT_ID,
        )

        return (
            f"{summary}.\n"
            f"Scope: {scope}{' (' + namespace + ')' if namespace else ''}\n"
            f"Enforcement report saved (id: {report_id[:8]})"
        )

    except Exception as e:
        logger.error(f"Failed to enforce lifecycle: {e}")
        return f"Lifecycle enforcement failed: {str(e)}"


@tool
async def generate_trust_report(
    description: str,
    record_id: Optional[str] = None,
) -> str:
    """
    Generate a full trust attestation report for a memory.

    Assembles: confidence score (from ConfidencePolicy), provenance chain
    (from get_lineage), outcome history (success/failure counts), lifecycle
    status, and RBAC access decisions.

    This is the compliance agent's signature output — a trust report that
    leadership can rely on to verify memory quality.

    Args:
        description: What the memory is about (e.g. "connection pool fix")
        record_id: Optional exact memory ID if known

    Returns:
        Formatted trust attestation report
    """
    if _memory_service is None:
        return "Memory is not enabled. Set ENABLE_MEMORY=true to use this feature."

    try:
        ml = await _memory_service._get_memledger()

        # Get lineage (includes confidence, provenance, supersession)
        lineage = await ml.get_lineage(
            record_id=record_id,
            description=description,
            max_hops=5,
        )

        if "error" in lineage:
            return f"Trust report failed — could not find memory: {lineage['error']}"

        rec = lineage["record"]
        rec_id = rec["id"]

        # Confidence score from ConfidencePolicy
        confidence_policy = ConfidencePolicy()
        confidence_score = lineage.get("confidence", None)
        if confidence_score is None:
            # Compute directly if not in lineage
            confidence_score = confidence_policy.compute(
                success_count=rec.get("success_count", 0),
                failure_count=rec.get("failure_count", 0),
                access_count=rec.get("access_count", 0),
            )

        # RBAC check for the record's namespace
        rbac = NamespaceRBAC()
        rec_namespace = rec.get("namespace", "/unknown")
        rbac_results = {}
        for agent_id in AGENT_IDS:
            read_ok = rbac.check_access(agent_id=agent_id, namespace=rec_namespace, operation="read")
            write_ok = rbac.check_access(agent_id=agent_id, namespace=rec_namespace, operation="write")
            rbac_results[agent_id] = {"read": read_ok, "write": write_ok}

        # Outcome history
        success_count = rec.get("success_count", 0)
        failure_count = rec.get("failure_count", 0)
        total_outcomes = success_count + failure_count

        # Trust grade based on confidence
        if isinstance(confidence_score, (int, float)):
            if confidence_score >= 0.8:
                grade = "HIGH"
            elif confidence_score >= 0.5:
                grade = "MEDIUM"
            elif confidence_score >= 0.2:
                grade = "LOW"
            else:
                grade = "UNTRUSTED"
        else:
            grade = "UNSCORED"

        # Build report
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = []
        lines.append("=" * 60)
        lines.append("TRUST ATTESTATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated:    {now}")
        lines.append(f"Attested by:  {COMPLIANCE_AGENT_ID}")
        lines.append("")
        lines.append("--- IDENTITY ---")
        lines.append(f"Record ID:    {rec_id}")
        lines.append(f"Content:      {rec['content'][:150]}")
        lines.append(f"Type:         {rec['record_type']}")
        lines.append(f"Namespace:    {rec_namespace}")
        lines.append(f"Status:       {rec['status']}")
        lines.append(f"Created:      {rec.get('created_at', 'unknown')}")
        lines.append("")

        lines.append("--- TRUST SCORE ---")
        lines.append(f"Confidence:   {confidence_score}")
        lines.append(f"Trust Grade:  {grade}")
        lines.append(f"Outcomes:     {success_count} success / {failure_count} failure (total: {total_outcomes})")
        if total_outcomes > 0:
            lines.append(f"Success Rate: {success_count / total_outcomes:.1%}")
        lines.append("")

        lines.append("--- PROVENANCE ---")
        creator = lineage.get("created_by", "UNATTRIBUTED")
        lines.append(f"Creator:      {creator}")
        if creator == "UNATTRIBUTED":
            lines.append("  ** COMPLIANCE GAP: No creator attribution **")

        accessors = lineage.get("accessed_by", [])
        lines.append(f"Accessors:    {', '.join(accessors) if accessors else '(none)'}")

        chain = lineage.get("supersedes_chain", [])
        if chain:
            lines.append(f"Supersession: {len(chain)} predecessor(s)")
            for i, pred in enumerate(chain, 1):
                lines.append(f"  {i}. [{pred['id'][:8]}] {pred['content'][:50]} ({pred['status']})")
        else:
            lines.append("Supersession: Original record (no predecessors)")
        lines.append("")

        lines.append("--- RBAC ACCESS ---")
        for agent_id, access in rbac_results.items():
            read_str = "ALLOW" if access["read"] else "DENY"
            write_str = "ALLOW" if access["write"] else "DENY"
            lines.append(f"  {agent_id:20s} | read: {read_str:5s} | write: {write_str:5s}")
        lines.append("")

        # Compliance flags
        flags = []
        if creator == "UNATTRIBUTED":
            flags.append("Missing creator attribution")
        if isinstance(confidence_score, (int, float)) and confidence_score < 0.5:
            flags.append(f"Low confidence ({confidence_score:.2f})")
        if failure_count > success_count and total_outcomes > 0:
            flags.append(f"More failures than successes ({failure_count}F > {success_count}S)")
        if rec["status"] in ("deprecated", "expired"):
            flags.append(f"Non-active status: {rec['status']}")

        lines.append("--- COMPLIANCE FLAGS ---")
        if flags:
            for flag in flags:
                lines.append(f"  ** {flag}")
        else:
            lines.append("  No compliance issues detected.")
        lines.append("")

        lines.append(f"VERDICT: {grade}")
        lines.append("=" * 60)

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to generate trust report: {e}")
        return f"Trust report failed: {str(e)}"


def get_memory_tools() -> list:
    """Get the list of memory tools for the compliance agent."""
    return [
        # Core memory tools (read-only / audit)
        recall_knowledge,
        memory_stats,
        memory_audit,
        memory_lineage,
        review_memories,
        manage_memory_lifecycle,
        # Compliance-specific tools
        audit_cross_agent_access,
        check_namespace_compliance,
        run_staleness_scan,
        enforce_lifecycle,
        generate_trust_report,
    ]
