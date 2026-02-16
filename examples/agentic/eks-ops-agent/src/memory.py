"""
Module 3: Long-term Memory with Redis

This module provides persistent storage for user defaults (cluster, namespace)
that survive across chat sessions. When a user sets defaults, they are stored
in Redis and automatically retrieved in future sessions.

Redis schema:
    Key:   user:{user_id}:defaults
    Value: JSON {"cluster": "eks-1", "namespace": "default"}

Usage:
    memory = MemoryService(redis_url="redis://localhost:6379")
    await memory.set_defaults(user_id, cluster="eks-1", namespace="default")
    defaults = await memory.get_defaults(user_id)
"""

import json
import logging
from typing import Optional

import redis.asyncio as redis
from langchain_core.tools import tool

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
    Redis-backed memory service for user defaults.

    Provides simple get/set operations for user preferences that
    persist across chat sessions.
    """

    def __init__(self, redis_url: str):
        """
        Initialize the memory service.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379")
        """
        self.redis_url = redis_url
        self._client: Optional[redis.Redis] = None

    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _defaults_key(self, user_id: str) -> str:
        """Generate Redis key for user defaults."""
        return f"user:{user_id}:defaults"

    async def get_defaults(self, user_id: str) -> UserDefaults:
        """
        Retrieve user's default settings.

        Args:
            user_id: User identifier from kagent session

        Returns:
            UserDefaults with cluster/namespace if set, empty otherwise
        """
        try:
            client = await self._get_client()
            data = await client.get(self._defaults_key(user_id))

            if data:
                defaults = UserDefaults.from_dict(json.loads(data))
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
        """
        Save user's default settings.

        Args:
            user_id: User identifier from kagent session
            cluster: Default EKS cluster name
            namespace: Default Kubernetes namespace

        Returns:
            Updated UserDefaults
        """
        try:
            client = await self._get_client()

            # Get existing defaults and merge with new values
            existing = await self.get_defaults(user_id)

            if cluster is not None:
                existing.cluster = cluster
            if namespace is not None:
                existing.namespace = namespace

            # Save to Redis
            await client.set(
                self._defaults_key(user_id),
                json.dumps(existing.to_dict()),
            )

            logger.info(f"Saved defaults for user {user_id}: {existing}")
            return existing

        except Exception as e:
            logger.error(f"Failed to save defaults for user {user_id}: {e}")
            raise

    async def clear_defaults(self, user_id: str) -> None:
        """
        Clear user's default settings.

        Args:
            user_id: User identifier from kagent session
        """
        try:
            client = await self._get_client()
            await client.delete(self._defaults_key(user_id))
            logger.info(f"Cleared defaults for user {user_id}")

        except Exception as e:
            logger.warning(f"Failed to clear defaults for user {user_id}: {e}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


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


def get_memory_tools() -> list:
    """Get the list of memory tools for the agent."""
    return [set_user_defaults, get_user_defaults, clear_user_defaults]
