"""Configuration for EKS Ops Agent."""

import os


class Config:
    """Environment-based configuration."""

    # AWS / Bedrock
    AWS_REGION: str = os.getenv("AWS_REGION", "us-west-2")
    BEDROCK_MODEL_ID: str = os.getenv(
        "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )

    # EKS
    EKS_CLUSTER_NAME: str = os.getenv("EKS_CLUSTER_NAME", "")

    # EKS MCP Server - Module 2
    ENABLE_MCP_TOOLS: bool = os.getenv("ENABLE_MCP_TOOLS", "false").lower() == "true"

    # Memory - Module 3
    ENABLE_MEMORY: bool = os.getenv("ENABLE_MEMORY", "false").lower() == "true"

    # kagent
    KAGENT_URL: str = os.getenv(
        "KAGENT_URL", "http://kagent-controller.kagent.svc.cluster.local:8083"
    )

    # Engram (persistent memory) - Module 3
    # Connection string to PostgreSQL+pgvector (Aurora or in-cluster Postgres)
    ENGRAM_PG_URL: str = os.getenv("ENGRAM_PG_URL", "")
    # Optional: path to engram YAML config (for composition/multi-backend mode)
    ENGRAM_CONFIG_PATH: str = os.getenv("ENGRAM_CONFIG_PATH", "")

    # Embedding config for engram
    ENGRAM_EMBEDDING_PROVIDER: str = os.getenv("ENGRAM_EMBEDDING_PROVIDER", "bedrock")
    ENGRAM_EMBEDDING_MODEL: str = os.getenv("ENGRAM_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    ENGRAM_EMBEDDING_DIMENSIONS: int = int(os.getenv("ENGRAM_EMBEDDING_DIMENSIONS", "1024"))

    # Langfuse (observability) - Module 4
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")


config = Config()
