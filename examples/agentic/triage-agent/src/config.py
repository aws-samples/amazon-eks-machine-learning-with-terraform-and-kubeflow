"""Configuration for Triage Agent."""

import os


class Config:
    """Environment-based configuration."""

    # Agent identity
    AGENT_ID: str = "triage-agent"

    # AWS / Bedrock
    AWS_REGION: str = os.getenv("AWS_REGION", "us-west-2")
    BEDROCK_MODEL_ID: str = os.getenv(
        "BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )

    # Memory — always enabled for triage agent
    ENABLE_MEMORY: bool = os.getenv("ENABLE_MEMORY", "true").lower() == "true"

    # kagent
    KAGENT_URL: str = os.getenv(
        "KAGENT_URL", "http://kagent-controller.kagent.svc.cluster.local:8083"
    )

    # Memledger (persistent memory)
    MEMLEDGER_PG_DSN: str = os.getenv("MEMLEDGER_PG_DSN", "")
    MEMLEDGER_CONFIG_PATH: str = os.getenv("MEMLEDGER_CONFIG_PATH", "")

    # Embedding config for memledger
    MEMLEDGER_EMBEDDING_PROVIDER: str = os.getenv("MEMLEDGER_EMBEDDING_PROVIDER", "bedrock")
    MEMLEDGER_EMBEDDING_MODEL: str = os.getenv("MEMLEDGER_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    MEMLEDGER_EMBEDDING_DIMENSIONS: int = int(os.getenv("MEMLEDGER_EMBEDDING_DIMENSIONS", "1024"))

    # Langfuse (observability)
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")


config = Config()
