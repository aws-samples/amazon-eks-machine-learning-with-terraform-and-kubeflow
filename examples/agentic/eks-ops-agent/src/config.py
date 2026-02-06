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

    # kagent
    KAGENT_URL: str = os.getenv(
        "KAGENT_URL", "http://kagent-controller.kagent.svc.cluster.local:8083"
    )

    # Redis (long-term memory) - Phase 3
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Langfuse (observability) - Phase 4
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")


config = Config()
