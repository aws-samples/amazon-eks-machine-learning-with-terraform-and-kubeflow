# EKS Ops Agent - Dockerfile
# Uses uv for fast Python dependency management

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# memledger version + extras are configurable via build args so the same
# Dockerfile builds both the v1 (OSS pgvector) and v2 (AWS composition) tracks.
#   v1 OSS:  --build-arg MEMLEDGER_VERSION=1.0.0 --build-arg MEMLEDGER_EXTRAS=pgvector,bedrock
#   v2 OSS:  defaults below — pgvector still works (default composition file)
#   v2 AWS:  defaults below + flip MEMLEDGER_CONFIG_PATH to memledger-composition-aws.yaml
ARG MEMLEDGER_VERSION=2.0.0
ARG MEMLEDGER_EXTRAS=aws,dynamodb,opensearch

# Copy all source files
COPY pyproject.toml .
COPY src/ ./src/

# Both composition configs ship in the image. The OSS pgvector config is the
# default at /app/memledger-composition.yaml; the AWS config is alongside it.
# Override MEMLEDGER_CONFIG_PATH on the pod to switch tracks.
COPY memledger-composition-pgvector.yaml /app/memledger-composition-pgvector.yaml
COPY memledger-composition-aws.yaml /app/memledger-composition-aws.yaml
RUN ln -sf /app/memledger-composition-pgvector.yaml /app/memledger-composition.yaml

# Install all dependencies at system level (no venv — container is the isolation)
RUN uv pip install --system . \
    && uv pip install --system "memledger[${MEMLEDGER_EXTRAS}]==${MEMLEDGER_VERSION}" \
    && uv pip install --system ".[memory]" \
    && uv pip install --system opentelemetry-instrumentation-langchain

# Verify packages installed correctly
RUN python -c "from kagent.langgraph import KAgentApp, KAgentCheckpointer; print('kagent OK')"
RUN python -c "from memledger import Memledger; print('memledger OK')"
RUN python -c "from opentelemetry.instrumentation.langchain import LangchainInstrumentor; print('langchain-otel OK')"

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# OTEL protocol: grpc (default) to OTEL Collector :4317 → Phoenix.
# LangChain instrumentation is wired up programmatically in app.py.

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["python", "src/app.py"]
