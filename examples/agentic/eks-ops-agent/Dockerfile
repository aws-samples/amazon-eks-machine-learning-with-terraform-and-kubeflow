# EKS Ops Agent - Dockerfile
# Uses uv for fast Python dependency management

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy all source files
COPY pyproject.toml .
COPY src/ ./src/
# v2: composition config (DynamoDB+OpenSearch) is reserved for v2; see v2-preview/.

# Install all dependencies at system level (no venv — container is the isolation)
# v1 ships with pgvector + bedrock only. v2 will re-add the dynamodb + opensearch extras.
RUN uv pip install --system . \
    && uv pip install --system "memledger[pgvector,bedrock]==1.0.0" \
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
