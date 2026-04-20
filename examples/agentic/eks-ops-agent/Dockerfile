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
COPY memledger-*.whl /tmp/
COPY memledger-composition.yaml /app/memledger-composition.yaml

# Install all dependencies at system level (no venv — container is the isolation)
# Composition mode needs the dynamodb + opensearch extras alongside pgvector + bedrock.
RUN uv pip install --system . \
    && uv pip install --system "/tmp/memledger-0.4.0-py3-none-any.whl[pgvector,bedrock,dynamodb,opensearch]" \
    && uv pip install --system ".[memory]" \
    && uv pip install --system aws-opentelemetry-distro opentelemetry-instrumentation-langchain \
    && rm /tmp/memledger-*.whl

# Verify packages installed correctly
RUN python -c "from kagent.langgraph import KAgentApp, KAgentCheckpointer; print('kagent OK')"
RUN python -c "from memledger import Memledger; print('memledger OK')"
RUN python -c "from opentelemetry.instrumentation.langchain import LangchainInstrumentor; print('langchain-otel OK')"

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# AWS OTEL distro env vars for AgentCore evaluation
ENV AGENT_OBSERVABILITY_ENABLED=true
ENV OTEL_PYTHON_DISTRO=aws_distro
ENV OTEL_PYTHON_CONFIGURATOR=aws_configurator
# OTEL protocol: use grpc (default) to match collector port 4317

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application with OTEL auto-instrumentation
CMD ["opentelemetry-instrument", "python", "src/app.py"]
