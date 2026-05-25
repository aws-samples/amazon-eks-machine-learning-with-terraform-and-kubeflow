# Triage Agent - Dockerfile
# Uses uv for fast Python dependency management

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# memledger version is configurable via build args (v1 vs v2). See the
# eks-ops-agent Dockerfile for the full description of the two tracks.
ARG MEMLEDGER_VERSION=2.0.0
ARG MEMLEDGER_EXTRAS=aws,dynamodb,opensearch
# Pass the Test PyPI flags here for pre-prod validation; empty after May 27.
ARG MEMLEDGER_INDEX_ARGS=""

# Copy all source files
COPY pyproject.toml .
COPY src/ ./src/

# Install all dependencies at system level (no venv — container is the isolation)
RUN uv pip install --system . \
    && uv pip install --system ${MEMLEDGER_INDEX_ARGS} "memledger[${MEMLEDGER_EXTRAS}]==${MEMLEDGER_VERSION}" \
    && uv pip install --system ".[memory]" \
    && uv pip install --system opentelemetry-instrumentation-langchain

# Verify packages installed correctly
RUN python -c "from kagent.langgraph import KAgentApp, KAgentCheckpointer; print('kagent OK')"
RUN python -c "from memledger import Memledger; print('memledger OK')"

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application
CMD ["python", "src/app.py"]
