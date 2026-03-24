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
COPY engram-*.whl /tmp/

# Install all dependencies at system level (no venv — container is the isolation)
RUN uv pip install --system "/tmp/engram-0.1.0-py3-none-any.whl[pgvector,bedrock]" \
    && uv pip install --system ".[memory]" \
    && rm /tmp/engram-*.whl

# Verify packages installed correctly
RUN python -c "from kagent.langgraph import KAgentApp, KAgentCheckpointer; print('kagent OK')"
RUN python -c "from engram import Engram; print('engram OK')"

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
