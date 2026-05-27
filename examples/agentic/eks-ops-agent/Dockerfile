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
# Dockerfile builds both v1 and v2.0 tracks. v2.0 ships three backends —
# Postgres + pgvector (OSS), Aurora + pgvector (IAM auth), and
# OpenSearch (SigV4) — picked at runtime via MEMLEDGER_CONFIG_PATH or
# direct backend config. DynamoDB + multi-backend composition land in
# v2.1; the preview lives under v2-preview/ in this directory.
#   v1 OSS:  --build-arg MEMLEDGER_VERSION=1.0.0 --build-arg MEMLEDGER_EXTRAS=pgvector,bedrock
#   v2:      defaults below
ARG MEMLEDGER_VERSION=2.0.0
ARG MEMLEDGER_EXTRAS=aws,opensearch
# MEMLEDGER_INDEX_ARGS lets you point pip/uv at Test PyPI for pre-prod
# validation (memledger 2.0.0 lands on Test PyPI before the Wed May 27
# prod publish). After May 27, leave this empty — the same wheel resolves
# from regular PyPI.
#
# `uv pip install` treats --index-url as authoritative by default and will
# refuse to consider --extra-index-url if any version of the package exists
# on the primary index. memledger 1.0.0 is on regular PyPI; 2.0.0 is only on
# Test PyPI, so we need --index-strategy=unsafe-best-match for the fallback
# to work. The flag is uv-specific; remove it for plain pip.
#   --build-arg MEMLEDGER_INDEX_ARGS="--index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match"
ARG MEMLEDGER_INDEX_ARGS=""

# Copy all source files
COPY pyproject.toml .
COPY src/ ./src/

# Backend config files. The pgvector file is the default and ships as
# /app/memledger-composition.yaml (works for in-cluster Postgres and
# Aurora-IAM via MEMLEDGER_PG_IAM_AUTH=true). The OpenSearch file is
# baked in too — point MEMLEDGER_CONFIG_PATH at it to swap backends.
COPY memledger-composition-pgvector.yaml /app/memledger-composition-pgvector.yaml
COPY memledger-composition-opensearch.yaml /app/memledger-composition-opensearch.yaml
RUN ln -sf /app/memledger-composition-pgvector.yaml /app/memledger-composition.yaml

# Install all dependencies at system level (no venv — container is the isolation).
#
# When MEMLEDGER_INDEX_ARGS pulls from Test PyPI, we pre-install Events from
# regular PyPI first. Test PyPI hosts a broken `events==1.0.0` sdist that
# uv's unsafe-best-match strategy would otherwise prefer over the working
# `Events==0.5` wheel on prod PyPI. Pinning here is harmless when
# MEMLEDGER_INDEX_ARGS is empty (post-May-27 prod publish).
RUN uv pip install --system . \
    && uv pip install --system "Events>=0.3,<1.0" \
    && uv pip install --system ${MEMLEDGER_INDEX_ARGS} "memledger[${MEMLEDGER_EXTRAS}]==${MEMLEDGER_VERSION}" \
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
