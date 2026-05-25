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

# Both composition configs ship in the image. The OSS pgvector config is the
# default at /app/memledger-composition.yaml; the AWS config is alongside it.
# Override MEMLEDGER_CONFIG_PATH on the pod to switch tracks.
COPY memledger-composition-pgvector.yaml /app/memledger-composition-pgvector.yaml
COPY memledger-composition-aws.yaml /app/memledger-composition-aws.yaml
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
