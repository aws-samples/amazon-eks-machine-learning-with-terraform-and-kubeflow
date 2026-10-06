# =============================================================================
# clm-serve: the CLM System One API (projection heads + HTTP API) on CPU
# =============================================================================
# The Qwen3-8B encoder runs separately on a GPU node (vLLM pooling server); this
# image only applies CLM's ~20M-parameter projection heads to the encoder's
# embeddings and serves POST /v1/systemone and POST /v1/rank.
#
# contrastive-lm declares vLLM as a dependency for the encoder side, which this
# process never imports, so it is installed with --no-deps on top of an explicit
# CPU-only runtime set.
# =============================================================================

FROM public.ecr.aws/docker/library/python:3.12-slim

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install --no-cache-dir numpy==2.5.3 requests==2.34.2 fastapi==0.142.2 uvicorn==0.54.0 \
 && pip install --no-cache-dir --no-deps contrastive-lm==0.1.0

RUN useradd -m -u 1000 -d /home/clm clm
USER 1000
ENV HOME=/home/clm

EXPOSE 8700

CMD ["clm-serve", "--port", "8700", "--no-ui"]
