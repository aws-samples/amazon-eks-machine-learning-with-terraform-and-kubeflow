# Co-location measurement harness (impl §6.1). CPU only -- it never touches a GPU.
#
# Runs on the `cpu-compute` nodepool, not on the g6e node. Putting the load driver
# on the node under test would have it compete for the same CPU that is feeding the
# GPUs, so the instrument would perturb the quantity it is measuring.
FROM python:3.12-slim

# transformers reaches for the network on import unless told not to. The tokenizer
# is read from the FSx mount, so any outbound call here is a failure mode, not a
# fallback -- make it fail loudly rather than hang on a DNS timeout mid-run.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY harness/ ./harness/
# Shipped in the image on purpose: it is how you verify the image that will actually
# run, not a copy of it built somewhere else.
#   docker run --rm --entrypoint python <image> -m tests.selftest
COPY tests/ ./tests/

# Non-root. The harness only reads /fsx and writes under /efs/colocation, both of
# which are world-writable RWX mounts.
RUN useradd --create-home --uid 1000 harness
USER 1000

ENTRYPOINT ["python", "-m", "harness"]
