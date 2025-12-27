FROM rayproject/ray:2.52.1-py312-cu128

# Combine environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    PYTHONUNBUFFERED=1

USER root
# Install build dependencies for flash-attn (ninja helps speed up compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER ray
# Upgrade pip and install heavy dependencies in one layer
# Using --no-cache-dir keeps the image size down
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio \
    ninja

# Install flash-attn separately (it often needs torch already present)
RUN pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation

CMD ["/bin/bash"]
