ARG BASE_IMAGE=public.ecr.aws/neuron/pytorch-inference-neuronx:2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PJRT_DEVICE=NEURON \
    VLLM_TARGET_DEVICE=neuron \
    PATH="/opt/program:/opt/aws/neuron/bin:/opt/tritonserver/bin:${PATH}"

# 1. System Build Tools + Modern CMake Setup
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg2 && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential git nginx pkg-config unzip \
    libssl-dev libcurl4-openssl-dev libgoogle-perftools-dev \
    libnuma-dev libarchive-dev libxml2-dev zlib1g-dev \
    autoconf automake libtool gperf scons patchelf \
    libre2-dev libb64-dev rapidjson-dev \
    cmake=3.27.7* \
    cmake-data=3.27.7* \
    && rm -rf /var/lib/apt/lists/*

# 2. Boost & Python Build Essentials
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel virtualenv && \
    wget -O /tmp/boost.tar.gz https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz && \
    tar -xzf /tmp/boost.tar.gz -C /tmp && \
    mv /tmp/boost_1_80_0/boost /usr/include/boost && \
    rm -rf /tmp/boost*

# 3. Triton Core Build
RUN git clone https://github.com/triton-inference-server/server.git /server && \
    cd /server && \
    git checkout 30230a04d9f40e0ad696e2ba88b07d11a1e503b2 && \
    ./build.py -v --no-container-build --build-dir=/server/build --backend=python \
    --enable-metrics --enable-logging --enable-stats --endpoint="http" --endpoint="grpc" --filesystem="s3" && \
    cp -r /server/build/opt/* /opt/ && \
    cd / && rm -rf /server

# 4. vLLM Installation (Optimized for Neuron)
ARG USE_NEURON_VLLM=false
RUN pip3 config set global.extra-index-url https://pip.repos.neuron.amazonaws.com && \
    if [ "$USE_NEURON_VLLM" = "true" ]; then \
      pip3 install --no-cache-dir git+https://github.com/aws-neuron/upstreaming-to-vllm.git@2.26.1; \
    else \
      pip3 install --no-cache-dir git+https://github.com/vllm-project/vllm.git@v0.9.0.1; \
    fi 
    
# 5. vLLM Backend for Triton Setup
# Clone, Move, and Clean in one layer to minimize size
RUN git clone https://github.com/triton-inference-server/vllm_backend.git /tmp/vllm_backend && \
    cd /tmp/vllm_backend && \
    git checkout 98947a7c641057a1e3cf8fc9805dc4e3b93e7b66 && \
    mkdir -p /opt/tritonserver/backends/vllm && \
    cp -r src/* /opt/tritonserver/backends/vllm/ && \
    rm -rf /tmp/vllm_backend

# 6. Final Python dependencies
RUN pip3 install --no-cache-dir pydantic pandas
