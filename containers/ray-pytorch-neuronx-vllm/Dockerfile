FROM rayproject/ray:2.58.0-py312

# 1. Environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PJRT_DEVICE=NEURON \
    PATH=/opt/aws/neuron/bin:$PATH \
    LD_LIBRARY_PATH=/opt/aws/neuron/lib:/home/ray/anaconda3/lib:$LD_LIBRARY_PATH

# 2. Switch to root for system installations
USER root

# 3. Consolidate APT installations and Repository setup
# This reduces image size by preventing the caching of intermediate APT lists.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg2 \
    wget \
    software-properties-common \
    && wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --dearmor > /etc/apt/trusted.gpg.d/amazon-aws-neuron.gpg \
    && echo "deb https://apt.repos.neuron.amazonaws.com jammy main" > /etc/apt/sources.list.d/neuron.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    build-essential \
    git \
    gperf \
    libre2-dev \
    libssl-dev \
    libtool \
    libcurl4-openssl-dev \
    libb64-dev \
    libgoogle-perftools-dev \
    patchelf \
    python3-dev \
    python3-pip \
    python3-setuptools \
    rapidjson-dev \
    scons \
    pkg-config \
    unzip \
    zlib1g-dev \
    libarchive-dev \
    libxml2-dev \
    libnuma-dev \
    aws-neuronx-collectives=2.33.10.0-068180c7a \
    aws-neuronx-runtime-lib=2.33.10.0-3dcef56f0 \
    aws-neuronx-tools=2.31.15.0-5c7949a6a \
    && rm -rf /var/lib/apt/lists/*

# 4. Switch back to the ray user for Python package installation
USER ray

# 5. Install Python dependencies
# Fixed typo: --no-cahce-dir -> --no-cache-dir
# Combined into a single RUN to reduce layers
RUN pip install --no-cache-dir \
    --extra-index-url https://pip.repos.neuron.amazonaws.com \
    libneuronxla==2.2.17544.0+fb9962bf \
    neuronx-cc==2.26.6360.0+6f180f47 \
    neuronx-distributed==0.19.28492+435aae2b \
    neuronx-distributed-inference==0.10.18399+ed62453e

# 6. Install required packages
RUN pip install --no-cache-dir --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    torch-neuronx==2.9.0.2.15.32035+de43f57c vllm==0.16.0

RUN pip install --no-cache-dir --no-deps git+https://github.com/vllm-project/vllm-neuron.git@0.5.3


CMD ["/bin/bash"]
