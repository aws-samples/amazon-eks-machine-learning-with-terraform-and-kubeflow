FROM rayproject/ray:2.52.1-py312

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
    aws-neuronx-collectives=2.* \
    aws-neuronx-runtime-lib=2.* \
    aws-neuronx-tools=2.* \
    && rm -rf /var/lib/apt/lists/*

# 4. Switch back to the ray user for Python package installation
USER ray

# 5. Install Python dependencies
# Fixed typo: --no-cahce-dir -> --no-cache-dir
# Combined into a single RUN to reduce layers
RUN pip install --no-cache-dir \
    --extra-index-url https://pip.repos.neuron.amazonaws.com \
    libneuronxla==2.2.14584.0+06ac23d1 \
    neuronx-cc==2.22.12471.0+b4a00d10 \
    neuronx-distributed==0.16.25997+f431c02e \
    neuronx-distributed-inference==0.7.14366+9adb71b8 


# 6. Install required packages
RUN pip install --no-cache-dir --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    torch-neuronx==2.9.0.2.11.19912+e48cd891 vllm==0.12.0

RUN pip install --no-cache-dir --no-deps git+https://github.com/vllm-project/vllm-neuron.git@0.2.0


# 7. Patch vllm_neuron for vLLM 0.12.0 compatibility
RUN VLLM_NEURON_DIR="/home/ray/anaconda3/lib/python3.12/site-packages/vllm_neuron" && \
    sed -i 's/from vllm\.utils import make_tensor_with_pad/from vllm.utils.torch_utils import make_tensor_with_pad/g' \
    $VLLM_NEURON_DIR/worker/neuronx_distributed_model_runner.py && \
    sed -i '/block_sizes=\[self\.block_size\],/a\            kernel_block_sizes=[self.block_size],' \
    $VLLM_NEURON_DIR/worker/neuronx_distributed_model_runner.py && \
    sed -i '/default_neuron_config_args = _get_default_neuron_config(/i\    if not hasattr(scheduler_config, "max_model_len"):\n        scheduler_config.max_model_len = model_config.max_model_len\n' \
    $VLLM_NEURON_DIR/worker/neuronx_distributed_model_loader.py

CMD ["/bin/bash"]
