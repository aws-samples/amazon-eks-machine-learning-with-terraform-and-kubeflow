ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt update \
 && apt install --yes \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    cgroup-tools \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.29/deb/Release.key \
  | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg \
 && chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg

RUN echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' \
  | tee /etc/apt/sources.list.d/kubernetes.list \
 && chmod 644 /etc/apt/sources.list.d/kubernetes.list

RUN apt update \
 && apt install --yes \
    kubectl \
 && apt autoremove --yes \
 && apt purge --yes \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/NVIDIA/TensorRT-LLM.git /workspace/TensorRT-LLM
RUN cd /workspace/TensorRT-LLM && git fetch origin 42a7b0922fc9e095f173eab9a7efa0bcdceadd0d
RUN cd /workspace/TensorRT-LLM && git reset --hard 42a7b0922fc9e095f173eab9a7efa0bcdceadd0d

RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend.git /workspace/tensorrtllm_backend
RUN cd /workspace/tensorrtllm_backend && git fetch origin ad209ced188cbae19c7041d968dd4e6d3800ece2
RUN cd /workspace/tensorrtllm_backend && git reset --hard ad209ced188cbae19c7041d968dd4e6d3800ece2

COPY --chmod=555 resources/kubessh .
COPY resources/server.py .