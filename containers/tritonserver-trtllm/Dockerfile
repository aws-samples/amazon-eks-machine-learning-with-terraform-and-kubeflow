# 26.07 is the last trtllm-python-py3 tag NGC published; the variant was discontinued after it.
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.07-trtllm-python-py3
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

RUN curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.35/deb/Release.key \
  | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg \
 && chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg

RUN echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.35/deb/ /' \
  | tee /etc/apt/sources.list.d/kubernetes.list \
 && chmod 644 /etc/apt/sources.list.d/kubernetes.list

RUN apt update \
 && apt install --yes \
    kubectl \
 && apt autoremove --yes \
 && apt purge --yes \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Tag v1.2.1, matching the tensorrt_llm 1.2.1 that the base image installs. The checkpoint
# converters and build scripts under TensorRT-LLM/examples must come from the same version
# as the installed library, so bump this whenever BASE_IMAGE moves.
ARG TRTLLM_COMMIT=376f7e1bd8ed543f75014309e3fd4b237e9b0e73
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 https://github.com/NVIDIA/TensorRT-LLM.git /workspace/TensorRT-LLM
RUN cd /workspace/TensorRT-LLM && git fetch origin ${TRTLLM_COMMIT}
RUN cd /workspace/TensorRT-LLM && GIT_LFS_SKIP_SMUDGE=1 git reset --hard ${TRTLLM_COMMIT}

COPY --chmod=555 resources/kubessh /usr/local/bin/kubessh
COPY resources/server.py .