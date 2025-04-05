ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.01-vllm-python-py3
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

COPY --chmod=555 resources/kubessh /usr/local/bin/kubessh
COPY resources/server.py .