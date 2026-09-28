ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.08-vllm-python-py3
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

# Ray was bundled in the base image through 25.10 but is no longer shipped as of 26.08,
# so install it explicitly. numpy is held below 2 because tritonserver requires numpy<2,
# while ray's resolver would otherwise pull numpy 2.x.
ARG RAY_VERSION=2.58.0
RUN pip install --no-cache-dir "ray[default]==${RAY_VERSION}" "numpy<2"

WORKDIR /workspace

COPY --chmod=555 resources/kubessh /usr/local/bin/kubessh
COPY resources/server.py .
ENV PYTHONPATH=/opt/tritonserver/backends/vllm