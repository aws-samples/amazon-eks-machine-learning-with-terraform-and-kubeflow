FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN git clone https://github.com/NVIDIA/apex /apex
RUN cd /apex && git fetch origin b496d85fb88a801d8e680872a12822de310951fd
RUN cd /apex && git reset --hard b496d85fb88a801d8e680872a12822de310951fd

RUN pip3 install --upgrade pip
RUN cd /apex && pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
RUN pip3 install deepspeed==0.13.4
RUN pip3 install git+https://github.com/microsoft/Megatron-DeepSpeed.git@a9856ce0e75dbe69c96d4e241e8a191b344118d7

RUN apt-get update
RUN apt-get install -y libaio-dev

CMD ["/bin/bash"]
