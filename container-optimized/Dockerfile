FROM nvidia/cuda:10.0-devel-ubuntu16.04

ENV TF_RELEASE https://github.com/armandmcqueen/tensorpack-mask-rcnn/releases/download/v0.0.0-WIP/tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl

ENV HOROVOD_VERSION=0.15.2
ENV CUDNN_VERSION=7.4.1.5-1+cuda10.0
ENV NCCL_VERSION=2.3.7-1+cuda10.0


RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
      build-essential \
      cmake \
      git \
      curl \
      vim \
      wget \
      ca-certificates \
      libcudnn7=${CUDNN_VERSION} \
      libnccl2=${NCCL_VERSION} \
      libnccl-dev=${NCCL_VERSION} \
      libjpeg-dev \
      libpng-dev  \
      openssl \
      libssl-dev \
      unzip \
      net-tools

RUN wget https://www.python.org/ftp/python/3.6.6/Python-3.6.6.tgz
RUN tar -xvf Python-3.6.6.tgz

RUN cd Python-3.6.6 && ./configure && make && make install


RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list


RUN ln -s /usr/local/bin/python3.6 /usr/bin/python
RUN ln -s /usr/local/bin/pip3 /usr/bin/pip

RUN apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages libgtk2.0-dev





# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz && \
    tar zxf openmpi-4.0.1.tar.gz && \
    cd openmpi-4.0.1 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi


# Install TensorFlow and Keras
RUN pip3 install --upgrade pip
RUN pip3 install awscli


RUN pip3 install $TF_RELEASE keras h5py

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1  pip3 install --no-cache-dir horovod==${HOROVOD_VERSION} && \
    ldconfig

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf

# Set default NCCL parameters
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf


ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server
RUN mkdir -p /var/run/sshd && \
  mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config



RUN apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
    ethtool \
    less \
    iproute



RUN pip3 install Cython
RUN pip3 install boto3 ujson opencv-python pycocotools matplotlib
RUN pip3 install --ignore-installed numpy==1.16.2

RUN git clone https://github.com/armandmcqueen/tensorpack-mask-rcnn.git /tensorpack-mask-rcnn
RUN cd /tensorpack-mask-rcnn && git fetch origin d6b712784f9d55ffda2d8be4e0b47b54b8b5bfea
RUN cd /tensorpack-mask-rcnn && git reset --hard d6b712784f9d55ffda2d8be4e0b47b54b8b5bfea 
RUN chmod -R +w /tensorpack-mask-rcnn
RUN pip3 install --ignore-installed -e /tensorpack-mask-rcnn/
