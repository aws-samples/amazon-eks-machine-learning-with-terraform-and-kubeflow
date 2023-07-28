FROM public.ecr.aws/lts/ubuntu:20.04_stable

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update
RUN apt-get -y install wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update

RUN apt-get -y install build-essential
RUN apt-get -y install gnupg2
RUN apt-get -y install git tar
RUN apt-get -y install protobuf-compiler

ARG CUDA=11.4
ARG CUDA_DASH=11-4
ARG CUDNN=8.2.2.26

RUN apt-get update
RUN apt-get -y install \
   cuda-command-line-tools-${CUDA_DASH} \
   cuda-cudart-dev-${CUDA_DASH} \
   libcufft-dev-${CUDA_DASH} \
   libcurand-dev-${CUDA_DASH} \
   libcusolver-dev-${CUDA_DASH} \
   libcusparse-dev-${CUDA_DASH} \
   libcublas-dev-${CUDA_DASH} \
   libcublas-${CUDA_DASH} \
   cuda-nvcc-${CUDA_DASH} \
   cuda-cupti-${CUDA_DASH} \
   libcupti-dev \
   libcudnn8=${CUDNN}-1+cuda${CUDA} \
   libcudnn8-dev=${CUDNN}-1+cuda${CUDA} \
   cuda-compat-11-8

RUN apt-get -y install libnccl2=2.11.4-1+cuda11.4 libnccl-dev=2.11.4-1+cuda11.4

RUN apt-get -y install python3-minimal
RUN apt-get -y install -y python3-pip

RUN pip3 install nvidia-cudnn-cu11==8.9.2.26

RUN pip3 install tensorflow==2.11.1 tensorflow-io==0.31.0

RUN apt-get -y install cmake
RUN apt-get -y install python3-setuptools

RUN apt-get -y install openmpi-common
RUN apt-get -y install openmpi-bin
RUN apt-get -y install libopenmpi-dev 
RUN apt-get -y install tzdata
RUN apt-get -y install python3-opencv
RUN apt-get -y install python3-tk 

RUN HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL pip3 install -v horovod==0.27.0
RUN pip3 install ujson==5.7.0
RUN pip3 install opencv-python==4.7.0.72
RUN pip3 install Cython==0.29.33
RUN pip3 install pycocotools==2.0.6
RUN pip3 install matplotlib==3.7.1
RUN pip3 install markdown==3.4.3
RUN pip3 install pybind11==2.10.0
RUN pip3 install scikit-image==0.19.3
RUN pip3 install numba==0.56.4
RUN pip3 install mpi4py==3.1.4
RUN pip3 install tensorboard_plugin_profile
RUN pip3 install tk==0.1.0

RUN git clone https://github.com/tensorpack/tensorpack.git /tensorpack
RUN cd /tensorpack && git fetch origin fac024f0f72fd593ea243f0b599a51b11fe4effd 
RUN cd /tensorpack && git reset --hard fac024f0f72fd593ea243f0b599a51b11fe4effd 
RUN pip3 install -e /tensorpack 

### Install Jupyter
RUN pip3 install jupyterlab==3.6.1
RUN pip3 install notebook==6.5.2

ADD notebooks /notebooks

RUN apt-get install -y unzip
RUN apt-get install -y nginx
RUN apt-get install -y openssl 
RUN openssl req -new -newkey rsa:2048 -days 365 -nodes -x509 -keyout /etc/ssl/domain.key -out /etc/ssl/domain.crt -subj "/CN=aws-samples/emailAddress=aws-samples@amazon.com/C=US/ST=Washington/L=Seattle/O=Amazon/OU=AWS"

RUN apt-get -y install openssh-server
RUN apt-get -y install openssh-client
RUN mkdir -p -m0755 /var/run/sshd
RUN systemctl enable ssh

RUN ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa
RUN cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
RUN echo "Host *" >> /root/.ssh/config
RUN echo "   StrictHostKeyChecking no" >> /root/.ssh/config
RUN echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config 

RUN echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-11-8
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.0
ENV CUDA_VERSION=${CUDA}.0
ENV TF_AUTOTUNE_THRESHOLD=2
ENV TF_NEED_CUDA=1
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.8/dist-packages/nvidia/cudnn/lib:/usr/lib/x86_64-linux-gnu/openmpi/lib
ENV PATH=/usr/local/cuda/bin:$PATH

CMD ["/bin/bash"]
