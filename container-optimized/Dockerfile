# Use AWS Deep Learning container as base container
FROM  763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.13-horovod-gpu-py36-cu100-ubuntu16.04-v2.0

ENV HOROVOD_VERSION=0.18.1

RUN apt-get install less

RUN pip install --upgrade pip

# Need to reinstall some libraries the DL container provides due to custom Tensorflow binary
RUN pip uninstall -y tensorflow tensorboard tensorflow-estimator keras h5py horovod numpy

# Download and install custom Tensorflow binary
RUN wget https://github.com/aws-samples/mask-rcnn-tensorflow/releases/download/v0.0.0/tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl && \
    pip install tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl && \
    pip install tensorflow-estimator==1.13.0 && \
    rm tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl

RUN pip install keras h5py

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1  pip install --no-cache-dir horovod==${HOROVOD_VERSION} && \
    ldconfig


# Install OpenSSH for MPI to communicate between containers
RUN rm -rf /root/.ssh
RUN mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

RUN pip install Cython
RUN pip install ujson opencv-python pycocotools matplotlib

RUN git clone https://github.com/aws-samples/mask-rcnn-tensorflow

RUN chmod -R +w /mask-rcnn-tensorflow
RUN pip install --ignore-installed -e /mask-rcnn-tensorflow/
