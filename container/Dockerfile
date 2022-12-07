FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.10.0-gpu-py39-cu112-ubuntu20.04-ec2

ARG CUDA=11.4
ARG CUDA_DASH=11-4
ARG CUDNN=8.2.4.15-1

RUN apt-get update && \
   apt-get install -y --no-install-recommends \
   --allow-unauthenticated --allow-downgrades  --allow-change-held-packages \
   cuda-command-line-tools-${CUDA_DASH} \
   cuda-cudart-dev-${CUDA_DASH} \
   libcufft-dev-${CUDA_DASH} \
   libcurand-dev-${CUDA_DASH} \
   libcusolver-dev-${CUDA_DASH} \
   libcusparse-dev-${CUDA_DASH} \
   libcudnn8=${CUDNN}+cuda11.4 \
   libcublas-dev-${CUDA_DASH} \
   libcublas-${CUDA_DASH}

RUN pip3 install --upgrade pip 
RUN pip3 install ujson==5.1.0
RUN pip3 install Cython==0.29.27
RUN pip3 install pycocotools==2.0.4
RUN pip3 install matplotlib==3.5.1

RUN git clone https://github.com/tensorpack/tensorpack.git /tensorpack
RUN cd /tensorpack && git fetch origin fac024f0f72fd593ea243f0b599a51b11fe4effd 
RUN cd /tensorpack && git reset --hard fac024f0f72fd593ea243f0b599a51b11fe4effd 
RUN pip3 install -e /tensorpack 

### Install Jupyter
RUN pip3 install jupyterlab==3.2.9
RUN pip3 install notebook==6.4.8 
RUN pip3 install pybind11==2.6.2
RUN pip3 install scikit-image==0.19.2
RUN pip3 install numba==0.55.1

ADD notebooks /notebooks

RUN apt-get install -y unzip
RUN apt-get install -y nginx
RUN openssl req -new -newkey rsa:2048 -days 365 -nodes -x509 -keyout /etc/ssl/domain.key -out /etc/ssl/domain.crt -subj "/CN=aws-samples/emailAddress=aws-samples@amazon.com/C=US/ST=Washington/L=Seattle/O=Amazon/OU=AWS"

RUN echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config
CMD ["/bin/bash"]
