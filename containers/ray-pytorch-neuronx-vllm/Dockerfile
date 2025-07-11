FROM rayproject/ray:2.44.0-py311

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN sudo apt-get update \
      && sudo apt-get install -y --no-install-recommends \
            ca-certificates \
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
            software-properties-common \
            pkg-config \
            unzip \
            wget \
            zlib1g-dev \
            libarchive-dev \
            libxml2-dev \
            libnuma-dev \
            wget \
            gnupg2 \
      && sudo rm -rf /var/lib/apt/lists/*

RUN sudo wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB > ./GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
RUN sudo gpg --no-default-keyring --keyring ./aws_neuron_keyring.gpg --import  ./GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
RUN sudo gpg --no-default-keyring --keyring ./aws_neuron_keyring.gpg  --export >  ./aws_neuron.gpg
RUN sudo mv ./aws_neuron.gpg /etc/apt/trusted.gpg.d/
RUN sudo rm ./GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB

RUN sudo add-apt-repository -y  "deb https://apt.repos.neuron.amazonaws.com jammy main"
RUN sudo apt-get -y update

RUN sudo apt-get -y install aws-neuronx-collectives=2.*
RUN sudo apt-get -y install aws-neuronx-runtime-lib=2.*
RUN sudo apt-get -y install aws-neuronx-tools=2.*

RUN pip3 config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
RUN pip3 install awscli
RUN pip3 install neuronx-cc==2.18.* torch-neuronx==2.6.* torchvision \
      transformers-neuronx neuronx-distributed neuronx-distributed-inference \
      git+https://github.com/huggingface/optimum-neuron.git@v0.1.0
      
ENV LD_LIBRARY_PATH=/opt/aws/neuron/lib:/home/ray/anaconda3/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/aws/neuron/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PJRT_DEVICE=NEURON

ENV VLLM_TARGET_DEVICE=neuron
RUN git clone https://github.com/vllm-project/vllm.git
RUN cd vllm && git fetch origin 5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7
RUN cd vllm && git reset --hard 5fbbfe9a4c13094ad72ed3d6b4ef208a7ddc0fd7
RUN pip3 install -r vllm/requirements/neuron.txt
RUN cd vllm && pip3 install -e .

RUN pip3 install pynvml==11.5.3 transformers==4.48.3 triton==3.2.0 protobuf==3.20.3

CMD ["/bin/bash"]
