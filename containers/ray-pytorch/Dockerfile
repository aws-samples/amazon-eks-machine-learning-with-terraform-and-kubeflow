FROM rayproject/ray:2.42.0-py311-cu124

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN pip install --upgrade pip
RUN pip install torch==2.5.1 torchvision torchaudio

CMD ["/bin/bash"]
