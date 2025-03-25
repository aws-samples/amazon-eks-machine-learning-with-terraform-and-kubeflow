FROM rayproject/ray:2.44.0-py311-cu125

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN pip install --upgrade pip
RUN pip install torch==2.5.1 torchvision torchaudio

CMD ["/bin/bash"]
