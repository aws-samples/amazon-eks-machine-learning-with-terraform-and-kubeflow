FROM rayproject/ray:2.49.0-py311-cu128

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN pip install --upgrade pip
RUN pip install torch==2.7.1 torchvision torchaudio

RUN pip install flash-attn==2.7.4.post1 --no-build-isolation

CMD ["/bin/bash"]
