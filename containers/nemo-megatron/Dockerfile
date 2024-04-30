FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update
RUN apt-get install -y libsndfile1 ffmpeg

RUN git clone https://github.com/NVIDIA/NeMo.git /NeMo
RUN cd /NeMo && git fetch origin b0bb807081623310c8cf0ab9a3052169fd875625
RUN cd /NeMo && git reset --hard b0bb807081623310c8cf0ab9a3052169fd875625
RUN cd /NeMo && pip install -e .

RUN cd /NeMo && pip install -r /NeMo/requirements/requirements_common.txt
RUN cd /NeMo && pip install -r /NeMo/requirements/requirements_lightning.txt
RUN cd /NeMo && pip install -r /NeMo/requirements/requirements_nlp.txt

RUN pip3 install git+https://github.com/NVIDIA/Megatron-LM.git@0d983e64afcd84cab83124e0b7ca89a3d8ec9655

CMD ["/bin/bash"]
