FROM rayproject/ray:2.22.0.a8ab7b-py311-cu121

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN pip install --upgrade pip
RUN pip install torch==2.3.0

CMD ["/bin/bash"]
