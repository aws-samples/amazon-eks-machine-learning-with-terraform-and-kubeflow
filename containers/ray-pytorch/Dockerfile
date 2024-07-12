FROM rayproject/ray:2.32.0-py311-cu123

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN pip install --upgrade pip
RUN pip install torch==2.3.0

CMD ["/bin/bash"]
