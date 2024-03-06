FROM rayproject/ray:1486a7-py311-cu121

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN pip install --upgrade pip
RUN pip install torch==2.2.1

CMD ["/bin/bash"]
