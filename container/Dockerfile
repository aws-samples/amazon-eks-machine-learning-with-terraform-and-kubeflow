FROM  763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.13-horovod-gpu-py36-cu100-ubuntu16.04-v2.0

RUN pip install --upgrade pip

RUN pip install awscli
RUN pip install boto3
RUN pip install ujson==1.35
RUN pip install opencv-python==4.1.0.25
RUN pip install Cython==0.28.4
RUN pip install pycocotools==2.0.0
RUN pip install matplotlib==3.0.3
RUN pip install markdown==3.1

RUN git clone https://github.com/tensorpack/tensorpack.git /tensorpack
RUN cd /tensorpack && git fetch origin 26664c3f1d58ae029ea6c3ba0af6ae11900b1e55 
RUN cd /tensorpack && git reset --hard 26664c3f1d58ae029ea6c3ba0af6ae11900b1e55 
RUN pip install --ignore-installed -e /tensorpack
