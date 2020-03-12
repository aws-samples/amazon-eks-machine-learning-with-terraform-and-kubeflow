FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.14.0-gpu-py36-cu100-ubuntu16.04

RUN pip install --upgrade pip

RUN pip install numpy==1.17.5
RUN pip install awscli
RUN pip install boto3
RUN pip install ujson==1.35
RUN pip install opencv-python==4.1.0.25
RUN pip install Cython==0.28.4
RUN git clone https://github.com/cocodataset/cocoapi.git
RUN cd /cocoapi && git fetch origin 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9
RUN cd /cocoapi && git reset --hard 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9
RUN cd  /cocoapi/PythonAPI && make
RUN pip install -e /cocoapi/PythonAPI
RUN pip install matplotlib==3.0.3
RUN pip install markdown==3.1

RUN git clone https://github.com/tensorpack/tensorpack.git /tensorpack
RUN cd /tensorpack && git fetch origin 4ac2e22b8f4f19949b2ce67b2fa37a926d4b5816 
RUN cd /tensorpack && git reset --hard 4ac2e22b8f4f19949b2ce67b2fa37a926d4b5816 
RUN pip install -e /tensorpack 
