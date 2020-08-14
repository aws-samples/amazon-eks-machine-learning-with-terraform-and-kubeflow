FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15.2-gpu-py36-cu100-ubuntu18.04

RUN apt update
RUN apt install -y python3-opencv
RUN pip install --upgrade pip

RUN pip install numpy==1.17.5
RUN pip install awscli
RUN pip install boto3
RUN pip install ujson==1.35
RUN pip install opencv-python==4.1.0.25
RUN pip install Cython==0.28.4
RUN pip install pycocotools==2.0.1
RUN pip install matplotlib==3.0.3
RUN pip install markdown==3.1

RUN git clone https://github.com/tensorpack/tensorpack.git /tensorpack
RUN cd /tensorpack && git fetch origin 9c1b1b7b200a66fe93afd00a47f3f198bb4e1393 
RUN cd /tensorpack && git reset --hard 9c1b1b7b200a66fe93afd00a47f3f198bb4e1393 
RUN pip install -e /tensorpack 
### Install Jupyter notebook and lab for viz
RUN pip install jupyterlab
RUN pip install pybind11
RUN pip install scikit-image 
RUN pip install jupyter
RUN pip install numba

RUN apt install unzip

ADD notebooks /notebooks

RUN openssl req -new -newkey rsa:2048 -days 365 -nodes -x509 -keyout /labs-key.key -out /labs-cert.pem -subj "/CN=mask-rcnn/emailAddress=admin@example.com/C=US/ST=Washington/L=Seattle/O=Amazon/OU=AWS"
