FROM public.ecr.aws/lts/ubuntu:22.04_stable

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update
RUN apt-get -y install software-properties-common
RUN apt-get -y install wget
RUN apt-get -y install apt-transport-https
RUN apt-get -y install gnupg2
RUN apt-get -y install git tar zip unzip

RUN wget -qO - https://baltocdn.com/helm/signing.asc  > ./helm-ubuntu-public-key.asc
RUN gpg --no-default-keyring --keyring ./helm_keyring.gpg --import  ./helm-ubuntu-public-key.asc
RUN gpg --no-default-keyring --keyring ./helm_keyring.gpg  --export > ./helm.gpg
RUN mv ./helm.gpg /etc/apt/trusted.gpg.d/
RUN rm ./helm-ubuntu-public-key.asc

RUN add-apt-repository -y  "deb [arch=amd64] https://baltocdn.com/helm/stable/debian/ all main"
RUN apt-get update

RUN apt-get -y install helm
RUN apt-get -y install python3-minimal
RUN apt-get -y install python3-pip
RUN apt-get -y install awscli

RUN pip3 install kubernetes
RUN pip3 install boto3
RUN pip3 install pyhelm
RUN pip3 install PyYAML