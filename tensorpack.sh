#!/bin/sh

# Add git hub authentication token below to avoid github request throttling
#export GITHUB_TOKEN=
# Create a namespace for kubeflow deployment.
NAMESPACE=kubeflow
kubectl create namespace ${NAMESPACE}

# Generate one-time ssh keys used by Open MPI.
SECRET=tensorpack-secret
mkdir -p .tmp
yes | ssh-keygen -N "" -f .tmp/id_rsa
kubectl delete secret ${SECRET} -n ${NAMESPACE} || true
kubectl create secret generic ${SECRET} -n ${NAMESPACE} --from-file=id_rsa=.tmp/id_rsa --from-file=id_rsa.pub=.tmp/id_rsa.pub --from-file=authorized_keys=.tmp/id_rsa.pub

# Which version of Kubeflow to use.
# For a list of releases refer to:
# https://github.com/kubeflow/kubeflow/releases
VERSION=v0.4.1

# Initialize a ksonnet app. Set the namespace for it's default environment.
APP_NAME=tensorpack
ks init ${APP_NAME}
cd ${APP_NAME}
ks env set default --namespace ${NAMESPACE}

# Install Kubeflow components.
ks registry add kubeflow github.com/kubeflow/kubeflow/tree/${VERSION}/kubeflow
ks pkg install kubeflow/openmpi@${VERSION}

# See the list of supported parameters.
ks prototype describe openmpi

# Generate openmpi components.
COMPONENT=tensorpack

# Customize docker image below
IMAGE=<aws-account-id>.dkr.ecr.<aws-region>.amazonaws.com/tf_tp_hvd_eks:tf1.12-hvd0.15.2-tp-860f7a3

# Customize node selector for ec2 instance type
NODE_SELECTOR='beta.kubernetes.io/instance-type=p3.16xlarge'
#VOLUMES='[{ "name": "efs", "persistentVolumeClaim": { "claimName": "pv-efs" }}, { "name": "ebs", "hostPath": { "path": "/local/faster-rcnn" , "type": "DirectoryOrCreate"}}]'
#VOLUME_MOUNTS='[{ "name": "efs", "mountPath": "/efs"}, { "name": "ebs", "mountPath": "/ebs"}]'
VOLUMES='[{ "name": "efs", "persistentVolumeClaim": { "claimName": "pv-efs" }}]'
VOLUME_MOUNTS='[{ "name": "efs", "mountPath": "/efs"}]'

# Customize number of workers
WORKERS=1
GPU=8

EXEC="/efs/run.sh"

ks generate openmpi ${COMPONENT} --image ${IMAGE} --imagePullPolicy "Always" --secret ${SECRET} --workers ${WORKERS} --gpu ${GPU} --exec "${EXEC}"  --nodeSelector "$NODE_SELECTOR" --volumes "$VOLUMES" --volumeMounts "$VOLUME_MOUNTS"

# Deploy to your cluster. 
#ks apply default

# Inspect the pod status.
#kubectl get pod -n ${NAMESPACE} -o wide
#kubectl logs -n ${NAMESPACE} -f ${COMPONENT}-master

# Clean up.
#ks delete default
