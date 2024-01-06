#!/bin/bash

pip install --upgrade pip
pip install awscli --upgrade --user

source ./set-cluster.sh
aws eks --region $AWS_REGION update-kubeconfig --name $EKS_CLUSTER
