#!/bin/bash

source ./set-cluster.sh

STACK_NAME=$EKS_CLUSTER-vpc

CFN_URL=https://amazon-eks.s3-us-west-2.amazonaws.com/cloudformation/2018-12-10/amazon-eks-vpc-sample.yaml
SUBNET1_BLOCK=192.168.64.0/18
SUBNET2_BLOCK=192.168.128.0/18
SUBNET3_BLOCK=192.168.192.0/18
VPC_BLOCK=192.168.0.0/16


aws cloudformation create-stack --region $AWS_REGION  --stack-name $STACK_NAME \
--template-url $CFN_URL \
--parameters \
ParameterKey=Subnet01Block,ParameterValue=$SUBNET1_BLOCK \
ParameterKey=Subnet02Block,ParameterValue=$SUBNET2_BLOCK \
ParameterKey=Subnet03Block,ParameterValue=$SUBNET3_BLOCK \
ParameterKey=VpcBlock,ParameterValue=$VPC_BLOCK
