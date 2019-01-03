#!/bin/bash

source ./set-cluster.sh

# Number of  eks workers
NUM_WORKERS=2

#Customize stack name as needed
STACK_NAME=$EKS_CLUSTER-workers

NODE_GROUP=ng1

# cfn template name
CFN_URL=https://amazon-eks.s3-us-west-2.amazonaws.com/cloudformation/2018-12-10/amazon-eks-nodegroup.yaml

# EC2 AMI for EKS worker nodes with GPU support 
# see https://docs.aws.amazon.com/eks/latest/userguide/gpu-ami.html
AMI_ID=ami-08a0bb74d1c9a5e2f

# EC2 instance type
INSTANCE_TYPE=p3.16xlarge

# EC2 key pair name
KEY_NAME=saga

# VPC ID 
VPC_ID=`aws eks --region $AWS_REGION describe-cluster --name $EKS_CLUSTER | grep vpcId | awk '{print $2}' | sed 's/,//g'`
echo "Using VpcId: $VPC_ID"

# Customize Subnet ID
SUBNETS=`aws eks --region $AWS_REGION  describe-cluster --name $EKS_CLUSTER | grep subnet- | sed 's/\"//g'| sed ':a;N;$!ba;s/\n//g' | sed 's/ //g' | sed 's/,/\\\\,/g'`
echo "Using Subnets: $SUBNETS"


CONTROL_SG=`aws eks --region $AWS_REGION  describe-cluster --name $EKS_CLUSTER | grep sg- | sed 's/ //g'`
echo "Using Cluster control security group: $CONTROL_SG"

VOLUME_SIZE=200


aws cloudformation create-stack --region $AWS_REGION  --stack-name $STACK_NAME \
--template-url $CFN_URL \
--capabilities CAPABILITY_NAMED_IAM \
--parameters \
ParameterKey=ClusterName,ParameterValue=$EKS_CLUSTER \
ParameterKey=ClusterControlPlaneSecurityGroup,ParameterValue=$CONTROL_SG \
ParameterKey=NodeGroupName,ParameterValue=$NODE_GROUP \
ParameterKey=NodeAutoScalingGroupMinSize,ParameterValue=$NUM_WORKERS \
ParameterKey=NodeAutoScalingGroupDesiredCapacity,ParameterValue=$NUM_WORKERS \
ParameterKey=NodeAutoScalingGroupMaxSize,ParameterValue=$NUM_WORKERS \
ParameterKey=NodeInstanceType,ParameterValue=$INSTANCE_TYPE \
ParameterKey=NodeImageId,ParameterValue=$AMI_ID \
ParameterKey=NodeVolumeSize,ParameterValue=$VOLUME_SIZE \
ParameterKey=KeyName,ParameterValue=$KEY_NAME \
ParameterKey=VpcId,ParameterValue=$VPC_ID \
ParameterKey=Subnets,ParameterValue=$SUBNETS
