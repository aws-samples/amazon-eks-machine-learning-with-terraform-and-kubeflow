# TensorFlow + TensorPack + Horovod + Amazon EKS

## Pre-requisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)

2. [Manage your service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) so your EC2 service limit allows you to launch required number of GPU enabled EC2 instanes, such as p3.16xlarge or p3dn.24xlarge. You would need a minimum limit of 2 GPU enabled instances. For the prupose of this setup, an EC2 service limit of 8 p3.16xlarge or p3dn.24xlarge instance types is recommended.

3. [Install and configure AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)

4. The steps described below require adequate [AWS IAM](https://docs.aws.amazon.com/IAM/latest/UserGuide/access.html) permissions.

## Create Amazon EKS Cluster

1. Before we can create an Amazon EKS cluster, we need to create a [VPC](https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html) that supports an EKS cluster. To create such a VPC, we need to execute following steps:

   i) Customize EKS_CLUSTER and AWS_REGION variables in eks-cluster/set-cluster.sh shell script in this project. The value of EKS_CLUSTER must be a unique cluster name in the selected AWS region in your account. 
   
   ii) In eks-cluster directory, execute ```./eks-cluster-vpc-stack.sh``` This script create an [AWS CloudFormation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-whatis-concepts.html#w2ab1b5c15b9) stack that creates the EKS cluster VPC. The output of the script is a CloudFormation Stack ID.
   
   iii) Check the status of the CloudFormation Stack for creating VPC in AWS Management Console. When the status is CREATE_COMPLETE, note the Outputs of the CloudFormation Stack in AWS Management Console: You will need it for the enxt step.
  
2. In AWS Management Console, [Create an Amazon EKS cluster](https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html) using the information obtained from CloudFormation Stack Outputs obtained in previous step. This creates an EKS cluster sans any Amazon EKS worker nodes.

3. To create Amazon EKS worker nodes, customize NUM_WORKERS variable in eks-cluster/eks-workers-stack.sh shell script and in eks-cluster directory execute: ```./eks-workers-stack.sh``` This script outputs a CloudFormation Stack ID for a stack that creates GPU enabled EKS worker nodes we will use for distributed training.

4. Check the status of the CloudFormation Stack in AWS Management Console. When the status is CREATE_COMPLETE, proceed to next step.

5. Next we install EKS kubectl client. For Linux client, in eks-cluster directory, execute: ```./install-kubectl-linux.sh``` For other operating systems, [install and configure kubectl for EKS](https://docs.aws.amazon.com/eks/latest/userguide/configure-kubectl.html).

6. In eks-cluster directory, execute: ```./update-kubeconfig.sh``` to update kube configuration 

7. In eks-cluster directory, execute: ```./apply-aws-auth-cm.sh``` to allow worker nodes to join EKS cluster

8. In eks-cluster directory, execute: ```./apply-nvidia-plugin.sh``` to create NVIDIA-plugin daemon set

## Prepare Amazon EFS File System

We will stage the data and code on an Amazon EFS File System that will be accessed as a shared persistent volume from all the Kubernetes Pods used for distributed training. 

While all the concepts described here are quite general, we will make them concrete by staging [Coco 2017](http://cocodataset.org/#download) dataset and [COCO-R50FPN-MaskRCNN-Standard](http://models.tensorpack.com/FasterRCNN/COCO-R50FPN-MaskRCNN-Standard.npz) pre-trained model, so we can do distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example 

To that end, we need to execute following steps:

1. In the same VPC as the EKS cluster you created above, [create a General Purpose, Bursting Amazon EFS file system](https://docs.aws.amazon.com/efs/latest/ug/gs-step-two-create-efs-resources.html). Create EFS mount points in each of the VPC subnets.

2. Using AWS Management console, in the same VPC as EKS cluster, launch an i3 EC2 instance with 200 GB storage using any linux AMI. The purpose of this instance is to mount the EFS file system created above and prepare the EFS file-system for machine-learning training.

3. Mount EFS file system on the instance created in Step 2 above at /efs. 

4. In the main project directory, customize and execute ```scp run.sh user@<i3 instance>:~/``` to copy run.sh file to the i3 instance. Also, customize and execute: ```scp eks-cluster/prepare-efs.sh user@<i3 instance>:~/``` to copy eks-cluster/prepare-efs.sh to the i3 instance.

5. SSH to i3 instance: ```sss user@<i3 instane>```

6. On the i3 instance, in the home directory, execute: ```nohup ./prepare-efs.sh &``` This step is may take a while.
