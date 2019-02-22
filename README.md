# TensorFlow + TensorPack + Horovod + Amazon EKS

## Pre-requisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)

2. [Manage your service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) so your EC2 service limit allows you to launch required number of GPU enabled EC2 instances, such as p3.16xlarge. You would need a minimum limit of 2 GPU enabled instances. For the purpose of this setup, an EC2 service limit of 8 p3.16xlarge instance types is recommended.

3. [Install and configure AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)

4. The steps described below require adequate [AWS IAM](https://docs.aws.amazon.com/IAM/latest/UserGuide/access.html) permissions.

## Overview

In this project, we are focused on distributed training using [TensorFlow](https://github.com/tensorflow/tensorflow), [TensorPack](https://github.com/tensorpack/tensorpack) and [Horovod](https://eng.uber.com/horovod/) on [Amazon EKS](https://aws.amazon.com/eks/).

While all the concepts described here are quite general and are applicable to running any combination of TensorFlow, TensorPack and Horovod based algorithms on Amazon EKS, we will make these concepts concrete by focusing on distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example. 

At a conceptual level, our objective is to:

  1. Create an Amazon EKS Cluster
  2. Stage data on an [Amazon Elastic File System (EFS)](https://aws.amazon.com/efs/)  in the same [Amazon VPC](https://aws.amazon.com/vpc/) as the EKS cluster
  3. Create EKS Persistent Volume and Peristent Volume Claim based on the EFS file-system
  4. Use [Ksonnet](https://github.com/ksonnet/ksonnet) to create an application that executes our training job using data from the EKS Persistent Volume mounted on the EKS Pods

## Create Amazon EKS Cluster VPC

As a first step, we need to create a [VPC](https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html) that supports an EKS cluster. To create such a VPC, we need to execute following steps:

   1. Customize EKS_CLUSTER and AWS_REGION variables in ```eks-cluster/set-cluster.sh``` shell script in this project. The value of EKS_CLUSTER must be a unique cluster name in the selected AWS region in your account. 
   
   2. In ```eks-cluster``` directory, execute ```./eks-cluster-vpc-stack.sh``` This script create an [AWS CloudFormation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-whatis-concepts.html#w2ab1b5c15b9) stack that creates the EKS cluster VPC. The output of the script is a CloudFormation Stack ID.
   
   3. Check the status of the CloudFormation Stack for creating VPC in AWS Management Console. When the status is CREATE_COMPLETE, note the Outputs of the CloudFormation Stack in AWS Management Console: You will need them later when you get ready to create an EKS cluster below.

## Prepare Amazon EFS File System

Next we will stage the data and code on an Amazon EFS File System that will be later accessed as a shared persistent volume from all the Kubernetes Pods used in distributed training. 

While the idea of using EFS to stage data is quite general, we will make the concept concrete by staging [Coco 2017](http://cocodataset.org/#download) dataset and [ImageNet-R50-AlignPadding](http://models.tensorpack.com/FasterRCNN/ImageNet-R50-AlignPadding.npz) pre-trained model, so we can do distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example 

To that end, we need to execute following steps:

1. In the same VPC as the EKS cluster you created above, [create a General Purpose, Bursting Amazon EFS file system](https://docs.aws.amazon.com/efs/latest/ug/gs-step-two-create-efs-resources.html). Create EFS mount points in each of the VPC subnets.

2. Using AWS Management console, in the same VPC as the EKS cluster, launch a general purpose computing EC2 instance with 200 GB storage using Ubuntu [Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/). The purpose of this instance is to mount the EFS file system created above and prepare the EFS file-system for machine-learning training.

3. Mount EFS file system on the instance created in Step 2 above at ```/efs```. 

4. In the main project directory, customize and execute ```scp run.sh user@<i3 instance>:~/``` to copy ```run.sh``` file to the i3 instance. Also, customize and execute: ```scp eks-cluster/prepare-efs.sh user@<i3 instance>:~/``` to copy ```eks-cluster/prepare-efs.sh``` to the i3 instance.

5. SSH to i3 instance: ```ssh user@<i3 instance>```

6. On the i3 instance, in the home directory, execute: ```nohup ./prepare-efs.sh &``` This step may take a while. You dont have to wait for this script to complete to proceed to next step. **You can use the [screen](https://linuxize.com/post/how-to-use-linux-screen/) command as an alternative to using ```nohup``` and ```screen``` appears to work more reliably than ```nohup``` command.**

## Create Amazon EKS Cluster

1. In AWS Management Console, using the information obtained from CloudFormation Stack Outputs when you created the EKS cluster VPC, [Create an Amazon EKS cluster](https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html). This creates an EKS cluster sans Amazon EKS worker nodes.

2. Next we install EKS kubectl client. For Linux client, in ```eks-cluster``` directory, execute: ```./install-kubectl-linux.sh``` For other operating systems, [install and configure kubectl for EKS](https://docs.aws.amazon.com/eks/latest/userguide/configure-kubectl.html).

3. *Ensure that you have at least version 1.16.73 of the AWS CLI installed, your system's Python version must be Python 3, or Python 2.7.9 or greater, and you have completed ```aws configure```. Also make sure the command ```aws-iam-authenticator help``` works.*
   
   In ```eks-cluster``` directory, execute: ```./update-kubeconfig.sh``` to update kube configuration 

4. In ```eks-cluster``` directory, execute: ```./apply-cni-patch.sh``` to apply CNI version 1.3 patch

5. To create Amazon EKS worker nodes, customize NUM_WORKERS and KEY_NAME variables in ```eks-cluster/eks-workers-stack.sh``` shell script and in ```eks-cluster``` directory execute: ```./eks-workers-stack.sh``` This script outputs a CloudFormation Stack ID for a stack that creates p3.16xlarge (or whatever instance type you specified in the script) GPU enabled EKS worker nodes we will use for distributed training. **This script uses an [EKS-optimized AMI with GPU support](https://aws.amazon.com/marketplace/pp/B07GRHFXGM). You would need to explicitly subscribe to this AMI in AWS Marketplace before you can execute ```eks-workers-stack.sh``` script.**

6. Check the status of the CloudFormation Stack in AWS Management Console. When the status is CREATE_COMPLETE, under the Outputs tab of the CloudFomration Stack in AWS Management Console, copy *NodeInstanceRole* and proceed to next step. 

7. In ```eks-cluster``` directory, customize *NodeInstanceRole* in ```aws-auth-cm.yaml``` and execute: ```./apply-aws-auth-cm.sh``` to allow worker nodes to join EKS cluster

8. In ```eks-cluster``` directory, execute: ```./apply-nvidia-plugin.sh``` to create NVIDIA-plugin daemon set


## Install ksonnet

We will use [Ksonnet](https://github.com/ksonnet/ksonnet) to manage the Kubernetes manifests needed for doing distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example in Amazon EKS. To that end, we need to install Ksonnet client on the machine you just installed EKS kubectl in the previous section.

To install Ksonnet, [download and install a pre-built ksonnet binary](https://github.com/ksonnet/ksonnet/releases) as an executable named ```ks``` under ```/usr/local/bin``` or some other directory in your PATH. If the pre-built binary option does not work for you, please see other [ksonnet install](https://github.com/ksonnet/ksonnet) options.

## Build and Upload Docker Image to ECR

We need to package TensorFlow, TensorPack and Horovod in a Docker image and upload the image to Amazon ECR. To that end, in ```container/build_tools``` directory in this project, customize for AWS region and execute: ```./build_and_push.sh``` shell script. This script creates and uploads the required Docker image to Amazon ECR in your default AWS region. It is recommended that the Docker Image be built on an EC2 instance based on [Amazon Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/).

## Create EKS Persistent Volume for EFS

1. Execute: ```kubectl create namespace kubeflow``` to create kubeflow namespace

2. In ```eks-cluster``` directory, customize ```pv-kubeflow-efs-gp-bursting.yaml``` for EFS file-system id and AWS region and execute: ``` kubectl apply -n kubeflow -f pv-kubeflow-efs-gp-bursting.yaml```

3. Check to see the persistent-volume was successfully created by executing: ```kubectl get pv -n kubeflow```

4. Execute: ```kubectl apply -n kubeflow -f pvc-kubeflow-efs-gp-bursting.yaml``` to create an EKS persistent-volume-claim

5. Check to see the persistent-volume was successfully bound to peristent-volume-claim by executing: ```kubectl get pv -n kubeflow```

## Build Ksonnet Application for Training

1. In the project folder, customize ```tensorpack.sh``` shell script to specify your IMAGE URL in ECR. You may optionally add an authentication GITHUB_TOKEN. You may customize WORKERS variable to specify number of available WORKER nodes you will like to use for training.

2. Execute: ```./tensorpack.sh``` The output of the script execution is a directory named ```tensorpack``` that contains the tensorpack Ksonnet application. 

3. In tensorpack directory created under your project, execute ```ks show default > /tmp/tensorpack.yaml``` to examine the Kubernetest manifest file corresponding to the Ksonnet appliction.

4. At this point, you need to verify that the ```prepare-efs.sh``` script has completed successfully and the data is staged on the EFS file ssytem.

5. In tensorpack directory created under your project, execute ```ks apply default``` to launch distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example.

6. Execute: ```kubectl get pods -n kubeflow``` to see the status of the pods

7. Execute: ```kubectl describe pods tensorpack-master -n kubeflow``` if the pods are in pending state

8. Execute: ```kubectl logs -f tensorpack-master -n kubeflow``` to see live log of training

9. Model checkpoints and logs will be placed on shared EFS file system
