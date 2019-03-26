# TensorFlow + TensorPack + Horovod + Amazon EKS

## Pre-requisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)

2. [Manage your service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) so your EC2 service limit allows you to launch required number of GPU enabled EC2 instances, such as p3.16xlarge. You would need a minimum limit of 2 GPU enabled instances. For the purpose of this setup, an EC2 service limit of 8 p3.16xlarge instance types is recommended.

3. [Install and configure AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)

4. The steps described below require adequate [AWS IAM](https://docs.aws.amazon.com/IAM/latest/UserGuide/access.html) permissions.

## Overview

In this project, we are focused on distributed training using [TensorFlow](https://github.com/tensorflow/tensorflow), [TensorPack](https://github.com/tensorpack/tensorpack) and [Horovod](https://eng.uber.com/horovod/) on [Amazon EKS](https://aws.amazon.com/eks/).

While all the concepts described here are quite general and are applicable to running any combination of TensorFlow, TensorPack and Horovod based algorithms on Amazon EKS, we will make these concepts concrete by focusing on distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example. 

At a high level, we will:

  1. Create an [Amazon EKS](https://aws.amazon.com/eks/) Cluster
  2. Download [COCO 2017 dataset](http://cocodataset.org/#download) and upload it to an AWS S3 bucket
  3. Stage COCO 2017 data for training on a shared file system, or replicate it on host attached volumes
  4. Create EKS Persistent Volume and Persistent Volume Claim based on the selected shared file-system
  5. Use [Helm charts](https://helm.sh/docs/developing_charts/) to manage training jobs in EKS cluster

## Create Amazon EKS Cluster VPC

As a first step, we need to create a [VPC](https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html) that supports an EKS cluster. To create such a VPC, we need to execute following steps:

   1. Customize EKS_CLUSTER and AWS_REGION variables in ```eks-cluster/set-cluster.sh``` shell script in this project. The value of EKS_CLUSTER must be a unique cluster name in the selected AWS region in your account. 
   
   2. In ```eks-cluster``` directory, execute ```./eks-cluster-vpc-stack.sh``` This script creates an [AWS CloudFormation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-whatis-concepts.html#w2ab1b5c15b9) stack that creates the EKS cluster VPC. The output of the script is a CloudFormation Stack ID.
   
   3. Check the status of the CloudFormation Stack for creating VPC in AWS Management Console. When the status is CREATE_COMPLETE, note the Outputs of the CloudFormation Stack in AWS Management Console: You will need them later when you get ready to create an EKS cluster below.

## Download COCO 2017 dataset and upload to AWS S3
Customize ```eks-cluster/prepare-s3-bucket.sh``` for ```S3_BUCKET``` and execute ```eks-cluster/prepare-s3-bucket.sh ``` to download COCO 2017 dataset and upload it to AWS S3 bucket. 

## Stage Data

Next we stage the data that will be later accessed as a persistent volume from all the Kubernetes Pods used in distributed training. We have three data store options for staging data:

1. [Amazon EFS](https://aws.amazon.com/efs/)
2. [Amazon FSx Lustre](https://aws.amazon.com/fsx/lustre/)
3. [Amazon EBS](https://aws.amazon.com/ebs/)

We will make the concept concrete by staging [Coco 2017](http://cocodataset.org/#download) dataset and [ImageNet-R50-AlignPadding](http://models.tensorpack.com/FasterRCNN/ImageNet-R50-AlignPadding.npz) pre-trained model, so we can do distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example 

First we need to select the option we would like to use for staging data: EFS, FSx or EBS. 

EFS and FSx are both shared file-systems accessible from multiple K8s Pods running on multiple hosts. EBS provides network attached storage volumes. Each EBS volume is accessible to K8s Pods running on a single host. The performance characterstics of each file-system are different and may perform differently during training.

The steps for staging data for each file-system option are shown below.

### Amazon EFS 
1. In the same VPC as the EKS cluster you created above, [create a General Purpose, Bursting Amazon EFS file system](https://docs.aws.amazon.com/efs/latest/ug/gs-step-two-create-efs-resources.html). Create EFS mount points in each of the VPC subnets.

2. Using AWS Management console, in the same VPC as the EKS cluster, launch a general purpose computing EC2 instance with 200 GB storage using Ubuntu [Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/). The purpose of this instance is to mount the EFS file system created above and prepare the EFS file-system for machine-learning training.

3. Mount EFS file system on the general-purpose instance created in Step 2 above at ```/efs```. 

4. Execute: ```scp eks-cluster/prepare-data.sh user@<gp-instance>:~/``` to copy ```eks-cluster/prepare-data.sh``` to the home directory on general-purpose instance. 

6. SSH to the general purpose instance: ```ssh user@<gp-instance>```. Customize ```S3_BUCKET``` and ```DATA_DIR``` variables in ```eks-cluster/prepare-data.sh```.

7. On the general purpose instance, in the home directory, execute: ```nohup ./prepare-data.sh &```. You dont have to wait for this script to complete to proceed to next step. 

### Amazon FSx
1. In the same VPC as the EKS cluster you created above, [create a FSx Lustre file system](https://docs.aws.amazon.com/fsx/latest/LustreGuide/getting-started.html). 

2. Using AWS Management console, in the same VPC as the EKS cluster, launch a general purpose computing EC2 instance with 200 GB storage using Ubuntu [Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/). The purpose of this instance is to mount the FSx file system created above and prepare the FSx file-system for machine-learning training.

3. [Mount FSx file system](https://docs.aws.amazon.com/fsx/latest/LustreGuide/getting-started.html#getting-started-step2)  on the general-purpose instance created in Step 2 above at ```/fsx```. 

4. Execute: ```scp eks-cluster/prepare-data.sh user@<gp-instance>:~/``` to copy ```eks-cluster/prepare-data.sh``` to the home directory on general-purpose instance. 

6. SSH to the general purpose instance: ```ssh user@<gp-instance>```. Customize ```S3_BUCKET``` and ```DATA_DIR``` variables in ```eks-cluster/prepare-data.sh```.

7. On the general purpose instance, in the home directory, execute: ```nohup ./prepare-data.sh &```. You dont have to wait for this script to complete to proceed to next step. 

### Amazon EBS

1. Customize ```S3_BUCKET``` variable and execute ```scp eks-cluster/prepare-data.sh user@<eks-worker-instance>:~/``` to copy ```eks-cluster/prepare-data.sh``` to the home directory on each EKS worker instance where you would like to stage the data as follows:

    a. SSH to the EKS worker instance: ```ssh user@<eks-worker-instance>```

    b. On the EKS worker instance, in the home directory, execute: ```nohup ./prepare-data.sh &```. You dont have to wait for this script to complete to proceed to next step. 

### Check Security Groups

Make sure the security groups used with the EFS or FSx file-system allow access from EKS worker instances

### Copy Run Script to Shared File System
**Only if you plan to use Ksonnet, not needed for Helm Charts** 

From the root directory of this project, customize and copy ```run.sh``` to the root directory of the shared file system you selected. Note, even if you selected EBS for staging your data, you must use a shared file system for your training logs, and if applicable, for staging run script file ```run.sh```.

## Create Amazon EKS Cluster

1. In AWS Management Console, using the information obtained from CloudFormation Stack Outputs when you created the EKS cluster VPC, [Create an Amazon EKS cluster](https://docs.aws.amazon.com/eks/latest/userguide/create-cluster.html). This creates an EKS cluster sans Amazon EKS worker nodes.

2. Next we install EKS kubectl client. For Linux client, in ```eks-cluster``` directory, execute: ```./install-kubectl-linux.sh``` For other operating systems, [install and configure kubectl for EKS](https://docs.aws.amazon.com/eks/latest/userguide/configure-kubectl.html).

3. *Ensure that you have at least version 1.16.73 of the AWS CLI installed, your system's Python version must be Python 3, or Python 2.7.9 or greater, and you have completed ```aws configure```. Also make sure the command ```aws-iam-authenticator help``` works.*
   
   In ```eks-cluster``` directory, execute: ```./update-kubeconfig.sh``` to update kube configuration 

4. [Upgrade Amazon CNI Plugin for Kubernetes](https://docs.aws.amazon.com/eks/latest/userguide/cni-upgrades.html) if needed

5. To create Amazon EKS worker nodes, customize AMI_ID, CFN_URL, NUM_WORKERS, and KEY_NAME variables in ```eks-cluster/eks-workers-stack.sh``` shell script and in ```eks-cluster``` directory execute: ```./eks-workers-stack.sh``` This script outputs a CloudFormation Stack ID for a stack that creates p3.16xlarge (or whatever instance type you specified in the script) GPU enabled EKS worker nodes we will use for distributed training. **This script uses an [EKS-optimized AMI with GPU support](https://aws.amazon.com/marketplace/pp/B07GRHFXGM). You would need to explicitly subscribe to this AMI in AWS Marketplace before you can execute ```eks-workers-stack.sh``` script.**

6. Check the status of the CloudFormation Stack in AWS Management Console. When the status is CREATE_COMPLETE, under the Outputs tab of the CloudFomration Stack in AWS Management Console, copy *NodeInstanceRole* and proceed to next step. 

7. In ```eks-cluster``` directory, customize *NodeInstanceRole* in ```aws-auth-cm.yaml``` and execute: ```./apply-aws-auth-cm.sh``` to allow worker nodes to join EKS cluster

8. In ```eks-cluster``` directory, execute: ```./apply-nvidia-plugin.sh``` to create NVIDIA-plugin daemon set

## Install Helm and Kubeflow

[Helm](https://helm.sh/docs/using_helm/) is package manager for Kubernetes. It uses a package format named *charts*. A Helm chart is a collection of files that define Kubernetes resources. Install helm according to instructions [here](https://helm.sh/docs/using_helm/#installing-helm).

[Kubeflow](https://www.kubeflow.org/docs/about/kubeflow/) project objective is to simplify the management of Machine Learning workflows on Kubernetes. Follow the [Kubeflow quick start guide](https://www.kubeflow.org/docs/started/getting-started/) to install Kubeflow.

## Install ksonnet
**Deprecated. Ksonnet project is ending. Use Kubeflow MPIJob with Helm charts as described above.**

We will use [Ksonnet](https://github.com/ksonnet/ksonnet) to manage the Kubernetes manifests needed for doing distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example in Amazon EKS. To that end, we need to install Ksonnet client on the machine you just installed EKS kubectl in the previous section.

To install Ksonnet, [download and install a pre-built ksonnet binary](https://github.com/ksonnet/ksonnet/releases) as an executable named ```ks``` under ```/usr/local/bin``` or some other directory in your PATH. If the pre-built binary option does not work for you, please see other [ksonnet install](https://github.com/ksonnet/ksonnet) options.

## Build and Upload Docker Image to ECR

We need to package TensorFlow, TensorPack and Horovod in a Docker image and upload the image to Amazon ECR. To that end, in ```container/build_tools``` directory in this project, customize for AWS region and execute: ```./build_and_push.sh``` shell script. This script creates and uploads the required Docker image to Amazon ECR in your default AWS region. It is recommended that the Docker Image be built on an EC2 instance based on [Amazon Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/).

## Create EKS Persistent Volume

### Persistent Volume for EFS

1. Execute: ```kubectl create namespace kubeflow``` to create kubeflow namespace

2. In ```eks-cluster``` directory, customize ```pv-kubeflow-efs-gp-bursting.yaml``` for EFS file-system id and AWS region and execute: ``` kubectl apply -n kubeflow -f pv-kubeflow-efs-gp-bursting.yaml```

3. Check to see the persistent-volume was successfully created by executing: ```kubectl get pv -n kubeflow```

4. Execute: ```kubectl apply -n kubeflow -f pvc-kubeflow-efs-gp-bursting.yaml``` to create an EKS persistent-volume-claim

5. Check to see the persistent-volume was successfully bound to peristent-volume-claim by executing: ```kubectl get pv -n kubeflow```

### Persistent Volume for FSx

0. [Install K8s Container Storage Interface (CS) driver for Amazon FSx Lustre file system](https://github.com/aws/csi-driver-amazon-fsx) in your EKS cluster

1. Execute: ```kubectl create namespace kubeflow``` to create kubeflow namespace

2. In ```eks-cluster``` directory, customize ```pv-kubeflow-fsx.yaml``` for FSx file-system id and AWS region and execute: ``` kubectl apply -n kubeflow -f pv-kubeflow-fsx.yaml```

3. Check to see the persistent-volume was successfully created by executing: ```kubectl get pv -n kubeflow```

4. Execute: ```kubectl apply -n kubeflow -f pvc-kubeflow-fsx.yaml``` to create an EKS persistent-volume-claim

5. Check to see the persistent-volume was successfully bound to peristent-volume-claim by executing: ```kubectl get pv -n kubeflow```

## Build Kubeflow MPIJob with Helm charts for EKS Training

1. In the ```charts``` folder in this project, execute ```helm install --name mpijob ./mpijob/``` to deploy Kubeflow **MPIJob** *CustomResouceDefintion* in EKS using *mpijob chart*. 

2. In the ```charts/maskrcnn``` folder in this project, customize ```values.yaml``` for ```shared_fs``` and ```shared_pvc``` variables as needed based on the shared file system selected, i.e. EFS or FSx.  

3. In the ```charts``` folder in this project, execute ```helm install --name maskrcnn ./maskrcnn/``` to create the MPI Operator Deployment resource and also define an MPIJob resource for Mask-RCNN Training. 

4. Execute: ```kubectl get pods -n kubeflow``` to see the status of the pods

5. Execute: ```kubectl logs -f maskrcnn-launcher-xxxxx -n kubeflow``` to see live log of training from the launcher (change xxxxx to your specific pod name).

9. Model checkpoints and logs will be placed on the ```shared_fs``` file-system  set in ```values.yaml```, i.e. ```efs``` or ```fsx```.

## Build Ksonnet Application for EKS Training
**Deprecated. Ksonnet project is ending. Use Kubeflow MPIJob with Helm charts as described above.**

1. In the project folder, customize ```tensorpack.sh``` shell script to specify your IMAGE URL in ECR. You may optionally add an authentication GITHUB_TOKEN. You may customize WORKERS variable to specify number of available WORKER nodes you will like to use for training.

2. Execute: ```./tensorpack.sh``` The output of the script execution is a directory named ```tensorpack``` that contains the tensorpack Ksonnet application. 

3. In tensorpack directory created under your project, execute ```ks show default > /tmp/tensorpack.yaml``` to examine the Kubernetest manifest file corresponding to the Ksonnet appliction.

4. At this point, you need to verify that the ```prepare-efs.sh``` script has completed successfully and the data is staged on the EFS file ssytem.

5. In tensorpack directory created under your project, execute ```ks apply default``` to launch distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example.

6. Execute: ```kubectl get pods -n kubeflow``` to see the status of the pods

7. Execute: ```kubectl describe pods tensorpack-master -n kubeflow``` if the pods are in pending state

8. Execute: ```kubectl logs -f tensorpack-master -n kubeflow``` to see live log of training

9. Model checkpoints and logs will be placed on shared EFS file system

