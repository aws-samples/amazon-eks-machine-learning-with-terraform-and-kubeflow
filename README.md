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

  1. Download [COCO 2017 dataset](http://cocodataset.org/#download) and upload it to an AWS S3 bucket
  2. Create an [Amazon EKS](https://aws.amazon.com/eks/) Cluster
  3. Create EKS Persistent Volume and Persistent Volume Claim for Amazon EFS or FSx file system
  4. Stage COCO 2017 data for training on Amazon EFS or FSx file system
  5. Use [Helm charts](https://helm.sh/docs/developing_charts/) to manage training jobs in EKS cluster

## Download COCO 2017 dataset and upload to AWS S3
  1. Use an EC2 instance with ```aws cli``` installed. Setup AWS access permissions to provide read-write access to your S3  bucket. 
  2. Customize ```eks-cluster/prepare-s3-bucket.sh``` for ```S3_BUCKET``` and execute ```eks-cluster/prepare-s3-bucket.sh ``` to download COCO 2017 dataset and upload it to AWS S3 bucket. 
  
## Create Amazon EKS Cluster

0. [Install Terraform](https://learn.hashicorp.com/terraform/getting-started/install.html)

1. In ```eks-cluster/terraform/aws-eks-cluster``` folder, run ```terraform init```, ```terraform plan``` and ```terraform apply``` to create an EKS cluster. Customize Terraform variables as appropriate. Use the outout for input into following steps.

2. *This step uses an [EKS-optimized AMI with GPU support](https://aws.amazon.com/marketplace/pp/B07GRHFXGM). You would need to explicitly subscribe to this AMI in AWS Marketplace before you can execute this step.*
  
    In ```eks-cluster/terraform/aws-eks-nodegroup``` folder, run ```terraform init```, ```terraform plan``` and ```terraform     apply``` to create an EKS cluster nodegroup. Customize Terraform variables as appropriate. Use the outout for input into     following steps.

3. Next we install EKS kubectl client. For Linux client, in ```eks-cluster``` directory, execute: ```./install-kubectl-linux.sh``` For other operating systems, [install and configure kubectl for EKS](https://docs.aws.amazon.com/eks/latest/userguide/configure-kubectl.html).

3. *Ensure that you have at least version 1.16.73 of the AWS CLI installed, your system's Python version must be Python 3, or Python 2.7.9 or greater, and you have completed ```aws configure```. Also make sure the command ```aws-iam-authenticator help``` works.*
   
   In ```eks-cluster``` directory, customize ```set-cluster.sh``` and execute: ```./update-kubeconfig.sh``` to update kube configuration 

4. [Upgrade Amazon CNI Plugin for Kubernetes](https://docs.aws.amazon.com/eks/latest/userguide/cni-upgrades.html) if needed

7. In ```eks-cluster``` directory, customize *NodeInstanceRole* in ```aws-auth-cm.yaml``` and execute: ```./apply-aws-auth-cm.sh``` to allow worker nodes to join EKS cluster. Note, if this is not your first node group, you must add the new node instnace role Amazon Resource Name (ARN), while perserving the existing role ARNs in ```aws-auth-cm.yaml```. 

8. In ```eks-cluster``` directory, execute: ```./apply-nvidia-plugin.sh``` to create NVIDIA-plugin daemon set

## Create EKS Persistent Volume

We have two shared file system options for staging data for distirbuted training:

1. [Amazon EFS](https://aws.amazon.com/efs/)
2. [Amazon FSx Lustre](https://aws.amazon.com/fsx/lustre/)

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

## Build and Upload Docker Image to ECR

We need to package TensorFlow, TensorPack and Horovod in a Docker image and upload the image to Amazon ECR. To that end, in ```container/build_tools``` directory in this project, customize for AWS region and execute: ```./build_and_push.sh``` shell script. This script creates and uploads the required Docker image to Amazon ECR in your default AWS region. It is recommended that the Docker Image be built on an EC2 instance based on [Amazon Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/).

## Stage Data

Next we stage the data that will be later accessed as a persistent volume from all the Kubernetes Pods used in distributed training. Customize ```eks-cluster/stage-data.ymal``` and execute ```kubectl apply -f stage-data.yaml -n kubeflow``` to stage data on selected persistent volume claim for EFS or FSX. Use the Docker image you just uploaded to ECR.

## Install Helm and Kubeflow

[Helm](https://helm.sh/docs/using_helm/) is package manager for Kubernetes. It uses a package format named *charts*. A Helm chart is a collection of files that define Kubernetes resources. Install helm according to instructions [here](https://helm.sh/docs/using_helm/#installing-helm).

[Kubeflow](https://www.kubeflow.org/docs/about/kubeflow/) project objective is to simplify the management of Machine Learning workflows on Kubernetes. Follow the [Kubeflow quick start guide](https://www.kubeflow.org/docs/started/getting-started/) to install Kubeflow.

## Install ksonnet
**Deprecated. Ksonnet project is ending. Use Kubeflow MPIJob with Helm charts as described above.**

We will use [Ksonnet](https://github.com/ksonnet/ksonnet) to manage the Kubernetes manifests needed for doing distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example in Amazon EKS. To that end, we need to install Ksonnet client on the machine you just installed EKS kubectl in the previous section.

To install Ksonnet, [download and install a pre-built ksonnet binary](https://github.com/ksonnet/ksonnet/releases) as an executable named ```ks``` under ```/usr/local/bin``` or some other directory in your PATH. If the pre-built binary option does not work for you, please see other [ksonnet install](https://github.com/ksonnet/ksonnet) options.


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

4. In tensorpack directory created under your project, execute ```ks apply default``` to launch distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example.

5. Execute: ```kubectl get pods -n kubeflow``` to see the status of the pods

6. Execute: ```kubectl describe pods tensorpack-master -n kubeflow``` if the pods are in pending state

7. Execute: ```kubectl logs -f tensorpack-master -n kubeflow``` to see live log of training

8. Model checkpoints and logs will be placed on shared EFS file system

