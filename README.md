# Distributed TensorFlow training using Kubeflow on Amazon EKS

## Prerequisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)

2. Subscribe to the [EKS-optimized AMI with GPU Support](https://aws.amazon.com/marketplace/pp/B07GRHFXGM) from the AWS Marketplace.

3. [Manage your service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) so you can launch at least 4 EKS-optimized GPU enabled [Amazon EC2 P3](https://aws.amazon.com/ec2/instance-types/p3/) instances.

3. We need a build environment with [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html), [Terraform](https://www.terraform.io/), and [Docker](https://www.docker.com/) installed. We recommend you launch a *m5.xlarge* Amazon EC2 instance using AWS Deep Learning AMI (Ubuntu) and use this instance as the build environment for this project. All steps described under *Step by step* section below must be executed on the build environment instance.

## Step by step

While all the concepts described here are quite general, we will make these concepts concrete by focusing on distributed TensorFlow training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) model. 

The high-level outline of steps is as follows:

  1. Download [COCO 2017 dataset](http://cocodataset.org/#download) and upload it to an AWS S3 bucket
  2. Create GPU enabled [Amazon EKS](https://aws.amazon.com/eks/) cluster
  3. Create [Persistent Volume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#persistent-volumes) and [Persistent Volume Claim](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#persistentvolumeclaims) for [Amazon EFS](https://aws.amazon.com/efs/) or [Amazon FSx](https://aws.amazon.com/fsx/) file system
  4. Stage COCO 2017 data for training on Amazon EFS or FSx file system
  5. Use [Helm charts](https://helm.sh/docs/developing_charts/) to manage training jobs in EKS cluster

## Download COCO 2017 dataset and upload to Amazon S3
  1. On the build environment instance with ```aws cli``` installed, setup read-write access to your Amazon S3 object store bucket. This typically requires use of [AWS access keys](https://docs.aws.amazon.com/general/latest/gr/aws-access-keys-best-practices.html).
  2. To download COCO 2017 dataset to your build environment instance and upload it to Amazon S3 bucket, customize ```eks-cluster/prepare-s3-bucket.sh``` script to specify your S3 bucket in ```S3_BUCKET``` variable and execute ```eks-cluster/prepare-s3-bucket.sh ``` 
  
## Create GPU Enabled Amazon EKS Cluster

1. [Install Terraform](https://learn.hashicorp.com/terraform/getting-started/install.html). 
2. In ```eks-cluster/terraform/aws-eks-cluster``` folder, run ```terraform init```, ```terraform plan``` and ```terraform apply``` to create an EKS cluster. Customize Terraform variables as appropriate. Use the summary output from ```terraform apply``` for input into following steps. K8s version can be specified using ```-var="k8s_version=x.xx"```. 

    Example:
    
    ```terraform plan -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]'```
    
    ```terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]'```
3. In ```eks-cluster/terraform/aws-eks-nodegroup``` folder, run ```terraform init```, ```terraform plan``` and ```terraform  apply``` to create an EKS cluster nodegroup. Customize Terraform variables as appropriate. Use the output for input into     following steps. 

   *To create more than one nodegroup in an EKS cluster, copy ```eks-cluster/terraform/aws-eks-nodegroup``` folder to a new folder under ```eks-cluster/terraform/``` and specify a unique value for ```nodegroup_name``` variable.*
    
    Example:
    
    ```terraform plan  -var="profile=default"  -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var="efs_id=fs-xxx" -var="subnet_id=subnet-xxx" -var="key_pair=xxx" -var="cluster_sg=sg-xxx" -var="nodegroup_name=xxx"```
    
     ```terraform apply  -var="profile=default"  -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var="efs_id=fs-xxx" -var="subnet_id=subnet-xxx" -var="key_pair=xxx" -var="cluster_sg=sg-xxx"  -var="nodegroup_name=xxx"```

4. In ```eks-cluster``` directory, execute: ```./install-kubectl-linux.sh``` to install ```kubectl``` on Linux clients. For other operating systems, [install and configure kubectl for EKS](https://docs.aws.amazon.com/eks/latest/userguide/configure-kubectl.html).
5. Install aws-iam-authenticator (https://docs.aws.amazon.com/eks/latest/userguide/install-aws-iam-authenticator.html) and make sure the command ```aws-iam-authenticator help``` works. In ```eks-cluster``` directory, customize ```set-cluster.sh``` and execute: ```./update-kubeconfig.sh``` to update kube configuration.

    *Ensure that you have at least version 1.16.73 of the AWS CLI installed. Your system's Python version must be Python 3, or Python 2.7.9 or greater.*

6. [Upgrade Amazon CNI Plugin for Kubernetes](https://docs.aws.amazon.com/eks/latest/userguide/cni-upgrades.html), if needed (optional step)
7. In ```eks-cluster``` directory, customize *NodeInstanceRole* in ```aws-auth-cm.yaml``` and execute: ```./apply-aws-auth-cm.sh``` to allow worker nodes to join EKS cluster. Note, if this is not your first EKS node group, you must add the new node instance role Amazon Resource Name (ARN) to ```aws-auth-cm.yaml```, while preserving the existing role ARNs in ```aws-auth-cm.yaml```. 
8. In ```eks-cluster``` directory, execute: ```./apply-nvidia-plugin.sh``` to create NVIDIA-plugin daemon set


## Create EKS Persistent Volume

We have two shared file system options for staging data for distributed training:

1. [Amazon EFS](https://aws.amazon.com/efs/)
2. [Amazon FSx Lustre](https://aws.amazon.com/fsx/lustre/)

### Persistent Volume for EFS

1. Execute: ```kubectl create namespace kubeflow``` to create kubeflow namespace

2. In ```eks-cluster``` directory, customize ```pv-kubeflow-efs-gp-bursting.yaml``` for EFS file-system id and AWS region and execute: ``` kubectl apply -n kubeflow -f pv-kubeflow-efs-gp-bursting.yaml```

3. Check to see the persistent-volume was successfully created by executing: ```kubectl get pv -n kubeflow```

4. Execute: ```kubectl apply -n kubeflow -f pvc-kubeflow-efs-gp-bursting.yaml``` to create an EKS persistent-volume-claim

5. Check to see the persistent-volume was successfully bound to peristent-volume-claim by executing: ```kubectl get pv -n kubeflow```

### Persistent Volume for FSx

1. [Install K8s Container Storage Interface (CS) driver for Amazon FSx Lustre file system](https://github.com/aws/csi-driver-amazon-fsx) in your EKS cluster

2. Execute: ```kubectl create namespace kubeflow``` to create kubeflow namespace

3. In ```eks-cluster``` directory, customize ```pv-kubeflow-fsx.yaml``` for FSx file-system id and AWS region and execute: ``` kubectl apply -n kubeflow -f pv-kubeflow-fsx.yaml```

4. Check to see the persistent-volume was successfully created by executing: ```kubectl get pv -n kubeflow```

6. Execute: ```kubectl apply -n kubeflow -f pvc-kubeflow-fsx.yaml``` to create an EKS persistent-volume-claim

7. Check to see the persistent-volume was successfully bound to peristent-volume-claim by executing: ```kubectl get pv -n kubeflow```

## Build and Upload Docker Image to ECR

We need to package TensorFlow, TensorPack and Horovod in a Docker image and upload the image to Amazon ECR. To that end, in ```container/build_tools``` directory in this project, customize for AWS region and execute: ```./build_and_push.sh``` shell script. This script creates and uploads the required Docker image to Amazon ECR in your selected AWS region. 

### Optimized MaskRCNN

To use an [optimized version of MaskRCNN](https://github.com/armandmcqueen/tensorpack-mask-rcnn), 
go into ```container-optimized/build_tools``` directory in this project, customize AWS region and execute: ```./build_and_push.sh``` shell script. This script creates and uploads the required Docker image to Amazon ECR in your default AWS region. 

## Stage Data

Next, we stage the data that will be later accessed through a persistent volume claim from all the Kubernetes Pods used in distributed TensorFlow training. We have two shared file system options for staging data:

### Use EFS, or FSx
To stage data on EFS or FSx, customize ```eks-cluster/stage-data.yaml``` and execute ```kubectl apply -f stage-data.yaml -n kubeflow``` to stage data on selected persistent volume claim for EFS or FSX. Use the Docker image you just uploaded to ECR in ```eks-cluster/stage-data.yaml```. To verify data has been staged correctly, custmize ```eks-cluster/attach-pvc.yaml``` and execute following commands:

  ```kubectl apply -f attach-pvc.yaml -n kubeflow```  
  ```kubectl exec attach-pvc -it -n kubeflow -- /bin/bash```  

You will be attached to the EFS or FSx file system persistent volume. Type ```exit``` once you have verified the data.

### Use EBS
Alternatively, you can replicate data on all EKS worker nodes by customizing and executing 

  ```kubectl apply -f replicate-data.yaml -n kubeflow```
  
This will create a K8s DeamonSet that will run on all EKS worker nodes, pulling down data from specififed S3 bucket to ```/ebs``` directory on the host. You can safely delete the DaemonSet once data has been replicated to all nodes:
  
    kubectl delete DaemonSet replicate-data -n kubeflow
    

## Install Helm and Kubeflow

[Helm](https://helm.sh/docs/using_helm/) is package manager for Kubernetes. It uses a package format named *charts*. A Helm chart is a collection of files that define Kubernetes resources. Install helm according to instructions [here](https://helm.sh/docs/using_helm/#installing-helm).

After installing Helm, initalize Helm as described below:
  1. In ```eks-cluster``` folder, execute ```kubectl create -f tiller-rbac-config.yaml```. You should see following two messages:
  
          serviceaccount "tiller" created  
          clusterrolebinding "tiller" created  
          
  2. Execute ```helm init --service-account tiller --history-max 200```

[Kubeflow](https://www.kubeflow.org/docs/about/kubeflow/) project objective is to simplify the management of Machine Learning workflows on Kubernetes. Follow the [Kubeflow quick start guide](https://www.kubeflow.org/docs/started/getting-started/) to install Kubeflow. Installing Kubeflow is optional for this project.

## Install Helm charts for training using Kubeflow

1. In the ```charts``` folder in this project, execute ```helm install --name mpijob ./mpijob/``` to deploy Kubeflow **MPIJob** *CustomResouceDefintion* in EKS using *mpijob chart*. 

2. 
    a) In the ```charts/maskrcnn``` folder in this project, customize ```image```, ```data_fs```, ```shared_fs``` and ```shared_pvc``` variables in ```valuex.yaml```. Set ```image``` to ECR docker image URL you built and uploaded in a previous step. Set ```shared_fs``` to ```efs``` or ```fsx```, as applicable. Set ```data_fs``` to ```efs```, ```fsx``` or ```ebs```, as applicable. Set ```shared_pvc``` to the name of the k8s persistent volume you created in relevant k8s namespace. 

    b) To use an optimized version of MaskRCNN under active development, in the ```charts/maskrcnn-optimized``` folder in this project, customize ```image```, ```data_fs```, ```shared_fs``` and ```shared_pvc``` variables in ```valuex.yaml```. Set ```image``` to the optimized MaskRCNN ECR docker image URL you built and uploaded in a previous step. Set ```shared_fs``` to ```efs``` or ```fsx```, as applicable. Set ```data_fs``` to ```efs```, ```fsx``` or ```ebs```, as applicable. Set ```shared_pvc``` to the name of the k8s persistent volume you created in relevant k8s namespace.  

    c) *To create a brand new Helm chart for defining a new MPIJOb, copy ```maskrcnn``` folder to a new folder under ```charts```. Update the chart name in ```Chart.yaml```. Update the ```namespace``` global variable  in ```values.yaml``` to specify a new K8s namespace.*

3. In the ```charts``` folder in this project, execute ```helm install --name maskrcnn ./maskrcnn/``` to create the MPI Operator Deployment resource and also define an MPIJob resource for Mask-RCNN Training. 

4. Execute: ```kubectl get pods -n kubeflow``` to see the status of the pods

5. Execute: ```kubectl logs -f maskrcnn-launcher-xxxxx -n kubeflow``` to see live log of training from the launcher (change xxxxx to your specific pod name).

9. Model checkpoints and logs will be placed on the ```shared_fs``` file-system  set in ```values.yaml```, i.e. ```efs``` or ```fsx```.

## Destroy GPU enabled EKS cluster

When you are done with distributed training, you can execute ```terraform destroy``` in ```eks-cluster/terraform/eks-cluster/terraform/aws-eks-nodegroup``` folder to destroy the GPU enabled EKS nodegroup, and then execute ```terraform destroy``` in ```eks-cluster/terraform/aws-eks-cluster``` to destroy EKS cluster. Pass the same arguments to ```terraform destroy``` that you passed in the apply step above when you created the EKS cluster.

## Install ksonnet
**Deprecated. Ksonnet project is ending. Use Kubeflow MPIJob with Helm charts as described above.**

We will use [Ksonnet](https://github.com/ksonnet/ksonnet) to manage the Kubernetes manifests needed for doing distributed training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) example in Amazon EKS. To that end, we need to install Ksonnet client on the machine you just installed EKS kubectl in the previous section.

To install Ksonnet, [download and install a pre-built ksonnet binary](https://github.com/ksonnet/ksonnet/releases) as an executable named ```ks``` under ```/usr/local/bin``` or some other directory in your PATH. If the pre-built binary option does not work for you, please see other [ksonnet install](https://github.com/ksonnet/ksonnet) options.

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

