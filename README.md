# Distributed TensorFlow training using Kubeflow on Amazon EKS

## Prerequisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)

2. Subscribe to the [EKS-optimized AMI with GPU Support](https://aws.amazon.com/marketplace/pp/B07GRHFXGM) from the AWS Marketplace.

3. [Manage your service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) so you can launch at least 4 EKS-optimized GPU enabled [Amazon EC2 P3](https://aws.amazon.com/ec2/instance-types/p3/) instances.

4. Create an [AWS Service role for an EC2 instance](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_terms-and-concepts.html#iam-term-service-role-ec2) and add [AWS managed policy for power user access](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html#jf_developer-power-user) to this IAM Role.

5. We need a build environment with [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html) and [Docker](https://www.docker.com/) installed. [Launch a *m5.xlarge* Amazon EC2 instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/LaunchingAndUsingInstances.html) from an [AWS Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/) (Ubuntu) using an [EC2 instance profile](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-ec2_instance-profiles.html) containing the Role created in Step 4. All steps described under *Step by step* section below must be executed on this build environment instance.

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

Below, you only need to create Persistent Volume and Persistent Volume Claim for EFS, or FSx, not both.

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

7. Check to see the persistent-volume was successfully bound to persistent-volume-claim by executing: ```kubectl get pv -n kubeflow```

## Build and Upload Docker Image to Amazon EC2 Container Registry (ECR)

We need to package TensorFlow, TensorPack and Horovod in a Docker image and upload the image to Amazon ECR. To that end, in ```container/build_tools``` directory in this project, customize for AWS region and execute: ```./build_and_push.sh``` shell script. This script creates and uploads the required Docker image to Amazon ECR in your selected AWS region. 

### Optimized MaskRCNN

To use an [optimized version of MaskRCNN](https://github.com/armandmcqueen/tensorpack-mask-rcnn), 
go into ```container-optimized/build_tools``` directory in this project, customize AWS region and execute: ```./build_and_push.sh``` shell script. This script creates and uploads the required Docker image to Amazon ECR in your default AWS region. 

## Stage Data
Next, we stage the data that will be later accessed through a persistent volume claim from all the Kubernetes Pods used in distributed TensorFlow training. We have two shared file system options for staging data: EFS or FSx. We need to use either EFS or FSx, not both. 

### Use EFS, or FSx
To stage data on EFS or FSx, customize ```eks-cluster/stage-data.yaml``` and execute ```kubectl apply -f stage-data.yaml -n kubeflow``` to stage data on selected persistent volume claim for EFS or FSX. Use the Docker image you just uploaded to ECR in ```eks-cluster/stage-data.yaml```. To verify data has been staged correctly, custmize ```eks-cluster/attach-pvc.yaml``` and execute following commands:

  ```kubectl apply -f attach-pvc.yaml -n kubeflow```  
  ```kubectl exec attach-pvc -it -n kubeflow -- /bin/bash```  

You will be attached to the EFS or FSx file system persistent volume. Type ```exit``` once you have verified the data. 

## Install Helm

[Helm](https://helm.sh/docs/using_helm/) is package manager for Kubernetes. It uses a package format named *charts*. A Helm chart is a collection of files that define Kubernetes resources. Install helm according to instructions [here](https://helm.sh/docs/using_helm/#installing-helm).

After installing Helm, initalize Helm as described below:
  1. In ```eks-cluster``` folder, execute ```kubectl create -f tiller-rbac-config.yaml```. You should see following two messages:
  
          serviceaccount "tiller" created  
          clusterrolebinding "tiller" created  
          
  2. Execute ```helm init --service-account tiller --history-max 200```

## Release Helm charts for training

1. In the ```charts``` folder in this project, execute ```helm install --name mpijob ./mpijob/``` to deploy Kubeflow **MPIJob** *CustomResouceDefintion* in EKS using *mpijob chart*. 

2. 
    a) In the ```charts/maskrcnn``` folder in this project, customize ```image```, ```data_fs```, ```shared_fs``` and ```shared_pvc``` variables in ```valuex.yaml```. Set ```image``` to ECR docker image URL you built and uploaded in a previous step. Set ```shared_fs``` to ```efs``` or ```fsx```, as applicable. Set ```data_fs``` to ```efs```, ```fsx``` or ```ebs```, as applicable. Set ```shared_pvc``` to the name of the k8s persistent volume you created in relevant k8s namespace. 

    b) To use an optimized version of MaskRCNN under active development, in the ```charts/maskrcnn-optimized``` folder in this project, customize ```image```, ```data_fs```, ```shared_fs``` and ```shared_pvc``` variables in ```valuex.yaml```. Set ```image``` to the optimized MaskRCNN ECR docker image URL you built and uploaded in a previous step. Set ```shared_fs``` to ```efs``` or ```fsx```, as applicable. Set ```data_fs``` to ```efs```, ```fsx``` or ```ebs```, as applicable. Set ```shared_pvc``` to the name of the k8s persistent volume you created in relevant k8s namespace.  

    c) *To create a brand new Helm chart for defining a new MPIJOb, copy ```maskrcnn``` folder to a new folder under ```charts```. Update the chart name in ```Chart.yaml```. Update the ```namespace``` global variable  in ```values.yaml``` to specify a new K8s namespace.*

3. In the ```charts``` folder in this project, execute ```helm install --name maskrcnn ./maskrcnn/``` to create the MPI Operator Deployment resource and also define an MPIJob resource for Mask-RCNN Training. 

4. Execute: ```kubectl get pods -n kubeflow``` to see the status of the pods

5. Execute: ```kubectl logs -f maskrcnn-launcher-xxxxx -n kubeflow``` to see live log of training from the launcher (change xxxxx to your specific pod name).

9. Model checkpoints and logs will be placed on the ```shared_fs``` file-system  set in ```values.yaml```, i.e. ```efs``` or ```fsx```.

## Tensorboard summaries
Execute: ```kubectl get services -n kubeflow``` to get Tensorboard service DNS address. Access the Tensorboard DNS service in a browser on port 80 to visualize Tensorboard summaries.

## Purge Helm charts after training
When training is complete, yoy may purge a release by exeucting ```helm del --purge maskrcnn```. This will destroy all pods used in training, including Tensorboard service pods. However, the training output will be preserved in the EFS or FSx shared file system used for training.

## Destroy GPU enabled EKS cluster

When you are done with distributed training, you can execute ```terraform destroy``` in ```eks-cluster/terraform/aws-eks-nodegroup``` folder to destroy the GPU enabled EKS nodegroup, and then execute ```terraform destroy``` in ```eks-cluster/terraform/aws-eks-cluster``` to destroy EKS cluster. Pass the same arguments to ```terraform destroy``` commands that you passed in the ```terraform apply``` commands, respectively.

This step will not destroy the shared EFS or FSx file-system used in training.


