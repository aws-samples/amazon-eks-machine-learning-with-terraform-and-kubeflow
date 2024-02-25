# Distributed training using Kubeflow on Amazon EKS

This is a tutorial on how to do [Kubeflow MPI Training](https://www.kubeflow.org/docs/components/training/mpi/) on [Amazon Elastic Kubernetes Service (EKS)](https://docs.aws.amazon.com/whitepapers/latest/overview-deployment-options/amazon-elastic-kubernetes-service.html). From a conceptual standpoint, the solution described in this tutorial is broadly applicable to any type of deep learning distributed training, and follows following outline:

1. Create required infrastructure using [Terraform](https://www.terraform.io/), with auto scaling via [Karpenter](https://karpenter.sh/), and [Cluster Autoscaler](https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html)
2. Prepare and stage the training data on [Amazon S3](https://aws.amazon.com/s3/)
3. Automatically serve Amazon S3 data using [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) file-system
3. Launch training job using [Helm Charts](https://helm.sh/docs/topics/charts/)
4. Test trained model using [Jupyter Lab](https://jupyter.org/) notebook deployed as a [Kubernetes Service](https://kubernetes.io/docs/concepts/services-networking/service/)

To make the above conceptual outline concrete, we provide a [Step by step tutorial](#step-by-step-tutorial).

## Prerequisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
2. Select your AWS Region. For the tutorial below, we assume the region to be ```us-west-2```
3. [Manage your service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) for required EC2 instances. Ensure your EC2 service limits in your selected AWS Region are set to at least 2 each for [p3.16xlarge, p3dn.24xlarge](https://aws.amazon.com/ec2/instance-types/p3/), and 2 for [g5.xlarge](https://aws.amazon.com/ec2/instance-types/g5/). If you use other EC2 instance types, ensure your EC2 service limit accordingly.

## Step by step tutorial

We make the tutorial concrete by focusing on distributed TensorFlow training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN), and [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow) models.  This tutorial has following steps:

  1. Setup the build machine
  2. Upload [COCO 2017 training dataset](https://cocodataset.org/#download) to your [Amazon S3](https://aws.amazon.com/s3/) bucket
  3. Use [Terraform](https://learn.hashicorp.com/terraform) to create the required infrastructure
  4. Build and Upload Docker Images to [Amazon EC2 Container Registry](https://aws.amazon.com/ecr/) (ECR)
  5. Use [Helm charts](https://helm.sh/docs/developing_charts/) to launch training jobs in the EKS cluster 
  6. Use [Jupyter](https://jupyter.org/) notebook to test the trained model
  
### Setup the build machine

For the *build machine*, we need [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html) and [Docker](https://www.docker.com/) installed. The AWS CLI must be configured for [Adminstrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html). You may use your laptop for your build machine if it has AWS CLI and Docker installed, or you may launch an EC2 instance for your build machine, as described below.

#### (Optional) Launch EC2 instance for the build machine 
To launch an EC2 instance for the *build machine*, you will need [Adminstrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html) access to [AWS Management Console](https://aws.amazon.com/console/). In the console, execute following steps:

1. Create an [Amazon EC2 key pair](https://docs.aws.amazon.com/en_pv/AWSEC2/latest/UserGuide/ec2-key-pairs.html) in your selected AWS region, if you do not already have one
2. Create an [AWS Service role for an EC2 instance](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_terms-and-concepts.html#iam-term-service-role-ec2), and add [AWS managed policy for Administrator access](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html#jf_administratorhttps://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html#jf_administrator) to this IAM Role.
3. [Launch](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html) a [m5.xlarge](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/LaunchingAndUsingInstances.html) instance from [Amazon Linux 2 AMI](https://aws.amazon.com/marketplace/pp/prodview-zc4x2k7vt6rpu) using  the IAM Role created in the previous step. Use 200 GB for ```Root``` volume size. 
4. After the instance state is ```Running```, [connect to your linux instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html) as ```ec2-user```. On the linux instance, install the required software tools as described below:

        sudo yum install -y docker git
        sudo systemctl enable docker.service
        sudo systemctl start docker.service
        sudo usermod -aG docker ec2-user
        exit

Now, reconnect to your linux instance. 

#### Clone git repository

Clone this git repository on the build machine using the following commands:

    cd ~
    git clone https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow.git

#### Install Kubectl

To install ```kubectl``` on Linux, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./eks-cluster/utils/install-kubectl-linux.sh

For non-Linux, [install and configure kubectl for EKS](https://docs.aws.amazon.com/eks/latest/userguide/configure-kubectl.html), install [aws-iam-authenticator](https://docs.aws.amazon.com/eks/latest/userguide/install-aws-iam-authenticator.html), and make sure the command ```aws-iam-authenticator help``` works. 

#### Install Terraform

[Install Terraform](https://learn.hashicorp.com/terraform/getting-started/install.html). Terraform configuration files in this repository are consistent with Terraform v1.1.4 syntax, but may work with other Terraform versions, as well.

#### Install Helm

[Helm](https://helm.sh/docs/intro/install/) is package manager for Kubernetes. It uses a package format named *charts*. A Helm chart is a collection of files that define Kubernetes resources. [Install helm](https://helm.sh/docs/intro/install/).

### Upload COCO 2017 dataset to Amazon S3 bucket

To download COCO 2017 dataset to your build environment instance, and upload it to your Amazon S3 bucket, replace S3_BUCKET with your bucket name and run following command:

    ./eks-cluster/utils/prepare-s3-bucket.sh S3_BUCKET

**Note:** 
In the script above, by default, data is uploaded under a top-level S3 folder named `ml-platform`. This folder is used in the `import_path` terraform variable in the section [Use Terraform to create infrastructure](#use-terraform-to-create-infrastructure). If you change this folder name, make sure to change it in both places.

### Use Terraform to create infrastructure

We use Terraform to create:

1. An [Amazon EKS](https://aws.amazon.com/eks/) cluster with [Cluster Autoscaler](https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html), and [Karpenter](https://karpenter.sh/)
2. A [managed node group](https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html) for running all pods that do not require a machine learning accelerator
3. [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) file-system
4. [Amazon EFS](https://aws.amazon.com/efs/) file-system

Set `region` to your selected AWS Region, set `cluster_name` to a unique EKS cluster name, set `azs` to your Availability Zones, replace `S3_BUCKET` with your S3 bucket name, and execute the commands below:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./eks-cluster/utils/s3-backend.sh S3_BUCKET
    
    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup
    terraform init

    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET/ml-platform"

In this solution, [Cluster Autoscaler](https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html) manages the CPU only nodes, and [Karpenter](https://karpenter.sh/) manages the GPU and AWS Neuron accelerator nodes.

### Build and Upload Docker Images to Amazon EC2 Container Registry (ECR)

Below, we will build and push all the Docker images to Amazon ECR. Replace ```aws-region``` below, and execute:

      cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
      ./build-ecr-images.sh aws-region

### Install Helm charts for model training

#### Install Mask-RCNN charts
 
You have two Helm charts available for training Mask-RCNN models. Both these Helm charts use the same Kubernetes namespace, which, by default, is set to ```kubeflow```.

To train [TensorPack Mask-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) model, customize  [values.yaml](charts/machine-learning/training/maskrcnn/values.yaml), as described below:

1. Set ```shared_fs``` and ```data_fs``` to  ```fsx``` (default) or ```efs``` (see [Stage Data on EFS](#optional-stage-data-on-efs)). Set ```shared_pvc``` to the corresponding ```persistent-volume-claim```: ```pv-fsx``` for `fsx` (default), and `pv-efs` for `efs`.  
2. Set `tf_device_min_sys_mem_mb` to `2560`, if `gpu_instance_type` is set to `p3.16xlarge`.

To install the ```maskrcnn``` chart, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug maskrcnn ./charts/machine-learning/training/maskrcnn/

To train [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow) optimized model, customize  [values.yaml](charts/machine-learning/training/maskrcnn-optimized/values.yaml), as described below:

1. Set ```shared_fs``` and ```data_fs``` to  ```fsx``` (default) or ```efs``` (see [Stage Data on EFS](#optional-stage-data-on-efs)). Set ```shared_pvc``` to the corresponding ```persistent-volume-claim```: ```pv-fsx``` for `fsx` (default), and `pv-efs` for `efs`. 
2. Set `tf_device_min_sys_mem_mb: 2560`, and `batch_size_per_gpu: 2`, if `gpu_instance_type` is set to `p3.16xlarge`.

To install the ```maskrcnn-optimized``` chart, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug maskrcnn-optimized ./charts/machine-learning/training/maskrcnn-optimized/

### Monitor training

Note, this solution uses [EKS autoscaling](https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html) to automatically scale-up (from zero nodes) and scale-down (to zero nodes) the size of the [EKS managed nodegroup](https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html) used for training. So, if currently your training node group has zero nodes, it may take several minutes (or longer, if GPU capacity is transiently unavailable) for the GPU nodes to be ```Ready``` and for the training pods to reach ```Running``` state. During this time, the ```maskrcnn-launcher-xxxxx``` pod may crash and restart automatically several times, and that is nominal behavior. Once the ```maskrcnn-launcher-xxxxx``` is in ```Running``` state, replace ```xxxxx``` with your launcher pod suffix below and execute:

    kubectl logs -f maskrcnn-launcher-xxxxx -n kubeflow

This will show the live training log from the launcher pod. 

### Training logs

Model checkpoints and all training logs are also available on the ```shared_fs``` file-system  set in ```values.yaml```, i.e. ```fsx``` (default), or `efs`.  For ```fsx``` (default), access your training logs as follows:

    kubectl apply -f eks-cluster/utils/attach-pvc-fsx.yaml -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc-fsx -- /bin/bash
    cd /fsx
    ls -ltr maskrcnn-*

Type ```exit``` to exit from the ```attach-pvc-fsx``` container. 

For ```efs```,  access your training logs as follows:

    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash
    cd /efs
    ls -ltr maskrcnn-*

Type ```exit``` to exit from the ```attach-pvc``` container. 

### Uninstall Helm charts after training
When training is complete, you may uninstall an installed chart by executing ```helm uninstall chart-name```, for example ```helm uninstall maskrcnn```. The logs and trained models will be preserved on the shared file system used for training. 

### Test trained model

#### Generate password hash

To password protect [TensorBoard](https://www.tensorflow.org/tensorboard), generate the password hash for your password using the command below:

    htpasswd -c .htpasswd tensorboard
   
Copy the generated password for `tensorboard` from `.htpasswd` file and save it to use in steps below. Finally, clean the generated password hash file:

    rm .htpasswd
    
#### Test TensorPack Mask-RCNN model

To test [TensorPack Mask-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) model, customize  [values.yaml](charts/machine-learning/testing/maskrcnn-jupyter/values.yaml), as described below:

1. Use [AWS check ip](http://checkip.amazonaws.com/) to get the public IP of your web browser client. Use this public IP to set ```global.source_cidr``` as a  ```/32``` CIDR. This will restrict Internet access to [Jupyter](https://jupyter.org/) notebook and [TensorBoard](https://www.tensorflow.org/tensorboard) services to your public IP.
2. Set `global.log_dir` to the **relative path** of your training log directory, for example, `maskrcnn-XXXX-XX-XX-XX-XX-XX`.
3. Set the generated password for `tensorboard`  as a quoted MD5 hash as shown in the example below:

    ```htpasswd: "your-generated-password-hash"```

To install the ```maskrcnn-jupyter``` chart, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug maskrcnn-jupyter ./charts/machine-learning/testing/maskrcnn-jupyter/

Execute ```kubectl logs -f maskrcnn-jupyter-xxxxx -n kubeflow -c jupyter``` to display Jupyter log. At the beginning of the Jupyter log, note the **security token** required to access Jupyter service in a browser. 

Execute ```kubectl get service maskrcnn-jupyter -n kubeflow``` to get the service DNS address. To test the trained model using a Jupyter notebook, access the service in a browser on port 443 using the service DNS and the security token.  Your URL to access the Jupyter service should look similar to the example below:

  https://xxxxxxxxxxxxxxxxxxxxxxxxx.elb.xx-xxxx-x.amazonaws.com?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  
Because the service endpoint in this tutorial uses a **self-signed certificate**, accessing Jupyter service in a browser will display a browser warning. If you deem it appropriate, proceed to access the service. Open the notebook, and run it it to test the trained model. Note, there may not be any trained model checkpoint available at a given time, while training is in progress.

To access TensorBoard via web, use the service DNS address noted above. Your URL to access the TensorBoard service should look similar to the example below:

  https://xxxxxxxxxxxxxxxxxxxxxxxxx.elb.xx-xxxx-x.amazonaws.com:6443/
  
Accessing TensorBoard service in a browser will display a browser warning, because the service endpoint uses a **self-signed certificate**. If you deem it appropriate, proceed to access the service. When prompted for authentication, use the default username ```tensorboard```, and your password.

#### Test AWS Mask-RCNN model 

To test [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow)  model, customize  [values.yaml](charts/machine-learning/testing/maskrcnn-optimized-jupyter/values.yaml) file, following the three steps shown for [TensorPack Mask-RCNN model](#test-tensorpack-mask-rcnn-model). Note, the `log_dir` will be different, for example, `maskrcnn-optimized-XXXX-XX-XX-XX-XX-XX`.

To install the ```maskrcnn-optimized-jupyter``` chart, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug maskrcnn-optimized-jupyter ./charts/machine-learning/testing/maskrcnn-optimized-jupyter/

Execute ```kubectl logs -f maskrcnn-optimized-jupyter-xxxxx -n kubeflow -c jupyter``` to display Jupyter log. At the beginning of the Jupyter log, note the **security token** required to access Jupyter service in a browser. 

Execute ```kubectl get service maskrcnn-optimized-jupyter -n kubeflow``` to get the service DNS address. The rest of the steps are the same as for [TensorPack Mask-RCNN model](#test-tensorpack-mask-rcnn-model).

### Uninstall Helm charts after testing
When testing is complete, you may uninstall an installed chart by executing ```helm uninstall chart-name```, for example ```helm uninstall maskrcnn-jupyter```, or ```helm uninstall maskrcnn-optimized-jupyter```.

### (Optional) Stage Data on EFS
The COCO 2017 training data used in the tutorial is **automatically imported** from the ```S3_BUCKET``` to the FSx for Lustre file-system. However, if you want to use the EFS file-system as the source for your training data, you need to customize ```S3_BUCKET``` variable in [stage-data.yaml](eks-cluster/utils/stage-data.yaml), and run following command:

    kubectl apply -f eks-cluster/utils/stage-data.yaml -n kubeflow

Execute ```kubectl get pods -n kubeflow``` to check the status of the staging Pod. Once the status of the Pod is marked ```Completed```, data is successfully staged on EFS.

### Use Terraform to destroy infrastructure

If you want to preserve the training output stored on the shared `fsx` or `efs` file-systems, you must upload it to your [Amazon S3](https://aws.amazon.com/s3/). To destroy all the infrastructure created in this tutorial, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup

    terraform destroy -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]'