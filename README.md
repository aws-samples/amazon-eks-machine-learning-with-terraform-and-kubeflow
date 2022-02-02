# Distributed TensorFlow training using Kubeflow on Amazon EKS

## Prerequisites
1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
2. Select your AWS Region. For the tutorial below, we assume the region to be ```us-west-2```
3. [Manage your service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) for GPU enabled EC2 instances. We recommend service limits be set to at least 4 instances each for [p3.16xlarge, p3dn.24xlarge](https://aws.amazon.com/ec2/instance-types/p3/), [p4d.24xlarge](https://aws.amazon.com/ec2/instance-types/p4/), and [g5.48xlarge](https://aws.amazon.com/ec2/instance-types/g5/) for training, and 2 instances each for [g4dn.xlarge](https://aws.amazon.com/ec2/instance-types/g4/), and [g5.xlarge](https://aws.amazon.com/ec2/instance-types/g5/) for testing. 

### Build machine

For the *build machine*, we need [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html) and [Docker](https://www.docker.com/) installed. The AWS CLI must be configured for [Adminstrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html). You may use your laptop for your build machine if it has AWS CLI and Docker installed, or you may launch an EC2 instance for your build machine, as described below.

#### (Optional) Launch EC2 instance for the build machine 
To launch an EC2 instance for the *build machine*, you will need [Adminstrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html) access to [AWS Management Console](https://aws.amazon.com/console/). In the console, execute following steps:

1. Create an [Amazon EC2 key pair](https://docs.aws.amazon.com/en_pv/AWSEC2/latest/UserGuide/ec2-key-pairs.html) in your selected AWS region, if you do not already have one
2. Create an [AWS Service role for an EC2 instance](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_terms-and-concepts.html#iam-term-service-role-ec2), and add [AWS managed policy for Administrator access](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html#jf_administratorhttps://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html#jf_administrator) to this IAM Role.
3. [Launch](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html) a [m5.xlarge](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/LaunchingAndUsingInstances.html) instance from [Amazon Linux 2 AMI](https://aws.amazon.com/marketplace/pp/prodview-zc4x2k7vt6rpu) using  the IAM Role created in the previous step. Use 100 GB for ```Root``` volume size. 
4. After the instance state is ```Running```, [connect to your linux instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstances.html) as ```ec2-user```. On the linux instance, install the required software tools as described below:

        sudo yum install -y docker git
        sudo systemctl enable docker.service
        sudo systemctl start docker.service
        sudo usermod -aG docker ec2-user
        exit

Now, reconnect to your linux instance. All steps described under *Step by step* section below must be executed on the *build machine*.

## Step by step tutorial

While the solution described in this tutorial is general, and can be used to train and test any type of deep learning network (DNN) model, we will make the tutorial concrete by focusing on distributed TensorFlow training for [TensorPack Mask/Faster-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN), and [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow) models.  The high-level outline of solution is as follows:

  1. Setup build environment on the *build machine*
  2. Use [Terraform](https://learn.hashicorp.com/terraform) to create infrastructure
  3. Stage training data on EFS, or FSx for Lustre shared file-system
  4. Use [Helm charts](https://helm.sh/docs/developing_charts/) to launch training jobs in the EKS cluster 
  5. Use [Jupyter](https://jupyter.org/) notebook to test the trained model
  
For this tutorial, we assume the region to be ```us-west-2```. You may need to adjust the commands below if you use a different AWS Region.

### Setup build environment on build machine

#### Clone git repository

Clone this git repository on the build machine using the following commands:

    cd ~
    git clone https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow.git

#### Install Kubectl

To install ```kubectl``` on Linux, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./eks-cluster/install-kubectl-linux.sh

For non-Linux, [install and configure kubectl for EKS](https://docs.aws.amazon.com/eks/latest/userguide/configure-kubectl.html), install [aws-iam-authenticator](https://docs.aws.amazon.com/eks/latest/userguide/install-aws-iam-authenticator.html), and make sure the command ```aws-iam-authenticator help``` works. 

#### Install Terraform

[Install Terraform](https://learn.hashicorp.com/terraform/getting-started/install.html). Terraform configuration files in this repository are consistent with Terraform v1.1.4 syntax, but may work with other Terraform versions, as well.

#### Install Helm

[Helm](https://helm.sh/docs/intro/install/) is package manager for Kubernetes. It uses a package format named *charts*. A Helm chart is a collection of files that define Kubernetes resources. [Install helm](https://helm.sh/docs/intro/install/).

### Use Terraform to create infrastructre

We recommend the [quick start option](#quick-start-option) for first-time walk-through.

#### Quick start option

This option creates an [Amazon EKS](https://aws.amazon.com/eks/) cluster, three [managed node groups](https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html) (```system```, ```inference```, ```training```), [Amazon EFS](https://aws.amazon.com/efs/) and [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) shared file-systems. The ```system``` node group size is fixed, and it runs the pods in the ```kube-system``` namespace. The EKS cluster uses [Cluster Autoscaler](https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html) to automatically scale the ```inference``` and ```training``` node groups up and down as needed. To complete quick-start, execute the commands below:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup
    terraform init
    
    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]'

#### Advanced option

The advanced option separates the creation of the EKS cluster from EKS managed node group for ```training```. 

##### Create EKS cluster

To create the EKS cluster, two managed node groups (```system```, ```inference```), [Amazon EFS](https://aws.amazon.com/efs/) and [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) shared file-systems, execute:
        
    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster
    terraform init
    
    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]'
   
   Save the output of the this command for creating the EKS managed node group.
   
##### Create EKS managed node group

This step creates an EKS managed node group for ```training```. Use the output of previous command for specifying ```node_role_arn```,  and ```subnet_ids``` below. Specify a unique value for ```nodegroup_name``` variable. To create the node group, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-nodegroup
    terraform init
   
    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster"  -var="node_role_arn=" -var="nodegroup_name=" -var="subnet_ids="

### Build and Upload Docker Image to Amazon EC2 Container Registry (ECR)

#### Tensorpack Mask-RCNN

Below, we build and push the Docker images for [TensorPack Mask-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) model. 

##### Training Image
For the training container image, replace ```aws-region``` below, and execute:

      cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
      ./container/build_tools/build_and_push.sh aws-region

##### Testing Image
For the testing container image, replace ```aws-region``` below, and execute:

      cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
      ./container-viz/build_tools/build_and_push.sh aws-region

#### AWS Mask-RCNN
Below, we build and push the Docker images for [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow) model. 

##### Training Image
For the training container image, replace ```aws-region``` below, and execute:

      cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
      ./container-optimized/build_tools/build_and_push.sh aws-region

##### Testing Image
For the testing container image, replace ```aws-region``` below, and execute:

      cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
      ./container-optimized-viz/build_tools/build_and_push.sh aws-region

### Stage COCO 2017 Dataset

To download COCO 2017 dataset to your build environment instance and upload it to Amazon S3 bucket, customize ```eks-cluster/prepare-s3-bucket.sh``` script to specify your S3 bucket in ```S3_BUCKET``` variable, and execute ```eks-cluster/prepare-s3-bucket.sh ``` 
 
Next, we stage the data on EFS and FSx file-systems, so you have the option to use either one in training.

#### Stage Data on EFS, and FSx for Lustre
To stage data on EFS,  customize ```S3_BUCKET``` variable in ```eks-cluster/stage-data.yaml``` and execute:

    kubectl apply -f stage-data.yaml -n kubeflow
  
To stage data on FSx for Lustre,  customize ```S3_BUCKET``` variable in ```eks-cluster/stage-data-fsx.yaml``` and execute:

    kubectl apply -f stage-data-fsx.yaml -n kubeflow

Execute ```kubectl get pods -n kubeflow``` to check the status of the two Pods. Once the status of the two Pods is marked ```Completed```, data is successfully staged.

### Install Helm charts to begin model training

#### Install mpijob

To deploy Kubeflow **MPIJob** *CustomResouceDefintion* using *mpijob chart*, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/charts
    helm install --debug mpijob ./mpijob/

#### Install Mask-RCNN training job
 
You have three options for training Mask-RCNN model:

  a) To train [TensorPack Mask-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) model, customize  ```charts/maskrcnn/valuex.yaml```, as needed.  Set ```shared_fs``` and ```data_fs``` to ```efs```, or ```fsx```, as applicable. Set ```shared_pvc``` to the name of the respective ```persistent-volume-claim```, which is ```tensorpack-efs-gp-bursting``` for ```efs```, and ```tensorpack-fsx``` for ```fsx```. To test the trained model using a Jupyter Lab notebook, customize ```values.yaml``` in the ```charts/maskrcnn/charts/jupyter``` directory. 

  b) To train [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow) optimized model, customize  ```charts/maskrcnn-optimized/values.yaml```, as needed.  Set ```shared_fs``` and ```data_fs``` to ```efs```, or ```fsx```, as applicable. Set ```shared_pvc``` to the name of the respective ```persistent-volume-claim```. To test the trained model using a Jupyter Lab notebook, customize ```values.yaml``` in the ```charts/maskrcnn-optimized/charts/jupyter``` directory. 

  c) To create a brand new Helm chart for defining a new MPIJOb, copy ```maskrcnn``` folder to a new folder under ```charts```. Update the chart name in ```Chart.yaml```. Update the ```namespace``` global variable  in ```values.yaml``` to specify a new K8s namespace.

Install the selected Helm chart. For example, to install the ```maskrcnn``` chart, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/charts
    helm install --debug maskrcnn ./maskrcnn/

Execute ```kubectl get pods -n kubeflow``` to see the status of the pods. Execute: ```kubectl logs -f maskrcnn-launcher-xxxxx -n kubeflow``` to see live log of training from the launcher (change xxxxx to your specific pod name). Model checkpoints and logs will be placed on the ```shared_fs``` file-system  set in ```values.yaml```, i.e. ```efs``` or ```fsx```.

### Visualize Tensorboard summaries
Execute ```kubectl get services -n kubeflow``` to get Tensorboard service DNS address. Access the Tensorboard DNS service in a browser on port 80 to visualize Tensorboard summaries.

### Test trained model
Execute ```kubectl logs -f jupyter-xxxxx -n kubeflow``` to display Jupyter pod log. At the beginning of the Jupyter pod log, note the **security token** required to access Jupyter service in a browser. 

Execute ```kubectl get services -n kubeflow``` to get Jupyter service DNS address. To test the trained model using a Jupyter Lab notebook, access the Jupyter service in a browser on port 443 using the security token provided in the pod log. Your URL to access the Jupyter service should look similar to the example below:

  https://xxxxxxxxxxxxxxxxxxxxxxxxx.elb.xx-xxxx-x.amazonaws.com/lab?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  
Accessing Jupyter service in a browser will display a browser warning, because the service endpoint uses a self-signed certificate. If you deem it appropriate, it is safe to ignore the warning and proceed to access the service. Open the notebook under ```notebook``` folder, and run it it to test the trained model.

### Delete Helm charts after training
When training is complete, you may delete an installed chart by executing ```helm delete chart-name```, for example ```helm delete maskrcnn```. This will destroy all pods used in training and testing, including Tensorboard and Jupyter service pods. However, the logs and trained models will be preserved on the shared file system used for training. When you delete all the helm charts, the kubenetes cluster autoscaler may scale down the ```inference``` and ```training``` node groups to zero size.

### Use Terraform to destroy infastructure

When you are done with this tutorial, you can destory all the infrastructure, including the shared EFS, and FSx for Lustre file-systems. If you want to preserve the data on the shared file-systems, you may want to first upload it to [Amazon S3](https://aws.amazon.com/s3/).

#### Quick start option

If you used the quick start option above, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup

    terraform destroy -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]'

#### Advanced option
If you used the advanced option above, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-nodegroup
   
    terraform destroy -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster"  -var="node_role_arn=" -var="nodegroup_name=" -var="subnet_ids="

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster
    
    terraform destroy -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]'


