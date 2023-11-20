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
3. [Launch](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html) a [m5.xlarge](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/LaunchingAndUsingInstances.html) instance from [Amazon Linux 2 AMI](https://aws.amazon.com/marketplace/pp/prodview-zc4x2k7vt6rpu) using  the IAM Role created in the previous step. Use 200 GB for ```Root``` volume size. 
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
  2. Upload [COCO 2017 dataset](https://cocodataset.org/#download) to your [Amazon S3](https://aws.amazon.com/s3/) bucket
  3. Use [Terraform](https://learn.hashicorp.com/terraform) to create infrastructure
  4. (Optional) Stage training data on [Amazon EFS](https://aws.amazon.com/efs/) file-system, if you plan to use EFS
  5. Build and Upload Docker Images to [Amazon EC2 Container Registry](https://aws.amazon.com/ecr/) (ECR)
  6. Use [Helm charts](https://helm.sh/docs/developing_charts/) to launch training jobs in the EKS cluster 
  7. Use [Jupyter](https://jupyter.org/) notebook to test the trained model
  
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

### Upload COCO 2017 dataset to Amazon S3 bucket

To download COCO 2017 dataset to your build environment instance, and upload it to your Amazon S3 bucket, customize [prepare-s3-bucket.sh](eks-cluster/prepare-s3-bucket.sh) script to specify your S3 bucket in ```S3_BUCKET``` variable, and run following command:

    ./eks-cluster/prepare-s3-bucket.sh

### Use Terraform to create infrastructure

We recommend the [quick start option](#quick-start-option) for first-time walk-through.

#### Quick start option

This option creates an [Amazon EKS](https://aws.amazon.com/eks/) cluster, three [managed node groups](https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html) (```system```, ```inference```, ```training```), [Amazon EFS](https://aws.amazon.com/efs/) and [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) shared file-systems. The ```system``` node group size is fixed, and it runs the pods in the ```kube-system``` namespace. The EKS cluster uses [Cluster Autoscaler](https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html) to automatically scale the ```inference``` and ```training``` node groups up and down as needed. To complete quick-start, execute the commands below:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup
    terraform init

Substitute `S3_BUCKET` with your S3 bucket name, and run following command:

    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET"

#### Advanced option

The advanced option separates the creation of the EKS cluster from EKS managed node group for ```training```. 

##### Create EKS cluster

To create the EKS cluster, two managed node groups (```system```, ```inference```), [Amazon EFS](https://aws.amazon.com/efs/) and [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) shared file-systems, execute:
        
    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster
    terraform init

Substitute `S3_BUCKET` with your S3 bucket name and run the following command:

    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET"
   
   Save the output of the this command for creating the EKS managed node group.
   
##### Create EKS managed node group

This step creates an EKS managed node group for ```training```. Use the output of previous command for specifying ```node_role_arn```,  and ```subnet_ids``` below. Specify a unique value for ```nodegroup_name``` variable. To create the node group, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-nodegroup
    terraform init
   
    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster"  -var="node_role_arn=" -var="nodegroup_name=" -var="subnet_ids="

#### Attach to shared file-systems

The infrastructure created above includes an EFS file-system, and a FSx for Lustre file-system.

Start ```attach-pvc``` container for access to the EFS shared file-system by executing following steps:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster
    kubectl apply -f attach-pvc.yaml  -n kubeflow

Start ```attach-pvc-fsx``` container for access to the FSx for Lustre shared file-system by executing following steps:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster
    kubectl apply -f attach-pvc-fsx.yaml  -n kubeflow

### (Optional) Stage Data on EFS
The data is automatically imported from the ```S3_BUCKET``` to the FSx for Lustre file-system. 

However, if you want to use the EFS file-system, we need to stage the COCO 2017 data on EFS. To stage the data, customize ```S3_BUCKET``` variable in [stage-data.yaml](eks-cluster/stage-data.yaml), and run following command:

    kubectl apply -f stage-data.yaml -n kubeflow

Execute ```kubectl get pods -n kubeflow``` to check the status of the staging Pod. Once the status of the Pod is marked ```Completed```, data is successfully staged on EFS.

### Build and Upload Docker Images to Amazon EC2 Container Registry (ECR)

Below, we will build and push all the Docker images to Amazon ECR. Replace ```aws-region``` below, and execute:

      cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
      ./build-ecr-images.sh aws-region

### Install Helm charts for model training

#### Install mpijob chart

To deploy Kubeflow **MPIJob** *CustomResouceDefintion* using *mpijob chart*, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/charts
    helm install --debug mpijob ./mpijob/

#### Install Mask-RCNN charts
 
You have two Helm charts available for training Mask-RCNN models. Both these Helm charts use the same Kubernetes namespace, namely ```kubeflow```. Do not install both Helm charts at the same time.

To train [TensorPack Mask-RCNN](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) model, customize  [values.yaml](charts/maskrcnn/values.yaml), as described below:

1. Set ```shared_fs``` and ```data_fs``` to  ```fsx``` (default) or ```efs``` (see [Stage Data on EFS](#optional-stage-data-on-efs)). Set ```shared_pvc``` to the corresponding ```persistent-volume-claim```: ```tensorpack-fsx``` for `fsx` (default), and `tensorpack-efs-gp-bursting` for `efs`. 
2. Use [AWS check ip](http://checkip.amazonaws.com/) to get the public IP of your web browser client. Use this public IP to set ```global.source_cidr``` as a  ```/32``` CIDR. This will restrict Internet access to [Jupyter](https://jupyter.org/) notebook and [TensorBoard](https://www.tensorflow.org/tensorboard) services to your public IP. 
3. Set `tf_device_min_sys_mem_mb` to `2560`, if `node_instance_type` in your EKS cluster training node group is `p3.16xlarge`.

To password protect [TensorBoard](https://www.tensorflow.org/tensorboard), generate the password hash for your password using the command below:

    htpasswd -c .htpasswd tensorboard
   
Copy the generated password for `tensorboard` from `.htpasswd` file and set it as a quoted MD5 hash in ```charts/maskrcnn/charts/jupyter/value.yaml``` file, as shown in the example below:

    htpasswd: "your-generated-password-hash" # MD5 password hash

Finally, clean the generated password hash:

    rm .htpasswd

To install the ```maskrcnn``` chart, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/charts
    helm install --debug maskrcnn ./maskrcnn/

To train [AWS Mask-RCNN](https://github.com/aws-samples/mask-rcnn-tensorflow) optimized model, customize  [maskrcnn-optimized/values.yaml](charts/maskrcnn-optimized/values.yaml), as described below:

1. Set ```shared_fs``` and ```data_fs``` to  ```fsx``` (default) or ```efs``` (see [Stage Data on EFS](#optional-stage-data-on-efs)). Set ```shared_pvc``` to the corresponding ```persistent-volume-claim```: ```tensorpack-fsx``` for `fsx` (default), and `tensorpack-efs-gp-bursting` for `efs`. 
2. Use [AWS check ip](http://checkip.amazonaws.com/) to get the public IP of your web browser client. Use this public IP to set ```global.source_cidr``` as a  ```/32``` CIDR. This will restrict Internet access to [Jupyter](https://jupyter.org/) notebook and [TensorBoard](https://www.tensorflow.org/tensorboard) services to your public IP.
3. Set `tf_device_min_sys_mem_mb: 2560`, and `batch_size_per_gpu: 2`, if `node_instance_type` in your EKS cluster training node group is `p3.16xlarge`.

To password protect [TensorBoard](https://www.tensorflow.org/tensorboard), you **must** set ```htpasswd```  in  ```charts/maskrcnn-optimized/charts/jupyter/value.yaml``` to a quoted MD5 password hash.

To install the ```maskrcnn-optimized``` chart, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/charts
    helm install --debug maskrcnn-optimized ./maskrcnn-optimized/

### Monitor training

Note, this solution uses [EKS autoscaling](https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html) to automatically scale-up (from zero nodes) and scale-down (to zero nodes) the size of the [EKS managed nodegroup](https://docs.aws.amazon.com/eks/latest/userguide/managed-node-groups.html) used for training. So, if currently your training node group has zero nodes, it may take several minutes (or longer, if GPU capacity is transiently unavailable) for the GPU nodes to be ```Ready``` and for the training pods to reach ```Running``` state. During this time, the ```maskrcnn-launcher-xxxxx``` pod may crash and restart automatically several times, and that is nominal behavior. Once the ```maskrcnn-launcher-xxxxx``` is in ```Running``` state, replace ```xxxxx``` with your launcher pod suffix below and execute:

    kubectl logs -f maskrcnn-launcher-xxxxx -n kubeflow

This will show the live training log from the launcher pod. 

### Training logs

Model checkpoints and all training logs are also available on the ```shared_fs``` file-system  set in ```values.yaml```, i.e. ```efs``` or ```fsx```. 

If you configured your ```shared_fs``` file-system to be ```efs```, you can access your training logs by going inside the ```attach-pvc``` container as follows:

    kubectl exec --tty --stdin -n kubeflow attach-pvc bash
    cd /efs
    ls -ltr maskrcnn-*

Type ```exit``` to exit from the ```attach-pvc``` container. 

If you configured your ```shared_fs``` file-system to be ```fsx```, you can access your training by going inside the ```attach-pvc-fsx``` container as follows :

    kubectl exec --tty --stdin -n kubeflow attach-pvc-fsx bash
    cd /fsx
    ls -ltr maskrcnn-*

Type ```exit``` to exit from the ```attach-pvc-fsx``` container. 

### Test trained model
Execute ```kubectl logs -f jupyter-xxxxx -n kubeflow -c jupyter``` to display Jupyter log. At the beginning of the Jupyter log, note the **security token** required to access Jupyter service in a browser. 

Execute ```kubectl get services -n kubeflow``` to get the service DNS address. To test the trained model using a Jupyter notebook, access the service in a browser on port 443 using the service DNS and the security token.  Your URL to access the Jupyter service should look similar to the example below:

  https://xxxxxxxxxxxxxxxxxxxxxxxxx.elb.xx-xxxx-x.amazonaws.com?token=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  
Because the service endpoint in this tutorial uses a **self-signed certificate**, accessing Jupyter service in a browser will display a browser warning. If you deem it appropriate, proceed to access the service. Open the notebook, and run it it to test the trained model. Note, there may not be any trained model checkpoint available at a given time, while training is in progress.

### Visualize TensorBoard summaries
To access TensorBoard via web, use the service DNS address noted above. Your URL to access the TensorBoard service should look similar to the example below:

  https://xxxxxxxxxxxxxxxxxxxxxxxxx.elb.xx-xxxx-x.amazonaws.com:6443/
  
Accessing TensorBoard service in a browser will display a browser warning, because the service endpoint uses a **self-signed certificate**. If you deem it appropriate, proceed to access the service. When prompted for authentication, use the default username ```tensorboard```, and your password.

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


