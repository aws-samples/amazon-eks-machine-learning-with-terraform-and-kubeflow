# MLOPs on Amazon EKS

This project defines a *prototypical* solution for  MLOps on [Amazon Elastic Kubernetes Service (EKS)](https://docs.aws.amazon.com/whitepapers/latest/overview-deployment-options/amazon-elastic-kubernetes-service.html). Key use cases for this solution are:

* Building a comprehensive sandbox environment for MLOps experimentation on EKS. 
* Defining a canonical *prototype* for building custom MLOPs platforms on EKS.

This solution uses a [modular](#enabling-modular-components) approach to MLOps, whereby, you can enable, or disable, various MLOPs modules, as needed. Supported ML Ops modules include: [Airflow](https://airflow.apache.org/), [Kubeflow](https://www.kubeflow.org/), [KServe](https://kserve.github.io/website/latest/), [Kueue](https://kueue.sigs.k8s.io/), [MLFlow](https://mlflow.org/), and 
[Slinky Slurm](https://github.com/slinkyproject).

For distributed training, the solution works with popular AI machine learning libraries, for example, [Nemo](https://github.com/NVIDIA/NeMo), [Hugging Face Accelerate](https://github.com/huggingface/accelerate), [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), [DeepSpeed](https://github.com/microsoft/DeepSpeed]), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed), [Ray Train](https://docs.ray.io/en/latest/train/train.html), [Neuronx Distributed](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html), among others. For distributed inference, the solution supports [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) with [vLLM](https://docs.vllm.ai/en/latest/), [Triton Inference Server](https://github.com/triton-inference-server)  with Python, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [vLLM](https://docs.vllm.ai/en/latest/) backends, and [Deep Java Library (DJL) Large Model Inference (LMI)](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html) with all [supported backends](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html).

**Legacy Note**: This project started as a companion to the [Mask R-CNN distributed training blog](https://aws.amazon.com/blogs/opensource/distributed-tensorflow-training-using-kubeflow-on-amazon-eks/), and that part of the project is documented in [this README](./tutorials/maskrcnn-blog/README.md). 

## Conceptual Overview

The key novel concept to understand is that this solution uses [Helm charts](https://helm.sh/docs/intro/using_helm/) to execute MLOps pipeline steps. See [tutorials](#tutorials) for a quick start.

Helm charts are commonly used to [deploy applications](https://docs.aws.amazon.com/eks/latest/userguide/helm.html) within an Amazon EKS cluster. While typical applications deployed using Helm charts are services that run until stopped, this solutions uses Helm Charts to execute discrete steps within arbitrary MLOps pipelines. The Helm chart based pipeline steps can be executed directly via [Helm CLI](https://helm.sh/docs/helm/), or can be used to compose MLOps pipelines using [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/), or [Apache Airflow](https://airflow.apache.org/).

Any MLOps pipeline in this solution can be conceptualized as a series of Helm chart installations, managed within a single Helm *Release*. Each step in the pipeline is executed via a Helm chart installation using a specific [YAML recipe](#yaml-recipes), whereby, the YAML recipe acts as a [Helm Values File](https://helm.sh/docs/chart_template_guide/values_files/). Once the Helm chart step completes, the Helm chart is uninstalled, and the next Helm chart in the pipeline is deployed within the same Helm *Release*. Using a single Helm *Release* for a given ML pipeline ensures that the discrete steps in the pipeline are executed atomically, and the dependency among the steps is maintained.

The included [tutorials](#tutorials) provide examples of MLOps pipelines that use pre-defined Helm charts, along with YAML recipe files that model each pipeline step. Typically, in order to define a new pipeline in this solution, you do not need to write new Helm Charts. The solution comes with a library of pre-defined [machine learning Helm charts](./charts/machine-learning/). However, you are required to write a YAML recipe file for each step in your MLOps pipeline. 

The [tutorials](#tutorials) walk you through each pipeline, step by step, where you manually execute each pipeline step by installing, and uninstalling, a pre-defined Helm chart with its associated [YAML recipe](#yaml-recipes). 

For complete end-to-end automation, we also provide an [example](./examples/training/accelerate/bert-glue-mrpc/pipeline.ipynb) where you can execute all the pipeline steps automatically using [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/concepts/). This option requires you to enable [Kubeflow platform module](#enabling-modular-components).

If you are a platform engineer, you may be interested in the [system architecture](#system-architecture) overview of this solution.

## Tutorials

After completing [prerequisites](#prerequisites), use the directory below to navigate the tutorials.

| Category      | Frameworks/Libraries |
| ----------- | ----------- |
| [Inference](./examples/inference/README.md)      | [DJL Serving](https://github.com/deepjavalibrary/djl-serving) , [RayServe](https://docs.ray.io/en/latest/serve/index.html), [Triton Inference Server](https://github.com/triton-inference-server/server)      |
| [Training](./examples/training/README.md)      | [Hugging Face Accelerate](https://github.com/huggingface/accelerate), [Lightning](https://lightning.ai/docs/pytorch/stable/), [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed), [Nemo](https://github.com/NVIDIA/NeMo), [Neuronx Distributed](https://github.com/aws-neuron/neuronx-distributed), [Neuronx Distributed Training](https://github.com/aws-neuron/neuronx-distributed-training), [RayTrain](https://docs.ray.io/en/latest/train/train.html) |
| [Legacy](./examples/legacy/README.md)      | [Mask R-CNN (TensorFlow)](https://github.com/aws-samples/mask-rcnn-tensorflow), [Neuronx Nemo Megatron](https://github.com/aws-neuron/neuronx-nemo-megatron)   |

## Prerequisites

* [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
* Select your AWS Region. For the tutorial below, we assume the region to be ```us-west-2```
* [Manage your Amazon EC2 service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) in your selected AWS Region. Increase service limits to at least 8 each for [p4d.24xlarge](https://aws.amazon.com/ec2/instance-types/p4/), [g6.xlarge, g6.2xlarge, g6.48xlarge](https://aws.amazon.com/ec2/instance-types/g5/), [inf2.xlarge, inf2.48xlarge](https://aws.amazon.com/machine-learning/inferentia/) and [trn1.32xlarge](https://aws.amazon.com/machine-learning/trainium/). If you use other Amazon EC2 GPU or AWS Trainium/Inferentia instance types in the tutorials, ensure your EC2 service limits are increased appropriately.
* If you do not already have an Amazon EC2 key pair, [create a new Amazon EC2 key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#prepare-key-pair). You will need the key pair name to specify the ```KeyName``` parameter when [launching the build machine desktop](#launch-build-machine-desktop).
* You will need an [Amazon S3](https://aws.amazon.com/s3/) bucket. If you don't have one, [create a new Amazon S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) in the AWS region you selected. The S3 bucket can be empty at this point.
* Use [AWS check ip](http://checkip.amazonaws.com/) to get your public IP address of your laptop. This will be the IP address you will need to specify the ```DesktopAccessCIDR``` parameter when creating the build machine desktop. 
* Clone this Git repository on your laptop using [```git clone ```](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository).

## Launch Build Machine Desktop

To launch the *build machine*, you will need [Administrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html) access to [AWS Management Console](https://aws.amazon.com/console/). Use the AWS CloudFormation template [ml-ops-desktop.yaml](./ml-ops-desktop.yaml) from your cloned  repository to create a new CloudFormation stack using the [ AWS Management console](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-create-stack.html), or using the [AWS CLI](https://docs.aws.amazon.com/cli/latest/reference/cloudformation/create-stack.html). 

The template [ml-ops-desktop.yaml](./ml-ops-desktop.yaml) creates [AWS Identity and Access Management (IAM)](https://aws.amazon.com/iam/) resources. If you are creating CloudFormation Stack using the console, in the review step, you must check 
**I acknowledge that AWS CloudFormation might create IAM resources.** If you use the ```aws cloudformation create-stack``` CLI, you must use ```--capabilities CAPABILITY_NAMED_IAM```. 

### Connect to Build Machine Desktop using SSH

* Once the stack status in CloudFormation console is ```CREATE_COMPLETE```, find the ML Ops desktop instance launched in your stack in the Amazon EC2 console, and [connect to the instance using SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html) as user ```ubuntu```, using your SSH key pair.
* When you connect using SSH, and you see the message ```"Cloud init in progress! Logs: /var/log/cloud-init-output.log"```, disconnect and try later after about 15 minutes. The desktop installs the Amazon DCV server on first-time startup, and reboots after the install is complete.
* If you see the message ```Amazon DCV server is enabled!```, run the command ```sudo passwd ubuntu``` to set a new password for user ```ubuntu```. Now you are ready to connect to the desktop using the [Amazon DCV client](https://docs.aws.amazon.com/dcv/latest/userguide/client.html)
* The build machine desktop uses EC2 [user-data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html) to initialize the desktop. Most *transient* failures in the desktop initialization can be fixed by rebooting the desktop.

#### Clone Git Repository

Clone this git repository on the build machine using the following commands:

    cd ~
    git clone https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow.git

#### Install Kubectl

Install ```kubectl``` on the build machine using following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./eks-cluster/utils/install-kubectl-linux.sh

### Use Terraform to Create ML Ops PLatform

 We use Terraform to create the ML Ops platform.

#### Enable S3 Backend for Terraform

Replace `S3_BUCKET`and `S3_PREFIX` with your S3 bucket name, and s3 prefix (no leading or trailing `/`), and execute the commands below

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./eks-cluster/utils/s3-backend.sh S3_BUCKET S3_PREFIX

#### Initialize Terraform
   
    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup
    terraform init

#### Apply Terraform

Logout from AWS Public ECR as otherwise `terraform apply` commands below may fail:

    docker logout public.ecr.aws

Specify at least three AWS Availability Zones from your AWS Region in `azs` below, ensuring that you  have access to your desired EC2 instance types. Replace `S3_BUCKET` with your S3 bucket name and execute:

    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET/ml-platform/"

If you need to use [AWS GPU accelerated instances](https://aws.amazon.com/ec2/instance-types/) with [AWS Elastic Fabric Adapter (EFA)](https://aws.amazon.com/hpc/efa/), you must specify an **AWS Availability Zone** for running these instances using `cuda_efa_az` variable, as shown in the example below:

    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2d","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET/ml-platform/" -var="cuda_efa_az=us-west-2c"

If you need to use [AWS Trainium instances](https://aws.amazon.com/machine-learning/trainium/), you must specify an **AWS Availability Zone** for running Trainium instances using `neuron_az` variable, as shown in the example below:

    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2d","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET/ml-platform/" -var="neuron_az=us-west-2d"

**Note:** Ensure that the AWS Availability Zone you specify for `neuron_az` or `cuda_efa_az` variable above supports requested instance types, and this zone is included in the `azs` variable.

#### Enabling Modular Components

This solution offers a suite of modular components for MLOps. All are disabled by default, and are not needed to work through included examples. You may toggle the modular components using following terraform variables:

| Component  | Terraform Variable | Default Value |
| ----------- | ----------- | ----------- |
| [Airflow](https://airflow.apache.org/) | airflow_enabled | false |
| [Kubeflow](https://www.kubeflow.org/) | kubeflow_platform_enabled | false |
| [KServe](https://kserve.github.io/website/latest/) | kserve_enabled | false |
| [Kueue](https://kueue.sigs.k8s.io/) | kueue_enabled | false |
| [MLFlow](https://mlflow.org/) | mlflow_enabled | false |
| [Nvidia DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) | dcgm_exporter_enabled | false |
| [SageMaker controller](https://github.com/aws-controllers-k8s/sagemaker-controller) | ack_sagemaker_enabled | false |
| [Slinky Slurm](https://github.com/slinkyproject) | slurm_enabled | false |


#### Retrieve Static User Password

 The static user's password is marked `sensitive` in the Terraform output. To show your static password, execute:

    terraform output static_password 

This password is used for Admin user for all web applications deployed within this solution.

### Create Home Folder on EFS and FSx for Lustre

Attach to the shared file-systems by executing following steps:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash

Inside the `attach-pvc` pod, for EFS file-system, execute:

    cd /efs
    mkdir home
    chown 1000:100 home
    exit

For Fsx for Lustre file-system, execute:

    cd /fsx
    mkdir home
    chown 1000:100 home
    exit

#### FSx for Lustre File-system Eventual Consistency with S3

FSx for Lustre file-system is configured to automatically import and export content from and to the configured S3 bucket. By default, `/fsx` is mapped to `ml-platform` top-level S3 folder in the S3 bucket. This automatic importing and exporting of content maintains *eventual consistency* between the FSx for Lustre file-system and the configured S3 bucket.

### Access Kubeflow Central Dashboard (Optional)

This section only applies if you [enable Kubeflow platform module](#enabling-modular-components).

If your web browser client machine is not the same as your build machine, before you can access Kubeflow Central Dashboard in a web browser, you must execute following steps on the your client machine:

1. [install `kubectl` client](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html)
2. [Enable IAM access](https://docs.aws.amazon.com/eks/latest/userguide/grant-k8s-access.html) to your EKS cluster. Before you execute this step, it is **highly recommended** that you backup your current configuration by executing following command on your **build machine**:

    `kubectl get configmap aws-auth -n kube-system -o yaml > ~/aws-auth.yaml`

After you have enabled IAM access to your EKS cluster, open a terminal on your client machine and start `kubectl` port-forwarding by using the local and remote ports as described below. Because we need to forward HTTPs port 
(443), we need root access to execute steps below:

For Mac:

    sudo kubectl port-forward svc/istio-ingressgateway -n ingress 443:443

For Linux:

Ensure `kubectl` is configured for `root` user by executing following commands (one time only):

    sudo su -
    aws eks update-kubeconfig --region us-west-2 --name my-eks-cluster

Connect using `kubectl port-forward`:

    kubectl port-forward svc/istio-ingressgateway -n ingress 443:443

**Note**: Leave the `kubectl port-forward` terminal open for next step below.

Next, modify your `/etc/hosts` file to add following entry:

    127.0.0.1 	istio-ingressgateway.ingress.svc.cluster.local

Open your web browser to the [KubeFlow Central Dashboard](https://istio-ingressgateway.ingress.svc.cluster.local/) URL to access the dashboard. For login, use the static username `user@example.com`, and [retrieve the static password from terraform](#retrieve-static-user-password).

**Note:** When you are not using the KubeFlow Central Dashboard, you can close the `kubectl port-forward` terminal.

### Use Terraform to Destroy ML Ops Platform

If you want to preserve any content from your EFS file-system, you must upload it to your [Amazon S3](https://aws.amazon.com/s3/) bucket, manually. The content stored on the  FSx for Lustre file-system is automatically exported to your [Amazon S3](https://aws.amazon.com/s3/) bucket under the `ml-platform` top-level folder.

Please verify your content in [Amazon S3](https://aws.amazon.com/s3/) bucket before destroying the ML Ops platform. You can recreate your ML Ops platform using the same S3 bucket. 

Use following command to check and uninstall all Helm releases:

    for x in $(helm list -q -n kubeflow-user-example-com); do echo $x; helm uninstall $x -n kubeflow-user-example-com; done

Wait at least 5 minutes for Helm uninstall to shut down all pods. Use following commands to check and delete all remaining pods in `kubeflow-user-example-com` namespace:

    kubectl get pods -n kubeflow-user-example-com
    kubectl delete --all pods -n kubeflow-user-example-com

Run following commands to delete `attach-pvc` pod:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl delete -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    
Wait 15 minutes to allow infrastructure to automatically scale down to zero.

Finally, to destroy all the infrastructure created in this tutorial, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup

    terraform destroy -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2d","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET/ml-platform/"

## Reference

### YAML Recipes

The YAML recipe file is a [Helm values](https://helm.sh/docs/chart_template_guide/values_files/) file that defines the runtime environment for a MLOps step. The key fields in the Helm values file that are common to all charts are described below:

* The `image` field specifies the required docker container image.
* The `resources` field specifies the required infrastructure resources.
* The `git` field describes the code repository we plan to use for running the job. The `git` repository is cloned into an implicitly defined location under `HOME` directory, and, the location is made available in the environment variable `GIT_CLONE_DIR`.
* The `inline_script` field is used to define an arbitrary script file.
* The `pre_script` field defines the shell script executed after cloning the `git` repository, but before launching the job.
* There is an optional `post-script` section for executing post training script.
* The training launch command and arguments are defined in the `train` field, and the data processing launch command and arguments are defined in the `process` field.
* The `pvc` field specifies the persistent volumes and their mount paths. EFS and Fsx for Lustre persistent volumes are available by default at `/efs` and `/fsx` mount paths, respectively, but these mount paths can be changed.
* The `ebs` field specifies optional [Amazon EBS](https://aws.amazon.com/ebs/) volume storage capacity and mount path. By default, no EBS volume is attached.

## System Architecture

The solution uses [Terraform](https://www.terraform.io/) to deploy [modular ML platform components](#enabling-modular-components) on top of [Amazon EKS](https://aws.amazon.com/eks/).  The hardware infrastructure is managed by [Karpenter](https://karpenter.sh/) and [Cluster Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler). Nvidia GPUs or AWS AI Chips ([AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/) and [AWS Inferentia](https://aws.amazon.com/ai/machine-learning/inferentia/)) based machines are automatically managed by [Karpenter](https://karpenter.sh/), while CPU-only machines are automatically managed by [Cluster Autoscaler](https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler).

The Kubeflow platform version that may be optionally deployed in this project is 1.9.2, and includes [Kubeflow Notebooks](https://awslabs.github.io/kubeflow-manifests/main/docs/component-guides/notebooks/), [Kubeflow Tensorboard](https://awslabs.github.io/kubeflow-manifests/main/docs/component-guides/tensorboard/). [Kubeflow Pipelines](https://awslabs.github.io/kubeflow-manifests/main/docs/component-guides/pipelines/). [Kubeflow Katib](https://www.kubeflow.org/docs/components/katib/overview/), and [Kubeflow Central Dashboard](https://www.kubeflow.org/docs/components/central-dash/).

The solution makes extensive use of [Amazon EFS](https://aws.amazon.com/efs/) and [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) shared file-systems to store the machine learning artifacts. Code, configuration, log files, and training checkpoints are stored on the EFS file-system. Data, and pre-trained model checkpoints are stored on the FSx for Lustre file system. FSx for Lustre file-system is configured to automatically import and export content from, and to, the configured S3 bucket. Any data stored on FSx for Lustre is automatically backed up to your S3 bucket.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

See the [LICENSE](./LICENSE) file.