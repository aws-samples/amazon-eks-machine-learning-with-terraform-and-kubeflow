# Distributed Training and Inference on Amazon EKS

This project defines a *prototypical* solution for  distributed training and inference on [Amazon Elastic Kubernetes Service (EKS)](https://docs.aws.amazon.com/whitepapers/latest/overview-deployment-options/amazon-elastic-kubernetes-service.html). The primary audience for this project is machine learning (ML) researchers, developers, applied engineers, and platform engineers who train, pre-train, fine-tune, test, or serve Generative AI, or general purpose deep neural networks (DNNs) based models.

The solution offers a framework and  accelerator agnostic approach to distributed training and inference. For training, it works with popular AI machine learning libraries, for example, [Nemo](https://github.com/NVIDIA/NeMo), [Hugging Face Accelerate](https://github.com/huggingface/accelerate), [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), [DeepSpeed](https://github.com/microsoft/DeepSpeed]), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed), [Ray Train](https://docs.ray.io/en/latest/train/train.html), [Neuronx Distributed](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html), among others. For inference, it supports [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) with [vLLM](https://docs.vllm.ai/en/latest/), [Triton Inference Server](https://github.com/triton-inference-server)  with Python, [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [vLLM](https://docs.vllm.ai/en/latest/) backends, and [Deep Java Library (DJL) Large Model Inference (LMI)](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html) with all [supported backends](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html).

#### Legacy Note: 
This project started as a companion to the [Mask R-CNN distributed training blog](https://aws.amazon.com/blogs/opensource/distributed-tensorflow-training-using-kubeflow-on-amazon-eks/), and that part of the project is documented in [this README](./tutorials/maskrcnn-blog/README.md). 

## Conceptual Overview

This solution uses [Helm charts](https://helm.sh/docs/intro/using_helm/) to execute inference and training pipelines. See [tutorials](#tutorials) for a quick start.

Helm charts are commonly used to [deploy applications](https://docs.aws.amazon.com/eks/latest/userguide/helm.html) within an Amazon EKS cluster. While typically, the applications deployed using Helm charts are services that run until stopped, Helm charts can be also used to deploy application that are not long running services. We leverage Helm Charts to execute discrete steps within any training or inference pipeline.

Any training or inference pipeline in this solution can be conceptualized as a series of Helm chart installations, managed within a single Helm *Release*. Each step in the workflow is executed via a Helm chart installation using a specific [YAML recipe](#yaml-recipes), whereby, the YAML recipe acts as a [Helm Values File](https://helm.sh/docs/chart_template_guide/values_files/). Once the Helm chart step completes successfully, the Helm chart is uninstalled, and the next Helm chart in the pipeline is deployed within the same Helm *Release*. Using a single Helm *Release* for a given ML pipeline ensures that the discrete steps in the pipeline are executed atomically, and the dependency among the steps is maintained.

The included [tutorials](#tutorials) provide examples of training and inference pipelines that use pre-defined Helm charts, along with YAML recipe files that model each pipeline step. Typically, in order to define a new pipeline in this solution, you do not need to write new Helm Charts. The solution comes with a library of pre-defined [machine learning Helm charts](./charts/machine-learning/). However, you are required to write a YAML recipe file for each step in your training or inference pipeline. 

The [tutorials](#tutorials) walk you through each pipeline, step by step, where you manually execute each pipeline step by installing, and uninstalling, a pre-defined Helm chart with its associated [YAML recipe](#yaml-recipes). 

For complete end-to-end automation, we also provide an [example](./examples/training/accelerate/bert-glue-mrpc/pipeline.ipynb) where you can execute all the pipeline steps automatically using [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/concepts/). This option requires you to enable [Kubeflow platform](#enabling-modular-components).

If you are a platform engineer, you may be interested in a [system architecture](#system-architecture) overview of this solution.

## Tutorials

Use the directory below to navigate tutorials.

| Category      | Frameworks/Libraries |
| ----------- | ----------- |
| [Inference](./examples/inference/README.md)      | [DJL Serving](https://github.com/deepjavalibrary/djl-serving) , [RayServe](https://docs.ray.io/en/latest/serve/index.html), [Triton Inference Server](https://github.com/triton-inference-server/server)      |
| [Training](./examples/training/README.md)      | [Hugging Face Accelerate](https://github.com/huggingface/accelerate), [Lightning](https://lightning.ai/docs/pytorch/stable/), [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed), [Nemo](https://github.com/NVIDIA/NeMo), [Neuronx Distributed](https://github.com/aws-neuron/neuronx-distributed), [Neuronx Distributed Training](https://github.com/aws-neuron/neuronx-distributed-training), [RayTrain](https://docs.ray.io/en/latest/train/train.html) |
| [Legacy](./examples/legacy/README.md)      | [Mask R-CNN (TensorFlow)](https://github.com/aws-samples/mask-rcnn-tensorflow), [Neuronx Nemo Megatron](https://github.com/aws-neuron/neuronx-nemo-megatron)   |

## Prerequisites

1. [Create and activate an AWS Account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/)
2. Select your AWS Region. For the tutorial below, we assume the region to be ```us-west-2```
3. [Manage your Amazon EC2 service limits](https://aws.amazon.com/premiumsupport/knowledge-center/manage-service-limits/) in your selected AWS Region. Increase service limits to at least 8 each for [p4d.24xlarge](https://aws.amazon.com/ec2/instance-types/p4/), [g5.xlarge, g5.12xlarge, g5.48xlarge](https://aws.amazon.com/ec2/instance-types/g5/), [`inf2.48xlarge`](https://aws.amazon.com/machine-learning/inferentia/) and [`trn1.32xlarge`](https://aws.amazon.com/machine-learning/trainium/). 
4. If you use other Amazon EC2 GPU or AWS Trainium/Inferentia instance types, ensure your EC2 service limits are increased appropriately.

## Getting started

To get started, we need to execute following steps:

  1. Setup the build machine
  2. Use [Terraform](https://learn.hashicorp.com/terraform) to create the required infrastructure
  3. Create `home` folder on [Amazon EFS](https://aws.amazon.com/efs/) and [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) shared file-systems
  
### Setup the build machine

For the *build machine*, we need a machine capable of building Docker images for the `linux/amd64` operating system architecture. The build machine will minimally need [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html) and [Docker](https://www.docker.com/) installed. The AWS CLI must be configured for [Administrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html). It is highly recommended that you  [launch an EC2 instance for the build machine](#optional-launch-ec2-instance-for-the-build-machine).

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

### Use Terraform to create infrastructure

 We use Terraform to create the EKS cluster, and deploy Kubeflow platform.

#### Enable S3 backend for Terraform

Replace `S3_BUCKET`and `S3_PREFIX` with your S3 bucket name, and s3 prefix (no leading or trailing `/`), and execute the commands below

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./eks-cluster/utils/s3-backend.sh S3_BUCKET S3_PREFIX

#### Initialize Terraform
   
    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup
    terraform init

#### Apply Terraform

Not all the AWS Availability Zones in an AWS Region have all the EC2 instance types. Specify at least three AWS Availability Zones from your AWS Region in `azs` below, ensuring that you  have access to your desired EC2 instance types. Replace `S3_BUCKET` with your S3 bucket name and execute:

    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET/ml-platform/"

If you need to use [AWS GPU accelerated instances](https://aws.amazon.com/ec2/instance-types/) with [AWS Elastic Fabric Adapter (EFA)](https://aws.amazon.com/hpc/efa/), you must specify an **AWS Availability Zone** for running these instances using `cuda_efa_az` variable, as shown in the example below:

    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2d","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET/ml-platform/" -var="cuda_efa_az=us-west-2c"

If you need to use [AWS Trainium instances](https://aws.amazon.com/machine-learning/trainium/), you must specify an **AWS Availability Zone** for running Trainium instances using `neuron_az` variable, as shown in the example below:

    terraform apply -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2d","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET/ml-platform/" -var="neuron_az=us-west-2d"

**Note:** Ensure that the AWS Availability Zone you specify for `neuron_az` or `cuda_efa_az` variable above supports requested instance types, and this zone is included in the `azs` variable.

#### Enabling modular components

You may toggle the enablement of following components using terraform variables:

| Component  | Terraform Variable | Default Value |
| ----------- | ----------- | ----------- |
| [Airflow](https://airflow.apache.org/) | airflow_enabled | false |
| [Kubeflow](https://www.kubeflow.org/) | kubeflow_platform_enabled | false |
| [KServe](https://kserve.github.io/website/latest/) | kserve_enabled | false |
| [Kueue](https://kueue.sigs.k8s.io/) | kueue_enabled | false |
| [Nvidia DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter) | dcgm_exporter_enabled | false |
| [SageMaker controller](https://github.com/aws-controllers-k8s/sagemaker-controller) | ack_sagemaker_enabled | false |
| [Slurm](https://github.com/stackhpc/slurm-k8s-cluster/tree/main) | slurm_enabled | false |


#### Retrieve static user password

This step is only needed if you plan to use the Kubeflow Central Dashboard, which is not required for running any of the examples and tutorials in this project. The static user's password is marked `sensitive` in the Terraform output. To show your static password, execute:

    terraform output static_password 

### Create `home` folder on shared file-systems

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

If your web browser client machine is not the same as your build machine, before you can access Kubeflow Central Dashboard in a web browser, you must execute following steps on the your client machine:

1. [install `kubectl` client](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html)
2. [Enable IAM access](https://docs.aws.amazon.com/eks/latest/userguide/grant-k8s-access.html) to your EKS cluster. Before you execute this step, it is **highly recommended** that you backup your current configuration by executing following command on your **build machine**:

    `kubectl get configmap aws-auth -n kube-system -o yaml > ~/aws-auth.yaml`

After you have enabled IAM access to your EKS cluster, open a terminal on your client machine and start `kubectl` port-forwarding by using the local and remote ports as described below. Because we need to forward HTTPs port 
(443), we need root access to execute steps below:

For Mac:

    sudo kubectl port-forward svc/istio-ingressgateway -n ingress 443:443

For Linux:

Ensure `kubectl` is configured for `root` user, and execute below:

    sudo su -
    kubectl port-forward svc/istio-ingressgateway -n ingress 443:443


**Note**: Leave the terminal open.

Next, modify your `/etc/hosts` file to add following entry:

    127.0.0.1 	istio-ingressgateway.ingress.svc.cluster.local

Open your web browser to the [KubeFlow Central Dashboard](https://istio-ingressgateway.ingress.svc.cluster.local/) URL to access the dashboard. For login, use the static username `user@example.com`, and [retrieve the static password from terraform](#retrieve-static-user-password).

### Use Terraform to destroy infrastructure

If you want to preserve any content from your EFS file-system, you must upload it to your [Amazon S3](https://aws.amazon.com/s3/) bucket, manually. The content stored on the  FSx for Lustre file-system is automatically exported to your [Amazon S3](https://aws.amazon.com/s3/) bucket under the `ml-platform` top-level folder.

Please verify your content in [Amazon S3](https://aws.amazon.com/s3/) bucket before destroying the infrastructure. You can recreate your infrastructure using the same S3 bucket. 

Use following command to check and uninstall all Helm releases:

    for x in $(helm list -q -n kubeflow-user-example-com); do echo $x; helm uninstall $x -n kubeflow-user-example-com; done

Wait 5 minutes for Helm uninstall to shut down pods. Use following commands to check and delete all remaining pods in `kubeflow-user-example-com` namespace:

    kubectl get pods -n kubeflow-user-example-com
    kubectl delete --all pods -n kubeflow-user-example-com

Run following commands to delete `attach-pvc` pod:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl delete -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    
Wait 15 minutes to allow accelerator infrastructure to automatically scale down to zero.

Finally, to destroy all the infrastructure created in this tutorial, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup

    terraform destroy -var="profile=default" -var="region=us-west-2" -var="cluster_name=my-eks-cluster" -var='azs=["us-west-2d","us-west-2b","us-west-2c"]' -var="import_path=s3://S3_BUCKET/ml-platform/"


#### (Optional) Launch EC2 instance for the build machine 
To launch an EC2 instance for the *build machine*, you will need [Administrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html) access to [AWS Management Console](https://aws.amazon.com/console/). In the console, execute following steps:

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

## Reference

### YAML Recipes

The YAML recipe file is a [Helm values](https://helm.sh/docs/chart_template_guide/values_files/) file that defines the runtime environment for a data pre-processing, or training job. The key fields in the Helm values file that are common to all charts are described below:

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