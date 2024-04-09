# Pre-train Llama2 70B on Redpajama dataset using Neuronx-Nemo-Megatron

This example shows how to use [pytorch-distributed](../../../charts/machine-learning/training/pytorchjob-elastic/Chart.yaml) Helm chart to pre-train Llama2-70B model on [Redpajama dataset](https://github.com/togethercomputer/RedPajama-Data) with [Neuronx-Nemo-Megatron](https://github.com/aws-neuron/neuronx-nemo-megatron). The example also shows use of [data-process](../../../charts/machine-learning/data-prep/data-process/Chart.yaml) Helm chart to tokenize the  dataset for use with LLAMA2-70B model.

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). In particular, you must [Apply Terraform](../../../README.md#apply-terraform) by specifying the variable `neuron_az` so you can automatically launch [`trn1.32xlarge`](https://aws.amazon.com/ec2/instance-types/trn1/) instances with [AWS Elastic Fabric Adapter (EFA)](https://aws.amazon.com/hpc/efa/).

See [What is in the YAML file](../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.


## Implicitly defined environment variables

Following variables are implicitly defined by the [pytorch-distributed](../../../charts/machine-learning/training/pytorchjob-distributed/Chart.yaml) Helm chart for use with [Torch distributed run](https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py):

1. `PET_NNODES` : Maps to `nnodes`
2. `PET_NPROC_PER_NODE` : Maps to `nproc_per_node` 
3. `PET_NODE_RANK` : Maps to `node_rank` 
4. `PET_MASTER_ADDR`: Maps to `master_addr` 
5. `PET_MASTER_PORT`: Maps to `master_port`

## Hugging Face Llama2 70B pre-trained model weights

Below, we show sample commands for downloading and uploading the pre-trained model weights for Hugging Face Llama2 70B model using [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (replace HF_TOKEN with your Hugging Face CLI token, and S3_BUCKET with your configured S3 bucket), below:

    huggingface-cli download --repo-type model --revision 97a63a0ef386379ab4175785da8352fbe338c6c5 --local-dir ./meta-llama/Llama-2-70b-hf --token HF_TOKEN  meta-llama/Llama-2-70b-hf

    aws s3 cp --recursive meta-llama/Llama-2-70b-hf s3://S3_BUCKET/ml-platform/pretrained-models/meta-llama/Llama-2-70b-hf

## Download Redpajama dataset 

We need to download the Redpajama dataset to the FSx for Lustre file-system, using commands shown below:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nx-nemo-megatron-llama2-70b \
        charts/machine-learning/data-prep/redpajama-data  -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall nx-nemo-megatron-llama2-70b -n kubeflow-user-example-com

## Pre-process Redpajama dataset

We define the runtime for pre-processing the dataset in [preprocess.yaml](./preprocess.yaml) values file. 

To launch the data processing job, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nx-nemo-megatron-llama2-70b \
        charts/machine-learning/data-prep/data-process \
        -f examples/neuronx-nemo-megatron/llama2_70b/preprocess.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-nx-nemo-megatron-llama2-70b -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall nx-nemo-megatron-llama2-70b -n kubeflow-user-example-com

## Compile

We define the runtime for the compile job in [compile.yaml](./compile.yaml) values file. 

To launch compile job:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nx-nemo-megatron-llama2-70b \
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/neuronx-nemo-megatron/llama2_70b/compile.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f pytorchjob-nx-nemo-megatron-llama2-70b-master-0  -n kubeflow-user-example-com

To uninstall the Helm chart:

    helm uninstall nx-nemo-megatron-llama2-70b -n kubeflow-user-example-com

## Pre-train

We define the runtime for pre-training job in [pretrain.yaml](./pretrain.yaml) values file. 

To launch the pre-training job:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nx-nemo-megatron-llama2-70b \
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/neuronx-nemo-megatron/llama2_70b/pretrain.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f pytorchjob-nx-nemo-megatron-llama2-70b-master-0  -n kubeflow-user-example-com

To uninstall the Helm chart:

   helm uninstall nx-nemo-megatron-llama2-70b -n kubeflow-user-example-com

## Output

To access the output stored on EFS and FSx for Lustre file-systems, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash


This will put you in a pod attached to the  EFS and FSx for Lustre file-systems, mounted at `/efs`, and `/fsx`, respectively. Type `exit` to exit the pod.

### Data

Pre-processed data is available in `/fsx/home/nx-nemo-megatron-llama2-70b/datasets/` folder.

### Logs

Pre-training `logs` are available in `/efs/home/nx-nemo-megatron-llama2-70b/logs` folder. 

### S3 Backup

Any content stored under `/fsx` is automatically backed up to your configured S3 bucket.