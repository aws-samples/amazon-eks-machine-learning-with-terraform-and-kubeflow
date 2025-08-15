# Pre-train Llama3.1-70B on Wikicorpus dataset using Neuronx Distributed library

This example shows how to use [pytorch-distributed](../../../charts/machine-learning/training/pytorchjob-elastic/Chart.yaml) Helm chart to pre-train Llama3.1-70B model on Wikicorpus dataset with [Neuronx-Distributed](https://github.com/aws-neuron/neuronx-distributed/tree/main) library, using distributed data-parallel and model-parallel. 

The example also shows use of [data-process](../../../charts/machine-learning/data-prep/data-process/Chart.yaml) Helm chart to pre-process the [Hugging Face Wikicorpus](https://huggingface.co/datasets/wikicorpus) dataset.

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). In particular, you must [Apply Terraform](../../../../README.md#apply-terraform) by specifying the variable `neuron_az` so you can automatically launch [`trn1.32xlarge`](https://aws.amazon.com/ec2/instance-types/trn1/) instances with [AWS Elastic Fabric Adapter (EFA)](https://aws.amazon.com/hpc/efa/).

See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.


## Implicitly defined environment variables

Following variables are implicitly defined by the [pytorch-distributed](../../../charts/machine-learning/training/pytorchjob-distributed/Chart.yaml) Helm chart for use with [Torch distributed run](https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py):

1. `PET_NNODES` : Maps to `nnodes`
2. `PET_NPROC_PER_NODE` : Maps to `nproc_per_node` 
3. `PET_NODE_RANK` : Maps to `node_rank` 
4. `PET_MASTER_ADDR`: Maps to `master_addr` 
5. `PET_MASTER_PORT`: Maps to `master_port`


## Hugging Face Llama3.1 70B model

To download Hugging Face Llama3.1 70B model configuration (without model weights, since we are pre-training from scratch), replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nxd-llama31-70b    \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Meta-Llama-3-70B"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"},{"name": "HF_TENSORS", "value": "false"}]' \
        -n kubeflow-user-example-com


Uninstall the Helm chart at completion:

    helm uninstall nxd-llama31-70b-n kubeflow-user-example-com

## Pre-process Wikicorpus dataset

We define the runtime for pre-processing the dataset in [wikicorpus.yaml](./wikicorpus.yaml) values file. 

To launch the data processing job, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nxd-llama31-70b\
        charts/machine-learning/data-prep/data-process \
        -f examples/training/neuronx-distributed/llama31_70b/wikicorpus.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-nxd-llama31-70b-n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall nxd-llama31-70b-n kubeflow-user-example-com

## Compile

We define the runtime for the compile job in [compile.yaml](./compile.yaml) values file. 

To launch compile job:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nxd-llama31-70b\
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/training/neuronx-distributed/llama31_70b/compile.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f pytorchjob-nxd-llama31-70b-master-0  -n kubeflow-user-example-com

To uninstall the Helm chart:

    helm uninstall nxd-llama31-70b-n kubeflow-user-example-com

## Pre-train

We define the runtime for pre-training job in [pretrain.yaml](./pretrain.yaml) values file. 

To launch the pre-training job:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nxd-llama31-70b\
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/training/neuronx-distributed/llama31_70b/pretrain.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f pytorchjob-nxd-llama31-70b-master-0  -n kubeflow-user-example-com

To uninstall the Helm chart:

    helm uninstall nxd-llama31-70b-n kubeflow-user-example-com

## Output

To access the output stored on EFS and FSx for Lustre file-systems, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash


This will put you in a pod attached to the  EFS and FSx for Lustre file-systems, mounted at `/efs`, and `/fsx`, respectively. Type `exit` to exit the pod.

### Data

Pre-processed data is available in `/fsx/home/nxd-llama31-70b/examples_datasets/` folder.

### Logs

Pre-training `logs` are available in `/efs/home/nxd-llama31-70b/logs` folder. 

### S3 Backup

Any content stored under `/fsx` is automatically backed up to your configured S3 bucket.