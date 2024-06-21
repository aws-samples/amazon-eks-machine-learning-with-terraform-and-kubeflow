# Fine-tune Llama2 7B on PubMedQA dataset using Neuronx-Nemo-Megatron

This example shows how to use [pytorch-distributed](../../../charts/machine-learning/training/pytorchjob-elastic/Chart.yaml) Helm chart to fine-tune (PEFT with LoRA) Llama2-7B model on [pubmedqa](https://github.com/pubmedqa/pubmedqa/tree/master) dataset with [Neuronx-Nemo-Megatron](https://github.com/aws-neuron/neuronx-nemo-megatron).

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

## Hugging Face Llama2 7B model

To download Hugging Face Llama2 7B model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nx-nemo-megatron-llama2-7b-peft     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Llama-2-7b-hf"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall nx-nemo-megatron-llama2-7b-peft -n kubeflow-user-example-com

## Convert HuggingFace Checkpoint to Nemo Checkpoint

To convert checkpoint:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nx-nemo-megatron-llama2-7b-peft \
        charts/machine-learning/data-prep/data-process \
        -f examples/neuronx-nemo-megatron/llama2_7b_peft/hf_to_nemo.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-nx-nemo-megatron-llama2-7b-peft -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall nx-nemo-megatron-llama2-7b-peft -n kubeflow-user-example-com

## Preprocess `pubmedqa`` dataset

To preprocess `pubmedqa` dataset:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nx-nemo-megatron-llama2-7b-peft \
        charts/machine-learning/data-prep/data-process \
        -f examples/neuronx-nemo-megatron/llama2_7b_peft/preprocess.yaml -n kubeflow-user-example-com


Uninstall the Helm chart at completion:

    helm uninstall nx-nemo-megatron-llama2-7b-peft -n kubeflow-user-example-com

## Compile

We define the runtime for the compile job in [compile.yaml](./compile.yaml) values file. 

To launch compile job:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nx-nemo-megatron-llama2-7b-peft \
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/neuronx-nemo-megatron/llama2_7b_peft/compile.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f pytorchjob-nx-nemo-megatron-llama2-7b-peft-master-0  -n kubeflow-user-example-com

To uninstall the Helm chart:

    helm uninstall nx-nemo-megatron-llama2-7b-peft -n kubeflow-user-example-com

## PEFT

We define the runtime for PEFT job in [peft.yaml](./peft.yaml) values file. 

To launch the pre-training job:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nx-nemo-megatron-llama2-7b-peft \
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/neuronx-nemo-megatron/llama2_7b_peft/peft.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f pytorchjob-nx-nemo-megatron-llama2-7b-peft-master-0  -n kubeflow-user-example-com

To uninstall the Helm chart:

   helm uninstall nx-nemo-megatron-llama2-7b-peft -n kubeflow-user-example-com

## Output

To access the output stored on EFS and FSx for Lustre file-systems, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash


This will put you in a pod attached to the  EFS and FSx for Lustre file-systems, mounted at `/efs`, and `/fsx`, respectively. Type `exit` to exit the pod.

### Data

Pre-processed data is available in `/fsx/home/nx-nemo-megatron-llama2-7b-peft/data/` folder.

### Logs

Pre-training `logs` are available in `/efs/home/nx-nemo-megatron-llama2-7b-peft/logs` folder. 

### S3 Backup

Any content stored under `/fsx` is automatically backed up to your configured S3 bucket.