# Fine-tuning Llama 2 using PyTorch FSDP with Accelerate library

This example illustrates how to use [pytorch-elastic](../../../../../charts/machine-learning/training/pytorchjob-elastic/) Helm chart for [fine-tuning Llama 2 using PyTorch FSDP](https://huggingface.co/blog/ram-efficient-pytorch-fsdp) with [Hugging Face Accelerate](https://github.com/huggingface/accelerate) library. We do the fine-tuning on [smangrul/code-chat-assistant-v1](https://huggingface.co/datasets/smangrul/code-chat-assistant-v1) dataset.

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../../../../README.md#prerequisites) and [Getting started](../../../../../../README.md#getting-started). In particular, if you plan to fine-tune the 70B model, you must [Apply Terraform](../../../../README.md#apply-terraform) by specifying the variable `cuda_efa_az` so you can automatically launch [`p4d.24xlarge`](https://aws.amazon.com/ec2/instance-types/p4/) instances with [AWS Elastic Fabric Adapter (EFA)](https://aws.amazon.com/hpc/efa/) elastic network interface enabled.

See [What is in the YAML file](../../../../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. 

## Hugging Face Llama 2 pre-trained model weights

Since we are fine-tuning Llama 2 models, we need to download Hugging Face Llama 2 pre-trained model weights.

### Llama 2 7B model weights

To download Hugging Face Llama2 7B Chat model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug accel-llama2-7b     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Llama-2-7b-chat-hf"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall accel-llama2-7b  -n kubeflow-user-example-com

### Llama 2 13B model weights

To download Hugging Face Llama2 13B Chat model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug accel-llama2-13b     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Llama-2-13b-chat-hf"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall accel-llama2-13b  -n kubeflow-user-example-com

### Llama 2 70B model weights

To download Hugging Face Llama2 70B Chat model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug accel-llama2-70b     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Llama-2-70b-chat-hf"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall accel-llama2-70b  -n kubeflow-user-example-com

## Implicitly defined environment variables

Following variables are implicitly defined by the [pytorch-elastic](../../../../../charts/machine-learning/training/pytorchjob-elastic/) Helm chart for use with [Torchrun elastic launch](https://pytorch.org/docs/stable/elastic/run.html):

1. `PET_NNODES` : Maps to `nnodes`
2. `PET_NPROC_PER_NODE` : Maps to `nproc_per_node` 
3. `PET_RDZV_ID` : Maps to `rdzv_id` 
4. `PET_RDZV_ENDPOINT`: Maps to `rdzv_endpoint` 

## Fine-tuning Llama 2 7B

The Helm values are defined in [7b.yaml](./7b.yaml). 

To launch fine-tuning job,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug accel-llama2-7b \
        ./charts/machine-learning/training/pytorchjob-elastic/ \
        -f examples/training/accelerate/llama2-ft-fsdp/7b.yaml -n kubeflow-user-example-com

You can tail the logs using following command:

    kubectl logs -f pytorchjob-accel-llama2-7b-worker-0 -n kubeflow-user-example-com


To uninstall the Helm chart for the fine-tuning job, execute:

    helm uninstall accel-llama2-7b -n kubeflow-user-example-com

## Fine-tuning Llama 2 13B

The Helm values are defined in [13b.yaml](./13b.yaml). 

To launch fine-tuning job,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug accel-llama2-13b \
        ./charts/machine-learning/training/pytorchjob-elastic/ \
        -f examples/training/accelerate/llama2-ft-fsdp/13b.yaml -n kubeflow-user-example-com

You can tail the logs using following command:

    kubectl logs -f pytorchjob-accel-llama2-13b-worker-0 -n kubeflow-user-example-com


To uninstall the Helm chart for fine-tuning job, execute:

    helm uninstall accel-llama2-13b -n kubeflow-user-example-com

## Fine-tuning Llama 2 70B

The Helm values are defined in [70b.yaml](./70b.yaml). 

To launch fine-tuning job,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug accel-llama2-70b \
        ./charts/machine-learning/training/pytorchjob-elastic/ \
        -f examples/training/accelerate/llama2-ft-fsdp/70b.yaml -n kubeflow-user-example-com

You can tail the logs using following command:

    kubectl logs -f pytorchjob-accel-llama2-70b-worker-0 -n kubeflow-user-example-com


To uninstall the Helm chart for fine-tuning job, execute:

    helm uninstall accel-llama2-70b -n kubeflow-user-example-com

## Output

To access the output stored on EFS and FSx for Lustre file-systems, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash

This will put you in a pod attached to the  EFS and FSx for Lustre file-systems, mounted at `/efs`, and `/fsx`, respectively. Type `exit` to exit the pod.

### Logs

Pre-training `logs` are available in `/efs/home/accel-llama2-*/logs` folder. 

### Checkpoints

Pre-training `checkpoints`, if any, are available in `/fsx/home/accel-llama2-*/checkpoints` folder. 

### S3 Backup

Any content stored under `/fsx` is automatically backed up to your configured S3 bucket.