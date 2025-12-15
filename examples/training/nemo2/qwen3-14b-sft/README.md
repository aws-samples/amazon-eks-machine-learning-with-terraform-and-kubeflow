# Fine-tuning Qwen3-14B using Nemo 2.0 Framework

This example illustrates fine-tuning Qwen3-14B using [Nemo 2.0 framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html).

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files.

## Hugging Face Qwen3-14B pre-trained model weights

To download Hugging Face Qwen3-14B model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nemo2-qwen3-14b-sft     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"Qwen/Qwen3-14B"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall nemo2-qwen3-14b-sft -n kubeflow-user-example-com

## Fine-tuning Qwen3-14B

The Helm values are defined in [fine-tune.yaml](./fine-tune.yaml).

To launch fine-tuning job, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nemo2-qwen3-14b-sft \
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/training/nemo2/qwen3-14b-sft/fine-tune.yaml -n kubeflow-user-example-com

To uninstall the Helm chart for the fine-tuning job, execute:

    helm uninstall nemo2-qwen3-14b-sft -n kubeflow-user-example-com

## Evaluating Qwen3-14B

The Helm values are defined in [evaluate.yaml](./evaluate.yaml).

To launch evaluation, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nemo2-qwen3-14b-sft \
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/training/nemo2/qwen3-14b-sft/evaluate.yaml -n kubeflow-user-example-com

To uninstall the Helm chart for the fine-tuning job, execute:

    helm uninstall nemo2-qwen3-14b-sft -n kubeflow-user-example-com

## Converting Qwen3-14B

The Helm values are defined in [convert-hf.yaml](./convert-hf.yaml).

To launch evaluation, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nemo2-qwen3-14b-sft \
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/training/nemo2/qwen3-14b-sft/convert-hf.yaml -n kubeflow-user-example-com

To uninstall the Helm chart for the fine-tuning job, execute:

    helm uninstall nemo2-qwen3-14b-sft -n kubeflow-user-example-com

## Output

To access the output stored on EFS and FSx for Lustre file-systems, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash

This will put you in a pod attached to the EFS and FSx for Lustre file-systems, mounted at `/efs`, and `/fsx`, respectively. Type `exit` to exit the pod.

Training outputs are available in `/efs/home/nemo2-qwen3-14b-sft/outputs` folder.

