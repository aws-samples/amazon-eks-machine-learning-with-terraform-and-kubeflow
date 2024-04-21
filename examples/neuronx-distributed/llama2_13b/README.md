# Pre-train Llama2-13B on Wikicorpus dataset using Neuronx Distributed library

This example shows how to use [pytorch-distributed](../../../charts/machine-learning/training/pytorchjob-elastic/Chart.yaml) Helm chart to pre-train Llama2-13B model on Wikicorpus dataset with [Neuronx-Distributed](https://github.com/aws-neuron/neuronx-distributed/tree/main) library, using distributed data-parallel, tensor-parallel, and [ZeRO-1](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html). 

The example also shows use of [data-process](../../../charts/machine-learning/data-prep/data-process/Chart.yaml) Helm chart to pre-process the [Hugging Face Wikicorpus](https://huggingface.co/datasets/wikicorpus) dataset for use with LLAMA2-13B model.

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). In particular, you must [Apply Terraform](../../../README.md#apply-terraform) by specifying the variable `neuron_az` so you can automatically launch `trn1.32xlarge` instances.

See [What is in the YAML file](../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Download Meta LLama2 Tokenizer Model file

[Download the Meta Llama2](https://llama.meta.com/llama2) model files. Upload the `tokenizer.model` file to your configured S3 bucket under the S3 path `ml-platform/llama2/tokenizer.model`. Once you upload it to the S3 bucket, it will be automatically imported to the FSx for Lustre file-system at the path `/fsx/llama2/tokenizer.model`.

## Pre-process Wikicorpus dataset

We define the runtime for pre-processing the dataset in [wikicorpus.yaml](./wikicorpus.yaml) values file. 

To launch the data processing job, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nxd-llama2-13b \
        charts/machine-learning/data-prep/data-process \
        -f examples/neuronx-distributed/llama2_13b/wikicorpus.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-nxd-llama2-13b -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall nxd-llama2-13b -n kubeflow-user-example-com

Processed data is available on EFS file-system at `/fsx/home/nxd-llama2-13b/data/wikicorpus`.

## Compile

We define the runtime for the compile job in [compile.yaml](./compile.yaml) values file. 

To launch compile job:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nxd-llama2-13b \
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/neuronx-distributed/llama2_13b/compile.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f pytorchjob-nxd-llama2-13b-master-0  -n kubeflow-user-example-com

To uninstall the Helm chart:

    helm uninstall nxd-llama2-13b -n kubeflow-user-example-com

## Pre-train

We define the runtime for pre-training job in [pretrain.yaml](./pretrain.yaml) values file. 

To launch the pre-training job:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug nxd-llama2-13b \
        charts/machine-learning/training/pytorchjob-distributed \
        -f examples/neuronx-distributed/llama2_13b/pretrain.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f pytorchjob-nxd-llama2-13b-master-0  -n kubeflow-user-example-com

To uninstall the Helm chart:

    helm uninstall nxd-llama2-13b -n kubeflow-user-example-com

## Output

To access the output stored on EFS and FSx for Lustre file-systems, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash


This will put you in a pod attached to the  EFS and FSx for Lustre file-systems, mounted at `/efs`, and `/fsx`, respectively. Type `exit` to exit the pod.

### Data

Pre-processed data is available in `/fsx/home/nxd-llama2-13b/examples_datasets/` folder.

### Logs

Pre-training `logs` are available in `/efs/home/nxd-llama2-13b/logs` folder. 

### Checkpoints

Pre-training `checkpoints`, if any, are available in `/fsx/home/nxd-llama2-13b/checkpoints` folder. 

### S3 Backup

Any content stored under `/fsx` is automatically backed up to your configured S3 bucket.