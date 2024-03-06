# Pre-train BERT on Glue MRPC dataset using Accelerate library

This example illustrates how to use [pytorch-elastic](../../../charts/machine-learning/training/pytorchjob-elastic/) Helm chart to pre-train BERT on Glue MRPC dataset with [Accelerate](https://github.com/huggingface/accelerate) library.

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). See [Key concepts](../../../README.md#key-concepts) to understand the structure of the Helm values files.

## Implicitly defined environment variables

Following variables are implicitly defined by the [pytorch-elastic](../../../charts/machine-learning/training/pytorchjob-elastic/) Helm chart for use with [Torchrun elastic launch](https://pytorch.org/docs/stable/elastic/run.html):

1. `PET_NNODES` : Maps to `nnodes`
2. `PET_NPROC_PER_NODE` : Maps to `nproc_per_node` 
3. `PET_RDZV_ID` : Maps to `rdzv_id` 
4. `PET_RDZV_ENDPOINT`: Maps to `rdzv_endpoint` 

## Launch pre-training

The pre-training Helm values are defined in [pretrain.yaml](./pretrain.yaml). 

To launch pre-training,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug accel-bert \
        ./charts/machine-learning/training/pytorchjob-elastic/ \
        -f ./examples/accelerate/bert-glue-mrpc/pretrain.yaml -n kubeflow-user-example-com

You can tail the logs using following command:

    kubectl logs -f pytorchjob-accel-bert-worker-0 -n kubeflow-user-example-com


To uninstall the Helm chart for pre-training job, execute:

    helm uninstall accel-bert  -n kubeflow-user-example-com

## Pre-training output

Pre-training output is available on the EFS file-system under `/efs/home/accel-bert` folder. To access the EFS file-system, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash
    cd efs

This will put you in a pod attached to the EFS file-system. Type `exit` to exit the pod attached to the EFS file-system.