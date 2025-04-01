# Fine-tune BERT using PyTorch Lightning and Ray Train libraries

This example illustrates how to use [raytrain](../../../charts/machine-learning/training/raytrain/) Helm chart to pre-train BERT with [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) and [Ray Train](https://docs.ray.io/en/latest/train/train.html) libraries.

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). See [What is in the YAML file](../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Train. Build and push this container using following command (replace `aws-region` with your AWS Region name):

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./containers/ray-pytorch/build_tools/build_and_push.sh aws-region

## Launch fine-training

The fine-training values files is defined in [fine-tune.yaml](fine-tune.yaml). 

To launch fine-tuning,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug ray-bert \
        charts/machine-learning/training/raytrain/ \
        -f examples/training/raytrain/lightning-bert/fine-tune.yaml -n kubeflow-user-example-com

The sequence of Ray training pod startup is as follows:

1. A Ray cluster is created with a `head` pod, named `rayjob-ray-bert-raycluster-*-head-*` 
2. A `training` worker pod named `rayjob-ray-bert-raycluster-*-worker-training-*`  is created
3. A job submitter pod named `rayjob-ray-bert-*` is created, which submits the fine-tuning job to the Ray cluster
4. The fine-tuning job runs in the Ray worker pod, but the logs are available from the Ray job pod

Once Ray job pod is running, you can see the logs using following command (where `*` below denotes a random string):

    kubectl logs -f rayjob-ray-bert-*  -n kubeflow-user-example-com

If you run:

    kubectl get pods -n kubeflow-user-example-com

and see  `rayjob-ray-*` pod with status `Not Ready`, this is normal after your job has completed.

To uninstall the Helm chart for pre-training job, execute:

    helm uninstall ray-bert  -n kubeflow-user-example-com

## Output

To access the output stored on EFS and FSx for Lustre file-systems, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash

This will put you in a pod attached to the  EFS and FSx for Lustre file-systems, mounted at `/efs`, and `/fsx`, respectively. Type `exit` to exit the pod.

### Logs

Pre-training `logs` are available in `/efs/home/ray-bert/logs` folder. 