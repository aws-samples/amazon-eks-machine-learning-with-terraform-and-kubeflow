# Serve Mistral 7B Instruct v0.1 using Triton Inference Server with vLLM

This example shows how to serve [mistral-7b-instruct-v01](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) model using [Triton Inference Server](https://github.com/triton-inference-server) with [vLLM backend](https://github.com/triton-inference-server/vllm_backend/tree/main). 

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). 

See [What is in the YAML file](../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.


## Hugging Face Mistral 7B Instruct v0.1 pre-trained model weights

To download Hugging Face Mistral 7B Instruct v0.1 pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-mistral-7b-instruct-v01-vllm     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"mistralai/Mistral-7B-Instruct-v0.1"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-mistral-7b-instruct-v01-vllm -n kubeflow-user-example-com


## Launch Triton Server

To launch Triton server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-mistral-7b-instruct-v01-vllm \
        charts/machine-learning/serving/triton-inference-server \
        -f examples/triton-server/mistral-7b-instruct-v01-vllm/triton_server.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-triton-server-mistral-7b-instruct-v01-vllm -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-mistral-7b-instruct-v01-vllm -n kubeflow-user-example-com

## Output

To access the output stored on EFS and FSx for Lustre file-systems, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash


This will put you in a pod attached to the  EFS and FSx for Lustre file-systems, mounted at `/efs`, and `/fsx`, respectively. Type `exit` to exit the pod.

### Data

Pre-processed data is available in `/fsx/home/triton-server-mistral-7b-instruct-v01-vllm/data/` folder.

### Logs

Pre-training `logs` are available in `/efs/home/triton-server-mistral-7b-instruct-v01-vllm/logs` folder. 

### S3 Backup

Any content stored under `/fsx` is automatically backed up to your configured S3 bucket.