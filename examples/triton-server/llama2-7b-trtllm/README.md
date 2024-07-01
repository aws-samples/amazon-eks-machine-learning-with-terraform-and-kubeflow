# Serve Llama2 7B using triton Inference Server

This example shows how to serve [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) model using triton Inference server.  

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). 

See [What is in the YAML file](../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.


## Hugging Face Llama2 7B pre-trained model weights

To download Hugging Face Llama2 7B pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-llama2-7b-trtllm     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Llama-2-7b-hf"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-llama2-7b-trtllm -n kubeflow-user-example-com

## Convert HuggingFace Checkpoint to TensorRT-LLM Checkpoint

To convert checkpoint:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-llama2-7b-trtllm \
        charts/machine-learning/data-prep/data-process \
        -f examples/triton-server/llama2-7b-trtllm/hf_to_trtllm.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-triton-server-llama2-7b-trtllm -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-llama2-7b-trtllm -n kubeflow-user-example-com

## Built TensorRT-LLM Engine

To build TensorRT-LLM engine:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-llama2-7b-trtllm \
        charts/machine-learning/data-prep/data-process \
        -f examples/triton-server/llama2-7b-trtllm/trtllm_engine.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-triton-server-llama2-7b-trtllm -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-llama2-7b-trtllm -n kubeflow-user-example-com

## Build Triton Model

To build Triton model:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-llama2-7b-trtllm \
        charts/machine-learning/data-prep/data-process \
        -f examples/triton-server/llama2-7b-trtllm/triton_model.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-triton-server-llama2-7b-trtllm -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-llama2-7b-trtllm -n kubeflow-user-example-com


## Launch Triton Server

To launch Triton server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-llama2-7b-trtllm \
        charts/machine-learning/serving/triton-inference-server \
        -f examples/triton-server/llama2-7b-trtllm/triton_server.yaml -n kubeflow-user-example-com


Uninstall the Helm chart at completion:

    helm uninstall triton-server-llama2-7b-trtllm -n kubeflow-user-example-com

## Output

To access the output stored on EFS and FSx for Lustre file-systems, execute following commands:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    kubectl apply -f eks-cluster/utils/attach-pvc.yaml  -n kubeflow
    kubectl exec -it -n kubeflow attach-pvc -- /bin/bash


This will put you in a pod attached to the  EFS and FSx for Lustre file-systems, mounted at `/efs`, and `/fsx`, respectively. Type `exit` to exit the pod.

### Data

Pre-processed data is available in `/fsx/home/triton-server-llama2-7b-trtllm/data/` folder.

### Logs

Pre-training `logs` are available in `/efs/home/triton-server-llama2-7b-trtllm/logs` folder. 

### S3 Backup

Any content stored under `/fsx` is automatically backed up to your configured S3 bucket.