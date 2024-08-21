# Serve Meta Llama 3 8B Instruct using Triton Inference Server

This example shows how to serve [Meta Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model using [Triton Inference Server](https://github.com/triton-inference-server) with [vLLM backend](https://github.com/triton-inference-server/vllm_backend/tree/main). 

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). 

See [What is in the YAML file](../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.


## Hugging Face Meta Llama 3 8B Instruct  pre-trained model weights

To download Hugging Face Meta Llama 3 8B Instruct  pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-llama3-8b-instruct-vllm    \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Meta-Llama-3-8B-Instruct"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-llama3-8b-instruct-vllm -n kubeflow-user-example-com

## Launch Triton Server

To launch Triton server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-llama3-8b-instruct-vllm \
        charts/machine-learning/serving/triton-inference-server \
        -f examples/triton-server/llama3-8b-instruct-vllm/triton_server.yaml -n kubeflow-user-example-com


## Stop Service

To stop the service:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall triton-server-llama3-8b-instruct-vllm -n kubeflow-user-example-com

## Logs

Triton server logs are available in `/efs/home/triton-server-llama3-8b-instruct-vllm/logs` folder. 