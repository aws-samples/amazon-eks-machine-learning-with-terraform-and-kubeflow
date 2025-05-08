# Serve Llama3 8B Instruct using Triton Inference Server with vLLM on AWS Neuron

This example shows how to serve [llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model using [Triton Inference Server](https://github.com/triton-inference-server) with [vLLM backend](https://github.com/triton-inference-server/vllm_backend/tree/main) on [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html).

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../../../README.md#prerequisites) and [Getting started](../../../../../README.md#getting-started). 

See [What is in the YAML file](../../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.
    
## Build and Push Docker Container

This example uses a custom Docker container for Triton Inference Server with vLLM for PyTorch Neuronx. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/tritonserver-neuronx-vllm/build_tools/build_and_push.sh aws-region

## Hugging Face Llama3 8B Instruct pre-trained model weights

To download Hugging Face Llama3 8B Instruct pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-llama3-8b-instruct-vllm-nx     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Meta-Llama-3-8B-Instruct"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-llama3-8b-instruct-vllm-nx -n kubeflow-user-example-com


## Launch Triton Server

To launch Triton server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-llama3-8b-instruct-vllm-nx \
        charts/machine-learning/serving/triton-inference-server \
        -f examples/inference/triton-inference-server/vllm_backend/llama3-8b-instruct-neuron/triton_server.yaml -n kubeflow-user-example-com

## Stop server

Uninstall the Helm chart to stop server:

    helm uninstall triton-server-llama3-8b-instruct-vllm-nx -n kubeflow-user-example-com