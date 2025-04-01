# Serve DeepSeek AI R1 Distill Llama 8B using Triton Inference Server with vLLM

This example shows how to serve [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) model using [Triton Inference Server](https://github.com/triton-inference-server) with [vLLM backend](https://github.com/triton-inference-server/vllm_backend/tree/main). 

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). 

See [What is in the YAML file](../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Hugging Face DeepSeek AI R1 Distill Llama 8B pre-trained model weights

To download Hugging Face Mistral 7B Instruct v0.1 pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-deepseek-r1-distill-llama-8b-vllm     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-deepseek-r1-distill-llama-8b-vllm -n kubeflow-user-example-com


## Launch Triton Server

To launch Triton server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-deepseek-r1-distill-llama-8b-vllm \
        charts/machine-learning/serving/triton-inference-server \
        -f examples/inference/triton-inference-server/vllm_backend/deepseek-r1-distill-llama-8b/triton_server.yaml -n kubeflow-user-example-com


## Stop Service

To stop the service:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall triton-server-deepseek-r1-distill-llama-8b-vllm -n kubeflow-user-example-com

## Logs

Triton server logs are available in `/efs/home/triton-server-deepseek-r1-distill-llama-8b-vllm/logs` folder. 