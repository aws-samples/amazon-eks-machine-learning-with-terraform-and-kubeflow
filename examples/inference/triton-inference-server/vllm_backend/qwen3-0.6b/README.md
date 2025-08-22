# Serve Qwen/Qwen3-0.6B using Triton Inference Server

This example shows how to serve [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model using [Triton Inference Server](https://github.com/triton-inference-server) with [vLLM backend](https://github.com/triton-inference-server/vllm_backend/tree/main). 

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../../../README.md#prerequisites) and [Launch desktop](../../../../../README.md##launch-build-machine-desktop).

See [What is in the YAML file](../../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Download Hugging Face Qwen3-0.6B pre-trained model weights

To download Hugging Face Qwen3-0.6B  pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-qwen3-0-6b-vllm    \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"Qwen/Qwen3-0.6B"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-qwen3-0-6b-vllm -n kubeflow-user-example-com

## Launch Triton Server

To launch Triton server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-qwen3-0-6b-vllm \
        charts/machine-learning/serving/triton-inference-server \
        -f examples/inference/triton-inference-server/vllm_backend/qwen3-0.6b/triton_server.yaml -n kubeflow-user-example-com


## Stop Service

To stop the service:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall triton-server-qwen3-0-6b-vllm -n kubeflow-user-example-com

## Logs

Triton server logs are available in `/efs/home/triton-server-qwen3-0-6b-vllm/logs` folder.
