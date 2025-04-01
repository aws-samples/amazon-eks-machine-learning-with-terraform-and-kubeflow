# Serve Mistral 7B Instruct v0.1 and Llama3 8B concurrently using Triton Inference Server with TensorRT-LLM

This example shows how to serve [mistral-7b-instruct-v01](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) and [Meta Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) models concurrently using [Triton Inference Server](https://github.com/triton-inference-server) with [TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend/tree/main).  

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). 

See [What is in the YAML file](../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Tritonserver TensorRT-LLM. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/tritonserver-trtllm/build_tools/build_and_push.sh aws-region
      
## Hugging Face Mistral 7B Instruct v0.1 pre-trained model weights

To download Hugging Face Mistral 7B Instruct v0.1 pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-mistral-7b-llama3-8b-trtllm     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"mistralai/Mistral-7B-Instruct-v0.1"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

## Hugging Face Meta Llama 3 8B Instruct  pre-trained model weights

To download Hugging Face Meta Llama 3 8B Instruct  pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-mistral-7b-llama3-8b-trtllm    \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Meta-Llama-3-8B-Instruct"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-mistral-7b-llama3-8b-trtllm -n kubeflow-user-example-com

## Convert Mistral 7B Instruct HuggingFace Checkpoint to TensorRT-LLM Checkpoint

To convert checkpoint:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-mistral-7b-llama3-8b-trtllm  \
        charts/machine-learning/data-prep/data-process \
        -f examples/inference/triton-inference-server/tensorrtllm_backend/mistral-7b-llama3-8b/mistral_7b_hf_to_trtllm.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

## Convert LLama3 8B Instruct HuggingFace Checkpoint to TensorRT-LLM Checkpoint

To convert checkpoint:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-mistral-7b-llama3-8b-trtllm \
        charts/machine-learning/data-prep/data-process \
        -f examples/inference/triton-inference-server/tensorrtllm_backend/mistral-7b-llama3-8b/llama3_8b_hf_to_trtllm.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com


## Build Mistral 7B TensorRT-LLM Engine

To build TensorRT-LLM engine:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-mistral-7b-llama3-8b-trtllm  \
        charts/machine-learning/data-prep/data-process \
        -f examples/inference/triton-inference-server/tensorrtllm_backend/mistral-7b-llama3-8b/trtllm_mistral_7b_engine.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

## Build Llama3 8B TensorRT-LLM Engine

To build TensorRT-LLM engine:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-mistral-7b-llama3-8b-trtllm  \
        charts/machine-learning/data-prep/data-process \
        -f examples/inference/triton-inference-server/tensorrtllm_backend/mistral-7b-llama3-8b/trtllm_llama3_8b_engine.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

## Build Mistral 7B Triton Model

To build Triton model:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-mistral-7b-llama3-8b-trtllm  \
        charts/machine-learning/data-prep/data-process \
        -f examples/inference/triton-inference-server/tensorrtllm_backend/mistral-7b-llama3-8b/triton_mistral_7b_model.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

## Build Llama3 8B Triton Model

To build Triton model:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-mistral-7b-llama3-8b-trtllm  \
        charts/machine-learning/data-prep/data-process \
        -f examples/inference/triton-inference-server/tensorrtllm_backend/mistral-7b-llama3-8b/triton_llama3_8b_model.yaml -n kubeflow-user-example-com

To monitor the logs, execute:

    kubectl logs -f data-process-triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

## Launch Triton Server

To launch Triton server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-mistral-7b-llama3-8b-trtllm  \
        charts/machine-learning/serving/triton-inference-server \
        -f examples/inference/triton-inference-server/tensorrtllm_backend/mistral-7b-llama3-8b/triton_server.yaml -n kubeflow-user-example-com


## Stop Service

To stop the service:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall triton-mistral-7b-llama3-8b-trtllm  -n kubeflow-user-example-com

### Logs

Triton server logs are available in `/efs/home/triton-mistral-7b-llama3-8b-trtllm /logs` folder. 
