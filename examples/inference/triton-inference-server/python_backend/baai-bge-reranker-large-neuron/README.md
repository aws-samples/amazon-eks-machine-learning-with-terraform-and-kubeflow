# Serve BAAI Bge Reranker Large using Triton Inference Server on AWS Neuron

This example shows how to serve [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) model using [Triton Inference Server](https://github.com/triton-inference-server) on [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html) with [torch-neuronx](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html).

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../../../README.md#prerequisites) and [Getting started](../../../../../README.md#getting-started). 

See [What is in the YAML file](../../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.
    
## Build and Push Docker Container

This example uses a custom Docker container for Triton Inference Server with PyTorch Neuronx. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/tritonserver-neuronx/build_tools/build_and_push.sh aws-region

## Hugging Face BAAI Bge Reranker Large pre-trained model weights

To download Hugging Face BAAI Bge Reranker Large pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-baai-bge-reranker-large-neuron     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"BAAI/bge-reranker-large"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-baai-bge-reranker-large-neuron -n kubeflow-user-example-com


## Launch Triton Server

To launch Triton server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-baai-bge-reranker-large-neuron \
        charts/machine-learning/serving/triton-inference-server \
        -f examples/inference/triton-inference-server/python_backend/baai-bge-reranker-large-neuron/triton_server.yaml -n kubeflow-user-example-com


## Stop Service

To stop the service:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall triton-server-baai-bge-reranker-large-neuron -n kubeflow-user-example-com

### Logs

Triton server logs are available in `/efs/home/triton-server-baai-bge-reranker-large-neuron/logs` folder. 
