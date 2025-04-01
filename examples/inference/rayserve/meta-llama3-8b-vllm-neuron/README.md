# Ray Serve Meta Llama3 8B Instruct Model using vLLM on Neuronx

This example illustrates how to use [Ray Serve](../../../charts/machine-learning/training/rayserve/) Helm chart to serve [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model using [vLLM](https://github.com/vllm-project/vllm) on [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html) with [transformers-neuronx](https://github.com/aws-neuron/transformers-neuronx).

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started). See [What is in the YAML file](../../../README.md#what-is-in-the-yaml-file) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./containers/ray-pytorch-neuronx-vllm/build_tools/build_and_push.sh aws-region


## Hugging Face Llama3 8B Instruct Pre-trained Model Weights

To download Hugging Face Llama3 8B Instruct pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-llama3-8b-instruct-vllm-nx      \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Meta-Llama-3-8B-Instruct"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-llama3-8b-instruct-vllm-nx  -n kubeflow-user-example-com

## Build Ray Serve Engine Config

To build Ray Serve engine config:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-llama3-8b-instruct-vllm-nx      \
            charts/machine-learning/data-prep/data-process   \
            -f examples/inference/rayserve/meta-llama3-8b-vllm-neuron/engine_config.yaml \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-llama3-8b-instruct-vllm-nx  -n kubeflow-user-example-com


## Build Ray Serve Engine

To build Ray Serve engine:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-llama3-8b-instruct-vllm-nx      \
            charts/machine-learning/model-prep/rayserve-vllm-asyncllmengine    \
            --set='engine_path=/fsx/rayserve/engines/vllm_asyncllmengine.zip' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-llama3-8b-instruct-vllm-nx  -n kubeflow-user-example-com

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-llama3-8b-instruct-vllm-nx  \
        charts/machine-learning/serving/rayserve \
        -f examples/inference/rayserve/meta-llama3-8b-vllm-neuron/rayservice.yaml -n kubeflow-user-example-com

To check the status of Ray Service:
    kubectl describe rayservice rayservice-rayserve-llama3-8b-instruct-vllm-nx \
        -n kubeflow-user-example-com

## Stop Service

To stop the service:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall rayserve-llama3-8b-instruct-vllm-nx  -n kubeflow-user-example-com