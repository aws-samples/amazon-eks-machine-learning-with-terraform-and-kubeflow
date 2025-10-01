# Ray Serve Qwen2.5-VL-32B-Instruct Model

This example illustrates how to use [Ray Serve](../../../charts/machine-learning/training/rayserve/) Helm chart to serve [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) model.

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/ray-pytorch/build_tools/build_and_push.sh aws-region


## Hugging Face Qwen2.5-VL-32B-Instruct Pre-trained Model Weights

To download Hugging Face Qwen2.5-VL-32B-Instruct pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-qwen25-vl-32b-instruct     \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"Qwen/Qwen2.5-VL-32B-Instruct"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-qwen25-vl-32b-instruct -n kubeflow-user-example-com

## Build Ray Serve Engine Config

To build Ray Serve engine:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-qwen25-vl-32b-instruct     \
            charts/machine-learning/data-prep/data-process   \
            -f examples/inference/rayserve/qwen25-vl-32B-instruct-vllm/engine_config.yaml \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-qwen25-vl-32b-instruct -n kubeflow-user-example-com

## Build Ray Serve Engine

To build Ray Serve engine:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-qwen25-vl-32b-instruct     \
            charts/machine-learning/model-prep/rayserve-vllm-asyncllmengine    \
            --set='engine_path=/fsx/rayserve/engines/vllm_asyncllmengine.zip' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-qwen25-vl-32b-instruct -n kubeflow-user-example-com

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-qwen25-vl-32b-instruct \
        charts/machine-learning/serving/rayserve/ \
        -f examples/inference/rayserve/qwen25-vl-32B-instruct-vllm/rayservice.yaml -n kubeflow-user-example-com

## Stop Service

To stop the service:

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     helm uninstall rayserve-qwen25-vl-32b-instruct -n kubeflow-user-example-com