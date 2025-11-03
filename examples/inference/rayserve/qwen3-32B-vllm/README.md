# Ray Serve Qwen3 32B Model

This example illustrates how to use [Ray Serve](../../../charts/machine-learning/training/rayserve/) Helm chart to serve [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) model.

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/ray-pytorch/build_tools/build_and_push.sh aws-region

## Hugging Face Qwen3 32B Pre-trained Model Weights

To download Hugging Face Qwen3 32B pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-qwen3-32b     \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"Qwen/Qwen3-32B"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-qwen3-32b -n kubeflow-user-example-com

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-qwen3-32b \
        charts/machine-learning/serving/rayserve/ \
        -f examples/inference/rayserve/qwen3-32B-vllm/rayservice.yaml -n kubeflow-user-example-com

## Stop Service

To stop the service:

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     helm uninstall rayserve-qwen3-32b -n kubeflow-user-example-com