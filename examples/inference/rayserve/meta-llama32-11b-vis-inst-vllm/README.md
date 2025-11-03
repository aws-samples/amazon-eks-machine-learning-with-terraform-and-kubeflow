# Ray Serve Meta Llama 3.2 11B Vision Instruct Model

This example illustrates how to use [Ray Serve](../../../charts/machine-learning/training/rayserve/) Helm chart to serve [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) multi-modal model.

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/ray-pytorch/build_tools/build_and_push.sh aws-region


## Hugging Face Meta Llama 3.2 11B Vision Instruct Pre-trained Model Weights

To download Hugging Face Llama3 8B Instruct pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-llama32-11b-vis-inst-vllm     \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Llama-3.2-11B-Vision-Instruct"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-llama32-11b-vis-inst-vllm -n kubeflow-user-example-com

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-llama32-11b-vis-inst-vllm \
        charts/machine-learning/serving/rayserve/ \
        -f examples/inference/rayserve/meta-llama32-11b-vis-inst-vllm/rayservice.yaml -n kubeflow-user-example-com

## Stop Service

To stop the service:

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     helm uninstall rayserve-llama32-11b-vis-inst-vllm -n kubeflow-user-example-com