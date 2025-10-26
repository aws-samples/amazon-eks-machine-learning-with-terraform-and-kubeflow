# Ray Serve Mistral Pixtral 12B 2409 Model

This example illustrates how to use [Ray Serve](../../../charts/machine-learning/training/rayserve/) Helm chart to serve [mistralai/Pixtral-12B-2409](https://huggingface.co/mistralai/Pixtral-12B-2409) model.

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/ray-pytorch/build_tools/build_and_push.sh aws-region


## Hugging Face Pixtral 12B 2409 Pre-trained Model Weights

To download Hugging Face Pixtral 12B 2409 pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-pixtral-12b-2409     \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"mistralai/Pixtral-12B-2409"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-pixtral-12b-2409 -n kubeflow-user-example-com

## Build Ray Serve Engine Config

To build Ray Serve engine:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-pixtral-12b-2409     \
            charts/machine-learning/data-prep/data-process   \
            -f examples/inference/rayserve/pixtral-12b-2409-vllm/engine_config.yaml \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-pixtral-12b-2409 -n kubeflow-user-example-com

## Build Ray Serve Engine

To build Ray Serve engine:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-pixtral-12b-2409     \
            charts/machine-learning/model-prep/rayserve-vllm-asyncllmengine    \
            --set='engine_path=/fsx/rayserve/engines/vllm_asyncllmengine.zip' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-pixtral-12b-2409 -n kubeflow-user-example-com

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-pixtral-12b-2409 \
        charts/machine-learning/serving/rayserve/ \
        -f examples/inference/rayserve/pixtral-12b-2409-vllm/rayservice.yaml -n kubeflow-user-example-com

## Stop Service

To stop the service:

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     helm uninstall rayserve-pixtral-12b-2409 -n kubeflow-user-example-com