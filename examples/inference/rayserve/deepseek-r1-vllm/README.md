# Ray Serve DeepSeek-R1 Model

This example illustrates how to use [Ray Serve](../../../charts/machine-learning/training/rayserve/) Helm chart to serve [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) model, using multi-GPU, multi-node deployment. 

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./containers/ray-pytorch/build_tools/build_and_push.sh aws-region


## Hugging Face DeepSeek-R1 Pre-trained Model Weights

To download Hugging Face DeepSeek-R1 pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-deepseek-r1    \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set 'ebs.storage=1000Gi' \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"deepseek-ai/DeepSeek-R1"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-deepseek-r1 -n kubeflow-user-example-com

## Build Ray Serve Engine Config

To build Ray Serve engine config:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-deepseek-r1    \
            charts/machine-learning/data-prep/data-process   \
            -f examples/inference/rayserve/deepseek-r1-vllm/engine_config.yaml \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-deepseek-r1 -n kubeflow-user-example-com

## Build Ray Serve Engine

To build Ray Serve engine:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-deepseek-r1    \
            charts/machine-learning/model-prep/rayserve-vllm-asyncllmengine    \
            --set='engine_path=/fsx/rayserve/engines/vllm_asyncllmengine.zip' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-deepseek-r1 -n kubeflow-user-example-com

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-deepseek-r1\
        charts/machine-learning/serving/rayserve/ \
        -f examples/inference/rayserve/deepseek-r1-vllm/rayservice.yaml -n kubeflow-user-example-com

## Stop Service

To stop the service:

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     helm uninstall rayserve-deepseek-r1 -n kubeflow-user-example-com