# Ray Serve DeepSeek R1 Distill Qwen 32B Model

This example illustrates how to use [Ray Serve](../../../charts/machine-learning/training/rayserve/) Helm chart to serve [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) model.

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/ray-pytorch/build_tools/build_and_push.sh aws-region


## Hugging Face DeepSeek R1 Distill Qwen 32B Pre-trained Model Weights

To download Hugging Face DeepSeek R1 Distill Qwen 32B pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-deepseek-r1-distill-qwen-32b     \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-deepseek-r1-distill-qwen-32b -n kubeflow-user-example-com

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-deepseek-r1-distill-qwen-32b \
        charts/machine-learning/serving/rayserve/ \
        -f examples/inference/rayserve/deepseek-r1-distill-qwen-32b-vllm/rayservice.yaml -n kubeflow-user-example-com

## Stop Service

To stop the service:

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     helm uninstall rayserve-deepseek-r1-distill-qwen-32b -n kubeflow-user-example-com