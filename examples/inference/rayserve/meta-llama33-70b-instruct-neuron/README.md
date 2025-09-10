# Ray Serve Meta Llama3 8B Instruct Model

## This example is deprecated.

This example illustrates how to use [Ray Serve](../../../charts/machine-learning/training/rayserve/) Helm chart to serve [Meta Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)  model with [Transformers Neuronx](https://github.com/aws-neuron/transformers-neuronx) in a multi-node deployment.

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.
    
## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/ray-pytorch-neuronx/build_tools/build_and_push.sh aws-region

## Hugging Face Llama3 8B Instruct Pre-trained Model Weights

To download Hugging Face Llama3 8B Instruct pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-llama33-70b-instruct-nx   \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Llama-3.3-70B-Instruct"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-llama33-70b-instruct-nx -n kubeflow-user-example-com

## Build Ray Serve Engine Config

To build Ray Serve engine config:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-llama33-70b-instruct-nx   \
            charts/machine-learning/data-prep/data-process   \
            -f examples/inference/rayserve/meta-llama33-70b-instruct-neuron/engine_config.yaml \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-llama33-70b-instruct-nx -n kubeflow-user-example-com

## Build Ray Serve Engine

To build Ray Serve engine:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-llama33-70b-instruct-nx   \
            charts/machine-learning/model-prep/rayserve-tnx-autocausalengine    \
            --set='engine_path=/fsx/rayserve/engines/tnx_autocausalengine.zip' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-llama33-70b-instruct-nx -n kubeflow-user-example-com

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-llama33-70b-instruct-nx \
        charts/machine-learning/serving/rayserve/ \
        -f examples/inference/rayserve/meta-llama33-70b-instruct-neuron/rayservice.yaml -n kubeflow-user-example-com

## Stop Service

To stop the service:

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     helm uninstall rayserve-llama33-70b-instruct-nx -n kubeflow-user-example-com