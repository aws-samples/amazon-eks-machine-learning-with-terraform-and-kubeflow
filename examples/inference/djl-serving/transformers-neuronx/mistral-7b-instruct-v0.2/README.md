# Serve Mistral 7B Instruct Instruct v0.2 using Transformer-Neuronx Engine in LMI

This example shows how to serve [Mistral 7B Instruct v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model using [Transformers-Neuronx Engine in LMI](https://docs.djl.ai/docs/serving/serving/docs/lmi/user_guides/tnx_user_guide.html).

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../../../README.md#prerequisites) and [Getting started](../../../../../README.md#getting-started). 

See [What is in the YAML file](../../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.


## Hugging Face Mistral 7B Instruct v0.2 pre-trained model weights

To download Hugging Face Mistral 7B Instruct v0.2 pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug djl-lmi-mistral-7b-instruct-v02-tnx     \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"mistralai/Mistral-7B-Instruct-v0.2"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall djl-lmi-mistral-7b-instruct-v02-tnx -n kubeflow-user-example-com


## Launch DJL LMI Server

To launch the server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug djl-lmi-mistral-7b-instruct-v02-tnx \
        charts/machine-learning/serving/djl-lmi-server \
        -f examples/inference/djl-serving/transformers-neuronx/mistral-7b-instruct-v0.2/server.yaml -n kubeflow-user-example-com


## Stop Service

To stop the service:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall djl-lmi-mistral-7b-instruct-v02-tnx -n kubeflow-user-example-com

### Logs

Triton server logs are available in `/efs/home/djl-lmi-mistral-7b-instruct-v02-tnx/logs` folder. 
