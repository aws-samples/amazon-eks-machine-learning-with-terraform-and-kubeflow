# Ray Server Mistral 8x22B Instruct v0.1 Model

This example shows how to serve [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) model using multi-GPU, multi-node deployment. 

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./containers/ray-pytorch/build_tools/build_and_push.sh aws-region


### Hugging Face Mistral 8x22B Instruct v0.1 pre-trained model weights

To download Hugging Face Mistral 8x22B Instruct v0.1 pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-mistral-8x22b-instruct-v01    \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"mistralai/Mixtral-8x22B-Instruct-v0.1"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-mistral-8x22b-instruct-v01 -n kubeflow-user-example-com

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-mistral-8x22b-instruct-v01\
        charts/machine-learning/serving/rayserve/ \
        -f examples/inference/rayserve/mistral-8x22b-instruct-v01-vllm/rayservice.yaml -n kubeflow-user-example-com

## Stop Service

To stop the service:

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     helm uninstall rayserve-mistral-8x22b-instruct-v01 -n kubeflow-user-example-com