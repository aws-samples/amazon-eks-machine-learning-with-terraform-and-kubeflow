# Serve Mistral 8x22B Instruct v0.1 using Triton Inference Server with vLLM and Ray

This example shows how to serve [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) model in a multi-GPU, multi-node deployment, using [Triton Inference Server](https://github.com/triton-inference-server) with [vLLM backend](https://github.com/triton-inference-server/vllm_backend/tree/main) and [Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html).  

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../../../README.md#prerequisites) and [Getting started](../../../../../README.md#getting-started). 

See [What is in the YAML file](../../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Tritonserver TensorRT-LLM. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/tritonserver-ray-vllm/build_tools/build_and_push.sh aws-region
     
## Hugging Face Mistral 8x22B Instruct v0.1 pre-trained model weights

To download Hugging Face Mistral 8x22B Instruct v0.1 pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-mistral-8x22b-instruct-v01-ray-vllm    \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \"mistralai/Mixtral-8x22B-Instruct-v0.1"
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall triton-server-mistral-8x22b-instruct-v01-ray-vllm-n kubeflow-user-example-com

## Launch Triton Server

To launch Triton server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug triton-server-mistral-8x22b-instruct-v01-ray-vllm\
        charts/machine-learning/serving/triton-inference-server-lws \
        -f examples/inference/triton-inference-server/ray_vllm_backend/mistral-8x22b-instruct-v01/triton_server.yaml -n kubeflow-user-example-com


## Stop Service

To stop the service:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall triton-server-mistral-8x22b-instruct-v01-ray-vllm-n kubeflow-user-example-com

### Logs

Triton server logs are available in `/efs/home/triton-server-mistral-8x22b-instruct-v01-ray-vllm/logs` folder. 
