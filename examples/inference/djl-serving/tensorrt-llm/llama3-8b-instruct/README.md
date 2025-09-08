# Serve Meta Llama 3 8B Instruct using TensorRT-LLM Engine in LMI

This example shows how to serve [Meta Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model using [TensorRT-LLM Engine in LMI](https://docs.djl.ai/docs/serving/serving/docs/lmi/user_guides/trt_llm_user_guide.html).

## Prerequisites

Before proceeding, complete the [Prerequisites](../../../../../README.md#prerequisites) and [Getting started](../../../../../README.md#getting-started). 

See [What is in the YAML file](../../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.


## Hugging Face Meta Llama 3 8B Instruct  pre-trained model weights

To download Hugging Face Meta Llama 3 8B Instruct  pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug djl-lmi-llama3-8b-instruct-trtllm    \
        charts/machine-learning/model-prep/hf-snapshot    \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"meta-llama/Meta-Llama-3-8B-Instruct"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
        -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall djl-lmi-llama3-8b-instruct-trtllm -n kubeflow-user-example-com


## Launch DJL LMI Server

To launch the server:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug djl-lmi-llama3-8b-instruct-trtllm\
        charts/machine-learning/serving/djl-lmi-server \
        -f examples/inference/djl-serving/tensorrt-llm/llama3-8b-instruct/server.yaml -n kubeflow-user-example-com


## Stop Service

To stop the service:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall djl-lmi-llama3-8b-instruct-trtllm -n kubeflow-user-example-com

### Logs

Triton server logs are available in `/efs/home/djl-lmi-llama3-8b-instruct-trtllm/logs` folder. 
