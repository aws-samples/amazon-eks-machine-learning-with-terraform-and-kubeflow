# Ray Serve BAAI BGE Reranker v2-m3 Model

This example illustrates how to use [Ray Serve](../../../charts/machine-learning/training/rayserve/) Helm chart to serve [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) model for text reranking and embedding tasks.

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files. There are some fields that are specific to a machine learning chart.

## Build and Push Docker Container

This example uses a custom Docker container for Ray Serve. Build and push this container using following command (replace `aws-region` with your AWS Region name):

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     ./containers/ray-pytorch/build_tools/build_and_push.sh aws-region


## Hugging Face BGE Reranker v2-m3 Pre-trained Model Weights

To download Hugging Face BGE Reranker v2-m3 pre-trained model weights, replace `YourHuggingFaceToken` with your Hugging Face token below, and execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-bge-reranker-v2-m3     \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"BAAI/bge-reranker-v2-m3"},{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}]' \
            -n kubeflow-user-example-com

Uninstall the Helm chart at completion:

    helm uninstall rayserve-bge-reranker-v2-m3 -n kubeflow-user-example-com

## Launch Ray Service

To launch Ray Service,  execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug rayserve-bge-reranker-v2-m3 \
        charts/machine-learning/serving/rayserve/ \
        -f examples/inference/rayserve/baai-bge-reranker-v2-m3-vllm/rayservice.yaml -n kubeflow-user-example-com

## Stop Service

To stop the service:

     cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
     helm uninstall rayserve-bge-reranker-v2-m3 -n kubeflow-user-example-com

## Model Usage

The BGE Reranker v2-m3 model is designed for:
- **Text Reranking**: Reorder search results based on relevance to a query
- **Cross-lingual Support**: Works with multiple languages including English, Chinese, and others
- **Embedding Generation**: Generate dense vector representations for text

### API Endpoints

Once deployed, the service exposes endpoints for:
- `/v1/rerank` - Rerank a list of documents given a query
- `/v1/embeddings` - Generate embeddings for input text

### Example Usage

```python
import requests

# Reranking example
response = requests.post("http://service-url:8000/v1/rerank", json={
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of AI",
        "Cooking recipes for beginners", 
        "Deep learning uses neural networks"
    ]
})

# Embedding example  
response = requests.post("http://service-url:8000/v1/embeddings", json={
    "texts": ["Hello world", "Machine learning is fascinating"]
})
```