## Inference Tutorials

### [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- | 
| [BAAI BGE Reranker v2 M3](./rayserve/baai-bge-reranker-v2-m3-vllm/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Devices=8, TP=8, PP=1 |
| [DeepSeek R1](./rayserve/deepseek-r1-vllm/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Devices=8, TP=8, PP=1 |
| [DeepSeek R1 Distill Qwen 32B](./rayserve/deepseek-r1-distill-qwen-32b-vllm/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Devices=8, TP=8, PP=1 |
| [Llama 3 8B Instruct](./rayserve/meta-llama3-8b-vllm/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Devices=8, TP=8, PP=1 |
| [Llama 3 8B Instruct](./rayserve/meta-llama3-8b-vllm-neuron/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | AWS Inferentia2 | Cores=8, TP=8, PP=1 |
| [Llama 3.2 11B Vision Instruct](./rayserve/meta-llama32-11b-vis-inst-vllm/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Devices=8, TP=8, PP=1, Multi-modal |
| [Llama 3.3 70B Instruct](./rayserve/meta-llama33-70b-instruct-vllm/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Nodes=2, Devices=8,  TP=8, PP=2, Multi-Node Inference |
| [Mixtral 8x22B Instruct v0.1](./rayserve/mixtral-8x22b-instruct-v01-vllm/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Nodes=2, Devices=8, TP=8, PP=2, Multi-node inference |
| [Pixtral 12B 2409](./rayserve/pixtral-12b-2409-vllm/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Devices=8, TP=8, PP=1, Multi-modal |
| [Qwen 2.5 VL 32B Instruct](./rayserve/qwen25-vl-32B-instruct-vllm/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Devices=8, TP=8, PP=1, Multi-modal |
| [Qwen 3 32B](./rayserve/qwen3-32B-vllm/serve.ipynb)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Devices=8, TP=8, PP=1 |

#### Streaming Ray Serve Logs

Access Ray Serve logs via kubectl using the `log-streamer` sidecar container:

```bash
# Controller/deployment logs (head pod)
kubectl logs <head-pod> -c log-streamer -n <namespace>

# Inference/replica logs (worker pod)
kubectl logs <worker-pod> -c log-streamer -n <namespace>

# Stream logs in real-time
kubectl logs -f <pod> -c log-streamer -n <namespace>
```

Ray Dashboard logs also remain accessible via the Dashboard UI.

### [Triton Inference Server](https://github.com/triton-inference-server/server)

#### [Python Backend](https://github.com/triton-inference-server/python_backend)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- |
| [BAAI BGE Reranker Large](./triton-inference-server/python_backend/baai-bge-reranker-large-neuron/serve.ipynb)    | [PyTorch](https://pytorch.org/)   | AWS Inferentia2 | Cores=1, TP=8, PP=1 |
| [XLM Roberta Base](./triton-inference-server/python_backend/xlm-roberta-base-neuron/serve.ipynb)    |[PyTorch](https://pytorch.org/)   | AWS Inferentia2 | Cores=1, TP=1, PP=1 |


#### [vLLM Backend](https://github.com/triton-inference-server/vllm_backend)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- |
| [DeepSeek R1 Distill Llama 8B](./triton-inference-server/vllm_backend/deepseek-r1-distill-llama-8b/serve.ipynb)    |[vLLM](https://github.com/vllm-project/vllm)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [DeepSeek R1 Distill Llama 8B](./triton-inference-server/vllm_backend/deepseek-r1-distill-llama-8b-neuron/serve.ipynb)    |[vLLM](https://github.com/vllm-project/vllm)   | AWS Trainium1 |  Cores=16, TP=16, PP=1 |
| [Llama 3 8B Instruct](./triton-inference-server/vllm_backend/llama3-8b-instruct/serve.ipynb)    |[vLLM](https://github.com/vllm-project/vllm)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [Llama 3 8B Instruct](./triton-inference-server/vllm_backend/llama3-8b-instruct-neuron/serve.ipynb)    |[vLLM](https://github.com/vllm-project/vllm)   | AWS Inferentia2 |  Cores=8, TP=8, PP=1 |
| [Mistral 7B Instruct v0.2](./triton-inference-server/vllm_backend/mistral-7b-instruct-v02/serve.ipynb)    |[vLLM](https://github.com/vllm-project/vllm)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [Mistral 7B Instruct v0.2](./triton-inference-server/vllm_backend/mistral-7b-instruct-v02-neuron/serve.ipynb)    |[vLLM](https://github.com/vllm-project/vllm)   | AWS Inferentia2 |  Cores=8, TP=8, PP=1 |
| [Qwen 3 0.6B](./triton-inference-server/vllm_backend/qwen3-0.6b/serve.ipynb)    |[vLLM](https://github.com/vllm-project/vllm)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |


#### [Ray vLLM Backend](https://github.com/triton-inference-server/vllm_backend)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- |
| [DeepSeek R1](./triton-inference-server/ray_vllm_backend/deepseek-R1/serve.ipynb)    |[vLLM](https://github.com/vllm-project/vllm)   | Nvidia GPU |  Node=2, Cores=8, TP=8, PP=2, Multi-Node inference using [lws](https://github.com/kubernetes-sigs/lws) with [Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html) |
| [Mixtral 8x22B Instruct v0.1](./triton-inference-server/ray_vllm_backend/mixtral-8x22b-instruct-v01/serve.ipynb)    |[vLLM](https://github.com/vllm-project/vllm)   | Nvidia GPU |  Node=2, Cores=8, TP=8, PP=2, Multi-Node inference using [lws](https://github.com/kubernetes-sigs/lws) with [Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html) |


#### [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- | 
| [Llama 3 8B Instruct](./triton-inference-server/tensorrtllm_backend/llama3-8b-instruct/serve.ipynb)    |[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [Mistral 7B Instruct v0.1](./triton-inference-server/tensorrtllm_backend/mistral-7b-instruct-v01/serve.ipynb)    |[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [Mixtral 8x22B Instruct v0.1](./triton-inference-server/tensorrtllm_backend/mixtral-8x22b-instruct-v01/serve.ipynb)    |[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)   | Nvidia GPU |  Nodes=2, Devices=8, TP=8, PP=2, Multi-node inference using [lws](https://github.com/kubernetes-sigs/lws) |
| [Mistral 7B Instruct v0.1, Llama 3 8B Instruct](./triton-inference-server/tensorrtllm_backend/mistral-7b-instruct-v01_llama3-8b/serve.ipynb)    |[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)   | Nvidia GPU |  Devices=8, TP=8, PP=1, [Multi-model concurrent model execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_execution.html) |
