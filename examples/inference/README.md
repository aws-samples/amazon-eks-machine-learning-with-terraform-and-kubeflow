## Inference Tutorials

### [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- | 
| [BART Large CNN](./rayserve/facebook-bart-large-cnn/README.md)    | [PyTorch](https://pytorch.org/)   | Nvidia GPU | Devices=1, TP=1, PP=1 |
| [Llama 3 8B Instruct](./rayserve/meta-llama3-8b-neuron/README.md)    | [Transfomers Neuronx](https://github.com/aws-neuron/transformers-neuronx)     | AWS Inferentia2 | Cores=8, TP=8, PP=1 (Deprecated) |
| [Llama 3 8B Instruct](./rayserve/meta-llama3-8b-vllm/README.md)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Devices=8, TP=8, PP=1 |
| [Llama 3 8B Instruct](./rayserve/meta-llama3-8b-vllm-neuron/README.md)    | [vLLM](https://github.com/vllm-project/vllm)    | AWS Inferentia2 | Cores=8, TP=8, PP=1 |
| [Llama 3.2 11B Vision Instruct](./rayserve/meta-llama32-11b-vis-inst-vllm/README.md)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Devices=8, TP=8, PP=1, Multi-modal |
| [Llama 3.3 70B Instruct](./rayserve/meta-llama33-70b-instruct-vllm/README.md)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Nodes=2, Devices=8,  TP=8, PP=2, Multi-Node Inference |
| [Llama 3.3 70B Instruct](./rayserve/meta-llama33-70b-instruct-neuron/README.md)    | [Transfomers Neuronx](https://github.com/aws-neuron/transformers-neuronx)     | AWS Trainium1 |  Nodes=2, TP=8, PP=2, Multi-node Inference (Deprecated) |
| [Mistral 8x22B Instruct v0.1](./rayserve/mistral-8x22b-instruct-v01-vllm/README.md)    | [vLLM](https://github.com/vllm-project/vllm)    | Nvidia GPU | Nodes=2, Devices=8, TP=8, PP=2, Multi-node inference |


### [Triton Inference Server](https://github.com/triton-inference-server/server)

#### [Python Backend](https://github.com/triton-inference-server/python_backend)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- |
| [BAAI BGE Reranker Large](./triton-inference-server/python_backend/baai-bge-reranker-large-neuron/README.md)    | [PyTorch](https://pytorch.org/)   | AWS Inferentia2 | Cores=1, TP=8, PP=1 |
| [Llama 3 8B Instruct](./triton-inference-server/python_backend/llama3-8b-instruct-lmi-neuron/README.md)    |[LMI Transformers Neuronx](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/tnx_user_guide.html)   | AWS Inferentia2 | Cores=8, TP=8, PP=1 (Deprecated) |
| [Llama 3 8B Instruct](./triton-inference-server/python_backend/llama3-8b-instruct-neuron/README.md)    |[Transfomers Neuronx](https://github.com/aws-neuron/transformers-neuronx)   | AWS Inferentia2 | Cores=8, TP=8, PP=1 (Deprecated) |
| [Mistral 7B Instruct v0.1](./triton-inference-server/python_backend/mistral-7b-instruct-v01-neuron/README.md)    |[Transfomers Neuronx](https://github.com/aws-neuron/transformers-neuronx)   | AWS Inferentia2 |  Cores=8, TP=8, PP=1 (Deprecated) |
| [XLM Roberta Base](./triton-inference-server/python_backend/xlm-roberta-base-neuron/README.md)    |[PyTorch](https://pytorch.org/)   | AWS Inferentia2 | Cores=1, TP=1, PP=1 |


#### [vLLM Backend](https://github.com/triton-inference-server/vllm_backend)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- |
| [DeepSeek R1 Distill Llama 8B](./triton-inference-server/vllm_backend/deepseek-r1-distill-llama-8b/README.md)    |[vLLM](https://github.com/vllm-project/vllm)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [DeepSeek R1 Distill Llama 8B](./triton-inference-server/vllm_backend/deepseek-r1-distill-llama-8b-neuron/README.md)    |[vLLM](https://github.com/vllm-project/vllm)   | AWS Trainium1 |  Cores=16, TP=16, PP=1 |
| [Llama 3 8B Instruct](./triton-inference-server/vllm_backend/llama3-8b-instruct/README.md)    |[vLLM](https://github.com/vllm-project/vllm)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [Llama 3 8B Instruct](./triton-inference-server/vllm_backend/llama3-8b-instruct-neuron/README.md)    |[vLLM](https://github.com/vllm-project/vllm)   | AWS Inferentia2 |  Cores=8, TP=8, PP=1 |
| [Mistral 7B Instruct v0.2](./triton-inference-server/vllm_backend/mistral-7b-instruct-v02/README.md)    |[vLLM](https://github.com/vllm-project/vllm)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [Mistral 7B Instruct v0.2](./triton-inference-server/vllm_backend/mistral-7b-instruct-v02-neuron/README.md)    |[vLLM](https://github.com/vllm-project/vllm)   | AWS Inferentia2 |  Cores=8, TP=8, PP=1 |
| [Mistral 8x22B Instruct v0.1](./triton-inference-server/ray_vllm_backend/mistral-8x22b-instruct-v01/README.md)    |[vLLM](https://github.com/vllm-project/vllm)   | Nvidia GPU |  Node=2, Cores=8, TP=8, PP=2, Multi-Node inference using [lws](https://github.com/kubernetes-sigs/lws) with [Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html) |


#### [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- | 
| [Llama2 7B](./triton-inference-server/tensorrtllm_backend/llama2-7b/README.md)    |[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [Llama 3 8B Instruct](./triton-inference-server/tensorrtllm_backend/llama3-8b-instruct/README.md)    |[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [Mistral 7B Instruct v0.1](./triton-inference-server/tensorrtllm_backend/mistral-7b-instruct-v01/README.md)    |[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)   | Nvidia GPU |  Devices=8, TP=8, PP=1 |
| [Mistral 8x22B Instruct v0.1](./triton-inference-server/tensorrtllm_backend/mistral-8x22b-instruct-v01/README.md)    |[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)   | Nvidia GPU |  Nodes=2, Devices=8, TP=8, PP=2, Multi-node inference using [lws](https://github.com/kubernetes-sigs/lws) |
| [Mistral 7B Instruct v0.1, Llama 3 8B Instruct](./triton-inference-server/tensorrtllm_backend/mistral-7b-instruct-v01_llama3-8b/README.md)    |[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)   | Nvidia GPU |  Devices=8, TP=8, PP=1, [Multi-model concurrent model execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_execution.html) |


### [DJL Serving](https://github.com/deepjavalibrary/djl-serving)

**Following examples are deprecated.**

#### [LMI TensorRT-LLM Engine](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/trt_llm_user_guide.html)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- | 
| [Llama3 8B Instruct](./djl-serving/tensorrt-llm/llama3-8b-instruct/README.md)    | [LMI TensorRT-LLM](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/trt_llm_user_guide.html)    | Nvidia GPU |  Devices=8, TP=8, PP=1 (Deprecated) |
| [Mistral 7B Instruct v0.2](./djl-serving/tensorrt-llm/mistral-7b-instruct-v0.2/README.md)    | [LMI TensorRT-LLM](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/trt_llm_user_guide.html)    | Nvidia GPU |  Devices=8, TP=8, PP=1 (Deprecated) |


#### [LMI Transformers Neuronx Engine](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/tnx_user_guide.html)

| Model      | Inference Engine | Accelerator | Notes |
| ----------- | ----------- | ------------ | ----------- | 
| [Llama3 8B Instruct](./djl-serving/transformers-neuronx/llama3-8b-instruct/README.md)    | [LMI Transformers-Neuronx](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/tnx_user_guide.html)    | AWS Inferentia2 |  Cores=8, TP=8, PP=1 (Deprecated) |
| [Mistral 7B Instruct v0.2](./djl-serving/transformers-neuronx/mistral-7b-instruct-v0.2/README.md)    | [LMI Transformers-Neuronx](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/tnx_user_guide.html)    | AWS Inferentia2 |  Cores=8, TP=8, PP=1 (Deprecated) |
