## Training Tutorials


### [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/en/index)

| Model      | Accelerator | Notes |
| ----------- | ----------- | -------- |
| [Bert GLUE MRPC Pre-training](./accelerate/bert-glue-mrpc/README.md)    | Nvidia GPU  | BERT Glue MRPC Pretraining    |
| [Llama 2 7B Fine-tuning](./accelerate/llama2-ft-fsdp/README.md)    | Nvidia GPU | Llama 2 7B Fine-tuning with FSDP on smangrul/code-chat-assistant-v1 |
| [Llama 2 13B Fine-tuning](./accelerate/llama2-ft-fsdp/README.md)    | Nvidia GPU  | Llama 2 13B Fine-tuning with FSDP on smangrul/code-chat-assistant-v1|
| [Llama 2 70B Fine-tuning](./accelerate/llama2-ft-fsdp/README.md)   | Nvidia GPU   | Llama 2 70B Fine-tuning with FSDP on smangrul/code-chat-assistant-v1|

### [Megatron DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed)


| Model      | Accelerator | Notes |
| ----------- | ----------- | -------- |
| [GPT2 345M Pre-train](./megatron-deepspeed/gpt2_345m/README.md)  | Nvidia GPU    | Pre-train GPT2 345M on Wikicorpus dataset  |

### [Nemo](https://github.com/NVIDIA/NeMo)

| Model      | Accelerator | Notes |
| ----------- | ----------- | -------- |
| [Llama 2 7B PEFT LoRA](./nemo-megatron/llama2-7b-peft/README.md)  | Nvidia GPU     | Llama 2 7B PEFT LoRA on Pubmedqa dataset |
| [Llama 3.1 8B PEFT LoRA](./nemo-megatron/llama31-8b-peft-dolphin/README.md) | Nvidia GPU       | Llama 3.1 8B PEFT LoRA on Dolphin dataset |
| [Mistral 7B v0.1 PEFT LoRA](./nemo-megatron/mistral-7b-v01-peft/README.md)   | Nvidia GPU     | Mistral 7B v0.1 PEFT LoRA on Pubmedqa dataset |
| [Mistral 7B v0.1  PEFT LoRA](./nemo-megatron/mistral-7b-v01-peft-dolphin/README.md) | Nvidia GPU       | Mistral 7B v0.1 PEFT LoRA  PEFT LoRA on Dolphin dataset |

### [Neuronx Distributed](https://github.com/aws-neuron/neuronx-distributed)

| Model      | Accelerator | Notes |
| ----------- | ----------- | -------- |
| [GPT Neox 6.9B Pre-train ](./neuronx-distributed/gpt_neox_6.9b/README.md)   | AWS Trainium1  | GPT Neox 6.9B pre-train on Wikicorpus |
| [GPT Neox 20B Pre-train ](./neuronx-distributed/gpt_neox_20b/README.md)     | AWS Trainium1  | GPT Neox 20B pre-train on Wikicorpus |
| [Llama 2 7B Pre-train ](./neuronx-distributed/llama2_7b/README.md)     | AWS Trainium1  | Llama2 7B pre-train on Wikicorpus |
| [Llama 2 7B Pre-train](./neuronx-distributed/llama2_7b_ptl/README.md) | AWS Trainium1     | Llama2 7B pre-train on Wikicorpus with PyTorch Lightning |
| [Llama 2 13B Pre-train ](./neuronx-distributed/llama2_13b/README.md)   | AWS Trainium1   | Llama2 13B pre-train on Wikicorpus |
| [Llama 2 13B Pre-train](./neuronx-distributed/llama2_13b_ptl/README.md) | AWS Trainium1     | Llama2 13B pre-train on Wikicorpus with PyTorch Lightning |
| [Llama 2 70B Pre-train ](./neuronx-distributed/llama2_70b/README.md)   | AWS Trainium1   | Llama2 70B pre-train on Wikicorpus |
| [Llama 2 70B Pre-train](./neuronx-distributed/llama2_70b_ptl/README.md)    | AWS Trainium1  | Llama2 70B pre-train on Wikicorpus with PyTorch Lightning |
| [Llama 3 8B Pre-train ](./neuronx-distributed/llama3_8b/README.md)  | AWS Trainium1    | Llama3 8B pre-train on Wikicorpus |
| [Llama 3 70B Pre-train ](./neuronx-distributed/llama3_70b/README.md)  | AWS Trainium1    | Llama3 70B pre-train on Wikicorpus |
| [Llama 3 70B Pre-train ](./neuronx-distributed/llama3_70b_ptl/README.md)  | AWS Trainium1    | Llama3 70B pre-train on Wikicorpus with PyTorch Lightning |
| [Llama 3.1 8B Pre-train ](./neuronx-distributed/llama31_8b/README.md)   | AWS Trainium1   | Llama3.1 8B pre-train on Wikicorpus |
| [Llama 3.1 70B Pre-train ](./neuronx-distributed/llama31_70b/README.md)  | AWS Trainium1    | Llama3.1 70B pre-train on Wikicorpus |

### [Neuronx Distributed Training](https://github.com/aws-neuron/neuronx-distributed-training)

| Model      | Accelerator | Notes |
| ----------- | ----------- | -------- |
| [Llama 3 70B ](./neuronx-distributed-training/llama3_70b/README.md)    | AWS Trainium1    | | Llama3 70B pre-train on Wikicorpus |

### [RayTrain](https://docs.ray.io/en/latest/train/train.html)


| Model      | Accelerator | Notes |
| ----------- | ----------- | -------- |
| [BERT](./raytrain/lightning-bert/README.md)   | Nvidia GPU  | Fine-tune BERT  using Lightning|






