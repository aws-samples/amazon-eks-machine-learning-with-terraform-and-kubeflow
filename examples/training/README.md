## Training Examples


### [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/en/index)

| Model      | Accelerator | Notes |
| ----------- | ----------- | -------- |
| [Bert GLUE MRPC Pre-training](./accelerate/bert-glue-mrpc/README.md)    | Nvidia GPU  | BERT Glue MRPC Pretraining    |
| [Llama 2 7B Fine-tuning](./accelerate/llama2-ft-fsdp/README.md)    | Nvidia GPU | Llama 2 7B Fine-tuning with FSDP on smangrul/code-chat-assistant-v1 |
| [Llama 2 13B Fine-tuning](./accelerate/llama2-ft-fsdp/README.md)    | Nvidia GPU  | Llama 2 13B Fine-tuning with FSDP on smangrul/code-chat-assistant-v1|
| [Llama 2 70B Fine-tuning](./accelerate/llama2-ft-fsdp/README.md)   | Nvidia GPU   | Llama 2 70B Fine-tuning with FSDP on smangrul/code-chat-assistant-v1|

### [Nemo](https://github.com/NVIDIA/NeMo)

| Model      | Accelerator | Notes |
| ----------- | ----------- | -------- |
| [Llama 2 7B PEFT LoRA](./nemo-megatron/llama2-7b-peft/README.md)  | Nvidia GPU     | Llama 2 7B PEFT LoRA on Pubmedqa dataset |
| [Llama 3.1 8B PEFT LoRA](./nemo-megatron/llama31-8b-peft-dolphin/README.md) | Nvidia GPU       | Llama 3.1 8B PEFT LoRA on Dolphin dataset |
| [Mistral 7B v0.1 PEFT LoRA](./nemo-megatron/mistral-7b-v01-peft/README.md)   | Nvidia GPU     | Mistral 7B v0.1 PEFT LoRA on Pubmedqa dataset |
| [Mistral 7B v0.1  PEFT LoRA](./nemo-megatron/mistral-7b-v01-peft-dolphin/README.md) | Nvidia GPU       | Mistral 7B v0.1 PEFT LoRA  PEFT LoRA on Dolphin dataset |


### [Neuronx Distributed Training](https://github.com/aws-neuron/neuronx-distributed-training)

| Model      | Accelerator | Notes |
| ----------- | ----------- | -------- |
| [Llama 3 70B ](./neuronx-distributed-training/llama3_70b/README.md)    | AWS Trainium1    |  Llama3 70B pre-train on Wikicorpus |

### [RayTrain](https://docs.ray.io/en/latest/train/train.html)


| Model      | Accelerator | Notes |
| ----------- | ----------- | -------- |
| [BERT](./raytrain/lightning-bert/README.md)   | Nvidia GPU  | Fine-tune BERT  using Lightning|






