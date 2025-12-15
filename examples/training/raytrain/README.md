# Ray Train Training on Amazon EKS

This directory contains examples for fine-tuning Large Language Models using [Ray Train](https://docs.ray.io/en/latest/train/train.html) with PyTorch FSDP (Fully Sharded Data Parallel) on Amazon EKS.

## Overview

The Ray Train framework is integrated into the [raytrain Helm chart](../../../charts/machine-learning/training/raytrain/) and provides scripts for:

- **Fine-tuning**: Parameter-efficient fine-tuning with LoRA using FSDP
- **Evaluation**: Testing checkpoints with vLLM for efficient inference
- **Conversion**: Converting FSDP checkpoints to HuggingFace format

## Features

- **Ray Train Framework**: Built on Ray Train for distributed, scalable training
- **Distributed Training**: Multi-node, multi-GPU training with FSDP for efficient memory usage
- **PEFT Methods**: Support for LoRA parameter-efficient fine-tuning via HuggingFace PEFT
- **Generalized Dataset Support**: Easy integration with any HuggingFace dataset through flexible templates
- **Advanced Callbacks**: Built-in support for early stopping, model checkpointing, and logging
- **Multiple Loggers**: Support for TensorBoard and Weights & Biases logging
- **Gradient Checkpointing**: Reduce memory usage for large models
- **Helm-Based Execution**: Deploy training jobs using Helm charts with YAML configuration files

## Architecture

### Helm Chart Integration

The Ray Train framework scripts are embedded in the raytrain Helm chart:

```
charts/machine-learning/training/raytrain/
├── templates/
│   └── train.yaml              # RayJob template with framework script mounting
└── scripts/
    └── ray_train/
        ├── fine_tune.py        # Main training script with Ray Train
        ├── test_checkpoint.py  # Checkpoint evaluation script
        ├── convert_checkpoint_to_hf.py  # Checkpoint conversion script
        └── dataset_module.py   # Dataset processing module
```

When you specify `framework: 'ray_train'` in your Helm values file, the chart automatically:

1. Creates a ConfigMap containing all Ray Train framework scripts
2. Mounts the scripts at `/etc/framework-scripts` in the RayJob pods
3. Makes the scripts available for execution within the training environment

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Helm Install with Values File (fine-tune.yaml)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  RayJob Created with:                                      │
│  • Framework scripts mounted at /etc/framework-scripts     │
│  • Ray Train environment variables configured              │
│  • Dependencies installed in pre_script                    │
│  • Training command executed via train.command/args       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Training Execution:                                        │
│  1. Git clone (if specified)                               │
│  2. Run pre_script (install dependencies, set env vars)    │
│  3. Execute: python /etc/framework-scripts/fine_tune.py    │
│     with Ray Train distributed training                    │
│  4. Run post_script (optional cleanup)                     │
└─────────────────────────────────────────────────────────────┘
```

## Helm Values Configuration

### Key Fields

#### Framework Selection

```yaml
framework: 'ray_train'  # Enables Ray Train script mounting
```

## Framework Scripts

### fine_tune.py

Main training script built with Ray Train with support for:

- **Ray Train Framework**: Distributed training with Ray Train for scalability
- **Models**: Any HuggingFace causal language model (Qwen, Llama, Mistral, etc.)
- **PEFT**: LoRA fine-tuning with configurable rank, alpha, dropout
- **Datasets**: Flexible HuggingFace dataset integration with custom templates
- **FSDP**: Fully sharded data parallel for memory efficiency
- **Advanced Callbacks**: Early stopping, model checkpointing, and logging
- **Multiple Loggers**: TensorBoard and Weights & Biases support

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--hf_model_id` | str | HuggingFace model name | `Qwen/Qwen3-8B` |
| `--model_path` | str | Local model path | None |
| `--max_steps` | int | Maximum training steps | `10000` |
| `--per_device_train_batch_size` | int | Batch size per device | `2` |
| `--gradient_accumulation_steps` | int | Gradient accumulation | `4` |
| `--learning_rate` | float | Learning rate | `5e-5` |
| `--min_learning_rate` | float | Minimum learning rate | `1e-6` |
| `--max_seq_length` | int | Maximum sequence length | `2048` |
| `--lora_rank` | int | LoRA rank | `32` |
| `--lora_alpha` | int | LoRA alpha | `32` |
| `--lora_dropout` | float | LoRA dropout | `0.1` |
| `--eval_steps` | int | Evaluation frequency | `100` |
| `--early_stopping_patience` | int | Early stopping patience | `3` |
| `--use_wandb` | bool | Enable Weights & Biases logging | `False` |

### test_checkpoint.py

Evaluates fine-tuned Ray Train checkpoints using vLLM:

- Automatically finds the latest Ray Train checkpoint
- Uses vLLM for fast batched inference
- Evaluates predictions using BERTScore
- Supports both LoRA and merged models
- Compatible with Ray Train checkpoint format

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--base_model` | str | Base model name | `Qwen/Qwen3-8B` |
| `--model_path` | str | Local model path | None |
| `--max_samples` | int | Max samples to evaluate | `1024` |
| `--batch_size` | int | Inference batch size | `128` |

### convert_checkpoint_to_hf.py

Converts Ray Train FSDP checkpoints to HuggingFace format:

- Automatically finds the latest Ray Train checkpoint
- Merges LoRA weights into base model by default
- Saves in standard HuggingFace format for deployment
- Optional: Save as LoRA adapter with `--no_merge`
- Handles Ray Train checkpoint structure

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--base_model` | str | Base model name | `Qwen/Qwen3-8B` |
| `--model_path` | str | Local model path | None |
| `--no_merge` | flag | Save as LoRA adapter | `False` |

## Dataset Configuration

The Ray Train framework uses a structured dataset configuration through the `HFDatasetConfig` class:

### Dataset Configuration Fields

The dataset configuration is embedded in the script's `Config` class:

```python
hf_dataset_config: HFDatasetConfig = field(default_factory=lambda: HFDatasetConfig(
    dataset_name="cognitivecomputations/dolphin",
    dataset_config='flan1m-alpaca-uncensored',
    split="train",
    train_split_ratio=0.9,
    val_test_split_ratio=0.5,
    input_template="### Instruction:\n{instruction}\n ### Input:\n{input}\n",
    output_template="### Response:\n{output}",
    field_mapping={
        "instruction": "instruction",
        "input": "input", 
        "output": "output"
    },
    num_proc=8
))
```

### Customizing Dataset via CLI Arguments

You can override dataset configuration through command-line arguments by modifying the script or using environment variables.

## Output Structure

Training outputs are stored on EFS with Ray Train-specific structure:

```
/efs/home/{release-name}/
├── logs/
│   ├── fine_tune.log
│   ├── test_checkpoint.log
│   └── convert_checkpoint_to_hf.log
├── results/
│   └── {model_name}/
│       ├── ray_results/
│       │   └── TorchTrainer_*/
│       │       ├── checkpoint_*/
│       │       └── params.json
│       ├── final/
│       └── hf_format/
└── tensorboard_logs/
    └── {model_name}/
```

## Accessing Results

To access training outputs:

```bash
kubectl apply -f eks-cluster/utils/attach-pvc.yaml -n kubeflow
kubectl exec -it -n kubeflow attach-pvc -- /bin/bash

# Navigate to results
cd /efs/home/raytrain-qwen3-14b-sft/results
cd /efs/home/raytrain-qwen3-14b-sft/logs

# View Ray Train checkpoints
cd /efs/home/raytrain-qwen3-14b-sft/results/Qwen3-14B/ray_results

# View TensorBoard logs
cd /efs/home/raytrain-qwen3-14b-sft/tensorboard_logs
```

## Supported Models

The framework supports any HuggingFace causal language model:

- **Qwen Family**: Qwen3-8B, Qwen3-14B, Qwen3-70B
- **Llama Family**: Llama-3-8B, Llama-3-70B, Meta-Llama-3.1-8B, Meta-Llama-3.1-70B
- **Mistral**: Mistral-7B-v0.1, Mixtral-8x7B-v0.1
- **Phi**: Phi-3-medium-4k-instruct

## GPU Requirements

### Small Models (1B - 13B parameters)

**Examples**: Qwen3-8B, Llama3-8B, Mistral-7B

**Configuration**:
- **GPUs**: 8x A100 (40GB or 80GB)
- **Batch size**: 2-4 per device
- **Gradient accumulation**: 4-8

### Medium Models (13B - 34B parameters)

**Examples**: Qwen3-14B, Llama2-13B, Yi-34B

**Configuration**:
- **GPUs**: 16x A100 (80GB) total (2 nodes)
- **Batch size**: 1-2 per device
- **Gradient accumulation**: 8-16

### Large Models (34B - 100B parameters)

**Examples**: Llama3.1-70B, Mixtral-8x22B

**Configuration**:
- **GPUs**: 32-64x A100 (80GB) or H100 (80GB)
- **Batch size**: 1 per device
- **Gradient accumulation**: 16-32
- **CPU Offload**: Consider enabling

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1: Reduce batch size**
```yaml
train:
  args:
    - --per_device_train_batch_size=1
```

**Solution 2: Increase gradient accumulation**
```yaml
train:
  args:
    - --gradient_accumulation_steps=16
```

**Solution 3: Reduce sequence length**
```yaml
train:
  args:
    - --max_seq_length=1024
```

**Solution 4: Enable CPU offload**
```yaml
train:
  args:
    - --cpu_offload=true
```

### Pod Failures

Check pod logs:
```bash
kubectl logs -n kubeflow-user-example-com rayjob-{release-name}-raycluster-head-0
```

Check RayJob status:
```bash
kubectl get rayjob -n kubeflow-user-example-com
kubectl describe rayjob rayjob-{release-name} -n kubeflow-user-example-com
```

### Storage Issues

Verify PVC mounts:
```bash
kubectl get pvc -n kubeflow-user-example-com
kubectl describe pvc pv-efs -n kubeflow-user-example-com
```

## Examples

- [qwen3-14b-sft](./qwen3-14b-sft/finetune.ipynb): Complete example with Jupyter notebook for fine-tuning Qwen3-14B using Ray Train

## Additional Resources

- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Ray Train with Transformers](https://docs.ray.io/en/latest/train/examples/transformers/huggingface_text_classification.html)
