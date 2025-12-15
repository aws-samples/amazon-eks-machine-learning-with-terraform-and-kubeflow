# NeMo 2.0 Training on Amazon EKS

This directory contains examples for fine-tuning Large Language Models using [NeMo 2.0 framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) on Amazon EKS.

## Overview

The NeMo 2.0 framework is integrated into the [pytorchjob-distributed Helm chart](../../../charts/machine-learning/training/pytorchjob-distributed/) and provides scripts for:

- **Fine-tuning**: Parameter-efficient fine-tuning with LoRA using NeMo 2.0 recipes
- **Evaluation**: Testing checkpoints with dynamic inference engine for efficient inference
- **Conversion**: Converting NeMo checkpoints to HuggingFace format

## Features

- **NeMo 2.0 Framework**: Built on NeMo 2.0 with structured recipes for scalable training
- **Distributed Training**: Multi-node, multi-GPU training with tensor, pipeline, and context parallelism
- **PEFT Methods**: Support for LoRA parameter-efficient fine-tuning
- **Generalized Dataset Support**: Easy integration with any HuggingFace dataset through flexible templates
- **Advanced Callbacks**: Built-in support for profiling, monitoring, and logging
- **Multiple Loggers**: Support for TensorBoard and Weights & Biases logging
- **Dynamic Inference**: Efficient batched inference with dynamic batching
- **Helm-Based Execution**: Deploy training jobs using Helm charts with YAML configuration files

## Architecture

### Helm Chart Integration

The NeMo 2.0 framework scripts are embedded in the pytorchjob-distributed Helm chart:

```
charts/machine-learning/training/pytorchjob-distributed/
├── templates/
│   └── train.yaml              # PyTorchJob template with framework script mounting
└── scripts/
    └── nemo2/
        ├── fine_tune.py        # Main training script with NeMo 2.0 recipes
        ├── test_checkpoint.py  # Checkpoint evaluation script with dynamic inference
        ├── convert_checkpoint_to_hf.py  # Checkpoint conversion script
        └── dataset_module.py   # Dataset processing module
```

When you specify `framework: 'nemo2'` in your Helm values file, the chart automatically:

1. Creates a ConfigMap containing all NeMo 2.0 framework scripts
2. Mounts the scripts at `/etc/framework-scripts` in the PyTorchJob pods
3. Makes the scripts available for execution within the training environment

### Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Helm Install with Values File (fine-tune.yaml)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  PyTorchJob Created with:                                  │
│  • Framework scripts mounted at /etc/framework-scripts     │
│  • NeMo 2.0 environment variables configured              │
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
│     with NeMo 2.0 distributed training                    │
│  4. Run post_script (optional cleanup)                     │
└─────────────────────────────────────────────────────────────┘
```

## Helm Values Configuration

### Key Fields

#### Framework Selection

```yaml
framework: 'nemo2'  # Enables NeMo 2.0 script mounting
```

## Framework Scripts

### fine_tune.py

Main training script built with NeMo 2.0 with support for:

- **NeMo 2.0 Recipes**: Pre-configured model recipes (qwen3_8b, qwen3_14b, etc.)
- **Models**: Any HuggingFace causal language model with NeMo 2.0 support
- **PEFT**: LoRA fine-tuning with configurable parameters
- **Datasets**: Flexible HuggingFace dataset integration with custom templates
- **Parallelism**: Tensor, pipeline, and context parallelism for efficient scaling
- **Advanced Callbacks**: Profiling, monitoring, and logging callbacks
- **Multiple Loggers**: TensorBoard and Weights & Biases support

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--hf_model_id` | str | HuggingFace model name | `Qwen/Qwen3-8B` |
| `--model_path` | str | Local model path | None |
| `--recipe_cls_name` | str | NeMo 2.0 recipe class name | `qwen3_8b` |
| `--num_nodes` | int | Number of nodes | `1` |
| `--gpus_per_node` | int | GPUs per node | `8` |
| `--tensor_parallel_size` | int | Tensor parallelism size | `8` |
| `--pipeline_parallel_size` | int | Pipeline parallelism size | `1` |
| `--context_parallel_size` | int | Context parallelism size | `1` |
| `--max_steps` | int | Maximum training steps | `10000` |
| `--micro_batch_size` | int | Batch size per device | `8` |
| `--accumulate_grad_batches` | int | Gradient accumulation | `8` |
| `--val_check_interval` | int | Validation check frequency | `800` |
| `--early_stopping_patience` | int | Early stopping patience | `3` |
| `--max_seq_length` | int | Maximum sequence length | `2048` |
| `--peft_scheme` | str | PEFT method | `lora` |
| `--full_ft` | bool | Enable full fine-tuning | `False` |
| `--use_wandb` | bool | Enable Weights & Biases logging | `False` |

### test_checkpoint.py

Evaluates fine-tuned NeMo checkpoints using dynamic inference engine:

- Automatically finds the latest NeMo checkpoint
- Uses dynamic batching for efficient inference
- Evaluates predictions using BERTScore
- Supports both LoRA and merged models
- Compatible with NeMo 2.0 checkpoint format

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--nemo_logs_dir` | str | NeMo logs directory path | None |
| `--max_samples` | int | Max samples to evaluate | `1024` |
| `--max_batch_size` | int | Maximum concurrent requests | `16` |
| `--tensor_parallel_size` | int | Tensor parallelism size | `8` |
| `--temperature` | float | Sampling temperature | `0.1` |
| `--top_p` | float | Top-p sampling | `0.95` |
| `--num_tokens_to_generate` | int | Tokens to generate | `512` |

### convert_checkpoint_to_hf.py

Converts NeMo 2.0 checkpoints to HuggingFace format:

- Automatically finds the latest NeMo checkpoint
- Merges LoRA weights into base model by default
- Saves in standard HuggingFace format for deployment
- Optional: Save as LoRA adapter with `--no_merge`
- Handles NeMo 2.0 checkpoint structure

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--nemo_logs_dir` | str | NeMo logs directory path | None |
| `--no_merge` | flag | Save as LoRA adapter | `False` |
| `--overwrite` | bool | Overwrite existing files | `False` |
| `--use_modelopt` | bool | Use ModelOpt for quantized models | `False` |

## Dataset Configuration

The NeMo 2.0 framework uses a structured dataset configuration through the `HFDatasetConfig` class:

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

Training outputs are stored on EFS with NeMo 2.0-specific structure:

```
/efs/home/{release-name}/
├── logs/
│   └── {node_rank}/
│       ├── fine_tune.log
│       ├── test_checkpoint.log
│       └── convert_checkpoint_to_hf.log
└── outputs/
    └── {timestamp}/
        ├── checkpoints/
        │   └── nemo_logs--{step}/
        │       ├── context/
        │       ├── model/
        │       └── optim/
        ├── nemo_logs--{step}.hf_model/  # Converted HF model
        └── nemo_logs--{step}.hf_peft/   # LoRA adapter (if --no_merge)
```

## Accessing Results

To access training outputs:

```bash
kubectl apply -f eks-cluster/utils/attach-pvc.yaml -n kubeflow
kubectl exec -it -n kubeflow attach-pvc -- /bin/bash

# Navigate to results
cd /efs/home/nemo2-qwen3-14b-sft/outputs
cd /efs/home/nemo2-qwen3-14b-sft/logs

# View NeMo checkpoints
cd /efs/home/nemo2-qwen3-14b-sft/outputs/{timestamp}/checkpoints

# View converted HuggingFace models
cd /efs/home/nemo2-qwen3-14b-sft/outputs/{timestamp}
ls *.hf_model *.hf_peft
```

## Supported Models

The framework supports NeMo 2.0 compatible models with pre-configured recipes:

- **Qwen Family**: Qwen3-8B (`qwen3_8b`), Qwen3-14B (`qwen3_14b`), Qwen3-70B (`qwen3_70b`)
- **Llama Family**: Llama-3-8B (`llama3_8b`), Llama-3-70B (`llama3_70b`)
- **Mistral**: Mistral-7B (`mistral_7b`)
- **Custom Models**: Any HuggingFace model with appropriate recipe configuration

## GPU Requirements

### Small Models (1B - 13B parameters)

**Examples**: Qwen3-8B, Llama3-8B, Mistral-7B

**Configuration**:
- **GPUs**: 8x A100 (40GB or 80GB)
- **Tensor Parallel**: 8
- **Pipeline Parallel**: 1
- **Batch size**: 8 per device
- **Gradient accumulation**: 8

### Medium Models (13B - 34B parameters)

**Examples**: Qwen3-14B, Llama2-13B

**Configuration**:
- **GPUs**: 16x A100 (80GB) total (2 nodes)
- **Tensor Parallel**: 8
- **Pipeline Parallel**: 1 or 2
- **Batch size**: 4-8 per device
- **Gradient accumulation**: 8-16

### Large Models (34B - 100B parameters)

**Examples**: Llama3.1-70B, Qwen3-70B

**Configuration**:
- **GPUs**: 32-64x A100 (80GB) or H100 (80GB)
- **Tensor Parallel**: 8
- **Pipeline Parallel**: 4-8
- **Batch size**: 1-4 per device
- **Gradient accumulation**: 16-32

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1: Reduce batch size**
```yaml
train:
  args:
    - --micro_batch_size=4
```

**Solution 2: Increase gradient accumulation**
```yaml
train:
  args:
    - --accumulate_grad_batches=16
```

**Solution 3: Increase parallelism**
```yaml
train:
  args:
    - --tensor_parallel_size=8
    - --pipeline_parallel_size=2
```

**Solution 4: Reduce sequence length**
```yaml
train:
  args:
    - --max_seq_length=1024
```

### Pod Failures

Check pod logs:
```bash
kubectl logs -n kubeflow-user-example-com pytorchjob-{release-name}-master-0
```

Check PyTorchJob status:
```bash
kubectl get pytorchjob -n kubeflow-user-example-com
kubectl describe pytorchjob pytorchjob-{release-name} -n kubeflow-user-example-com
```

### Storage Issues

Verify PVC mounts:
```bash
kubectl get pvc -n kubeflow-user-example-com
kubectl describe pvc efs-pvc -n kubeflow-user-example-com
```

### NeMo 2.0 Specific Issues

**Recipe not found**:
- Ensure the `recipe_cls_name` matches available NeMo 2.0 recipes
- Check NeMo 2.0 documentation for supported model recipes

**Checkpoint loading errors**:
- Verify checkpoint directory structure has `context/`, `model/`, and `optim/` subdirectories
- Ensure checkpoint was saved with compatible NeMo 2.0 version

## Examples

- [qwen3-14b-sft](./qwen3-14b-sft/finetune.ipynb): Complete example with Jupyter notebook for fine-tuning Qwen3-14B using NeMo 2.0

## Additional Resources

- [NeMo 2.0 Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)