# PyTorch Lightning Training on Amazon EKS

This directory contains examples for fine-tuning Large Language Models using [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) with PyTorch FSDP (Fully Sharded Data Parallel) on Amazon EKS.

## Overview

The PyTorch Lightning framework is integrated into the [pytorchjob-distributed Helm chart](../../../charts/machine-learning/training/pytorchjob-distributed/) and provides scripts for:

- **Fine-tuning**: Parameter-efficient fine-tuning with LoRA using FSDP
- **Evaluation**: Testing checkpoints with vLLM for efficient inference
- **Conversion**: Converting FSDP checkpoints to HuggingFace format

## Features

- **Lightning Framework**: Built on PyTorch Lightning for structured, scalable training
- **Distributed Training**: Multi-node, multi-GPU training with FSDP for efficient memory usage
- **PEFT Methods**: Support for LoRA parameter-efficient fine-tuning via HuggingFace PEFT
- **Generalized Dataset Support**: Easy integration with any HuggingFace dataset through flexible templates
- **Advanced Callbacks**: Built-in support for early stopping, model checkpointing, and logging
- **Multiple Loggers**: Support for TensorBoard and Weights & Biases logging
- **Gradient Checkpointing**: Reduce memory usage for large models
- **Helm-Based Execution**: Deploy training jobs using Helm charts with YAML configuration files

## Architecture

### Helm Chart Integration

The PyTorch Lightning framework scripts are embedded in the pytorchjob-distributed Helm chart:

```
charts/machine-learning/training/pytorchjob-distributed/
├── templates/
│   └── train.yaml              # PyTorchJob template with framework script mounting
└── scripts/
    └── pytorch_lightning/
        ├── fine_tune.py        # Main training script with Lightning modules
        ├── test_checkpoint.py  # Checkpoint evaluation script
        ├── convert_checkpoint_to_hf.py  # Checkpoint conversion script
        └── dataset_module.py   # Dataset processing module
```

When you specify `framework: 'pytorch_lightning'` in your Helm values file, the chart automatically:

1. Creates a ConfigMap containing all PyTorch Lightning framework scripts
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
│  • Lightning environment variables configured              │
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
│     with Lightning distributed training                    │
│  4. Run post_script (optional cleanup)                     │
└─────────────────────────────────────────────────────────────┘
```

## Helm Values Configuration

### Key Fields

#### Framework Selection

```yaml
framework: 'pytorch_lightning'  # Enables PyTorch Lightning script mounting
```

#### Lightning Environment Configuration

Configure distributed training environment in `pre_script`:

```yaml
pre_script:
  - export MASTER_ADDR="$PET_MASTER_ADDR"
  - export MASTER_PORT="$PET_MASTER_PORT"
  - export NODE_RANK="$PET_NODE_RANK"
  - export WORLD_SIZE="$((PET_NNODES*PET_NPROC_PER_NODE))"
  - mkdir -p $HOME/logs/$PET_NODE_RANK
  - PROCESS_LOG=$HOME/logs/$PET_NODE_RANK/fine_tune.log
  - cd $SCRIPTS_DIR
```

#### Dependency Installation

Install required packages in `pre_script`:

```yaml
pre_script:
  - pip3 install --upgrade pip
  - pip install lightning==2.5.6
      transformers==4.57.1
      datasets==4.4.1
      peft==0.18.0
      accelerate==1.12.0
      tensorboard==2.20.0
      sentencepiece==0.2.1
      torchao==0.14.1
      wandb==0.23.0
      mpi4py==4.1.1
```

#### Training Command

Execute the framework script:

```yaml
train:
  env:
    - name: HOME
      value: "/efs/home/{{ .Release.Name }}"
    - name: MODEL_PATH
      value: "/fsx/pretrained-models/Qwen/Qwen3-14B"
    - name: SCRIPTS_DIR
      value: /etc/framework-scripts
  command:
    - python
  args:
    - fine_tune.py  # Script from /etc/framework-scripts
    - --num_nodes=$PET_NNODES
    - --gpus_per_node=$PET_NPROC_PER_NODE
    - --hf_model_id="Qwen/Qwen3-14B"
    - --model_path=$MODEL_PATH
    - --max_steps=10000
    - --micro_batch_size=1
    - --accumulate_grad_batches=4
```

#### Resource Configuration

```yaml
resources:
  requests:
    "nvidia.com/gpu": 8
    "vpc.amazonaws.com/efa": 4
  limits:
    "nvidia.com/gpu": 8
    "vpc.amazonaws.com/efa": 4
  nnodes: 2
  nproc_per_node: 8
  node_type: 'p4d.24xlarge'
```

#### Storage Configuration

```yaml
pvc:
  - name: efs-pvc
    mount_path: /efs
  - name: fsx-pvc
    mount_path: /fsx

ebs:
  storage: 200Gi
  mount_path: /tmp
```

## Framework Scripts

### fine_tune.py

Main training script built with PyTorch Lightning with support for:

- **Lightning Framework**: Structured training with Lightning modules and trainers
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
| `--num_nodes` | int | Number of nodes | `1` |
| `--gpus_per_node` | int | GPUs per node | `8` |
| `--max_steps` | int | Maximum training steps | `10000` |
| `--micro_batch_size` | int | Batch size per device | `2` |
| `--accumulate_grad_batches` | int | Gradient accumulation | `4` |
| `--max_learning_rate` | float | Maximum learning rate | `1e-5` |
| `--min_learning_rate` | float | Minimum learning rate | `1e-7` |
| `--max_seq_length` | int | Maximum sequence length | `2048` |
| `--lora_rank` | int | LoRA rank | `32` |
| `--lora_alpha` | int | LoRA alpha | `32` |
| `--lora_dropout` | float | LoRA dropout | `0.1` |
| `--val_check_interval` | int | Validation check frequency | `400` |
| `--early_stopping_patience` | int | Early stopping patience | `3` |
| `--use_wandb` | bool | Enable Weights & Biases logging | `False` |

### test_checkpoint.py

Evaluates fine-tuned Lightning checkpoints using vLLM:

- Automatically finds the latest Lightning checkpoint
- Uses vLLM for fast batched inference
- Evaluates predictions using BERTScore
- Supports both LoRA and merged models
- Compatible with Lightning checkpoint format

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--base_model` | str | Base model name | `Qwen/Qwen3-8B` |
| `--model_path` | str | Local model path | None |
| `--max_samples` | int | Max samples to evaluate | `1024` |
| `--batch_size` | int | Inference batch size | `128` |

### convert_checkpoint_to_hf.py

Converts Lightning FSDP checkpoints to HuggingFace format:

- Automatically finds the latest Lightning checkpoint
- Merges LoRA weights into base model by default
- Saves in standard HuggingFace format for deployment
- Optional: Save as LoRA adapter with `--no_merge`
- Handles Lightning checkpoint structure

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--base_model` | str | Base model name | `Qwen/Qwen3-8B` |
| `--model_path` | str | Local model path | None |
| `--no_merge` | flag | Save as LoRA adapter | `False` |

## Dataset Configuration

The PyTorch Lightning framework uses a structured dataset configuration through the `HFDatasetConfig` class:

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

Training outputs are stored on EFS with Lightning-specific structure:

```
/efs/home/{release-name}/
├── logs/
│   └── {node_rank}/
│       ├── fine_tune.log
│       ├── test_checkpoint.log
│       └── convert_checkpoint_to_hf.log
├── results/
│   └── {model_name}/
│       ├── lightning_logs/
│       │   └── version_0/
│       │       ├── checkpoints/
│       │       │   ├── epoch=X-step=Y.ckpt
│       │       │   └── last.ckpt
│       │       └── hparams.yaml
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
cd /efs/home/lightning-qwen3-14b-sft/results
cd /efs/home/lightning-qwen3-14b-sft/logs

# View Lightning checkpoints
cd /efs/home/lightning-qwen3-14b-sft/results/Qwen3-14B/lightning_logs/version_0/checkpoints

# View TensorBoard logs
cd /efs/home/lightning-qwen3-14b-sft/tensorboard_logs
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
    - --micro_batch_size=1
```

**Solution 2: Increase gradient accumulation**
```yaml
train:
  args:
    - --accumulate_grad_batches=16
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

## Examples

- [qwen3-14b-sft](./qwen3-14b-sft/finetune.ipynb): Complete example with Jupyter notebook for fine-tuning Qwen3-14B using PyTorch Lightning

## Additional Resources

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Lightning FSDP Strategy](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)
