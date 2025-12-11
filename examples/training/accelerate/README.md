# Hugging Face Accelerate Training on Amazon EKS

This directory contains examples for fine-tuning Large Language Models using [Hugging Face Accelerate](https://github.com/huggingface/accelerate) with PyTorch FSDP (Fully Sharded Data Parallel) on Amazon EKS.

## Overview

The accelerate framework is integrated into the [pytorchjob-distributed Helm chart](../../../charts/machine-learning/training/pytorchjob-distributed/) and provides scripts for:

- **Fine-tuning**: Parameter-efficient fine-tuning with LoRA using FSDP
- **Evaluation**: Testing checkpoints with vLLM for efficient inference
- **Conversion**: Converting FSDP checkpoints to HuggingFace format

## Features

- **Distributed Training**: Multi-node, multi-GPU training with FSDP for efficient memory usage
- **PEFT Methods**: Support for LoRA parameter-efficient fine-tuning via HuggingFace PEFT
- **Generalized Dataset Support**: Easy integration with any HuggingFace dataset through flexible templates
- **Flash Attention 2**: Optimized attention implementation for faster training
- **Gradient Checkpointing**: Reduce memory usage for large models
- **Helm-Based Execution**: Deploy training jobs using Helm charts with YAML configuration files

## Architecture

### Helm Chart Integration

The accelerate framework scripts are embedded in the pytorchjob-distributed Helm chart:

```
charts/machine-learning/training/pytorchjob-distributed/
├── templates/
│   └── train.yaml              # PyTorchJob template with framework script mounting
└── scripts/
    └── accelerate/
        ├── fine_tune.py        # Main training script
        ├── test_checkpoint.py  # Checkpoint evaluation script
        ├── convert_checkpoint_to_hf.py  # Checkpoint conversion script
        └── dataset_module.py   # Dataset processing module
```

When you specify `framework: 'accelerate'` in your Helm values file, the chart automatically:

1. Creates a ConfigMap containing all accelerate framework scripts
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
│  • Accelerate config generated in inline_script            │
│  • Dependencies installed in pre_script                    │
│  • Training command executed via train.command/args       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Training Execution:                                        │
│  1. Git clone (if specified)                               │
│  2. Run inline_script (generate configs)                   │
│  3. Run pre_script (install dependencies)                  │
│  4. Execute: accelerate launch --config_file ...           │
│     /etc/framework-scripts/fine_tune.py                    │
│  5. Run post_script (optional cleanup)                     │
└─────────────────────────────────────────────────────────────┘
```

## Helm Values Configuration

### Key Fields

#### Framework Selection

```yaml
framework: 'accelerate'  # Enables accelerate script mounting
```

#### Accelerate Configuration

Generate the accelerate config in `inline_script`:

```yaml
inline_script:
  - |+
    cat > /tmp/accel_config.yaml <<EOF
    compute_environment: LOCAL_MACHINE
    distributed_type: FSDP
    mixed_precision: bf16
    num_machines: $PET_NNODES
    num_processes: $((PET_NPROC_PER_NODE * PET_NNODES))
    machine_rank: $PET_NODE_RANK
    main_process_ip: $PET_MASTER_ADDR
    main_process_port: $PET_MASTER_PORT
    fsdp_config:
      fsdp_sharding_strategy: FULL_SHARD
      fsdp_state_dict_type: SHARDED_STATE_DICT
      fsdp_cpu_ram_efficient_loading: true
    EOF
```

#### Dependency Installation

Install required packages in `pre_script`:

```yaml
pre_script:
  - pip3 install --upgrade pip
  - pip3 install transformers==4.57.1 datasets==4.4.1 peft==0.18.0 
      accelerate==1.12.0 tensorboard==2.20.0
  - cd $SCRIPTS_DIR  # Navigate to /etc/framework-scripts
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
    - accelerate
  args:
    - launch
    - --config_file
    - /tmp/accel_config.yaml
    - fine_tune.py  # Script from /etc/framework-scripts
    - --hf_model_id="Qwen/Qwen3-14B"
    - --model_path=$MODEL_PATH
    - --max_steps=10000
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

Main training script with support for:

- **Models**: Any HuggingFace causal language model (Qwen, Llama, Mistral, etc.)
- **PEFT**: LoRA fine-tuning with configurable rank, alpha, dropout
- **Datasets**: Flexible HuggingFace dataset integration with custom templates
- **FSDP**: Fully sharded data parallel for memory efficiency
- **Checkpointing**: Automatic checkpoint saving and resumption

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--hf_model_id` | str | HuggingFace model name | `Qwen/Qwen3-8B` |
| `--model_path` | str | Local model path | None |
| `--max_steps` | int | Maximum training steps | `10000` |
| `--per_device_train_batch_size` | int | Batch size per device | `2` |
| `--gradient_accumulation_steps` | int | Gradient accumulation | `4` |
| `--learning_rate` | float | Learning rate | `5e-5` |
| `--max_seq_length` | int | Maximum sequence length | `2048` |
| `--lora_rank` | int | LoRA rank | `32` |
| `--lora_alpha` | int | LoRA alpha | `32` |
| `--lora_dropout` | float | LoRA dropout | `0.1` |
| `--output_dir` | str | Output directory | `results/{hf_model_id}` |
| `--eval_steps` | int | Evaluation frequency | `100` |
| `--early_stopping_patience` | int | Early stopping patience | `3` |

### test_checkpoint.py

Evaluates fine-tuned checkpoints using vLLM:

- Automatically finds the latest checkpoint
- Uses vLLM for fast batched inference
- Evaluates predictions using BERTScore
- Supports both LoRA and merged models

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--base_model` | str | Base model name | `Qwen/Qwen3-8B` |
| `--model_path` | str | Local model path | None |
| `--max_samples` | int | Max samples to evaluate | `1024` |
| `--batch_size` | int | Inference batch size | `128` |

### convert_checkpoint_to_hf.py

Converts FSDP checkpoints to HuggingFace format:

- Automatically finds the latest checkpoint
- Merges LoRA weights into base model by default
- Saves in standard HuggingFace format for deployment
- Optional: Save as LoRA adapter with `--no_merge`

#### Key Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--base_model` | str | Base model name | `Qwen/Qwen3-8B` |
| `--model_path` | str | Local model path | None |
| `--no_merge` | flag | Save as LoRA adapter | `False` |

## Dataset Configuration

The framework uses flexible dataset configuration through CLI arguments:

### HuggingFace Dataset Arguments

| Argument | Description |
|----------|-------------|
| `--hfdc_dataset_name` | HuggingFace dataset name |
| `--hfdc_dataset_config` | Dataset configuration/subset |
| `--hfdc_split` | Initial split to load |
| `--hfdc_train_split_ratio` | Training data ratio |
| `--hfdc_val_test_split_ratio` | Val/test split ratio |
| `--hfdc_input_template` | Input formatting template |
| `--hfdc_output_template` | Output formatting template |
| `--hfdc_field_mapping` | Field name mapping (JSON) |

### Example: Custom Dataset

```yaml
train:
  args:
    - --hfdc_dataset_name="databricks/databricks-dolly-15k"
    - --hfdc_input_template="### Instruction:\n{instruction}\n### Context:\n{context}\n"
    - --hfdc_output_template="### Response:\n{response}"
    - --hfdc_field_mapping='{"instruction":"instruction","context":"context","response":"response"}'
```

## Output Structure

Training outputs are stored on EFS:

```
/efs/home/{release-name}/
├── logs/
│   └── {node_rank}/
│       ├── fine_tune.log
│       ├── test_checkpoint.log
│       └── convert_checkpoint_to_hf.log
└── results/
    └── {model_name}/
        ├── checkpoint-{step}/
        │   ├── model.safetensors
        │   ├── adapter_config.json
        │   └── adapter_model.safetensors
        ├── final/
        └── hf_format/
```

## Accessing Results

To access training outputs:

```bash
kubectl apply -f eks-cluster/utils/attach-pvc.yaml -n kubeflow
kubectl exec -it -n kubeflow attach-pvc -- /bin/bash

# Navigate to results
cd /efs/home/accel-qwen3-14b-sft/results
cd /efs/home/accel-qwen3-14b-sft/logs
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
inline_script:
  - |+
    fsdp_config:
      fsdp_offload_params: true
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

- [qwen3-14b-sft](./qwen3-14b-sft/finetune.ipynb): Complete example with Jupyter notebook for fine-tuning Qwen3-14B

## Additional Resources

- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [PEFT Documentation](https://huggingface.co/docs/peft)
