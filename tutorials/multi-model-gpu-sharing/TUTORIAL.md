# Tutorial: deploy several models on the fewest GPUs

A platform team needs to deploy M models on N GPUs. How can it use the fewest GPUs while every model
still meets its service level objectives (SLOs)?

This tutorial answers that question for one example: five models on one `g6e.48xlarge` node, which
has 8 NVIDIA L40S GPUs. It runs each step of the method and shows what to read in its output.

## The method in brief

| step | what it does | GPU |
|---|---|---|
| `plan` | Lists the placements that fit in GPU memory, from each model's size and SLOs. | no |
| demand sweep | Shows how lower request rates would change the GPU count, before you measure anything. | no |
| `profile` | Runs each model alone, to find the fewest GPUs it needs to meet its SLOs. The sum is the **baseline**: the GPU count without sharing. | yes |
| `search` | An agent (a Claude model on Amazon Bedrock) picks placements that share GPUs. Each one is deployed with NVIDIA MPS and measured under load. A placement counts only if every model meets its SLOs. | yes |
| `export` | Writes the best measured placement in a deployable form, beside the baseline. | no |
| deploy on KAI | Serves the exported placement with the KAI scheduler. | yes |

[USER-GUIDE.md](USER-GUIDE.md) describes each step in detail: §8 `plan`, §10 `profile`, §11
`search`, §12 reading the result, §13 `export`, §14 KAI.

The tools and the guide call a step *free* when it uses no GPU. It still runs on billed
infrastructure.

## Two paths

- **CPU-only path.** Runs on a laptop, with no cluster and no GPU. Two scripts stand in for
  `profile` and `search`: they write illustrative results into a separate store, so that `plan`,
  `export` and the KAI render have data to read.
- **GPU path.** Runs the real steps on this repository's cluster, with Claude Code. Steps that lease
  a GPU node are marked **(GPU)**.

## The example fleet

Both paths use [`fleets/examples/g6e.48xlarge/qwen-anchor-riders.yaml`](fleets/examples/g6e.48xlarge/qwen-anchor-riders.yaml).

| tenant | model | use | SLO: mean inter-token latency (ITL) / p95 time to first token (TTFT) |
|---|---|---|---|
| `reason` | Qwen3-30B-A3B | long reasoning answers | 60 ms / 2000 ms |
| `chat` | Qwen3-14B | interactive chat | 50 ms / 1200 ms |
| `route` | Qwen3-0.6B | request routing | 20 ms / 300 ms |
| `guard` | Qwen3-0.6B | safety filter | 80 ms / 500 ms |
| `title` | Qwen3-0.6B | document titling | 150 ms / 3000 ms |

The fleet file also states each tenant's request rate and request size. It does not state how many
GPUs each model gets: the method works that out. The comments in the file explain each SLO. USER-GUIDE
§7 describes the file format.

---

## Path 1: CPU only

You need Python 3.11 or later and a clone of this repository.

```bash
git clone https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow.git
cd amazon-eks-machine-learning-with-terraform-and-kubeflow/tutorials/multi-model-gpu-sharing
pip install ortools pyyaml boto3
```

Run every command from this directory. Set three variables first:

```bash
export PYTHONPATH=src
FLEET=fleets/examples/g6e.48xlarge/qwen-anchor-riders.yaml
SYN=~/.colocation/synthetic
```

`SYN` is the synthetic store. Real results live in `~/.colocation/store`, and the scripts refuse to
write there.

Pass `--local-only` to `plan` and `export` on this path. It stops them from contacting a cluster.

### 1. Run the self-test

```bash
python3 -m colocation.tests.selftest
```

The last line reads `all ... self-tests passed`. No GPU is used. If your kubectl is connected to a
cluster, two checks make a read-only call to it; with no cluster, they pass anyway.

### 2. Plan with an empty store

```bash
mkdir -p /tmp/empty-store
python3 -m colocation.agent plan --fleet $FLEET --store /tmp/empty-store --local-only
```

`plan` refuses. It needs two facts about the hardware: the number of GPUs in a `g6e.48xlarge` and the
memory of one GPU. The tool records these facts the first time it leases a node of that type, so an
empty store has neither. The GPU path meets the same refusal, and its first `profile` fixes it.

### 3. Write illustrative profiles (in place of `profile`)

```bash
python3 tools/synthetic_profile.py --fleet $FLEET --store $SYN
```

The script writes what a real `profile` writes:

- the node's GPU count and the GPU's memory, from the L40S specification;
- one result per model, alone on the fewest GPUs that fit it, with latencies at half of its SLOs.

These numbers are illustrative. Do not pass `$SYN` to a real `profile` or `search`.

### 4. Plan again

```bash
python3 -m colocation.agent plan --fleet $FLEET --store $SYN --local-only
```

The output has three parts:

- **What each SLO requires.** The KV cache each model needs at its request rate and latency SLO.
- **The baseline.** 6 of the 8 GPUs, with each model on its own GPUs. A shared placement must use
  fewer.
- **Feasible placements, densest first.** The densest use 4 GPUs. They fit in memory, but `plan`
  cannot tell whether they meet the SLOs. Only `search` can, by measuring them.

### 5. Sweep the request rates

```bash
mkdir -p /tmp/syn-home/.colocation && ln -sfn $SYN /tmp/syn-home/.colocation/store
HOME=/tmp/syn-home python3 tools/demand_sweep.py --fleet $FLEET
```

The sweep repeats the plan at lower request rates. Its latencies are predictions, marked `~`. They
can disagree with measurements: here the sweep predicts that `chat` needs two GPUs, while the
illustrative profile gives it one. On the GPU path, the measurement decides.

The sweep reads only the default store, so `HOME` points it at the synthetic store.

### 6. Write an illustrative search result (in place of `search`)

```bash
python3 tools/synthetic_search.py --fleet $FLEET --store $SYN
```

The script runs the real search loop, with an in-memory cluster in place of a real one and no agent.
It measures one placement, the solver's first, with latencies at three quarters of each SLO. It then
prints the same report as a real search:

- **BASELINE** 6 GPUs, **STRICT** 4 GPUs: 2 GPUs freed, every SLO met.
- **vs solo**: each model's latency when sharing, against its latency alone. Most ratios are
  withheld, because the model had more KV cache alone than when sharing, so the two runs are not
  comparable.

### 7. Read the record

```bash
python3 -m colocation.inspect record <the path that step 6 printed>
```

The record shows which models shared which GPU, the latency of each model, and the KV cache each one
needed against what it got.

### 8. Export the result

```bash
python3 -m colocation.agent export --fleet $FLEET --store $SYN --local-only
```

The export describes the placement by GPU slot and memory size, not by device ID, so it applies to any
node of the same type. It reads `certified` because every illustrative latency is under its SLO. On
illustrative data, that label means nothing.

### 9. Render the KAI deployment

```bash
python3 -m colocation.kai render $SYN/observations/*/exports/synthetic.yaml --store $SYN -o /tmp/kai-values.yaml
```

This writes the Helm values that would deploy each model on a fraction of a GPU with the KAI
scheduler, and prints how each GPU's memory is divided.

### 10. Try a real search

```bash
python3 -m colocation.agent search --fleet $FLEET --store $SYN
```

`search` refuses at once, because it needs the load harness image and the Claude Agent SDK. The GPU
path sets up both.

---

## Path 2: GPU

### Before you start

Build the cluster with this repository's
[Quick Start (Basic)](https://github.com/aws-samples/amazon-eks-machine-learning-with-terraform-and-kubeflow#quick-start-basic).
It creates an EKS cluster and an EC2 desktop with kubectl, helm, docker, Python and Claude Code
(set up for Amazon Bedrock). The repository is cloned at
`/home/ubuntu/amazon-eks-machine-learning-with-terraform-and-kubeflow`. Run everything below on the
desktop.

You also need:

- **EC2 quota for one `g6e.48xlarge`**: 192 vCPUs of *Running On-Demand G and VT instances* in the
  cluster's region.
- **Amazon Bedrock access to `us.anthropic.claude-opus-5` in `us-west-2`.** The search calls Bedrock
  in `us-west-2`, whatever the cluster's region.
- **A Hugging Face token**, to download model weights.

A leased GPU node is billed until it is released. Step 10 releases it.

### 1. Prepare the cluster

Enable the KAI scheduler, and turn off the CloudWatch add-on's GPU metrics exporter:

```bash
cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup
terraform apply -var-file=basic.tfvars -var kai_enabled=true -var cloudwatch_dcgm_exporter_enabled=false
```

Each run deploys its own DCGM exporter on the GPU node it leases, and a GPU allows only one exporter
to read its profiling metrics (USER-GUIDE §0). Turning off the add-on's exporter removes the Container
Insights GPU dashboards and nothing else. Check that it is gone:

```bash
kubectl get dcgmexporters -A
kubectl get ds -A | grep -i dcgm
```

Both commands should list nothing. If an exporter remains, the run refuses before it leases a node,
and prints the `aws eks update-addon` command that removes it.

The tools reach the shared store on EFS through a pod named `attach-pvc` in the
`kubeflow-user-example-com` namespace. Start it:

```bash
cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
kubectl apply -f eks-cluster/utils/attach-pvc.yaml -n kubeflow-user-example-com
kubectl wait --for=condition=ready pod/attach-pvc -n kubeflow-user-example-com --timeout=300s
```

### 2. Install the dependencies

```bash
cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
git pull
cd tutorials/multi-model-gpu-sharing
pip install ortools pyyaml boto3 claude-agent-sdk
PYTHONPATH=src python3 -m colocation.tests.selftest
```

### 3. Install the Claude Code plugin

Start Claude Code in this directory:

```bash
claude
```

Then, in the session:

```
/plugin marketplace add ./plugins
/plugin install colocation-pipeline@colocation-plugins
```

The plugin has two skills, one command per step, and hooks that block commands that would break a
running search. `run-fleet` runs the steps in order. `kai-deploy` deploys a result, only when you ask.
USER-GUIDE §15 describes the plugin.

### 4. Check the cluster

```
/fleet-preflight
```

This changes nothing. It checks the DCGM exporters, the GPU node holder, the harness image, and the
cluster settings in the harness Job template.

### 5. Build the load harness image

```
/fleet-image
```

This builds the harness, runs its self-test inside the image, and pushes the image to Amazon ECR in
your account.

### 6. Download the model weights

The models read their weights from FSx, at `/fsx/pretrained-models/<org>/<model>`. Store your Hugging
Face token as a secret, then download the three models with the repository's `hf-snapshot` chart:

```bash
NS=kubeflow-user-example-com
kubectl create secret generic hf-token -n $NS --from-file=token=$HOME/.hf_token \
  --dry-run=client -o yaml | kubectl apply -f -
cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
for m in Qwen/Qwen3-30B-A3B-Base Qwen/Qwen3-14B Qwen/Qwen3-0.6B; do
  r=weights-$(echo ${m#*/} | tr 'A-Z.' 'a-z-')
  helm install $r charts/machine-learning/model-prep/hf-snapshot \
    --set hf_model_id=$m --set hf_secret_name=hf-token -n $NS
done
```

When each job completes, uninstall its release with `helm uninstall <release> -n $NS`. Check the
weights:

```bash
kubectl exec -n $NS attach-pvc -- ls /fsx/pretrained-models/Qwen
```

USER-GUIDE §4 has more on staging weights.

### 7. Run the fleet (GPU)

In Claude Code, ask:

```
> Run fleets/examples/g6e.48xlarge/qwen-anchor-riders.yaml through the pipeline.
```

The `run-fleet` skill runs `plan`, the demand sweep, `profile`, `plan` again, `search` and `export`.
It asks you before each step that leases a GPU node. The first `profile` on a new instance type only
records the node's GPU count and memory, then stops; the skill runs it again to measure.

### 8. Read the result

The search output lists each placement the agent measured, the agent's reason for choosing it, and
the measured latencies. It ends with the BASELINE, STRICT and RELAXED placements (USER-GUIDE §12).
To check one measurement in full:

```
/fleet-inspect <record id>
```

The export is in `~/.colocation/store/observations/<key>/exports/`.

### 9. Deploy on KAI (GPU)

Ask Claude Code to deploy the export, or run `/fleet-kai-deploy <export or run token>`. The
`kai-deploy` skill deploys each model on a fraction of a GPU on the `kai-gpu` node pool, checks that
each model landed where the export says, and retries if one did not.

Then `/fleet-kai-load` drives the deployment with a load-test scenario from `load-tests/examples/`,
and `/fleet-kai-drift` compares each model's measured latency with its SLOs.

If the search measured a model with an SM cap (a limit on its share of GPU compute; USER-GUIDE §11),
KAI deploys it with the same cap. This is experimental.

### 10. Clean up

Remove the KAI deployment:

```
/fleet-kai-teardown <deployment name>
```

Ask Claude Code to tear down the run and release the GPU node. The skill removes the models and the
MPS daemon, and hands the node to `gpu-node-holder`, a pod that keeps the node between runs. **The
node stays up, and billed, while that pod exists.** Delete it from your own terminal; the plugin's
hooks block Claude Code from deleting it, because during a run that would lose the node.

```bash
NS=kubeflow-user-example-com
helm list -n $NS                                   # no mps-... or model releases should remain
kubectl delete pod gpu-node-holder -n $NS --ignore-not-found
kubectl get nodes -l karpenter.sh/nodepool=cuda
kubectl get nodes -l karpenter.sh/nodepool=kai-gpu
```

Karpenter removes empty nodes. Repeat the last two commands until both list nothing.

### Going further

- **A fleet that needs two nodes.** [`three-teams.yaml`](fleets/examples/g6e.48xlarge/three-teams.yaml)
  has 15 models. With `hardware.max_nodes: 2`, `search` splits them across two nodes (USER-GUIDE §11).
- **Other serving engines.** The `engine:` field in a fleet file serves models on Triton Inference
  Server or NVIDIA Dynamo instead of vLLM (USER-GUIDE §7).
- **Your own fleet.** Write a fleet file at the top of `fleets/` (USER-GUIDE §7).

### Using another cluster

The CPU-only path runs anywhere. The GPU path expects this repository's cluster by name: Karpenter
NodePools named `cuda` and `kai-gpu`, a tainted `cpu-compute` pool, the `attach-pvc` pod, shared
`/efs` and `/fsx` volumes, and, for KAI, the settings in `kai.tf`. USER-GUIDE §0 and §1 list these
requirements. The GPU path has not been tested on other clusters.
