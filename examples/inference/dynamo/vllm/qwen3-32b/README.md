# Qwen3-32B multi-node inference with NVIDIA Dynamo and vLLM

## What this is

[Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) served as a **single vLLM engine whose
parallelism spans two GPU nodes**, with the single-node deployment alongside it as a baseline.
[`serve.ipynb`](serve.ipynb) runs every configuration end to end.

| Values file | Nodes | GPUs | Parallelism | Crosses the node boundary |
| --- | --- | --- | --- | --- |
| [`dgd-singlenode-tp4.yaml`](dgd-singlenode-tp4.yaml) | 1 | 4 | TP=4 | no — baseline |
| [`dgd-multinode-tp8.yaml`](dgd-multinode-tp8.yaml) | 2 | 8 | TP=8 | every layer, over EFA |
| [`dgd-multinode-tp8-tcp.yaml`](dgd-multinode-tp8-tcp.yaml) | 2 | 8 | TP=8 | every layer, over ENA/TCP |
| [`dgd-multinode-tp4-pp2.yaml`](dgd-multinode-tp4-pp2.yaml) | 2 | 8 | TP=4, PP=2 | once per token, over EFA |
| [`dgd-router-kv-2x.yaml`](dgd-router-kv-2x.yaml) | 2 | 8 | TP=4 × 2 replicas | no — two independent engines |

Multi-node and disaggregated serving are independent choices. The [Qwen3-8B
example](../qwen3-8b/) splits prefill from decode across pods; this one widens a single engine
past one machine. A component can be either, both, or neither.

## Why

**Qwen3-32B does not need two nodes** — ~65 GB in bf16 against 192 GB of VRAM on one
`g6e.12xlarge` (4 × L40S 48 GB). That is deliberate. Because the model fits, the single-node arm
is a usable control and the two-node arms can be priced against it. Expect single-node to win.
For a model that genuinely cannot be served on one node, and therefore has no control to compare
against, see the [Qwen3-235B-A22B-FP8 example](../qwen3-235b-a22b-fp8/).

Two deployment decisions come out of the comparison:

**Where to cut the model.** At TP=8 every transformer layer ends in an all-reduce across all eight
ranks, so a 64-layer model crosses the node boundary ~128 times per forward pass, each crossing
roughly `hidden_size × 2 B` = 10 KB and therefore latency-bound rather than bandwidth-bound — the
least favourable traffic shape for an Ethernet-class fabric, even at 100 Gbps. At TP=4 × PP=2 the
tensor-parallel groups stay inside a node and the boundary is crossed once per token, passing one
activation tensor between pipeline stages. "TP within a node, PP across nodes" is the standard
guidance. `dgd-multinode-tp8.yaml` is included because it is the configuration people reach for
first, and measuring what it costs is more useful than asserting that it is wrong. PP's own cost
is pipeline bubbles: stages idle unless enough requests are in flight.

**One wide engine or several narrow ones.** `node_count: 2` is one engine occupying two nodes —
one KV cache sharded across 8 GPUs, nothing to route between. `replicas: 2` is the opposite
arrangement: two complete engines with independent caches and a routing decision on every
request. The two knobs look alike in a values file and have almost nothing in common.
`dgd-router-kv-2x.yaml` is the second shape, and the only one where **KV-aware routing** does
anything — the frontend scoring each worker on how much of the prompt's prefix it already holds
and sending the request where prefill can be skipped. With `replicas: 1` a router has one
candidate and is a no-op in every mode.

## How Dynamo spans two nodes

**You do not write a LeaderWorkerSet.** `node_count: 2` is the entire multi-node input:

```yaml
VllmWorker:
  type: worker
  node_count: 2
```

The [chart](../../../../../charts/machine-learning/serving/dynamo/) renders that to
`spec.components[].multinode.nodeCount`, and the operator creates one `LeaderWorkerSet` per
multi-node component: `size: nodeCount` (1 leader + `nodeCount - 1` workers),
`restartPolicy: RecreateGroupOnPodRestart`, and separate leader and worker pod templates. LWS then
injects into every pod in the group:

| Variable | Meaning |
| --- | --- |
| `LWS_LEADER_ADDRESS` | stable DNS name of the leader, backed by a headless service |
| `LWS_WORKER_INDEX` | this pod's rank, 0 on the leader |
| `LWS_GROUP_SIZE` | `nodeCount` |

Stable identity is the whole reason an LWS is used rather than a Deployment: rank 0 has to be
findable by name before any process has started.

**vLLM spans the nodes with its native multiprocessing backend, not Ray.** When
`--tensor-parallel-size × --pipeline-parallel-size` exceeds the container's GPU request, the
operator rewrites the command:

    leader:  python3 -m dynamo.vllm ... --distributed-executor-backend mp \
               --nnodes 2 --master-addr $(LWS_LEADER_ADDRESS) --master-port 29500 --node-rank 0
    worker:  python3 -m dynamo.vllm ... --distributed-executor-backend mp \
               --nnodes 2 --master-addr $(LWS_LEADER_ADDRESS) --master-port 29500 \
               --node-rank $(LWS_WORKER_INDEX) --headless

So the cross-node transport is **torch.distributed over port 29500, with NCCL carrying the
collectives** — and NCCL is what uses EFA. There is no Ray, etcd, NATS or MPI on this path.
Whether the operator picks `mp` or Ray is a version gate rather than a documented default, so
every values file here says which it means:

```yaml
annotations:
  nvidia.com/vllm-distributed-executor-backend: "mp"
```

Three further operator behaviours, all of which matter while debugging:

- **Worker pods lose their probes.** Only the leader serves HTTP; a worker is a rank in a process
  group and has nothing to answer a health check with.
- **Each worker gets a `wait-for-leader-mp` init container** that polls the leader's pod status
  and then TCP-connects to port 29500. It has **no timeout** — which is what lets a group wait out
  Karpenter provisioning the second node, and also what makes a genuinely stuck group sit in
  `Init:0/1` indefinitely instead of crash-looping. Read that container's log first.
- **`VLLM_NIXL_SIDE_CHANNEL_HOST` is set from `status.podIP`**, so NIXL advertises a routable
  address.

`spec.components[].roles` is omitted everywhere here. The CRD derives the implicit leader/worker
layout from `multinode.nodeCount`, and `roles.providerOverride` requires Grove, which is not
installed.

## Prerequisites

`dynamo_enabled = true` installs both the [Dynamo operator](../../README.md#what-gets-installed)
and Volcano. The rest is EFA. Each item below fails in its own quiet way, so check them before
deploying rather than after.

### 1. Volcano, installed with `dynamo_enabled`

The operator decides **at startup** whether its LeaderWorkerSet path is available by probing for
two API groups — `leaderworkerset.x-k8s.io` and `scheduling.volcano.sh`. LWS alone is not enough:
with `Orchestrators.LWS.Enabled` unset the gate resolves to `lwsAvailable && volcanoAvailable`, and
when it resolves false the operator starts up perfectly normally. A `DynamoGraphDeployment` with
`multinode.nodeCount` then goes to:

    $ kubectl get dgd dyn-qwen3-32b -o jsonpath='{.status.conditions}'
    Ready  False  no_multinode_orchestrator_available  "no multinode orchestrator available"

No pods, no LeaderWorkerSet, nothing logged at default verbosity, and single-node deployments on
the same cluster keep working — which is why this gets discovered late.

Because the probe runs at operator startup, **installing Volcano afterwards is not enough**; the
operator pod has to restart. Terraform encodes that by listing `helm_release.volcano` in the
operator's `depends_on`.

Note what this does not buy: Volcano is present so its API group exists. Nothing here sets
`schedulerName: volcano`, so pods are still placed by kube-scheduler. Real gang scheduling is a
follow-on change, not something a two-node deployment needs.

### 2. The `cudaefa` NodePool must admit your instance type

The pool carries a floor of `karpenter.k8s.aws/instance-network-bandwidth Gt 99999`, which keeps
low-bandwidth instances out of a pool meant for collective communication:

| Size | GPUs | Network | EFA interfaces | In `cudaefa`? |
| --- | --- | --- | --- | --- |
| `g6e.8xlarge` | 1 | 25 Gbps | 1 | no — below the floor |
| `g6e.12xlarge` | 4 | 100 Gbps | 1 | yes, by 1 Mbps |
| `g6e.24xlarge` | 4 | 200 Gbps | 2 | yes |
| `g6e.48xlarge` | 8 | 400 Gbps | 4 | yes |

`g6e.12xlarge` is rated exactly 100 Gigabit, so it clears the floor with nothing to spare. Add the
sizes you intend to use to `charts/karpenter-components/templates/node-pool.yaml` if they are not
there.

### 3. The EFA device plugin must advertise on them

Independent of the NodePool, and the more insidious of the two. The `aws-efa-k8s-device-plugin`
release in `main.tf` carries an allow-list:

```yaml
supportedInstanceLabels:
  keys: ["node.kubernetes.io/instance-type"]
  values: [..., "g6e.12xlarge", "g6e.24xlarge", "g6e.48xlarge"]
```

A type missing from it still launches into `cudaefa` and still comes up `Ready` — it simply never
advertises `vpc.amazonaws.com/efa`, so the pod sits `Pending` on a node that looks healthy.

**A louder failure looks like this one and is not it.** The plugin crash-loops with
`Instance type g6e.12xlarge is not EFA enabled / No valid EFA devices found`. The message names
the instance type, so it reads like the allow-list. It is not: **Karpenter attaches an EFA ENI only
when a pod on that node requests `vpc.amazonaws.com/efa`**, so a `cudaefa` node running the
single-node arm, the TCP arm, or nothing at all genuinely has no EFA device and the plugin is
describing that accurately. Check whether anything on the node asked for EFA before editing
`supportedInstanceLabels`.

### 4. Pin multi-node pods to the `cudaefa` pool, not just to an instance type

This one has no error message at all. **EFA traffic is not routed between subnets.** The `cuda`
pool's EC2NodeClass selects subnets in every AZ, so if the two pods of one LWS group land in
different AZs, libfabric falls back to TCP: the deployment comes up, answers correctly, and runs
at a fraction of the expected speed with nothing reporting a problem.

Hence `node_selector: {karpenter.sh/nodepool: cudaefa}` on every multi-node component — that pool
is single-AZ by construction, selecting the subnet tagged from `cuda_efa_az`.

Use the EFA runtime image too, `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0-efa`. The plain
`1.5.0` tag has no `aws-ofi-nccl`, so NCCL cannot use the fabric even when the interface is
attached.

### 5. That pool's AZ has to supply your instance type

Because prerequisite 4 forces the whole group into one subnet, there is no second zone to fall
back to. It does not fail, it hangs: Karpenter retries every ten seconds and the pods stay
`Pending`, with a fleet error on the NodeClaim as the only evidence.

```
  Warning  FailedLaunch  failed launching nodeclaim: creating instance,
                         with fleet error(s), UnfulfillableCapacity: ...
```

Widening `node_types` is the reflex and it does not help much — another 4-GPU size in the same
family leaves the geometry unchanged. So: **stand up your cluster where you have GPU capacity**,
and point `cuda_efa_az` at a zone where you have it. `kubectl get ec2nodeclass cudaefa -o
jsonpath='{.status.subnets}'` reports where the pool actually resolved; a zone your VPC has no
subnet in is no use. Terraform tags every private subnet with `karpenter.sh/discovery/cudaefa`,
setting it to the cluster name on the chosen AZ's subnet and `nil` elsewhere, so the entire effect
of that variable is one tag value.

If you cannot place the group where you have GPUs to spare, drop `vpc.amazonaws.com/efa` and the
`cudaefa` pin and run on the multi-AZ `cuda` pool over TCP —
[`dgd-multinode-tp8-tcp.yaml`](dgd-multinode-tp8-tcp.yaml). That exercises every Dynamo and LWS
mechanism above; it just stops being a measurement of EFA.

### 6. Disable custom all-reduce on GPUs without NVLink

Not an EKS or Dynamo prerequisite — vLLM — but the only item here that stops multi-node dead
rather than slowing it down, and it applies to every g5 and g6e shape. Every multi-node values
file sets `--disable-custom-all-reduce`.

Without it all ranks hang in distributed init and no weight is ever loaded. vLLM detects the
situation, logs `Custom collectives are disabled because this multi-node group does not support
MNNVL multicast`, and then calls `_init_mnnvl_buffer()` anyway, blocking forever in
`torch.distributed._symmetric_memory.rendezvous()`. It presents as a slow model load rather than a
crash: the pod stays `Running`, the leader is `0/1`, and nothing is logged after that line.

To tell it apart from a genuinely slow cold FSx read: a deadlocked rank never prints
`Loading safetensors checkpoint shards`, its `rchar` in `/proc/<pid>/io` is flat across samples,
and `py-spy dump --pid <rank>` shows `rendezvous` / `_init_mnnvl_buffer` instead of a
weight-loading frame.

The flag costs nothing on this hardware — custom all-reduce needs NVLink or working P2P, and L40S
has neither, so the collectives were going over NCCL regardless. On NVLink-connected shapes
(p5/p6) leave it alone.

### Cluster variables

No `.tfvars` is committed in this repository, templates included: a variable file pins a region
and a cluster name, and the buckets it references usually carry an account ID. `*.tfvars` is
gitignored and the working copies belong outside any checkout. Three variables are worth calling
out:

| Variable | Why it matters |
| --- | --- |
| `dynamo_enabled` | defaults to `false`; omitting it plans to **destroy** the operator and its cluster-scoped CRDs |
| `cuda_efa_az` | the single AZ the `cudaefa` pool resolves to — see prerequisite 5 |
| `import_path` | pass via `TF_VAR_import_path`; it is the value most likely to carry an account ID |

Use a separate checkout and a separate Terraform state key per cluster. `s3-backend.sh` rewrites
`backend.tf` in place and `backend.tf` is gitignored, so a shared checkout means one untracked file
decides which cluster your next `apply` targets. Read the plan before confirming — a fresh cluster
should be all creates.

Model weights also do not travel: an FSx for Lustre data repository association needs its S3
bucket in the filesystem's own region, so a new cluster means re-staging the weights. Qwen3-32B is
~65 GB and needs no GPU, so it can download while the cluster builds.

## How to run it

[`serve.ipynb`](serve.ipynb) is the path of least resistance — it checks each prerequisite, stages
the weights into FSx, and deploys the single-node and multi-node arms in turn. By hand, from the
repository root:

```bash
helm upgrade --install dyn-qwen3-32b charts/machine-learning/serving/dynamo \
  -n kubeflow-user-example-com \
  -f examples/inference/dynamo/vllm/qwen3-32b/dgd-multinode-tp8.yaml
```

Then confirm the group formed the way you asked, because every failure mode above is quiet:

| Check | Command | Pass |
| --- | --- | --- |
| Operator authored the LWS | `kubectl get lws -n kubeflow-user-example-com` | one object, `size: 2` |
| One pod per node, same zone | `kubectl get pod -o wide`, then the nodes' `topology.kubernetes.io/zone` | identical zone |
| EFA advertised and requested | `kubectl get node <n> -o jsonpath='{.status.allocatable}'` | `vpc.amazonaws.com/efa: "1"` |
| **NCCL chose the fabric** | set `NCCL_DEBUG=INFO`, then read the leader's log | `NET/OFI Selected Provider is efa` |
| Engine built the layout you asked for | leader log at startup | the TP/PP values from your values file |
| Worker waited, not hung | `kubectl logs <worker> -c wait-for-leader-mp` | finishes; a long tail is Karpenter, not a bug |

The NCCL line matters most: every other EFA prerequisite can be satisfied and libfabric can still
fall back to sockets, in which case the EFA and TCP arms measure the same thing.

## Measuring it on your own cluster

Four configurations, one variable at a time. Hold the model, `--max-model-len`, the memory
fraction, the instance type and the load profile fixed, and vary only where the parallelism is cut
and what carries it.

| Arm | Values file | Nodes | TP | PP | Boundary | Cross-node events per decoded token |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | `dgd-singlenode-tp4.yaml` | 1 | 4 | 1 | none | 0 |
| TP across, EFA | `dgd-multinode-tp8.yaml` | 2 | 8 | 1 | EFA | ~128 all-reduces, ~10 KB each |
| TP across, TCP | `dgd-multinode-tp8-tcp.yaml` | 2 | 8 | 1 | ENA/TCP | ~128 all-reduces, ~10 KB each |
| PP across, EFA | `dgd-multinode-tp4-pp2.yaml` | 2 | 4 | 2 | EFA | 1 activation send |

Baseline → TP-across is the cost of taking tensor parallelism off the local bus. TP-EFA → TP-TCP
is the only pair that isolates what the fabric is worth: same nodes, same zone, same instance
type, differing only in whether the pod requested `vpc.amazonaws.com/efa` and whether the image
carried `aws-ofi-nccl`. TP-across → PP-across is the cost of the placement decision rather than
the fabric.

Load comes from the 8B example's harness, unchanged, so numbers stay comparable across examples.
It runs in-cluster as a Job on purpose — `kubectl port-forward` funnels every stream through one
userspace tunnel and saturates long before eight GPUs do:

```bash
cd ../qwen3-8b/bench
FRONTEND=dyn-qwen3-32b-frontend MODEL=qwen3-32b \
  ./run-bench.sh tp8-efa --concurrency 1,4,16 --requests 64 \
                         --prompt-tokens 1000 --max-tokens 128
```

Run every arm at concurrency 1, 4 and 16. Concurrency 1 isolates latency, which is where a slow
boundary shows worst; 16 shows whether throughput recovers once there is enough work in flight to
overlap the collectives. See [bench/README.md](../qwen3-8b/bench/README.md) for what the harness
holds constant and why.

Three things that will make the numbers lie:

- **Prefix cache leaking between arms.** The harness gives every request a unique prefix *and*
  salts it per invocation, so two runs at the same concurrency do not generate identical prompts.
  Verify rather than assume: `bench/engine-cache.sh` should show a near-zero hit rate for any arm
  meant to be measuring topology.
- **No warmup.** The first request after engine start pays CUDA graph capture. The harness
  discards four and settles for five seconds; driving load by hand usually does not.
- **Silent TCP fallback.** Check the NCCL provider line for every EFA arm, not once.

### Testing KV-aware routing

A different axis, and the only arm where routing exists at all:
[`dgd-router-kv-2x.yaml`](dgd-router-kv-2x.yaml) runs two independent TP=4 workers on the same two
nodes with `--router-mode kv`. The control is the same deployment with `--router-mode round-robin`,
a one-line change that restarts only the frontend — so both arms see the same worker pods with the
same loaded weights.

Five things have to line up, and **none of them fails loudly**. `--router-mode kv` is accepted on
a single-worker deployment, with prefix caching off, with workers that publish nothing, and on a
workload that shares no prefixes; in all four cases it behaves like round-robin.

1. **The frontend runs a KV router** — `--router-mode kv` on `dynamo.frontend`. The default is
   `round-robin`.
2. **The workers publish cache state, which needs its own flag.** `--router-mode kv` on the
   frontend does not make the workers feed it. `dynamo.vllm` builds the publisher only if
   `--kv-events-config` is passed with both keys, because `publisher` defaults to `None`:

   ```yaml
   - --kv-events-config
   - '{"enable_kv_cache_events": true, "publisher": "zmq"}'
   ```

   Without it the worker logs `use_kv_events=False` once at startup and the frontend's radix tree
   stays empty for the life of the deployment.
3. **The frontend maintains a radix tree** of which blocks live on which worker, built from those
   events. `--no-router-kv-events` substitutes a prediction from the router's own past decisions,
   for workers that cannot publish.
4. **Prefix caching must be on in the engine** — on by default in vLLM V1, set explicitly in the
   values file so a future default change breaks loudly instead of flattening the result to zero.
5. **The workload has to share prefixes**, on more distinct prefixes than you have workers, and
   its prefix sequence must not be phase-locked to the router. This is a property of your traffic,
   not your configuration, and it is where this measurement most easily lies to you.

Use **both** instruments, because they measure different things and can disagree:

```bash
# what the frontend's radix tree believed, and how much overlap it thought it was buying
NS=kubeflow-user-example-com FRONTEND=dyn-qwen3-32b-frontend ../qwen3-8b/bench/router-metrics.sh

# how many prompt tokens vLLM actually skipped prefilling
NS=kubeflow-user-example-com WORKER=dyn-qwen3-32b-vllmworker ../qwen3-8b/bench/engine-cache.sh
```

The frontend's `dynamo_component_router_*` metrics exist **only** in `kv` mode — in round-robin
they are flat zero — so they cannot compare a mode against its own control. The workers'
`vllm:prefix_cache_hits_total` is engine-side and reported in both modes, which is the only reason
the comparison is possible. Take a delta around each run; the counters are cumulative.

| `routing decisions` | `KV event batches` | mean overlap | engine hit rate vs control | diagnosis |
| --- | --- | --- | --- | --- |
| 0 | 0 | n/a | — | frontend is not in `kv` mode |
| > 0 | 0 | 0.0000 | — | `kv` mode, workers not publishing — add `--kv-events-config` |
| > 0 | > 0 | > 0 | same as control | router is fine; the **workload** cannot tell the modes apart |
| > 0 | > 0 | > 0 | above control | working, and worth something |

Matching is at KV block granularity, 16 tokens per block here, so a 790-token shared prefix
matches 49 whole blocks and a perfect hit reads as ~0.98 rather than 1.0. Two reporting traps:
`worker_id` on the router metrics is the frontend's own instance id, not the backend's — the
per-target identifier is `router_worker_id`, carried only by `router_worker_registered` — and that
gauge is never cleared when a worker goes away, so cross-check worker counts against
`kubectl get pod`.

### What to expect

- **Do not go multi-node for a model that fits on one node.** TP across nodes loses to the single
  node at every concurrency level, and the gap widens with load rather than closing. Twice the
  GPUs, less throughput.
- **If you must cross nodes, cut at the pipeline boundary.** PP loses slightly at concurrency 1
  for exactly the predicted reason — nothing fills stage 1 while stage 0 works — and wins under
  load, with a markedly better latency tail than TP=8 across the same two nodes.
- **EFA is worth having, and helps most at low concurrency**, where there is no other work to
  overlap collective latency with. It does not rescue a bad topology: choosing the cut point
  correctly is worth about as much as adding the fabric.
- **Graph capture time is a usable proxy for fabric quality**, visible before any benchmark runs.
  If a run meant to be on EFA captures as slowly as the TCP arm, check the provider line.
- **KV-aware routing is a prefill-cost lever.** On short-output traffic decode dominates
  end-to-end time, so eliminating all prefill would buy little; it shows up in the TTFT tail
  rather than the median. It pays on long-prompt, short-output workloads — RAG over a fixed
  corpus, long system prompts, agent loops replaying a transcript. Much of any shared-prefix win
  belongs to vLLM's engine-local prefix cache, not the router; the router earns its keep when the
  prefix set is too large to replicate on every worker.
- **A router and its control scoring the same is usually a statement about your workload**, not
  about the router. The engine-side counter is the only number reported in both modes, so it is
  the only one that can tell you which.

## Notes

- **`--gpu-memory-utilization 0.85`, not the default 0.9.** vLLM sizes the KV cache to fill
  whatever fraction it is given, and CUDA graph capture and NIXL registration buffers then have to
  fit on top. At TP=8 the weights are only ~8 GB per rank, so the cache grows aggressively and the
  trap applies despite 48 GB cards.
- **L40S has no NVLink.** Even the single-node TP=4 baseline runs its collectives over PCIe, so
  the two-node comparison is PCIe versus 100 Gbps EFA, not NVLink versus network. Do not
  generalise the gap to NVLink hardware.
- **One instance type per multi-node component, not a list.** Elsewhere a list of `node_types`
  keeps a deployment schedulable; here the pods schedule independently, so a list lets the leader
  take a `g6e.12xlarge` and the worker a `g6e.24xlarge`, and mismatched GPU counts per node fail
  during distributed init.
- **Qwen3 is a reasoning model.** It emits a `<think>` block before its answer, so keep
  `max_tokens` generous — at 128 the response truncates mid-reasoning and reads like a broken
  deployment.
- **Three mechanisms get called "KV cache" something.** TP/PP collectives (NCCL) move activations
  and partial sums every layer and are what crosses nodes here; KV transfer (NIXL) moves KV blocks
  from a prefill worker to a decode worker and belongs to the [8B
  example](../qwen3-8b/README.md); KV-aware routing moves nothing and is a routing decision.
