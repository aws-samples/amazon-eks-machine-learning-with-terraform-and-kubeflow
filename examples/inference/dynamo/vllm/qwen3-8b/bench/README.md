# Does disaggregation make Qwen3-8B faster?

A reproducible comparison at a fixed GPU budget. **Measured latency, throughput and bandwidth
figures are deliberately not published here** — they are specific to one cluster, one driver stack
and one image version, and a published table invites exactly the wrong kind of comparison. What is
recorded is the method, the controls, and the qualitative findings, so you can run the arms and get
numbers that describe *your* hardware.

The short version, from the clean experiment — four identical A10Gs on one node, no network between
the workers:

- **Aggregated wins throughput and time-to-first-token**, at every concurrency level tested.
- **Disaggregation wins the decode tail, decisively.** Under prefill-heavy load the aggregated arm's
  p99 inter-token latency degraded by more than an order of magnitude while the disaggregated arm's
  stayed flat. Its mean ITL barely moved across the whole concurrency sweep.
- **The reason disaggregation loses throughput is the KV transfer, and it is quantifiable on your
  own hardware.** The cache is 144 KiB per prompt token (`2 × 36 layers × 8 KV heads × 128 head_dim
  × 2 bytes`). Divide your measured transfer bandwidth by that to get the prompt tokens/s the link
  can carry, shared across every in-flight request. On A10G-class hardware that ceiling was not far
  enough above what one prefill worker produces, so the transfer became the binding constraint.

So at 8B on A10G-class hardware, serve aggregated unless smooth decode under mixed load is what you
are optimising for. That is a statement about this model size and this interconnect, not about
Dynamo: both arms are Dynamo, and the phase isolation it promises was clearly present in the data —
it just could not pay for itself here.

## Two experiments

| | Hardware | GPUs are | Workers are | Status |
| --- | --- | --- | --- | --- |
| [Experiment 1](#experiment-1--4-gpus-one-node-identical-gpus) | one `g5.12xlarge`, 4× A10G 24GB | identical | on the same node | **primary** |
| [Experiment 2](#experiment-2--2-gpus-two-nodes-heterogeneous) | `g6e.xlarge` + `g6.xlarge` | different (L40S, L4) | in different AZs | secondary |

Experiment 1 is the one to trust: it holds the GPUs identical and removes the network. It only
became possible after fixing a `TP=2` out-of-memory bug in
[`../dgd-disagg.yaml`](../dgd-disagg.yaml) — see [What was actually run](#what-was-actually-run).
Experiment 2 was run first, on the single-GPU nodes available at the time, and is kept because its
two role assignments isolate an effect Experiment 1 cannot show.

## The fair-comparison argument

Measuring disaggregation on N GPUs against aggregation on one measures the extra GPUs. To attribute
anything to the topology, the hardware has to be held fixed:

| Arm | Topology | GPU budget |
| --- | --- | --- |
| aggregated | 2 aggregated workers, each doing its own prefill and decode | N |
| disaggregated | 1 prefill worker + 1 decode worker, KV cache over NIXL | N |

Both arms are Dynamo, so this isolates *disaggregation*, not Dynamo-versus-something-else. The
aggregated arm doubles as a vLLM baseline: `dynamo.vllm` runs the same vLLM engine, so the
aggregated workers are ordinary vLLM replicas behind a router. In Experiment 1 both arms also shard
at `TP=2` and set the same `--gpu-memory-utilization`, so the *only* difference is the split.

## Files

| File | Purpose |
| --- | --- |
| [`loadgen.py`](loadgen.py) | Streaming load generator. Standard library only, so it runs in any Python image with no `pip install`. Reports TTFT, inter-token latency and output throughput per concurrency level. |
| [`run-bench.sh`](run-bench.sh) | Ships `loadgen.py` into the cluster as a ConfigMap and runs it as a Job against the frontend Service. |
| [`router-metrics.sh`](router-metrics.sh) | Reads the frontend's `dynamo_component_router_*` metrics. KV-routing arms only — these are flat zero in round-robin. |
| [`engine-cache.sh`](engine-cache.sh) | Reads each worker's own `vllm:prefix_cache_*` counters. Engine-side, so reported in every routing mode. |
| [`dgd-agg-4gpu.yaml`](dgd-agg-4gpu.yaml) | Experiment 1 aggregated arm: 2 replicas × `TP=2` on one 4-GPU node. Pairs with [`../dgd-disagg.yaml`](../dgd-disagg.yaml). |
| [`dgd-agg-2x.yaml`](dgd-agg-2x.yaml) | Experiment 2 aggregated arm: 2 replicas × `TP=1`. |
| [`dgd-disagg-2x.yaml`](dgd-disagg-2x.yaml) | Experiment 2 disaggregated arm: prefill + decode, `TP=1` each. |

The load must be generated **inside** the cluster. A `kubectl port-forward` tunnel is a single
userspace hop on your workstation and saturates long before the GPUs do, which would flatten exactly
the differences being measured.

The Experiment 2 files pin `node_types` to a single instance type, unlike the example files one
directory up which list many for capacity resilience. The comparison is only valid if the arms run
on identical GPUs. If you repoint a pin, repoint it in **both** files.

## How to run

The weights and the Dynamo platform are prerequisites — see the [example README](../README.md).
Then, for each arm:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    EX=examples/inference/dynamo/vllm/qwen3-8b

    helm install dyn-qwen3-8b charts/machine-learning/serving/dynamo/ \
        -f $EX/dgd-disagg.yaml -n kubeflow-user-example-com

    kubectl get pods -n kubeflow-user-example-com -w    # wait for 3/3 Ready

    $EX/bench/run-bench.sh disagg4-a \
        --prompt-tokens 1000 --max-tokens 128 --concurrency 1,4,16,32 --requests 32
    $EX/bench/run-bench.sh disagg4-b \
        --prompt-tokens 4000 --max-tokens 16 --concurrency 1,4,16 --requests 24

    helm uninstall dyn-qwen3-8b -n kubeflow-user-example-com

Then repeat with `bench/dgd-agg-4gpu.yaml`. **Uninstall between arms rather than `helm upgrade`** —
the admission webhook rejects a change to the set of component names with `component topology is
immutable`, and even for a permitted change the operator hashes the component spec into the
`DynamoComponentDeployment` name, so an upgrade strands the previous generation on its GPUs.

`run-bench.sh` prints one JSON row per concurrency level plus a `# JSON_SUMMARY` line. Set
`LOG=<path>` to tee the output to a file — this namespace garbage-collects completed Job pods, and
the Job still reads `Complete` long after `kubectl logs job/...` has nothing left to return.

## Experiment 1 — 4 GPUs, one node, identical GPUs

One `g5.12xlarge`: 4× A10G 24GB, 48 vCPU, 192 GiB. Both arms `TP=2`,
`--gpu-memory-utilization 0.80`. `ignore_eos` is set so every request emits exactly `max_tokens` and
throughput stays comparable. All requests succeeded in every arm.

Two workloads, run against both arms:

| Workload | Shape | Why |
| --- | --- | --- |
| A | chat-shaped — ~620 prompt tokens, 128 output tokens | the common case; decode dominates |
| B | prefill-heavy — ~2,580 prompt tokens, 16 output tokens | the shape disaggregation is supposed to favour |

What the sweep showed:

- **Workload A.** Aggregated led on output throughput at every concurrency, by a small margin at one
  request and a wide one at 32, and led on TTFT throughout. Disaggregation's ITL stayed nearly level
  as concurrency rose while aggregated's mean more than doubled and its p99 grew by more than an
  order of magnitude.
- **Workload B — the sharpest contrast in the study.** Aggregated moved substantially more requests
  per second, but its decode stream fell apart: mean and p99 ITL both degraded badly, which is
  prefill chunks stalling in-flight decodes — the textbook interference disaggregation exists to
  remove. The disaggregated arm's decode GPUs never see a prefill, so its ITL stayed flat with a
  tight p99.

If you are streaming tokens to a human, an order-of-magnitude better ITL tail is not a rounding
error — it is the difference between smooth output and visible stutter.

## Experiment 2 — 2 GPUs, two nodes, heterogeneous

Run first, before a 4-GPU node was available. One `g6e.xlarge` (L40S) and one `g6.xlarge` (L4), in
different availability zones, `TP=1` each. Because the fleet is heterogeneous the disaggregated arm
was run **twice with the roles swapped**, which bounds the effect of role-to-hardware assignment
rather than hiding it.

The throughput column where disaggregation appears to win here is an artifact of heterogeneity, and
the arithmetic proves it. Single-stream ITL differed sharply between decode-on-L40S and decode-on-L4.
The aggregated arm had one replica on each, so its mean should be the average of those two — and
measured, it was, to two decimal places. The aggregated arm is not slower per token; it is a 50/50
mix of a fast and a slow decoder, while `disagg, decode on L40S` puts *all* decode on the fast GPU.

That is a genuine capability — disaggregation lets you assign hardware by phase, so the
memory-bandwidth-bound phase can sit on your high-bandwidth GPUs — but it is not a topology speedup,
and it vanishes on the homogeneous fleet of Experiment 1.

Workload B on this fleet showed the same pattern as Experiment 1, more extremely.

## Analysis: where the time goes

**The TTFT penalty is the KV transfer, and it is measurable.** Read it from the decode worker's own
`KV Transfer metrics` log line, one request at a time, with no competing load:

    KV Transfer metrics: Num successful transfers=1, Avg xfer time (ms)=...

Collect it at two prompt lengths and at both `TP=1` and `TP=2`. At `TP=2` each rank holds half the
cache, so the metrics report two transfers of half the size running in parallel — sum them.

Three independent checks should agree on the volume: MB moved divided by prompt tokens at each of
your two prompt lengths, and the architecture, which says `2 × 36 layers × 8 KV heads × 128 head_dim
× 2 bytes` = **144 KiB/token** exactly. If your measured per-token volume disagrees with that, the
arm is not configured the way you think.

**Staying on one node bought far less than expected.** This was the surprise. `g5`'s A10Gs have no
NVLink, so peer transfers go over PCIe, and the intra-node bandwidth came out the same order of
magnitude as the cross-AZ network — not orders of magnitude better. Divide by the per-token volume
and the intra-node path carries only a few thousand prompt tokens/s shared across all in-flight
requests, which is not enough headroom over a single prefill worker's output to absorb concurrency.
Under concurrent load in Experiment 2, per-transfer throughput collapsed by roughly an order of
magnitude and transfer time rose accordingly, which is what produced the very long TTFT there.

**Phase isolation is real and is the one unambiguous win.** In Experiment 1 it cannot be attributed
to GPU heterogeneity (the GPUs are identical) or to networking (there is none between the workers).
The aggregated arm's p99 ITL degraded badly under prefill-heavy load; the disaggregated arm's stayed
flat. That is the mechanism working as designed.

## Conclusions

- **At 8B on A10G-class GPUs, serve aggregated with multiple replicas** if you are optimising
  throughput or TTFT. It won both at every concurrency in the clean experiment.
- **Serve disaggregated if smooth decode under mixed load is the goal.** The ITL tail under
  prefill-heavy traffic is the strongest effect measured here, and it is exactly what disaggregation
  is for.
- **Budget the interconnect explicitly: 144 KiB per prompt token.** This is the number that decides
  whether disaggregation pays. Neither PCIe-attached A10Gs nor a cross-AZ VPC hop had the headroom.
  What plausibly does: NVLink (`p4d`/`p5`, which have it, unlike `g5`) or EFA at 400 Gbps via the
  cluster's `cudaefa` NodePool. Both remain unmeasured here.
- **Disaggregation's case strengthens with model size and TP degree**, neither of which this study
  varies. An 8B model at `TP=2` is close to the least favourable case: the weights are small enough
  that aggregation wastes little, while the KV cache per token is unchanged.
- **Prefer a homogeneous fleet, and co-locate if you disaggregate across nodes.** Experiment 2 shows
  how much a mixed fleet can distort a read of the numbers.

## Limitations

Stated plainly, because they bound how far these findings travel:

- **One model, one size.** Qwen3-8B only, `TP=1` and `TP=2`. No larger model, no MoE.
- **No NVLink and no EFA measured.** The strongest case for disaggregation is the one not tested,
  because `g5` has neither. This is the main reason not to read these findings as a verdict on
  disaggregation in general.
- **Experiment 2 used heterogeneous GPUs in different AZs,** because two identical nodes were not
  available at the time. Both role assignments were measured to bound the effect, and the ITL
  arithmetic quantifies it, but Experiment 1 is the trustworthy one.
- **Prompt lengths are approximate.** `loadgen.py` sizes prompts at ~4 characters per token; the
  prose it uses actually runs about 6.1, so `--prompt-tokens 1000` produces appreciably fewer real
  tokens. Derive the true count from the server's own KV volumes rather than from the flag.
- **Short runs.** 24–128 requests per concurrency level. Enough to separate effects this large; not
  enough for tight confidence intervals on the p99s.
- **No cost axis.** Aggregated winning throughput on the same hardware means it also wins on cost
  per token here, but no dollar figures were computed.

## What was actually run

The functional proof that Dynamo serves at all is [`../serve.ipynb`](../serve.ipynb), which runs both
topologies end to end. For the benchmark specifically:

**Experiment 2 first**, before a 4-GPU node was available. A homogeneous pin could not place both
pods, so the two GPU nodes already held were annotated `karpenter.sh/do-not-disrupt=true` to stop
consolidation reclaiming the idle one during the re-plan, and the per-arm placement was applied as
an override rather than committed, keeping the files homogeneously pinned:

    --set-json 'components.VllmPrefillWorker.node_types=["g6e.xlarge"]' \
    --set-json 'components.VllmDecodeWorker.node_types=["g6.xlarge"]'

Five arms followed: workload A on disagg/prefill-on-L40S, disagg/decode-on-L40S and aggregated;
workload B on aggregated and disagg/prefill-on-L40S.

**Experiment 1 second**, on a `g5.12xlarge`. The first install
failed: both workers died with `CUDA out of memory` during startup at near-full GPU occupancy. GPU
isolation was ruled out first — each pod had two distinct devices with distinct per-rank PIDs —
leaving vLLM's default memory target as the cause, since sharding the weights lets the KV cache grow
until CUDA graph capture and NIXL registration overrun the card. Adding
`--gpu-memory-utilization 0.80` to [`../dgd-disagg.yaml`](../dgd-disagg.yaml) fixed it. Four arms
followed: workloads A and B on disaggregated, then the same two on
[`dgd-agg-4gpu.yaml`](dgd-agg-4gpu.yaml).

Checks performed along the way, each of which could have invalidated a result:

- **Both aggregated replicas really registered and really got their own GPUs.** The frontend logged
  `KubeDiscoveryClient::list returning 2 instances for query=AllModels`, and in Experiment 1 both
  replicas landed on the one node with 2 GPUs each.
- **Remote prefill genuinely engages**, rather than the decode worker prefilling short prompts
  locally — which would have made the disaggregated arms aggregated in disguise. At both prompt
  lengths the prefill worker showed the prompt-throughput spike and the decode worker showed none.
- **Uncontended `KV Transfer metrics` were collected separately** from the loaded runs, so the
  bandwidth readings in the analysis are not confounded by queueing.

One observation not chased down: during the Experiment 2 prefill-heavy run the prefill worker logged
`Releasing expired KV blocks for request ... which were retrieved by 0 remote worker(s) before lease
expired`. Under a saturated transfer link some prefill results appear to be computed and then
discarded before decode can fetch them, wasting prefill capacity on top of the bandwidth limit.
Worth investigating if you pursue disaggregation on a slow interconnect; it does not change the
conclusions, since the link was already the bottleneck.
