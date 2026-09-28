# Does disaggregation make Qwen3-8B faster?

A measured answer for this cluster, at a fixed GPU budget. The short version, from the clean
experiment — four identical A10Gs on one node, no network between the workers:

- **Aggregated wins throughput and time-to-first-token.** 655 vs 434 output tok/s at 32-way
  concurrency (disaggregation is 34% slower), and TTFT is 2–6× better at every concurrency.
- **Disaggregation wins the decode tail, decisively.** Under prefill-heavy load the aggregated
  arm's p99 inter-token latency degraded to 620 ms while the disaggregated arm held at 29 ms —
  a **21× better tail**. Its mean ITL never moved from 20 ms.
- **The reason disaggregation loses throughput is the KV transfer, and it is quantified.** The
  cache is 144 KiB per prompt token, and the transfer ran at 474–532 MB/s intra-node, giving
  ~3,800 prompt tokens/s shared across every in-flight request. That is not enough headroom
  over what one prefill worker produces, so the transfer becomes the binding constraint.

So at 8B on A10G-class hardware, serve aggregated unless smooth decode under mixed load is
what you are optimising for. That is a statement about this model size and this interconnect,
not about Dynamo: both arms are Dynamo, and the phase isolation it promises was clearly
present in the data — it just could not pay for itself here.

## Two experiments

| | Hardware | GPUs are | Workers are | Status |
| --- | --- | --- | --- | --- |
| [Experiment 1](#experiment-1--4-gpus-one-node-identical-gpus) | one `g5.12xlarge`, 4× A10G 24GB | identical | on the same node | **primary** |
| [Experiment 2](#experiment-2--2-gpus-two-nodes-heterogeneous) | `g6e.xlarge` + `g6.xlarge` | different (L40S, L4) | in different AZs | secondary |

Experiment 1 is the one to trust: it holds the GPUs identical and removes the network. It only
became possible after fixing a `TP=2` out-of-memory bug in
[`../dgd-disagg.yaml`](../dgd-disagg.yaml) — see [What was actually run](#what-was-actually-run).
Experiment 2 was run first, when 4-GPU capacity was unavailable, and is kept because its two
role assignments isolate an effect Experiment 1 cannot show.

## The fair-comparison argument

Measuring disaggregation on N GPUs against aggregation on one measures the extra GPUs. To
attribute anything to the topology, the hardware has to be held fixed:

| Arm | Topology | GPU budget |
| --- | --- | --- |
| aggregated | 2 aggregated workers, each doing its own prefill and decode | N |
| disaggregated | 1 prefill worker + 1 decode worker, KV cache over NIXL | N |

Both arms are Dynamo, so this isolates *disaggregation*, not Dynamo-versus-something-else. The
aggregated arm doubles as a vLLM baseline: `dynamo.vllm` runs the same vLLM engine, so the
aggregated workers are ordinary vLLM replicas behind a router. In Experiment 1 both arms also
shard at `TP=2` and set the same `--gpu-memory-utilization`, so the *only* difference is the
split.

## Files

| File | Purpose |
| --- | --- |
| [`loadgen.py`](loadgen.py) | Streaming load generator. Standard library only, so it runs in any Python image with no `pip install`. Reports TTFT, inter-token latency and output throughput per concurrency level. |
| [`run-bench.sh`](run-bench.sh) | Ships `loadgen.py` into the cluster as a ConfigMap and runs it as a Job against the frontend Service. |
| [`dgd-agg-4gpu.yaml`](dgd-agg-4gpu.yaml) | Experiment 1 aggregated arm: 2 replicas × `TP=2` on one 4-GPU node. Pairs with [`../dgd-disagg.yaml`](../dgd-disagg.yaml). |
| [`dgd-agg-2x.yaml`](dgd-agg-2x.yaml) | Experiment 2 aggregated arm: 2 replicas × `TP=1`. |
| [`dgd-disagg-2x.yaml`](dgd-disagg-2x.yaml) | Experiment 2 disaggregated arm: prefill + decode, `TP=1` each. |

The load must be generated **inside** the cluster. A `kubectl port-forward` tunnel is a single
userspace hop on your workstation and saturates long before the GPUs do, which would flatten
exactly the differences being measured.

The Experiment 2 files pin `node_types` to a single instance type, unlike the example files one
directory up which list many for capacity resilience. The comparison is only valid if the arms
run on identical GPUs. If you repoint a pin, repoint it in **both** files.

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

Then repeat with `bench/dgd-agg-4gpu.yaml`. **Uninstall between arms rather than
`helm upgrade`** — the admission webhook rejects a change to the set of component names with
`component topology is immutable`, and even for a permitted change the operator hashes the
component spec into the `DynamoComponentDeployment` name, so an upgrade strands the previous
generation on its GPUs.

`run-bench.sh` prints one JSON row per concurrency level plus a `# JSON_SUMMARY` line.

## Experiment 1 — 4 GPUs, one node, identical GPUs

One `g5.12xlarge`: 4× A10G 24GB, 48 vCPU, 192 GiB. Both arms `TP=2`,
`--gpu-memory-utilization 0.80`. `ignore_eos` is set so every request emits exactly
`max_tokens` and throughput stays comparable. All requests succeeded in every arm.

### Workload A — chat-shaped: ~620 prompt tokens, 128 output tokens

| Concurrency | | aggregated | disaggregated |
| --- | --- | --- | --- |
| 1 | output tok/s | **48.4** | 45.0 |
| | mean TTFT ms | **188** | 391 |
| | mean ITL ms | 19.33 | **19.32** |
| 4 | output tok/s | **180.8** | 151.8 |
| | mean TTFT ms | **249** | 691 |
| | mean ITL ms | 20.09 | **19.93** |
| 16 | output tok/s | **481.4** | 385.7 |
| | mean TTFT ms | **550** | 2065 |
| | mean ITL ms | 28.39 | **23.81** |
| | p99 ITL ms | 189.5 | **28.0** |
| 32 | output tok/s | **654.9** | 434.3 |
| | mean TTFT ms | **869** | 5264 |
| | mean ITL ms | 40.72 | **26.46** |
| | p99 ITL ms | 374.4 | **35.4** |

Aggregated is 8% faster at one request and 51% faster at 32 (equivalently, disaggregation is
34% slower there), with far better TTFT throughout.
Disaggregation's ITL is level (19.3 → 26.5 ms) while aggregated's more than doubles
(19.3 → 40.7 ms) and its p99 grows 18×.

### Workload B — prefill-heavy: ~2,580 prompt tokens, 16 output tokens

The shape disaggregation is supposed to favour, since prefill dominates.

| Concurrency | | aggregated | disaggregated |
| --- | --- | --- | --- |
| 1 | req/s | **1.131** | 0.516 |
| | mean TTFT ms | **588** | 1643 |
| | mean ITL ms | 19.66 | 19.66 |
| 4 | req/s | **2.493** | 0.844 |
| | mean TTFT ms | **709** | 4311 |
| | mean ITL ms | 55.63 | **20.74** |
| | p99 ITL ms | 593.0 | **23.3** |
| 16 | req/s | **2.507** | 0.986 |
| | mean TTFT ms | **1917** | 15365 |
| | mean ITL ms | 287.23 | **20.81** |
| | p99 ITL ms | 620.2 | **29.1** |

This is the sharpest contrast in the whole study. Aggregated moves 2.5× the requests, but its
decode stream falls apart: mean ITL 287 ms, p99 620 ms — prefill chunks stalling in-flight
decodes, the textbook interference disaggregation exists to remove. The disaggregated arm's
decode GPUs never see a prefill, so its ITL is flat at 20.8 ms with a p99 of 29 ms.

If you are streaming tokens to a human, a 14× better mean ITL and 21× better p99 is not a
rounding error — it is the difference between smooth output and visible stutter.

## Experiment 2 — 2 GPUs, two nodes, heterogeneous

Run first, when no 4-GPU type had capacity. One `g6e.xlarge` (L40S, 864 GB/s) and one
`g6.xlarge` (L4, 300 GB/s), in different availability zones, `TP=1` each. Because the fleet is
heterogeneous the disaggregated arm was run **twice with the roles swapped**, which bounds the
effect of role-to-hardware assignment rather than hiding it.

Workload A, output tokens/s:

| Concurrency | aggregated | disagg, decode on L40S | disagg, decode on L4 |
| --- | --- | --- | --- |
| 1 | 24.0 | **38.4** | 15.9 |
| 4 | 83.2 | **123.7** | 54.9 |
| 16 | 253.0 | **312.9** | 166.3 |
| 32 | **406.0** | 352.1 | 274.4 |

Workload A, mean TTFT ms:

| Concurrency | aggregated | disagg, decode on L40S | disagg, decode on L4 |
| --- | --- | --- | --- |
| 1 | **130** | 574 | 429 |
| 4 | **199** | 960 | 1012 |
| 16 | **401** | 2872 | 2542 |
| 32 | **620** | 7342 | 3338 |

Workload A, mean ITL ms:

| Concurrency | aggregated | disagg, decode on L40S | disagg, decode on L4 |
| --- | --- | --- | --- |
| 1 | 40.9 | **21.7** | 60.1 |
| 4 | 44.0 | **23.3** | 64.2 |
| 16 | 51.5 | **25.0** | 69.7 |
| 32 | 64.1 | **27.3** | 80.6 |

The throughput column where disaggregation appears to win is an artifact of heterogeneity, and
the arithmetic proves it: single-stream ITL was 21.7 ms with decode on the L40S and 60.1 ms
with decode on the L4. The aggregated arm had one replica on each, so its mean should be the
average of the two: `(21.7 + 60.1) / 2 = 40.9`. Measured: **40.91 ms**. The aggregated arm is
not slower per token — it is a 50/50 mix of a fast and a slow decoder, while
`disagg, decode on L40S` puts *all* decode on the fast GPU.

That is a genuine capability — disaggregation lets you assign hardware by phase, so the
memory-bandwidth-bound phase can sit on your high-bandwidth GPUs — but it is not a topology
speedup, and it vanishes on the homogeneous fleet of Experiment 1.

Workload B on this fleet showed the same pattern as Experiment 1, more extremely: aggregated
2.36 req/s against 0.77, and disaggregated ITL flat at 61 → 67 ms (p99 130) where aggregated
degraded 42 → 206 ms (p99 627).

## Analysis: where the time goes

**The TTFT penalty is the KV transfer, and it is measurable.** From the decode worker's
`KV Transfer metrics`, one request at a time:

| Experiment | Prompt tokens | Total MB moved | Transfer time | Aggregate throughput |
| --- | --- | --- | --- | --- |
| 2 (cross-AZ, `TP=1`) | 845 | 119.25 | 424 ms | 281 MB/s |
| 2 (cross-AZ, `TP=1`) | 5,605 | 789.75 | 2,771 ms | 285 MB/s |
| 1 (intra-node, `TP=2`) | 845 | 119.25 | 252 ms | 474 MB/s |
| 1 (intra-node, `TP=2`) | 5,605 | 789.75 | 1,483 ms | 532 MB/s |

At `TP=2` each rank holds half the cache, so the metrics report two transfers of half the size
running in parallel; the table gives the totals.

Three independent checks agree on the volume: 119.25 MB / 845 tokens = 144.5 KiB/token,
789.75 MB / 5,605 = 144.3 KiB/token, and the architecture says
`2 × 36 layers × 8 KV heads × 128 head_dim × 2 bytes` = **144 KiB/token** exactly.

**Staying on one node bought only ~1.8×, not orders of magnitude.** This was the surprise.
`g5`'s A10Gs have no NVLink, so peer transfers go over PCIe, and 532 MB/s is the same order as
the 285 MB/s cross-AZ network. Divide by the per-token volume and the intra-node path carries
~3,800 prompt tokens/s shared across all in-flight requests — not enough headroom over a
single prefill worker's output to absorb concurrency. Under 16 concurrent requests in
Experiment 2 per-transfer throughput collapsed from 285 MB/s to 28–37 MB/s and transfer time
rose to ~11 s, which is what produced the 18 s TTFT there.

**Phase isolation is real and is the one unambiguous win.** In Experiment 1 it cannot be
attributed to GPU heterogeneity (the GPUs are identical) or to networking (there is none
between the workers). The aggregated arm's p99 ITL rose to 620 ms under prefill-heavy load;
the disaggregated arm's stayed at 29 ms. That is the mechanism working as designed.

## Conclusions

- **At 8B on A10G-class GPUs, serve aggregated with multiple replicas** if you are optimising
  throughput or TTFT. It won both at every concurrency in the clean experiment.
- **Serve disaggregated if smooth decode under mixed load is the goal.** A 14× better mean and
  21× better p99 inter-token latency under prefill-heavy traffic is the strongest effect
  measured here, and it is exactly what disaggregation is for.
- **Budget the interconnect explicitly: 144 KiB per prompt token.** This is the number that
  decides whether disaggregation pays. Neither PCIe-attached A10Gs nor a cross-AZ VPC hop had
  the headroom. What plausibly does: NVLink (`p4d`/`p5`, which have it, unlike `g5`) or EFA at
  400 Gbps via the cluster's `cudaefa` NodePool. Both remain unmeasured here.
- **Disaggregation's case strengthens with model size and TP degree**, neither of which this
  study varies. An 8B model at `TP=2` is close to the least favourable case: the weights are
  small enough that aggregation wastes little, while the KV cache per token is unchanged.
- **Prefer a homogeneous fleet, and co-locate if you disaggregate across nodes.** Experiment 2
  shows how much a mixed fleet can distort a read of the numbers.

## Limitations

Stated plainly, because they bound how far these numbers travel:

- **One model, one size.** Qwen3-8B only, `TP=1` and `TP=2`. No larger model, no MoE.
- **No NVLink and no EFA measured.** The strongest case for disaggregation is the one not
  tested, because `g5` has neither. This is the main reason not to read these numbers as a
  verdict on disaggregation in general.
- **Experiment 2 used heterogeneous GPUs in different AZs.** `g6e.xlarge` and then `g6.xlarge`
  were both at `InsufficientInstanceCapacity` in all three of the cluster's availability zones,
  so two identical nodes could not be obtained. Both role assignments were measured to bound
  the effect, and the ITL arithmetic quantifies it, but Experiment 1 is the trustworthy one.
- **Prompt lengths are approximate.** `loadgen.py` sizes prompts at ~4 characters per token;
  the prose it uses actually runs about 6.1, so `--prompt-tokens 1000` produced ~620 real
  tokens. The tables report the measured values, derived from the server's own KV volumes.
- **Short runs.** 24–128 requests per concurrency level. Enough to separate effects this large;
  not enough for tight confidence intervals on the p99s.
- **No cost axis.** Aggregated winning throughput on the same hardware means it also wins on
  cost per token here, but no dollar figures were computed.

## What was actually run

The functional proof that Dynamo serves at all is [`../serve.ipynb`](../serve.ipynb), which runs
both topologies end to end. For the benchmark specifically:

**Experiment 2 first**, because 4-GPU capacity was unavailable. `dgd-disagg-2x.yaml` pinned to
`g6e.xlarge` could not get a second node — `InsufficientInstanceCapacity` in all three AZs —
and repinning to `g6.xlarge` hit the same wall. The two GPU nodes already held were annotated
`karpenter.sh/do-not-disrupt=true` so consolidation would not reclaim the idle one during the
re-plan, and the per-arm placement was applied as an override rather than committed, keeping
the files homogeneously pinned:

    --set-json 'components.VllmPrefillWorker.node_types=["g6e.xlarge"]' \
    --set-json 'components.VllmDecodeWorker.node_types=["g6.xlarge"]'

Five arms followed: workload A on disagg/prefill-on-L40S, disagg/decode-on-L40S and
aggregated; workload B on aggregated and disagg/prefill-on-L40S.

**Experiment 1 second**, once GPU cost was explicitly deprioritised and a `g5.12xlarge` was
obtainable in `us-west-2c`. The first install failed: both workers died with `CUDA out of
memory. Tried to allocate 150.00 MiB. GPU 0 has a total capacity of 22.30 GiB of which
100.69 MiB is free` at 99.5% occupancy. GPU isolation was ruled out first — each pod had two
distinct devices with distinct per-rank PIDs — leaving vLLM's default memory target as the
cause, since sharding the weights to 7.6 GiB/rank lets the KV cache grow until CUDA graph
capture and NIXL registration overrun the card. Adding `--gpu-memory-utilization 0.80` to
[`../dgd-disagg.yaml`](../dgd-disagg.yaml) fixed it; all pods Ready in 220 s. Four arms
followed: workloads A and B on disaggregated, then the same two on
[`dgd-agg-4gpu.yaml`](dgd-agg-4gpu.yaml).

Checks performed along the way, each of which could have invalidated a result:

- **Both aggregated replicas really registered and really got their own GPUs.** The frontend
  logged `KubeDiscoveryClient::list returning 2 instances for query=AllModels`, and in
  Experiment 1 both replicas landed on the one node with 2 GPUs each.
- **Remote prefill genuinely engages**, rather than the decode worker prefilling short prompts
  locally — which would have made the disaggregated arms aggregated in disguise. At both 845
  and 5,605 prompt tokens the prefill worker showed the prompt-throughput spike and the decode
  worker showed none.
- **Uncontended `KV Transfer metrics` were collected separately** from the loaded runs, so the
  bandwidth figures in the analysis are not confounded by queueing.

One observation not chased down: during the Experiment 2 prefill-heavy run the prefill worker
logged `Releasing expired KV blocks for request ... which were retrieved by 0 remote worker(s)
before lease expired`. Under a saturated transfer link some prefill results appear to be
computed and then discarded before decode can fetch them, wasting prefill capacity on top of
the bandwidth limit. Worth investigating if you pursue disaggregation on a slow interconnect;
it does not change the conclusions, since the link was already the bottleneck.
