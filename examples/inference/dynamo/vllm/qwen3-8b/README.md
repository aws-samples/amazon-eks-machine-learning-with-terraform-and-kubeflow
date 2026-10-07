# NVIDIA Dynamo Qwen3 8B Model

This example illustrates how to use the [Dynamo](../../../../../charts/machine-learning/serving/dynamo/) Helm chart to serve [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) with [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo), first **aggregated** on a single GPU, then **disaggregated**.

There are three values files, in increasing order of GPU appetite:

| File | Topology | GPUs |
| --- | --- | --- |
| [`dgd-agg.yaml`](dgd-agg.yaml) | prefill and decode in one worker | 1 |
| [`dgd-disagg-tp1.yaml`](dgd-disagg-tp1.yaml) | prefill and decode as separate workers, `TP=1` each, on two single-GPU nodes | 2 |
| [`dgd-disagg.yaml`](dgd-disagg.yaml) | prefill and decode as separate workers, `TP=2` each, on one 4-GPU node | 4 |

All three have been run end to end and served real completions.

[`serve.ipynb`](serve.ipynb) walks through the aggregated and disaggregated topologies in one pass and is the quickest way to try this example; the steps below are the same thing outside a notebook.

**If you are here to decide whether to serve disaggregated, read [bench/README.md](bench/README.md) first.** At a fixed GPU budget on identical GPUs, aggregated with two replicas won on both output throughput and time-to-first-token. Disaggregation's win is decode smoothness under mixed load, where it held p99 inter-token latency dramatically lower. Which one you want depends on what you are optimising.

Before proceeding, complete the [Prerequisites](../../../../../README.md#prerequisites) and [Getting started](../../../../../README.md#getting-started). See [What is in the YAML file](../../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files.

Dynamo 1.5.0 also requires CUDA 13.0 and an NVIDIA driver of 580.xx or newer. That comes from the GPU AMI your Karpenter NodePool provisions rather than from the runtime image, so an older AMI is a hard blocker no values file can work around. See [../../README.md](../../README.md) for the full support matrix.

## Dynamo Platform

Unlike the Ray Serve examples, there is no container to build and nothing to install by hand. The Dynamo operator is installed by Terraform from the local [dynamo-platform](../../../../../charts/dynamo-platform/) chart, but it is **off by default**. Set `dynamo_enabled = true` in your `.tfvars` and apply, then confirm it is running:

    kubectl get pods -n dynamo-system
    kubectl get crd | grep nvidia.com

You should see one `dynamo-platform-controller-manager` pod and six `nvidia.com` CRDs — [What gets installed](../../README.md#what-gets-installed) lists what each CRD is for, and why there is no etcd or NATS. An empty result for both means `dynamo_enabled` is still `false`.

The runtime images are pulled anonymously from `nvcr.io`, so no NGC credentials or image pull secret are required.

## Hugging Face Qwen3 8B Pre-trained Model Weights

Qwen3-8B is not a gated model, so no Hugging Face token is needed. To download the weights to shared storage, execute:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug dyn-qwen3-8b-weights     \
            charts/machine-learning/model-prep/hf-snapshot    \
            --set-json='env=[{"name":"HF_MODEL_ID","value":"Qwen/Qwen3-8B"}]' \
            -n kubeflow-user-example-com

This lands the weights at `/fsx/pretrained-models/Qwen/Qwen3-8B`, which is the path all three values files below read. Wait for the job to complete, then uninstall the chart:

    kubectl wait --for=condition=complete job/hf-snapshot-dyn-qwen3-8b-weights \
        -n kubeflow-user-example-com --timeout=3600s
    helm uninstall dyn-qwen3-8b-weights -n kubeflow-user-example-com

## Aggregated Serving

Prefill and decode run in the same process on one GPU. This is the simplest configuration and the one to get working first.

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug dyn-qwen3-8b \
        charts/machine-learning/serving/dynamo/ \
        -f examples/inference/dynamo/vllm/qwen3-8b/dgd-agg.yaml -n kubeflow-user-example-com

Karpenter provisions one single-GPU node — the cheapest of the `node_types` in the values file that has capacity, so which type you get varies. Allow ~5 minutes for the node, the image pull and the vLLM engine to load. Watch progress with:

    kubectl get dynamographdeployment dyn-qwen3-8b -n kubeflow-user-example-com
    kubectl get pods -n kubeflow-user-example-com -w

## Test the Endpoint

The frontend serves an OpenAI-compatible API on port 8000. Port-forward to it:

    kubectl port-forward -n kubeflow-user-example-com \
        svc/dyn-qwen3-8b-frontend 8000:8000

Confirm the worker has registered its model, then send a completion:

    curl -s localhost:8000/v1/models | jq .

    curl -s localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
              "model": "qwen3-8b",
              "messages": [{"role": "user", "content": "What is disaggregated serving?"}],
              "max_tokens": 600
            }' | jq -r '.choices[0].message.content'

Qwen3 is a reasoning model and emits a `<think>` block before its answer, so keep `max_tokens` generous — at 128 the response is truncated mid-reasoning and looks like a broken deployment rather than a budget you set too low.

An empty `data` array from `/v1/models` means the frontend is up but no worker has registered yet — check the worker pod's logs rather than the frontend's.

## Tool Calling

Agents need the model's tool calls as structured `tool_calls`, not as text. [`dgd-agg-tools.yaml`](dgd-agg-tools.yaml) is `dgd-agg.yaml` with two worker flags added:

| flag | what it does |
|---|---|
| `--dyn-tool-call-parser hermes` | Qwen3 writes tool calls as Hermes-style `<tool_call>{...}</tool_call>` blocks. The parser returns them in `message.tool_calls`, with `finish_reason: tool_calls`. |
| `--dyn-reasoning-parser qwen3` | Moves the `<think>` block out of `content` and into `message.reasoning_content`. |

Without them, as in `dgd-agg.yaml`, the model still decides to call the tool, but the call comes back as `<tool_call>` text inside `content`, after the `<think>` block, with `finish_reason: stop` and no `tool_calls`. An agent loop never sees it.

Changing the worker args makes `helm upgrade` leave the old worker running (see [Notes](#notes)), so uninstall the aggregated deployment first:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall dyn-qwen3-8b -n kubeflow-user-example-com
    helm install --debug dyn-qwen3-8b \
        charts/machine-learning/serving/dynamo/ \
        -f examples/inference/dynamo/vllm/qwen3-8b/dgd-agg-tools.yaml -n kubeflow-user-example-com

Port-forward to the frontend as in [Test the Endpoint](#test-the-endpoint), then run [`tool_call_check.py`](tool_call_check.py):

    pip install openai
    python examples/inference/dynamo/vllm/qwen3-8b/tool_call_check.py --base-url http://localhost:8000/v1

It makes the calls an agent loop makes, and prints `PASS` or `FAIL` for each check:

1. A question with one `get_weather` tool. The model must call the tool, with JSON arguments that name the city, and leave no `<tool_call>` or `<think>` markup in `content`.
2. The tool's result, sent back as a `tool` message. The model must answer from it, without calling the tool again.
3. The same question with `tool_choice: "none"`. The model must not call the tool.
4. The same question with `stream: true`. The streamed deltas must add up to the same tool call.

Keep `max_tokens` generous here too: Qwen3 reasons before it calls a tool. The script uses 2048.

## Disaggregated Serving

Prefill and decode run as separate workers exchanging the KV cache over NIXL. This is what Dynamo exists to do: the two phases have very different compute profiles, and separating them stops long prompts from stalling in-flight decodes.

Uninstall the aggregated deployment first. This is not optional — the admission webhook rejects a change to the set of component names on an existing `DynamoGraphDeployment` with `component topology is immutable and cannot be modified after creation`, so `helm upgrade` from `dgd-agg.yaml` to either disaggregated file fails:

    helm uninstall dyn-qwen3-8b -n kubeflow-user-example-com

Then pick one of the two topologies.

**Two GPUs, one per worker** ([`dgd-disagg-tp1.yaml`](dgd-disagg-tp1.yaml)). Each worker runs at `TP=1` on its own single-GPU node, so the KV cache crosses the network between pods:

    helm install --debug dyn-qwen3-8b \
        charts/machine-learning/serving/dynamo/ \
        -f examples/inference/dynamo/vllm/qwen3-8b/dgd-disagg-tp1.yaml -n kubeflow-user-example-com

**Four GPUs, two per worker** ([`dgd-disagg.yaml`](dgd-disagg.yaml)). Each worker runs at `TP=2`, both on one 4-GPU node, so the transfer stays on the host. Worth knowing before you assume that makes it fast: on a `g5.12xlarge` the intra-node transfer measured the same order of magnitude as the cross-AZ network rate, because `g5`'s A10Gs have no NVLink and peer transfers fall back to PCIe. "Same node" is not the same as "GPU fabric":

    helm install --debug dyn-qwen3-8b \
        charts/machine-learning/serving/dynamo/ \
        -f examples/inference/dynamo/vllm/qwen3-8b/dgd-disagg.yaml -n kubeflow-user-example-com

Test either exactly as above; the API surface is unchanged, which is the point — disaggregation is a serving-topology decision, not a client-visible one.

To confirm the request really is being split rather than served entirely by the decode worker, look for the KV transfer in the decode worker's log:

    kubectl logs -n kubeflow-user-example-com \
        -l nvidia.com/dynamo-component-type=decode --tail=200 | grep -E 'NIXL|Transfer'

A served request produces `NIXL compatibility check passed`, a `TransferTopology(...)` line, and `KV Transfer metrics: Num successful transfers=1, ...` with a non-zero throughput. The matching `component="prefill"` request in the prefill worker's log is the other half. If the decode worker reports transfers but the prefill worker never receives a request, the two are not paired and you are effectively running aggregated.

## Observing Prefill and Decode

Two different questions, two different instruments. DCGM tells you what each *GPU* is doing; it cannot tell you time-to-first-token or inter-token latency, which are properties of a *request* and are only visible where requests are — the frontend, or your client.

### Which GPU each phase landed on

Start from the pods, and surface the component type as a column:

    kubectl get pods -n kubeflow-user-example-com -o wide \
        -L nvidia.com/dynamo-component-type

Then ask each worker which physical GPU the device plugin gave it:

    kubectl exec -n kubeflow-user-example-com \
        $(kubectl get pod -n kubeflow-user-example-com \
            -l nvidia.com/dynamo-component-type=prefill -o name) \
        -- nvidia-smi -L

**Join DCGM metrics on the `UUID` label, not on a pod name.** `dcgm-exporter` runs here with `--kubernetes=false`, deliberately — see the comment above the `dcgm_exporter` release in [`main.tf`](../../../../../eks-cluster/terraform/aws-eks-cluster-and-nodegroup/main.tf). Where a GPU workload bypasses the device plugin, the pod label does not come back empty, it comes back naming whichever pod holds the plugin allocation, so every series gets attributed with full confidence to the wrong pod. The `UUID` label is correct by construction.

With the UUIDs in hand, in Grafana or the Prometheus expression browser:

| Question | Metric |
| --- | --- |
| Is this GPU doing anything? | `DCGM_FI_PROF_GR_ENGINE_ACTIVE{UUID="GPU-..."}` |
| Is it doing matmuls? | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` |
| Is it memory-bandwidth-bound? | `DCGM_FI_PROF_DRAM_ACTIVE` |
| How big did the KV cache get? | `DCGM_FI_DEV_FB_USED` |
| Is the GPU moving anything over the bus? | `DCGM_FI_PROF_PCIE_TX_BYTES`, `..._RX_BYTES` — but see below, this does **not** isolate the KV transfer |

A working disaggregation looks *asymmetric*, but only the decode half of that asymmetry is legible from DCGM, and only under real load. Observed on a two-node `TP=1` deployment (prefill on an A10G, decode on an L4) serving one chat-shaped request with a short prompt, sampling every 5 s:

| | prefill (A10G) | decode (L4) |
| --- | --- | --- |
| `GR_ENGINE_ACTIVE` peak | floor | **essentially saturated** |
| `DRAM_ACTIVE` peak | floor | **tracks the engine almost 1:1** |
| `PIPE_TENSOR_ACTIVE` peak | floor | floor |
| `PCIE_RX_BYTES` peak | negligible | negligible |

Decode behaves exactly as the theory says: engine essentially saturated, DRAM activity tracking it almost one-to-one, tensor pipes idle. That is memory-bandwidth-bound autoregressive decode, and it is visible from a single stream.

**Prefill is invisible at this scale, and that is not a fault.** A short prefill is a few milliseconds of compute against a 5 s collection interval, so it never lands in a sample. Neither do the tensor pipes on either GPU — at batch size 1, decode is GEMV, which barely touches them. To see prefill on a GPU trace you need long prompts and sustained concurrency, which means [`bench/loadgen.py`](bench/loadgen.py) with something like `--prompt-tokens 4000 --concurrency 16`, not a single chat request.

**Do not try to read the KV transfer off the PCIe counters.** A single chat request's transfer completes in tens of milliseconds, so smeared across a 5 s window it is indistinguishable from idle bus chatter. On a two-node deployment the transfer also goes GPU → host → NIC → network → host → GPU, so the PCIe counter mixes it with ordinary host traffic rather than isolating it. The authoritative source is the decode worker's own `KV Transfer metrics` log line, which is what the notebook asserts on:

    KV Transfer metrics: Num successful transfers=1, Avg xfer time (ms)=...,
    Avg MB per transfer=..., Throughput (MB/s)=..., Avg number of descriptors=...

The MB figure is its own consistency check: divide it by 144 KiB per prompt token and you should recover the prompt length you sent. Two DCGM traces that look alike tell you very little at low load; a decode worker reporting zero transfers tells you the split is nominal.

Two practical notes. `dcgm-exporter` is a DaemonSet selecting `karpenter.k8s.aws/instance-gpu-manufacturer=nvidia`, so it has **zero pods until Karpenter provisions a GPU node** — an empty `kubectl get pods -n kube-system -l app.kubernetes.io/name=dcgm-exporter` before you deploy is expected, not a broken exporter. And its `--collect-interval=5000` is what makes the `DCGM_FI_PROF_*` fields refresh faster than a measurement batch; at the exporter's own 30 s default, SM_ACTIVE returns the byte-identical value for consecutive scrapes and looks like a stuck metric.

### TTFT and inter-token latency

Dynamo publishes both as Prometheus histograms on the frontend's HTTP port at `/metrics`, enabled by default. The quickest read needs nothing but the port-forward you already have:

    curl -s localhost:8000/metrics | grep -E 'time_to_first_token|inter_token_latency'

**Note the metric prefix.** These are `dynamo_component_router_*`, not `dynamo_frontend_*`, even though the frontend pod is what serves them — the router is a component *inside* the frontend process. NVIDIA's metrics catalog documents a `dynamo_frontend_*` naming that release 1.5.0 does not use; the names below were read off a running 1.5.0 deployment. In 1.5.0 the `dynamo_frontend_*` families are tokenizer, template and event-loop internals, with nothing request-latency-shaped among them.

| Metric | Notes |
| --- | --- |
| `dynamo_component_router_time_to_first_token_seconds` | histogram. Labelled `dynamo_component` (`backend`), `dynamo_namespace`, `router_id`, `worker_id` — there is **no** `model` label on this one |
| `dynamo_component_router_inter_token_latency_seconds` | histogram, same labels |
| `dynamo_component_router_input_sequence_tokens`, `..._output_sequence_tokens` | the real token counts, which is how the prompt lengths in [bench/README.md](bench/README.md) were corrected |
| `dynamo_component_router_kv_hit_rate` | prefix-cache hit rate. Only appears once KV-aware routing has something to route; absent on a fresh single-worker deployment |
| `dynamo_component_request_duration_seconds` | histogram, from the *worker*. This one does carry `model` and `model_name`, plus `dynamo_endpoint` and `dynamo_component_type` |
| `dynamo_component_inflight_requests` | queue depth per worker endpoint |
| `dynamo_request_plane_roundtrip_ttft_seconds` | TTFT measured on the request plane rather than at the router; useful for separating transport from engine time |

Worker-side series are on the worker's `system` port rather than its HTTP port, which is why the shipped PodMonitors scrape `port: system` for workers and `port: http` for the frontend.

Being histograms, quantiles are a query, not a metric:

    histogram_quantile(0.99, sum by (le) (
        rate(dynamo_component_router_inter_token_latency_seconds_bucket[1m])))

**Prometheus is not scraping these yet unless you have re-applied the Terraform.** `dynamo-platform` ships five PodMonitors (frontend, worker, router, planner, epp) carrying only `app.kubernetes.io/managed-by: Helm`, and kube-prometheus-stack leaves `podMonitorSelector` at `{matchLabels: {release: prometheus}}` — so the PodMonitors exist, `kubectl get podmonitor -A` lists them, and Prometheus simply has no such targets. [`main.tf`](../../../../../eks-cluster/terraform/aws-eks-cluster-and-nodegroup/main.tf) sets `podMonitorSelectorNilUsesHelmValues = false` to fix it for every chart at once; until that apply lands, either read the metrics with `curl` as above or label them in place:

    kubectl label podmonitor -n dynamo-system --all release=prometheus

That label is reverted by the next `dynamo-platform` upgrade, so it is a bridge, not the fix. Either way, check that the worker pods carry `nvidia.com/metrics-enabled=true` — the worker PodMonitor selects on it, and the operator omits it if the `DynamoGraphDeployment` is annotated `nvidia.com/enable-metrics: "false"`:

    kubectl get pods -n kubeflow-user-example-com --show-labels | grep metrics-enabled

For a *comparison* rather than a spot reading, use [`bench/loadgen.py`](bench/loadgen.py), which times the streaming response client-side and reports mean and p99 TTFT and ITL per concurrency level. That is the instrument behind the findings in [bench/README.md](bench/README.md). It and the frontend histogram agree on the mean; the histogram excludes the network hop out to your client, and `loadgen.py` includes it, which is one more reason to generate load from inside the cluster.

## Stop Service

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm uninstall dyn-qwen3-8b -n kubeflow-user-example-com

Karpenter deprovisions the GPU node once the pods are gone, after the
`karpenter_consolidate_after` delay.

## Notes

**Constrain the instance type.** Every GPU component sets `node_types`, a list rendered as a required node affinity. The `cuda` NodePool admits all 42 GPU instance types from `g4dn.xlarge` upward, and Karpenter picks the cheapest that satisfies the pod. For a bare `nvidia.com/gpu: 1` request that is a `g4dn.xlarge`, whose T4 has 16GB — enough for Qwen3-8B's weights in bf16 and almost nothing left for a KV cache, so vLLM fails to allocate and the pod crash-loops. Constraining the type is the difference between a working deployment and a confusing OOM.

A *list* rather than a single type, because a given AZ's GPU capacity is finite and pinning to one type turns any shortage of it into a deployment that never schedules. Each instance type *and size* is its own EC2 capacity pool, so listing several sizes per family helps as much as listing several families. If everything stays `Pending`, the reason is in the Karpenter log:

    kubectl logs -n kube-system -l app.kubernetes.io/name=karpenter --tail=200 | grep -i insufficient

Note also that an AZ named in that output may be one the cluster has no subnet for — such a suggestion is not actionable without a VPC change.

**Changing a worker spec leaves the previous generation running.** The operator names each `DynamoComponentDeployment` with a hash of the component spec, so editing anything about a worker — resources, `node_types` — makes `helm upgrade` create a new `...-<newhash>` deployment while the old one keeps its pods and its GPU. A two-worker deployment briefly wants four GPUs, and on a constrained cluster the new pods sit `Pending` behind the old ones forever. `helm uninstall` does clean up correctly, because the component deployments are garbage-collected with their owning `DynamoGraphDeployment`; it is only the upgrade path that strands them. Either uninstall and reinstall, or delete the stale generation explicitly:

    kubectl get dynamocomponentdeployment -n kubeflow-user-example-com
    kubectl delete dynamocomponentdeployment -n kubeflow-user-example-com <stale-name>

**The CRD is v1beta1, and it is not shaped like the published examples.** Most Dynamo material online targets v1alpha1, where components are a *map* under `spec.services`. In v1beta1 — the storage version installed here — they are a *list* under `spec.components`, keyed by `name`, and `podTemplate.spec.containers` is required by the admission webhook. Copying an upstream example verbatim will be rejected.

**`--kv-transfer-config` is mandatory for prefill workers.** Dynamo 1.5.0 rejects the old `--connector` flag outright and refuses to start a `--disaggregation-mode prefill` worker without an explicit `--kv-transfer-config`. Both disaggregated workers set `NixlConnector` with `kv_role: kv_both`.

**Set `--gpu-memory-utilization` explicitly at `TP>1`.** Counter-intuitively, tensor parallelism makes a 24GB card *more* likely to OOM, not less. Sharding shrinks the weights per rank, so vLLM grows the KV cache to fill the default fraction, and CUDA graph capture plus the NIXL registration buffers then have no room — both 4-GPU workers died at near-full occupancy, a little short of what capture needed. [`dgd-disagg.yaml`](dgd-disagg.yaml) sets `0.80`.
