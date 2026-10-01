# NVIDIA Dynamo

[Dynamo](https://github.com/ai-dynamo/dynamo) serves models through a pluggable inference
backend, with an OpenAI-compatible frontend in front of it. Examples are grouped by backend,
and each directory is named exactly as the `backend_framework` value it uses, so the directory
name is the string you put in the values file.

| Backend | Directory | Runtime image | Status in this repository |
| --- | --- | --- | --- |
| vLLM | [`vllm/`](vllm/) | `nvcr.io/nvidia/ai-dynamo/vllm-runtime` | Verified — see [vllm/qwen3-8b](vllm/qwen3-8b/) |
| SGLang | — | `nvcr.io/nvidia/ai-dynamo/sglang-runtime` | Not yet |
| TensorRT-LLM | — | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime` | Not yet |

vLLM is first because NVIDIA's [support matrix](https://docs.nvidia.com/dynamo/dev/reference/compatibility)
gives it the broadest feature coverage of the three. All three are GA for disaggregated
serving, KV-aware routing and the SLA planner, but they differ elsewhere:

| | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| KV block manager | yes | work in progress | yes |
| LoRA adapters | yes | no | no |
| Multimodal | image, video, audio (experimental) | image, video | image |
| Request cancellation | yes | yes | incomplete |

Dynamo also ships `dynamo.triton` and `dynamo.mocker`, which are deliberately not covered
here. `dynamo.triton` wraps Triton's tensor backends over KServe gRPC — it rejects FP16 and
BF16 tensors and has no KV-aware routing, so it is not an LLM path, and this repository already
has [Triton Inference Server examples](../triton-inference-server/) that would be easy to
confuse it with. `dynamo.mocker` is a GPU-free mock engine for exercising routing and the
planner. Neither has a published runtime image, so unlike the three backends above, either
would require a container build.

## No container build

Unlike the Ray Serve and Triton examples, there is nothing under
[`containers/`](../../../containers/) for Dynamo. NVIDIA publishes a complete runtime image per
backend, and each already contains the inference engine, the `dynamo.*` entrypoint modules and
NIXL. The images are anonymously pullable from `nvcr.io`, so no NGC credentials, no
`imagePullSecret`, and no push to your own registry.

Pin an exact tag. The `vllm-runtime` repository carries several hundred tags, most of them
development builds like `1.5.0-kimi-k3-dev.1-cuda13`. There are also `-efa` variants of all
three runtimes for multi-node deployments over Elastic Fabric Adapter.

## Requirements

Dynamo 1.5.0 needs **CUDA 13.0** — 13.1 for TensorRT-LLM — and an **NVIDIA driver of 580.xx or
newer**. These come from the GPU AMI your Karpenter NodePool provisions, not from the runtime
image, so an older AMI is a hard blocker rather than something a values file can work around.

## What gets installed

The operator installs from the local [dynamo-platform](../../../charts/dynamo-platform/) chart
via `terraform apply`. It is **off by default** — set `dynamo_enabled = true` in your `.tfvars`
and apply before working through these examples. There is no manual installation step and no
external chart repository.

It is a single pod:

    $ kubectl get pods -n dynamo-system
    NAME                                                  READY   STATUS    RESTARTS   AGE
    dynamo-platform-controller-manager-...                1/1     Running   0          4d6h

That one pod is the controller manager, the conversion webhook and the validating webhook. There
is **no etcd and no NATS**, which is worth knowing because most Dynamo material online installs
both. Dynamo 1.5.0 defaults to `discoveryBackend: kubernetes` — workers register through the
Kubernetes API, with TCP and ZMQ request and event planes — so neither is required. Upstream's
platform chart declares them as conditional subcharts that default to `install: false`; this
chart drops the dependency list entirely, which is what lets Terraform install it straight from
the local `charts/` directory with no `helm dependency build` and no reachable chart repository.

Six CRDs are registered, all in group `nvidia.com`:

    kubectl get crd -o name | grep nvidia.com

| Kind | Short | Stored version | What it is |
| --- | --- | --- | --- |
| `DynamoGraphDeployment` | `dgd` | `v1beta1` | The deployment you write. This is what [`charts/machine-learning/serving/dynamo`](../../../charts/machine-learning/serving/dynamo/) renders. |
| `DynamoComponentDeployment` | `dcd` | `v1beta1` | Per-component child created by the operator, one per frontend/prefill/decode. Its name carries a hash of the component spec — see the note on stranded generations in the [vLLM example](vllm/qwen3-8b/README.md#notes). |
| `DynamoGraphDeploymentRequest` | `dgdr` | `v1beta1` | SLA-driven alternative: state a model and latency targets, and the controller profiles the hardware and picks the topology for you. Not used by these examples, which specify topologies directly. |
| `DynamoGraphDeploymentScalingAdapter` | `dgdsa` | `v1beta1` | Implements the Kubernetes scale subresource per component, so HPA or KEDA can drive replicas without racing the operator for ownership of the DGD. |
| `DynamoModel` | `dm` | `v1alpha1` | Model registry entry. |
| `DynamoWorkerMetadata` | `dwm` | `v1alpha1` | Discovery metadata for a worker pod. This is the Kubernetes-native discovery that replaces etcd. |

The first four also serve `v1alpha1` through the conversion webhook, but `v1beta1` is the stored
version and the two shapes differ materially — in `v1alpha1` components are a map under
`spec.services`, in `v1beta1` a list under `spec.components`. Copying an upstream `v1alpha1`
example verbatim gets rejected by the admission webhook.

The CRDs do **not** come from a Helm `crds/` directory. The chart deliberately has none; a
`crd-apply` initContainer server-side-applies the schemas baked into the operator image instead.
`nvidia.com_dynamographdeployments` alone is ~1.6 MB of generated schema, which exceeds the
262144-byte ceiling on the annotation that client-side apply writes, and Helm's `crds/` is
install-only and never upgraded.

## Picking a topology

Every example offers aggregated and disaggregated variants as separate values files, because
which one you want is a real decision rather than a default:

- **Aggregated** — prefill and decode in one process. Simplest, and at 8B on A10G-class GPUs it
  won both throughput and time-to-first-token at a fixed GPU budget.
- **Disaggregated** — prefill and decode as separate workers exchanging the KV cache over NIXL.
  Its win is decode smoothness under mixed load, where it held p99 inter-token latency
  dramatically lower.

[vllm/qwen3-8b/bench/README.md](vllm/qwen3-8b/bench/README.md) has the method and the
findings. The short version: budget **144 KiB of interconnect per prompt token**, and check
whether your instance family actually has a GPU fabric — `g5`, `g6` and `g6e` have no NVLink,
so peer transfers fall back to PCIe.

Moving between topologies requires `helm uninstall` and reinstall. The admission webhook rejects
any change to the set of component names on an existing `DynamoGraphDeployment` with
`component topology is immutable and cannot be modified after creation`.
