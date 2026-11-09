"""NVIDIA Dynamo with the vLLM backend, aggregated: one Dynamo frontend and one vLLM worker per tenant.

**Why this exists.** Dynamo is the serving stack whose reason to exist is what this pipeline does not
do -- disaggregated prefill and decode, KV-aware routing across replicas. None of that is used here,
deliberately: a tenant is ONE aggregated worker behind ONE frontend, so the thing measured is what the
Dynamo runtime costs a tenant that the other two stacks do not charge, under the same envelope, at the
same demand. Disaggregation would put a second pod on a second envelope and turn one tenant into two,
which the solver has no way to price; see `values` for the refusal.

**What is genuinely different from the sibling adapters.**

  1. The server is a *graph*. The chart renders a `DynamoGraphDeployment` and the Dynamo operator
     makes the pods, so the readiness gate sees two pods per tenant (the frontend and the worker)
     rather than one. Both carry the tenant labels; the gate needs every pod of a release Ready.
  2. The engine is newer. `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0` carries vLLM 0.28.0 on CUDA
     13.0, where the two NGC 26.07 images carry 0.24.0. So a delta between this stack and the others
     is a delta between two runtimes AND two engine builds, and the record says which image produced
     it. That is recorded rather than hidden: there is no Dynamo build on the 26.07 engine.
  3. The image has no forward-compatibility driver of its own, and under MPS that matters: the client
     must load the libcuda the daemon loaded (finding 11), and the daemon runs on the 26.07 compat
     driver 610.43.02. The chart copies that driver out of the DAEMON's image into the worker pod.
     Measured without it: the attach hangs forever. With it: attached, `cuDriverGetVersion` 13030.

Everything else -- the envelope, the node pin, the UUID-keyed limit, the fraction-of-the-card budget
-- is the shared half of the schema, and the vLLM flags are the sibling's flags, because `dynamo.vllm`
takes vLLM's own engine arguments on its command line.
"""

from __future__ import annotations

from . import Engine, register

# Pinned for the same reason the other two images are: the image decides the engine build, and the
# engine build is an input to every memory number measured under it. 1.5.0 is the release the
# cluster's Dynamo operator runs; a runtime and an operator from different releases are not a
# combination anybody tested.
DEFAULT_IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0"

# Flags `dynamo.vllm` takes that make a tenant something other than one aggregated worker. Each one
# turns the worker into half of a prefill/decode pair, which needs a second pod on a second envelope
# and a KV transfer between them -- a shape the solver cannot place and the store cannot key.
_DISAGGREGATION_FLAGS = ("--disaggregation-mode", "--is-prefill-worker", "--is-decode-worker")

# Set by the chart from the shared half of the schema, so a second copy in extra_args would be a
# second place for the weights path or the model id to be wrong.
_CHART_OWNED_FLAGS = ("--model", "--served-model-name")


class DynamoFlagError(ValueError):
    """An `extra_args` entry this adapter will not pass to a Dynamo worker."""


def _refuse(extra_args) -> None:
    for raw in extra_args or ():
        flag = str(raw).strip().split("=", 1)[0].split(" ", 1)[0]
        if flag in _DISAGGREGATION_FLAGS:
            raise DynamoFlagError(
                f"extra_args entry {raw!r} makes this tenant half of a disaggregated prefill/decode "
                "pair. A dynamo-vllm tenant is one aggregated worker: a second worker would need an "
                "envelope of its own on its own GPUs and a KV transfer between them, and the solver "
                "prices a tenant as one envelope. Remove it.")
        if flag in _CHART_OWNED_FLAGS:
            raise DynamoFlagError(
                f"extra_args entry {raw!r} restates what the chart sets from model.path and "
                "model.served_name. Two places for it means two chances for the weights the worker "
                "loads and the name the frontend routes by to disagree. Remove it.")


def values(*, tp: int, util: float, max_num_seqs: int, max_num_batched_tokens: int,
           max_model_len: int, dtype: str, extra_args=()) -> dict:
    """The `dynamo_vllm:` block of the chart's values.

    The same fields and spellings as the `vllm-openai` block, because they reach the same place: the
    worker's command line, which `dynamo.vllm` parses with vLLM's own argument parser. Held identical
    on purpose, so that the two stacks differ in what serves the flags and not in the flags.
    """
    _refuse(extra_args)
    return {
        "tensor_parallel_size": int(tp),
        # A fraction of the whole card, as on the sibling: under MPS the engine sees the envelope as
        # free memory and the card as total. Six decimals because `render.check_agreement` compares.
        "gpu_memory_utilization": round(float(util), 6),
        "max_num_seqs": int(max_num_seqs),
        "max_num_batched_tokens": int(max_num_batched_tokens),
        "max_model_len": int(max_model_len),
        "dtype": dtype,
        "extra_args": list(extra_args),
    }


def read_tp(block: dict) -> int:
    return int(block["tensor_parallel_size"])


def read_util(block: dict) -> float:
    return float(block["gpu_memory_utilization"])


# Held equal to the sibling's for the Triton adapter's reason: a different default cap would make a
# delta between the stacks partly a delta between two caps, with nothing in the record separating them.
DEFAULT_MAX_NUM_SEQS = 64


ENGINE = register(Engine(
    name="dynamo-vllm",
    chart="charts/machine-learning/model-serving/dynamo-vllm",
    chart_release_prefix="dynamo-vllm-",
    default_image=DEFAULT_IMAGE,
    values_key="dynamo_vllm",
    values=values,
    default_max_num_seqs=DEFAULT_MAX_NUM_SEQS,
    read_tp=read_tp,
    read_util=read_util,
    # The worker would take an image -- it is vLLM -- but nothing about it has been checked on this
    # stack: not the frontend's handling of image content parts, not the multimodal processor path
    # `dynamo.vllm` builds, not whether the frontend's own tokenizer agrees with the engine about the
    # placeholder tokens. A capability nobody measured is not one to claim, so an image tenant is
    # refused until someone does.
    serves_images=False,
    # The frontend and the worker, both carrying the tenant labels.
    pods_per_tenant=2,
    # The operator rolls an updated worker out new-before-old (`status.rollingUpdate`), which under
    # MPS is two envelopes on one GPU for as long as the overlap lasts. So a changed tenant is taken
    # down and seen to be gone before it comes back.
    replace_on_change=True,
    requires_crds=("dynamographdeployments.nvidia.com",),
    # The blueprint's dynamo-platform release, in `dynamo-system`.
    requires_controller="app.kubernetes.io/name=dynamo-platform,control-plane=controller-manager",
))
