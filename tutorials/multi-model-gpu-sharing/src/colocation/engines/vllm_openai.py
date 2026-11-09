"""vLLM's own OpenAI-compatible server, which is the stack every stored measurement was taken on.

This file is a *move*, not a design. Every value and every comment in it was lifted from the place
that previously hard-coded it -- `render.render_values`, `spec.DEFAULT_IMAGE`, `deploy.DEFAULT_CHART`
-- and the render snapshot (`tests/snapshot.py`) is what says the move changed nothing. If a number
here looks arbitrary it is because it was arbitrary before, and the honest place to argue with it is
the fleet file that overrides it.
"""

from __future__ import annotations

from . import Engine, register

# Pinned, not floating. The image decides the engine version, the engine version decides what a
# start-up log says and how much memory a graph capture takes, and both of those are inputs to
# measurements already in the store -- so it is part of the hardware validity key
# (`store/hwfacts.py`), and moving it invalidates memory facts rather than merely upgrading a
# dependency.
DEFAULT_IMAGE = "nvcr.io/nvidia/vllm:26.07-py3"


def values(*, tp: int, util: float, max_num_seqs: int, max_num_batched_tokens: int,
           max_model_len: int, dtype: str, extra_args=()) -> dict:
    """The `vllm:` block of the chart's values. THE one place that knows this engine's vocabulary.

    `gpu_memory_utilization` is this engine's way of being told a memory budget, and it is a
    *fraction of the whole card* rather than an absolute -- which is why the envelope has to be
    divided by the device total before it gets here (`solver/memory.gpu_memory_utilization`), and why
    a rounded envelope and an unrounded utilisation are a real failure rather than a cosmetic one
    (`render._envelope_gib_int`). An engine that takes absolute bytes will not need that division,
    and that is the first thing a second adapter changes.
    """
    return {
        "tensor_parallel_size": int(tp),
        # Six decimals, matching what was rendered before this moved: the fraction is compared
        # against a recomputed one by `render.check_agreement`, so its precision is load-bearing.
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


# The `vllm-openai` chart's own default for `max_num_seqs`, and the number every stored row taken
# before impl §14.9 was measured at -- not vLLM's library default, which is larger and which this
# chart does not use. It is the floor under the derived cap and the reason most tenants keep their
# rows across §14.9: a tenant whose burst arithmetic asks for less than this is served at this, which
# is what it was already being served at.
DEFAULT_MAX_NUM_SEQS = 64


ENGINE = register(Engine(
    name="vllm-openai",
    chart="charts/machine-learning/model-serving/vllm-openai",
    chart_release_prefix="vllm-openai-",
    default_image=DEFAULT_IMAGE,
    values_key="vllm",
    values=values,
    default_max_num_seqs=DEFAULT_MAX_NUM_SEQS,
    read_tp=read_tp,
    read_util=read_util,
))
