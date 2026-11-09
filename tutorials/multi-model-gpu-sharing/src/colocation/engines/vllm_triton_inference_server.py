"""Triton Inference Server with the vLLM backend, served through Triton's OpenAI frontend.

**Why this exists.** The same model, the same weights and the same request format, hosted by a
different server. That is what makes it useful: a delta measured between two stacks is only readable
if everything except the stack is held still, so this adapter deliberately changes as little as it
can. The frontend is Triton's OpenAI-compatible one rather than its native `v2/.../generate_stream`,
so the body the harness sends is the body it sends to `vllm-openai` (see this package's harness-side
twin), and the same frontend is what a later Triton + TensorRT-LLM adapter would use.

**What is genuinely different, and it is one thing.** vLLM's server takes its engine arguments as
CLI flags; Triton's vLLM backend reads them as JSON out of a model repository -- one `model.json`
per model, holding vLLM `AsyncEngineArgs` by their Python names. So a fleet file's `extra_args`,
which are written as flags because that is what the other stack takes, have to be converted. The
conversion happens here, once, in Python, rather than in a Helm template: it has to refuse what it
cannot convert, and a refusal is worth writing in a language that can say why.

Nothing else about a tenant changes. The envelope is still MPS's, the memory budget is still a
fraction of the whole card (the backend passes `gpu_memory_utilization` straight through to the same
vLLM engine), and the placement, the node pin and the UUID-keyed limit are the shared half of the
chart's schema that no serving stack gets a say in.
"""

from __future__ import annotations

import json

from . import Engine, register

# Pinned for the same reason the vLLM image is: the image decides the engine build, and the engine
# build is an input to every memory number measured under it.
#
# `-vllm-python-py3` is the NGC variant that carries the vLLM backend; the plain `-py3` tag does not
# have it. The 26.07 line is not a free choice either -- it must be the same NGC release as the
# mps-daemon's image, because a client that attaches to an MPS server through a *different*
# forward-compatibility libcuda hangs forever with no error (finding 11). 26.06 and 26.08 exist if
# this line ever has to move; the daemon moves with it.
DEFAULT_IMAGE = "nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3"


class FlagConversionError(ValueError):
    """An `extra_args` entry this stack cannot be told about."""


def engine_args_from_flags(extra_args) -> dict:
    """vLLM CLI flags -> the JSON keys the Triton vLLM backend reads from `model.json`.

    Three shapes, and nothing else:

      ``--flag=value``  -> ``{"flag": value}``, the value parsed as JSON when it parses (so
                           ``--mm-processor-cache-gb=0`` is the number 0 and
                           ``--limit-mm-per-prompt={"image": 8}`` is a mapping) and left as a
                           string when it does not (``--dtype=bfloat16``).
      ``--no-flag``     -> ``{"flag": false}``. vLLM spells a negated boolean this way and the
                           fleets in this repo use it, so dropping the ``no-`` and writing `false`
                           is the conversion, not a guess.
      ``--flag``        -> ``{"flag": true}``.

    Anything else is refused. In particular a space-separated pair (``--max-model-len 4096``) is
    refused rather than joined with the next entry: a converter that guessed would turn a typo into
    an engine argument nobody wrote, and an engine argument nobody wrote is a measurement of a
    configuration nobody asked for.
    """
    out: dict = {}
    for raw in extra_args or ():
        flag = str(raw).strip()
        if not flag.startswith("--"):
            raise FlagConversionError(
                f"extra_args entry {flag!r} is not a flag. This stack takes its engine arguments as "
                "JSON, so each entry must be `--flag`, `--no-flag` or `--flag=value` and be "
                "convertible on its own -- a value in a separate entry has nothing to attach to.")
        name, eq, value = flag[2:].partition("=")
        if " " in name:
            raise FlagConversionError(
                f"extra_args entry {flag!r} puts a flag and its value in one entry separated by a "
                "space. Write it as `--flag=value`: this stack has no command line to hand them "
                "to, so the two halves have to be told apart here.")
        key = name.replace("-", "_")
        if eq:
            try:
                parsed = json.loads(value)
            except ValueError:
                parsed = value
        elif key.startswith("no_"):
            key, parsed = key[3:], False
        else:
            parsed = True
        if key in out:
            raise FlagConversionError(
                f"extra_args sets {key!r} twice ({flag!r} is the second). On a command line the last "
                "one wins silently; in a JSON object one of them simply disappears, so say it once.")
        out[key] = parsed
    return out


def values(*, tp: int, util: float, max_num_seqs: int, max_num_batched_tokens: int,
           max_model_len: int, dtype: str, extra_args=()) -> dict:
    """The `triton_vllm:` block of the chart's values.

    One key, `engine_args`, which the chart writes verbatim into each tenant's `model.json`. Names
    are vLLM's Python argument names, which is what the backend reads -- not the CLI spellings, and
    not names of this repo's own invention.

    `model` and `served_model_name` are deliberately absent: they come from the shared half of the
    schema (`model.path`, `model.served_name`), and a second copy here would be a second place for
    the weights path to be wrong.
    """
    args = {
        "tensor_parallel_size": int(tp),
        # The same fraction-of-the-whole-card the other stack takes, for the same reason: under MPS
        # the engine sees the envelope as free memory but the *card* as total, so the budget has to
        # be expressed against the card. Six decimals because `render.check_agreement` recomputes it
        # and compares.
        "gpu_memory_utilization": round(float(util), 6),
        "max_num_seqs": int(max_num_seqs),
        "max_num_batched_tokens": int(max_num_batched_tokens),
        "max_model_len": int(max_model_len),
        "dtype": dtype,
    }
    converted = engine_args_from_flags(extra_args)
    clash = sorted(set(converted) & set(args))
    if clash:
        raise FlagConversionError(
            f"extra_args restates {', '.join(clash)}, which the solver decides. On this stack both "
            "end up in the same JSON object, so the flag would silently replace a computed value -- "
            "a hand-written `gpu_memory_utilization` in particular would hand the engine an envelope "
            "nothing verified. Remove it from extra_args.")
    args.update(converted)
    return {"engine_args": args}


def read_tp(block: dict) -> int:
    return int(block["engine_args"]["tensor_parallel_size"])


def read_util(block: dict) -> float:
    return float(block["engine_args"]["gpu_memory_utilization"])


# Held equal to the other stack's default on purpose. vLLM's own library default is larger, and
# taking it here would mean the two stacks were serving different batch caps at the same stated
# demand -- so a delta between them would be partly a delta between two caps, and nothing in the
# record would separate the two.
DEFAULT_MAX_NUM_SEQS = 64


ENGINE = register(Engine(
    name="vllm-triton-inference-server",
    chart="charts/machine-learning/model-serving/vllm-triton-inference-server",
    chart_release_prefix="vllm-triton-inference-server-",
    default_image=DEFAULT_IMAGE,
    values_key="triton_vllm",
    values=values,
    default_max_num_seqs=DEFAULT_MAX_NUM_SEQS,
    read_tp=read_tp,
    read_util=read_util,
    # Not an omission, and not the backend's limitation: the vLLM backend under this server does
    # serve images, through Triton's own protocol as a base64 `image` input tensor. The frontend is
    # what cannot. Triton's OpenAI-compatible frontend raises on any non-text content part of a chat
    # message ("only text message is supported, but got image_url"), checked in the 26.07 image and
    # again in 26.08, so this is not a version accident to wait out.
    #
    # Reaching the backend's image path would mean sending this stack a different request shape than
    # the sibling gets, and then a measured difference between the two would include the difference
    # between two protocols. Refusing the tenant keeps the comparison; a fleet says `engine:` per
    # tenant so that one multimodal tenant does not decide the stack for the whole fleet.
    serves_images=False,
))
