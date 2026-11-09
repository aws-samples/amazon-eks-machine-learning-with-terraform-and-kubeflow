"""vLLM's OpenAI-compatible server, as the load driver sees it.

A *move*, not a design: every path and field below was previously spelled in `workload.py` or
`__main__.py`, and the comments came with them.

This used to map four of vLLM's metric names onto the harness's own vocabulary as well. That half is
gone (impl §14.6): nothing reads a serving engine's account of its own internals, so there is no
mapping to supply and no per-build verification of metric names to keep current. What is left is three
things about *talking to* this server, all of which fail visibly at the first request if they are
wrong.
"""

from __future__ import annotations

from . import Engine, register


def endpoint(image_count: int) -> str:
    """`/v1/chat/completions` for an image request, `/v1/completions` otherwise.

    Images have no route through the completions endpoint -- there is no field to put them in -- so a
    multimodal tenant is driven through chat, template and all. A text tenant stays on completions,
    where there is no template between the stated prompt and the tokens the engine sees.
    """
    return "/v1/chat/completions" if image_count else "/v1/completions"


def pin_output(osl_tokens: int) -> dict:
    """`max_tokens` + `min_tokens` + `ignore_eos`, which is what makes OSL an input.

    Without all three, a model that decides to stop early turns output length into a variable the
    harness does not control (impl §6.2, design §8.5).
    """
    return {"max_tokens": int(osl_tokens), "min_tokens": int(osl_tokens), "ignore_eos": True}


def stream_fields() -> dict:
    return {
        # Required, not a preference: inter-token latency cannot be computed without per-token
        # arrival times (impl §6.3).
        "stream": True,
        # The server's own completion_tokens count. The delivered-fraction gate must not rest on the
        # client's chunk observations -- see load.RequestResult.tokens for what that cost once.
        "stream_options": {"include_usage": True},
    }


ENGINE = register(Engine(
    name="vllm-openai",
    health_path="/health",
    endpoint=endpoint,
    pin_output=pin_output,
    pins_output=True,
    stream_fields=stream_fields,
))
