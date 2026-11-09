"""Triton Inference Server running the vLLM backend, as the load driver sees it.

Driven through Triton's OpenAI-compatible frontend, so the request bodies are the SAME bodies the
`vllm-openai` adapter sends. That is the whole reason this file is short: the two stacks wrap one
engine, and if the driver spoke a different protocol to each of them, a measured difference between
them would include the difference between the two protocols and nothing could be attributed.

Three things genuinely differ, and each one is a fact about the frontend rather than a preference:

  1. **Readiness is at `/health/ready`,** not `/health`. Triton's frontend follows the KServe naming
     even when its KServe ports are not open.
  2. **Images are refused, loudly.** See `endpoint`.
  3. Nothing else. `pin_output` is identical, and that was checked in the image rather than assumed --
     see the note on `pin_output`.
"""

from __future__ import annotations

from . import Engine, register


def endpoint(image_count: int) -> str:
    """`/v1/completions`. An image request is refused rather than routed.

    This frontend cannot serve images at all, whatever the engine behind it can do. Its chat handler
    walks the content parts of a message and raises on anything that is not text
    (`engine/utils/chat.py::_parse_chat_message_content_parts`: "only text message is supported, but
    got image_url"), so `/v1/chat/completions` is not an alternative route -- it is the route that
    rejects them. Checked on 26.07 and 26.08; it is not a version accident.

    The vLLM backend underneath does support images, but only through Triton's own protocol, as a
    base64 `image` input tensor that the backend turns into `multi_modal_data`. Reaching that means
    leaving the OpenAI frontend, i.e. sending this stack a different request shape than its sibling
    gets, which would forfeit the comparison the two charts exist for. So the honest answer is that
    this adapter does not serve image tenants, and a fleet that needs one gives that tenant an engine
    that can.

    Raising is deliberate. Returning the chat path would produce a run that issues requests, collects
    400s, and reports a latency distribution over failures.
    """
    if image_count:
        raise ValueError(
            f"this tenant's workload carries {image_count} image(s) per request, and "
            "vllm-triton-inference-server cannot serve images: Triton's OpenAI-compatible frontend "
            "rejects any non-text content part, so both /v1/completions and /v1/chat/completions are "
            "closed to them. Give this tenant an engine that serves images (vllm-openai does) -- a "
            "fleet may state `engine:` per tenant precisely so that one multimodal tenant does not "
            "decide the stack for the whole fleet."
        )
    return "/v1/completions"


def pin_output(osl_tokens: int) -> dict:
    """`max_tokens` + `min_tokens` + `ignore_eos`, exactly as on the sibling.

    Not assumed -- read out of the image, because a field that is silently dropped rather than
    forwarded would turn OSL from an input into an unmeasured variable and nothing downstream would
    notice. In `nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3`:

      - `min_tokens` and `ignore_eos` are declared fields of the frontend's own request models
        (`schemas/openai.py`, both the completion and the chat request), so they are not extras.
      - those models are `extra="forbid"`, so a field this frontend does not know is a 400 rather
        than a silent drop. That is what makes the check above conclusive: if a future version drops
        these fields, the run fails at the first request instead of measuring an unpinned length.
      - neither field is excluded when the frontend builds the backend request
        (`engine/utils/triton.py::_create_vllm_generate_request`), so both arrive in
        `sampling_parameters` and reach the same vLLM sampler the sibling reaches.
    """
    return {"max_tokens": int(osl_tokens), "min_tokens": int(osl_tokens), "ignore_eos": True}


def stream_fields() -> dict:
    return {
        # Required, not a preference: inter-token latency cannot be computed without per-token
        # arrival times (impl §6.3). On this stack streaming is also why the generated config.pbtxt
        # sets Triton's decoupled transaction policy -- one request, many responses.
        "stream": True,
        # The server's own completion_tokens count. The delivered-fraction gate must not rest on the
        # client's chunk observations -- see load.RequestResult.tokens for what that cost once.
        "stream_options": {"include_usage": True},
    }


ENGINE = register(Engine(
    name="vllm-triton-inference-server",
    # KServe's name for it, used by this frontend even though `--enable-kserve-frontends` is off and
    # the KServe ports are never opened. `/health` is not an alias: it 404s.
    health_path="/health/ready",
    # 200 here means the server is live AND every model in the repository is loaded, which for a
    # one-model repository is the only thing we need it to mean. So the default body check stands:
    # the response carries no body to read.
    endpoint=endpoint,
    pin_output=pin_output,
    pins_output=True,
    stream_fields=stream_fields,
))
