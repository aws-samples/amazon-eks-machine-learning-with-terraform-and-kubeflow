"""NVIDIA Dynamo with the vLLM backend, as the load driver sees it: one Dynamo frontend per tenant.

Driven through the frontend's OpenAI-compatible API with the SAME bodies the two sibling adapters
send, for the reason the Triton adapter gives: a difference measured between stacks is only readable
if the request is not part of it.

Two things differ, and both are facts about the frontend:

  1. **Readiness is "my model is listed", not a health path.** A Dynamo frontend is its own process,
     separate from the worker that serves the model, and it answers `/health` and `/v1/models` with a
     200 before any worker has registered. So a 200 means nothing here; the body has to name a model.
  2. **Images are refused**, because nothing about image requests has been checked on this stack. See
     `endpoint`.
"""

from __future__ import annotations

from . import Engine, register


def endpoint(image_count: int) -> str:
    """`/v1/completions`. An image request is refused rather than routed.

    Not because the frontend is known to reject one -- it is not known either way. The agent-side
    adapter declares `serves_images=False` for the same reason: the frontend's handling of image
    content parts, the multimodal path `dynamo.vllm` builds and the frontend's own tokenization of the
    placeholders were never measured here, and a run against an unmeasured path would be filed as
    evidence about it.
    """
    if image_count:
        raise ValueError(
            f"this tenant's workload carries {image_count} image(s) per request, and dynamo-vllm "
            "does not serve image tenants: its image path has not been checked on this stack. Give "
            "this tenant an engine that serves images (vllm-openai does) -- a fleet may state "
            "`engine:` per tenant for exactly this.")
    return "/v1/completions"


def pin_output(osl_tokens: int) -> dict:
    """`max_tokens` + `min_tokens` + `ignore_eos`, exactly as on the siblings.

    Checked on the first boot (2026-09-29, Dynamo 1.5.0, Qwen3-0.6B and Qwen3-14B) that the frontend
    passes both through to the worker's sampler rather than dropping them: a prompt the model ends
    after 2 tokens unpinned came back with `completion_tokens` equal to `max_tokens` (128), finish
    reason `length`, under this pin -- and under each of the two fields alone.
    """
    return {"max_tokens": int(osl_tokens), "min_tokens": int(osl_tokens), "ignore_eos": True}


def stream_fields() -> dict:
    return {
        "stream": True,
        # The server's own completion_tokens count, for the delivered-fraction gate.
        "stream_options": {"include_usage": True},
    }


def ready_ok(body) -> bool:
    """A 200 from `/v1/models` means serving only if it lists a model.

    One model per frontend, because each tenant is its own graph with its own discovery namespace, so
    any listed model is this tenant's. An empty list is a frontend whose worker is still loading or
    has died, and it answers 200 all the same.
    """
    return isinstance(body, dict) and bool(body.get("data"))


ENGINE = register(Engine(
    name="dynamo-vllm",
    health_path="/v1/models",
    ready_ok=ready_ok,
    endpoint=endpoint,
    pin_output=pin_output,
    pins_output=True,
    stream_fields=stream_fields,
))
