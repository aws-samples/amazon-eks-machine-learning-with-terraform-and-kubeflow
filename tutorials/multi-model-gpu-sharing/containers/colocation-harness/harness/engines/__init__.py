"""One serving engine behind one name, load-driver side (impl §13).

The agent's half of this seam (`colocation.engines`) answers *what to deploy*. This half answers
*how to talk to it*, and it is three questions:

  1. **Where a request goes** and what the body looks like -- a path and a shape, not a protocol. An
     OpenAI-compatible server and Triton's generate endpoint disagree about both.
  2. **How output length is pinned.** This one is load-bearing rather than cosmetic. OSL is an
     *input* to every measurement here, and it is an input only because `min_tokens` + `ignore_eos`
     make it one. An engine that cannot pin it does not get a hidden approximation: it reports that
     it cannot, and the run records the achieved distribution instead. Then the comparison gets
     weaker and stays honest, which is the only acceptable direction.
  3. **How to tell it is ready** -- a path, and whether a 200 from it means anything.

**There was a fourth, and its removal is the point (impl §14.6).** An engine also had to say what its
runtime gauges were called: the harness reasoned about `requests_waiting` and `kv_usage`, each engine
supplied its own spelling, and an engine that published neither mapped them to nothing. The machinery
was careful -- absent never read as zero -- and it was still the wrong shape twice over. `_mean_kv` had
to be written so an engine with no KV gauge could not *satisfy* a stability threshold it never reported
against; then `too_narrow_reason` was found resting the authority to widen a degree on three of these
gauges before consulting the driver at all. Two recurrences of one mistake, in code written to avoid
it, is an argument for not having the vocabulary rather than for guarding it better. So nothing here
scrapes a server, the questions a run answers are answered from the driver's books and from DCGM, and
an adapter has three things to get right instead of four -- all three of which fail at the first
request if they are wrong, whereas a name mapping failed only under load.

Imports nothing from the rest of the harness: `config`, `controller`, `observation` and `workload` all
read this, and a cycle through any of them would be a cycle through all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

DEFAULT_ENGINE = "vllm-openai"


@dataclass(frozen=True)
class Engine:
    """One serving stack, as the three questions above."""

    name: str
    # Readiness. `ready_ok` gets the parsed JSON body (or None if it was not JSON) and says whether
    # a 200 actually means serving -- some stacks answer 200 on a liveness path while still loading.
    health_path: str = "/health"
    ready_ok: Callable[[object], bool] = lambda _body: True
    # (image_count) -> request path.
    endpoint: Callable[[int], str] = lambda _n: "/v1/completions"
    # (osl_tokens) -> the fields that fix output length, or `{}` from an engine that cannot fix it.
    pin_output: Callable[[int], dict] = lambda _n: {}
    # Whether `pin_output` actually pins. False makes OSL an observation rather than an input, and
    # the record says so instead of quietly reporting a number nobody controlled.
    pins_output: bool = True
    # The fields that turn on token-by-token streaming plus the server's own usage counts. Streaming
    # is required, not preferred: ITL cannot be computed without per-token arrival times (impl §6.3).
    stream_fields: Callable[[], dict] = lambda: {"stream": True}


_REGISTRY: dict[str, Engine] = {}


def register(engine: Engine) -> Engine:
    if engine.name in _REGISTRY:
        raise ValueError(f"engine {engine.name!r} is already registered")
    _REGISTRY[engine.name] = engine
    return engine


def get(name: str | None) -> Engine:
    """The engine of that name. None or empty means the default, for a run config written before
    the field existed -- which describes a `vllm-openai` run, because that was the only engine."""
    if not name:
        return _REGISTRY[DEFAULT_ENGINE]
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"the run config names serving engine {name!r}, which this harness image has no adapter "
            f"for. It knows {', '.join(sorted(_REGISTRY))}. Refusing rather than falling back: a "
            "fallback would drive one stack's request format at another's server, and the most "
            "likely outcome is a run that completes, reports latency, and measured something else."
        ) from None


def names() -> tuple:
    return tuple(sorted(_REGISTRY))


# `spellings()` and `first_present()` were here: they resolved one canonical signal name against
# whichever engine's spelling a scrape dict happened to carry, for readers holding a dict without the
# engine that produced it. There are no scrape dicts any more (impl §14.6), so there is nothing to
# resolve. Noted rather than silently dropped, because the mechanism was sound and the reason it went
# is not a defect in it.

from . import vllm_openai as _vllm_openai      # noqa: E402,F401  (registers on import)
from . import vllm_triton_inference_server as _vllm_tis      # noqa: E402,F401  (same)
from . import dynamo_vllm as _dynamo_vllm      # noqa: E402,F401  (same)

__all__ = ["DEFAULT_ENGINE", "Engine", "get", "names", "register"]
