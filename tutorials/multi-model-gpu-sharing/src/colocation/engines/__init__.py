"""One serving engine behind one name (impl §13).

**What this is for.** Everything else in this package is engine-agnostic by construction -- the
solver reasons about memory and GPUs, the store files measurements, the profiler judges degrees --
and exactly six places assumed vLLM's OpenAI server. This module is where those assumptions go, so
that adding Triton + vLLM, or Triton + TensorRT-LLM, is a new file rather than a search for
`"vllm"` across a codebase that has 972 self-tests resting on the current answers.

**The seam is deliberately narrow, and its boundary is a question, not a layer.** The question is:
*would a different serving stack answer this differently?* The chart to install, the image to pull,
the words the engine's own values block uses, and how a memory budget is expressed to it -- yes. How
many GPUs a tenant needs, whether a latency row may be cited, what a validity key hashes -- no, and
those stay out. A thing that ends up here because it was convenient rather than because the answer
differs is a thing that will have to be re-derived per engine for no reason.

**What used to be listed here as missing, and is now moot.** The start-up log patterns were vLLM's
regexes over an engine's own memory report, and the gap was that nobody had the second engine's log
to write the Triton equivalent from. Impl §14.4 deleted the patterns along with the mode that used
them, so there is nothing to port: an engine's account of its own internals is not evidence this
design reads, on either engine. What a second engine has to supply instead is what a *driven run*
observes -- which is what the rest of this module is for.

**Nothing in this module imports from the rest of the package.** Both `colocation.store` and
`colocation.agent` read it, and the store's ability to read and judge a record with no cluster and
no OR-Tools is a property the self-test asserts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# The engine a fleet file gets when it does not say. It is vLLM's OpenAI server because that is the
# stack every measurement in the store was taken on, and a default that renamed the existing
# condition would silently re-key history. See `store/serving.py` for why the name is omitted from a
# digest when it equals this.
DEFAULT_ENGINE = "vllm-openai"


@dataclass(frozen=True)
class Engine:
    """One serving stack, as the four questions only it can answer.

    Frozen because a registry entry names a chart that gets installed and an image that gets pulled;
    a mutable one would let a caller change what a later caller deploys.
    """

    # The name a fleet file states, and the name that travels into a solo row's identity. Kept to
    # RFC-1123 characters because it appears in a Helm release and in a store path.
    name: str
    # Helm chart, relative to the blueprint checkout. `agent/deploy.py` joins it to `--chart-dir`.
    chart: str
    # What an installed release's `chart` field starts with, which is how `deploy.py` recognises a
    # release of ours from `helm list` output. Helm appends the chart version, so this is a prefix.
    chart_release_prefix: str
    # The container image a fleet file gets when it does not state one.
    default_image: str
    # The key in the chart's values under which this engine's own flags sit. vLLM's chart calls it
    # `vllm`; a Triton chart will not.
    values_key: str
    # (tp, util, max_num_seqs, max_num_batched_tokens, max_model_len, dtype, extra_args) -> dict.
    # THE one place that knows how this engine is told its shard count and its memory budget.
    values: Callable[..., dict]
    # The batch cap this engine gives a tenant that asks for no particular one, and therefore the
    # floor under the derived cap (`spec.TenantTemplate.max_num_seqs`). It belongs here and not in the
    # solver for the same reason the image does: it is what *this* stack does when nobody says, it is
    # already baked into every stored row taken at the default, and a second engine's default is its
    # own business. Deliberately a floor and never a ceiling -- the derivation may raise it, and a cap
    # below the offered burst diverges rather than degrades.
    default_max_num_seqs: int
    # values-sub-dict -> int. The inverse of `values` for the two fields `render.check_agreement`
    # re-derives, and it must be the inverse: that check exists because DCGM cannot see through MPS,
    # so a tenant's envelope is only ever verifiable against what was written into its own values.
    read_tp: Callable[[dict], int]
    read_util: Callable[[dict], float]
    # Whether this stack can serve a request that carries an image at all.
    #
    # Here, rather than in the harness alone, because the consequence is a fleet file that cannot be
    # run and the fleet file is read offline. The harness-side adapter also refuses -- it has to, as
    # the backstop for a run configured some other way -- but that refusal arrives after a node is
    # leased, a daemon is installed and weights are pulled off FSx. This one arrives before `plan`
    # prints anything.
    #
    # A capability and not a preference, which is why it is a plain bool: the one stack that says
    # False says it because its frontend rejects every non-text content part, so there is no request
    # the harness could send instead. Default True because the question only came up when the second
    # engine arrived, and the first one serves images.
    serves_images: bool = True
    # How many pods one tenant is, counted by the tenant labels. The readiness gate needs every one of
    # them Ready and needs to know how many there are: a stack whose frontend comes up before its
    # worker has even been created would otherwise pass the gate with one pod out of two.
    pods_per_tenant: int = 1
    # Whether a release whose values changed must be uninstalled, and seen to be gone, before it is
    # installed again -- rather than upgraded in place. True for a stack whose controller rolls an
    # update out by starting the new server BEFORE stopping the old one: under MPS the newcomer's
    # envelope is claimed while the outgoing one still holds its own, which double-books the GPU. A
    # Deployment with `Recreate` does not do that, so the default is to upgrade.
    replace_on_change: bool = False
    # What must exist in the cluster before a tenant of this stack can come up at all: CRD names, and a
    # label selector (all namespaces) for the controller that acts on them. Checked before a node is
    # leased. A missing CRD fails the first install loudly; a missing controller fails nothing -- the
    # custom resource is accepted and no pod ever appears -- so without this the run would learn it
    # thirty minutes later, from a readiness timeout, with a GPU node billing.
    requires_crds: tuple = ()
    requires_controller: str = ""


_REGISTRY: dict[str, Engine] = {}


def register(engine: Engine) -> Engine:
    if engine.name in _REGISTRY:
        raise ValueError(f"engine {engine.name!r} is already registered")
    _REGISTRY[engine.name] = engine
    return engine


def get(name: str) -> Engine:
    """The engine of that name, or a refusal that lists the ones there are.

    A `KeyError` with the alternatives in it rather than a bare one, because the caller is a fleet
    file's `engine:` line and the person who wrote it wants to know what they may write instead.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no serving engine named {name!r}. This fleet's `engine:` must name one of "
            f"{', '.join(sorted(_REGISTRY))}. An engine is a chart plus an image plus a request "
            "format plus a metrics vocabulary, and there is no generic fallback: a fleet served by "
            "an engine nobody wrote an adapter for would deploy, measure, and be filed as evidence "
            "about a condition the store cannot name.") from None


def names() -> tuple:
    return tuple(sorted(_REGISTRY))


from . import vllm_openai as _vllm_openai      # noqa: E402,F401  (registers on import)
from . import vllm_triton_inference_server as _vllm_tis      # noqa: E402,F401  (same)
from . import dynamo_vllm as _dynamo_vllm      # noqa: E402,F401  (same)

__all__ = ["DEFAULT_ENGINE", "Engine", "get", "names", "register"]
