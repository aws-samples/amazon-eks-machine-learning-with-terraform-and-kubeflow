"""What the search is over, and what it holds fixed (impl §11).

The agent is not a solution for one fleet. It takes a `FleetSpec` -- a set of tenants and a
workload -- and searches the placements of *that* set. Nothing below names a model, a node, or
a rate; those all arrive as arguments. This module exists so that the distinction is a type
rather than a convention.

The split is the point:

  - `TenantTemplate` holds everything the search does **not** vary: which model, at what TP
    degree, offered at what rate, in what request shape, with which serving limits, and what it
    must achieve. Two of these are decision variables *for the solver* and therefore absent: the
    GPU assignment and the memory envelope.
  - `NodeInventory` holds the GPU ordinal -> UUID mapping, which is node-specific and must be
    read from the live node rather than written down (impl §7). A UUID typo does not fail; the
    MPS limit is silently ignored (finding 9).
  - `WorkloadSpec` holds what is genuinely fleet-wide and nothing else, which is now only the
    delivery gate. The tokenizer used to live here and does not: see its docstring.
  - `TenantSlo` holds what the tenant must achieve, which is **not** part of the validity key.

**Rate and sequence shape are per tenant, and that is not a generalization for its own sake.**
Design §8.3: a fleet whose members shared one rate and one request shape would be a fleet of one
workload wearing several models' names, and the whole problem this design addresses is a mix. A
summarizer offering 8k in and 200 out is not the same workload as a chat model offering 300 in
and 600 out, and a solver told they are identical is solving a fiction. Both are in the validity
key, because both change what the hardware does (§4.1).

`TenantTemplate.request()` is the only bridge to the solver, and it is deliberately narrow --
the solver sees seven facts per tenant and nothing about Helm, images, or paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from statistics import NormalDist

from ..solver.cpsat import NodeSpec, TenantRequest
from .. import engines
from ..solver.memory import ModelMemory
from ..store.serving import ServingConfig

# The image every measurement on this stack is taken through. Part of the validity key, so it
# lives here as a default rather than being hardcoded in the renderer: a run under a different
# image is a different key, and that has to be expressible.
#
# Now read off the default engine's adapter rather than spelled here, because an image belongs to an
# engine -- `nvcr.io/nvidia/vllm` is not a sensible default for a Triton fleet, and a fleet that
# names an engine without naming an image must get *that* engine's image, not this constant. Kept
# under this name because it is imported in five places and is the right default for the default
# engine; `engine_for(tenant).default_image` is what a non-default engine resolves through
# (`fleetfile._tenant_image`).
DEFAULT_IMAGE = engines.get(engines.DEFAULT_ENGINE).default_image

# The agent solves on whole-GiB envelopes. The chart's MPS limit is rendered as an integer
# `<N>G` (see the `mpsMemLimit` helper), so a 12.5 GiB envelope would be enforced at 12 GiB
# while `gpu_memory_utilization` was computed for 12.5 -- the engine would then over-ask by
# half a GiB and die in sampler warmup, exactly the failure impl §4.2's second correction
# describes. Rather than round silently at render time, the search never proposes a
# non-integral envelope. Recovering the solver's 512 MiB resolution means teaching the chart
# to emit the limit in MiB, which is a chart change with a measurable consequence and so is
# not made incidentally.
AGENT_ENVELOPE_STEP_MIB = 1024

# How far past a stated ceiling a placement may be and still be worth reporting as a RELAXED
# reading, when the tenant does not say. One number in the codebase: `TenantSlo.tolerance_pct`
# overrides it per tenant and `--relax-targets-pct` can only lower it.
DEFAULT_TOLERANCE_PCT = 5.0

# Which arrival burst a tenant's KV pool and batch cap are sized for, when the fleet file does not
# say (impl §14.9). Arrivals at a stated mean rate are not evenly spaced, and a pool sized for the
# mean is a pool that preempts whenever they clump -- which the engine reports as an ITL the tenant
# then misses its target by, with nothing in the record able to say the pool was the cause.
#
# It is a **policy** number and the only new input Fix 3 adds: how much of the arrival distribution
# the fleet wants to hold without preempting. Not hardware, not a model property, and nothing here
# fits it. A fleet that knows its own bursts better states `peak_rate_hz` instead and this is not
# consulted at all.
BURST_QUANTILE_DEFAULT = 0.99

# The minimum viable KV pool, in whole requests at the engine's own context window. Two, because one
# is not a pool: a pool that can hold a single `max_model_len` request serves it and then serves
# nothing else until it retires, so the tenant's ITL is whatever a serial queue gives and no target
# is reachable at any rate. Two is also the smallest number that can be a *floor* rather than a
# prediction -- it makes no claim about how many requests will actually be in flight, which is what
# the burst arithmetic above is for.
MIN_POOL_REQUESTS = 2


@dataclass(frozen=True)
class TenantSlo:
    """What one tenant must achieve. An *input* to the search (design §2), not an output.

    Design §2 states the problem as "given a node, a set of models, and per-model SLOs, produce
    a configuration that serves all of them", and §4.3 lists SLO targets among the solver's
    constraints. So a target is something the fleet declares, in the same file as the rate it
    declares -- not a threshold discovered afterwards by looking at what the hardware happened
    to do.

    **Targets are deliberately not part of the validity key.** The key answers one question:
    may these two measurements be compared? A target does not bear on that. The same placement,
    at the same rate, on the same image and driver, produces the same latency whatever anybody
    was hoping for. Putting a target in the key would mean tightening a threshold invalidated a
    store full of perfectly good measurements and forced a re-measure to learn nothing. The
    offered *rate* is in the key and must stay there, because it changes what the hardware does.
    So a tenant's SLO is split across two places on purpose: `rate_hz` is a condition of
    measurement, and this is a criterion for judging it.

    The same reasoning says records store raw metrics and are judged when read, never stamped
    with a met/missed verdict when written.

    **Both ceilings are optional and there is no default.** A tenant that states no target has
    no target; nothing here invents one, because an invented threshold is a decision wearing a
    default's clothes, and it would later be reported as though somebody had asked for it. What
    follows from an absent target is that no measurement can certify that tenant -- which is a
    true and useful thing for the search to have to say.

    `ttft_p95_ms` is a p95 and `itl_mean_ms` is a mean, and the asymmetry is deliberate.
    Design §8.1: a vLLM scheduler step that carries a prefill emits no token for anyone in the
    batch, so an ITL p95 measures how often prefills land rather than the quality of decode.

    **`tolerance_pct` is how far past its ceilings this tenant is willing to be reported as
    RELAXED, and it belongs here for the same reason the ceilings do.** A concession stated in
    advance by whoever owns the requirement is a policy; the same concession discovered after
    the run, by reading how far the measurement missed, is a rationalisation. Those are not the
    same fact and the report must not be able to confuse them, so the number is declared in the
    fleet file next to the target it applies to -- not chosen at the command line once the
    result is known. `--relax-targets-pct` can only *lower* what is declared here, never raise
    it; a flag that could raise it would reopen exactly the after-the-fact loosening this field
    exists to close.

    It is a tolerance and never a target. It is not folded into the ceilings, because a
    tolerance that moved the ceiling would silently become the ceiling -- STRICT is always
    computed and reported against the unrelaxed numbers, and RELAXED is reported beside it, so
    both are on screen at once. Like the ceilings, it is not in the validity key: it changes how
    a measurement is judged, not what was measured.
    """

    ttft_p95_ms: float | None = None
    itl_mean_ms: float | None = None
    tolerance_pct: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("ttft_p95_ms", "itl_mean_ms"):
            value = getattr(self, field_name)
            if value is not None and not value > 0:
                raise ValueError(f"{field_name} must be > 0 when given; got {value!r}")
        if self.tolerance_pct is not None and not 0.0 <= self.tolerance_pct <= 100.0:
            # 0 is meaningful and allowed: "this ceiling does not move", which is how a tenant
            # opts out of ever being reported as RELAXED. Above 100% the reading stops being a
            # concession and becomes a different requirement.
            raise ValueError("tolerance_pct must be between 0 and 100 when given; got "
                             f"{self.tolerance_pct!r}")
        if self.ttft_p95_ms is None and self.itl_mean_ms is None:
            # An `slo:` block with nothing in it is a typo, not a statement of indifference.
            # Indifference is expressed by omitting the block, which yields `slo=None` and is
            # distinguishable from a stated target downstream.
            raise ValueError(
                "an slo block must state at least one of ttft_p95_ms or itl_mean_ms. To say a "
                "tenant has no target, omit the block entirely -- 'no target stated' and "
                "'target stated as nothing' have to stay distinguishable.")

    @property
    def stated(self) -> tuple:
        """The names of the targets this actually carries. For reports and briefs."""
        return tuple(n for n in ("ttft_p95_ms", "itl_mean_ms") if getattr(self, n) is not None)

    @property
    def tolerance_is_declared(self) -> bool:
        """Whether the tolerance below is this tenant's own statement or the fallback.

        Reported, because "chat accepts 1%" and "nobody said, so 5%" support different claims and
        a reader cannot tell them apart from the number alone.
        """
        return self.tolerance_pct is not None

    def tolerance(self, override_pct: float | None = None) -> float:
        """The effective tolerance in percent: declared, or the default, lowered by an override.

        `min` and not the override, deliberately. `--relax-targets-pct` is a way to ask a narrower
        question than the fleet file allows -- "what holds within 1%?" -- and never a way to widen
        what it allows.
        """
        declared = DEFAULT_TOLERANCE_PCT if self.tolerance_pct is None else self.tolerance_pct
        return declared if override_pct is None else min(declared, max(float(override_pct), 0.0))

    def cap_factor(self, override_pct: float | None = None) -> float:
        """`tolerance` as the multiplier a measured/ceiling ratio is compared against."""
        return 1.0 + self.tolerance(override_pct) / 100.0


@dataclass(frozen=True)
class TenantTemplate:
    """One tenant, minus the two things the solver decides.

    `name` is the Helm release name and the tenant's identity in every record, so it must be
    RFC-1123 and stable across runs: the store keys per-tenant results by it, and renaming a
    tenant mid-search silently splits its history.
    """

    name: str
    model_id: str                 # catalog identity, e.g. Qwen/Qwen3-0.6B
    model_path: str               # where the weights actually are, e.g. /fsx/pretrained-models/...
    served_name: str              # --served-model-name; what the harness asks for
    tp: int
    rate_hz: float
    # This tenant's request shape. Required, with no default, for the same reason `rate_hz` has
    # none: a shape the loader invented would be reported afterwards as though somebody had
    # asked for it, and it decides both the KV arithmetic (§1.7) and what the harness sends.
    isl_tokens: int
    osl_tokens: int
    mem: ModelMemory
    slo: TenantSlo | None = None   # None means no target was stated. Never defaulted.

    # The arrival burst this tenant is sized for, and the two ways of stating it. Both optional and
    # both about the *offered load*, which is the only thing a fleet file may still say about a KV
    # floor or a batch cap -- the floor and the cap themselves are derived (`min_kv_tokens`,
    # `max_num_seqs`).
    #
    # `peak_rate_hz` is a rate this tenant is known to reach in bursts, and stating it replaces the
    # arrival model entirely: the sizing arithmetic runs at that rate instead of inflating the mean.
    # For a tenant behind a rate limiter, or one whose traffic is a batch job with a known shape, that
    # is a better statement than any quantile of a distribution it does not have.
    # `burst_quantile` overrides `BURST_QUANTILE_DEFAULT` for a tenant that wants more or less
    # headroom than the fleet default. Both are reported in the floor's own provenance string, so a
    # placement certified against a burst says which burst.
    peak_rate_hz: float | None = None
    burst_quantile: float | None = None

    # Where the tokenizer that builds this tenant's prompts comes from. Belongs to the tenant,
    # not the fleet: it decides the tenant's real ISL, and the real ISL is what the tenant's
    # neighbours pay for in prefill.
    #
    # None resolves to `model_path` in `__post_init__`, so by the time anything reads this field
    # it names a real directory. Defaulting to the tenant's own weights invents nothing -- a
    # model's tokenizer *is* part of the model, and it is the only choice for which the requested
    # ISL and the delivered ISL agree. Setting it to something else is allowed and occasionally
    # useful (driving two tenants with a byte-identical prompt, say), which is why it is a field
    # and not a derivation; a fleet that does that gets `isl_exact: false` in its own record and
    # can be read accordingly.
    tokenizer_path: str | None = None

    # This tenant's multimodal payload: how many images per request, and how many pixels each.
    # Zero images is a text tenant, which is the default because most tenants are.
    #
    # Two numbers rather than one because they answer different questions. `image_count` is part
    # of the request shape -- a document assistant sends two dozen pages and a captioner sends
    # one -- and `image_pixels` is what each of those costs. Neither is used to *predict* a visual
    # token count anywhere in this codebase: the harness settles the real prompt length by asking
    # the engine (`workload.calibrate`) rather than deriving it from these fields, for the same
    # reason the KV page is not derived from a config product. What these fields do is fix the payload
    # so that the number the engine reports is the number this tenant asked for.
    #
    # They must agree with the engine flags in `extra_args` -- `--limit-mm-per-prompt` and the
    # `min_pixels`/`max_pixels` in `--mm-processor-kwargs` -- and `fleetfile` refuses a fleet
    # where they do not. Above `max_pixels` the server's preprocessor resizes the image and the
    # driver is then sending one resolution while the engine costs another; above the mm limit the
    # request is rejected outright. Both are stated in one file, so both can be checked there.
    image_count: int = 0
    image_pixels: int = 0

    max_model_len: int = 4096
    max_num_batched_tokens: int = 2048
    dtype: str = "bfloat16"
    image: str = DEFAULT_IMAGE
    # Which serving stack runs this tenant. A name, resolved through `colocation.engines`, and not
    # an `Engine` object: a `TenantTemplate` is compared, hashed into a validity key and written to
    # JSON, and a callable-bearing dataclass does none of those honestly. Per tenant rather than per
    # fleet so one fleet can be measured on two engines and the rows stay distinguishable -- which
    # is the whole point of the exercise (impl §13) -- with the fleet-level `engine:` as the default
    # (`fleetfile`).
    engine: str = engines.DEFAULT_ENGINE
    shm_size_gi: int = 16
    model_revision: str = ""
    extra_args: tuple = ()
    extra_env: tuple = ()         # ({"name":..., "value":...}, ...)

    def __post_init__(self) -> None:
        # Resolved here rather than exposed as a second accessor, so there is no unresolved form
        # for a caller to read by accident: every reader of `tokenizer_path` sees a real path, and
        # the renderer, the record and any report all agree about which tokenizer was used.
        if self.tokenizer_path is None:
            object.__setattr__(self, "tokenizer_path", self.model_path)
        # The other resolution this class owns, and it is here for the same reason: a hybrid stack's
        # KV page is its attention geometry plus its per-sequence state divided by the context it is
        # served at, so the page is not a property of the checkpoint alone. `ModelMemory` arrives from
        # the loader unbound and refuses to price a hybrid page until it is bound (`solver/memory.py`),
        # and this is the one place every tenant in this codebase is constructed -- so binding it here
        # means no caller can price one tenant's context against another's, which is exactly what a
        # single hand-written page per model did.
        if self.mem.context_tokens != self.max_model_len:
            object.__setattr__(self, "mem",
                               replace(self.mem, context_tokens=int(self.max_model_len)))
        if self.tp < 1:
            raise ValueError(f"{self.name}: tp must be >= 1")
        if self.rate_hz <= 0:
            raise ValueError(f"{self.name}: rate_hz must be > 0")
        if self.isl_tokens < 1 or self.osl_tokens < 1:
            raise ValueError(
                f"{self.name}: isl_tokens and osl_tokens must both be >= 1; got "
                f"{self.isl_tokens} and {self.osl_tokens}")
        if self.image_count < 0 or self.image_pixels < 0:
            raise ValueError(
                f"{self.name}: image_count and image_pixels must both be >= 0; got "
                f"{self.image_count} and {self.image_pixels}")
        if bool(self.image_count) != bool(self.image_pixels):
            raise ValueError(
                f"{self.name}: image_count={self.image_count} and "
                f"image_pixels={self.image_pixels} -- state both or neither. A count with no "
                "pixel size does not describe an image, and a pixel size with no count describes "
                "an image nobody sends.")
        if self.image_count >= self.isl_tokens:
            # A deliberately weak check, and weak on purpose. Every image costs at least one
            # token, so this much is certain without knowing anything about the model; the real
            # bound is `image_count x visual_tokens_per_image` plus the chat template, and
            # visual_tokens_per_image depends on patch size and spatial merge, which are the model
            # family's business and not this dataclass's. The authority on that is the harness's
            # start-up probe, which reports the measured cost and refuses to invent a shape that
            # fits (`harness/workload.py`). What is checked here is only what is provable here.
            raise ValueError(
                f"{self.name}: image_count={self.image_count} with isl_tokens={self.isl_tokens}. "
                "Each image costs at least one token, so the stated input length cannot hold "
                "them, let alone the text. Raise isl_tokens or send fewer images.")
        if self.mem.model_id != self.model_id:
            raise ValueError(
                f"{self.name}: mem.model_id is {self.mem.model_id!r} but model_id is "
                f"{self.model_id!r}. The envelope map keys off `mem` and every record keys off "
                "`model_id`, so a mismatch prices one model with another's memory facts.")
        # The workload cannot exceed the engine's own context window: vLLM rejects a request
        # longer than max_model_len, which reads as a delivery failure rather than as the
        # configuration error it is.
        if self.max_model_len < 1:
            raise ValueError(f"{self.name}: max_model_len must be >= 1")
        if self.max_model_len < self.isl_tokens + self.osl_tokens:
            raise ValueError(
                f"{self.name}: max_model_len is {self.max_model_len} but its own request shape "
                f"is {self.isl_tokens}+{self.osl_tokens}="
                f"{self.isl_tokens + self.osl_tokens} tokens. vLLM rejects an over-long "
                "request, which the harness records as a delivery failure -- so this would look "
                "like an infeasible placement rather than a misconfigured tenant.")
        if self.peak_rate_hz is not None and self.peak_rate_hz < self.rate_hz:
            raise ValueError(
                f"{self.name}: peak_rate_hz is {self.peak_rate_hz:g} but rate_hz is "
                f"{self.rate_hz:g}. A peak below the mean is not a burst -- it is a lower offered "
                "rate, and the rate the harness will actually drive is rate_hz. State the rate this "
                "tenant is offered and, if its arrivals clump, the rate they clump to.")
        if self.burst_quantile is not None and not 0.5 <= self.burst_quantile < 1.0:
            raise ValueError(
                f"{self.name}: burst_quantile is {self.burst_quantile!r}; it must be at least 0.5 and "
                "below 1. Below the median it sizes for less than the mean load, which is an "
                "under-provision wearing a burst allowance's name; at 1.0 the quantile is unbounded "
                "and every floor derived from it is infinite.")
        # THE TARGET HAS TO BE REACHABLE BY THE ENGINE THAT SERVES IT. Steady state means retiring
        # `rate_hz x osl_tokens` tokens/s; with N sequences in flight each retiring one token per
        # ITL the engine retires N/ITL, so holding the target requires N = rate x osl x ITL sequences
        # resident -- and `--max-num-seqs` caps N. Above the cap the queue diverges: the tenant does
        # not miss its target by a margin, it never reaches steady state at all, which the harness
        # reports as infeasibility of the *placement*. `search-r2hk3z` was refused for exactly that,
        # its anchor needing 82 concurrent against the chart default of 64.
        #
        # That was a refusal here while the cap was a fleet-file field. It is now an invariant of the
        # derivation: `max_num_seqs` sizes the cap from an occupancy that has this tenant's whole
        # decode time inside it, so it cannot return a number below this one. The check stays, cheap,
        # and if it ever fires the derivation is wrong rather than the fleet file.
        want = self.concurrency_at_target()
        if want is not None and math.ceil(want) > self.max_num_seqs:
            raise ValueError(
                f"{self.name}: derived max_num_seqs {self.max_num_seqs} is below the {want:.2f} "
                f"requests in flight its own target needs at its own rate ({self.rate_hz:g} Hz x "
                f"{self.osl_tokens} osl x {self.slo.itl_mean_ms:g} ms ITL). That is a bug in "
                "`TenantTemplate.max_num_seqs`, not a fleet-file error.")

    @property
    def span_tokens(self) -> int:
        """Tokens one request of this tenant's occupies in KV at its longest."""
        return self.isl_tokens + self.osl_tokens

    def burst_concurrency(self, mean: float) -> tuple[float, str]:
        """A mean occupancy, raised to the burst this tenant is sized for, and how (impl §14.9).

        Two ways, and which one applies is the fleet file's to state:

          - `peak_rate_hz` stated: the occupancy scales with the rate, so the burst occupancy is the
            mean one at the peak rate. No distribution is assumed, because the fleet supplied the
            number a distribution would have been used to guess.
          - otherwise: arrivals are treated as Poisson at `rate_hz`, which makes the number in flight
            Poisson with mean `mean` and standard deviation `sqrt(mean)`, and the quantile is
            `mean + z_q sqrt(mean)`. `z_q` comes from the normal approximation, which is the usual
            one at these occupancies and is never the binding uncertainty here -- the arrival process
            being Poisson at all is a much stronger assumption than the shape of its tail.

        **Poisson is an assumption about the offered load and nothing measures it**, which is why it
        is a quantile the fleet may set (`burst_quantile`) rather than a constant, and why what it
        produces is a floor that chooses experiments rather than a claim about what will happen. The
        direction matters: sizing for the mean guarantees the pool is too small half the time, and a
        pool that is too small preempts, which the engine serves as a worse ITL -- so the tenant
        misses a target for a reason that is nowhere in its record. That failure is invisible; an
        over-large floor is merely a placement that ranks worse than it deserves, and the walk finds
        out. Between an invisible error and a visible one this sizes for the visible one.
        """
        if self.peak_rate_hz:
            scaled = mean * (float(self.peak_rate_hz) / self.rate_hz)
            return scaled, (f"{mean:.2f} at {self.rate_hz:g} Hz -> {scaled:.2f} at the stated "
                            f"{self.peak_rate_hz:g} Hz peak")
        q = BURST_QUANTILE_DEFAULT if self.burst_quantile is None else float(self.burst_quantile)
        z = NormalDist().inv_cdf(q)
        return mean + z * math.sqrt(mean), (
            f"{mean:.2f} mean + {z:.2f}*sqrt({mean:.2f}) = "
            f"{mean + z * math.sqrt(mean):.2f} at the q{q:g} arrival burst"
            f"{'' if self.burst_quantile is None else ' stated by this tenant'}")

    @property
    def min_kv_tokens(self) -> int:
        """The minimum viable KV pool: `MIN_POOL_REQUESTS` requests at this tenant's own context.

        Derived, and it was a fleet-file field until impl §14.9. Two reasons it could not stay one.
        It is not a fact about the workload that anybody knows independently -- every shipped fleet
        stated exactly 2x or 4x its own `max_model_len`, i.e. this arithmetic, transcribed by hand and
        then obliged to move whenever the context did. And it is the floor that has to survive a
        tenant with no ITL target at all, where there is no other floor: the SLO floor is zero there,
        and zero would let the solver hand a tenant a pool it cannot admit one request into.

        Stated in `max_model_len` rather than in `span_tokens` deliberately. The engine admits on its
        own context window, not on the shape the harness happens to send, so a pool measured in spans
        says nothing about whether a request can be admitted; and on a hybrid stack the pool is also
        where the per-sequence state slots come from, one per `max_model_len` of pool
        (`solver/memory.ModelMemory.state_bytes_per_token_tp1`), so `MIN_POOL_REQUESTS` full contexts
        is also `MIN_POOL_REQUESTS` state slots. One number does both jobs because the engine sizes
        both from the same pool.
        """
        return MIN_POOL_REQUESTS * int(self.max_model_len)

    @property
    def max_num_seqs(self) -> int:
        """The batch cap: the burst occupancy of a *slot*, or the engine's own default if that is more.

        Derived, and a fleet-file field until impl §14.9. What it has to cover is different from what
        `kv_tokens_at_target` covers, and the difference is prefill. A slot is held from the moment a
        request is admitted until its last token: `ttft_p95_ms` (the tenant's own ceiling on how long
        the prefill half may take) plus `osl x itl`. The KV floor is derived from the decode half
        alone, because that is the half whose *tokens* are resident and growing -- a request waiting
        on its first token holds a slot and (on this engine) already holds its prompt's pages, but the
        thing that runs out first is slots.

        That distinction is not theoretical. The GLM-4.5-Air anchor was driven alone at a cap of 96
        with a measured TTFT p95 of 56 s: the decode-only arithmetic wanted 41 slots and it failed,
        because at 0.5 Hz a 56-second prefill queue is 28 slots nobody had counted.

        **The floor under it is the engine's own default, not this arithmetic** (`engines/*.py`,
        `default_max_num_seqs`). A cap is not free to be tight: below the offered burst the queue
        diverges, and divergence is a cliff rather than a gradient -- the tenant does not serve
        slightly worse, it stops reaching steady state. Where the arithmetic asks for less than the
        engine would have given anyway, taking the engine's number costs some activation memory and
        buys a graceful over-provision, and it also keeps every stored row whose cap was that default
        (`store/serving.py` keys on it). Where the arithmetic asks for more, it wins, and the row is
        re-keyed because the measurement really would be of a different engine configuration.
        """
        want = self.concurrency_at_slot_occupancy()
        floor = engines.get(self.engine).default_max_num_seqs
        return floor if want is None else max(floor, math.ceil(want))

    def concurrency_at_slot_occupancy(self) -> float | None:
        """Slots the burst holds: `rate x (ttft ceiling + osl x itl)`, burst-adjusted. None if no ITL.

        None rather than zero when no ITL ceiling is stated, for `kv_tokens_at_target`'s reason: a
        tenant with no target has no requirement, which is different from a requirement of nothing.
        A stated `ttft_p95_ms` with no `itl_mean_ms` is still None -- the prefill half of an occupancy
        whose decode half is unstated is not an occupancy.
        """
        if self.slo is None or self.slo.itl_mean_ms is None:
            return None
        decode_s = self.osl_tokens * (self.slo.itl_mean_ms / 1000.0)
        prefill_s = (self.slo.ttft_p95_ms or 0.0) / 1000.0
        return self.burst_concurrency(self.rate_hz * (prefill_s + decode_s))[0]

    @property
    def serving(self) -> ServingConfig:
        """The engine flags this tenant is served with, as the value the store keys on.

        Assembled here rather than in the loader so there is one definition of which flags are part
        of a measurement's identity, and it lives next to the fields themselves. `tp` and the
        envelope are excluded: TP has its own field in the validity key, and the envelope is what
        the search varies (`store/serving.py`).
        """
        return ServingConfig(engine=self.engine,
                             max_num_seqs=self.max_num_seqs,
                             max_num_batched_tokens=self.max_num_batched_tokens,
                             max_model_len=self.max_model_len,
                             dtype=self.dtype,
                             extra_args=tuple(self.extra_args))

    def concurrency_at_target(self) -> float | None:
        """How many requests must be in flight at once for this tenant to hold its own target.

        `rate_hz x osl_tokens x itl_target_s` -- the mean, un-inflated, and the first half of
        `kv_tokens_at_target`'s arithmetic. Named separately because it answers a different question:
        that method asks how much *memory* the target needs, and this one how much *batch*. The
        no-burst mean is what `__post_init__` checks the derived cap against and what
        `concurrency_load` reports, because both are about the load a tenant is known to offer rather
        than the burst its pool is sized for; `concurrency_at_slot_occupancy` is the sized one.

        None when no ITL target was stated, for `kv_tokens_at_target`'s reason.
        """
        if self.slo is None or self.slo.itl_mean_ms is None:
            return None
        return self.rate_hz * self.osl_tokens * (self.slo.itl_mean_ms / 1000.0)

    def concurrency_load(self) -> float | None:
        """`concurrency_at_target / max_num_seqs`: how much of the batch cap the target consumes.

        Reported rather than thresholded. A tenant at 0.96 is feasible and one small change from
        not being -- and which side of that a fleet wants to sit on is a judgement about how much
        interference it expects, not something this file can decide. Making the number visible at
        load is what a threshold would have hidden.
        """
        want = self.concurrency_at_target()
        return None if want is None else want / float(self.max_num_seqs)

    @property
    def request(self) -> TenantRequest:
        """The solver's view: the memory facts and the floors, nothing about how it deploys.

        **The SLO-derived floor is computed here and handed over**, rather than the solver being
        given the `TenantSlo` and deriving it. The solver enforces floors; it does not interpret
        targets. Keeping the derivation on this side means there is exactly one place that turns a
        latency ceiling into a token count (`kv_tokens_at_target`), and the solver cannot grow a
        second opinion about what a target means.

        A tenant with no ITL ceiling passes zero, which is impl §9's "no floor from this rule".
        """
        floor = self.kv_tokens_at_target()
        tokens, how = floor if floor is not None else (0, "no ITL ceiling stated")
        return TenantRequest(name=self.name, model_id=self.model_id, tp=self.tp,
                             rate_hz=self.rate_hz, isl_tokens=self.isl_tokens,
                             osl_tokens=self.osl_tokens, mem=self.mem,
                             min_kv_tokens=self.min_kv_tokens,
                             kv_floor_tokens=tokens, floor_source=how)

    def kv_tokens_at_target(self) -> tuple | None:
        """`(tokens, how it was derived)` needed to hold the ITL target at the offered rate.

        Little's law, run in the direction design §1.7 now specifies: not to forecast a latency
        but to state a requirement from a target.

            concurrency = rate_hz x osl_tokens x itl_target_s   (the mean)
            burst       = burst_concurrency(concurrency)        (what the pool is sized for, §14.9)
            tokens      = burst x (isl_tokens + osl_tokens)

        **The burst step is why the mean is not the answer.** Little's law gives the *average* number
        in flight; arrivals at an average rate are not evenly spaced, and a pool sized for the average
        is one that preempts on every clump. `burst_concurrency` raises it to the quantile the fleet
        sizes for, which is the one new input impl §14.9 adds and the only thing about a KV floor a
        fleet file may still state.

        Both the rate and the shape are this tenant's own. A requirement, not a prediction: it
        says only that for this tenant to hold its stated latency at its stated rate, this much
        KV cache has to be resident. Nothing here claims it will be achieved -- interference can
        miss a target at any pool size, and only a measurement settles that. That is why the
        solver may enforce it as a floor (§4.3) while nothing in the architecture predicts a
        latency (§4.4).

        Note which way the target pushes: a *tighter* ITL ceiling implies *less* KV cache, not
        more, because requests that finish sooner are resident for less time. So the two halves
        of an SLO pull on different resources -- the rate demands memory and the latency
        ceiling demands compute -- which is why neither half alone determines a configuration.

        None when no ITL target was stated. Deliberately not zero: 'no requirement' and 'a
        requirement of nothing' are different, and only one of them can be certified.
        """
        if self.slo is None or self.slo.itl_mean_ms is None:
            return None
        span = self.span_tokens
        mean = self.rate_hz * self.osl_tokens * (self.slo.itl_mean_ms / 1000.0)
        burst, how = self.burst_concurrency(mean)
        tokens = math.ceil(burst * span)
        return tokens, (f"{self.rate_hz:g} Hz x {self.osl_tokens} osl x "
                        f"{self.slo.itl_mean_ms:g} ms = {how} concurrent, "
                        f"x {span} tokens/request")


@dataclass(frozen=True)
class NodeInventory:
    """A node's GPUs, by host ordinal, with the UUID each ordinal resolves to.

    Read from the `<daemon-release>-gpu-uuids-<node>` ConfigMap the daemon publishes, never
    written by hand -- so `mps-daemon-gpu-uuids-<node>` for the one hand-installed daemon and
    `mps-<token>-gpu-uuids-<node>` for a run that captured its own node (agent/capture.py).
    The ordinal is what `CUDA_VISIBLE_DEVICES` takes and the UUID is what the MPS
    memory limit must be keyed by, so both are needed for every tenant and they must agree in
    order. `uuids` is indexed by position in `gpu_ids`.
    """

    node_name: str
    gpu_ids: tuple           # host ordinals, ascending
    uuids: tuple             # same length and order as gpu_ids, or () when not leased
    # What the driver reports as the size of one of these GPUs. Required: it is measured per card
    # (`store/hwfacts.py`) and there is no product table to fall back on, so an inventory that does
    # not know it cannot be built rather than being built against a guess.
    device_total_gib: float

    def __post_init__(self) -> None:
        if not self.gpu_ids:
            raise ValueError(f"{self.node_name}: no GPUs")
        if not self.uuids:
            # An UNLEASED inventory: the shape of a machine nobody has captured yet, which is
            # what `plan` reasons about. Legitimate, and deliberately not filled with
            # placeholders -- a synthetic UUID that reached `render_tenant_values` would key an
            # MPS memory limit to a device that does not exist, and that is the one failure on
            # this stack with no symptom. `uuid_for` refuses instead.
            return
        if len(self.gpu_ids) != len(self.uuids):
            raise ValueError(
                f"{self.node_name}: {len(self.gpu_ids)} GPU ordinals but "
                f"{len(self.uuids)} UUIDs. These are matched by position and an off-by-one "
                "would key each tenant's memory limit to its neighbour's GPU -- which does "
                "not error, it silently lifts the limit.")
        bad = [u for u in self.uuids if not str(u).startswith("GPU-")]
        if bad:
            raise ValueError(
                f"{self.node_name}: {bad} are not GPU UUIDs. An ordinal used as an MPS "
                "memory-limit key references nothing and is ignored without warning.")

    @property
    def leased(self) -> bool:
        """True once a real node has been captured and its UUIDs read off it."""
        return bool(self.uuids)

    @property
    def node_spec(self) -> NodeSpec:
        """The solver's view: how many interchangeable GPUs, and how much fits on one.

        Available unleased, which is the point: how many GPUs a `g6e.48xlarge` has and how much
        of one an engine gets are facts about the instance type, so feasibility can be settled
        before any money is spent provisioning one.
        """
        return NodeSpec(gpu_ids=tuple(self.gpu_ids), device_total_gib=self.device_total_gib)

    def uuid_for(self, gpu_id: int) -> str:
        if not self.uuids:
            raise KeyError(
                f"{self.node_name} has no GPU UUIDs because no node has been captured. UUIDs "
                "are read from the daemon's ConfigMap on the node a run actually leased "
                "(agent/capture.py) and are never written by hand or invented: a UUID that "
                "names no device is accepted by MPS and ignored, so the tenant runs with no "
                "memory envelope at all and the record looks healthy.")
        try:
            return self.uuids[list(self.gpu_ids).index(gpu_id)]
        except ValueError:
            raise KeyError(
                f"GPU ordinal {gpu_id} is not in {self.node_name}'s inventory "
                f"{list(self.gpu_ids)}") from None


@dataclass(frozen=True)
class WorkloadSpec:
    """What is fleet-wide about the load, which is now only one thing.

    **The sequence shape used to live here and no longer does.** It moved onto the tenant
    (design §8.3): a single fleet-wide ISL/OSL is not a simplification of a mixed fleet, it is a
    different problem, and holding it here made every caller read the fiction.

    **The tokenizer followed it, for the same reason.** `tokenizer_path` was fleet-wide with a
    stated limit -- exact for the model it belongs to, approximate for any other -- and that
    limit is not a rounding error once a fleet holds more than one model: Qwen3's vocabulary is
    151936 entries and Llama-3's is 128256, so one tokenizer building a "512 token" prompt for
    both delivers 512 tokens to one tenant and some other number to the other, while the
    validity key names 512 for both. Prefill cost is most of what a long-input tenant does to
    its neighbours, so that is not a small fiction. It is now `TenantTemplate.tokenizer_path`,
    defaulting to the tenant's own weights, which is the only answer that is always right.

    What is left is genuinely shared: the delivery gate that decides whether any latency figure
    means anything at all (impl §6.3). One field is not a reason to delete the type -- the gate
    is a property of the run and not of any tenant, and giving it a home says so.
    """

    # Fraction of offered tokens that must arrive for latency to mean anything (impl §6.3).
    delivered_fraction_threshold: float = 0.99

    def __post_init__(self) -> None:
        if not 0.0 < self.delivered_fraction_threshold <= 1.0:
            raise ValueError(
                "delivered_fraction_threshold must be in (0, 1]; got "
                f"{self.delivered_fraction_threshold!r}")


@dataclass(frozen=True)
class FleetSpec:
    """The search problem: these tenants, on this node.

    Everything the agent needs and nothing it decides. Swap the tenants and the same agent
    searches a different fleet; that is what makes this a search agent rather than a
    configuration for one deployment.
    """

    tenants: tuple
    node: NodeInventory
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)

    def __post_init__(self) -> None:
        if not self.tenants:
            raise ValueError("a fleet needs at least one tenant")
        names = [t.name for t in self.tenants]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tenant names: {names}")

    @property
    def requests(self) -> tuple:
        return tuple(t.request for t in self.tenants)

    @property
    def sequence_shapes(self) -> dict:
        """tenant -> `(isl, osl, image_count, image_pixels)`, for the validity key (design §4.1).

        Taken from the tenants rather than accepted as a separate declaration, so the key cannot
        name a shape the fleet was not driven at. The image pair travels with the token pair for
        the same reason it shares the key's tuple: it is part of what the request *is*, and the
        harness is handed all four from this one place.
        """
        return {t.name: (t.isl_tokens, t.osl_tokens, t.image_count, t.image_pixels)
                for t in self.tenants}

    @property
    def tp_degrees(self) -> dict:
        """tenant -> tensor-parallel degree, for the validity key (design §4.1).

        Taken from the tenants for the same reason `sequence_shapes` is: the key must not be able
        to name a shard count the fleet was not served at. TP is a fixed input here -- the search
        chooses which GPUs a tenant lands on, never how many it runs across -- so it conditions
        every measurement under the key rather than varying within it.
        """
        return {t.name: t.tp for t in self.tenants}

    @property
    def serving_configs(self) -> dict:
        """tenant -> `ServingConfig`, for the validity key.

        Taken from the tenants for `tp_degrees`' reason, and in the key for `tp_degrees`' reason:
        these flags are fixed inputs that condition every measurement under the key rather than
        things the search varies (`store/serving.py`).
        """
        return {t.name: t.serving for t in self.tenants}

    @property
    def untargeted(self) -> tuple:
        """Tenants that stated no SLO, in fleet order.

        Surfaced rather than filled in. A mixed fleet -- some tenants with targets, some
        without -- is allowed, because a fleet may legitimately carry a best-effort tenant
        alongside ones that must hold a number. What it must not do is look like a fleet where
        everything is targeted, so the names are available to every report and brief that
        claims a configuration serves this fleet.
        """
        return tuple(t.name for t in self.tenants if t.slo is None)

    def tenant(self, name: str) -> TenantTemplate:
        for t in self.tenants:
            if t.name == name:
                return t
        raise KeyError(f"no tenant named {name!r} in this fleet")
