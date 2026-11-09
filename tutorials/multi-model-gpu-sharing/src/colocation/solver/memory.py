"""The envelope arithmetic the solver owns (impl §4.2, consumed by §9).

Two directions, and the second is what makes §9's memory envelope a *decision* variable
rather than a value copied out of a config file:

    envelope_gib  ->  gpu_memory_utilization   (what the chart needs)
    envelope_gib  ->  kv_cache_tokens          (what §9's KV floors are stated in)

Without the second map the envelope has no effect on anything the solver can check, and a KV
floor derived from a latency target could not be turned into a constraint on a decision variable
at all. With it, an envelope is what a floor is enforced against and a margin is measured in.

The map is affine in the envelope, exactly:

    kv_gib    = (envelope_gib - mps_reserve_gib) * headroom - weight_gib/tp - activation_gib
    kv_tokens = kv_gib * 2**30 / (kv_bytes_per_token / tp)

`device_total_gib` cancels out of the KV calculation — it appears in `util` as a divisor and
comes straight back as a multiplier — so KV capacity depends on the envelope and not on the
card, *provided the same figure is used on both sides*. It is one number, measured
(`store/hwfacts.py`), and it is the driver's: what `nvidia-smi --query-gpu=memory.total`
reports, because that is what the engine multiplies the utilisation by. Two different
denominators — a smaller "usable" one here and the driver's one in the engine — do not cancel,
and the residue is unpriced KV memory on every GPU. It still bounds the envelope, which is where
§9's hard constraint lives.

**Validated against every measured row in impl §4.2, to better than 0.2%:**

    tp  envelope  predicted kv_gib  measured  predicted tokens  measured  error
    1     8 GiB       5.124          5.13         47,972         48,032   0.12%
    1    20 GiB      16.524         16.5        154,702        154,432   0.17%
    2     8 GiB       5.336          5.33         99,904         99,776   0.13%
    2    20 GiB      16.736         16.74       313,365        313,408   0.01%

`kv_bytes_per_token` is not fitted — it is arithmetic on the model's own config, and on a *dense*
model it lands exactly. Qwen3-0.6B has 28 layers, 8 KV heads, head_dim 128, bf16:
`2 (K and V) * 2 bytes * 128 * 8 * 28` = 114,688 B = 112 KiB per token, and 112/2 = 56 KiB per
shard at TP=2. Both are what the measured block counts imply to the byte.

**The assumption underneath that, and where it fails.** The product above holds only because every
layer of a dense model keeps a KV page. A hybrid attention family breaks it and the arithmetic has
no way to notice: Qwen3.5-9B's `layer_types` marks 8 of its 32 layers `full_attention` and gives the
other 24 a linear-attention state instead, so the all-layers product comes out at 131,072 B/token
against an engine-delivered 40,129 — 3.27x too large — while counting only the attention layers
gives 32,768, which is 18% too *small*, the direction that OOMs under load. The over-estimate is
merely conservative for occupancy but not harmless: it inflates the envelope every KV floor is
enforced against, so a fleet that fits on one L40S is declared infeasible even at TP=2, which
attacks §0's premise rather than a margin.

So this arithmetic is the `derived` tier and is labelled as such (`ModelMemory.kv_bytes_per_token_tier`,
`colocation/tiers.py`). It stays because `plan` must work offline with no GPU, and because for a
family nobody has measured an over-safe number is the right default. What supersedes it is a
`measured` value carried into the fleet file as an explicit `kv_bytes_per_token`. Note that
"over-safe" was an empirical observation on this family and never a proof -- a hybrid whose linear
state dominates its attention pages could exceed the all-layers product -- which is why a plan
resting on the derived tier is reported as resting on it.

**That caveat is now a measurement, so read "over-safe" as a hope and not a property.**
Qwen3.5-0.8B derives 12,288 B/token from its attention layers alone (6 of 24 layers, 2 KV heads,
head_dim 256) and an engine delivers **52,230** at `max_model_len: 512` -- four times as much, which
under-states every KV floor computed from it and fails as an OOM rather than as a refusal. The
mechanism is the one this docstring predicted: a hybrid page is the attention-only geometry plus a
per-sequence recurrent state divided by the context length, so the shorter the context the larger the
per-token page.

**So the page is now that sum, and the context is part of it** (`page_bytes_per_token_tp1`). The
state bytes come from the checkpoint's own `linear_*` config fields (`store/hubfacts.py`) and the
context from the tenant's `max_model_len`, which is why a `ModelMemory` is bound to a tenant before
anything prices it: two tenants of one checkpoint at different contexts have different pages, and a
single number per model was wrong for at least one of them. On the five tenants of
`qwen35-multimodal.yaml` the sum lands 1.6% to 5.9% below what the engine delivered -- low,
and that is the OOM direction, so it may choose which degree to boot first and may not rule one out
(§14.2). What it replaces was worse in both directions at once: one hand-written page per *model*, so the
0.8B's was the figure an engine delivered at `max_model_len: 512` and the fleet's other tenant on that
checkpoint serves 1024 -- priced 2.1x over what its own context implies.

**What follows from that, since impl §14.2.** There is no longer a mode that measures this page, and
the reason is not that measuring it stopped being useful: the only way to read it is off the engine's
own start-up report, which does not generalise past the one engine whose log was parsed. So the
number stays derived, and what changed is what a derived number is *allowed to do*. It may choose
which degree to boot first. It may not rule a degree out. Only the weight floor may (§14.2), and a
degree whose boot fails for want of memory is a witness that moves the walk -- which is an answer of
the right shape, measured on the driver rather than read out of the engine.

`activation_gib` — PyTorch's peak working set, inside the `gpu_memory_utilization` budget and
therefore *not* part of `mps_reserve_gib` — is the one term that is fitted, and it is fitted to
a constant: the residual is 0.416 ± 0.003 GiB at TP=1 and 0.2845 ± 0.0005 GiB per shard at
TP=2 across a 2.5x envelope range. Flat, like the reserve, and for the same reason.

What the four rows actually pin down is the *sum* `weight_gib/tp + activation_gib` — 1.526 GiB
at TP=1 and 0.8395 GiB per shard at TP=2. The split is not identified by them, so
`weight_gib` is taken from the checkpoint (596M parameters at 2 bytes = 1.110 GiB, a fact about
the file rather than a fit) and the activation term is what remains. The split matters because
only the weights divide by TP degree, so getting it wrong would mispredict TP=4 and TP=8 while
still reproducing every measured row. It is also why `activation_gib` is keyed by TP degree:
activations shard too, just not by the same factor, and forcing one number to cover both
degrees would push the error into the weight term where it does not belong.

**Both fallback defaults are safe, not accurate, and are not to be tuned** — impl §4.2 argues
this at length for `mps_reserve_gib` and the argument transfers unchanged to
`activation_gib`. Over-stating either one under-states KV tokens, which over-states occupancy,
which makes an unmeasured candidate rank worse than it deserves. That is the direction an
error should point when the alternative is a runtime OOM in the middle of a measurement.

Pure standard library. The solver's CP-SAT layer needs OR-Tools; this file does not.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tiers import TIER_DERIVED

BYTES_PER_GIB = 1 << 30

# impl §4.2. Covers allocator fragmentation. No longer load-bearing on this stack; kept as
# insurance, because removing it and being wrong costs a runtime OOM.
HEADROOM_DEFAULT = 0.95

# impl §4.2's reserve: extra memory the engine uses on the device that
# `gpu_memory_utilization` does not account for -- CUDA context, cuBLAS/cuDNN scratch and
# CUDA-graph pools, and at TP>1 the NCCL collective buffers. Subtracted, never multiplied: it
# is a roughly constant number of bytes, and a multiplier cannot express that.
MPS_RESERVE_TP1_GIB = 1.0
MPS_RESERVE_TPN_GIB = 1.5

# PyTorch's peak activation working set, per shard. Inside the util budget, so distinct from
# the reserve. Measured 0.416 at TP=1 and 0.2845 at TP=2 on Qwen3-0.6B; the default is
# deliberately well above both, for models nobody has profiled.
ACTIVATION_GIB_DEFAULT = 1.0

# Bytes per element for the KV cache dtype. bf16/fp16 on this stack.
KV_DTYPE_BYTES = 2


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int,
                       dtype_bytes: int = KV_DTYPE_BYTES) -> int:
    """KV cache bytes one token occupies at TP=1. Arithmetic, not a measurement.

    `2 *` because K and V are both cached. At TP=N each shard holds `num_kv_heads / N` heads,
    so per-shard cost is this divided by N -- which is why `ModelMemory` stores the TP=1
    figure and divides at the point of use.
    """
    return 2 * int(dtype_bytes) * int(head_dim) * int(num_kv_heads) * int(num_layers)


@dataclass(frozen=True)
class ModelMemory:
    """Per-model memory facts, derived from the checkpoint's own config (impl §14.8).

    `weight_gib` is the whole model; it divides by TP degree because TP shards the weights.
    `activation_gib` and `measured_reserve_gib` are per shard, and both are None-able: None
    means "never profiled", and the fallback path is the only place the generous defaults
    apply.

    **One of these is not a per-model fact, and it is the reason this class is bound to a tenant.**
    A hybrid stack's page depends on the context it is served at (`context_tokens`), so a
    `ModelMemory` fresh out of the loader is not yet priceable and `kv_bytes_per_token` refuses it.
    `spec.TenantTemplate.__post_init__` does the binding, in the one place every tenant goes through.
    """

    model_id: str
    weight_gib: float
    # Attention KV only: `layers x KV heads x head_dim x 2 x dtype` over the layers that cache K and
    # V. NOT the page the engine allocates on a hybrid stack -- that is `page_bytes_per_token_tp1`,
    # which adds the state term below. Named for what it is rather than for what most models make it.
    kv_bytes_per_token_tp1: int
    # GiB of `weight_gib` that the checkpoint stores but the engine does not reside, per impl
    # §14.2. The one known instance is a tied `lm_head`: with `tie_word_embeddings: true` the
    # safetensors holds both `model.embed_tokens.weight` and `lm_head.weight` while the engine
    # resides the embedding once, so the checkpoint over-reads by `vocab_size x hidden_size x
    # dtype_bytes`. Default 0.0, which is the right default twice over: a model with no tied head
    # has nothing to subtract, and a `weight_gib` that is already a *resident* figure (a measured
    # sheet, or a fleet file that did the subtraction itself) must not have it subtracted again.
    # State it only when `weight_gib` is the gross checkpoint figure -- which is what makes
    # double-subtraction unrepresentable rather than merely discouraged.
    tied_embedding_gib: float = 0.0
    # tp -> measured activation GiB per shard. Keyed by TP degree because activations shard
    # too: measured 0.416 at TP=1 and 0.2845 per shard at TP=2 on Qwen3-0.6B.
    activation_gib: dict = None      # type: ignore[assignment]
    # tp -> measured extra memory per shard, from a §6 profile of that exact pair.
    measured_reserve_gib: dict = None  # type: ignore[assignment]
    # Bytes one *sequence* occupies at TP=1 in the fixed-size recurrent state a linear-attention or
    # mamba layer keeps instead of a growing KV page. Derived from the checkpoint's `linear_*` config
    # fields (`store/hubfacts.KvGeometry`); 0 on a dense stack, which is every stack that has no such
    # layer. Charged through the page rather than as a fixed term, because that is the accounting the
    # engine's own delivered page size agrees with: it provisions one state slot per `max_model_len`
    # of KV pool, so a pool sized for N full-length requests is a pool with N state slots in it.
    state_bytes_per_seq: int = 0
    # KV heads this stack has, from the same config as the page. It is here for one purpose: below
    # `tp == num_kv_heads` the heads divide and the attention page shards; at and above it they are
    # REPLICATED instead, so the page per shard stops falling. Qwen3.5-0.8B has 2, and an engine
    # delivered it 52,230 B/token at tp1 and tp2 and 65,281 at tp4 -- a 25% jump that a page divided
    # by the degree cannot see, and it under-states a KV floor in the OOM direction. 0 means "not
    # known", which divides by the degree as before.
    num_kv_heads: int = 0
    # The context the state above is amortised over: the `max_model_len` of the tenant this was bound
    # to. 0 means unbound, which is what the loader returns and what `kv_bytes_per_token` refuses on a
    # hybrid -- the alternative is pricing a hybrid page as attention-only, silently, which is the
    # 4x-too-small figure §14.9 exists to remove.
    context_tokens: int = 0
    # How `kv_bytes_per_token_tp1` was obtained (`colocation/tiers.py`). Defaults to `derived`
    # because that is what the product of config scalars above this class is, and a default of
    # anything stronger would describe every unmeasured model as measured. A `measured` value
    # comes from a fact sheet by way of the fleet file, never from this module.
    kv_bytes_per_token_tier: str = TIER_DERIVED

    def reserve_gib(self, tp: int) -> tuple[float, str]:
        """`mps_reserve_gib` for this (model, TP) pair, and where the number came from.

        impl §9 requires the measurement when one exists and the safe default otherwise. The
        provenance string travels with it because a placement priced on the default is a
        placement nobody has run, and that is worth being able to see in a ranking.
        """
        measured = (self.measured_reserve_gib or {}).get(int(tp))
        if measured is not None:
            return float(measured), f"measured@tp{tp}"
        default = MPS_RESERVE_TP1_GIB if int(tp) == 1 else MPS_RESERVE_TPN_GIB
        return default, f"default(tp{'1' if int(tp) == 1 else 'N'})"

    def activation_for(self, tp: int) -> tuple[float, str]:
        measured = (self.activation_gib or {}).get(int(tp))
        if measured is not None:
            return float(measured), f"measured@tp{tp}"
        return ACTIVATION_GIB_DEFAULT, "default"

    @property
    def state_bytes_per_token_tp1(self) -> int:
        """The per-sequence state, spread over the context it is held for. Zero on a dense stack.

        Rounded up, because the alternative rounds a memory requirement down. Refuses rather than
        assuming a context: see `context_tokens`.
        """
        if not self.state_bytes_per_seq:
            return 0
        if not self.context_tokens:
            raise ValueError(
                f"{self.model_id} keeps {self.state_bytes_per_seq} B of recurrent state per sequence "
                "but this ModelMemory is not bound to a tenant's max_model_len, so there is nothing "
                "to amortise it over. A page priced without it is the attention-only figure, which "
                "is 4x too small on this family and fails as an OOM rather than as a refusal "
                "(impl §14.9). Bind it through `spec.TenantTemplate`, which is where every tenant "
                "gets its context.")
        return -(-int(self.state_bytes_per_seq) // int(self.context_tokens))

    @property
    def page_bytes_per_token_tp1(self) -> int:
        """What the engine allocates per token of pool at TP=1: attention pages plus state share."""
        return int(self.kv_bytes_per_token_tp1) + self.state_bytes_per_token_tp1

    def kv_bytes_per_token(self, tp: int) -> float:
        """Page bytes per token **per shard** at this degree. Not simply the tp1 page over `tp`.

        Two halves shard differently. The attention half divides by the degree only while there are
        heads left to divide: at `tp > num_kv_heads` the heads are replicated and every shard holds
        the same pages, so the divisor stops at `num_kv_heads`. The state half divides by the degree
        throughout -- a hybrid stack's linear layers carry tens of value heads, more than any degree
        this node offers, so their cliff is out of reach and is not modelled (if a family ever puts it
        in reach, this is the line that has to learn about it).

        Checked against the one measurement of the cliff that exists: Qwen3.5-0.8B at 512 tokens of
        context, 2 KV heads, measured 52,230 B/token at tp1 and tp2 and 65,281 at tp4. This gives
        12,288/1 + 36,864/1 = 49,152 at tp1, and 12,288/2 + 36,864/4 = 15,360 per shard at tp4, i.e.
        61,440 tp1-equivalent -- 5.9% below each, the same residual at both degrees, which is the
        state estimate and not the cliff.
        """
        heads = int(self.num_kv_heads) or int(tp)
        attention = int(self.kv_bytes_per_token_tp1) / float(min(int(tp), heads))
        return attention + self.state_bytes_per_token_tp1 / float(tp)

    def fixed_gib(self, tp: int) -> float:
        """Everything inside the util budget that is not KV cache: weights plus activation."""
        return self.weight_gib / float(tp) + self.activation_for(tp)[0]

    def weight_lower_bound_gib(self, tp: int) -> float:
        """The least device memory one shard can possibly occupy at this degree (impl §14.2).

        **This is the only quantity in this module a refusal may rest on**, and the reason is the
        direction of its error rather than its provenance. Every other term here can come back
        smaller than stated: `activation_for` falls back to 1.0 GiB and the engine wanted 0.2256 on
        the 0.6B, `reserve_gib` falls back to 1.0/1.5, `kv_bytes_per_token` was 3.27x too large on
        the 9B (and 6% too small on the 0.8B), and `floor_tokens` is Little's law against a stated
        target. A degree eliminated with any of those in the sum is a degree eliminated on a number
        nothing has checked -- which is how `plan` came to rule out tp1 for a model nobody ever
        booted at tp1. This one cannot come back smaller, so "it does not fit" is a fact about the
        device and not a prediction.

        Deliberately optimistic in every other direction: no `headroom` divisor, no MPS reserve, no
        activation, and a KV pool of zero. No engine can actually start with zero KV blocks, so a
        shard that clears this bound may still fail to come up -- and that is the point. Clearing
        the bound does not mean the degree works, it means only a measurement may say it does not.

        Soundness rests on two things being true of `weight_gib`, both of which the *caller* owns:
        that `tied_embedding_gib` accounts for anything the checkpoint stores and the engine does
        not reside, and that the serving dtype is no narrower than the checkpoint's. A bf16
        checkpoint served quantized resides *less* than its bytes and has no bound here at all;
        the reverse (fp8 served bf16) resides more, which only makes a refusal safer.
        """
        return max(0.0, self.weight_gib - float(self.tied_embedding_gib)) / float(tp)


def gpu_memory_utilization(envelope_gib: float, reserve_gib: float,
                           device_total_gib: float,
                           headroom: float = HEADROOM_DEFAULT) -> float:
    """impl §4.2's formula. The chart consumes this; the solver computes it.

    `device_total_gib` is **what the driver reports as the size of one device**, because that is
    the number vLLM multiplies the utilisation it is handed by. It is required and has no
    default, and that is the whole point of this signature: a default here is a claim about a
    card, this function is the one place a utilisation is produced, and every KV pool on the node
    is `util * device_total - fixed`. A default that runs low hands the engine a utilisation that
    buys more memory than the packing priced, on every GPU, with nothing anywhere reporting it --
    which is what the constant this parameter used to default to did. The figure
    is read off a leased node by `agent/capture._observe` -- `memory.total` minus `memory.reserved`
    -- and filed in `store/hwfacts.HardwareFacts.total_gib`; there is nowhere else it may come from,
    and in particular not from a table keyed by product name.
    """
    if envelope_gib <= reserve_gib:
        raise ValueError(
            f"envelope {envelope_gib} GiB does not cover the {reserve_gib} GiB MPS reserve, "
            "so there is nothing left for weights or KV cache. The engine would fail vLLM's "
            "startup gate rather than run small.")
    return ((envelope_gib - reserve_gib) * headroom) / device_total_gib


def kv_gib_for_envelope(envelope_gib: float, tp: int, mem: ModelMemory,
                        headroom: float = HEADROOM_DEFAULT) -> float:
    """KV cache GiB per shard. Negative means the envelope cannot host the model at all."""
    reserve, _ = mem.reserve_gib(tp)
    return (envelope_gib - reserve) * headroom - mem.fixed_gib(tp)


def kv_tokens_for_envelope(envelope_gib: float, tp: int, mem: ModelMemory,
                           headroom: float = HEADROOM_DEFAULT) -> int:
    """KV cache tokens per shard. Floored, because a partial token is not a token."""
    kv_gib = kv_gib_for_envelope(envelope_gib, tp, mem, headroom)
    if kv_gib <= 0:
        return 0
    return int(kv_gib * BYTES_PER_GIB / mem.kv_bytes_per_token(tp))


def envelope_gib_for_kv_tokens(kv_tokens: int, tp: int, mem: ModelMemory,
                               headroom: float = HEADROOM_DEFAULT) -> float:
    """The inverse: the smallest envelope that yields at least `kv_tokens` per shard.

    This is what turns impl §9's "envelope >= weight footprint plus a minimum viable KV pool"
    into a number the solver can bound a variable with.
    """
    reserve, _ = mem.reserve_gib(tp)
    kv_gib = kv_tokens * mem.kv_bytes_per_token(tp) / BYTES_PER_GIB
    return (kv_gib + mem.fixed_gib(tp)) / headroom + reserve


def kv_tokens_per_mib(tp: int, mem: ModelMemory,
                      headroom: float = HEADROOM_DEFAULT) -> float:
    """Slope of the affine map, in tokens per MiB of envelope.

    Split out because CP-SAT needs the slope and the intercept as separate integer
    coefficients; see `cpsat._kv_tokens_expr`.
    """
    return headroom * BYTES_PER_GIB / (mem.kv_bytes_per_token(tp) * 1024.0)


def kv_tokens_intercept(tp: int, mem: ModelMemory,
                        headroom: float = HEADROOM_DEFAULT) -> float:
    """Constant term of the affine map, in tokens. Positive; it is subtracted."""
    reserve, _ = mem.reserve_gib(tp)
    fixed = reserve * headroom + mem.fixed_gib(tp)
    return fixed * BYTES_PER_GIB / mem.kv_bytes_per_token(tp)
