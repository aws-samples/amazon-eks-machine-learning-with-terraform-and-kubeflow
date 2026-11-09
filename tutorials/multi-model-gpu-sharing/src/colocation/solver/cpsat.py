"""The placement solver (impl §9): OR-Tools CP-SAT, CPU-only, deterministic, prediction-free.

**Nothing here estimates what a placement will do.** There is no interference model, no ITL
table and no expected latency in the objective. Each tenant's latency ceiling enters as a *KV
floor* -- the cache a tenant must hold to serve its own stated rate and shape at its own stated
ITL (design §1.7 run in the requirement direction, `TenantTemplate.kv_tokens_at_target`) -- and a
floor is a constraint, not a forecast. This is what keeps the search's correctness independent of
anything's ability to predict interference: there is no predictor to be wrong.

The previous version of this file did the opposite. It built a linear surrogate for an occupancy
predictor, minimized it in CP-SAT, then re-scored every placement in Python against its actual
co-residency and ranked by predicted ITL. Both stages are gone. What survives is the algebra,
with a different constant in it: `maximize min_t (kv_tokens_t / floor_t)` has exactly the shape
the old `maximize min_t (kv_tokens_t / demand_t)` had, except that `floor_t` is a requirement the
tenant stated and `demand_t` was a number the model guessed.

**The objective is packing density, in strict priority order** (impl §9):

1. Minimize the number of GPUs the fleet occupies.
2. Among placements using that many, maximize the smallest margin any tenant holds over its own
   KV floor. Density alone would drive every tenant onto as few GPUs as the memory arithmetic
   permits, which is exactly the condition that breaks latency targets; this spends the leftover
   capacity where it is scarcest instead of leaving it wherever the search happened to put it.
3. Among those, maximize the total envelope allocated. Not cosmetic: max-min is indifferent to
   every tenant that is not the bottleneck, and without this pass it handed one fleet two tenants
   at 44.0 GiB and two at 20.5 GiB on equally empty GPUs -- a difference nothing could justify.
   Unused envelope buys nothing, so the tie-break is to allocate it.

Three solves rather than one weighted objective, because the weight that makes a single objective
strictly lexicographic spans it over more decimal digits than CP-SAT will close inside a
time budget. Split up it is milliseconds.

**The return value is a feasible set, not a ranking of predictions.** `FeasibleSet` carries the
candidates and, crucially, whether enumeration was **exhausted** or **truncated** at the cap. An
agent told the set is complete may conclude the fleet is infeasible; one told it was truncated may
not, and the difference has to be in the data rather than in a convention.

**The capacity constraint is an invariant, not a preference.** impl §9 is explicit: a violation is
a solver bug, not a runtime hazard. So `Placement.verify()` re-derives the per-GPU sums, the TP
distinctness and every KV floor in plain Python and raises `SolverBug` -- checking CP-SAT's answer
with arithmetic that shares no code with the model that produced it. That check earns its keep
because of §4.2.2: two over-subscribed tenants can both reach Ready and serve correct completions
while receiving 24.15 GiB and 14.23 GiB of KV cache, a 41% deficit decided by a ~1 s profiling
race. Nothing crashes, both report the `gpu_memory_utilization` the solver asked for, and the
resulting latency gap is indistinguishable from co-location interference. There is no runtime
signal to fall back on; prevention is the mechanism.

**A TTFT ceiling cannot be expressed here** -- impl §9. Prefill queueing under MPS is not a
function of the decision variables and is not approximated. A tenant that states only
`ttft_p95_ms` contributes no floor, and its target is checked by measurement alone.

**Determinism** comes from `num_workers = 1` and a fixed seed. Multi-worker CP-SAT races its
strategies, so with 8 workers the *same* model can return different equally-optimal placements
between runs. For a search whose whole record is "this candidate was proposed, then measured",
that would make the history unreproducible.

**No adjacency constraint**, and this is a simplifying assumption rather than a fact about
hardware. Design §1.10 takes GPU-to-GPU cost to be uniform -- on an NVSwitch node because every
pair is one hop, on a no-NVLink node because every pair is staged through the host -- so that no
subset of GPUs is a better home for a TP group than another. It is here to keep the model small,
and it is expected to be violated by some machines.

It is violated by g6e.48xlarge, measured 2026-09-04: `reason` at TP=4, uncontended, identical
envelope, ran 13.957 ms on GPUs 4-7 (run `search-rknwpq-000-disjoint-8gpu`) and 21.138 ms on
GPUs 2,3,4,5 (`...-003-chat-summ-pair`), a 1.51x gap at ~30x the run-to-run spread. The node has
two NUMA complexes, GPUs 0-3 and 4-7, and a TP group spanning both puts its collectives across
the socket interconnect.

Nothing was added here in response, on purpose. The assumption's job is to be cheap and
falsifiable, not correct: it costs the solver nothing when it is wrong, because the enumeration
is a scratchpad and not the search space -- `solve_pinned` accepts any assignment a reasoner
names, with every symmetry break switched off, which is how the placements that exposed this
were measured in the first place. Modelling the interconnect here would move a judgement about
one machine into code that must then be right about every machine. What the run does instead is
record the shard ordinals (`loop._provenance`) so the comparison stays visible in the store.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .memory import (ModelMemory, envelope_gib_for_kv_tokens, gpu_memory_utilization,
                     kv_tokens_for_envelope, kv_tokens_intercept, kv_tokens_per_mib,
                     HEADROOM_DEFAULT)
from .topology import coresidency

log = logging.getLogger(__name__)

# Envelope granularity. MPS enforces at a nominal limit written into an environment variable;
# sub-GiB precision buys no measurable KV cache and multiplies the candidate space, so the
# solver quantizes rather than searching every MiB.
ENVELOPE_STEP_MIB_DEFAULT = 512

# Fixed-point scales for CP-SAT, which is integer-only. MILLI carries the affine map's
# coefficients; MARGIN_SCALE carries the ratio `kv_tokens / floor_tokens`. Both chosen so every
# product stays far inside CP-SAT's +/-2**62 domain limit at fleet sizes this design targets.
MILLI = 1000
MARGIN_SCALE = 10_000


class SolverBug(Exception):
    """A returned placement violates a constraint the model was supposed to enforce.

    Deliberately not a warning and not a re-solve. impl §9 classifies a capacity violation as
    a solver bug precisely because the runtime does not reliably reject one.
    """


class Infeasible(Exception):
    """No placement satisfies the hard constraints. Carries why, when the reason is local."""


class Unproven(Infeasible):
    """The arithmetic found no placement, and the arithmetic is not entitled to say so (impl §14.2).

    A subclass rather than a sibling because every caller that catches `Infeasible` is asking "did
    the solver hand me a placement?", and the answer is no either way. What differs is the remedy,
    and the remedy is the whole point: an `Infeasible` means change the fleet -- fewer tokens, a
    smaller model, a lower rate -- while an `Unproven` means **go and measure**, because the numbers
    that produced it are the ones this stack has caught being wrong. The KV page size was 3.27x too
    large on the 9B and 6% too small on the 0.8B; the activation fallback over-reserved by 4.4x on
    the 0.6B; the reserve is a constant. A fleet closed on any of those is a fleet nobody ran.

    So `plan` exits 0 on this and points at a degree to boot, which is the same exit and the same
    advice it already gives for `BASELINE NOT ESTABLISHED` -- another "the arithmetic has gone as
    far as it can" state. Exiting 1 would read as "the fleet file has to change", which is the one
    conclusion that is not available here.
    """


@dataclass(frozen=True)
class NodeSpec:
    """The node being packed.

    `device_total_gib` is **what the driver reports as the size of one device** -- `memory.total`
    minus `memory.reserved`, read off a leased node by `agent/capture._observe` and filed in a
    `store/hwfacts` sheet. It has no default, on purpose, and no per-product table stands behind it.

    It is the only capacity figure on this stack, and it plays two roles that used to be played by
    two different numbers: it is the denominator of every tenant's `gpu_memory_utilization`, and it
    is the capacity the sum of envelopes on one GPU must fit inside. Two numbers there do not
    cancel. Whatever the smaller one under-counted was memory the engine handed out and the packing
    never priced -- in the direction that over-subscribes every GPU while reporting that it had
    not, which is the failure mode §9 makes an invariant against.

    Not the nameplate. A card's box, its marketing capacity and what its driver reports are three
    different figures; the driver is the one the engine asks, so the driver is the one this is.
    """

    gpu_ids: tuple
    device_total_gib: float

    @property
    def capacity_mib(self) -> int:
        return int(self.device_total_gib * 1024)


@dataclass(frozen=True)
class TenantRequest:
    """One model to place, with the memory facts needed to price it and the floors it must clear.

    `min_kv_tokens` is impl §9's "minimum viable KV pool". It is a workload statement, not a
    model property: a pool that cannot hold one request at `max_model_len` cannot serve that
    request at all, and a pool that cannot hold a few concurrent ones will spend the run
    preempting. The caller derives it -- `MIN_POOL_REQUESTS` requests at the tenant's own context
    window (`agent/spec.TenantTemplate.min_kv_tokens`) -- and no fleet file states it.

    `kv_floor_tokens` is impl §9's SLO-derived floor: the cache this tenant must hold to serve
    its own rate and shape at its own stated ITL ceiling. Zero means **no ITL ceiling was
    stated**, and zero is the right encoding for that -- a tenant with no target gets no floor
    from this rule and the solver has nothing to enforce on its behalf. It is deliberately not
    defaulted to something: an invented ceiling would be reported afterwards as a requirement
    somebody asked for. `floor_source` says where the number came from, and travels into the
    record, because a placement certified against a floor nobody stated is not certified.

    Note the direction the ceiling pushes: a *tighter* ITL ceiling implies a *smaller* floor,
    because requests that finish sooner occupy the cache for less time. It is the rate that
    demands memory.

    `isl_tokens` and `osl_tokens` are **this tenant's** shape, not the fleet's (design §8.3): the
    KV a tenant needs is `rate x osl x itl x (isl + osl)` and every factor is the tenant's own.
    """

    name: str
    model_id: str
    tp: int
    rate_hz: float
    isl_tokens: int
    osl_tokens: int
    mem: ModelMemory
    min_kv_tokens: int
    kv_floor_tokens: int = 0
    floor_source: str = "no ITL ceiling stated"

    @property
    def span_tokens(self) -> int:
        return self.isl_tokens + self.osl_tokens

    @property
    def floor_tokens(self) -> int:
        """The binding KV floor: the larger of the workload minimum and the SLO-derived one.

        One number rather than two constraints, because they are the same kind of claim -- "this
        tenant cannot be given less than this" -- and the solver has nothing useful to do with
        the distinction. `floor_detail` keeps the distinction for the record.
        """
        return max(int(self.min_kv_tokens), int(self.kv_floor_tokens))

    @property
    def floor_detail(self) -> str:
        """Which of the two floors binds, and where it came from. Recorded, never enforced."""
        if self.floor_tokens == 0:
            return "no floor: no ITL ceiling stated and no minimum viable pool set"
        if int(self.kv_floor_tokens) >= int(self.min_kv_tokens):
            return f"SLO floor {self.kv_floor_tokens} tokens ({self.floor_source})"
        return (f"minimum viable pool {self.min_kv_tokens} tokens, above the SLO floor of "
                f"{self.kv_floor_tokens} ({self.floor_source})")

    @property
    def signature(self) -> tuple:
        """What makes two tenants interchangeable, for symmetry breaking.

        The shape and the floor are in here for a reason that is easy to miss: two tenants of the
        same model at the same rate but different ISL, or the same shape but different ITL
        ceilings, are *not* interchangeable -- and treating them as such would let the symmetry
        break order them by GPU index and refuse the placement that puts the demanding one
        somewhere else.
        """
        return (self.model_id, self.tp, self.rate_hz, self.isl_tokens, self.osl_tokens,
                self.min_kv_tokens, self.kv_floor_tokens, self.mem.model_id,
                # The whole page, state term included: two tenants of one hybrid checkpoint at two
                # contexts have different pages and are NOT interchangeable, and the attention-only
                # field cannot see the difference (`solver/memory.ModelMemory`).
                self.mem.weight_gib, self.mem.page_bytes_per_token_tp1, self.mem.activation_gib)


@dataclass(frozen=True)
class Placement:
    """One feasible placement, with everything the chart and the record need."""

    assignment: dict            # name -> tuple of gpu ids
    envelope_gib: dict          # name -> per-shard envelope
    kv_tokens: dict             # name -> per-shard KV tokens, recomputed exactly
    util: dict                  # name -> gpu_memory_utilization for the chart
    mps_reserve_gib: dict       # name -> reserve used
    reserve_source: dict        # name -> "measured@tpN" or "default(tpN)"

    def gpu_load_mib(self) -> dict:
        out: dict = {}
        for name, gpus in self.assignment.items():
            for g in gpus:
                out[g] = out.get(g, 0.0) + self.envelope_gib[name] * 1024.0
        return out

    @property
    def gpus_used(self) -> int:
        """Distinct GPUs the fleet occupies. The solver's first objective, counted from the
        answer rather than read out of the model that produced it."""
        return len({g for gpus in self.assignment.values() for g in gpus})

    def verify(self, node: NodeSpec, requests) -> None:
        """Re-check every hard constraint in plain arithmetic. Raises `SolverBug`.

        Shares no code with the CP-SAT model on purpose. A bug in how a constraint was
        *expressed* is the failure mode this catches, and re-asking the same model would not.
        """
        by_name = {r.name: r for r in requests}
        for g, mib in self.gpu_load_mib().items():
            if mib > node.capacity_mib + 1e-6:
                raise SolverBug(
                    f"GPU {g} is promised {mib / 1024:.3f} GiB against a measured device total "
                    f"of {node.device_total_gib:.2f} GiB. impl §9 makes this an invariant, not a "
                    "preference: over-subscription need not crash -- two tenants can both "
                    "reach Ready and serve correctly with a 41% KV deficit between them "
                    "(§4.2.2), which then reads as co-location interference.")
        for name, gpus in self.assignment.items():
            r = by_name[name]
            if len(set(gpus)) != r.tp:
                raise SolverBug(
                    f"tenant {name} has tp={r.tp} but occupies {len(set(gpus))} distinct "
                    f"GPUs {gpus}. TP shards must be on distinct devices.")
            if self.kv_tokens[name] < r.floor_tokens:
                raise SolverBug(
                    f"tenant {name} gets {self.kv_tokens[name]} KV tokens per shard, below its "
                    f"floor of {r.floor_tokens}: {r.floor_detail}. A floor is impl §9's whole "
                    "substitute for predicting a latency, so a placement that violates one is "
                    "not a worse candidate -- it is an uncertified one wearing a ranking.")
            if self.envelope_gib[name] <= self.mps_reserve_gib[name]:
                raise SolverBug(
                    f"tenant {name}'s envelope {self.envelope_gib[name]} GiB does not clear "
                    f"its {self.mps_reserve_gib[name]} GiB MPS reserve.")


@dataclass(frozen=True)
class Candidate:
    """A feasible placement and the facts impl §9 says accompany it.

    No predicted latency, by construction. `margin` is what replaces it as the thing worth
    looking at: how much more KV cache than its own floor each tenant holds. It is a statement
    about capacity, not about what will happen -- a tenant at margin 3.0 may still miss its
    target, and only a measurement settles that.
    """

    placement: Placement
    margin: dict                # name -> kv_tokens / floor_tokens, or None if it has no floor
    floor_tokens: dict          # name -> the floor it was solved against
    floor_detail: dict          # name -> which floor bound, and where it came from
    coresidency: dict           # name -> per-GPU co-resident count actually achieved

    @property
    def gpus_used(self) -> int:
        return self.placement.gpus_used

    @property
    def min_margin(self) -> float | None:
        """The tightest margin in the fleet, or None when no tenant stated a floor.

        None rather than infinity: "every tenant has room to spare" and "nobody said how much
        room is needed" are different answers, and only one of them is about this placement.
        """
        held = [m for m in self.margin.values() if m is not None]
        return min(held) if held else None

    @property
    def tightest_tenant(self) -> str | None:
        held = {n: m for n, m in self.margin.items() if m is not None}
        return min(held, key=lambda n: held[n]) if held else None

    @property
    def rank_key(self) -> tuple:
        """Fewest GPUs first, then the largest smallest-margin. A tenant-free fleet sorts last."""
        m = self.min_margin
        return (self.gpus_used, -m if m is not None else float("inf"))


@dataclass(frozen=True)
class FeasibleSet:
    """What `solve` returns: the candidates, and whether that is all of them.

    impl §9 requires the distinction to be in the data: "an agent told the set is complete may
    conclude the fleet is infeasible; one told the set was truncated may not." So `exhausted` is
    only true when CP-SAT actually proved there is no further distinct assignment. A solve that
    ran out of time sets neither -- `exhausted` is false and `stopped_because` says so, because
    reporting a timeout as an exhausted search is precisely the lie the flag exists to prevent.

    Sequence-like, so callers that only want the candidates can index and iterate it.
    """

    candidates: tuple
    exhausted: bool
    cap: int
    stopped_because: str

    @property
    def truncated(self) -> bool:
        return not self.exhausted

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    def __getitem__(self, i):
        return self.candidates[i]

    def __bool__(self) -> bool:
        return bool(self.candidates)


@dataclass
class _Model:
    """CP-SAT model plus the handles the solve phases and extraction need."""
    m: object
    x: dict
    steps: dict
    floors: dict
    used: list
    worst: object          # min margin over tenants that have a floor, or None if none do
    margins: list


class PlacementSolver:
    """impl §9's solver. Construct with the node and the fleet -- and nothing else.

    There is deliberately no model parameter. The previous signature took an `OccupancyModel`,
    which meant a solver could not be built at all until something had been measured, and the
    first run of a new validity key had to be seeded with another key's ITL. A prediction-free
    solver needs no history: the floors come from the fleet's own targets.
    """

    def __init__(self, node: NodeSpec, requests,
                 envelope_step_mib: int = ENVELOPE_STEP_MIB_DEFAULT,
                 headroom: float = HEADROOM_DEFAULT):
        self.node = node
        self.requests = list(requests)
        self.step = int(envelope_step_mib)
        self.headroom = headroom
        # Only a *single tenant's* shards must be on distinct GPUs. Tenants share GPUs freely
        # -- that is the entire design. Comparing the fleet's total shard count against the
        # GPU count would reject every packed placement while looking like a sanity check.
        too_wide = [r for r in self.requests if r.tp > len(node.gpu_ids)]
        if too_wide:
            raise Infeasible(
                f"{', '.join(r.name for r in too_wide)} need TP degrees up to "
                f"{max(r.tp for r in too_wide)} on distinct GPUs, but the node has "
                f"{len(node.gpu_ids)}. No envelope assignment can rescue this.")

    # --- who may refuse ------------------------------------------------------

    def witnessed_infeasibility(self) -> str | None:
        """Why this fleet cannot be hosted *on the weights alone*, or None (impl §14.2).

        The question a fleet-level refusal has to answer before it is allowed to be one: is there a
        quantity here that cannot come back smaller than stated? Two of them are, and both are the
        weight bound (`ModelMemory.weight_lower_bound_gib`) rather than an envelope:

          1. some tenant's shard weighs more than one whole GPU, at the degree it is being priced
             at -- no packing and no measurement changes that;
          2. the fleet's weights, summed over every shard, exceed every GPU on the node put
             together -- likewise, and it is worth checking separately because a fleet can be
             impossible while no single tenant is.

        Neither reads a KV page size, an activation figure or a reserve, which is what makes them
        citable. Returning None does not mean the fleet fits; it means *this* function cannot prove
        it does not, so the caller owes the answer to a boot rather than to arithmetic.
        """
        total = self.node.device_total_gib
        for r in self.requests:
            b = r.mem.weight_lower_bound_gib(r.tp)
            if b > total:
                return (f"tenant {r.name} at tp{r.tp} puts {b:.2f} GiB of weights on one GPU, and "
                        f"one GPU has {total:.2f} GiB. Weights alone, with no KV cache, no "
                        f"activation and no reserve -- so no envelope and no measurement helps")
        node_total = total * len(self.node.gpu_ids)
        fleet = sum(r.mem.weight_lower_bound_gib(r.tp) * r.tp for r in self.requests)
        if fleet > node_total:
            return (f"the fleet's weights alone come to {fleet:.2f} GiB across "
                    f"{sum(r.tp for r in self.requests)} shard(s), and the node has "
                    f"{node_total:.2f} GiB across {len(self.node.gpu_ids)} GPU(s). Every tenant "
                    f"would have to be resident at once for the baseline to exist")
        return None

    # --- the model -----------------------------------------------------------

    def _check_avoid(self, avoid) -> tuple:
        """Validate agent-supplied cuts. Unknown tenant names raise rather than being ignored.

        impl §9 lets the agent forbid patterns -- "not these two on one GPU" -- and scopes the cut
        to the run. A cut naming a tenant that does not exist has to be an error: silently
        dropping it would leave the agent believing it had constrained a search that then handed
        back the very placement it ruled out, and the next iteration would be spent explaining a
        result the agent thinks is impossible.
        """
        known = {r.name for r in self.requests}
        out = []
        for group in avoid or ():
            names = tuple(str(n) for n in group)
            unknown = sorted(set(names) - known)
            if unknown:
                raise ValueError(
                    f"the cut {names} names {unknown}, which this fleet does not have. It has "
                    f"{sorted(known)}. A cut that matches nothing would be applied silently and "
                    "the search would look like it had ignored you.")
            if len(set(names)) < 2:
                raise ValueError(
                    f"the cut {names} needs at least two distinct tenants: it means \"no two of "
                    "these share a GPU\", which is vacuous for one.")
            out.append(tuple(sorted(set(names))))
        return tuple(out)

    def _build(self, forbidden=(), pinned=None, avoid=()):
        """Build a fresh model, with `forbidden` assignments and `avoid` cuts applied.

        Rebuilt per candidate rather than mutated, because each lexicographic phase adds a bound
        on the previous phase's optimum and those bounds are only valid for one candidate.
        Rebuilding costs milliseconds; carrying a stale bound would silently cap every later
        candidate at the first one's margin.

        `pinned` is `{name: (gpu_id, ...)}` and fixes the assignment, leaving only the envelopes
        to optimize. **All three symmetry breaks are switched off when it is given**, and that is
        not an optimization detail: they are valid only up to relabelling GPUs, so against a
        caller-supplied assignment they would reject a perfectly good one -- pinning tenant 0 to
        GPU 3 contradicts the prefix cut and comes back INFEASIBLE with nothing to say why.
        """
        m = cp_model.CpModel()
        G = len(self.node.gpu_ids)
        cap = self.node.capacity_mib
        avoid = self._check_avoid(avoid)

        steps, x, kv, margin, min_gpu, margin_hi, floors = {}, {}, {}, {}, {}, {}, {}
        prefix = 0
        for r in self.requests:
            floor = r.floor_tokens
            floors[r.name] = floor
            min_env = envelope_gib_for_kv_tokens(floor, r.tp, r.mem, self.headroom)
            lo = math.ceil(min_env * 1024 / self.step)
            hi = cap // self.step
            if lo > hi:
                # The envelope does not fit. Whether that is a refusal or a question depends on
                # which term put it over (impl §14.2): if the weights alone are over, nothing can
                # rescue the degree and saying so costs no GPU time. Otherwise the overflow is in
                # the KV pool, the activation or the reserve -- the three terms this stack has
                # caught being wrong -- and the honest answer is that it has not been measured.
                bound = r.mem.weight_lower_bound_gib(r.tp)
                common = (f"tenant {r.name} needs a {min_env:.2f} GiB envelope for its KV floor of "
                          f"{floor} tokens per shard ({r.floor_detail}), but one GPU only has "
                          f"{self.node.device_total_gib:.2f} GiB on it.")
                if bound > self.node.device_total_gib:
                    raise Infeasible(
                        f"{common} Its weights alone are {bound:.2f} GiB per shard at tp{r.tp}, so "
                        "the envelope is beside the point. Raise the TP degree or use a smaller "
                        "model -- packing cannot help, and neither can a measurement.")
                raise Unproven(
                    f"{common} Its weights are only {bound:.2f} GiB per shard at tp{r.tp}, so what "
                    f"does not fit is the {min_env - bound:.2f} GiB of KV pool, activation and "
                    f"reserve above them -- and those are computed, not measured. Boot this tenant "
                    f"alone at tp{r.tp} before believing it: `profile` settles this, and until it "
                    "does nothing here has ruled the degree out.")
            steps[r.name] = m.NewIntVar(lo, hi, f"env_steps[{r.name}]")

            # Assignment. `sum == tp` with one boolean per (tenant, GPU) makes "TP shards on
            # distinct GPUs" structural rather than a constraint that could be forgotten.
            x[r.name] = [m.NewBoolVar(f"x[{r.name},{g}]") for g in range(G)]
            m.Add(sum(x[r.name]) == r.tp)

            if pinned is not None:
                want = pinned.get(r.name)
                if want is None:
                    raise Infeasible(
                        f"a pinned assignment must name every tenant; {r.name} is missing. "
                        "Leaving one free would let the solver move it while the others are "
                        "held, which is a different experiment than the one being asked for.")
                if len(set(want)) != r.tp:
                    raise Infeasible(
                        f"pinned assignment gives {r.name} GPUs {tuple(want)} but its TP "
                        f"degree is {r.tp}: one shard per GPU, on distinct GPUs.")
                for g in range(G):
                    m.Add(x[r.name][g] == (1 if self.node.gpu_ids[g] in want else 0))
            else:
                # Symmetry break: with interchangeable GPUs (design §1.10) any solution can be
                # relabelled so tenant i uses only GPUs below the running total of TP degrees.
                # Without this, the candidate set fills with relabellings of one placement.
                # Inherits the uniform-topology assumption in this module's header, so on a
                # machine where ordinals differ it prunes placements that are not in fact
                # equivalent. That costs nothing but enumeration: `solve_pinned` reaches them.
                prefix += r.tp
                for g in range(prefix, G):
                    m.Add(x[r.name][g] == 0)

            # KV tokens, affine in the envelope. `<=` not `==`: the coefficients are floored
            # and the intercept ceiled, so `kv` is a conservative lower bound and the
            # maximizing objectives push it to the floor of the true value.
            a = int(kv_tokens_per_mib(r.tp, r.mem, self.headroom) * MILLI)
            b = math.ceil(kv_tokens_intercept(r.tp, r.mem, self.headroom) * MILLI)
            kv_hi = kv_tokens_for_envelope(hi * self.step / 1024.0, r.tp, r.mem, self.headroom)
            kv[r.name] = m.NewIntVar(floor, max(kv_hi, floor), f"kv[{r.name}]")
            m.Add(MILLI * kv[r.name] <= a * self.step * steps[r.name] - b)

            # Margin over the floor, scaled. `margin * floor <= MARGIN_SCALE * kv` with `floor` a
            # constant is linear, so maximizing the minimum margin is exact rather than an
            # approximation of something. A tenant with no floor contributes no margin variable:
            # there is no ratio to hold, and inventing a denominator of 1 would make an
            # unconstrained tenant look like the most comfortable one in the fleet.
            if floor > 0:
                margin_hi[r.name] = max(1, (MARGIN_SCALE * max(kv_hi, floor)) // floor)
                margin[r.name] = m.NewIntVar(0, margin_hi[r.name], f"margin[{r.name}]")
                m.Add(margin[r.name] * floor <= MARGIN_SCALE * kv[r.name])

            # Lowest GPU index this tenant occupies, for the identical-tenant break below.
            aux = []
            for g in range(G):
                v = m.NewIntVar(0, G, f"aux[{r.name},{g}]")
                m.Add(v == g).OnlyEnforceIf(x[r.name][g])
                m.Add(v == G).OnlyEnforceIf(x[r.name][g].Not())
                aux.append(v)
            min_gpu[r.name] = m.NewIntVar(0, G, f"min_gpu[{r.name}]")
            m.AddMinEquality(min_gpu[r.name], aux)

        # Per-GPU capacity. The invariant. `env_on` is the envelope when the tenant is on the
        # GPU and zero otherwise -- a linear stand-in for `envelope * assigned`, which would
        # otherwise be a product of two variables.
        for g in range(G):
            terms = []
            for r in self.requests:
                on = m.NewIntVar(0, cap, f"env_on[{r.name},{g}]")
                m.Add(on == self.step * steps[r.name]).OnlyEnforceIf(x[r.name][g])
                m.Add(on == 0).OnlyEnforceIf(x[r.name][g].Not())
                terms.append(on)
            m.Add(sum(terms) <= cap)

        # Whether each GPU hosts anything. Built even when pinned, because it is what the first
        # objective counts -- under a pin that objective is a constant, which is harmless.
        used = []
        for g in range(G):
            u = m.NewBoolVar(f"used[{g}]")
            m.AddMaxEquality(u, [x[r.name][g] for r in self.requests])
            used.append(u)

        if pinned is None:
            # Symmetry break: GPUs are interchangeable, so fill them in order. The strongest of
            # the three, and the first to mislead on a machine that is not uniform -- a placement
            # that leaves a hole to keep a TP group on one side of the node is not enumerable
            # here. Same escape hatch: name the assignment and it is measured as given.
            for g in range(1, G):
                m.Add(used[g] <= used[g - 1])

            # Symmetry break: two interchangeable tenants are ordered by their lowest GPU, so
            # the candidate set does not fill with mirror images of one placement.
            for i in range(len(self.requests) - 1):
                a_r, b_r = self.requests[i], self.requests[i + 1]
                if a_r.signature == b_r.signature:
                    m.Add(min_gpu[a_r.name] <= min_gpu[b_r.name])

        # Agent-supplied cuts (impl §9). Run-scoped and never persisted: a cut is a belief the
        # agent formed from retrieved runs, and a mistaken belief must cost iterations rather
        # than correctness. Nothing here writes them anywhere.
        for group in avoid:
            for g in range(G):
                m.Add(sum(x[name][g] for name in group) <= 1)

        # Assignment patterns already returned. Cut here rather than mutated in, so that the
        # k-th model is built once with all k-1 cuts present.
        for pattern in forbidden:
            ones = [x[name][g] for name, g in pattern]
            m.Add(sum(ones) <= len(ones) - 1)

        worst = None
        if margin:
            worst = m.NewIntVar(0, max(margin_hi.values()), "min_margin")
            m.AddMinEquality(worst, list(margin.values()))
        return _Model(m, x, steps, floors, used, worst, list(margin.values()))

    # --- solving -------------------------------------------------------------

    def _new_solver(self, max_seconds: float):
        solver = cp_model.CpSolver()
        # Determinism, not speed. See the module docstring.
        solver.parameters.num_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = max_seconds
        return solver

    def _optimize(self, solver, b) -> tuple:
        """The three lexicographic passes. Returns `(ok, status)`; `status` is the last one.

        Shared by `solve` and `solve_pinned` so the two cannot drift: a pinned run that optimized
        only the first objective would hand every non-bottleneck tenant an arbitrary envelope,
        and a co-residency comparison would then be varying two things at once.
        """
        def failed(phase: int, status) -> None:
            raise SolverBug(
                f"phase {phase} returned {solver.StatusName(status)} on a model an earlier "
                "phase had already satisfied. Bounding an objective at its own optimum cannot "
                "remove every solution, so the bound or the objective is wrong.")

        # 1. Fewest GPUs. The only phase that may legitimately find nothing.
        b.m.Minimize(sum(b.used))
        status = solver.Solve(b.m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return False, status
        # Values read one variable at a time rather than off the objective, so this does not
        # depend on `CpSolver.Value` accepting a linear expression.
        b.m.Add(sum(b.used) <= sum(solver.Value(u) for u in b.used))

        # 2. Largest smallest margin over the KV floors. Skipped entirely when no tenant stated
        # one: there is no ratio to maximize, and the third phase still allocates the capacity.
        if b.worst is not None:
            b.m.Maximize(b.worst)
            status = solver.Solve(b.m)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                failed(2, status)
            b.m.Add(b.worst >= solver.Value(b.worst))

        # 3. Spend what is left. Unused envelope buys nothing.
        b.m.Maximize(sum(b.steps.values()))
        status = solver.Solve(b.m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            failed(3, status)
        return True, status

    def _extract(self, solver, b) -> Placement:
        assignment, env, kv, util, reserve, rsource = {}, {}, {}, {}, {}, {}
        for r in self.requests:
            gpus = tuple(self.node.gpu_ids[g] for g in range(len(self.node.gpu_ids))
                         if solver.Value(b.x[r.name][g]))
            assignment[r.name] = gpus
            e = solver.Value(b.steps[r.name]) * self.step / 1024.0
            env[r.name] = e
            res, src = r.mem.reserve_gib(r.tp)
            reserve[r.name], rsource[r.name] = res, src
            # Recomputed in floating point rather than read out of the integer model: the
            # solver's `kv` is a floored lower bound, and the chart and the record should both
            # see the real number.
            kv[r.name] = kv_tokens_for_envelope(e, r.tp, r.mem, self.headroom)
            util[r.name] = gpu_memory_utilization(e, res, self.node.device_total_gib, self.headroom)
        return Placement(assignment=assignment, envelope_gib=env, kv_tokens=kv, util=util,
                         mps_reserve_gib=reserve, reserve_source=rsource)

    def _describe(self, p: Placement) -> Candidate:
        """Attach the facts impl §9 says travel with a placement. No prediction is made here."""
        margin, floors, detail = {}, {}, {}
        for r in self.requests:
            floor = r.floor_tokens
            floors[r.name] = floor
            detail[r.name] = r.floor_detail
            margin[r.name] = (p.kv_tokens[r.name] / floor) if floor > 0 else None
        return Candidate(placement=p, margin=margin, floor_tokens=floors, floor_detail=detail,
                         coresidency=coresidency(dict(p.assignment)))

    def solve_pinned(self, assignment: dict, max_seconds: float = 10.0) -> Candidate:
        """Best envelopes for one caller-chosen assignment. impl §9's pinned feasibility check.

        This is how the agent asks "can this placement be hosted at all, and with what room to
        spare" about a placement the enumeration did not offer. It fixes *which GPUs are shared*
        while leaving everything else at its best -- otherwise a difference in a later
        measurement could just as well come from one arrangement having been handed worse
        envelopes. Same objectives as `solve`, same verification.
        """
        solver = self._new_solver(max_seconds)
        b = self._build(pinned=assignment)
        ok, status = self._optimize(solver, b)
        if not ok:
            witness = self.witnessed_infeasibility()
            base = (f"no envelope assignment fits the pinned placement {assignment} "
                    f"({solver.StatusName(status)}). Every GPU's envelope sum must stay within "
                    f"{self.node.device_total_gib:.2f} GiB while each tenant clears its own KV "
                    "floor -- with the assignment fixed, packing cannot help.")
            if witness is not None:
                raise Infeasible(f"{base} And it is not the envelopes: {witness}.")
            raise Unproven(
                f"{base} No tenant's weights exceed a GPU, so what did not fit is KV pool, "
                "activation and reserve -- computed terms (impl §14.2). This placement is "
                "unproven rather than impossible; measure it before discarding it.")
        p = self._extract(solver, b)
        p.verify(self.node, self.requests)
        got = {n: tuple(sorted(g)) for n, g in p.assignment.items()}
        want = {n: tuple(sorted(g)) for n, g in assignment.items()}
        if got != want:
            raise SolverBug(
                f"solve_pinned was asked for {want} and returned {got}. The pin is expressed "
                "as equality constraints on the assignment booleans, so a mismatch means the "
                "pin was not applied -- and measuring a different placement than was recorded "
                "would corrupt the store.")
        return self._describe(p)

    def solve(self, cap: int = 3, max_seconds: float = 10.0, avoid=()) -> FeasibleSet:
        """The feasible set, up to `cap` members, in packing-density order.

        Each candidate is found by solving, then forbidding that exact *assignment* (not its
        envelopes) and solving again. So the members are distinct placement structures, each with
        its own best envelope split, rather than near-duplicates differing by one envelope step.
        Because every solve re-optimizes the same lexicographic objective over what is left, the
        members come out already ordered -- fewest GPUs first, then largest smallest-margin.

        `avoid` is a list of tenant-name groups meaning "no two of these share a GPU". Run-scoped
        (impl §9): nothing derived from a cut is written to the store or may become a hard
        constraint on a later run.

        `max_seconds` applies per solve, and a timeout is not an error -- CP-SAT returns the best
        placement it found, which is feasible and verified. It just may not be optimal, which is
        why the returned set says how enumeration ended rather than leaving it to be assumed.
        """
        solver = self._new_solver(max_seconds)
        out: list = []
        forbidden: list = []
        exhausted = False
        why = f"the cap of {cap} was reached; there may be more"
        for _ in range(max(1, int(cap))):
            b = self._build(forbidden, avoid=avoid)
            ok, status = self._optimize(solver, b)
            if not ok:
                if not out:
                    witness = self.witnessed_infeasibility()
                    base = (f"CP-SAT found no placement ({solver.StatusName(status)}). Every GPU's "
                            "envelope sum must stay within the measured device total while each "
                            "tenant clears its own KV floor; on this node those cannot both hold "
                            "for this fleet.")
                    if witness is not None:
                        raise Infeasible(f"{base} And the weights alone settle it: {witness}.")
                    raise Unproven(
                        f"{base} The weights alone do fit, so what closed the fleet is KV pool, "
                        "activation and reserve -- computed terms (impl §14.2), and the KV page "
                        "size behind them was 3.27x too large the one time it was checked against "
                        "a booted engine. Nothing here has ruled this fleet out; `profile` can.")
                # Only a proved INFEASIBLE means there is nothing left. A timeout means the
                # search ran out of budget, and calling that "exhausted" is the exact
                # misreading the flag exists to prevent (impl §9).
                exhausted = status == cp_model.INFEASIBLE
                why = ("every distinct assignment has been enumerated"
                       if exhausted else
                       f"the solve returned {solver.StatusName(status)} after {len(out)} "
                       f"candidate(s), so enumeration stopped without proving there are no more")
                break
            p = self._extract(solver, b)
            p.verify(self.node, self.requests)
            out.append(self._describe(p))
            forbidden.append([(r.name, g) for r in self.requests
                              for g in range(len(self.node.gpu_ids))
                              if solver.Value(b.x[r.name][g])])

        # Ordered by construction, but sorted anyway: a per-solve timeout can return a feasible
        # non-optimal candidate, and then the enumeration order is not the ranking. Sorting makes
        # the returned order true whatever happened, and costs nothing at these sizes.
        out.sort(key=lambda c: c.rank_key)
        best = out[0]
        log.info("solver returned %d candidate(s) on %d GPUs; tightest margin %s (%s); %s",
                 len(out), best.gpus_used,
                 "none stated" if best.min_margin is None else f"{best.min_margin:.3f}x",
                 best.tightest_tenant or "-", why)
        return FeasibleSet(tuple(out), exhausted=exhausted, cap=int(cap), stopped_because=why)
