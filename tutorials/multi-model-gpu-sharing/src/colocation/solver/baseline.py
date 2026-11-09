"""The baseline co-location has to beat, and the two refutations that are free (impl §0).

§0 states the objective as *GPUs the fleet consumes while every tenant meets its target*, and the
baseline as the same objective with overlap forbidden: `sum(tp)` GPUs, one shard per GPU. Two things
follow that this module computes, and both are arithmetic on the fleet file -- no node, no
measurement, no GPU time:

  - **The TP degree memory alone forces.** A shard has to fit in one GPU together
    with the KV pool its own target requires, so for each tenant there is a smallest legal TP degree
    that fits at all. That is a *lower bound* on minimal TP and nothing more: the true minimal degree
    may be higher, because a degree that fits may still miss the latency target, and only a solo
    measurement settles that (`profile`). What the bound is good for is the other direction -- a
    degree *above* it is a claim about latency that somebody has to have measured, and a degree above
    it with no such measurement is exactly the mistake impl §0 was written about (an example fleet once had TP=4
    chosen "to leave room to share", when TP=2 fits). That is why the fleet file no longer lets a
    degree be typed in at all: it walks this floor upward against solo measurements instead
    (`agent/fleetfile.resolve_tp`).
  - **Whether a co-resident placement can exist at all.** Overlap means some GPU holds shards from
    two different tenants, so it requires at least one *pair* of tenants whose minimum envelopes fit
    together on one GPU. If no pair does, no co-resident placement is feasible at any envelope the
    solver could choose, and a search would spend a node-hour confirming the disjoint answer. This
    is the check that would have decided the 2026-09-04 run before it was launched.

Both are refutations rather than predictions, which is why they are worth stating this early: they
say what the search *cannot* find, and being wrong about that is not possible from arithmetic that
uses the solver's own feasibility test.

**The fits-on-one-GPU test is the solver's, not a second opinion.** It replicates `cpsat._build`'s
`lo > hi` check step for step -- same `envelope_gib_for_kv_tokens`, same quantization to
`envelope_step_mib`, same `int(device_total_gib * 1024)` capacity. A "fits" verdict here that the solver
then called infeasible would be worse than no verdict, so `envelope_step_mib` is a required argument:
there is no default to drift from the one the search actually runs at.

Pure standard library, like `memory.py`, and deliberately does not import `cpsat`: nothing here needs
CP-SAT, and a caller that only wants the baseline should not need OR-Tools to get it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .memory import HEADROOM_DEFAULT, envelope_gib_for_kv_tokens


def candidate_tp_degrees(gpu_count: int) -> tuple:
    """The TP degrees a tenant could be sharded at on a node with this many GPUs.

    Powers of two, and that is a statement about what is legal rather than a search shortcut.
    A TP group of degree N splits the attention heads N ways, so N has to divide the head count;
    every model on this stack has a power-of-two head count, and a fleet file does not state the
    attention-head count at all (only the *KV* head count, in `models.<name>.kv`). Enumerating
    non-powers-of-two would mean proposing degrees that vLLM may refuse at startup, which reads as
    a broken profile run rather than as an illegal shard count.

    A model whose head count admits, say, TP=3 is not reachable, and that is a real limitation
    rather than a shortcut: since the fleet file may not state a degree, this set is every degree any
    tenant can be resolved to. A fleet that needs TP=3 needs this function changed, which is the
    right place for that decision -- one list of legal degrees, checkable against the models.
    """
    out, k = [], 1
    while k <= int(gpu_count):
        out.append(k)
        k *= 2
    return tuple(out)


def min_envelope_gib(req, tp: int, headroom: float = HEADROOM_DEFAULT) -> float:
    """The smallest per-shard envelope that holds this tenant's KV floor at this TP degree."""
    return envelope_gib_for_kv_tokens(req.floor_tokens, int(tp), req.mem, headroom)


def envelope_steps(envelope_gib: float, envelope_step_mib: int) -> int:
    """The envelope in whole solver steps, rounded up. `cpsat._build`'s `lo`."""
    return math.ceil(envelope_gib * 1024 / int(envelope_step_mib))


def capacity_steps(device_total_gib: float, envelope_step_mib: int) -> int:
    """One GPU's capacity in whole solver steps, rounded down. `cpsat._build`'s `hi`."""
    return int(device_total_gib * 1024) // int(envelope_step_mib)


@dataclass(frozen=True)
class WeightFloor:
    """The narrowest TP degree the *weight* term alone permits, and what it ruled out (impl §14.2).

    Distinct from `TenantBaseline.floor_tp`, and the distinction is the whole of §14.2. That floor
    is where a *walk should start*: it prices the full envelope -- weights, activation, reserve, and
    a KV pool big enough for the tenant's floor tokens -- so it is the narrowest degree arithmetic
    expects to work, which makes it the cheapest first experiment. This floor is where a *refusal
    may rest*: below it the weights alone do not fit, so no measurement can rescue the degree and
    spending a round on it would be theatre. Between the two is a band of degrees that arithmetic
    doubts and only a boot can settle, and every degree in that band stays a candidate.

    `floor_tp` is None when even the widest candidate cannot hold the weights of one shard. That is
    the one case where "cannot be placed on this node" is a statement about the node.
    """

    model_id: str
    floor_tp: int | None
    # The bound at `floor_tp`, i.e. the least memory one shard can occupy there.
    bound_gib: float | None
    device_total_gib: float
    # (tp, bound GiB) for every candidate below `floor_tp`, ascending. Kept for the same reason
    # `TenantBaseline.ruled_out` is: "tp2 is the floor" is only checkable against tp1's number.
    ruled_out: tuple

    @property
    def why(self) -> str:
        """One line: what the weights alone settle."""
        if self.floor_tp is None:
            return (f"the weights of one shard exceed {self.device_total_gib:.2f} GiB at every "
                    f"candidate degree, so no shard count puts this model on this device")
        if not self.ruled_out:
            return (f"the weights alone permit tp{self.floor_tp}, the narrowest degree there is, "
                    f"so the weights eliminate nothing")
        worst = self.ruled_out[0]
        return (f"tp{self.floor_tp} holds {self.bound_gib:.2f} GiB of weights per shard of "
                f"{self.device_total_gib:.2f}; tp{worst[0]} would need {worst[1]:.2f} GiB of "
                f"weights alone, before any KV cache, activation or reserve")


def weight_floor_tp(mem, *, device_total_gib: float, candidates=None,
                    gpu_count: int | None = None) -> WeightFloor:
    """The narrowest candidate degree whose *weight bound* fits on one GPU.

    Ascending, first fit, and here monotonicity does hold rather than nearly holding: the bound is
    a constant divided by the degree, with none of the terms that make `memory_floor_tp` scan (no
    reserve that rises at TP>1, no measured activation that need not fall). Scanning anyway, because
    the candidate list is four long on this hardware and agreeing with the function above about how
    a floor is found is worth more than three saved divisions.

    Compared as floats against the raw device total, not in solver steps: steps round an envelope
    *up* to what the solver can allocate, and rounding up a bound makes a refusal stricter than the
    hardware does.
    """
    if candidates is None:
        if gpu_count is None:
            raise ValueError("pass either `candidates` or `gpu_count`")
        candidates = candidate_tp_degrees(gpu_count)
    floor_tp, bound, ruled_out = None, None, []
    for k in sorted(candidates):
        b = mem.weight_lower_bound_gib(k)
        if b <= float(device_total_gib):
            floor_tp, bound = k, b
            break
        ruled_out.append((k, b))
    return WeightFloor(model_id=mem.model_id, floor_tp=floor_tp, bound_gib=bound,
                       device_total_gib=float(device_total_gib), ruled_out=tuple(ruled_out))


@dataclass(frozen=True)
class TenantBaseline:
    """What memory alone says about one tenant's shard count.

    `resolved_tp` is the degree the request carries -- which, since `tp:` was deleted from the fleet
    file, is the degree `fleetfile.resolve_tp` measured for it (or, under `allow_unresolved_tp`, the
    memory floor standing in for a measurement). Nothing in this module knows which; it prices
    whatever degree it is handed, and the caller is the one that knows what the number means.

    `floor_tp` is None when no candidate degree fits -- the tenant cannot be hosted on this
    instance type at its own KV floor however it is sharded, which is a fact about the fleet and
    not an error here.
    """

    name: str
    resolved_tp: int
    floor_tp: int | None
    floor_tokens: int
    floor_detail: str
    envelope_at_resolved_gib: float
    envelope_at_floor_gib: float | None
    device_total_gib: float
    # (tp, required envelope GiB) for every candidate degree below `floor_tp`, in ascending order.
    # Kept because "tp2 is the floor" is only checkable against what tp1 would have needed.
    ruled_out: tuple

    @property
    def above_floor(self) -> bool:
        """True when this tenant carries more shards than the weights and the KV floor require.

        Arithmetic, and on its own it is neither good nor bad news. It was called `inflated` while
        the fleet file could state `tp:`, because a hand-written degree above the floor was §0's
        worked mistake -- the extra shards were GPUs the baseline paid for before any search
        started. With `tp:` deleted, the ordinary reason for a degree above the floor is that the
        floor degree was *measured* to miss the tenant's target, which is a baseline that is
        established rather than inflated. What still inflates one is a degree resolved while some
        narrower degree was never measured, and that is not visible from here: it lives in the
        resolution's `certain` flag (`fleetfile.upper_bound_tp`). So this property answers "wider
        than memory alone requires?" and nothing more; the caller decides what that is worth.
        """
        return self.floor_tp is not None and self.resolved_tp > self.floor_tp

    @property
    def excess_gpus(self) -> int:
        """GPUs this tenant's degree costs above the memory floor, per replica."""
        return max(0, self.resolved_tp - (self.floor_tp or self.resolved_tp))

    @property
    def why(self) -> str:
        """One line: what makes the floor the floor."""
        if self.floor_tp is None:
            return (f"no TP degree fits: even the widest candidate needs more than "
                    f"{self.device_total_gib:.2f} GiB per shard for {self.floor_tokens} KV tokens")
        at = (f"tp{self.floor_tp} needs {self.envelope_at_floor_gib:.2f} GiB/shard of "
              f"{self.device_total_gib:.2f} on the device")
        if not self.ruled_out:
            return f"{at}, so memory forces no sharding at all"
        prev_tp, prev_env = self.ruled_out[-1]
        return f"{at}; tp{prev_tp} would need {prev_env:.2f} GiB/shard, which does not fit"

    def as_dict(self) -> dict:
        return {"tenant": self.name, "resolved_tp": self.resolved_tp,
                "memory_floor_tp": self.floor_tp,
                "floor_tokens": self.floor_tokens, "floor_detail": self.floor_detail,
                "envelope_at_resolved_gib": round(self.envelope_at_resolved_gib, 3),
                "envelope_at_floor_gib": (None if self.envelope_at_floor_gib is None
                                          else round(self.envelope_at_floor_gib, 3)),
                "above_floor": self.above_floor, "excess_gpus": self.excess_gpus, "why": self.why}


@dataclass(frozen=True)
class SharePair:
    """Two tenants and whether one GPU can hold a shard of each."""

    a: str
    b: str
    envelope_a_gib: float
    envelope_b_gib: float
    device_total_gib: float
    fits: bool

    @property
    def total_gib(self) -> float:
        return self.envelope_a_gib + self.envelope_b_gib

    @property
    def deficit_gib(self) -> float:
        """How much over one GPU the pair is. Zero or negative when it fits."""
        return self.total_gib - self.device_total_gib

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "total_gib": round(self.total_gib, 3),
                "device_total_gib": round(self.device_total_gib, 3), "fits": self.fits,
                "deficit_gib": round(self.deficit_gib, 3)}


@dataclass(frozen=True)
class FleetBaseline:
    """The disjoint baseline for a whole fleet, plus what overlap would need.

    `resolved_gpus` is the baseline §0 names: the GPUs this fleet occupies with overlap forbidden,
    at the TP degrees its tenants resolved to. `floor_gpus` is what that baseline would be at the
    memory floor, and the gap between them is GPUs re-sharding alone would free -- which is why it is
    reported before the search rather than discovered after it. The gap is not by itself a fault: a
    resolved degree is above the floor precisely when the floor degree was measured to miss its
    target. It is a fault when nobody measured, and only the resolution knows that
    (`fleetfile.upper_bound_tp`).
    """

    tenants: tuple
    gpu_count: int
    device_total_gib: float
    pairs: tuple

    @property
    def resolved_gpus(self) -> int:
        return sum(t.resolved_tp for t in self.tenants)

    @property
    def floor_gpus(self) -> int:
        """`sum(memory floor tp)`. Tenants that fit at no degree are counted at their resolved
        degree, since there is no floor to count and dropping them would understate the baseline."""
        return sum((t.floor_tp or t.resolved_tp) for t in self.tenants)

    @property
    def above_floor(self) -> tuple:
        return tuple(t.name for t in self.tenants if t.above_floor)

    @property
    def unplaceable(self) -> tuple:
        return tuple(t.name for t in self.tenants if t.floor_tp is None)

    @property
    def fits_disjoint(self) -> bool:
        """Whether the fleet can be served with no GPU shared, at the resolved degrees."""
        return self.resolved_gpus <= self.gpu_count

    @property
    def floor_fits_disjoint(self) -> bool:
        return self.floor_gpus <= self.gpu_count

    @property
    def shareable(self) -> tuple:
        return tuple(p for p in self.pairs if p.fits)

    @property
    def can_coreside(self) -> bool:
        """Whether ANY co-resident placement is feasible, at any envelope.

        Overlap means one GPU holds shards of two different tenants, so a fleet where no pair fits
        together on one GPU has no co-resident placement -- not a hard one, none. A refutation, and
        the only claim in this module strong enough to cancel a run.
        """
        return bool(self.shareable)

    @property
    def tightest_pair(self) -> SharePair | None:
        """The pair closest to fitting. What a fleet would have to change to become shareable."""
        return min(self.pairs, key=lambda p: p.deficit_gib) if self.pairs else None

    def tenant(self, name: str) -> TenantBaseline:
        for t in self.tenants:
            if t.name == name:
                return t
        raise KeyError(f"no tenant named {name!r} in this baseline")

    def as_dict(self) -> dict:
        return {"gpu_count": self.gpu_count, "device_total_gib": round(self.device_total_gib, 3),
                "baseline_gpus_at_resolved_tp": self.resolved_gpus,
                "baseline_gpus_at_memory_floor": self.floor_gpus,
                "fits_disjoint": self.fits_disjoint,
                "tenants_above_floor": list(self.above_floor),
                "unplaceable_tenants": list(self.unplaceable),
                "coresidency_possible": self.can_coreside,
                "tightest_pair": (None if self.tightest_pair is None
                                  else self.tightest_pair.as_dict()),
                "tenants": [t.as_dict() for t in self.tenants]}


def memory_floor_tp(req, *, device_total_gib: float, envelope_step_mib: int,
                    candidates=None, gpu_count: int | None = None,
                    headroom: float = HEADROOM_DEFAULT) -> TenantBaseline:
    """The smallest candidate TP degree at which this tenant's shard fits on one GPU.

    Candidates are walked in ascending order and the first that fits wins; monotonicity is not
    assumed. It very nearly holds -- weights and per-token KV both divide by the degree -- but the
    per-shard reserve *rises* from 1.0 to 1.5 GiB at TP>1 (`memory.MPS_RESERVE_TP1_GIB`) and a
    measured `activation_gib` need not fall, so a wider degree can cost more per shard than a
    narrower one. Scanning is cheap and assuming is a silent wrong answer.
    """
    if candidates is None:
        if gpu_count is None:
            raise ValueError("pass either `candidates` or `gpu_count`")
        candidates = candidate_tp_degrees(gpu_count)
    hi = capacity_steps(device_total_gib, envelope_step_mib)
    floor_tp, floor_env, ruled_out = None, None, []
    for k in sorted(candidates):
        env = min_envelope_gib(req, k, headroom)
        if envelope_steps(env, envelope_step_mib) <= hi:
            floor_tp, floor_env = k, env
            break
        ruled_out.append((k, env))
    return TenantBaseline(
        name=req.name, resolved_tp=int(req.tp), floor_tp=floor_tp,
        floor_tokens=req.floor_tokens, floor_detail=req.floor_detail,
        envelope_at_resolved_gib=min_envelope_gib(req, req.tp, headroom),
        envelope_at_floor_gib=floor_env, device_total_gib=float(device_total_gib),
        ruled_out=tuple(ruled_out))


def share_pairs(requests, *, device_total_gib: float, envelope_step_mib: int,
                headroom: float = HEADROOM_DEFAULT) -> tuple:
    """Every unordered pair of tenants, and whether one GPU can hold a shard of each.

    Priced at each tenant's *minimum* envelope at its *resolved* TP degree, which is the most
    generous reading available: the solver may raise an envelope above its floor but never below,
    so a pair that does not fit here fits at no envelope the solver could choose.
    """
    reqs = list(requests)
    hi = capacity_steps(device_total_gib, envelope_step_mib)
    envs = {r.name: min_envelope_gib(r, r.tp, headroom) for r in reqs}
    steps = {r.name: envelope_steps(envs[r.name], envelope_step_mib) for r in reqs}
    out = []
    for i, a in enumerate(reqs):
        for b in reqs[i + 1:]:
            out.append(SharePair(a=a.name, b=b.name, envelope_a_gib=envs[a.name],
                                 envelope_b_gib=envs[b.name], device_total_gib=float(device_total_gib),
                                 fits=steps[a.name] + steps[b.name] <= hi))
    return tuple(out)


def fleet_baseline(node, requests, *, envelope_step_mib: int,
                   headroom: float = HEADROOM_DEFAULT) -> FleetBaseline:
    """The whole offline picture: the disjoint baseline, the memory floor, and whether overlap fits.

    `node` is a `cpsat.NodeSpec` (or anything with `gpu_ids` and `device_total_gib`), so this works on
    the *shape* of an instance type and needs no captured node.
    """
    gpu_count = len(node.gpu_ids)
    candidates = candidate_tp_degrees(gpu_count)
    tenants = tuple(memory_floor_tp(r, device_total_gib=node.device_total_gib,
                                    envelope_step_mib=envelope_step_mib,
                                    candidates=candidates, headroom=headroom)
                    for r in requests)
    return FleetBaseline(tenants=tenants, gpu_count=gpu_count,
                         device_total_gib=float(node.device_total_gib),
                         pairs=share_pairs(requests, device_total_gib=node.device_total_gib,
                                           envelope_step_mib=envelope_step_mib,
                                           headroom=headroom))


def at_tp(req, tp: int):
    """The same tenant request sharded differently. What a floor search and `profile` compare.

    A `TenantRequest` carries its KV floor in *tokens*, and the floor does not move with the shard
    count: at TP=N every shard holds all of the tokens and 1/N of each token's KV, so the token
    count is the invariant and `mem.kv_bytes_per_token(tp)` is what shrinks. That is why re-sharding
    is a one-field replacement here and not a re-derivation of the requirement.
    """
    return replace(req, tp=int(tp))
