"""Splitting a fleet over up to `max_nodes` nodes of one instance type, each tenant on one node.

**A group is a fleet.** Nodes share no GPU, so nothing a tenant does on one node reaches a tenant on
another except through the network its Service sits on. A split is therefore a set of groups, and
each group is an ordinary single-node fleet: placed by `cpsat.PlacementSolver` on one node, measured
on one leased node, keyed and recalled as a fleet of exactly those tenants. Nothing below the split
changes, and this module adds nothing to what a group is.

**Two stages, so that the search does not solve every subset.** Solving each candidate group on
its own is exponential in the tenant count: a 15-tenant fleet over two nodes has 16384 groupings,
and did not finish in ten minutes. So one CP-SAT model chooses which node each tenant is on, with
every tenant's envelope at its own floor -- the same `lo` that `PlacementSolver._build` computes,
at the same step and the same capacity, so a split admitted here is one the solver can place. Then
each group of each split it returns is placed by `PlacementSolver`, unchanged, which is where the
margins and the envelopes come from.

**What it settles is only what arithmetic can.** Which split meets its targets is a measurement, as
it is for a placement inside one node, and the ranking here predicts none of it: fewest nodes, then
fewest GPUs, then the largest smallest-margin over every group -- `Candidate.rank_key` with the node
count in front. Choosing which split to measure is the reasoner's, among the ones listed here (impl
§9, §11). A refusal is a fact only when the weights alone make it (impl §14.2), so "no split
exists" is said only when the same model with every envelope at its weights is infeasible too.

Pure, like `cpsat`: no node, no cluster, no store.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .cpsat import (ENVELOPE_STEP_MIB_DEFAULT, Candidate, Infeasible, NodeSpec, PlacementSolver,
                    Unproven)
from .memory import HEADROOM_DEFAULT, envelope_gib_for_kv_tokens

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Group:
    """Some tenants, on one node, with the densest placement the solver found for them there."""

    tenants: tuple              # names, in fleet order
    candidate: Candidate

    @property
    def gpus(self) -> int:
        return self.candidate.gpus_used

    @property
    def min_margin(self) -> float | None:
        return self.candidate.min_margin


@dataclass(frozen=True)
class Split:
    """Every tenant in exactly one group, and every group on its own node."""

    groups: tuple

    @property
    def nodes(self) -> int:
        return len(self.groups)

    @property
    def gpus(self) -> int:
        return sum(g.gpus for g in self.groups)

    @property
    def min_margin(self) -> float | None:
        held = [g.min_margin for g in self.groups if g.min_margin is not None]
        return min(held) if held else None

    @property
    def rank_key(self) -> tuple:
        """Fewest nodes, then fewest GPUs, then the largest smallest-margin; names break ties."""
        m = self.min_margin
        return (self.nodes, self.gpus, -m if m is not None else float("inf"),
                tuple(g.tenants for g in self.groups))


@dataclass(frozen=True)
class Splits:
    """What `split` returns: the ranked splits, and how much of the space they are."""

    splits: tuple               # ranked
    max_nodes: int
    exhausted: bool             # True when there are no splits beyond these
    unplaced: dict              # tenant -> why it fits no node even alone
    proven_none: bool           # no split, and the weights alone say so


def disjoint_nodes(tps, gpu_count: int) -> int:
    """Nodes the disjoint baseline needs: every shard on a GPU of its own, each tenant on one node.

    First-fit decreasing, which is exact here: every degree is a power of two no wider than the
    node (`baseline.candidate_tp_degrees`), so each size divides every larger one and FFD leaves no
    node with room another shard could have used.
    """
    free: list = []
    for tp in sorted(tps, reverse=True):
        for i, room in enumerate(free):
            if room >= tp:
                free[i] -= tp
                break
        else:
            free.append(gpu_count - tp)
    return len(free)


def _floor_mib(r, step: int, headroom: float, weights_only: bool) -> int:
    """One shard's envelope at the tenant's floor, in MiB, quantized as `PlacementSolver._build`
    quantizes it. With `weights_only`, the weights alone: the bound nothing can come in under."""
    gib = (r.mem.weight_lower_bound_gib(r.tp) if weights_only
           else envelope_gib_for_kv_tokens(r.floor_tokens, r.tp, r.mem, headroom))
    return math.ceil(gib * 1024 / step) * step


def _split_model(node, requests, max_nodes, step, headroom, weights_only, forbidden,
                 apart=(), together=()):
    """Which node each tenant is on, with each shard at its floor envelope on its own GPU."""
    m = cp_model.CpModel()
    G, N, cap = len(node.gpu_ids), max_nodes, node.capacity_mib
    need = {r.name: _floor_mib(r, step, headroom, weights_only) for r in requests}
    z = {r.name: [m.NewBoolVar(f"z[{r.name},{n}]") for n in range(N)] for r in requests}
    x = {r.name: [[m.NewBoolVar(f"x[{r.name},{n},{g}]") for g in range(G)] for n in range(N)]
         for r in requests}
    for r in requests:
        m.AddExactlyOne(z[r.name])
        for n in range(N):
            m.Add(sum(x[r.name][n]) == r.tp * z[r.name][n])
    used = [[m.NewBoolVar(f"used[{n},{g}]") for g in range(G)] for n in range(N)]
    for n in range(N):
        for g in range(G):
            m.Add(sum(need[r.name] * x[r.name][n][g] for r in requests) <= cap)
            m.AddMaxEquality(used[n][g], [x[r.name][n][g] for r in requests])
            # GPUs on one node are interchangeable (design §1.10), so they fill in order.
            if g:
                m.Add(used[n][g] <= used[n][g - 1])
    # Nodes are interchangeable too, so each split has one labelling: a tenant may open node n only
    # if an earlier tenant is on node n-1. That is what lets a cut on `z` forbid a split rather
    # than one of its N! relabellings.
    for i, r in enumerate(requests):
        for n in range(1, N):
            m.Add(z[r.name][n] <= sum(z[q.name][n - 1] for q in requests[:i]))
    for pattern in forbidden:
        m.Add(sum(z[name][n] for name, n in pattern) <= len(pattern) - 1)
    # The reasoner's, after a group it measured: pairs that must not share a node, or must.
    for a, b in apart:
        for n in range(N):
            m.Add(z[a][n] + z[b][n] <= 1)
    for a, b in together:
        for n in range(N):
            m.Add(z[a][n] == z[b][n])
    nodes = [used[n][0] for n in range(N)]
    return m, z, nodes, [u for row in used for u in row]


def _solve_split(node, requests, max_nodes, step, headroom, max_seconds, forbidden,
                 weights_only=False, apart=(), together=()):
    """`{tenant: node index}` for the split with fewest nodes, then fewest GPUs; None if none."""
    m, z, nodes, gpus = _split_model(node, requests, max_nodes, step, headroom, weights_only,
                                     forbidden, apart, together)
    solver = cp_model.CpSolver()
    # Determinism, as in `cpsat`.
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = max_seconds
    m.Minimize(sum(nodes))
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, status
    m.Add(sum(nodes) <= sum(solver.Value(v) for v in nodes))
    m.Minimize(sum(gpus))
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, status
    return {r.name: next(n for n in range(max_nodes) if solver.Value(z[r.name][n]))
            for r in requests}, status


def _group(node, members, step, max_seconds):
    try:
        feasible = PlacementSolver(node, members, envelope_step_mib=step).solve(
            cap=1, max_seconds=max_seconds)
    except Infeasible:      # and Unproven: not expected at a floor the split model admitted
        return None
    return Group(tenants=tuple(r.name for r in members), candidate=feasible[0])


def split(node: NodeSpec, requests, max_nodes: int, *, cap: int = 5, max_seconds: float = 10.0,
          envelope_step_mib: int = ENVELOPE_STEP_MIB_DEFAULT,
          headroom: float = HEADROOM_DEFAULT, apart=(), together=()) -> Splits:
    """Up to `cap` splits of `requests` over at most `max_nodes` nodes shaped like `node`, best
    first. `max_seconds` applies per solve, as it does in `PlacementSolver.solve`.

    `apart` and `together` are pairs of tenant names that must be on different nodes, or the same
    one. They are constraints like any other, so a proof that no split exists is a proof under them.
    """
    if max_nodes < 1:
        raise ValueError(f"max_nodes is at least 1; got {max_nodes}")
    requests = list(requests)
    names = {r.name for r in requests}
    for a, b in list(apart) + list(together):
        if a not in names or b not in names or a == b:
            raise ValueError(f"({a}, {b}) is not a pair of two of this fleet's tenants")
    step = int(envelope_step_mib)
    unplaced = {}
    for r in requests:
        try:
            PlacementSolver(node, [r], envelope_step_mib=step).solve(cap=1, max_seconds=max_seconds)
        except Infeasible as exc:
            unplaced[r.name] = (str(exc), not isinstance(exc, Unproven))
    if unplaced:
        return Splits(splits=(), max_nodes=max_nodes, exhausted=True,
                      unplaced={n: why for n, (why, _) in unplaced.items()},
                      proven_none=any(proven for _, proven in unplaced.values()))
    found, forbidden, exhausted = [], [], False
    for _ in range(max(1, int(cap))):
        where, status = _solve_split(node, requests, max_nodes, step, headroom, max_seconds,
                                     forbidden, apart=apart, together=together)
        if where is None:
            # As in `cpsat.solve`: only a proved INFEASIBLE means there are no more.
            exhausted = status == cp_model.INFEASIBLE
            break
        forbidden.append(list(where.items()))
        groups = [_group(node, [r for r in requests if where[r.name] == n], step, max_seconds)
                  for n in sorted(set(where.values()))]
        if all(groups):
            found.append(Split(tuple(groups)))
    proven_none = False
    if not found and exhausted:
        where, _ = _solve_split(node, requests, max_nodes, step, headroom, max_seconds, [],
                                weights_only=True, apart=apart, together=together)
        proven_none = where is None
    found.sort(key=lambda s: s.rank_key)
    log.info("%d split(s) of %d tenants over at most %d node(s); %s", len(found), len(requests),
             max_nodes, "no more exist" if exhausted else "there may be more")
    return Splits(splits=tuple(found), max_nodes=int(max_nodes), exhausted=exhausted,
                  unplaced={}, proven_none=proven_none)
