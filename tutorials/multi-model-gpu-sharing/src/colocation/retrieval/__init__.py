"""Retrieval: prior runs that bear on a candidate (impl §10).

**This is a query, not a model.** Given a candidate placement it returns the runs the store already
holds that bear on it -- each one intact: its run id, the assignment it actually ran, every tenant's
offered rate and sequence shape, its measured TTFT and ITL, and whether each tenant met its own
target, computed here at read time against the targets in force *now*. It returns nothing that was
not measured. It predicts nothing.

Why there is no predictor in this package, said once here so it is not re-argued at each call site:

A model that turns a candidate into an expected latency needs an equivalence class -- a key, a
bucket, a distance -- deciding which past runs speak for this candidate. Writing that rule makes the
similarity judgement **in advance and uniformly**, for every query, before anyone knows what any
particular query turns on. And it fails invisibly: a lookup returns a confident number for a
neighbourhood the store has never been near, and nothing in the answer says so.

Averaging into a cell destroys the very structure that decides the answer. The store holds two
placements that any co-residency count calls the same and that measured 1.89x apart, because *which*
tenant a shard shares a device with matters more than *how many* it shares with, and a count cannot
recover it. So retrieval returns runs. Where two runs disagree, both come back rather than their
mean. The judgement of which prior run speaks for this candidate is left to the reasoner, which can
make it per query, with the evidence in front of it, and be argued with.

What that costs, honestly (impl §10): there is no cheap offline ranking of candidates by expected
latency any more. Every belief about latency costs either a retrieved run or a measurement. What it
buys is that the beliefs are inspectable and that "no evidence" is sayable.

**The invariant.** Retrieval feeds proposals and cuts. It never decides feasibility -- that is the
solver's, on stated capacity and stated floors. A wrong belief about what a run implies must cost
iterations, not correctness.

`required_relaxation` reads a record the other way round from `targets_met` -- not "did it pass" but
"by how much would each target have to move for it to" -- and it is in the same category: arithmetic
over metrics already measured, against the targets in force now. It relaxes nothing and judges
nothing. What the caller does with the number is the caller's; `agent/__main__.py` prints it beside
every refused placement and claims none of them.

There is no cap on how many runs come back, and that is deliberate rather than an oversight
(impl §13). A cap would be the similarity rule this package refuses to write, only cruder: a
truncation by recency or by count instead of by relevance, with the same property of being applied
before anyone knows what the query turns on. Breadth is one summary line per run from `recall`;
depth is `read_run` on the ids that look worth opening, which spends the reasoner's context on the
runs the reasoner chose.

Dependencies point one way: `store/` and `solver/topology.py` (stdlib) and nothing else. In
particular **not** `colocation.inspect`, which imports `agent.mirror`; the agent imports retrieval,
so retrieval importing anything under `agent/` would close a cycle. A practical consequence worth
keeping: a record can be retrieved and judged with no OR-Tools and no cluster.
"""

from .recall import (
    EXCLUDED_KEY,
    EXCLUDED_SUSPECT,
    EXCLUDED_TENANTS,
    Evidence,
    Relation,
    RetrievedRun,
    TenantRun,
    coresident_pairs,
    placement_of,
    qualifiers,
    read_run,
    recall,
    required_relaxation,
    sm_caps_of,
    target_lines,
    targets_met,
)

__all__ = [
    "EXCLUDED_KEY",
    "EXCLUDED_SUSPECT",
    "EXCLUDED_TENANTS",
    "Evidence",
    "Relation",
    "RetrievedRun",
    "TenantRun",
    "coresident_pairs",
    "placement_of",
    "qualifiers",
    "read_run",
    "recall",
    "required_relaxation",
    "sm_caps_of",
    "target_lines",
    "targets_met",
]
