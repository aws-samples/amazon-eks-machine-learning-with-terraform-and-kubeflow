"""Who shares a device with whom. A description of a placement, not a prediction about one.

This is the one thing worth keeping from the deleted `interference` package. It computed
co-residency in order to *index a predicted ITL by it*, and that indexing is exactly what impl
§10 rejects -- "which tenant a shard is co-resident with matters more than how many it has, and a
count cannot recover it". The count itself is still a fact about a placement, and a useful one to
print next to a measurement, so it lives on with the predictor it used to feed removed.

Nothing in the solver reads it. It is descriptive only: `inspect` prints it beside a record, the
`recall` prints it beside a run, and `calibrate` buckets repeats by it. If anything ever ranks
candidates by this number again, that is the predictor coming back in through a side door.

Pure standard library, so a reader of records does not need OR-Tools to use it.
"""

from __future__ import annotations


def coresidency(placement: dict) -> dict:
    """Co-resident tenant count per tenant, from `{name: (gpu, ...)}`.

    **Counted per GPU, not per node**, and that distinction is not cosmetic. A tenant contends
    for SMs with the tenants sharing its devices; a tenant on the other side of the node
    contends only for host memory bandwidth and `/dev/shm` (impl §4.2.1), which is a different
    and much weaker channel. A node-wide count would label a fully disjoint placement as
    maximally contended -- four tenants on six GPUs sharing none of them is "3 neighbours" by a
    node-wide count and "0" by this one, and 0 is what the hardware does.

    For a TP group the count is the **max over its shards**, per design §1.2: shards synchronize
    on an AllReduce roughly twice per layer, and a barrier runs at the speed of its last arrival,
    so the worst-placed shard sets the group's pace.

    Keys may be GPU ordinals or UUIDs -- whatever the caller has. Only equality is used.
    """
    occupants: dict = {}
    for name, gpus in placement.items():
        for g in gpus:
            occupants.setdefault(g, set()).add(name)
    out: dict = {}
    for name, gpus in placement.items():
        per_shard = [len(occupants[g]) - 1 for g in gpus if g in occupants]
        out[name] = max(per_shard) if per_shard else 0
    return out
