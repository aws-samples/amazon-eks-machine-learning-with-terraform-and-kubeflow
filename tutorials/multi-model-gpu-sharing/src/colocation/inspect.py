"""Read back what a run measured.

    # one record: the headline numbers, and the signals that qualify them
    python -m colocation.inspect record <run-id> --key <validity-key-hash>
    python -m colocation.inspect record /path/to/record.json

It mutates nothing and it is not a test: it prints. That is why it is here and not under
`tests/`, where every module either passes or fails.

**There used to be a second mode.** `inspect model` printed the interference model's ITL table and
interrogated it at co-residency degrees it had never measured, which is where extrapolation error
lived -- and it earned its place by making a real defect visible (an optimistic tie-break priced a
cell bounded above by 18.733 ms at 5.734 ms). It is gone with the predictor it inspected (impl §9).
There is no table to print because nothing estimates a latency any more: the numbers a run is
judged against are the tenant's own stated target and the KV floor it implies, and both travel in
the record's own provenance, printed below alongside what was measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent.mirror import EfsMirror
from .agent.render import DEFAULT_NAMESPACE, DEFAULT_STORE_ROOT
from .store.schema import Observation, SuppressedMetric

DEFAULT_LOCAL_STORE = Path.home() / ".colocation" / "store"


# --- record ------------------------------------------------------------------

def _all_uuids(obs: Observation) -> list:
    """Every GPU UUID this record mentions, from the tenant configs and the per-GPU gauges."""
    seen = list(obs.per_gpu)
    for name in obs.per_tenant:
        for u in obs.tenant_config(name).get("gpu_uuids") or []:
            if u not in seen:
                seen.append(u)
    return seen


def gpu_aliases(uuids, *, width: int = 4) -> dict:
    """Short readable labels for GPU UUIDs: the shortest unambiguous prefix of the hex part.

    A record stores UUIDs and nothing else, because that is the only GPU identity the harness
    can see from inside a container. Host ordinals live in the node's ConfigMap, which a pure
    function over one record has no access to -- so callers that know the ordinals should pass
    their own `gpu_names`, and everyone else gets a label that is at least comparable *within*
    a record. That is enough to see which tenants overlap, which is the point.
    """
    keys = [str(u) for u in uuids]
    hexes = {k: (k[4:] if k.startswith("GPU-") else k).replace("-", "") for k in keys}
    w = width
    while w < 32 and len({h[:w] for h in hexes.values()}) < len(set(hexes.values())):
        w += 4
    return {k: h[:w] for k, h in hexes.items()}


def record_lines(obs: Observation, *, gpu_names: dict | None = None) -> list:
    """The headline numbers of one record, with the signals that qualify them.

    Deliberately prints the qualifiers next to the numbers rather than below them. A record's
    ITL means nothing without its verdict, and a record's sampled series mean nothing without
    `subsamples_per_sample` -- live run 2 read occupancy of exactly 0.0 while DCGM saw the GPU
    47% busy, and it looked perfectly healthy in every field a summary would normally show.

    **The `gpus` column prints identities, not a count, and that is a correction.** It used to
    print `len(gpu_uuids)`, which made two structurally different placements read identically:
    `r2` and `t2` sharing *both* their GPUs and sharing only *one* both show two GPUs and
    co-residency degree 1, and they measured 5.76 ms and 10.88 ms. Whatever explains that gap is
    in which GPUs overlap, and a count cannot express it. Neighbour structure is deliberately
    *not* summarised here -- the overlap is left visible for a reader to draw, because naming
    the pattern would pre-empt the diagnosis rather than support it.
    """
    # Two different nodes, and conflating them is how a record gets read as if it were measured
    # somewhere it was not. `obs.node` is written by the harness from its own environment, so it
    # is the load driver's node -- a cpu-compute node Karpenter reclaims, typically gone by the
    # time anyone reads the record. The node under test is in provenance, written by the agent.
    out = [f"run {obs.run_id}   key {obs.validity_key_hash}",
           f"  gpu node           {obs.provenance.get('gpu_node') or 'not recorded'}",
           f"  driver node        {obs.node or '?'}   (the harness Job's node, not the GPU's)",
           f"  verdict            {obs.verdict}"
           f"{'' if obs.measured else '   <- not a latency measurement'}",
           f"  window             {obs.window_s:.1f}s"
           f"   steady state at {_fmt(obs.steady_state_reached_s, 's')}",
           f"  sampling           {obs.subsamples_per_sample:.1f} sub-samples/sample"
           f"{'   <- SUSPECT, may be phase-locked to the arrivals' if obs.sampling_suspect else ''}",
           f"  usable as evidence {obs.usable_as_evidence}"]
    if obs.provenance.get("synthesized_by"):
        out.append(f"  synthesized by     {obs.provenance['synthesized_by']}"
                   "   <- no harness ran; this is a feasibility fact")
    if obs.notes.get("readiness_gate"):
        out.append(f"  readiness gate     {obs.notes['readiness_gate']}: "
                   f"{obs.notes.get('detail', '')}")

    if obs.per_tenant:
        out.append("")
        # `preempt`, `KV max` and `blocks` were the three columns to the right of TTFT until impl
        # §14.6. All three were a serving engine's account of its own internals, and `in flight`
        # replaces them with the driver's: the count at the start of the window and at the end, which
        # is what says whether the backlog drained. Flat is the healthy reading; a climb means
        # completions ran below arrivals, and it is the same series that decided steady state.
        out.append(f"  {'tenant':<8}{'ITL':>10}{'TTFT':>10}{'in flight':>16}"
                   f"{'degree':>7}  gpus")
        degrees = _degrees(obs)
        names = gpu_names or gpu_aliases(_all_uuids(obs))
        for name, t in sorted(obs.per_tenant.items()):
            gpus = obs.tenant_config(name).get("gpu_uuids") or []
            out.append(
                f"  {name:<8}{_lat(t, 'itl_mean_ms'):>10}{_lat(t, 'ttft_mean_ms'):>10}"
                f"{_in_flight(t):>16}{degrees.get(name, '?'):>7}"
                f"  {','.join(names.get(u, u) for u in gpus) or '-'}")
            if t.latency_suppressed:
                out.append(f"           latency suppressed: {t.latency_suppressed}")
            if t.chunk_divergence:
                # Affects no reported number -- the gate and ITL both read the server's count
                # -- but it is a live unexplained divergence and must not be invisible.
                out.append(f"           {t.chunk_divergence} tokens the client never saw as "
                           "chunks (unexplained; affects no number here)")

    throttled = [u for u, g in obs.per_gpu.items() if g.throttled]
    if throttled:
        # A clock drop from a power or thermal cap is indistinguishable from one caused by
        # co-location, so this has to be louder than a field in a table.
        out.append("")
        out.append(f"  THROTTLED: {len(throttled)} GPU(s) hit a power or thermal cap. Any "
                   "latency difference in this record may be the cap, not the placement.")

    out += _floor_lines(obs)
    out += _prediction_lines(obs)
    return out


def _floor_lines(obs: Observation) -> list:
    """What was required of this placement when it was chosen, beside what it got.

    The requirement, not a forecast: `kv_floor_tokens` is the KV cache the tenant's own stated ITL
    ceiling needs at its own rate, and the margin is how much the solver gave it over that. A floor
    cleared is not a target met -- the measured ITL above settles that and nothing here does -- so
    the two blocks are printed separately and neither is labelled as evidence for the other.
    """
    floors = obs.provenance.get("kv_floor_tokens")
    if not floors:
        return []
    kv = obs.provenance.get("kv_tokens") or {}
    margins = obs.provenance.get("kv_margin") or {}
    sources = obs.provenance.get("kv_floor_source") or {}
    out = ["", "  what was required of this placement when it was chosen:"]
    for name in sorted(floors):
        floor = floors[name]
        margin = margins.get(name)
        out.append(f"    {name:<8} floor {_fmt(floor):>8} tokens   got {_fmt(kv.get(name)):>8}"
                   f"   {f'{margin:.2f}x' if margin is not None else 'no floor stated':>16}")
        if sources.get(name):
            out.append(f"             {sources[name]}")
    return out


def _prediction_lines(obs: Observation) -> list:
    """The prediction block, for records old enough to have one.

    Kept deliberately. impl §9 is prediction-free from here on, but records written before it are
    still in the store and still say what the deleted model forecast; a reader comparing them with
    what was measured is doing something legitimate. Rewriting or hiding them would be editing a
    history, so they are printed and labelled as historical.
    """
    pred = obs.provenance.get("predicted_itl_ms")
    if not pred:
        return []
    out = ["", "  what the agent predicted when it chose this (a record from before impl §9; "
                "nothing predicts a latency now):"]
    for name in sorted(pred):
        got = _lat(obs.per_tenant[name], "itl_mean_ms") if name in obs.per_tenant else "-"
        out.append(f"    {name:<8} predicted {pred[name]:>8.3f}ms   measured {got:>10}"
                   f"   [{obs.provenance.get('itl_sources', {}).get(name, '?')}]")
    return out


def _degrees(obs: Observation) -> dict:
    """Co-residency degree per tenant, from the UUIDs the harness recorded."""
    from .solver.topology import coresidency
    placement = {n: obs.tenant_config(n).get("gpu_uuids") or [] for n in obs.per_tenant}
    if not all(placement.values()):
        return {}
    return coresidency(placement)


def _lat(t, which: str) -> str:
    """A latency field, or why it is absent. Never a zero: see `SuppressedMetric`."""
    try:
        return f"{getattr(t, which):.3f}ms"
    except SuppressedMetric:
        return "suppressed"


def _in_flight(t) -> str:
    """`first->last` from the driver's own in-flight series, or `-` when the row has none.

    Two numbers rather than one, because the direction is the information: a settled window is flat
    and a window that came apart climbs. `-` is shown for a record written before impl §14.6, which
    carried a queue depth off the server instead -- that field is still parsed and is deliberately
    not printed here, because a reader comparing it against this column would be comparing a
    server's account of itself against the driver's.
    """
    a, b = t.in_flight_first_quarter_mean, t.in_flight_last_quarter_mean
    if a is None or b is None:
        return "-"
    return f"{a:.3g}->{b:.3g}"


def _fmt(v, unit: str = "") -> str:
    return "-" if v is None else (f"{v:.1f}{unit}" if isinstance(v, float) else f"{v}{unit}")


def _pct(v) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


# --- plumbing ----------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m colocation.inspect",
                                description="read back one record")
    # A single choice rather than a bare `target`, so the call that worked yesterday still works:
    # `model` was the other mode and is gone with the predictor. Naming `record` explicitly also
    # keeps the door open for a second mode without changing anyone's command line again.
    p.add_argument("mode", choices=("record",))
    p.add_argument("target", nargs="?", help="a run id, or a path to a .json")
    p.add_argument("--key", help="validity key hash (required for a run id)")
    p.add_argument("--store", default=str(DEFAULT_LOCAL_STORE))
    p.add_argument("--efs-store", default=DEFAULT_STORE_ROOT)
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p.add_argument("--json", action="store_true", help="dump the raw record")
    p.add_argument("--ordinals", action="store_true",
                   help="label GPUs by host ordinal instead of by UUID prefix. Free for records "
                        "carrying `provenance.gpu_ordinal_by_uuid`; for older ones it falls back "
                        "to reading the node's UUID ConfigMap, which needs the cluster and needs "
                        "the node to still exist")
    return p


def _ordinal_names(obs: Observation, args) -> dict | None:
    """UUID -> `gpu<n>`, from the record if it carries the map and the cluster otherwise.

    Not the default. The cluster route needs a node that still exists, and a record outlives its
    node -- so an inspector that required it would stop working on exactly the records that are
    oldest and most worth reading.

    Records written from 2026-09-05 carry `gpu_ordinal_by_uuid` and need no cluster at all, which
    is the point of writing it: the mapping is what makes two records on two different machines
    comparable by position rather than only by identity. Preferred over the ConfigMap when both
    are available, because it is what this run actually deployed against, whereas the ConfigMap
    describes whatever daemon holds the node now.
    """
    stored = obs.provenance.get("gpu_ordinal_by_uuid")
    if stored:
        return {u: f"gpu{g}" for u, g in stored.items()}
    node = obs.provenance.get("gpu_node")
    if not node:
        print("--ordinals: this record has no provenance.gpu_node, so there is no node whose "
              "ConfigMap could be read. Falling back to UUID prefixes.", file=sys.stderr)
        return None
    from .agent.fleetfile import DEFAULT_DAEMON_RELEASE, read_gpu_uuids
    # The ConfigMap is named after the daemon release that published it, which is per-run since
    # runs capture their own node (agent/capture.py). Records written before that carry no
    # `daemon_release` and were all published by the one hand-installed daemon.
    release = obs.provenance.get("daemon_release") or DEFAULT_DAEMON_RELEASE
    try:
        by_ordinal = read_gpu_uuids(node, namespace=args.namespace, release=release)
    except Exception as exc:                            # noqa: BLE001 - degrade, never fail
        print(f"--ordinals: could not read the UUID map for {node} ({exc}). Falling back to "
              "UUID prefixes.", file=sys.stderr)
        return None
    return {u: f"gpu{g}" for g, u in by_ordinal.items()}


def _load_record(args) -> Observation:
    if not args.target:
        raise SystemExit("record mode needs a run id or a path")
    path = Path(args.target)
    if path.exists():
        return Observation.parse(json.loads(path.read_text()), source_path=str(path))
    if not args.key:
        raise SystemExit(f"{args.target} is not a file, so it is read as a run id -- which "
                         "needs --key to say which validity key it belongs to")
    local = Path(args.store) / "observations" / args.key / "observations" / f"{args.target}.json"
    if local.exists():
        return Observation.parse(json.loads(local.read_text()), source_path=str(local))
    # Not mirrored locally: read it straight off EFS rather than pulling the whole key, since
    # inspecting one run should not require syncing every run under it.
    mirror = EfsMirror(args.key, args.store, efs_root=args.efs_store, namespace=args.namespace)
    raw = mirror.read_record(args.target)
    if raw is None:
        known = mirror.list_records()
        raise SystemExit(f"no record {args.target} locally or on EFS under {args.key}."
                         + (f" EFS has: {', '.join(known)}" if known else " EFS has none."))
    return Observation.parse(raw, source_path=f"efs:{args.target}")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    obs = _load_record(args)
    if args.json:
        print(json.dumps(json.loads(Path(obs.source_path).read_text())
                         if obs.source_path and Path(obs.source_path).exists() else {},
                         indent=2, sort_keys=True))
        return 0
    print("\n".join(record_lines(
        obs, gpu_names=_ordinal_names(obs, args) if args.ordinals else None)))
    # Non-zero when the record is not a usable measurement, so this composes in a shell.
    return 0 if obs.usable_as_evidence else 1


if __name__ == "__main__":
    sys.exit(main())
