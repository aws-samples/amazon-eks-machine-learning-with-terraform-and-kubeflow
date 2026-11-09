#!/usr/bin/env python3
"""Fill a separate, synthetic store with what a real `profile` would have filed. CPU only.

WHAT THIS IS FOR. `plan` refuses an instance type the store knows nothing about, and resolves every
tenant's tensor-parallel degree from solo profile rows that only `profile` can measure, on a leased
GPU node. This script stands in for those two steps so that the rest of the pipeline -- `plan`, the
demand sweep, `export`, `python -m colocation.kai render` -- can be run and read with no GPU at all.
It writes, through the code's own store writers and key functions:

  - an instance shape and a device sheet for the fleet's instance type, which is what the first
    `profile` or `search` on a new instance type files when it captures a node; and
  - one solo profile row per tenant, at the degree a real `profile` measures first (the memory
    floor), with latencies at half of the tenant's own targets, so every row meets them.

EVERY NUMBER HERE IS ILLUSTRATIVE. The latencies are arithmetic on the fleet file's targets, not
measurements, and are deliberately round. The device total and power limit are the card's nameplate
figures, not a reading from any node. Nothing filed here says anything about how a model performs.

WHY IT CANNOT BE MISTAKEN FOR A REAL STORE. Three guards:

  - It writes only to a store you name with `--store`, and refuses the default store
    (`~/.colocation/store`) and anything under `/efs`.
  - Every row is filed under an environment no real node can report. A real node's link topology is
    a 12-hex-digit digest of `nvidia-smi topo -m`; the synthetic shape's is the word `synthetic`, so
    its store address can never be the address of a real node's rows.
  - Every row says so: run ids, node name and `notes.synthetic` all name it.

Never point `profile` or `search` at the synthetic store. They lease a real node, read its real link
topology, find it differs from the synthetic shape, and stop with `EnvironmentMoved` after the node
has been leased.

Run from the tutorial directory:

    PYTHONPATH=src python3 tools/synthetic_profile.py \\
        --fleet fleets/examples/g6e.48xlarge/qwen-anchor-riders.yaml \\
        --store ~/.colocation/synthetic
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from colocation.agent import fleetfile                                          # noqa: E402
from colocation.agent.__main__ import AGENT_ENVELOPE_STEP_MIB                   # noqa: E402
from colocation.agent.profiler import (SoloProfiler, profile_from_record,       # noqa: E402
                                       solo_fleet, solo_key)
from colocation.solver.cpsat import Infeasible, PlacementSolver                 # noqa: E402
from colocation.store import (VERDICT_MEASURED, HardwareFacts, HardwareIndex,   # noqa: E402
                              HardwareKey, HardwareStore, HubFactsStore, InstanceShape,
                              ProfileStore)

log = logging.getLogger("synthetic_profile")

# The marker every synthetic store carries, and the value that keeps its address apart from every
# real node's. A real topology is a hex digest (`store/environment.topology_digest`), never a word.
SYNTHETIC_TOPOLOGY = "synthetic"
SYNTHETIC_NODE = "synthetic-node"
SYNTHETIC_RUN_TOKEN = "synthetic"
MARKER = "SYNTHETIC-STORE.txt"

# Nameplate figures for the instance types the shipped fleets use: how many GPUs, which card, and the
# card's published memory and power limit. Illustrative inputs, not observations of any node.
# 45 GiB per device is the L40S's 48 GB, as a round figure.
NAMEPLATE = {
    "g6e.12xlarge": {"gpu_count": 4, "gpu_product": "l40s", "device_gib": 45.0, "power_w": 350},
    "g6e.24xlarge": {"gpu_count": 4, "gpu_product": "l40s", "device_gib": 45.0, "power_w": 350},
    "g6e.48xlarge": {"gpu_count": 8, "gpu_product": "l40s", "device_gib": 45.0, "power_w": 350},
}

# Where the illustrative latencies sit: this fraction of each tenant's own ceiling.
LATENCY_FRACTION = 0.5

FIXTURE_HUB = Path(__file__).resolve().parents[1] / "src/colocation/tests/fixtures/hubfacts"


def refuse_unsafe_store(store: Path, *, must_be_synthetic: bool) -> Path:
    """Resolve `--store` and refuse the two places real measurements live."""
    store = store.expanduser().resolve()
    default = (Path.home() / ".colocation" / "store").resolve()
    if store == default or default in store.parents:
        raise SystemExit(f"refused: {store} is the default store, where real measurements live. "
                         "Name a separate directory, e.g. --store ~/.colocation/synthetic")
    if str(store).startswith("/efs"):
        raise SystemExit(f"refused: {store} is on EFS, where real measurements live.")
    if must_be_synthetic and not (store / MARKER).is_file():
        raise SystemExit(f"refused: {store} has no {MARKER}, so it was not made by "
                         "tools/synthetic_profile.py. Run that first.")
    if not must_be_synthetic and store.exists() and any(store.iterdir()) \
            and not (store / MARKER).is_file():
        raise SystemExit(f"refused: {store} already holds files and was not made by this script. "
                         "Name an empty or new directory.")
    return store


def seed_hub_cache(store: Path, doc: dict) -> None:
    """Copy the checkpoint manifests the self-test ships into the store's hub cache.

    A manifest is a published, immutable artifact (weights and KV geometry read off a checkpoint
    at one commit), so this is a cache, not a measurement. It is what lets `plan` price the fleet
    with no network. Models the fixtures do not cover are fetched from the hub as usual.
    """
    hub = HubFactsStore(store)
    for model in (doc.get("models") or {}).values():
        dest = hub._dir(str((model or {}).get("model_id") or ""))   # the cache's own layout
        src = FIXTURE_HUB / dest.name
        if src.is_dir() and not dest.exists():
            shutil.copytree(src, dest)


def file_hardware(store: Path, doc: dict, instance_type: str) -> None:
    plate = NAMEPLATE.get(instance_type)
    if plate is None:
        raise SystemExit(f"refused: no nameplate figures here for {instance_type}. This script "
                         f"knows {', '.join(sorted(NAMEPLATE))}.")
    shape = InstanceShape(
        instance_type=instance_type, gpu_count=plate["gpu_count"],
        gpu_product=plate["gpu_product"], addressable_mib=int(plate["device_gib"] * 1024),
        power_limit_w=plate["power_w"], p2p_topology=SYNTHETIC_TOPOLOGY,
        observed_at="synthetic", node=SYNTHETIC_NODE,
        notes={"synthetic": "nameplate figures written by tools/synthetic_profile.py, "
                            "not observed on any node"})
    index = HardwareIndex(store)
    index.write_shape(shape)
    sheets = HardwareStore(store, HardwareKey(env=shape.device_environment(
        fleetfile.fleet_image(doc))))
    sheets.write_key()
    sheets.write(HardwareFacts(
        gpu_product=plate["gpu_product"], total_gib=plate["device_gib"],
        total_source="synthetic", driver="synthetic", image=fleetfile.fleet_image(doc),
        gpu_node=SYNTHETIC_NODE, measured_at="synthetic",
        notes={"synthetic": "nameplate figure, not measured"}))
    print(f"filed {instance_type}: {plate['gpu_count']}x{plate['gpu_product']}, "
          f"{plate['device_gib']:g} GiB and {plate['power_w']} W per device (nameplate), "
          f"link topology '{SYNTHETIC_TOPOLOGY}'")


def illustrative_record(tenant, threshold: float, kv_tokens: int, reserve_source: str) -> dict:
    """A harness record shaped like a real solo round, with latencies at half the targets."""
    slo = tenant.slo
    itl = round(slo.itl_mean_ms * LATENCY_FRACTION)
    ttft = None if slo.ttft_p95_ms is None else round(slo.ttft_p95_ms * LATENCY_FRACTION)
    return {
        "verdict": VERDICT_MEASURED,
        "node": SYNTHETIC_NODE,
        "per_tenant": {tenant.name: {
            "itl_mean_ms": float(itl), "ttft_p95_ms": None if ttft is None else float(ttft),
            "delivered_fraction": 1.0, "delivered_fraction_threshold": float(threshold)}},
        "sampling": {"subsamples_per_sample_mean": 36.0},
        "provenance": {"gpu_node": SYNTHETIC_NODE,
                       "kv_tokens": {tenant.name: int(kv_tokens)},
                       "mps_reserve_source": {tenant.name: reserve_source}},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fleet", required=True)
    p.add_argument("--store", required=True, type=Path,
                   help="a separate directory for the synthetic store, never the default one")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)-7s %(name)s: %(message)s")

    store = refuse_unsafe_store(args.store, must_be_synthetic=False)
    store.mkdir(parents=True, exist_ok=True)
    (store / MARKER).write_text(
        "Everything under this directory was written by tools/synthetic_profile.py and\n"
        "tools/synthetic_search.py. It is illustrative and measures nothing.\n"
        "Never pass this directory to `profile` or `search`.\n")

    doc = fleetfile.read_doc(args.fleet)
    instance_type = fleetfile.fleet_instance_type(doc)
    if fleetfile.fleet_max_nodes(doc) > 1:
        raise SystemExit("refused: this fleet is split over more than one node "
                         "(hardware.max_nodes); the synthetic path covers one-node fleets.")
    seed_hub_cache(store, doc)
    file_hardware(store, doc, instance_type)

    index = HardwareIndex(store)
    profiles = ProfileStore(store, fleetfile.profile_key(args.fleet, index))
    profiles.write_key()
    hub = HubFactsStore(store)
    # The load below is BEFORE any row exists, so it warns that every degree is a memory floor.
    # True, and exactly what this script is about to fill in, so it is not shown unless asked.
    if not args.verbose:
        logging.getLogger("colocation.agent.fleetfile").setLevel(logging.ERROR)
    loaded = fleetfile.load(args.fleet, index=index, profiles=profiles, hub=hub,
                            allow_unresolved_tp=True)
    fleet, key, hardware = loaded["fleet"], loaded["key"], loaded["hardware"]
    threshold = fleet.workload.delivered_fraction_threshold
    # Used for its arithmetic only (where a real walk starts, which degrees it leaves below):
    # nothing here deploys, so it is given no agent.
    walk = SoloProfiler(fleet, key, hardware, profiles, make_agent=None, store_root=store)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"\nsolo rows under profile key {profiles.key_hash} (illustrative, "
          f"latency = {LATENCY_FRACTION:g} x each tenant's own target):")
    for tenant in fleet.tenants:
        tp = walk.start_tp_for(tenant)
        if tp is None:
            print(f"  {tenant.name:<8} no degree fits its weights on a {instance_type}; no row")
            continue
        if tenant.slo is None or tenant.slo.itl_mean_ms is None:
            print(f"  {tenant.name:<8} states no itl_mean_ms target; no row")
            continue
        solo = solo_fleet(fleet, tenant, tp)
        gpus = tuple(fleet.node.gpu_ids[:tp])
        try:
            candidate = PlacementSolver(solo.node.node_spec, solo.requests,
                                        envelope_step_mib=AGENT_ENVELOPE_STEP_MIB
                                        ).solve_pinned({tenant.name: gpus}, max_seconds=10.0)
        except Infeasible as exc:
            print(f"  {tenant.name:<8} tp{tp} cannot be placed alone ({exc}); no row")
            continue
        placement = candidate.placement
        record = illustrative_record(tenant, threshold, placement.kv_tokens[tenant.name],
                                     placement.reserve_source[tenant.name])
        row = profile_from_record(
            record, tenant, tp, envelope_gib=placement.envelope_gib[tenant.name],
            device_total_gib=fleet.node.device_total_gib,
            run_id=f"synthetic-{tenant.name}-tp{tp}", run_token=SYNTHETIC_RUN_TOKEN,
            key_hash=solo_key(key, tenant, tp).hash, gpu_ids=gpus,
            util=placement.util.get(tenant.name), measured_at=now)
        row = dataclasses.replace(row, notes={
            **row.notes, "synthetic": "illustrative latencies from tools/synthetic_profile.py; "
                                      "nothing was measured"})
        profiles.write(row)
        below = [k for k in walk.band_for(tenant)]
        print(f"  {tenant.name:<8} {tenant.model_id} tp{tp}: ITL {row.itl_mean_ms:g} ms "
              f"(target {tenant.slo.itl_mean_ms:g}), TTFT p95 "
              f"{'-' if row.ttft_p95_ms is None else f'{row.ttft_p95_ms:g} ms'}, "
              f"envelope {row.envelope_gib:.2f} GiB/shard"
              + (f"; tp{', tp'.join(map(str, below))} left unmeasured" if below else ""))

    print(f"\nsynthetic store ready at {store}. Next:\n"
          f"  PYTHONPATH=src python3 -m colocation.agent plan --fleet {args.fleet} --store {store} "
          "--local-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
