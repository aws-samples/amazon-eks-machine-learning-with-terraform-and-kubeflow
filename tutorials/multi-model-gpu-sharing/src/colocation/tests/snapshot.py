"""What every shipped fleet renders to today, as a byte-comparable artifact.

This exists for one job: to make "the engine seam changed nothing" a **check** rather than a claim.
The refactor that puts a serving engine behind an adapter (impl §13) touches the one function that
knows the chart's value schema and the one that writes the harness's run config, and the failure it
risks is not a crash -- it is a fleet that still deploys, still measures, and quietly asks the engine
for a different amount of memory or a different shard count than it did before. Nothing downstream
would flag that. A stored snapshot would.

**What is snapshotted, and why it is not a placement.** Every shipped fleet file, every tenant,
rendered at every legal degree, with a fixed envelope and a fixed reserve. Deliberately *not* the
solver's chosen placement: a solver improvement would then show up here as an engine regression, and
a snapshot that fails for unrelated reasons is one people learn to regenerate without reading. What
is held constant is the mapping from (fleet file, degree, envelope) to deployable values -- which is
exactly the surface an engine adapter sits on.

**The hardware is a fixture and the numbers in it are arbitrary.** They have to be *some* value for
the rendering to be a function of anything, and none of them is a fact about a card: a snapshot
proving 44.99 GiB is a snapshot of the fixture. What it proves is that the same inputs produce the
same output before and after a refactor.

Regenerate with `python3 -m colocation.tests.snapshot --write` and **read the diff**. A diff is not
a failure -- an intended change to the chart's schema produces one -- but an unexplained line in it
is the whole point of the file.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ..agent import fleetfile, render
from ..agent.spec import DEFAULT_IMAGE
from ..solver.baseline import candidate_tp_degrees
from ..solver.memory import HEADROOM_DEFAULT, gpu_memory_utilization
from ..store.environment import DeviceEnvironment, topology_digest
from ..store.hwfacts import (HardwareFacts, HardwareIndex, HardwareKey, HardwareStore,
                             InstanceShape)
from .hubfixture import fixture_hub

# Where the snapshot lives, and where the fleets it covers live. Both relative to this file so the
# suite does not depend on the working directory a run was started from.
SNAPSHOT_PATH = Path(__file__).with_name("render-snapshot.json")
FLEETS_DIR = Path(__file__).resolve().parents[3] / "fleets"

# Fixture, not a measurement. See the module docstring: these are the constants that make the
# rendering a function, and a snapshot of them proves nothing about any card.
FIXTURE_CARD_GIB = 44.99
FIXTURE_GPU_COUNT = 8
FIXTURE_GPU_PRODUCT = "l40s"
# Provenance on the sheet below, not part of any key: see `store/environment.py`.
FIXTURE_DRIVER = "580.65.06"
# The three device facts that name this store's directories (`store/environment.py`). Fixtures like
# everything else here, and the MiB figure is `FIXTURE_CARD_GIB` in the unit a driver reports so that
# the fixture cannot be read as two different cards. The topology is digested rather than written as a
# literal hash: a hand-copied digest is a fixture that silently stops matching what the digest
# function produces.
FIXTURE_ADDRESSABLE_MIB = 46070
FIXTURE_POWER_LIMIT_W = 350
FIXTURE_TOPOLOGY = topology_digest({(a, b): "NODE" for a in range(FIXTURE_GPU_COUNT)
                                    for b in range(FIXTURE_GPU_COUNT) if a != b})
# One integer GiB, because the chart enforces the MPS limit as an integer and `render_values`
# refuses a fractional envelope rather than rounding it (`render._envelope_gib_int`).
FIXTURE_ENVELOPE_GIB = 40
# Zero, so the utilisation in the snapshot is a function of the envelope and the card alone. A
# non-zero reserve is a real term (`solver/memory.py`) and it is exercised elsewhere; here it would
# only add a constant nobody reading a diff could attribute.
FIXTURE_RESERVE_GIB = 0.0


def fixture_node():
    """A node inventory with fixture UUIDs, which only a test may build.

    `planned_node` deliberately carries none: a UUID that names no device is accepted by MPS and
    silently ignored, so the tenant runs with no memory envelope while the record looks healthy, and
    the refusal that produces is one of this system's load-bearing ones. Supplying them here is not a
    way round it -- the values dict pairs `gpu_indices[i]` with `gpu_uuids[i]` positionally, and that
    pairing is part of what this snapshot is holding still.
    """
    from ..agent.spec import NodeInventory

    return NodeInventory(node_name="fixture-node", gpu_ids=tuple(range(FIXTURE_GPU_COUNT)),
                         uuids=tuple(f"GPU-{i:04d}" for i in range(FIXTURE_GPU_COUNT)),
                         device_total_gib=FIXTURE_CARD_GIB)


class _Placement:
    """The three fields `render_run_config` and `render_releases` read off a placement.

    A stand-in rather than the real thing, because building a real one means running the solver, and
    then a solver change moves this snapshot. See the module docstring.
    """

    def __init__(self, assignment: dict, envelope_gib: dict, util: dict):
        self.assignment = assignment
        self.envelope_gib = envelope_gib
        self.util = util


def fixture_index(root: str) -> HardwareIndex:
    """A hardware store holding exactly what the shipped fleets ask about."""
    index = HardwareIndex(root)
    index.write_shape(InstanceShape(
        instance_type="g6e.48xlarge", gpu_count=FIXTURE_GPU_COUNT,
        gpu_product=FIXTURE_GPU_PRODUCT, addressable_mib=FIXTURE_ADDRESSABLE_MIB,
        power_limit_w=FIXTURE_POWER_LIMIT_W, p2p_topology=FIXTURE_TOPOLOGY,
        observed_at="2026-09-04T00:00:00Z", node="fixture-node"))
    # The shape carries the device facts because an offline load derives the store's address from it
    # (`agent/hardware.Hardware.environment`), so a fixture without them is a fixture no fleet can be
    # loaded against.
    key = HardwareKey(env=DeviceEnvironment(
        gpu_product=FIXTURE_GPU_PRODUCT, addressable_mib=FIXTURE_ADDRESSABLE_MIB,
        power_limit_w=FIXTURE_POWER_LIMIT_W, p2p_topology=FIXTURE_TOPOLOGY,
        engine_image=DEFAULT_IMAGE))
    sheet = HardwareStore(root, key)
    sheet.write_key()
    sheet.write(HardwareFacts(
        gpu_product=FIXTURE_GPU_PRODUCT, total_gib=FIXTURE_CARD_GIB, total_source="slope",
        total_uncertainty_gib=0.01, total_falsifiable=True, driver=FIXTURE_DRIVER,
        image=DEFAULT_IMAGE, gpu_node="fixture-node"))
    return index


def facts_missing(doc: dict, index, hub) -> list:
    """What an offline load of this fleet would need and the fixtures do not hold, worded for a reader.

    Empty means it can be loaded here. A fleet for an instance type nobody has filed a shape for, or
    naming a checkpoint nobody has fixtured, is not loadable offline, and inventing the missing facts
    would put a device figure or a weight count in source. Callers name such a fleet as not checked
    rather than skipping it silently; `fleetfile.check_doc` still holds it to every rule a file can
    break by itself.
    """
    it = fleetfile.fleet_instance_type(doc)
    miss = [] if index.read_shape(it) is not None else [f"no {it} shape"]
    ids = sorted({str(d["model_id"]) for d in (doc.get("models") or {}).values()
                  if d.get("model_id") and hub.newest_cached(str(d["model_id"])) is None})
    return miss + ([f"no hub facts for {', '.join(ids)}"] if ids else [])


def build(fleets_dir: Path | None = None) -> dict:
    """`{fleet path under fleets/: {tenant: {degree: values}}}` plus one run config per fleet.

    Every fleet is loaded with `allow_unresolved_tp=True`, because a shipped fleet's degrees come
    from measurements that are not in this fixture -- and the snapshot pins each degree explicitly
    anyway, which is more coverage than one resolved degree would be.

    The hub is the fixture one (`hubfixture.py`), so the weight bytes and the KV page behind every
    shipped fleet come off a cached manifest on disk. That matters more here than anywhere else in the
    suite: this snapshot is a byte-comparison, so a load that reached the network would make it depend
    on what the Hub answers today.
    """
    fleets_dir = FLEETS_DIR if fleets_dir is None else fleets_dir
    out: dict = {"fixture": {"card_gib": FIXTURE_CARD_GIB, "gpu_count": FIXTURE_GPU_COUNT,
                             "envelope_gib": FIXTURE_ENVELOPE_GIB,
                             "reserve_gib": FIXTURE_RESERVE_GIB,
                             "headroom": HEADROOM_DEFAULT,
                             "default_image": DEFAULT_IMAGE},
                 "fleets": {}, "not_snapshotted": {}}
    util = gpu_memory_utilization(FIXTURE_ENVELOPE_GIB, FIXTURE_RESERVE_GIB, FIXTURE_CARD_GIB,
                                  HEADROOM_DEFAULT)
    with tempfile.TemporaryDirectory() as root:
        index = fixture_index(root)
        # Recursive: the shipped fleets sit under `examples/<instance type>/`, and a non-recursive
        # glob here would snapshot nothing while still comparing equal to a snapshot of nothing.
        for path in sorted(fleets_dir.rglob("*.yaml")):
            missing = facts_missing(fleetfile.read_doc(path), index, fixture_hub())
            if missing:
                out["not_snapshotted"][path.relative_to(fleets_dir).as_posix()] = missing
                continue
            loaded = fleetfile.load(str(path), index=index, hub=fixture_hub(),
                                    allow_unresolved_tp=True)
            fleet = _replace_node(loaded["fleet"], fixture_node())
            entry: dict = {"validity_key_hash": loaded["key"].hash, "tenants": {}}
            for t in fleet.tenants:
                by_degree = {}
                for tp in candidate_tp_degrees(FIXTURE_GPU_COUNT):
                    pinned = _replace_tp(t, tp)
                    by_degree[str(tp)] = render.render_tenant_values(
                        pinned, fleet.node, list(range(tp)), float(FIXTURE_ENVELOPE_GIB), util)
                entry["tenants"][t.name] = by_degree
            # One run config for the whole fleet, at every tenant's narrowest legal degree. It is
            # the second artifact `render.py` produces and it names the same UUIDs as the first --
            # a disagreement between them files every per-GPU signal against the wrong tenant and
            # nothing reports it (`render.check_agreement`), so it belongs in the same snapshot.
            at_one = [_replace_tp(t, 1) for t in fleet.tenants]
            placement = _Placement(
                assignment={t.name: [0] for t in at_one},
                envelope_gib={t.name: float(FIXTURE_ENVELOPE_GIB) for t in at_one},
                util={t.name: util for t in at_one})
            entry["run_config"] = render.render_run_config(
                placement, _replace_tenants(fleet, at_one), run_id="snapshot",
                validity_key_hash=loaded["key"].hash)
            # Keyed by the path under `fleets/`, not the file name: two hardware directories are
            # entitled to hold the same fleet retargeted, and a base name would let one silently
            # overwrite the other's entry.
            out["fleets"][path.relative_to(fleets_dir).as_posix()] = entry
    return out


def _replace_tp(tenant, tp: int):
    import dataclasses

    return dataclasses.replace(tenant, tp=int(tp))


def _replace_node(fleet, node):
    import dataclasses

    return dataclasses.replace(fleet, node=node)


def _replace_tenants(fleet, tenants):
    import dataclasses

    return dataclasses.replace(fleet, tenants=tuple(tenants))


def dumps(snap: dict) -> str:
    """Canonical JSON: sorted keys and a trailing newline, so a diff is a real diff."""
    return json.dumps(snap, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    snap = dumps(build())
    if "--write" in argv:
        SNAPSHOT_PATH.write_text(snap)
        print(f"wrote {SNAPSHOT_PATH} ({len(snap)} bytes)")
        return 0
    if not SNAPSHOT_PATH.exists():
        print(f"no snapshot at {SNAPSHOT_PATH}; run with --write", file=sys.stderr)
        return 1
    if SNAPSHOT_PATH.read_text() == snap:
        print(f"{SNAPSHOT_PATH.name}: identical")
        return 0
    import difflib

    sys.stdout.writelines(difflib.unified_diff(
        SNAPSHOT_PATH.read_text().splitlines(keepends=True), snap.splitlines(keepends=True),
        fromfile="stored", tofile="rendered"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
