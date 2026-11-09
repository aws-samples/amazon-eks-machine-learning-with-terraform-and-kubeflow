#!/usr/bin/env python3
"""Write one synthetic search result into a synthetic store, so `export` has something to read. CPU only.

WHAT THIS IS FOR. A real `search` leases a GPU node, lets a reasoner choose placements, deploys each
one, drives it under load and files what the harness measured. This script stands in for all of
that, for one placement, so the steps after a search -- `colocation.inspect record`, `export` and
`python -m colocation.kai render` -- can be run and read with no GPU.

WHAT IT DOES, using the code's own machinery throughout:

  - loads the fleet against a store made by `tools/synthetic_profile.py`, so every tenant's degree
    resolves from the synthetic solo rows;
  - takes the solver's top-ranked candidate, exactly as the search loop's fallback policy does
    (`agent/policy.TopRanked`), with no reasoner involved;
  - runs it through the search loop (`agent/loop.SearchAgent`) with the package's own in-memory
    deployer and harness (`FakeDeployer`, `FakeHarness`), so the record is rendered, keyed and filed
    by the same code a real run uses; the harness returns latencies at three quarters of each
    tenant's own target, so every target is met;
  - writes the run summary where a real `search` leaves it, built by the same report function.

EVERY NUMBER IN THE RECORD IS ILLUSTRATIVE, and `certified: true` in the export it leads to means
nothing here: it only says the illustrative latencies are under the targets. No reasoner was asked
anything and no transcript is written; the summary records that the solver's first candidate was
taken.

The GPU identities are placeholders (`GPU-synthetic-0` ...). They live only in this store's record;
the export addresses slots, not devices, so none of them reaches a deployable document.

Run from the tutorial directory, after `tools/synthetic_profile.py`:

    PYTHONPATH=src python3 tools/synthetic_search.py \\
        --fleet fleets/examples/g6e.48xlarge/qwen-anchor-riders.yaml \\
        --store ~/.colocation/synthetic
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_profile import (SYNTHETIC_NODE, SYNTHETIC_RUN_TOKEN,            # noqa: E402
                               refuse_unsafe_store)

from colocation.agent import fleetfile                                          # noqa: E402
from colocation.agent.__main__ import _baseline, _report, _targets_met                        # noqa: E402
from colocation.agent.deploy import FakeDeployer                                # noqa: E402
from colocation.agent.loop import SearchAgent                                   # noqa: E402
from colocation.agent.measure import FakeHarness                                # noqa: E402
from colocation.agent.packer import PackerResult                                # noqa: E402
from colocation.agent.policy import TopRanked                                   # noqa: E402
from colocation.agent.prompt import DERIVED_VERDICT                             # noqa: E402
from colocation.agent.spec import NodeInventory                                 # noqa: E402
from colocation.store import (VERDICT_MEASURED, HardwareIndex, HubFactsStore,   # noqa: E402
                              ObservationStore, ProfileStore)

log = logging.getLogger("synthetic_search")

# Where the illustrative latencies sit: this fraction of each tenant's own ceiling. Above the solo
# rows' 0.5, so a co-resident tenant reads slower than it did alone, as sharing makes it.
LATENCY_FRACTION = 0.75


def illustrative_record(run_config: dict, run_id: str, fleet) -> dict:
    """A harness record shaped like a real one, every tenant at 3/4 of its own targets."""
    slos = {t.name: t.slo for t in fleet.tenants}
    per_tenant = {}
    for t in run_config["tenants"]:
        slo = slos[t["name"]]
        itl = float(round(slo.itl_mean_ms * LATENCY_FRACTION))
        ttft = None if slo.ttft_p95_ms is None else float(round(slo.ttft_p95_ms * LATENCY_FRACTION))
        per_tenant[t["name"]] = {
            "requests_dispatched": 100, "requests_completed": 100, "requests_failed": 0,
            "delivered_fraction": 1.0,
            "delivered_fraction_threshold": fleet.workload.delivered_fraction_threshold,
            "itl_mean_ms": itl, "ttft_p95_ms": ttft,
            # The mean is what `colocation.inspect record` prints; half the p95, as round as the rest.
            "ttft_mean_ms": None if ttft is None else float(round(ttft / 2))}
    return {"run_id": run_id, "validity_key_hash": run_config["validity_key_hash"],
            "verdict": VERDICT_MEASURED, "node": "synthetic-driver-node", "window_s": 60.0,
            "steady_state_reached_s": 30.0, "per_tenant": per_tenant, "per_gpu": {},
            "workload": {"tenants": {t["name"]: {"isl_tokens": t["isl_tokens"],
                                                 "osl_tokens": t["osl_tokens"],
                                                 "tokenizer_path": t.get("tokenizer_path")}
                                     for t in run_config["tenants"]}},
            "configuration": {"tenants": run_config["tenants"]},
            "sampling": {"subsamples_per_sample_mean": 36.0},
            "provenance": {**dict(run_config.get("provenance") or {}),
                           "gpu_node": SYNTHETIC_NODE},
            "notes": {"synthetic": "illustrative latencies from tools/synthetic_search.py; "
                                   "nothing was measured"}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fleet", required=True)
    p.add_argument("--store", required=True, type=Path,
                   help="the synthetic store tools/synthetic_profile.py made")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)-7s %(name)s: %(message)s")
    store_root = refuse_unsafe_store(args.store, must_be_synthetic=True)

    index = HardwareIndex(store_root)
    profiles = ProfileStore(store_root, fleetfile.profile_key(args.fleet, index))
    try:
        loaded = fleetfile.load(args.fleet, index=index, profiles=profiles,
                                hub=HubFactsStore(store_root))
    except fleetfile.FleetFileError as exc:
        raise SystemExit(f"{args.fleet}: {exc}\n(Run tools/synthetic_profile.py against this "
                         "fleet and store first.)")
    if loaded["max_nodes"] > 1:
        raise SystemExit("refused: this fleet is split over more than one node; the synthetic "
                         "path covers one-node fleets.")
    fleet, key = loaded["fleet"], loaded["key"]

    # The one thing a real search has that an offline load does not: a node with GPU identities,
    # which rendering a tenant's values needs. Placeholders, named as such.
    node = NodeInventory(node_name=SYNTHETIC_NODE, gpu_ids=fleet.node.gpu_ids,
                         uuids=tuple(f"GPU-synthetic-{g}" for g in fleet.node.gpu_ids),
                         device_total_gib=fleet.node.device_total_gib)
    fleet = dataclasses.replace(fleet, node=node)

    store = ObservationStore(store_root, key)
    store.write_key()
    agent = SearchAgent(
        fleet, store, deployer=FakeDeployer(),
        harness=FakeHarness(lambda cfg, run_id: illustrative_record(cfg, run_id, fleet)),
        run_prefix=SYNTHETIC_RUN_TOKEN,
        run_config_kwargs={"out_dir": str(store_root), "exporter_url": None})
    searched = agent.run(TopRanked(top_k=4, max_iterations=1), teardown_at_end=False)

    result = PackerResult(attempts=list(searched.attempts),
                          stopped_because="synthetic: the solver's first candidate, one "
                                          "measurement, no reasoner")
    # `met` is derived from the records, as the real loop derives it: every tenant's stated
    # target met on a measured record.
    met = [a for a in result.attempts
           if a.record and all(v is True for v in _targets_met(a.record, fleet).values())]
    result.verdict = DERIVED_VERDICT if met else "unresolvable"

    summary = _report(result, fleet, _baseline(fleet), profiles, None, store.read_calibration(),
                      fleetfile.upper_bound_tp(loaded))
    summary.update(fleet_file=str(args.fleet), validity_key_hash=key.hash,
                   validity_key=key.canonical(), mode="search", driver="none",
                   reasoner="none (synthetic: the solver's first candidate was taken)",
                   budget=1, elapsed_s=0.0, run_token=SYNTHETIC_RUN_TOKEN, lease_release="",
                   gpu_node=SYNTHETIC_NODE, gpu_node_observed={"synthetic": True},
                   synthetic=("written by tools/synthetic_search.py; every latency in it is "
                              "illustrative and nothing was measured"))
    path = store.key_dir / "searches" / f"{SYNTHETIC_RUN_TOKEN}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    ids = ", ".join(a.run_id for a in result.attempts)
    print(f"\nsynthetic search summary written to {path}")
    print(f"record(s) filed under validity key {key.hash}: {ids}")
    print("Next:\n"
          f"  PYTHONPATH=src python3 -m colocation.inspect record "
          f"{store.observations_dir / (result.attempts[0].run_id + '.json')}\n"
          f"  PYTHONPATH=src python3 -m colocation.agent export --fleet {args.fleet} "
          f"--store {store_root} --local-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
