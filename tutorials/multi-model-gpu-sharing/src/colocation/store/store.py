"""The observation store (impl §8).

Files on EFS, not a database: readable by the agent, diffable, and versionable alongside the
recipes that produced them.

    /efs/colocation/observations/
      <validity-key-hash>/
        key.json                    # full key, so the directory name is attributable
        calibration.json            # noise floor per regime, solo saturation rates
        feasibility/
          <model>-tp<N>.json        # probe results
        observations/
          <run-id>.json             # one deployed configuration + everything measured

Two other stores sit beside this one under different keys and are not part of it: `profile.py`
(solo rows, keyed by the environment alone) and `hwfacts.py` (what a *device* is, keyed by driver
and image -- no capacity constant lives in source).

Two behaviours are worth stating outright because both are the opposite of a normal cache:

  - **Append-only.** A repeat of a configuration is a new observation, never an overwrite.
    Between-run spread is a quantity the design measures (design §8.1); overwriting would
    destroy exactly the data calibration exists to collect.
  - **A record that fails to parse is counted and named, never skipped quietly.** Infeasible
    observations are what stop the agent re-proposing a placement that already failed
    (impl §8). A store that silently returns fewer records than it holds would make the
    agent re-propose them, and the symptom -- an agent that keeps retrying a bad placement
    -- looks like a bad exploration policy rather than a broken read.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .schema import Calibration, Observation
from .validity_key import ValidityKey

log = logging.getLogger(__name__)


@dataclass
class LoadReport:
    """What `observations()` found, including what it could not read."""
    observations: list
    unreadable: list          # (path, reason)

    @property
    def ok(self) -> bool:
        return not self.unreadable


class ObservationStore:
    """One validity key's worth of observations.

    Constructed from a `ValidityKey` (normal use) or a bare hash string (for reading a
    directory whose key predates key computation -- runs 1-6 were stamped `smoke` by hand).
    """

    def __init__(self, root: str | Path, key: ValidityKey | str):
        self.root = Path(root)
        if isinstance(key, ValidityKey):
            self.key: ValidityKey | None = key
            self.key_hash = key.hash
        else:
            self.key = None
            self.key_hash = key

    # --- layout -------------------------------------------------------------

    @property
    def key_dir(self) -> Path:
        return self.root / "observations" / self.key_hash

    @property
    def observations_dir(self) -> Path:
        return self.key_dir / "observations"

    @property
    def feasibility_dir(self) -> Path:
        return self.key_dir / "feasibility"

    @property
    def exports_dir(self) -> Path:
        """Canonical forms of this key's result, one file per search (`agent/export.py`).

        Beside `searches/` and for the same reason it is not under `observations/`: everything in
        there is globbed and parsed as an `Observation`, so a document that is not one would be a
        parse failure at best and read as evidence at worst. An export is derived -- it can be
        rebuilt from the summary and the fleet file at any time -- which is why, unlike a record,
        it may be overwritten.
        """
        return self.key_dir / "exports"

    # --- key ----------------------------------------------------------------

    def write_key(self) -> Path:
        """Write `key.json`. Refuses to change an existing one.

        A key file that disagrees with the directory it sits in means two different
        environments have been filed together, and every comparison across them is invalid.
        Better to fail here than to let the agent average across hardware.
        """
        if self.key is None:
            raise ValueError("no ValidityKey: this store was opened by hash only")
        path = self.key_dir / "key.json"
        payload = self.key.key_json()
        if path.exists():
            existing = path.read_text()
            if existing != payload:
                raise ValueError(
                    f"{path} already exists and does not match this key. Same hash, "
                    "different contents means a hash collision or a changed canonical "
                    "form -- either way, refusing to overwrite the provenance of "
                    f"{len(list(self.observations_dir.glob('*.json')))} observations.")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
        return path

    def read_key(self) -> ValidityKey | None:
        path = self.key_dir / "key.json"
        if not path.exists():
            return None
        return ValidityKey.from_dict(json.loads(path.read_text()))

    # --- calibration --------------------------------------------------------

    def read_calibration(self) -> Calibration | None:
        """None means "not calibrated" -- the agent's cue to run §12 step 2.

        Deliberately not a default-valued object. A default noise floor would let an entire
        search run at a discrimination threshold nobody measured, and the agent would
        terminate on a criterion it invented.
        """
        path = self.key_dir / "calibration.json"
        if not path.exists():
            return None
        return Calibration.parse(json.loads(path.read_text()))

    def write_calibration(self, cal: Calibration) -> Path:
        path = self.key_dir / "calibration.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cal.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    # --- feasibility --------------------------------------------------------

    def feasibility_path(self, model: str, tp: int) -> Path:
        # Model ids contain slashes (Qwen/Qwen3-0.6B); a raw one would silently create a
        # nested directory and the read would then miss it.
        safe = model.replace("/", "_")
        return self.feasibility_dir / f"{safe}-tp{tp}.json"

    def read_feasibility(self, model: str, tp: int) -> dict | None:
        path = self.feasibility_path(model, tp)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def write_feasibility(self, model: str, tp: int, facts: dict) -> Path:
        """Feasibility facts are provisional by design (design §4.2).

        Memory feasibility cannot be computed from solo measurement -- compute contention
        from a neighbour raises ITL, which raises required concurrency, which can drive a
        tenant into memory starvation it never saw alone (design §1.7). So these are
        overwritable: a later in-context measurement supersedes an earlier solo probe. That
        is the opposite of the append-only rule for observations, and the difference is that
        a probe result is a current best estimate while an observation is a historical fact.
        """
        path = self.feasibility_path(model, tp)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**facts, "provisional": True},
                                   indent=2, sort_keys=True) + "\n")
        return path

    # --- observations -------------------------------------------------------

    def append_observation(self, record: dict) -> Path:
        """Write one record. Never overwrites; suffixes on collision."""
        run_id = record.get("run_id") or "unknown"
        got = record.get("validity_key_hash")
        if got and got != self.key_hash:
            raise ValueError(
                f"record {run_id} is stamped validity_key_hash={got!r} but this store is "
                f"{self.key_hash!r}. The harness stamps observations at collection time so "
                "they cannot be misfiled later; filing this one here would do exactly that.")
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        path = self.observations_dir / f"{run_id}.json"
        n = 1
        while path.exists():
            path = self.observations_dir / f"{run_id}-{n}.json"
            n += 1
        path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
        return path

    def load(self) -> LoadReport:
        """Every observation under this key, plus an explicit list of what would not read."""
        obs, bad = [], []
        if not self.observations_dir.exists():
            return LoadReport(observations=[], unreadable=[])
        for path in sorted(self.observations_dir.glob("*.json")):
            try:
                obs.append(Observation.parse(json.loads(path.read_text()), str(path)))
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                bad.append((str(path), f"{type(exc).__name__}: {exc}"))
        if bad:
            log.error("%d of %d records under %s could not be read: %s",
                      len(bad), len(bad) + len(obs), self.key_hash, bad)
        return LoadReport(observations=obs, unreadable=bad)

    def observations(self) -> list:
        return self.load().observations

    def usable_as_evidence(self) -> list:
        """Records that may be cited as evidence about latency (impl §10).

        Filters out both infeasible verdicts and records whose sampling is suspect. The
        second filter is the one that matters: an infeasible record announces itself, while
        a phase-locked one reads as a healthy measurement of a placement doing no work.
        """
        return [o for o in self.observations() if o.usable_as_evidence]
