"""`search` over more than one node: one ordinary search per group, and which split's groups to run.

**A group is a fleet file.** `solver/partition.py` splits a fleet into groups, one per node, and a
group is the parent file cut down to its tenants: same hardware, same engine, same models, same
tenants number for number, and no `max_nodes`. So a group is searched by today's `search`, unchanged
-- its own lease, its own validity key, its own store folder, its own summary and export -- and a
group searched once is found again under the same key by any later split that contains it.

**What this module decides is only the order.** `drive` is the deterministic driver, built first
so that a reasoner choosing splits has a fallback, as `TopRanked` is to a reasoner choosing
placements (impl §11.2). It walks the ranked splits and, within a split, searches the group with
the least KV margin first: that is the group most likely to miss, and a split one of whose groups
misses is finished without leasing a node for the rest. It stops at the first split whose every
group came back `met`, or when the group budget is spent.

Pure: running a group is `run_group`'s, which `__main__` supplies.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .prompt import DERIVED_VERDICT


def group_doc(doc: dict, tenants) -> dict:
    """The parent fleet file cut down to `tenants`: a fleet on one node, of exactly those tenants."""
    names = set(tenants)
    unknown = names - {t["name"] for t in doc["tenants"]}
    if unknown:
        raise ValueError(f"the fleet has no tenant {', '.join(sorted(unknown))}")
    out = copy.deepcopy(doc)
    out["hardware"] = {k: v for k, v in out["hardware"].items() if k != "max_nodes"}
    out["tenants"] = [t for t in out["tenants"] if t["name"] in names]
    used = {t["model"] for t in out["tenants"]}
    out["models"] = {k: v for k, v in out["models"].items() if k in used}
    return out


def group_path(root, fleet_path, tenants) -> Path:
    """Where a group's fleet file is written. Named by the parent and the tenant set, so the same
    group of the same fleet is the same file in every split and every run."""
    digest = hashlib.sha256("\n".join(sorted(tenants)).encode()).hexdigest()[:10]
    return Path(root) / "groups" / Path(fleet_path).stem / f"{Path(fleet_path).stem}-{digest}.yaml"


def group_text(fleet_path, doc: dict, tenants) -> str:
    """The text `write_group` writes. Its digest is what an earlier run's summary is matched on."""
    head = (f"# A group of {fleet_path}, written by `search` for one node: tenants "
            f"{', '.join(tenants)}.\n# Generated; edit the parent file instead.\n")
    return head + yaml.safe_dump(group_doc(doc, tenants), sort_keys=False)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_group(root, fleet_path, doc: dict, tenants) -> Path:
    path = group_path(root, fleet_path, tenants)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(group_text(fleet_path, doc, tenants))
    return path


def recall_group(root, fleet_path, doc: dict, tenants):
    """The newest earlier run of exactly this group, or None.

    Matched on the digest of the group's fleet file, which each run's summary records as
    `group_fleet_sha256`: the file's name says only which tenants, and a target, a rate or a model
    edited in the parent changes the text but not the name. A run that wrote no summary, or whose
    summary has no verdict, is not a result and is not recalled.
    """
    path = group_path(root, fleet_path, tenants)
    sha = text_sha256(group_text(fleet_path, doc, tenants))
    for p in sorted(path.parent.glob(f"{path.stem}.*.summary.json"), reverse=True):
        try:
            summary = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if summary.get("group_fleet_sha256") == sha and summary.get("verdict"):
            return GroupRun(tuple(tenants), str(path), summary["verdict"], summary, recalled=True)
    return None


@dataclass
class GroupRun:
    """One group's search, as its summary reports it."""

    tenants: tuple
    fleet_file: str
    verdict: str | None             # the summary's; `met` only when a record met every target
    summary: dict | None = None     # None when the run wrote none (it failed before the end)
    recalled: bool = False          # an earlier run's, found by `recall_group`: no budget spent

    @property
    def met(self) -> bool:
        return self.verdict == DERIVED_VERDICT


@dataclass
class DriveResult:
    split: object | None            # the first split whose every group was met, or None
    runs: list = field(default_factory=list)    # every group searched, in order
    tried: list = field(default_factory=list)   # (split rank, the group that missed or None)
    stopped_because: str = ""


def _order(groups) -> list:
    """Least margin first, so the group most likely to miss is the one measured first."""
    return sorted(groups, key=lambda g: (g.min_margin if g.min_margin is not None
                                         else float("inf"), g.tenants))


def drive(splits, run_group, budget: int, recall=None) -> DriveResult:
    """Search the ranked `splits` group by group until one is met throughout.

    `run_group(tenants) -> GroupRun` searches one group; `budget` is how many may be run. A group
    already run is never run again: its result is the same group's result in every split. Nor is
    one `recall(tenants)` finds an earlier run of, and that costs none of the budget.
    """
    out = DriveResult(split=None)
    done: dict = {}
    for rank, s in enumerate(splits, 1):
        missed = None
        for g in _order(s.groups):
            run = done.get(frozenset(g.tenants))
            if run is None and recall is not None:
                run = recall(tuple(g.tenants))
                if run is not None:
                    done[frozenset(g.tenants)] = run
                    out.runs.append(run)
            if run is None:
                if sum(not r.recalled for r in done.values()) >= budget:
                    out.tried.append((rank, None))
                    out.stopped_because = (f"the group budget of {budget} is spent, at split "
                                           f"#{rank}")
                    return out
                run = run_group(tuple(g.tenants))
                done[frozenset(g.tenants)] = run
                out.runs.append(run)
            if not run.met:
                missed = run
                break
        out.tried.append((rank, missed))
        if missed is None:
            out.split = s
            out.stopped_because = f"split #{rank} met every target on every node"
            return out
    out.stopped_because = (f"none of the {len(splits)} split(s) met every target on every node"
                           if splits else "there was no split to search")
    return out


def next_group(splits, done) -> tuple | None:
    """The group `drive` would search next, given `done` (tenant frozenset -> GroupRun); None when
    a split is already met throughout or every split has a group that missed.

    One step of `drive`, for a caller that interleaves its own choices with the driver's: the
    reasoner's fallback (`splitpacker`) takes this step on its behalf, and only this one.
    """
    for s in splits:
        ordered = _order(s.groups)
        if any(frozenset(g.tenants) in done and not done[frozenset(g.tenants)].met
               for g in ordered):
            continue
        for g in ordered:
            if frozenset(g.tenants) not in done:
                return tuple(g.tenants)
        return None
    return None
