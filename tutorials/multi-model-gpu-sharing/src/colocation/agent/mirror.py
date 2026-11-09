"""Keeping a local store in step with the canonical one on EFS (impl §8, §11).

The agent runs on the ML Ops desktop; `/efs` is only reachable from inside the cluster. So the
store the agent reads and writes is a **local mirror** of the one the harness writes to, and
this module is the only thing that moves records between them. Everything goes through
`kubectl exec` into the standing `attach-pvc` pod -- never an ad-hoc reader pod.

Why a mirror at all, rather than just reading records back one at a time as they are produced:

  - The whole store is re-read every iteration, and the records themselves are what the reasoner
    reasons from (impl §10). Starting from an empty local directory when EFS holds a history means
    the agent silently reasons from less than it knows, which looks like a bad exploration policy
    rather than a missing read.
  - Records the *agent* synthesizes -- a placement that failed its readiness gate -- exist only
    locally, because no harness Job ever ran. Those are precisely the records that stop the next
    agent re-proposing a placement that already failed, so they have to be pushed up. An
    infeasibility that lives on one laptop is an infeasibility the fleet will rediscover.

Direction is asymmetric on purpose: EFS wins on pull (the harness wrote it there, that is the
measurement), and push refuses to overwrite (a local file must never replace a real record).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .deploy import DeployError, _run
from .render import DEFAULT_NAMESPACE, DEFAULT_STORE_ROOT

log = logging.getLogger(__name__)

DEFAULT_ATTACH_POD = "attach-pvc"
SIDECAR_FILES = ("key.json", "calibration.json")


class MirrorError(Exception):
    pass


class PodMirror:
    """The `attach-pvc` plumbing, shared by everything that moves files to and from EFS.

    Split out because there is now more than one kind of directory on EFS worth mirroring --
    a validity key's observations, and an environment's solo profiles (`store/profile.py`) --
    and the alternative was a second copy of the `kubectl exec` argv. The subdirectory under
    `efs_root` is the only thing that differs, so subclasses set `remote_subdir` and get the
    rest. `EfsMirror`'s constructor signature is unchanged: it is called positionally.
    """

    remote_subdir = ""

    def __init__(self, key_hash: str, local_root: str | Path,
                 efs_root: str = DEFAULT_STORE_ROOT,
                 namespace: str = DEFAULT_NAMESPACE,
                 pod: str = DEFAULT_ATTACH_POD, kubectl: str = "kubectl"):
        self.key_hash = key_hash
        self.local_root = Path(local_root)
        self.efs_root = efs_root.rstrip("/")
        self.namespace = namespace
        self.pod = pod
        self.kubectl = kubectl

    # --- paths ---------------------------------------------------------------

    @property
    def remote_key_dir(self) -> str:
        return f"{self.efs_root}/{self.remote_subdir}/{self.key_hash}"

    @property
    def local_key_dir(self) -> Path:
        return self.local_root / self.remote_subdir / self.key_hash

    # --- the pod -------------------------------------------------------------

    def _exec(self, script: str, timeout_s: float = 120.0) -> str:
        return _run([self.kubectl, "exec", "-n", self.namespace, self.pod,
                     "-c", self.pod, "--", "sh", "-c", script], timeout_s=timeout_s)

    def _exec_in(self, script: str, stdin_text: str, timeout_s: float = 120.0) -> str:
        return _run([self.kubectl, "exec", "-i", "-n", self.namespace, self.pod,
                     "-c", self.pod, "--", "sh", "-c", script],
                    timeout_s=timeout_s, input_text=stdin_text)

    def _cat(self, path: str) -> str | None:
        try:
            return self._exec(f"cat {path}")
        except DeployError:
            return None

    def _ls(self, remote_dir: str) -> list:
        """`.json` names in a remote directory, or [] if it does not exist.

        A missing directory is not an error anywhere in this module: a brand-new key has none
        until its first run, and treating that as a failure would make opening one impossible.
        """
        out = self._exec(f"test -d {remote_dir} && ls -1 {remote_dir} 2>/dev/null || true")
        return [n.strip() for n in out.splitlines() if n.strip().endswith(".json")]

    def _chmod_dirs(self, *dirs: str) -> str:
        """A shell fragment creating directories both uids can write into.

        This is not incidental. `attach-pvc` runs as root and the harness Job runs as uid 1000,
        so a directory the mirror creates with a default umask is one the harness cannot write
        into -- and the harness discovers that *after* the measurement window has closed, which
        costs a run of GPU time to learn. It is how the first live sweep failed: the key
        directory existed as `root:root 0755`, `write_record`'s `mkdir` raised EPERM, and a
        completed `verdict: measured` record survived only in the pod log.

        1777 rather than 0777: the sticky bit means neither uid can delete the other's files,
        which is the property an append-only store wants anyway. `chmod` is unconditional so it
        also repairs a directory an earlier version created too strictly.
        """
        joined = " ".join(dirs)
        return f"mkdir -p {joined} && chmod 1777 {joined}"


class EfsMirror(PodMirror):
    """One validity key's directory, moved between EFS and a local path."""

    remote_subdir = "observations"

    # --- pull ----------------------------------------------------------------

    def pull(self) -> dict:
        """Copy this key's directory down. Returns `{"observations": n, "sidecars": [...]}`.

        A missing remote directory is not an error: a brand-new validity key has none until its
        first run, and treating that as a failure would make opening a new key impossible.
        """
        names = self._ls(f"{self.remote_key_dir}/observations")
        (self.local_key_dir / "observations").mkdir(parents=True, exist_ok=True)
        pulled = 0
        for name in names:
            body = self._cat(f"{self.remote_key_dir}/observations/{name}")
            if body is None:
                continue
            try:
                json.loads(body)
            except json.JSONDecodeError as exc:
                # Written locally anyway. `ObservationStore.load` counts and names unreadable
                # records, and the agent refuses to search against a partial history -- which is
                # the behaviour that should decide this, not a silent skip here.
                log.warning("%s on EFS is not valid JSON (%s); mirroring it as-is so the "
                            "store's own unreadable-record check sees it", name, exc)
            (self.local_key_dir / "observations" / name).write_text(body)
            pulled += 1
        sidecars = []
        for name in SIDECAR_FILES:
            body = self._cat(f"{self.remote_key_dir}/{name}")
            if body is not None:
                (self.local_key_dir / name).write_text(body)
                sidecars.append(name)
        # Past searches' own summaries, brought down with everything else so the local store is
        # actually a mirror. Nothing in the agent reads them -- the evidence is the records, and a
        # conclusion must not be re-consumed as one -- but a person asking "what has already been
        # tried under this key" should not have to exec into a pod to find out.
        pulled_searches = 0
        for name in self._ls(f"{self.remote_key_dir}/searches"):
            body = self._cat(f"{self.remote_key_dir}/searches/{name}")
            if body is None:
                continue
            (self.local_key_dir / "searches").mkdir(parents=True, exist_ok=True)
            (self.local_key_dir / "searches" / name).write_text(body)
            pulled_searches += 1
        log.info("mirrored %d observations, %d search summaries and %s from %s to %s",
                 pulled, pulled_searches, sidecars or "no sidecars",
                 self.remote_key_dir, self.local_key_dir)
        return {"observations": pulled, "sidecars": sidecars, "searches": pulled_searches}

    def read_record(self, run_id: str) -> dict | None:
        """One record straight off EFS, without mirroring the whole key. For inspection."""
        body = self._cat(f"{self.remote_key_dir}/observations/{run_id}.json")
        return None if body is None else json.loads(body)

    def list_records(self) -> list:
        """Run ids on EFS under this key, or [] if the key has no directory yet."""
        return sorted(n[:-5] for n in self._ls(f"{self.remote_key_dir}/observations"))

    # --- push ----------------------------------------------------------------

    def push_record(self, run_id: str, record: dict) -> bool:
        """Upload one record the agent synthesized. False if EFS already has that run id.

        Never overwrites. A record already on EFS was written by a harness Job, and a
        synthesized one replacing a measured one would be an unrecoverable loss of the only copy
        of a measurement.
        """
        remote = f"{self.remote_key_dir}/observations/{run_id}.json"
        exists = self._exec(f"test -f {remote} && echo yes || echo no").strip()
        if exists == "yes":
            log.warning("not pushing %s: EFS already has a record for that run id", run_id)
            return False
        body = json.dumps(record, indent=2, sort_keys=True) + "\n"
        self._exec_in(f"{self._mkdirs()} && cat > {remote}", body)
        log.info("pushed synthesized record %s to %s", run_id, remote)
        return True

    def _mkdirs(self) -> str:
        """Create the key directory and its `observations/`, writable by the harness too."""
        return self._chmod_dirs(self.remote_key_dir, f"{self.remote_key_dir}/observations")

    def push_calibration(self, calibration_json: str) -> None:
        """Write `calibration.json` beside the observations, replacing any existing one.

        Overwrites, unlike every other push here, and that is the correct asymmetry: a record is
        a measurement and must never be replaced, while a calibration is a *summary* of the
        records under it. More repeats give a better floor from the same data, and refusing to
        update would pin the search's stop condition to whatever the store held first.
        """
        remote = f"{self.remote_key_dir}/calibration.json"
        self._exec_in(f"{self._mkdirs()} && cat > {remote}", calibration_json)
        log.info("wrote %s", remote)

    def push_summary(self, run_token: str, summary_json: str) -> str:
        """Write a search's own summary under `searches/<run_token>.json`.

        The store held the evidence and not the conclusion. Every measurement a search took was in
        EFS within seconds of being taken, while what the search *decided* -- the verdict, the
        reasoner's turns, the placements as ordinals, the targets they were judged against -- lived
        only in whatever local file `--summary` happened to name. That is backwards for the artifact
        a person actually reads, and it has already cost something: the first 14B/30B search's
        records were superseded and deleted, and its summary survives only because it was copied
        into the repo by hand.

        Kept out of `observations/`, which is deliberate rather than tidy. `ObservationStore` globs
        `observations/*.json` and parses everything it finds as an `Observation`, so a summary in
        there would be either a parse failure or -- worse -- read as evidence. Up here it is a
        document beside `key.json` and `calibration.json`, which is what it is.

        One file per run token, so searches accumulate instead of overwriting. Two searches under
        one key are two answers, and the second does not supersede the first: it was taken at a
        different time, possibly on a different node, and its own record says so.
        """
        remote = f"{self.remote_key_dir}/searches/{run_token}.json"
        self._exec_in(f"{self._mkdirs()} && "
                      f"{self._chmod_dirs(f'{self.remote_key_dir}/searches')} && "
                      f"cat > {remote}", summary_json)
        log.info("wrote %s", remote)
        return remote

    def push_export(self, name: str, document: str) -> str:
        """Write a canonical export under `exports/<name>.yaml`.

        Overwrites, like `push_calibration` and unlike everything else here, and for the same
        reason: an export is *derived*. It is the stored summary plus the fleet file rewritten in
        terms that outlive the node (`agent/export.py`), so re-exporting the same search must be
        allowed to produce the file again -- a better exporter should be able to improve an existing
        document rather than having to invent a new name to sit beside it.

        Pushed at all because this is the artifact somebody deploys from, and the store already
        learned this lesson once with search summaries: the evidence was on EFS within seconds while
        the conclusion lived in whatever local file `--summary` happened to name.
        """
        remote = f"{self.remote_key_dir}/exports/{name}.yaml"
        self._exec_in(f"{self._mkdirs()} && "
                      f"{self._chmod_dirs(f'{self.remote_key_dir}/exports')} && "
                      f"cat > {remote}", document)
        log.info("wrote %s", remote)
        return remote

    def push_drift(self, deployment: str, document: str) -> str:
        """Write a KAI deployment's drift records under `drift/<deployment>.jsonl`, beside the
        exports. Overwrites, like `push_export`: the records are derived from Prometheus, and the
        local file is the one that accumulates them (`colocation.kai drift`)."""
        remote = f"{self.remote_key_dir}/drift/{deployment}.jsonl"
        self._exec_in(f"{self._mkdirs()} && "
                      f"{self._chmod_dirs(f'{self.remote_key_dir}/drift')} && "
                      f"cat > {remote}", document)
        log.info("wrote %s", remote)
        return remote

    def push_key(self, key_json: str) -> None:
        """Write `key.json` beside the observations if it is not already there.

        Without it the directory name is unattributable later, which defeats keying at all
        (see `ValidityKey.key_json`). Refuses to change an existing one for the same reason
        `ObservationStore.write_key` does.
        """
        remote = f"{self.remote_key_dir}/key.json"
        if self._exec(f"test -f {remote} && echo yes || echo no").strip() == "yes":
            existing = self._cat(remote)
            if existing is not None and existing != key_json:
                raise MirrorError(
                    f"{remote} exists and does not match this key. Same hash, different "
                    "contents means two environments have been filed together, and every "
                    "comparison across them is invalid.")
            # Still fix the permissions: this is the path a re-run of an existing key takes,
            # and the harness's write must succeed on it.
            self._exec(self._mkdirs())
            return
        self._exec_in(f"{self._mkdirs()} && cat > {remote}", key_json)
        log.info("wrote %s", remote)


class HardwareMirror(PodMirror):
    """The hardware store on EFS: every environment's device laws, and the instance shapes.

    **Not keyed to this run's environment, unlike every other mirror here**, and the reason is the
    one `store/hwfacts.HardwareIndex` gives for reading across hardware keys: how much memory a card
    has is not conditioned by the driver or the image, and `plan` needs that figure *before* it has
    leased anything or agreed with anybody about which image it will run. So `pull` brings down every
    key directory it finds and lets the local reader decide whether they agree -- `device_total`
    refuses when two of them do not, which is where a disagreement belongs.

    This exists because deleting the GPU capacity table would otherwise have traded a constant that
    was wrong everywhere for a measurement that is right in one place. The table was in the source, so
    every checkout had it; a store that never leaves the desktop that captured means a second
    operator's `plan` refuses an instance type a colleague leased last week, and the two boots that
    measured the card get paid for again.

    Push is per-file and asymmetric, matching what each file is. A device law supersedes, on
    `HardwareStore.write`'s argument -- a later slope replaces a driver reading and carries its own
    boots. A shape does NOT: two nodes disagreeing about how many GPUs an instance type has means one
    of them was not what it claimed, and `push_shape` refuses rather than erasing the evidence, which
    is `write_shape`'s rule enforced on the shared copy as well as the local one.
    """

    remote_subdir = "hwfacts"

    def __init__(self, local_root: str | Path, efs_root: str = DEFAULT_STORE_ROOT,
                 namespace: str = DEFAULT_NAMESPACE,
                 pod: str = DEFAULT_ATTACH_POD, kubectl: str = "kubectl"):
        # No key hash: this mirror is the whole `hwfacts/` tree. Passed as "" rather than made
        # optional so `remote_key_dir` cannot be used here by accident -- it would name the root.
        super().__init__("", local_root, efs_root=efs_root, namespace=namespace, pod=pod,
                         kubectl=kubectl)

    # --- paths ---------------------------------------------------------------

    @property
    def remote_base(self) -> str:
        return f"{self.efs_root}/{self.remote_subdir}"

    @property
    def local_base(self) -> Path:
        return self.local_root / self.remote_subdir

    def _ls_dirs(self, remote_dir: str) -> list:
        """Subdirectory names under a remote directory, or [] if it does not exist."""
        out = self._exec(f"test -d {remote_dir} && find {remote_dir} -mindepth 1 -maxdepth 1 "
                         "-type d -exec basename {} \\; 2>/dev/null || true")
        return sorted(n.strip() for n in out.splitlines() if n.strip())

    # --- pull ----------------------------------------------------------------

    def pull(self) -> dict:
        """Copy the whole hardware store down. Returns `{"devices": n, "shapes": m, "keys": [...]}`.

        Every key directory, not this run's, because the consumer reads across them. A missing remote
        tree is not an error: nobody has captured on this cluster yet, and the first run to do so
        is what creates it.
        """
        keys, devices = [], 0
        for key_hash in self._ls_dirs(self.remote_base):
            if key_hash == "instances":
                continue                     # keyed by nothing; pulled below with its own rules
            local = self.local_base / key_hash
            for name in self._ls(f"{self.remote_base}/{key_hash}/devices"):
                body = self._cat(f"{self.remote_base}/{key_hash}/devices/{name}")
                if body is None:
                    continue
                try:
                    json.loads(body)
                except json.JSONDecodeError as exc:
                    # Mirrored as-is, on `HardwareStore.all`'s reasoning: an unreadable device law is
                    # a hard error there, because the fallback for a missing device total is a
                    # refusal and a silent skip here would turn the error into that refusal --
                    # reading as "nobody has measured this card" when somebody did.
                    log.warning("%s on EFS is not valid JSON (%s); mirroring it as-is", name, exc)
                (local / "devices").mkdir(parents=True, exist_ok=True)
                (local / "devices" / name).write_text(body)
                devices += 1
            body = self._cat(f"{self.remote_base}/{key_hash}/key.json")
            if body is not None:
                local.mkdir(parents=True, exist_ok=True)
                (local / "key.json").write_text(body)
            keys.append(key_hash)
        shapes = 0
        for name in self._ls(f"{self.remote_base}/instances"):
            body = self._cat(f"{self.remote_base}/instances/{name}")
            if body is None:
                continue
            (self.local_base / "instances").mkdir(parents=True, exist_ok=True)
            (self.local_base / "instances" / name).write_text(body)
            shapes += 1
        log.info("mirrored %d device law(s) under %d environment(s) and %d instance shape(s) from "
                 "%s to %s", devices, len(keys), shapes, self.remote_base, self.local_base)
        return {"devices": devices, "shapes": shapes, "keys": keys}

    # --- push ----------------------------------------------------------------

    def _mkdirs(self, key_hash: str = "") -> str:
        dirs = [self.remote_base, f"{self.remote_base}/instances"]
        if key_hash:
            dirs += [f"{self.remote_base}/{key_hash}", f"{self.remote_base}/{key_hash}/devices"]
        return self._chmod_dirs(*dirs)

    def push_sheet(self, key_hash: str, facts_id: str, facts: dict) -> str:
        """Upload one device law under one environment, superseding any sheet for the same card."""
        remote = f"{self.remote_base}/{key_hash}/devices/{facts_id}.json"
        body = json.dumps(facts, indent=2, sort_keys=True) + "\n"
        self._exec_in(f"{self._mkdirs(key_hash)} && cat > {remote}", body)
        log.info("wrote %s", remote)
        return remote

    def push_shape(self, shape_id: str, shape: dict) -> str:
        """Upload one instance shape, refusing to contradict one already there.

        Compared on the two fields that are the fact -- how many GPUs of what product -- and not
        byte-for-byte, because `observed_at` and `node` legitimately differ between two captures of
        the same instance type and refusing on those would make the second capture look like a
        conflict.
        """
        remote = f"{self.remote_base}/instances/{shape_id}.json"
        existing = self._cat(remote)
        if existing is not None:
            try:
                old = json.loads(existing)
            except json.JSONDecodeError:
                old = None
            if old is not None and (old.get("gpu_count"), old.get("gpu_product")) != (
                    shape.get("gpu_count"), shape.get("gpu_product")):
                raise MirrorError(
                    f"{remote} says {old.get('instance_type')} is {old.get('gpu_count')}x"
                    f"{old.get('gpu_product')} (from node {old.get('node') or 'unknown'}) and this "
                    f"capture says {shape.get('gpu_count')}x{shape.get('gpu_product')}. An instance "
                    "type has one shape, so one of the two nodes was not what it claimed. Nothing "
                    "overwritten.")
            if old is not None:
                self._exec(self._mkdirs())
                return remote
        body = json.dumps(shape, indent=2, sort_keys=True) + "\n"
        self._exec_in(f"{self._mkdirs()} && cat > {remote}", body)
        log.info("wrote %s", remote)
        return remote

    def push_key(self, key_hash: str, key_json: str) -> None:
        """Write one environment's `key.json`, refusing to change an existing one."""
        remote = f"{self.remote_base}/{key_hash}/key.json"
        existing = self._cat(remote)
        if existing is not None:
            if existing != key_json:
                raise MirrorError(
                    f"{remote} exists and does not match this hardware key. Same hash, different "
                    "contents means two environments have been filed together, and every device law "
                    "under it would be attributed to the wrong driver or image.")
            self._exec(self._mkdirs(key_hash))
            return
        self._exec_in(f"{self._mkdirs(key_hash)} && cat > {remote}", key_json)
        log.info("wrote %s", remote)


class ProfileMirror(PodMirror):
    """One environment's solo profiles, moved between EFS and a local path (`store/profile.py`).

    Same plumbing as `EfsMirror`, and one difference that matters: **push overwrites**. A profile
    row is the current best statement about a model at a TP degree on this hardware, so a
    re-profile supersedes -- the argument `ProfileStore.write` makes. The measurement behind each
    row is still append-only in its own validity key's `observations/`, and each row names the run
    that produced it, so superseding here loses no measurement.

    The profiles are what makes the *second* fleet on a machine cheap to place, so they are worth
    pulling even when nothing under this fleet's key exists yet. `pull` is therefore safe on an
    empty remote and returns 0 rather than raising.
    """

    remote_subdir = "profiles"

    def pull(self) -> dict:
        """Copy this environment's profile rows down. Returns `{"solo": n, "sidecars": [...]}`."""
        names = self._ls(f"{self.remote_key_dir}/solo")
        (self.local_key_dir / "solo").mkdir(parents=True, exist_ok=True)
        pulled = 0
        for name in names:
            body = self._cat(f"{self.remote_key_dir}/solo/{name}")
            if body is None:
                continue
            try:
                json.loads(body)
            except json.JSONDecodeError as exc:
                # Mirrored anyway, for the same reason an unreadable observation is: the store's
                # own reader is what should report it (`ProfileStore.all` logs and treats the
                # degree as unprofiled), not a silent skip here that makes the file vanish.
                log.warning("%s on EFS is not valid JSON (%s); mirroring it as-is", name, exc)
            (self.local_key_dir / "solo" / name).write_text(body)
            pulled += 1
        sidecars = []
        body = self._cat(f"{self.remote_key_dir}/key.json")
        if body is not None:
            (self.local_key_dir / "key.json").write_text(body)
            sidecars.append("key.json")
        log.info("mirrored %d solo profiles and %s from %s to %s", pulled,
                 sidecars or "no sidecars", self.remote_key_dir, self.local_key_dir)
        return {"solo": pulled, "sidecars": sidecars}

    def _mkdirs(self) -> str:
        return self._chmod_dirs(self.remote_key_dir, f"{self.remote_key_dir}/solo")

    def push_profile(self, profile_id: str, profile: dict) -> str:
        """Upload one profile row, replacing any existing row for the same identity."""
        remote = f"{self.remote_key_dir}/solo/{profile_id}.json"
        body = json.dumps(profile, indent=2, sort_keys=True) + "\n"
        self._exec_in(f"{self._mkdirs()} && cat > {remote}", body)
        log.info("wrote %s", remote)
        return remote

    def push_key(self, key_json: str) -> None:
        """Write `key.json` for this environment, refusing to change an existing one.

        Identical in intent to `EfsMirror.push_key` and not shared with it, because the two write
        different files under different roots and folding them together would need the subclass to
        supply both a path and a `_mkdirs`, which is more indirection than the six lines saves.
        """
        remote = f"{self.remote_key_dir}/key.json"
        if self._exec(f"test -f {remote} && echo yes || echo no").strip() == "yes":
            existing = self._cat(remote)
            if existing is not None and existing != key_json:
                raise MirrorError(
                    f"{remote} exists and does not match this profile key. Same hash, different "
                    "contents means two environments have been filed together, and every profile "
                    "under it would be attributed to the wrong machine.")
            self._exec(self._mkdirs())
            return
        self._exec_in(f"{self._mkdirs()} && cat > {remote}", key_json)
        log.info("wrote %s", remote)
