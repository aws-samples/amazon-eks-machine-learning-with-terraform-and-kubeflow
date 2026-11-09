"""Running the §6 harness and getting the record back (impl §11).

One measurement is one Job, named after the run. `HARNESS_RUN_ID` is set to that same name as a
literal, overriding the template's `fieldRef: metadata.name` -- which resolves to the *pod* name
and so would file the record under an id containing a suffix the API server invents. The run id
is the record's filename, so the agent has to have chosen it to be able to read it back.

Four things this has to work around, all of them measured facts about this cluster rather than
design choices:

  - **The record lands on EFS, and EFS is only reachable from inside the cluster.** The agent
    runs on the ML Ops desktop (impl §11). So the record is read by exec'ing into the standing
    `attach-pvc` pod, not by mounting anything locally.
  - **The run config is a ConfigMap, and a shared one is a race.** The harness's own Job
    manifest warns that a placeholder copy silently overwrites the real config. Here each run
    gets its own ConfigMap named after the run, so two runs cannot collide even if one is
    still terminating.
  - **A Job can produce a record and still exit non-zero, or exit non-zero with no record at
    all.** `write_record` dumps the record into the log on a filesystem failure precisely so a
    run is recoverable from logs, so that path is followed here rather than treated as a lost
    run: a measurement window costs minutes of GPU time.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from pathlib import Path

import yaml

from .deploy import DeployError, _run
from .render import DEFAULT_NAMESPACE, DEFAULT_STORE_ROOT

log = logging.getLogger(__name__)

DEFAULT_JOB_TEMPLATE = "containers/colocation-harness/deploy/job.yaml"
DEFAULT_ATTACH_POD = "attach-pvc"

# The harness image has **no default registry**, and that is deliberate. It used to default to
# `<account>.dkr.ecr.<region>.amazonaws.com/colocation-harness:v6` -- a private repository in
# one AWS account. Anyone running this anywhere else got the pull refused, and an ImagePullBackOff
# on a private ECR repo reports `no basic auth credentials` rather than "this is not your registry",
# so the cause is not visible from the pod's events. That is a wrong default rather than a missing
# one: the image does not exist until someone builds it from `containers/colocation-harness/` and
# pushes it somewhere their own cluster can pull from (USER-GUIDE.md, step 3). Refusing with that
# instruction costs a sentence; guessing costs a node lease and a confusing pod.
HARNESS_IMAGE_ENV = "COLOCATION_HARNESS_IMAGE"


def default_harness_image() -> str | None:
    """The harness image from the environment, or None if it is not set.

    Read at call time rather than bound at import, so exporting the variable after this module is
    imported still works -- which is what happens in a notebook or a long-lived shell.
    """
    return os.environ.get(HARNESS_IMAGE_ENV) or None

_RECORD_RE = re.compile(
    r"BEGIN OBSERVATION RECORD (?P<id>\S+)\n(?P<body>.*?)\nEND OBSERVATION RECORD",
    re.DOTALL)


class MeasureError(Exception):
    """The run produced no record that could be recovered."""


class HarnessRunner:
    """One verb: drive this placement and hand back the observation record."""

    def measure(self, run_config: dict, run_id: str, timeout_s: float) -> dict:
        raise NotImplementedError


class JobHarness(HarnessRunner):
    """Runs the harness as a Job and reads the record off EFS.

    The Job manifest is read from disk and patched rather than constructed here. Everything in
    it that matters -- the `cpu-compute` nodepool so the load driver does not compete with the
    GPUs it is measuring, the NoSchedule toleration, and the no-Envoy annotation -- is reasoned
    about in that file, and duplicating it in Python would let the two drift.

    The one placement decision that is made here rather than there is the zone, because it is
    not a property of the repo: `node_name` is the node under test, and the Job is pinned to
    that node's zone. The template used to carry a literal AZ, which wedged the pipeline the
    first time this ran in a different region -- a Job that cannot schedule sits Pending for as
    long as the measurement timeout allows, with an engine loaded and burning GPU behind it.
    """

    ZONE_LABEL = "topology.kubernetes.io/zone"

    def __init__(self, job_template: str, namespace: str = DEFAULT_NAMESPACE,
                 image: str | None = None, kubectl: str = "kubectl",
                 store_root: str = DEFAULT_STORE_ROOT,
                 attach_pod: str = DEFAULT_ATTACH_POD,
                 poll_s: float = 20.0, node_name: str | None = None):
        self.template = yaml.safe_load(Path(job_template).read_text())
        if self.template.get("kind") != "Job":
            raise MeasureError(f"{job_template} is not a Job manifest")
        # Refused rather than defaulted. This is the backstop, not the check that saves money: by
        # the time a `JobHarness` exists the node is already leased, so `__main__` refuses a missing
        # image before it captures anything. Both paths, because a caller that builds one of these
        # directly should not get an ImagePullBackOff instead of a sentence.
        image = image or default_harness_image()
        if not image:
            raise MeasureError(
                "no harness image. Build it from containers/colocation-harness/ and push it to a "
                "registry this cluster can pull from, then either export "
                f"{HARNESS_IMAGE_ENV}=<registry>/colocation-harness:<tag> or pass "
                "--harness-image. There is no default, because the only image that ever existed "
                "was in one private ECR repository -- see USER-GUIDE.md, step 3.")
        self.namespace = namespace
        self.image = image
        self.kubectl = kubectl
        self.store_root = store_root
        self.attach_pod = attach_pod
        self.poll_s = poll_s
        self.node_name = node_name
        self._zone: str | None = None

    # --- the run -------------------------------------------------------------

    def measure(self, run_config: dict, run_id: str, timeout_s: float = 5400.0) -> dict:
        cm_name = f"harness-run-{run_id}"
        self._write_configmap(cm_name, run_config)
        manifest = self._job_manifest(run_id, cm_name)
        # A previous Job with this name would make `create` fail and `apply` illegal (a Job's
        # pod template is immutable), so a re-run of the same id starts from a clean slate.
        _run([self.kubectl, "delete", "job", run_id, "-n", self.namespace,
              "--ignore-not-found", "--wait=true"], timeout_s=300.0)
        _run([self.kubectl, "apply", "-n", self.namespace, "-f", "-"], timeout_s=120.0,
             input_text=yaml.safe_dump(manifest, sort_keys=False))
        log.info("harness Job %s applied", run_id)

        state = self._wait(run_id, timeout_s)
        hash_ = run_config["validity_key_hash"]
        record = self._read_record(hash_, run_id)
        if record is None:
            record = self._record_from_logs(run_id)
        if record is None:
            logs = self._logs(run_id, tail=80)
            raise MeasureError(
                f"harness Job {run_id} finished as {state} and produced no record, on EFS or "
                f"in its logs. Last 80 log lines:\n{logs}")
        if state == "failed" and record.get("verdict") is None:
            raise MeasureError(f"harness Job {run_id} failed and its record has no verdict")
        # Cleaned up only on success. A ConfigMap left behind after a failure is how you see
        # what the failed run was actually asked to do.
        _run([self.kubectl, "delete", "configmap", cm_name, "-n", self.namespace,
              "--ignore-not-found"], timeout_s=60.0)
        log.info("run %s verdict=%s", run_id, record.get("verdict"))
        return record

    def _write_configmap(self, name: str, run_config: dict) -> None:
        body = yaml.safe_dump(run_config, sort_keys=False)
        cm = {"apiVersion": "v1", "kind": "ConfigMap",
              "metadata": {"name": name, "namespace": self.namespace},
              "data": {"run.yaml": body}}
        _run([self.kubectl, "apply", "-n", self.namespace, "-f", "-"], timeout_s=60.0,
             input_text=yaml.safe_dump(cm, sort_keys=False))

    def _job_manifest(self, run_id: str, cm_name: str) -> dict:
        m = copy.deepcopy(self.template)
        m["metadata"]["name"] = run_id
        m["metadata"]["namespace"] = self.namespace
        spec = m["spec"]["template"]["spec"]
        containers = spec["containers"]
        if len(containers) != 1:
            raise MeasureError("the Job template must have exactly one container")
        containers[0]["image"] = self.image
        self._pin_run_id(containers[0], run_id)
        for vol in spec.get("volumes", []):
            if vol.get("configMap", {}).get("name") == "colocation-harness-run":
                vol["configMap"]["name"] = cm_name
                break
        else:
            raise MeasureError(
                "the Job template has no volume mounting the `colocation-harness-run` "
                "ConfigMap, so there is nothing to point at this run's config")
        zone = self._lease_zone()
        if zone:
            spec.setdefault("nodeSelector", {})[self.ZONE_LABEL] = zone
        return m

    def _lease_zone(self) -> str | None:
        """The zone of the node under test, looked up once and reused.

        None when there is no node to follow or the node carries no zone label, and then the
        template's placement is left exactly as written -- an unpinned driver still measures
        correctly, it just may cross an AZ to reach the engines, so this degrades the instrument's
        repeatability rather than the run.
        """
        if self.node_name is None or self._zone is not None:
            return self._zone
        try:
            out = _run([self.kubectl, "get", "node", self.node_name, "-o", "json"],
                       timeout_s=60.0)
            zone = json.loads(out).get("metadata", {}).get("labels", {}).get(self.ZONE_LABEL)
        except (DeployError, json.JSONDecodeError) as exc:
            log.warning("could not read %s off node %s, leaving the harness unpinned: %s",
                        self.ZONE_LABEL, self.node_name, exc)
            return None
        if not zone:
            log.warning("node %s carries no %s, leaving the harness unpinned",
                        self.node_name, self.ZONE_LABEL)
            return None
        log.info("harness pinned to zone %s, the zone of %s", zone, self.node_name)
        self._zone = zone
        return zone

    @staticmethod
    def _pin_run_id(container: dict, run_id: str) -> None:
        """Set HARNESS_RUN_ID to a literal, replacing the template's downward-API reference.

        The template reads it from `fieldRef: metadata.name`, which inside a pod resolves to the
        *pod* name -- `<job>-<random suffix>` for a Job. For a hand-run Job that is harmless (it
        is where runs 1-6's ids like `colocation-harness-5-56pz7` come from), but the agent has
        to be able to *find* the record it just caused, and `record_path` can only name a file
        whose id the agent chose. A five-character suffix the API server invents is not
        predictable, so the record would be unreadable on EFS and the run would fall back to
        scraping the pod log.
        """
        env = container.setdefault("env", [])
        for e in env:
            if e.get("name") == "HARNESS_RUN_ID":
                e.pop("valueFrom", None)
                e["value"] = run_id
                return
        env.append({"name": "HARNESS_RUN_ID", "value": run_id})

    def _wait(self, run_id: str, timeout_s: float) -> str:
        t0 = time.monotonic()
        while True:
            out = _run([self.kubectl, "get", "job", run_id, "-n", self.namespace,
                        "-o", "json"], timeout_s=60.0)
            st = json.loads(out).get("status", {})
            if st.get("succeeded"):
                log.info("harness Job %s succeeded after %.0fs", run_id,
                         time.monotonic() - t0)
                return "succeeded"
            if st.get("failed"):
                log.error("harness Job %s failed after %.0fs", run_id, time.monotonic() - t0)
                return "failed"
            if time.monotonic() - t0 > timeout_s:
                # Not deleted: the Job is the only place its logs live, and a stuck run is
                # exactly the thing you need to look at.
                raise MeasureError(
                    f"harness Job {run_id} neither succeeded nor failed within {timeout_s:.0f}s. "
                    "It is left running deliberately; inspect it before deleting.")
            time.sleep(self.poll_s)

    # --- reading the record --------------------------------------------------

    def record_path(self, validity_key_hash: str, run_id: str) -> str:
        return (f"{self.store_root}/observations/{validity_key_hash}/observations/"
                f"{run_id}.json")

    def _read_record(self, validity_key_hash: str, run_id: str) -> dict | None:
        path = self.record_path(validity_key_hash, run_id)
        try:
            out = _run([self.kubectl, "exec", "-n", self.namespace, self.attach_pod,
                        "-c", self.attach_pod, "--", "cat", path], timeout_s=120.0)
        except DeployError as exc:
            log.warning("could not read %s from EFS: %s", path, exc)
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise MeasureError(f"{path} is not valid JSON: {exc}") from exc

    def _record_from_logs(self, run_id: str) -> dict | None:
        """Recover a record the harness could not write. See `observation.write_record`."""
        logs = self._logs(run_id, tail=None)
        m = _RECORD_RE.search(logs)
        if not m:
            return None
        log.warning("recovered run %s's record from the Job log: the harness could not write "
                    "it to EFS", run_id)
        return _parse_interleaved(m.group("body"), run_id)

    def _logs(self, run_id: str, tail: int | None) -> str:
        argv = [self.kubectl, "logs", "-n", self.namespace, f"job/{run_id}"]
        if tail is not None:
            argv += [f"--tail={tail}"]
        try:
            return _run(argv, timeout_s=120.0)
        except DeployError as exc:
            return f"(could not read logs: {exc})"


MAX_INTERLEAVED_LINES = 40


def _parse_interleaved(body: str, run_id: str) -> dict:
    """Parse a JSON record that other output has been spliced into.

    `kubectl logs` merges the container's stdout and stderr into one stream in arrival order, so
    a traceback written while the record was being printed lands *inside* it. That is not a
    hypothetical: the first live sweep's record was cut by two lines of the very traceback the
    write failure raised, and a completed `verdict: measured` measurement was thrown away for it.

    The repair is to drop the line the parser stopped on and try again, because a foreign line is
    exactly where JSON becomes invalid. Bounded, and every dropped line is logged in full: if a
    line of the *record* were ever dropped this way the result would be a record missing fields,
    and the schema is what has to catch that -- so the evidence has to be in the log.
    """
    lines = body.split("\n")
    dropped: list[str] = []
    while True:
        try:
            return json.loads("\n".join(lines))
        except json.JSONDecodeError as exc:
            i = exc.lineno - 1
            if len(dropped) >= MAX_INTERLEAVED_LINES or not 0 <= i < len(lines):
                raise MeasureError(
                    f"run {run_id}'s record could not be recovered from its log even after "
                    f"dropping {len(dropped)} interleaved line(s): {exc}. The record is in the "
                    f"Job's log between BEGIN and END OBSERVATION RECORD; recover it by hand "
                    f"rather than re-running, because the measurement itself may be intact."
                ) from exc
            dropped.append(lines.pop(i))
            log.warning("run %s: dropped a line interleaved into its record: %r",
                        run_id, dropped[-1])


class FakeHarness(HarnessRunner):
    """Returns scripted records. Takes a callable so a test can react to the placement.

    `builder(run_config, run_id) -> record`. Every call is recorded in `calls`.
    """

    def __init__(self, builder):
        self.builder = builder
        self.calls: list = []

    def measure(self, run_config: dict, run_id: str, timeout_s: float = 0.0) -> dict:
        self.calls.append((run_id, run_config))
        return self.builder(run_config, run_id)
