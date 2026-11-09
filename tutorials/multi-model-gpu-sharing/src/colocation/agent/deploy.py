"""Making a placement real, and the readiness gate (impl §5, §11).

The agent talks to this through `FleetDeployer`, never to `helm` directly. Two implementations
ship: `HelmDeployer`, which drives the real cluster, and `FakeDeployer`, which does not. The
interface exists so the loop in `loop.py` is testable offline -- the whole store, solver and
model layer already is, and a loop that can only be exercised against a live g6e node would be
the one component nobody can check before spending GPU time on it.

**The readiness gate is a classifier, not a boolean.** impl §5 requires all tenant pods Ready
*and* all verification init containers passed. Those cannot actually disagree -- an init
container that fails blocks the pod forever, so it can never be Ready -- but the agent needs to
know *which* happened, because they mean different things:

  - verification failed  -> the MPS envelope was not applied as requested. A configuration
    error or a mis-keyed UUID; re-proposing it is pointless.
  - engine crash-looped   -> the envelope was applied and the engine could not live in it.
    That is a real feasibility fact about this placement and belongs in the store.
  - timed out still pulling weights -> nothing about the placement; retryable.

`helm --wait` collapses all three into "timed out", which is why it is not used.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import engines
from .render import DEFAULT_NAMESPACE

log = logging.getLogger(__name__)

# Chart path relative to the tutorial directory (tutorials/multi-model-gpu-sharing).
DEFAULT_CHART = engines.get(engines.DEFAULT_ENGINE).chart

# tutorials/multi-model-gpu-sharing/, which is what an engine's `chart` is relative to.
REPO = Path(__file__).resolve().parents[3]

READY = "ready"
NOT_READY_VERIFY = "verification_failed"
NOT_READY_ENGINE = "engine_failed"
NOT_READY_TIMEOUT = "timeout"
NOT_READY_SCHEDULE = "unschedulable"

# The hand-installed daemon release, protected by name. This is a FLOOR, not the mechanism:
# `mps-daemon` is what a human installs when working on the cluster directly, it is not owned by
# any run, and no run may remove it. Uninstalling it would tear down every MPS server on the node
# and with it every tenant, and release the 8-GPU reservation that keeps Karpenter from reclaiming
# the node. (`dcgm-prototype` was here until 2026-09-04 and is gone because the release is: the
# exporter now ships inside the mps-daemon chart, so this one name protects both.)
#
# THE MECHANISM IS OWNERSHIP, and it is what `HelmDeployer(lease_release=..., tenant_prefix=...)`
# expresses. Once each run installs its own daemon and prefixes its own tenants, "infrastructure"
# stops being a set of names and becomes a relationship: the releases this run created are its own
# to remove, this run's *lease* is untouchable even though this run created it, and every other
# release in the namespace -- including another run's daemon, exporter and tenants -- is somebody
# else's. A name list cannot say any of that, because the names are not known when it is written.
PROTECTED_RELEASES = frozenset({"mps-daemon"})


class DeployError(Exception):
    """A command failed, or a request was refused as unsafe."""


def guard_release(name: str, lease_release: str = "", tenant_prefix: str = "") -> None:
    """Refuse to install or uninstall anything this run does not own. Three ways to fail.

    Shared by `HelmDeployer` and `FakeDeployer` so the offline tests exercise the same rule the
    cluster gets, rather than a simplified copy of it that passes for a different reason.

    The prefix check is the one that matters most and it is the one that looks least alarming.
    Without it, a release name is a global in a shared namespace: two runs both call a tenant
    `m1`, and `helm upgrade --install` does not report a conflict -- it adopts the other run's
    release and re-pins its pod to this run's node (see `render.release_name`). Requiring the
    prefix turns that from a silent takeover into a refusal here.
    """
    if name in PROTECTED_RELEASES:
        raise DeployError(
            f"refusing to touch release {name!r}: it is infrastructure, not a tenant, and it "
            "belongs to whoever installed it by hand. Uninstalling mps-daemon tears down every "
            "MPS server on its node -- and with it every tenant and the 8-GPU reservation that "
            "keeps the node from being reclaimed.")
    if lease_release and name == lease_release:
        raise DeployError(
            f"refusing to touch release {name!r}: it is THIS run's node lease. This run created "
            "it, and it is still not a tenant -- the daemon pod is what holds the node, so "
            "uninstalling it here would drop the lease mid-search and hand the node back to "
            "Karpenter. Releasing it is `capture.release()`'s job, which also re-applies the "
            "holder.")
    if tenant_prefix and not name.startswith(f"{tenant_prefix}-"):
        raise DeployError(
            f"refusing to touch release {name!r}: this run owns only releases prefixed "
            f"{tenant_prefix + '-'!r}. Anything else in this namespace is another run's -- its "
            "daemon, its exporter or its tenants -- and installing over it would re-pin its pods "
            "to this run's node while it is measuring them.")


@dataclass
class ReadyResult:
    """The readiness gate's verdict, and the evidence for it."""

    state: str                       # one of the READY / NOT_READY_* constants
    waited_s: float
    per_tenant: dict = field(default_factory=dict)   # release -> short status string
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.state == READY


class FleetDeployer:
    """What the loop needs from the world. Three verbs."""

    def apply(self, releases) -> None:
        raise NotImplementedError

    def wait_ready(self, releases, timeout_s: float) -> ReadyResult:
        raise NotImplementedError

    def destroy(self, release_names) -> None:
        raise NotImplementedError

    def live_tenant_releases(self) -> list:
        """Tenant releases currently installed. Used to clear a previous iteration."""
        raise NotImplementedError


def _run(argv, timeout_s: float = 300.0, input_text: str | None = None) -> str:
    log.debug("$ %s", " ".join(shlex.quote(a) for a in argv))
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                           input=input_text)
    except subprocess.TimeoutExpired as exc:
        raise DeployError(f"timed out after {timeout_s}s: {' '.join(argv)}") from exc
    if p.returncode != 0:
        raise DeployError(
            f"exit {p.returncode}: {' '.join(argv)}\n--- stdout ---\n{p.stdout}\n"
            f"--- stderr ---\n{p.stderr}")
    return p.stdout


class HelmDeployer(FleetDeployer):
    """`helm upgrade --install` per tenant, plus a polled readiness gate.

    One release per tenant rather than one umbrella release. impl §5 describes an umbrella
    chart, and this is the same rendering with a flatter Helm topology: a per-tenant release
    means a tenant whose values did not change is not touched at all, so a search that moves
    one tenant does not restart the other three. Under an umbrella every iteration would
    re-render every subchart, and vLLM startup is minutes.
    """

    def __init__(self, chart_path: str = "", namespace: str = DEFAULT_NAMESPACE,
                 kubectl: str = "kubectl", helm: str = "helm",
                 tenant_label: str = "colocation/tenant",
                 lease_release: str = "", tenant_prefix: str = ""):
        # Empty means: each tenant comes from its own engine's chart, which is the normal case and
        # the only one that can serve a fleet whose tenants do not all share a stack. A path here is
        # an override for a chart under development, and `_chart_for` refuses it against a tenant it
        # does not belong to rather than installing the wrong templates over the right values.
        self.chart_override = str(Path(chart_path).resolve()) if chart_path else ""
        if self.chart_override:
            self._chart_name(self.chart_override)
        self.namespace = namespace
        self.kubectl = kubectl
        self.helm = helm
        self.tenant_label = tenant_label
        # This run's own daemon release (`capture.NodeLease.release`), which this deployer must
        # never touch even though the same run created it, and the prefix every release it *may*
        # touch carries (`capture.NodeLease.tenant_prefix`). Both empty means the old
        # one-of-everything behaviour: the name list below is the only guard.
        self.lease_release = lease_release
        self.tenant_prefix = tenant_prefix

    # --- which chart ---------------------------------------------------------

    @staticmethod
    def _chart_name(path: str) -> str:
        """The name in a directory's Chart.yaml, or a refusal that it is not a chart."""
        meta = Path(path, "Chart.yaml")
        if not meta.exists():
            raise DeployError(f"{path} is not a Helm chart (no Chart.yaml)")
        try:
            name = (yaml.safe_load(meta.read_text()) or {}).get("name") or ""
        except (OSError, yaml.YAMLError) as exc:
            raise DeployError(f"{meta} could not be read: {exc}") from exc
        if not name:
            raise DeployError(f"{meta} names no chart, so nothing can be checked against it")
        return str(name)

    def _chart_for(self, release: str, values: dict) -> str:
        """The chart this tenant's values must be installed from.

        A tenant's values say which engine they describe (`render.render_values`), and an engine
        says which chart it deploys. That pairing is the whole of the decision, and it has to be per
        tenant rather than per run: a fleet may state `engine:` on individual tenants, and a single
        chart for the whole run would render one stack's values block into the other stack's
        templates. Helm does not complain about that -- an unrecognised values block is simply
        ignored -- so the tenant would come up on the chart's own defaults, serving a model at a
        batch cap and a memory budget nobody chose, and nothing downstream would say so.

        An explicit `--chart` overrides the pairing only where it is the *same* chart, compared by
        the name in its Chart.yaml against the name the engine's release prefix carries. That keeps
        the flag useful for a copy of a chart under development and refuses it for the case it
        cannot serve.
        """
        eng = engines.get(values.get("engine") or engines.DEFAULT_ENGINE)
        if not self.chart_override:
            return str(REPO / eng.chart)
        want = eng.chart_release_prefix.rstrip("-")
        got = self._chart_name(self.chart_override)
        if got != want:
            raise DeployError(
                f"--chart points at the {got!r} chart, but release {release!r} is a {eng.name!r} "
                f"tenant, which is served by the {want!r} chart. Installing it from the wrong chart "
                "would not fail: Helm ignores a values block a chart does not know, so the tenant "
                "would start on that chart's own defaults instead of the envelope and batch cap this "
                "run solved for. Drop --chart to let each tenant come from its own engine's chart.")
        return self.chart_override

    # --- apply ---------------------------------------------------------------

    def apply(self, releases) -> None:
        releases = list(releases)
        # Down, gone, then up -- never upgraded in place, for a stack that says so. Its controller
        # rolls an update out by starting the new server before it stops the old one, and under MPS
        # that overlap is two envelopes on one GPU: the newcomer's verification reads the outgoing
        # one's memory as a co-tenant and fails, or, worse, passes on a card whose memory is briefly
        # booked twice. An unchanged release is left alone, as for every stack, because an
        # identical upgrade changes nothing and costs nothing.
        #
        # Every changed release goes down before any comes up, not each in turn. Replaced one at a
        # time, the first tenant to return lands on a GPU its old neighbours have not left yet --
        # they are later in the list -- and verifies against their memory. A search that spreads
        # three co-resident tenants onto three GPUs did exactly that on 2026-10-02.
        doomed = []
        for name, values in releases:
            self._guard(name)
            self._chart_for(name, values)
            eng = engines.get(values.get("engine") or engines.DEFAULT_ENGINE)
            if not eng.replace_on_change:
                continue
            installed = self._installed_values(name)
            if installed is not None and _drop_nulls(installed) != _drop_nulls(values):
                log.info("release %s changed and %s does not upgrade in place: replacing it",
                         name, eng.name)
                doomed.append(name)
        if doomed:
            self.destroy(doomed)
        for name, values in releases:
            chart = self._chart_for(name, values)
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
                yaml.safe_dump(values, fh, sort_keys=False)
                path = fh.name
            try:
                # No --wait: the readiness gate below classifies the outcome, and --wait would
                # report a verification failure and a slow weight load identically.
                _run([self.helm, "upgrade", "--install", name, chart,
                      "-n", self.namespace, "-f", path, "--wait=false"], timeout_s=300.0)
            finally:
                Path(path).unlink(missing_ok=True)
            log.info("applied release %s", name)

    def _installed_values(self, name: str):
        """The values a release is installed with, or None if there is no such release."""
        try:
            out = _run([self.helm, "get", "values", name, "-n", self.namespace, "-o", "json"],
                       timeout_s=60.0)
        except DeployError as exc:
            if "not found" in str(exc):
                return None
            raise
        return json.loads(out or "null") or {}

    # --- readiness gate ------------------------------------------------------

    def wait_ready(self, releases, timeout_s: float = 1800.0) -> ReadyResult:
        names = [n for n, _ in releases]
        # How many pods each tenant is. One for a stack that is one server, more for one that is a
        # graph -- and the count matters, because the pods of a graph are not created together.
        want = {n: engines.get(v.get("engine") or engines.DEFAULT_ENGINE).pods_per_tenant
                for n, v in releases}
        t0 = time.monotonic()
        last: dict = {}
        while True:
            pods = self._pods(names)
            last = {}
            terminal: tuple | None = None
            all_ready = True
            for name in names:
                state, why = _classify_tenant(pods.get(name, []), want[name])
                last[name] = why
                if state == READY:
                    continue
                all_ready = False
                if state != NOT_READY_TIMEOUT and terminal is None:
                    terminal = (state, name, why)
            if all_ready:
                waited = time.monotonic() - t0
                log.info("fleet ready after %.0fs: %s", waited, names)
                return ReadyResult(READY, waited, last)
            if terminal is not None:
                state, name, why = terminal
                waited = time.monotonic() - t0
                log.error("readiness gate failed after %.0fs: %s is %s (%s)",
                          waited, name, state, why)
                return ReadyResult(state, waited, last, {"tenant": name, "reason": why})
            if time.monotonic() - t0 > timeout_s:
                waited = time.monotonic() - t0
                log.error("readiness gate timed out after %.0fs: %s", waited, last)
                return ReadyResult(NOT_READY_TIMEOUT, waited, last,
                                   {"reason": f"not all Ready within {timeout_s:.0f}s"})
            time.sleep(10.0)

    def _pods(self, release_names) -> dict:
        """release -> its live pods, newest first. A release with no pod is absent.

        A list, not one pod, because a tenant of a graph-shaped stack is several pods under one
        release (`engines.Engine.pods_per_tenant`), and judging it on the newest alone would pass
        the gate on whichever of them happened to be created last.
        """
        out = _run([self.kubectl, "get", "pods", "-n", self.namespace,
                    "-l", f"{self.tenant_label}", "-o", "json"], timeout_s=60.0)
        items = json.loads(out).get("items", [])
        by_release: dict = {}
        wanted = set(release_names)
        for p in items:
            labels = p.get("metadata", {}).get("labels", {})
            release = labels.get("app.kubernetes.io/instance")
            if release not in wanted:
                continue
            # A Recreate rollout leaves the outgoing pod Terminating for a while. Ignore it:
            # judging readiness on a pod that is going away would either pass the gate on the
            # previous placement's engine or fail it on a normal rollout.
            if p.get("metadata", {}).get("deletionTimestamp"):
                continue
            by_release.setdefault(release, []).append(p)
        for release in by_release:
            by_release[release].sort(key=_creation, reverse=True)
        return by_release

    # --- destroy -------------------------------------------------------------

    def _guard(self, name: str) -> None:
        guard_release(name, self.lease_release, self.tenant_prefix)

    def destroy(self, release_names) -> None:
        gone = []
        for name in release_names:
            self._guard(name)
            try:
                _run([self.helm, "uninstall", name, "-n", self.namespace,
                      "--wait", "--ignore-not-found"], timeout_s=600.0)
                log.info("uninstalled release %s", name)
                gone.append(name)
            except DeployError as exc:
                # Teardown failure must not abort the search: the next iteration's apply is an
                # upgrade of whatever is left, and a stale release is visible in
                # `live_tenant_releases()`. Losing the whole run to a transient API error would
                # be worse.
                log.error("uninstall %s failed, continuing: %s", name, exc)
        self._await_departed(gone)

    def _await_departed(self, release_names, timeout_s: float = 300.0) -> None:
        """Block until these releases have no pod left on the node, Terminating included.

        `helm uninstall --wait` does not cover this and cannot: it waits for the *objects* it
        created to be deleted, and a Deployment is deleted the moment the API server accepts it
        while the pod it made keeps running until its container exits. Measured here: uninstall
        returned in 0.9 s and the outgoing pod was still holding 43 GiB of weights half a minute
        later, because a container that does not exit on SIGTERM lives until the grace period
        runs out.

        That gap is not cosmetic, because of what the *next* tenant does first. Every tenant
        boots with a verification init container that measures the free memory on its devices and
        refuses to start if it does not match the envelope it was given (§4.4). A predecessor
        still resident on one of those devices makes that measurement come back low, so the new
        tenant fails its own envelope check over memory that was already on its way out -- and
        the gate is deliberately unforgiving about a failed verification, because the fault it
        exists to catch is a mis-keyed UUID that must never be retried into passing.

        So the wait belongs here rather than in the gate: once teardown means *departed*, a low
        reading is an over-subscribed placement again, which is exactly what it should mean.

        Engine-agnostic on purpose. How promptly a server dies is a property of the image -- one
        stack's entrypoint forwards the signal and another's leaves a child holding the device --
        and a run's correctness must not depend on which one it got.
        """
        names = [n for n in release_names if n]
        if not names:
            return
        t0 = time.monotonic()
        while True:
            left = self._any_pods(names)
            if not left:
                waited = time.monotonic() - t0
                if waited > 1.0:
                    log.info("released the device(s): %s gone after %.0fs", sorted(names), waited)
                return
            if time.monotonic() - t0 > timeout_s:
                # Not raised, for `destroy`'s reason above, and not silent either: whatever is
                # still there is about to make the next tenant's envelope check read low, and
                # this line is what tells the reader why.
                log.error("%s still has pod(s) %s after %.0fs; going ahead, and the next "
                          "tenant's envelope check may read low because of it",
                          sorted(left), timeout_s, time.monotonic() - t0)
                return
            time.sleep(5.0)

    def _any_pods(self, release_names) -> set:
        """Which of these releases still have a pod object, Terminating ones counted.

        `_pods` skips a pod with a `deletionTimestamp` because readiness must not be judged on
        one. Here the opposite is true: a Terminating pod is precisely what this is waiting for.
        """
        out = _run([self.kubectl, "get", "pods", "-n", self.namespace,
                    "-l", f"{self.tenant_label}", "-o", "json"], timeout_s=60.0)
        wanted = set(release_names)
        return {r for p in json.loads(out).get("items", [])
                if (r := (p.get("metadata", {}).get("labels", {})
                          .get("app.kubernetes.io/instance"))) in wanted}

    def live_tenant_releases(self) -> list:
        """This run's tenant releases. Filtered by chart AND by prefix.

        By chart, because a `helm list` in a shared namespace also returns daemons, exporters and
        whatever else lives there. By prefix, because the caller uses this to clear the previous
        iteration -- so anything returned here is about to be uninstalled, and another run's
        tenants must not be in the list. That is also why the result is passed back through
        `_guard`: this method deciding what to return and the guard deciding what may be removed
        are the same rule, and only one of them is looked at during an incident.
        """
        out = _run([self.helm, "list", "-n", self.namespace, "-o", "json"], timeout_s=60.0)
        names = sorted(r["name"] for r in json.loads(out)
                       if any(r.get("chart", "").startswith(engines.get(n).chart_release_prefix)
                              for n in engines.names())
                       and r["name"] not in PROTECTED_RELEASES
                       and r["name"] != self.lease_release
                       and (not self.tenant_prefix
                            or r["name"].startswith(f"{self.tenant_prefix}-")))
        for n in names:
            self._guard(n)
        return names


def _creation(pod) -> str:
    return pod.get("metadata", {}).get("creationTimestamp", "")


def _drop_nulls(v):
    """`v` with every null-valued key removed, recursively. Helm treats a null value as "unset" and
    need not report it back as given, so a comparison that counted nulls could call a release with an
    unset field changed and replace it for nothing."""
    if isinstance(v, dict):
        return {k: _drop_nulls(x) for k, x in v.items() if x is not None}
    if isinstance(v, list):
        return [_drop_nulls(x) for x in v]
    return v


def _classify_tenant(pods, want: int = 1) -> tuple:
    """(state, why) for one tenant's pods, of which there should be `want`.

    The worst pod decides: any terminal verdict wins over waiting, and waiting wins over Ready. Too
    few pods is waiting, not failure -- a controller that makes the pods of a graph one after another
    is not broken while it is still making them. For a one-pod tenant this is `_classify` exactly,
    which is the property that keeps the single-server stacks' gate unchanged.
    """
    if want <= 1:
        # The newest, as before this took a list: a Recreate rollout can briefly show the outgoing
        # pod beside the incoming one, and the gate is about the incoming one.
        return _classify(pods[0] if pods else None)
    verdicts = [(_classify(p), p.get("metadata", {}).get("name", "?")) for p in pods]
    for (state, why), pod in verdicts:
        if state not in (READY, NOT_READY_TIMEOUT):
            return state, f"{pod}: {why}"
    if len(pods) < want:
        return NOT_READY_TIMEOUT, f"{len(pods)} of {want} pods exist yet"
    for (state, why), pod in verdicts:
        if state != READY:
            return state, f"{pod}: {why}"
    return READY, f"Ready ({len(pods)} pods)"


def _classify(pod) -> tuple:
    """(state, human-readable why) for one pod. `None` means no pod exists yet."""
    if pod is None:
        return NOT_READY_TIMEOUT, "no pod yet"
    status = pod.get("status", {})
    phase = status.get("phase")

    # Verification first: it is an init container, so it gates everything after it, and its
    # failure is the one that says the envelope itself was wrong.
    for c in status.get("initContainerStatuses", []) or []:
        for where in ("state", "lastState"):
            term = (c.get(where) or {}).get("terminated") or {}
            if term and term.get("exitCode", 0) != 0:
                return (NOT_READY_VERIFY,
                        f"init {c.get('name')} exited {term.get('exitCode')} "
                        f"({term.get('reason')})")
        waiting = (c.get("state") or {}).get("waiting") or {}
        if waiting.get("reason") in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull"):
            return NOT_READY_VERIFY, f"init {c.get('name')} {waiting['reason']}"

    for c in status.get("containerStatuses", []) or []:
        waiting = (c.get("state") or {}).get("waiting") or {}
        if waiting.get("reason") in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull"):
            return NOT_READY_ENGINE, f"{c.get('name')} {waiting['reason']}"
        term = (c.get("state") or {}).get("terminated") or {}
        if term and term.get("exitCode", 0) != 0:
            return NOT_READY_ENGINE, f"{c.get('name')} exited {term.get('exitCode')}"

    if phase == "Failed":
        return NOT_READY_ENGINE, f"pod Failed: {status.get('reason')}"
    if phase == "Pending":
        for cond in status.get("conditions", []) or []:
            if cond.get("type") == "PodScheduled" and cond.get("status") == "False":
                # The chart pins `nodeName`, so the scheduler is bypassed and this normally
                # cannot happen. When it does it means the node is gone or cordoned, which no
                # amount of waiting fixes.
                return NOT_READY_SCHEDULE, f"unscheduled: {cond.get('reason')}"

    for cond in status.get("conditions", []) or []:
        if cond.get("type") == "Ready" and cond.get("status") == "True":
            return READY, "Ready"
    return NOT_READY_TIMEOUT, f"phase={phase}, not Ready yet"


class FakeDeployer(FleetDeployer):
    """An in-memory cluster. Records what it was asked to do and answers as scripted.

    `outcomes` maps a frozenset of release names to the `ReadyResult` state to return, so a
    test can make one specific placement fail its readiness gate. Anything unlisted is READY.
    """

    def __init__(self, outcomes: dict | None = None, ready_after_s: float = 0.0,
                 lease_release: str = "", tenant_prefix: str = ""):
        self.outcomes = dict(outcomes or {})
        self.ready_after_s = ready_after_s
        self.applied: list = []       # list of [(release, values), ...] per apply
        self.destroyed: list = []     # list of [release, ...] per destroy
        self.live: set = set()
        self.waits: list = []
        self.lease_release = lease_release
        self.tenant_prefix = tenant_prefix

    def _guard(self, name: str) -> None:
        guard_release(name, self.lease_release, self.tenant_prefix)

    def apply(self, releases) -> None:
        for name, _ in releases:
            self._guard(name)
        self.applied.append([(n, v) for n, v in releases])
        self.live.update(n for n, _ in releases)

    def wait_ready(self, releases, timeout_s: float = 1800.0) -> ReadyResult:
        names = frozenset(n for n, _ in releases)
        self.waits.append(names)
        state = self.outcomes.get(names, READY)
        return ReadyResult(state, self.ready_after_s,
                           {n: state for n in sorted(names)},
                           {} if state == READY else {"reason": "scripted"})

    def destroy(self, release_names) -> None:
        names = list(release_names)
        for n in names:
            self._guard(n)
        self.destroyed.append(names)
        self.live.difference_update(names)

    def live_tenant_releases(self) -> list:
        return sorted(self.live)


def check_prerequisites(engine_names, kubectl: str = "kubectl") -> None:
    """Refuse, before a node is leased, a fleet whose stacks this cluster cannot run.

    Only what an engine declares (`engines.Engine.requires_crds`, `.requires_controller`) is
    checked, so a fleet on the single-server stacks asks the cluster nothing. The controller check is
    the one worth having: a custom resource with no controller behind it is accepted, installed and
    never acted on, and the only symptom is the readiness gate timing out half an hour later.
    """
    for name in sorted(set(engine_names)):
        eng = engines.get(name)
        for crd in eng.requires_crds:
            try:
                _run([kubectl, "get", "crd", crd, "-o", "name"], timeout_s=60.0)
            except DeployError as exc:
                raise DeployError(
                    f"this fleet has {name!r} tenants, and the cluster has no {crd} resource type. "
                    "That stack is served by an operator, which is not installed here (for Dynamo: "
                    "the blueprint's dynamo-platform chart, terraform `dynamo_enabled=true`). "
                    "Refused before a node is leased.") from exc
        if eng.requires_controller:
            out = _run([kubectl, "get", "pods", "-A", "-l", eng.requires_controller, "-o", "json"],
                       timeout_s=60.0)
            ready = [p for p in json.loads(out).get("items", [])
                     if _classify(p)[0] == READY]
            if not ready:
                raise DeployError(
                    f"this fleet has {name!r} tenants, and no Ready pod in the cluster matches "
                    f"{eng.requires_controller!r}, which is the operator that turns those tenants "
                    "into pods. Its resource types are installed, so an install would be accepted -- "
                    "and nothing would ever come up. Refused before a node is leased.")
