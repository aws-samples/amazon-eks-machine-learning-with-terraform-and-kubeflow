"""Leasing a GPU node, by installing the daemon that holds it (impl §3, §5).

**Capture is discovery, and the scheduler is the lock manager.** A run does not ask which node
to use and is not told one. It installs its own `mps-daemon` release, whose pod requests every
GPU on the node; the pod reaching Running *is* the capture, and the agent then reads
`pod.spec.nodeName` to **learn** which node it got. Nothing writes the node name anywhere first,
because at install time there is nothing to write -- that is what capture means.

Four consequences, each the reason something below looks the way it does:

**The pod is the lease, and it is self-cleaning.** The obvious alternative -- label the node
`colocation/leased-by=<run>` -- is a cache, and a cache of a lock lies exactly when it matters:
an agent that dies leaves the label behind, and the next run reads it as "somebody else holds
this" forever, or clears it and double-books the node. A pod cannot outlive its release. The
one piece of cached state the daemon does produce, the GPU-UUID ConfigMap, is therefore treated
here as **suspect data rather than as evidence**: `capture` refuses to start if one already
exists for its release name, and checks that the one it ends up reading carries an
`ownerReference` to the pod it just watched come up.

**A Pending pod is a purchase order, not a stall.** Karpenter provisions a node for an
unschedulable 8-GPU pod, which is how the node comes to exist at all -- so `_wait_for_daemon`
treats Pending as normal progress, unlike the tenant readiness gate in `deploy.py`, where
Pending means the pinned node is gone. It also means every failure path here has to uninstall,
and has to decide whether re-applying `gpu-node-holder` would preserve a node or buy one.

**Releasing a lease re-applies the holder** (decided 2026-09-04 by the user). Uninstalling the
daemon releases the last claim on the node and Karpenter reclaims it ~600 s later -- and a
replacement node has different GPU UUIDs, which fail *silently* as an unbounded MPS envelope
(finding 9). So `release()` hands the node to `manifests/gpu-node-holder.yaml`, which
holds it at priority -1000 so the next run's daemon preempts it. The alternative, letting the
node go, stops paying for an idle g6e but makes every record's `gpu_node` and every stored
UUID refer to a machine that no longer exists.

**And because the holder is what blocks a node's replacement, capturing is what unblocks it.**
The holder carries `karpenter.sh/do-not-disrupt`; preempting it removes that protection, so a
node the cluster already wanted to replace starts going away the moment this run takes it. Two
things follow, and they are a pair.

`_protect_held_nodes` runs *before* the install and puts the same annotation on the **node** that
each holder is currently sitting on. That is the only moment the protection can be transferred
without a gap: after the install, the holder is already gone, and on 2026-09-10 the claim took a
`deletionTimestamp` three seconds later -- well inside the daemon pod's own start-up. Protection
on the node does not depend on any pod, so it survives the handover, and `release` removes it
again once the holder is back to carrying it.

`_refuse_if_disrupted` then runs the instant the node name is known and before anything is
measured, because that is the earliest the question can be asked at all -- capture is discovery,
so there is no node to ask about before the pod lands. What it refuses is narrower than it looks:
a node marked `Drifted` while protected is a node the cluster *wants* to replace and *cannot*,
which is a fact about a fleet keyed to what this node reports about itself, not a hazard. A node
already being disrupted or deleted is refused whatever its annotations say, since neither can be
called off.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..store.environment import topology_digest
from . import smi
from .deploy import DeployError, _run
from .hardware import SHAPE_OBSERVED, Hardware, normalize_gpu
from .render import DEFAULT_NAMESPACE, exporter_service_url
from .spec import NodeInventory

log = logging.getLogger(__name__)

# The lease release is `mps-<token>`. Kept short on purpose: the chart derives the exporter's
# resource names as `<release>-dcgm-exporter` and truncates at 63, and the runtime UUID
# ConfigMap is `<release>-gpu-uuids-<node>` where the node name alone is ~45 characters.
DAEMON_RELEASE_PREFIX = "mps"
DEFAULT_DAEMON_CHART = "charts/machine-learning/model-serving/mps-daemon"
DEFAULT_HOLDER_MANIFEST = "manifests/gpu-node-holder.yaml"

# The container that runs `nvidia-cuda-mps-control`, and the one whose `nvidia-smi` is asked
# what driver and what GPU this actually is.
DAEMON_CONTAINER = "daemon"

# What a DCGM exporter looks like from the outside, whoever installed it. Matched on the container
# image, because the thing that conflicts is the DCGM host engine embedded in that binary and the
# two deployments seen here agree on nothing else: ours is a Deployment named after the release,
# the EKS addon's is a DaemonSet in another namespace, reconciled by an operator from a
# `DcgmExporter` custom resource, with its own labels and its own image registry.
DCGM_IMAGE_HINT = "dcgm-exporter"

# The Karpenter pool every lease is taken from: the daemon's nodeSelector names it, and the DCGM
# check reads its taints to tell an exporter that can follow a lease from one that cannot.
LEASE_NODEPOOL = "cuda"

# The refusal above, and the one command that fixes it. `dcgmExporter.enabled` is a documented key
# of the `amazon-cloudwatch-observability` addon's own configuration schema (checked against
# v6.6.0-eksbuild.1; `aws eks describe-addon-configuration` prints the schema for any version).
# Turning the addon's exporter off there, rather than deleting its DaemonSet or its custom
# resource, is the only durable form: the operator recreates both within seconds.
DCGM_ADDON_REMEDY = """\
CLUSTER=$(kubectl config current-context | cut -d/ -f2)   # or whatever names this cluster
REGION=$(kubectl config current-context | cut -d: -f4)
aws eks update-addon --cluster-name $CLUSTER --region $REGION \\
  --addon-name amazon-cloudwatch-observability \\
  --configuration-values '{"dcgmExporter":{"enabled":false}}' --resolve-conflicts OVERWRITE"""

# Conditions on the node's own claim that mean the cluster has already decided to take it away.
# `Drifted` is the one that has cost a machine here: it is set when the node no longer matches the
# NodePool or node class it was made from -- an AMI bump is the usual cause -- and from then on
# Karpenter is waiting only for a disruption budget and for nothing to block the drain. Matched by
# type and only when `True`, because these are added when they apply rather than carried permanently
# False, and a condition type this does not know about is not treated as one of them.
DISRUPTION_CONDITIONS = ("Drifted", "Disrupted")

# Of those, the one a run may hold a node through. `Drifted` means the node no longer matches the
# NodePool it was made from and the autoscaler would like to replace it -- a standing intention, not
# an action, and one that cannot proceed while `DO_NOT_DISRUPT` is on the node. `Disrupted` is the
# action itself, already begun, and no annotation calls it back.
TOLERABLE_WHEN_PROTECTED = ("Drifted",)

# Karpenter's own opt-out from voluntary disruption -- drift, consolidation and emptiness. It is on
# the holder pod in `manifests/gpu-node-holder.yaml`, which is why an idle drifted node survives
# here for days; a capture preempts that pod, so the same annotation is put on the node for the
# length of the lease. Not invented by this repo and not cluster-specific: the annotation is part of
# the autoscaler's API, and a cluster without that autoscaler has nothing to read it, in which case
# `_protect` writes an annotation nothing acts on and nothing is worse for it.
DO_NOT_DISRUPT = "karpenter.sh/do-not-disrupt"

# How a holder is recognised. Matches the labels in `manifests/gpu-node-holder.yaml`; the coupling
# is deliberate and one-directional -- capture has to be able to find the holders *before* it
# removes them, and the manifest is the thing that decides what a holder is.
HOLDER_SELECTOR = "app.kubernetes.io/name=gpu-node-holder"

# What owns a node here, when anything does. Read-only, and its absence is not an error: a cluster
# with no such resource has no autoscaler that could remove a node under a run, so there is nothing
# to check rather than something that failed to be checked.
NODE_CLAIM_RESOURCE = "nodeclaims.karpenter.sh"

# A run token has to be a DNS-1123 label and it prefixes release names, so it starts with a
# letter and stays in [a-z0-9]. 5 random characters over a 36-symbol alphabet is ~60M
# possibilities -- and a collision is caught by the pre-flight checks rather than tolerated,
# so this only has to make collisions rare, not impossible.
_TOKEN_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
TOKEN_CHARS = 5


class CaptureError(Exception):
    """A node could not be leased, or a lease could not be trusted."""


class EnvironmentMoved(CaptureError):
    """The leased node is not the machine the store was keyed to, so this run stops and re-keys.

    Its own type because it is the one capture failure that is *not* a fault: nothing is broken, the
    node is fine, and the remedy is to run the same command again. Every store this run opened -- the
    solo profiles it resolved TP degrees from, the observations it was about to file -- was addressed
    from the last node of this type, and the one in hand reports something different, which by
    construction means those rows describe another machine.

    Stopping is chosen over re-keying in place, and the reason is what has already happened by the time
    this is raised: degrees were resolved and a baseline was printed from rows under the old address.
    Re-keying mid-run would leave a report whose numbers came from one directory and whose measurements
    went to another, with nothing in either saying so. The new observation is filed first, so the
    re-run needs no flag, no edit and no decision from anybody.
    """


def new_run_token() -> str:
    """A short unique identity for one run, used to name every release it installs.

    One token, two uses: `mps-<token>` is the lease, and `<token>-<tenant>` is each tenant
    release. Both have to be per-run for the same reason -- two runs in one namespace otherwise
    collide on a fixed name, and `helm upgrade --install` resolves a collision by *adopting*
    the other run's release and re-pinning its pods, which is not an error anywhere.
    """
    return "r" + "".join(secrets.choice(_TOKEN_ALPHABET) for _ in range(TOKEN_CHARS))


@dataclass(frozen=True)
class NodeLease:
    """A node this run holds, and everything that was observed about it while holding it.

    Every field here is read off the cluster rather than configured, `hardware` included. It arrives
    resolved from the hardware store and leaves *reconciled with this node*: the GPU count and product
    checked against what the node reports, and the device total either checked against what the driver
    says a context can address (`memory.total` minus `memory.reserved` -- `smi.py` is where that
    subtraction is argued) or, when nothing had measured the card yet, taken from it. So a run that boots an
    engine is multiplying its utilisation by a figure that came off the machine it is holding, which
    is the only thing that makes an envelope mean anything.

    `driver` is recorded here and keys nothing. It is provenance: it attributes a number six months
    later and lets a report say two rows span two drivers, while what it used to be a proxy for -- how
    much of the board a context can address -- is now read and keyed directly (`store/environment.py`).
    """

    token: str
    release: str                 # the mps-daemon release that IS the lease
    node_name: str
    pod_name: str
    pod_uid: str
    hardware: Hardware
    driver: str                  # from nvidia-smi on the node, not from a file
    inventory: NodeInventory
    exporter_url: str
    captured_at: str
    observed: dict = field(default_factory=dict)
    # Nodes this run annotated `DO_NOT_DISRUPT` and is therefore responsible for un-annotating.
    # Only the ones it added: an annotation somebody else put there is somebody else's to remove,
    # and clearing it on release would quietly hand away a node they were protecting.
    protected_nodes: tuple = ()

    @property
    def tenant_prefix(self) -> str:
        """What tenant release names are prefixed with, so two runs cannot collide."""
        return self.token

    def summary(self) -> str:
        return (f"lease {self.release} on {self.node_name}: "
                f"{self.hardware.instance_type}, driver {self.driver}, "
                f"{self.hardware.provenance()}")


def _mib(text: str, field_name: str) -> int:
    """A `--format=nounits` memory field as whole MiB, which is the only unit the driver prints it in.

    Integer, and that is load-bearing twice over. Anything that divides it does so by 1024 and never
    by 1000 -- the engine's own KV report is GiB and the MPS memory limit is a percentage of the
    device, so a decimal gibibyte would put a ~7% error into the one number every envelope on the node
    is a fraction of. And the integer itself is a term in the store's address
    (`store/environment.py`): the quantum the driver reports at is what makes the address hashable
    without any tolerance anywhere, so it must not become a float on the way in.
    """
    try:
        return int(str(text).strip())
    except ValueError:
        raise CaptureError(
            f"nvidia-smi reported {field_name} as {text!r}, which is not a whole number of MiB. The "
            "query asks for --format=csv,noheader,nounits; a unit suffix here means the flag did not "
            "take, and parsing it loosely would risk reading MB as MiB.") from None


def parse_topology(text: str) -> dict:
    """`nvidia-smi topo -m` to `{(gpu_a, gpu_b): link_type}`, GPU-to-GPU pairs only.

    The matrix is what tells a multi-shard tenant's collectives apart -- NVLink against PCIe against a
    hop over the SMP interconnect -- so it is in the environment the store is keyed by
    (`store/environment.py`), as a digest. Parsed here rather than digested here, because the raw
    matrix is also filed on the instance shape: a digest that nobody can read back is a directory name
    with no explanation in it.

    Only the columns whose header is `GPU<n>` are read. The real output carries `CPU Affinity`,
    `NUMA Affinity` and `GPU NUMA ID` columns on the right and may carry `NIC<n>` ones, and those are
    properties of the host rather than links between devices: a NUMA id in the digest would re-key the
    store because a scheduler placed the pod differently.

    Self-pairs (` X `) are dropped by `topology_digest`. ANSI underline escapes are stripped -- real
    output has them around the header -- because a digest that depended on whether the driver thought
    it was writing to a terminal would hash one node two ways.
    """
    clean = re.sub(r"\x1b\[[0-9;]*m", "", str(text))
    columns: dict[int, int] = {}
    links: dict[tuple, str] = {}
    for line in clean.splitlines():
        if line.strip().lower().startswith("legend"):
            break
        cells = [c.strip() for c in line.split("\t")]
        if not columns:
            found = {i: int(m.group(1)) for i, c in enumerate(cells)
                     if (m := re.fullmatch(r"GPU(\d+)", c))}
            if found:
                columns = found
            continue
        head = re.fullmatch(r"GPU(\d+)", cells[0] if cells else "")
        if not head:
            continue
        row = int(head.group(1))
        for i, gpu in columns.items():
            if i < len(cells) and cells[i] and cells[i] != "X":
                links[(row, gpu)] = cells[i]
    return links


def reconcile_hardware(hardware: Hardware, observed: dict, node_name: str, index,
                       image: str = "") -> Hardware:
    """The `Hardware` this run will price with, and what gets filed about the machine.

    Called only after `verify`, so anything the run already claimed and used has been checked.
    What is left is what it did *not* claim, and there are two such things:

    A shape that was stated on the command line rather than read off a node is now an
    observation, so it is adopted and filed. Filing is what makes `--gpu-count` a bootstrap
    rather than a habit: the next run on this instance type resolves it offline.

    A device total nobody had measured is taken from the driver, because the alternative is
    refusing to boot on a card that is sitting right there reporting its own size. It is filed as
    a `driver`-sourced sheet so that `plan` can use it too -- but not as a falsifiable one:
    **nothing says the engine's arithmetic agrees, and after impl §14.4 nothing will.** The
    cross-check was a fit to the engine's own KV report, which is the engine's account of its own
    internals; it is deleted, and the answer of the right shape is a boot that fails and moves the
    walk (§14.2). `HardwareStore.write` still supersedes, so a better reading replaces this one.

    And the third thing, which is not a claim the run made but the address it is about to write under:
    the device facts. They are adopted from the node unconditionally -- it is the better authority than
    the store's memory of the last one -- and when they *differ* from what was filed, the new
    observation is filed and `EnvironmentMoved` stops the run. See that exception for why stopping is
    the honest answer rather than re-keying in place.
    """
    if hardware.shape_source != SHAPE_OBSERVED:
        log.warning("%s's shape was stated on the command line and has now been verified "
                    "against %s (%d x %s). It is filed as an observation, so no run after this "
                    "one needs --gpu-count for this instance type.",
                    hardware.instance_type, node_name, observed["gpu_count"],
                    observed["gpu_product"])
        hardware = hardware.with_shape(gpu_count=observed["gpu_count"],
                                       gpu_product=observed["gpu_product"], node=node_name)
    if not hardware.total_known:
        log.warning("nothing had measured how much memory a %s has, so this run takes the "
                    "driver's own figure from %s: %.3f GiB per device (memory.total minus "
                    "memory.reserved, which is what a CUDA context can address). That is enough to "
                    "size envelopes with, and nothing cross-checks that the engine multiplies the "
                    "same number.",
                    hardware.gpu_product, node_name, observed["device_total_gib"])
        hardware = hardware.with_driver_total(observed["device_total_gib"], node=node_name)

    # What the store thought a node of this type presents, against what this one does. Read before the
    # adoption below, because after it there is nothing left to compare.
    was = (hardware.addressable_mib, hardware.power_limit_w, hardware.p2p_topology)
    now = (int(observed["addressable_mib"]), int(observed["power_limit_w"]),
           str(observed["p2p_topology"]))
    moved = hardware.environment_known and was != now
    hardware = hardware.with_device_facts(addressable_mib=now[0], power_limit_w=now[1],
                                          p2p_topology=now[2], node=node_name)
    if index is not None:
        _file_hardware(hardware, observed, node_name, index, image)
    if moved:
        names = ("addressable_mib", "power_limit_w", "p2p_topology")
        changed = "; ".join(f"{n}: {o!r} -> {v!r}" for n, o, v in zip(names, was, now) if o != v)
        raise EnvironmentMoved(
            f"{node_name} is not the machine the store was keyed to. {changed}. The new observation "
            f"is filed against {hardware.instance_type}, so re-running this command now reads and "
            "writes under the new address with nothing further to do. This run stops instead of "
            "continuing, because it opened its stores under the old address before it had a node: "
            "every degree it resolved and every baseline it printed came from rows taken on the other "
            "machine, and filing this run's measurements beside them -- or its rows under a key it did "
            "not read -- is the one failure nothing downstream can detect.")
    return hardware


def _file_hardware(hardware: Hardware, observed: dict, node_name: str, index,
                   image: str = "") -> None:
    """File the shape, and the driver's total when it is the only measurement of this card.

    Failures here are logged and not raised. The lease is real either way, and a store that
    cannot be written is a lost saving rather than a wrong number -- while aborting a capture
    that has already provisioned a node would turn it into a wasted one.
    """
    from ..store.hwfacts import HardwareFacts, HardwareKey, HardwareStore
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        # The raw link matrix and the driver go into the shape's notes: the digest in the address is
        # one-way, so without them a directory name is unattributable, and the driver version is
        # exactly the kind of thing a reader wants six months later and nothing keys on.
        index.write_shape(hardware.shape(
            observed_at=at, node=node_name,
            notes={"p2p_links": dict(observed.get("p2p_links") or {}),
                   "driver": str(observed.get("driver") or ""),
                   "board_total_gib": observed.get("device_total_board_gib"),
                   "read_by": "capture, nvidia-smi on the leased node"}))
    except Exception as exc:            # noqa: BLE001 -- see the docstring
        log.warning("could not file %s's shape: %s", hardware.instance_type, exc)
    if hardware.total_source != "driver":
        return              # a measured sheet already exists; this run adds nothing to it
    try:
        store = HardwareStore(index.root,
                              HardwareKey(env=hardware.environment(image).device()))
        store.write_key()
        store.write(HardwareFacts(
            gpu_product=hardware.gpu_product,
            total_gib=hardware.require_total_gib(),
            total_source="driver",
            total_uncertainty_gib=hardware.total_uncertainty_gib,
            total_falsifiable=False,
            driver_total_gib=hardware.require_total_gib(),
            driver=str(observed["driver"]), image=str(image),
            gpu_node=node_name, measured_at=at,
            notes={"read_by": "capture, nvidia-smi memory.total minus memory.reserved",
                   "board_total_gib": observed.get("device_total_board_gib"),
                   "reserved_gib": observed.get("device_reserved_gib"),
                   "instance_type": hardware.instance_type}))
    except Exception as exc:            # noqa: BLE001 -- see the docstring
        log.warning("could not file %s's device total: %s", hardware.gpu_product, exc)


class NodeCapture:
    """What the agent needs from the world in order to hold a node. Two verbs."""

    def capture(self, token: str, hardware: Hardware, timeout_s: float,
                index=None, image: str = "") -> NodeLease:
        raise NotImplementedError

    def release(self, lease: NodeLease) -> None:
        raise NotImplementedError


class HelmCapture(NodeCapture):
    """`helm install` a per-run mps-daemon release, then read the node it landed on."""

    def __init__(self, chart_path: str, namespace: str = DEFAULT_NAMESPACE,
                 holder_manifest: str | None = None,
                 kubectl: str = "kubectl", helm: str = "helm"):
        self.chart = str(Path(chart_path).resolve())
        if not Path(self.chart, "Chart.yaml").exists():
            raise CaptureError(f"{self.chart} is not a Helm chart (no Chart.yaml)")
        self.namespace = namespace
        # None means "let the node go on release". Not the default: see the module docstring.
        self.holder_manifest = str(Path(holder_manifest).resolve()) if holder_manifest else None
        self.kubectl = kubectl
        self.helm = helm

    # --- capture -------------------------------------------------------------

    def capture(self, token: str, hardware: Hardware, timeout_s: float = 1800.0,
                index=None, image: str = "") -> NodeLease:
        """Lease a node of `hardware`'s instance type and return it reconciled with what it is.

        `index` is a `store/hwfacts.HardwareIndex`. Given one, this files what it observed -- the
        instance shape, and the driver's `memory.total` when no environment had measured the card --
        so the next run resolves the same machine offline and for free, and the `--gpu-count` that may
        have been needed to get here is needed exactly once. Without one nothing is filed and the
        lease still works; the mode that only wants a node does not have to open a store.

        `image` is the engine image the run will boot, and it is here only because a hardware sheet is
        keyed by it (`store/hwfacts.HardwareKey`) -- an engine build is part of the environment a
        device figure was read in, so everything read on this node lands in one directory.
        """
        release = f"{DAEMON_RELEASE_PREFIX}-{token}"
        self._preflight(release)
        # Before the install, because the install is what removes the holder that carries this.
        # There is no ordering that closes the gap afterwards -- see the module docstring.
        protected = self._protect_held_nodes()

        values = {
            "gpuRequest": hardware.gpu_count,
            "nodeSelector": {
                "karpenter.sh/nodepool": LEASE_NODEPOOL,
                "node.kubernetes.io/instance-type": hardware.instance_type,
            },
        }
        log.info("capturing a %s node: helm install %s (%d GPUs requested)",
                 hardware.instance_type, release, hardware.gpu_count)
        self._install(release, values)

        node_name = ""
        try:
            pod = self._wait_for_daemon(release, timeout_s)
            node_name = pod["spec"]["nodeName"]
            # A node no holder was on -- a fresh one Karpenter just bought, or one this run is the
            # first to touch -- gets its protection here instead. Late by the length of the pod's
            # start-up, which is a risk only for a node that was already drifted, and a node that
            # did not exist a minute ago is not.
            if node_name not in protected and self._protect(node_name):
                protected.append(node_name)
            # Before anything is measured on it, and before the lease is worth anything: is this a
            # node the cluster is already going to take away? Free, and the only moment it can be
            # asked -- the node is not known until the pod lands, and the capture is what removes
            # the holder that was blocking the removal.
            self._refuse_if_disrupted(node_name)
            self._refuse_foreign_dcgm_on(node_name)
            observed = self._observe(release, node_name, pod)
            # Every claim this run arrived with, against the machine it actually got. The device
            # total is checked here too, and that check is the one with no symptom if it is skipped:
            # a wrong denominator does not fail, it hands every engine on the node slightly more
            # memory than the packing priced.
            hardware.verify(gpu_count=observed["gpu_count"],
                            gpu_product=observed["gpu_product"],
                            driver_total_gib=observed["device_total_gib"])
            if observed["instance_type"] != hardware.instance_type:
                raise CaptureError(
                    f"asked for a {hardware.instance_type} and got {node_name}, which the "
                    f"cluster labels {observed['instance_type']!r}. The validity key names the "
                    "instance type, so measuring here would file these records under a machine "
                    "they were not taken on.")
            hardware = reconcile_hardware(hardware, observed, node_name, index, image)
            inventory = self._read_inventory(release, node_name, pod, hardware, timeout_s=180.0)
        except Exception:
            # Never leave a Pending 8-GPU pod behind: it is a standing order for another node.
            self._abandon(release, node_name, protected)
            raise

        lease = NodeLease(
            token=token, release=release, node_name=node_name,
            pod_name=pod["metadata"]["name"], pod_uid=pod["metadata"]["uid"],
            hardware=hardware, driver=observed["driver"], inventory=inventory,
            exporter_url=exporter_service_url(release, self.namespace),
            captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            observed=observed, protected_nodes=tuple(protected))
        log.info("%s", lease.summary())
        return lease

    def _preflight(self, release: str) -> None:
        """Refuse to start if anything already exists under this run's release name.

        Three checks, and the second is the one worth explaining. A
        `<release>-gpu-uuids-<node>` ConfigMap is written at runtime by the daemon's publisher
        sidecar, so it is not part of the Helm release and `helm uninstall` used to orphan it
        (fixed 2026-09-04 with an ownerReference, but a chart old enough to lack that may have
        run here). An orphan is a UUID set for a node nobody holds, and using it deploys
        tenants whose MPS memory limits key to devices this run does not own -- which does not
        error. It silently lifts every envelope.
        """
        try:
            _run([self.helm, "status", release, "-n", self.namespace], timeout_s=60.0)
        except DeployError:
            pass        # the good case: no such release
        else:
            raise CaptureError(
                f"a Helm release named {release!r} already exists in {self.namespace}. It is "
                "either another run's lease or the wreckage of one, and this run will not adopt "
                "it: adopting a lease means measuring on a node somebody else may still be "
                "using. Pick a new run token, or `helm uninstall` it if you know it is dead.")
        out = _run([self.kubectl, "get", "configmap", "-n", self.namespace,
                    "-o", "jsonpath={.items[*].metadata.name}"], timeout_s=60.0)
        stale = [n for n in out.split() if n.startswith(f"{release}-gpu-uuids-")]
        if stale:
            raise CaptureError(
                f"{stale} already exist(s) and no pod of this run published it. A GPU-UUID "
                "ConfigMap is a cache, and the only thing that proves a node is held is the "
                "daemon pod -- so this is treated as an error rather than as data. Delete it "
                "and capture again.")
        self._check_one_dcgm_exporter()

    # --- keeping the node the run is standing on -------------------------------

    def _held_nodes(self) -> list[str]:
        """Every node a holder pod is currently sitting on. Empty is normal, not an error."""
        out = _run([self.kubectl, "get", "pods", "-n", self.namespace, "-l", HOLDER_SELECTOR,
                    "-o", 'jsonpath={range .items[*]}{.spec.nodeName}{"\\n"}{end}'],
                   timeout_s=60.0)
        return sorted({n.strip() for n in out.splitlines() if n.strip()})

    def _protect_held_nodes(self) -> list[str]:
        """Put `DO_NOT_DISRUPT` on every node a holder is on, and say which ones this added.

        Called before the daemon goes in, which is the last moment the holder's protection still
        exists to be copied. A failed read is logged and not raised: the capture is still worth
        doing, and refusing one because a `kubectl get` did not answer would cost a node to save
        a node.
        """
        try:
            nodes = self._held_nodes()
        except DeployError as exc:
            log.warning("could not list holder pods, so no node was protected before the daemon "
                        "preempts them: %s. A drifted node may start going away mid-run.", exc)
            return []
        if not nodes:
            log.info("no holder pod is on a node, so there is no protection to carry over")
        return [n for n in nodes if self._protect(n)]

    def _disrupt_annotation(self, node_name: str) -> str:
        """The node's `DO_NOT_DISRUPT` annotation, or "" -- including when it could not be read."""
        path = "{.metadata.annotations." + DO_NOT_DISRUPT.replace(".", r"\.") + "}"
        try:
            return _run([self.kubectl, "get", "node", node_name, "-o", f"jsonpath={path}"],
                        timeout_s=60.0).strip()
        except DeployError as exc:
            log.warning("could not read %s on %s: %s", DO_NOT_DISRUPT, node_name, exc)
            return ""

    def _protect(self, node_name: str) -> bool:
        """Annotate a node against voluntary disruption. True only if this call added it.

        False when it was already there, and the annotation is left exactly as it is: it may be
        somebody else's, and this run must not take responsibility for removing it.
        """
        if self._disrupt_annotation(node_name) == "true":
            log.info("%s already carries %s, so it stays as it is", node_name, DO_NOT_DISRUPT)
            return False
        try:
            _run([self.kubectl, "annotate", "node", node_name, f"{DO_NOT_DISRUPT}=true",
                  "--overwrite"], timeout_s=60.0)
        except DeployError as exc:
            log.warning("could not put %s on %s: %s. The node is unprotected for the length of "
                        "this lease, so a pending replacement can begin the moment the holder "
                        "goes.", DO_NOT_DISRUPT, node_name, exc)
            return False
        log.info("%s now carries %s for the length of this lease: the holder that was carrying it "
                 "is about to be preempted", node_name, DO_NOT_DISRUPT)
        return True

    def _hand_back_protection(self, protected, timeout_s: float = 60.0) -> None:
        """Give the protection back to the holder, and only where the holder is there to take it.

        The annotation and the holder guard the same thing, and the whole reason for the ordering
        around both callers is that the node is never carrying neither. So this waits briefly for a
        holder to land and leaves the annotation alone on any node where none has: an over-protected
        node is a machine that stays, and an unprotected node with no holder is one that goes.
        """
        if not protected:
            return
        deadline = time.monotonic() + timeout_s
        back: set = set()
        while True:
            try:
                back = set(self._held_nodes())
            except DeployError as exc:
                log.warning("could not list holder pods: %s", exc)
                break
            if set(protected) <= back or time.monotonic() > deadline:
                break
            time.sleep(5.0)
        self._unprotect([n for n in protected if n in back])
        left = [n for n in protected if n not in back]
        if left:
            log.warning("left %s on %s: no holder pod is on it, so that annotation is now the only "
                        "thing keeping the machine. Apply %s and then remove the annotation by "
                        "hand, or leave both in place and the node stays.",
                        DO_NOT_DISRUPT, left,
                        self.holder_manifest or DEFAULT_HOLDER_MANIFEST)

    def _unprotect(self, nodes) -> None:
        """Remove only what this run added, once the holder is back to carrying it."""
        for node_name in nodes:
            try:
                _run([self.kubectl, "annotate", "node", node_name, f"{DO_NOT_DISRUPT}-"],
                     timeout_s=60.0)
            except DeployError as exc:
                log.warning("could not remove %s from %s: %s. Nothing is at risk -- the node is "
                            "over-protected rather than under-protected -- but the autoscaler "
                            "cannot replace it until that annotation goes.",
                            DO_NOT_DISRUPT, node_name, exc)
                continue
            log.info("removed %s from %s; the holder carries it again", DO_NOT_DISRUPT, node_name)

    def _refuse_if_disrupted(self, node_name: str) -> None:
        """Refuse a node the cluster has already decided to remove. Free, and before any spend.

        **The holder is what was stopping it, so a capture is what starts it.**
        `manifests/gpu-node-holder.yaml` carries `karpenter.sh/do-not-disrupt`, and while it sits on
        the node nothing can replace it -- including for a drift the autoscaler has been wanting to
        act on for days. The daemon installed above *preempts the holder*, which removes the only
        blocker, so a run that captures a drifted node is the event that schedules its own node's
        deletion. On 2026-09-10 the claim took a `deletionTimestamp` three seconds after the daemon
        went in; that run finished only because its own pods held the drain open for two hours, the
        instance was terminated four seconds after the tenant releases came out, and the replacement
        could not be acquired in any AZ of the region.

        So this is checked here rather than trusted to timing, and it is a refusal rather than a
        warning: an unprotected drifted node will be taken away mid-run, the measurement it would
        produce is one nobody can finish, and the lease is not cheaper for having been started. The
        remedy is to let the replacement happen and capture again -- which is why the refusal says so
        instead of offering a flag to proceed. Note that the replacement arrives on a **new AMI**, so
        its driver may not be the one the fleet's `validity_key` names, in which case the next capture
        refuses for that reason (correctly) and the stored rows need re-measuring.

        **`Drifted` on a node this run has protected is not that situation**, and refusing it would
        be refusing arithmetic. `capture` puts `DO_NOT_DISRUPT` on the node before the daemon
        preempts the holder, and drift is voluntary disruption: while the annotation is there the
        autoscaler is not waiting for a disruption budget, it is not going to get one. What is left
        is a node the cluster would replace if it could, which is worth one loud line in the log --
        an AMI bump is pending, and taking it means a new driver and a re-keyed store -- and is not
        worth losing a machine over. `Disrupted`, and a claim already carrying a `deletionTimestamp`,
        are refused whatever the annotations say: those are the disruption itself, already under way,
        and no annotation calls one back.

        Three things are deliberately not errors. **No claim resource** -- a cluster with no
        autoscaler of this kind has nothing that could remove a node under a run. **No claim for this
        node** -- a node somebody made by hand is not going to be consolidated away. **A failed
        read** -- refusing a lease because a `kubectl get` did not answer would turn a transient API
        error into a lost node, so it is logged loudly and the run goes on. Each of those says so in
        the log, because "checked and clean" and "could not check" must not read the same.
        """
        try:
            out = _run([self.kubectl, "get", NODE_CLAIM_RESOURCE, "-o", "json"], timeout_s=60.0)
        except DeployError as exc:
            log.warning("could not read %s, so %s was NOT checked for drift and this run may be "
                        "holding a node the cluster is about to replace: %s",
                        NODE_CLAIM_RESOURCE, node_name, exc)
            return
        claims = [c for c in json.loads(out).get("items", [])
                  if c.get("status", {}).get("nodeName") == node_name]
        if not claims:
            log.info("no %s owns %s, so nothing there can take it away mid-run",
                     NODE_CLAIM_RESOURCE, node_name)
            return
        # Set when a claim was marked and the mark was held off rather than absent. Without it the
        # closing line reads "nothing scheduled to remove it" on the very node whose drift was just
        # warned about two lines above -- a log that contradicts itself is worse than a quiet one,
        # because the reassuring half is the one a reader remembers.
        tolerated = False
        for claim in claims:
            name = claim.get("metadata", {}).get("name", "?")
            if claim.get("metadata", {}).get("deletionTimestamp"):
                raise CaptureError(
                    f"{node_name} is already being deleted: its claim {name} has a "
                    f"deletionTimestamp of {claim['metadata']['deletionTimestamp']}. The node this "
                    "run just captured is on its way out, and a measurement taken on it cannot be "
                    "finished. Wait for the replacement to register, confirm the holder is on it, "
                    "and capture again.")
            hit = sorted(c.get("type") for c in claim.get("status", {}).get("conditions", [])
                         if c.get("type") in DISRUPTION_CONDITIONS and c.get("status") == "True")
            if hit:
                why = "; ".join(
                    f"{c.get('type')}: {c.get('reason') or '?'} {c.get('message') or ''}".strip()
                    for c in claim.get("status", {}).get("conditions", [])
                    if c.get("type") in hit)
                if (all(h in TOLERABLE_WHEN_PROTECTED for h in hit)
                        and self._disrupt_annotation(node_name) == "true"):
                    log.warning(
                        "%s is marked %s on its claim %s (%s), and it carries %s so the "
                        "autoscaler cannot act on that while this run holds it. Proceeding. What "
                        "this means beyond the run: the cluster wants to replace this machine, and "
                        "the replacement arrives on a new AMI -- so taking it means a new driver, "
                        "which re-keys every row in the store. Nothing here decides that.",
                        node_name, " and ".join(hit), name, why, DO_NOT_DISRUPT)
                    tolerated = True
                    continue
                raise CaptureError(
                    f"{node_name} is marked {' and '.join(hit)} on its claim {name} ({why}), and "
                    f"it does not carry {DO_NOT_DISRUPT}. The "
                    "holder this run's daemon just preempted was the only thing keeping the node, "
                    "so continuing means measuring on a machine the cluster will remove as soon as "
                    "the drain is not blocked -- and the last time that happened the instance was "
                    "not reacquirable. Let the replacement come up, confirm the holder is on it, "
                    "and capture again. The replacement may carry a different driver than the "
                    "fleet's validity key names, in which case the next capture will say so.")
        if tolerated:
            log.info("%s is held against the mark above for the length of this lease; nothing can "
                     "act on it while %s is on the node", node_name, DO_NOT_DISRUPT)
            return
        log.info("%s is not marked %s on its claim: nothing scheduled to remove it",
                 node_name, "/".join(DISRUPTION_CONDITIONS))

    def _check_one_dcgm_exporter(self) -> None:
        """Refuse to start while some other DCGM exporter is on the cluster. Free, and before spend.

        **A GPU may have exactly one DCGM host engine watching its profiling fields.** The second
        one to ask gets `error watching fields: The third-party Profiling module returned an
        unrecoverable error` and exits, so it crash-loops, so every record of the run it was
        instrumenting loses its `per_gpu` block -- the entire cause-signal half of §6.5. The harness
        degrades on a missing exporter instead of failing, deliberately (`_wait_for_daemon`), which
        is right for an instrument that dies mid-run and exactly wrong as a way to *start* one: the
        run would spend its whole budget measuring latency with the cause signals silently absent.

        Observed 2026-09-14 on this cluster: the `amazon-cloudwatch-observability` EKS addon ships
        its own dcgm-exporter DaemonSet, it landed on the leased g6e, and it had already claimed the
        profiling groups (`Successfully queried DCGM profiling metric groups ... gpu_model="NVIDIA
        L40S"`). Ours restarted 11 times. Disabling the addon's copy and nothing else brought all
        seven `DCGM_FI_PROF_*` fields back on ours within one pod lifetime -- which is also what
        makes this a cause rather than a correlation.

        Checked cluster-wide rather than on the leased node, because there is no leased node yet and
        because the conflict arrives with the node either way: a DaemonSet that tolerates GPU nodes
        schedules onto the machine this call is about to buy, whether or not it is running anywhere
        right now. So DaemonSets count even at zero replicas, and this refuses before the install.

        **Except one that cannot reach the lease's pool.** An exporter that does not tolerate a
        NoSchedule or NoExecute taint of `LEASE_NODEPOOL` cannot be scheduled onto any node of it,
        so it is not counted. That is the GPU Operator's exporter on a cluster that also runs KAI
        (2026-10-08): it tolerates only the KAI pool's taint. A pool whose taints cannot be read
        excuses nothing. What this cannot see -- a pod that got there anyway -- is checked again
        on the node itself once it is known (`_refuse_foreign_dcgm_on`).
        """
        taints = self._pool_taints()
        found = []
        for kind, items in (("daemonset", self._listing("daemonsets")),
                            ("pod", self._listing("pods", "--field-selector",
                                                  "status.phase=Running"))):
            for item in items:
                who, spec = _who(item), _pod_spec(item)
                if DCGM_IMAGE_HINT not in _images(spec) or (
                        kind == "pod" and _ours(who, self.namespace)):
                    continue
                untolerated = [t for t in taints if not _tolerates(spec.get("tolerations"), t)]
                if untolerated:
                    log.info("%s %s is a DCGM exporter, but it cannot reach the %s pool: it does "
                             "not tolerate %s", kind, who, LEASE_NODEPOOL,
                             _taint_text(untolerated[0]))
                    continue
                found.append(f"{kind} {who}")
        if not found:
            return
        raise CaptureError(
            "another DCGM exporter is already on this cluster: " + ", ".join(sorted(found)) +
            ". A GPU allows one DCGM host engine to own its profiling fields, so this run's own "
            "exporter would crash-loop and every record it produced would come back with no "
            "`per_gpu` block -- the run would look successful and be missing half its evidence. "
            "No node has been leased; nothing has been spent. If that is the EKS CloudWatch "
            "addon, turn its exporter off (Container Insights loses its GPU dashboards, nothing "
            "else) and capture again:\n\n" + DCGM_ADDON_REMEDY + "\n\n"
            "Background: USER-GUIDE.md step 0, 'Only one DCGM exporter per GPU'.")

    def _listing(self, kind: str, *extra) -> list:
        out = _run([self.kubectl, "get", kind, "-A", *extra, "-o", "json"], timeout_s=90.0)
        return json.loads(out).get("items", []) if out.strip() else []

    def _pool_taints(self) -> list:
        """The taints that keep pods off `LEASE_NODEPOOL`'s nodes; [] when they cannot be read."""
        try:
            out = _run([self.kubectl, "get", "nodepool", LEASE_NODEPOOL, "-o",
                        "jsonpath={.spec.template.spec.taints}"], timeout_s=60.0)
            taints = json.loads(out) if out.strip() else []
        except (DeployError, ValueError) as exc:
            log.warning("could not read the %s pool's taints (%s), so every DCGM exporter on the "
                        "cluster counts", LEASE_NODEPOOL, exc)
            return []
        return [t for t in taints if t.get("effect") in ("NoSchedule", "NoExecute")]

    def _refuse_foreign_dcgm_on(self, node_name: str) -> None:
        """The pre-flight's question again, on the node this run got: is anyone else's DCGM host
        engine already on it? The pre-flight excused exporters that cannot reach the pool; this is
        what makes that excuse safe."""
        out = _run([self.kubectl, "get", "pods", "-A", "--field-selector",
                    f"spec.nodeName={node_name}", "-o", "jsonpath="
                    '{range .items[*]}{.metadata.namespace}/{.metadata.name}{"\\t"}'
                    '{range .spec.containers[*]}{.image}{" "}{end}{"\\n"}{end}'],
                   timeout_s=90.0)
        found = [who for who, images in _tab_rows(out)
                 if DCGM_IMAGE_HINT in images and not _ours(who, self.namespace)]
        if found:
            raise CaptureError(
                f"another DCGM exporter is on {node_name}, the node this run just leased: "
                f"{', '.join(sorted(found))}. The pre-flight did not count it, because it does not "
                f"tolerate the {LEASE_NODEPOOL} pool's taints, so something placed it here anyway. "
                "Nothing has been measured; the lease is being given up.")

    def _install(self, release: str, values: dict) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            yaml.safe_dump(values, fh, sort_keys=False)
            path = fh.name
        try:
            # No --wait, for the same reason `deploy.py` does not use it, plus one specific to
            # capture: --wait would report "the node is still being provisioned" and "the daemon
            # cannot start on this node" as one timeout, and those need opposite responses.
            _run([self.helm, "install", release, self.chart, "-n", self.namespace,
                  "-f", path, "--wait=false"], timeout_s=300.0)
        finally:
            Path(path).unlink(missing_ok=True)

    def _wait_for_daemon(self, release: str, timeout_s: float) -> dict:
        """Poll until the daemon pod is Ready. Pending is progress here, not a failure.

        Gates on the DAEMON only. The release also ships a dcgm-exporter, and that is an
        instrument: if it never becomes Ready the records lose their `per_gpu` block, which the
        harness degrades on, and that must not cost the run its node.
        """
        t0 = time.monotonic()
        last = ""
        while True:
            pod = self._daemon_pod(release)
            if pod is not None:
                state, why = _classify_daemon(pod)
                if state == "ready":
                    log.info("daemon %s Ready after %.0fs on %s", release,
                             time.monotonic() - t0, pod["spec"].get("nodeName"))
                    return pod
                if state == "fatal":
                    raise CaptureError(
                        f"the daemon pod for {release} cannot run: {why}. The node was not "
                        "captured, and this is not something waiting fixes.")
                last = why
            else:
                last = "no pod yet"
            waited = time.monotonic() - t0
            if waited > timeout_s:
                raise CaptureError(
                    f"no node captured within {timeout_s:.0f}s ({last}). If the pod is still "
                    "Pending, either the cluster cannot provision this instance type right now "
                    "or something else holds the GPUs at a priority this cannot preempt.")
            log.info("waiting for a node (%.0fs): %s", waited, last)
            time.sleep(10.0)

    def _daemon_pod(self, release: str) -> dict | None:
        out = _run([self.kubectl, "get", "pods", "-n", self.namespace, "-l",
                    f"app.kubernetes.io/name=mps-daemon,app.kubernetes.io/instance={release}",
                    "-o", "json"], timeout_s=60.0)
        items = [p for p in json.loads(out).get("items", [])
                 if not p.get("metadata", {}).get("deletionTimestamp")]
        if not items:
            return None
        return max(items, key=lambda p: p.get("metadata", {}).get("creationTimestamp", ""))

    def _observe(self, release: str, node_name: str, pod: dict) -> dict:
        """What the leased node says it is. Two independent sources, both recorded.

        `nvidia-smi` inside the daemon container is the authority on everything the store is keyed by
        (`store/environment.py`): the addressable total -- which is `memory.total` minus
        `memory.reserved`, not `memory.total`; see `smi.py` -- the enforced power limit, and the
        link matrix. **That whole bag is read here and nowhere else**, because it is read on the node
        the measurements are taken on, which is the only place it is a fact rather than a claim. The
        driver version is read with it and recorded as provenance; it names no directory.

        The Karpenter labels are the authority on the instance type and on *which GPU this is* --
        hardware sheets are filed under the label's token (`store/hwfacts.py`), and an `nvidia-smi`
        product name does not reduce to it in general ("Tesla T4" against a label of `t4`). The
        `nvidia-smi` name is recorded beside it as `gpu_product_raw` and used only when there is
        no label. The GPU count is read from both, since they can disagree only if something is
        very wrong.
        """
        out = _run([self.kubectl, "get", "node", node_name, "-o", "json"], timeout_s=60.0)
        node = json.loads(out)
        labels = node.get("metadata", {}).get("labels", {})
        allocatable = node.get("status", {}).get("allocatable", {})

        # Named for what it holds rather than for the tool, because the module `smi` is imported here
        # and read three lines below: one CSV of per-GPU rows, not the reader.
        gpu_csv = _run([self.kubectl, "exec", "-n", self.namespace, pod["metadata"]["name"],
                        "-c", DAEMON_CONTAINER, "--", "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader,nounits"], timeout_s=120.0)
        rows = [r.strip() for r in gpu_csv.splitlines() if r.strip()]
        if not rows:
            raise CaptureError(
                f"nvidia-smi on {node_name} listed no GPUs. The daemon is Ready, so the node "
                "exists and MPS started -- but a node with no visible devices cannot be "
                "measured on, and the cause is upstream of this agent.")
        products = {r.split(",")[0].strip() for r in rows}
        drivers = {r.split(",")[1].strip() for r in rows}
        if len(products) != 1 or len(drivers) != 1:
            raise CaptureError(
                f"{node_name} reports mixed GPUs or drivers ({sorted(products)}, "
                f"{sorted(drivers)}). The solver treats the node's GPUs as interchangeable and "
                "the validity key names one driver, so neither holds here.")

        # `memory.total` in whole MiB, `nounits`. This is the driver's own answer to the one question
        # nothing else on the node can answer -- how much memory a device has -- and it costs nothing
        # extra here because the query was already being made for the driver version. It is read for
        # every GPU and required to agree: the solver treats the node's devices as interchangeable, so
        # two different sizes would make "per GPU" meaningless for every envelope on the node.
        totals = {_mib(r.split(",")[2], "memory.total") for r in rows}
        if len(totals) != 1:
            raise CaptureError(
                f"{node_name} reports GPUs of different sizes ({sorted(totals)} MiB). Every "
                "envelope on this stack is a fraction of one device total, and the solver assumes "
                "the node's GPUs are interchangeable, so there is no honest single figure here.")
        board_total_mib = totals.pop()
        # And then the part of the board no CUDA context can address, subtracted. See `smi.py`: the
        # engine's utilisation multiplies the ADDRESSABLE total, so this subtraction is what makes the
        # filed figure the same quantity the engine is dividing up.
        reserved_mib = smi.reserved_mib(self._exec_prefix(pod), where=node_name)
        # One subtraction, in the units the driver prints, and the GiB figure is that integer divided
        # rather than a second subtraction of its own. Two spellings of one quantity is exactly the
        # defect `smi.py` was split out to prevent, and here it would be worse than a wrong number: the
        # MiB figure names the store directory and the GiB figure sizes the envelopes, so they would
        # disagree about which machine the rows in it describe.
        addressable_mib = int(board_total_mib) - int(reserved_mib)
        if addressable_mib < 1:
            raise CaptureError(
                f"{node_name} reports memory.total {board_total_mib} MiB and reserves "
                f"{reserved_mib} MiB of it, leaving {addressable_mib} MiB a context can address. "
                "Nothing can be served on that, and the two figures cannot both be right.")
        device_total_gib = addressable_mib / 1024.0
        power_limit_w = self._power_limit_w(release, node_name, pod)
        links = self._topology(release, node_name, pod)
        # Digested here so that a matrix that cannot be digested fails while the node is in hand,
        # rather than at the moment a key is built from a filed shape hours later.
        p2p_topology = topology_digest(links)

        gpu_count = len(rows)
        label_count = labels.get("karpenter.k8s.aws/instance-gpu-count")
        if label_count is not None and int(label_count) != gpu_count:
            raise CaptureError(
                f"{node_name} is labelled as having {label_count} GPU(s) and nvidia-smi lists "
                f"{gpu_count}. A daemon that can see fewer devices than the node has creates "
                "MPS server contexts whose device numbering tenants cannot address.")
        raw_product = products.pop()
        label_product = labels.get("karpenter.k8s.aws/instance-gpu-name") or ""
        return {
            "node_name": node_name,
            "instance_type": labels.get("node.kubernetes.io/instance-type", ""),
            "gpu_count": gpu_count,
            "gpu_product": normalize_gpu(label_product or raw_product),
            "device_total_gib": device_total_gib,
            # The three fields that, with the product and the count above, ARE the environment the
            # store is keyed by. In the driver's own units, and integers, so the key needs no
            # tolerance: equality at the instrument's quantum is the comparison.
            "addressable_mib": addressable_mib,
            "power_limit_w": power_limit_w,
            "p2p_topology": p2p_topology,
            # The matrix behind that digest. A digest alone would name a directory nobody can read
            # back, so the links themselves are filed on the instance shape.
            "p2p_links": {f"{a}-{b}": link for (a, b), link in sorted(links.items())},
            # Both halves of the subtraction, kept because the difference between them is the whole
            # finding: a reader who only sees the result cannot tell a card from a card with a large
            # reserved region, and the driver's own board figure is what every other tool prints.
            "device_total_board_gib": board_total_mib / 1024.0,
            "device_reserved_gib": reserved_mib / 1024.0,
            # Provenance, and provenance only. It used to be half of every key in the store; what it
            # was a proxy for is the three fields above, measured (`store/environment.py`).
            "driver": drivers.pop(),
            "gpu_product_label": label_product,
            "gpu_product_raw": raw_product,
            "allocatable_gpus": allocatable.get("nvidia.com/gpu"),
            "capacity_type": labels.get("karpenter.sh/capacity-type", ""),
            "zone": labels.get("topology.kubernetes.io/zone", ""),
            "kubelet": node.get("status", {}).get("nodeInfo", {}).get("kubeletVersion", ""),
            "os_image": node.get("status", {}).get("nodeInfo", {}).get("osImage", ""),
            "daemon_release": release,
        }

    def _exec_prefix(self, pod: dict) -> list:
        """The argv that gets a command into the container that can see this node's GPUs.

        One place, because three readers use it now -- the reserve, the power limit and the link
        matrix -- and a second spelling of it would be a second container name to get wrong.
        """
        return [self.kubectl, "exec", "-n", self.namespace, pod["metadata"]["name"],
                "-c", DAEMON_CONTAINER, "--"]

    def _power_limit_w(self, release: str, node_name: str, pod: dict) -> int:
        """The enforced power limit, whole watts, required to agree across the node's devices.

        Asked in its own query rather than appended to the one above, for `smi.RESERVED_QUERY`'s
        reason: a driver too old to know a field fails the whole query it appears in, and bundling
        would turn an unknown power limit into "this node has no GPUs".

        **Not defaulted and not degraded**, unlike the reserve. A missing reserve costs accuracy in a
        known direction and the run can still measure; a missing power limit would be *fabricated* into
        the address every row of this run is filed under, which is the failure the whole key exists to
        prevent. This is the field that catches the capped or throttled card that every other one sails
        past, so a run that cannot read it has not established the machine it is on.

        `enforced.power.limit` rather than `power.limit`: the enforced figure is what the board will
        actually hold to, and it is the lower of the requested limit and any cap the driver is applying.
        """
        out = _run([*self._exec_prefix(pod), "nvidia-smi",
                    "--query-gpu=enforced.power.limit", smi.SMI_FORMAT], timeout_s=120.0)
        watts = set()
        for row in out.splitlines():
            text = row.strip()
            if not text:
                continue
            try:
                watts.add(int(round(float(text))))
            except ValueError:
                raise CaptureError(
                    f"nvidia-smi on {node_name} reported enforced.power.limit as {text!r}. `[N/A]` is "
                    "what that field looks like on a card that will not state it, and it is a term in "
                    "the address every measurement from this node is filed under -- so there is "
                    "nothing to fall back on: a guessed limit would file rows taken on a capped card "
                    "under a key claiming an uncapped one.") from None
        if not watts:
            raise CaptureError(
                f"nvidia-smi on {node_name} reported no power limit for any device, and it is part of "
                "the environment this run's records are keyed by.")
        if len(watts) != 1:
            raise CaptureError(
                f"the GPUs of {node_name} are held to different power limits ({sorted(watts)} W). The "
                "solver treats the node's devices as interchangeable and a latency number taken on the "
                "capped one is not comparable with one taken on the rest, so there is no single figure "
                "to key this node by.")
        # Rounded to the watt, which is the unit a limit is set in (`nvidia-smi -pl` takes whole
        # watts); the driver prints it as `350.00`. The raw string is not kept, because a limit that
        # differed from an integer by centiwatts would be an instrument artefact rather than a
        # different machine, and hashing it would re-key the store for one.
        return watts.pop()

    def _topology(self, release: str, node_name: str, pod: dict) -> dict:
        """The pairwise link matrix, parsed. Refused rather than defaulted, for `_power_limit_w`'s
        reason: a node whose links are unknown is a node whose collectives are unknown.

        A single-GPU node has no pairs and `topology_digest` refuses an empty set. That is correct
        rather than an oversight -- this stack co-locates tenants on a multi-GPU node, and every mode
        that reaches here has just requested every GPU on one -- but it does mean a one-card machine
        needs a decision about what its topology *is* before it can be captured, and inventing one here
        would make that decision silently.
        """
        out = _run([*self._exec_prefix(pod), "nvidia-smi", "topo", "-m"], timeout_s=120.0)
        links = parse_topology(out)
        if not links:
            raise CaptureError(
                f"could not read a GPU link matrix from `nvidia-smi topo -m` on {node_name}. It is "
                "part of the environment this run's records are keyed by -- NVLink against PCIe "
                "against a hop between NUMA nodes is most of what a multi-shard latency number "
                f"depends on -- so it is not something to proceed without. What it printed:\n{out}")
        return links

    def _read_inventory(self, release: str, node_name: str, pod: dict, hardware: Hardware,
                        timeout_s: float) -> NodeInventory:
        """The GPU ordinal -> UUID map, and proof that this run's pod is what published it.

        Polled rather than read once: the publisher runs as a second container in the daemon
        pod and has no readiness probe of its own, so the pod can be Ready a moment before the
        ConfigMap exists.

        The `ownerReference` check is the whole point of reading it here instead of trusting
        `_preflight`. `_preflight` establishes that no ConfigMap existed before the install;
        this establishes that the one now present belongs to the pod whose Running state *is*
        this run's lease. Anything else is a cache, and the failure mode of a wrong UUID is
        an MPS memory limit that references a device the tenant does not have -- silently
        ignored, so the tenant runs with no envelope at all and the record looks healthy.
        """
        from . import fleetfile          # local: fleetfile imports render, render imports none
        deadline = time.monotonic() + timeout_s
        cm_name = fleetfile.node_name_to_configmap(node_name, release)
        while True:
            try:
                inventory = fleetfile.read_node_inventory(
                    node_name, gpu_ids=hardware.gpu_ids,
                    device_total_gib=hardware.require_total_gib("the node inventory"),
                    namespace=self.namespace, kubectl=self.kubectl, release=release)
                break
            except (DeployError, fleetfile.FleetFileError) as exc:
                if time.monotonic() > deadline:
                    raise CaptureError(
                        f"{cm_name} never became readable ({exc}). The daemon holds the node, "
                        "but without the ordinal->UUID map no tenant can be given an MPS "
                        "envelope -- an ordinal used as a memory-limit key is ignored without "
                        "warning.") from None
                time.sleep(5.0)
        owners = self._owner_uids(cm_name)
        if pod["metadata"]["uid"] not in owners:
            raise CaptureError(
                f"{cm_name} is not owned by this run's daemon pod "
                f"{pod['metadata']['name']} (uid {pod['metadata']['uid']}); its owners are "
                f"{sorted(owners) or 'nobody'}. Refusing to read UUIDs this run did not watch "
                "being published: they may describe a node this run does not hold, and a wrong "
                "UUID lifts an MPS envelope silently rather than failing.")
        return inventory

    def _owner_uids(self, cm_name: str) -> set:
        out = _run([self.kubectl, "get", "configmap", cm_name, "-n", self.namespace,
                    "-o", "jsonpath={.metadata.ownerReferences[*].uid}"], timeout_s=60.0)
        return set(out.split())

    # --- release -------------------------------------------------------------

    def release(self, lease: NodeLease) -> None:
        """Give the node up, and hand it to the holder so its identity survives.

        The order is uninstall-then-apply and it is not interchangeable. The holder requests
        every GPU on the node, so applying it while the daemon still holds them leaves it
        Pending -- and a Pending 8-GPU pod under Karpenter provisions a second g6e.48xlarge.
        Which is why the holder is applied only after the daemon pod is *observed gone*, and
        why failing to observe that is a loud error rather than a best-effort apply: while the
        daemon is still there the node is still held, so nothing about the node's identity is
        at risk and there is nothing for the holder to protect.
        """
        log.info("releasing lease %s on %s", lease.release, lease.node_name)
        _run([self.helm, "uninstall", lease.release, "-n", self.namespace,
              "--wait", "--ignore-not-found"], timeout_s=600.0)
        if not self._daemon_gone(lease.release, timeout_s=300.0):
            raise CaptureError(
                f"uninstalled {lease.release} but its daemon pod is still on "
                f"{lease.node_name}. Not applying the holder: it requests every GPU on the "
                "node, so it would sit Pending behind the daemon and buy a second node. The "
                "node is still held meanwhile, so its UUIDs are not at risk -- but remove that "
                f"pod and apply {self.holder_manifest or DEFAULT_HOLDER_MANIFEST} by hand, "
                "inside ~600s, or the node goes.")
        if self.holder_manifest is None:
            # Releasing for real: the node is meant to go, so the protection has to go with it.
            # Left on, it would keep a node nobody is holding and nobody wants.
            self._unprotect(lease.protected_nodes)
            log.warning("no holder manifest configured, so %s is now unheld. Karpenter "
                        "reclaims it in ~600s and the replacement has different GPU UUIDs.",
                        lease.node_name)
            return
        # Idempotent: the holder is a fixed name, so a re-apply of one already Running is a
        # no-op rather than a second 8-GPU request.
        _run([self.kubectl, "apply", "-f", self.holder_manifest], timeout_s=120.0)
        log.info("applied %s; %s is held at priority -1000, so the next run's daemon preempts "
                 "it and inherits the node with its UUIDs intact",
                 Path(self.holder_manifest).name, lease.node_name)
        # Only now, and in this order: the holder pod carries `DO_NOT_DISRUPT` itself, so the node
        # is never unprotected between the two. Removed rather than left, because an annotation that
        # outlives its lease is state nobody chose -- the next capture puts it back.
        self._hand_back_protection(lease.protected_nodes)

    def _daemon_gone(self, release: str, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            if self._daemon_pod(release) is None:
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(5.0)

    def _abandon(self, release: str, node_name: str, protected=()) -> None:
        """Undo a capture that did not complete, without buying a node in the process.

        The holder is applied only if a node was actually learned. Applying it after a capture
        that never scheduled would request 8 GPUs nothing is holding, which provisions a node
        for a run that has already failed -- the holder exists to PRESERVE a node identity, and
        there is no identity to preserve if the pod never landed.

        `protected` is undone last, after the holder is back, for the same reason `release` does it
        in that order: between the two the node would be carrying nothing. A capture that failed
        *because* the node is drifted and unprotected leaves nothing to undo, which is why this is
        driven by what was actually annotated rather than by what was attempted.
        """
        try:
            _run([self.helm, "uninstall", release, "-n", self.namespace,
                  "--wait", "--ignore-not-found"], timeout_s=600.0)
        except DeployError as exc:
            log.error("could not uninstall %s after a failed capture: %s. Its pod may still be "
                      "Pending, which under Karpenter is a standing order for a node -- check "
                      "and remove it by hand.", release, exc)
            return
        if node_name and self.holder_manifest:
            log.warning("capture failed after landing on %s; applying the holder so the node "
                        "and its GPU UUIDs survive", node_name)
            try:
                _run([self.kubectl, "apply", "-f", self.holder_manifest], timeout_s=120.0)
            except DeployError as exc:
                log.error("could not apply the holder: %s. %s is unheld and Karpenter reclaims "
                          "it in ~600s.", exc, node_name)
                return          # nothing carries the protection now, so it stays where it is
        self._hand_back_protection(protected)


def _tab_rows(out: str) -> list:
    """`[(name, images), ...]` from a two-column `kubectl -o jsonpath` listing.

    Tolerant of a blank listing and of a row with no containers at all, both of which a `jsonpath`
    over an empty `.items` produces -- a check that raises on "there is nothing here" would refuse
    every run on a cluster that is fine.
    """
    rows = []
    for line in out.splitlines():
        who, _, images = line.partition("\t")
        if who.strip():
            rows.append((who.strip(), images))
    return rows


def _who(item: dict) -> str:
    meta = item.get("metadata", {})
    return f"{meta.get('namespace', '')}/{meta.get('name', '')}"


def _pod_spec(item: dict) -> dict:
    """A pod's spec, or a DaemonSet's pod template's."""
    spec = item.get("spec", {})
    return spec.get("template", {}).get("spec", spec) if "template" in spec else spec


def _images(spec: dict) -> str:
    return " ".join(c.get("image", "") for c in spec.get("containers", []))


def _tolerates(tolerations, taint: dict) -> bool:
    """Kubernetes' own rule: an empty key with `Exists` tolerates everything; otherwise the key must
    match, and the value too unless the operator is `Exists`; an empty effect matches any."""
    for tol in tolerations or []:
        if tol.get("effect") and tol["effect"] != taint.get("effect"):
            continue
        if tol.get("operator") == "Exists":
            if not tol.get("key") or tol["key"] == taint.get("key"):
                return True
        elif tol.get("key") == taint.get("key") and tol.get("value", "") == taint.get("value", ""):
            return True
    return False


def _taint_text(taint: dict) -> str:
    value = f"={taint['value']}" if taint.get("value") else ""
    return f"{taint.get('key')}{value}:{taint.get('effect')}"


def _ours(who: str, namespace: str) -> bool:
    """Is `namespace/pod` an exporter belonging to one of this project's own leases?

    Another lease's exporter is not a configuration mistake and is not what this check is for: it
    is a run holding a node, which the GPU request settles by itself. Only a foreign exporter --
    one that came with the cluster and will follow the lease onto whatever node it lands on -- is
    something a human has to go and switch off.
    """
    ns, _, name = who.partition("/")
    return ns == namespace and name.startswith(f"{DAEMON_RELEASE_PREFIX}-")


def _classify_daemon(pod: dict) -> tuple:
    """(state, why) for the daemon pod, where state is ready / waiting / fatal.

    Pending is `waiting`, not `fatal`, and that is the difference from `deploy._classify`.
    There, a Pending pod means the node it was pinned to is gone. Here it means Karpenter has
    not finished buying one yet, which is the normal path on a cold cluster.
    """
    status = pod.get("status", {})
    for c in status.get("containerStatuses", []) or []:
        waiting = (c.get("state") or {}).get("waiting") or {}
        if waiting.get("reason") in ("CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull",
                                     "CreateContainerConfigError", "InvalidImageName"):
            return "fatal", f"{c.get('name')} {waiting['reason']}"
        term = (c.get("state") or {}).get("terminated") or {}
        if term and term.get("exitCode", 0) != 0:
            return "fatal", f"{c.get('name')} exited {term.get('exitCode')}"
    if status.get("phase") == "Failed":
        return "fatal", f"pod Failed: {status.get('reason')}"
    for cond in status.get("conditions", []) or []:
        if cond.get("type") == "Ready" and cond.get("status") == "True":
            return "ready", "Ready"
    if not pod.get("spec", {}).get("nodeName"):
        return "waiting", "Pending: waiting for a node to be provisioned"
    return "waiting", f"phase={status.get('phase')}, not Ready yet"


class FakeCapture(NodeCapture):
    """An in-memory cluster that hands out one scripted lease. Records what it was asked.

    Exists so the whole loop -- capture, deploy, gate, measure, record, release -- runs offline.
    A capture path testable only against a live g6e is one nobody can check before it spends
    money on a node, and it is the step that decides which machine every subsequent number in
    the run describes.
    """

    def __init__(self, inventory: NodeInventory, hardware: Hardware,
                 driver: str = "580.178.04", namespace: str = DEFAULT_NAMESPACE,
                 fail_with: str | None = None, power_limit_w: int = 350,
                 p2p_link: str = "NODE"):
        self.inventory = inventory
        self.hardware = hardware
        self.driver = driver
        # What this scripted node reports as its enforced power limit and the link between every pair
        # of its devices -- the two halves of the environment that are not derivable from the
        # inventory. Defaulted to the g6e.48xlarge this stack measured on so an offline test describes
        # a machine that exists, and settable so a test can move the environment and watch the store
        # re-key.
        self.power_limit_w = int(power_limit_w)
        self.p2p_link = str(p2p_link)
        self.namespace = namespace
        self.fail_with = fail_with
        self.captured: list = []
        self.released: list = []

    def capture(self, token: str, hardware: Hardware, timeout_s: float = 1800.0,
                index=None, image: str = "") -> NodeLease:
        release = f"{DAEMON_RELEASE_PREFIX}-{token}"
        # The scripted node's own device size, which on a real one is the driver's ADDRESSABLE total
        # (`smi.py`). Taken from the inventory rather than from `hardware` so that the
        # verify-and-reconcile path below is exercised against a figure that did not come from the
        # thing being checked. `device_reserved_gib` is zero here and the board total equals the
        # addressable one: a scripted cluster has no driver to reserve anything, and inventing a
        # reserve would make every fake lease disagree with its own inventory.
        gpu_ids = list(self.inventory.gpu_ids)
        observed = {"node_name": self.inventory.node_name,
                    "instance_type": hardware.instance_type,
                    "gpu_count": len(gpu_ids),
                    "gpu_product": self.hardware.gpu_product,
                    "device_total_gib": self.inventory.device_total_gib,
                    "device_total_board_gib": self.inventory.device_total_gib,
                    "device_reserved_gib": 0.0,
                    # The device facts a real `_observe` reads off the node. Derived from the
                    # inventory's own total rather than stated, so a scripted lease cannot claim to
                    # address a different amount of memory than it hands out; the power limit and the
                    # link type are constructor arguments because a test that varies the environment
                    # has to be able to vary something.
                    "addressable_mib": int(round(self.inventory.device_total_gib * 1024.0)),
                    "power_limit_w": int(self.power_limit_w),
                    "p2p_topology": topology_digest({(a, b): self.p2p_link
                                                     for a in gpu_ids for b in gpu_ids if a != b}),
                    "p2p_links": {f"{a}-{b}": self.p2p_link
                                  for a in gpu_ids for b in gpu_ids if a < b},
                    "driver": self.driver, "daemon_release": release, "fake": True}
        if self.fail_with:
            raise CaptureError(self.fail_with)
        hardware.verify(gpu_count=observed["gpu_count"], gpu_product=observed["gpu_product"],
                        driver_total_gib=observed["device_total_gib"])
        hardware = reconcile_hardware(hardware, observed, self.inventory.node_name, index, image)
        lease = NodeLease(
            token=token, release=release, node_name=self.inventory.node_name,
            pod_name=f"{release}-mps-daemon-fake", pod_uid=f"uid-{token}",
            hardware=hardware, driver=self.driver, inventory=self.inventory,
            exporter_url=exporter_service_url(release, self.namespace),
            captured_at="1970-01-01T00:00:00Z", observed=observed)
        self.captured.append(lease)
        return lease

    def release(self, lease: NodeLease) -> None:
        self.released.append(lease.release)
