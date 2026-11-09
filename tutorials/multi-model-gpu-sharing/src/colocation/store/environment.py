"""The environment a stored number is valid in, as things the machine says about itself.

Every key in this store used to name the environment with two hand-written strings: a driver version
and a `metrics_version`, both typed into the fleet file by whoever wrote it. That was wrong in a way
worth writing down, because the fix is the whole shape of this module.

**A driver version is a proxy, and it is wrong in both directions.** It is over-sensitive: a bump that
changes nothing measurable mints a new directory, and an empty directory is indistinguishable from a
fresh start, so the agent loses its history with nothing anywhere reporting it. It is also
under-sensitive: a power cap, a clock throttle, a changed reserved region or a different sharing mode
all move the numbers while the string holds still. And it was asked of a *fleet author*, who cannot
know it -- it is a property of a node nobody has leased yet.

**So the environment is named by what the device reports about itself**, and every field below is
discrete, so equality is the comparison and no tolerance is needed anywhere:

    gpu_product      what card. From the Karpenter label, or `nvidia-smi` when there is none.
    gpu_count        how many of them the node presents.
    addressable_mib  `memory.total` minus `memory.reserved`, in whole MiB -- the only unit the
                     driver reports it in, which is what makes hashing it stable. This is the
                     quantity an engine's utilisation is a fraction of.
    power_limit_w    the enforced limit, whole watts. This is the field that catches the throttled
                     or capped card that a driver string sails straight past.
    p2p_topology     a digest of the pairwise link-type matrix (`nvidia-smi topo -m`). NVLink
                     against PCIe against a hop over the SMP interconnect is most of what a
                     multi-shard latency number depends on, and it is what makes naming the
                     instance type unnecessary: two SKUs presenting the same devices with the same
                     links have the same answer, and today they pay for two sets of measurements.

Two fields are not observations of the node but of what we chose to run on it, and they belong here
for the same reason:

    engine_image     the serving container. The CUDA runtime and the engine version are inside it.
    sharing_mode     how the GPUs are shared. Every co-location number in this store was taken
                     under MPS, and `compute_mode` does NOT record that -- a node running MPS
                     reports `Default` -- so nothing else would notice a run that was not.

    metrics_version  which definitions of ITL, TTFT and delivered fraction are in force. A constant
                     in code, next to nothing a fleet author ever sees, bumped deliberately when a
                     definition changes. It is NOT a digest of the source: a digest moves when a
                     comment moves, and that discards the store for an edit that changed no number.

**The driver version is still recorded, on every row, as provenance.** It attributes a number six
months later and lets a report say that two rows span two drivers. It addresses nothing, so a bump
costs whatever it actually changed -- which the fields above detect if it changed anything at all --
rather than the entire history.

**The instance type is also recorded rather than addressed**, for the reason `hwfacts.py` already
gives about the device key: it would spend a measurement per SKU rediscovering the same card.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# How the GPUs are shared while a fleet is served. One value today, because the deployer installs an
# MPS daemon and nothing else -- but it is in the key rather than assumed, so the first run that
# shares a card another way cannot land in the same directory as these.
SHARING_MODE_MPS = "mps"

# Which definitions of ITL, TTFT and delivered fraction are in force (design §8.4). **Bump this when
# one of those definitions changes** -- a percentile estimator, what counts as delivered, whether
# queueing time is inside the inter-token interval. It lives here rather than in a fleet file because
# it is a property of this code, and nobody writing a fleet could answer it.
METRICS_VERSION = "v1"

# Hash prefix length for a topology digest. Same length as a key hash, for the same reason: long
# enough that a collision is not a practical concern, short enough to read in a directory listing.
TOPOLOGY_HASH_CHARS = 12


def topology_digest(links: dict[tuple[int, int], str]) -> str:
    """A digest of the pairwise link-type matrix, canonicalized so one topology has one spelling.

    `links` maps an unordered device pair to the link type between them (`NODE`, `SYS`, `NV4`, ...).
    Both orderings of a pair are accepted and the low index is written first, because `nvidia-smi
    topo -m` prints the matrix twice and a digest that depended on which half was read would make one
    node hash two ways.

    Self-pairs are dropped rather than refused: `X` is not a link, and including it would make the
    digest depend on the device count, which is a field of its own.
    """
    canon = {}
    for (a, b), link in links.items():
        a, b = int(a), int(b)
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        token = str(link).strip().upper()
        if not token:
            raise ValueError(f"the link between GPU{lo} and GPU{hi} is blank; a topology with an "
                             "unknown link is half a topology, and hashing it would name an "
                             "environment nobody observed")
        prior = canon.get((lo, hi))
        if prior is not None and prior != token:
            raise ValueError(
                f"GPU{lo}<->GPU{hi} reads {prior!r} one way and {token!r} the other. The matrix is "
                "symmetric by construction, so this is a parse error rather than a machine.")
        canon[(lo, hi)] = token
    if not canon:
        raise ValueError("no links: a single-GPU node has no topology to digest, and an empty "
                         "digest would read as 'the same links' on every machine")
    blob = ";".join(f"{a}-{b}={canon[(a, b)]}" for a, b in sorted(canon))
    return hashlib.sha256(blob.encode()).hexdigest()[:TOPOLOGY_HASH_CHARS]


def validate_device_facts(gpu_product: str, addressable_mib: int, power_limit_w: int,
                         p2p_topology: str) -> None:
    """The four facts a node reports about its GPUs, checked once for everything that carries them.

    A function rather than a method because `hwfacts.InstanceShape` holds the same four fields -- it is
    what the key is computed *from* -- and two copies of these checks would eventually let a shape be
    filed that the key derived from it refuses, which reads as a node that cannot be planned against
    for no stated reason.

    Validated rather than normalized, deliberately. Normalizing would let two spellings of one machine
    both be accepted and quietly agree -- which is fine -- but it would also hide the case where a
    caller hands over a *raw* `nvidia-smi` name for one run and a Karpenter label for the next, and
    those do not reduce to each other in general ("Tesla T4" against `t4`). A refusal makes that a
    visible bug once instead of two directories forever.
    """
    product = str(gpu_product)
    if not product.strip():
        raise ValueError("gpu_product is required: an environment that does not say which card it is "
                         "cannot be the address of a memory measurement")
    if product != product.strip().lower() or " " in product:
        raise ValueError(
            f"gpu_product {product!r} is not in the store's key form. Pass it through "
            "`agent.hardware.normalize_gpu` first: two spellings of one card are two directories, "
            "and every row in each of them is individually valid, so nothing would report the split.")
    if int(addressable_mib) < 1:
        raise ValueError(
            f"addressable_mib must be >= 1; got {addressable_mib!r}. It is `memory.total` minus "
            "`memory.reserved` in whole MiB, read off the node -- there is no default, because a card "
            "assumed smaller than it is refuses fleets that fit and one assumed larger "
            "over-subscribes every GPU on the node.")
    if int(power_limit_w) < 1:
        raise ValueError(
            f"power_limit_w must be >= 1; got {power_limit_w!r}. It is the enforced limit in whole "
            "watts, and it is here because it is the field that catches a capped or throttled card "
            "that every other one sails straight past.")
    if not str(p2p_topology).strip():
        raise ValueError("p2p_topology is required: it is a digest of `nvidia-smi topo -m`, and "
                         "without it two nodes presenting the same cards over different links are "
                         "one address")


@dataclass(frozen=True)
class DeviceEnvironment:
    """The part of an environment that conditions a measurement of one *device*.

    Its own type rather than a projection of a dict, because it is what `hwfacts.HardwareKey` is
    written and read under, and a key whose stored block could be parsed back as a full
    `Environment` would silently invent the two fields it leaves out.

    `gpu_count` and `metrics_version` are the omissions and both are deliberate: a card's memory law
    is valid for every model that will ever run on that device, so it does not expire when the box
    holds a different number of them, nor when a latency percentile is redefined.
    """

    gpu_product: str
    addressable_mib: int
    power_limit_w: int
    p2p_topology: str
    engine_image: str
    sharing_mode: str = SHARING_MODE_MPS

    def __post_init__(self) -> None:
        validate_device_facts(self.gpu_product, self.addressable_mib, self.power_limit_w,
                             self.p2p_topology)
        for name in ("engine_image", "sharing_mode"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")

    def canonical(self) -> dict:
        return {
            "gpu_product": str(self.gpu_product),
            "addressable_mib": int(self.addressable_mib),
            "power_limit_w": int(self.power_limit_w),
            "p2p_topology": str(self.p2p_topology),
            "engine_image": str(self.engine_image),
            "sharing_mode": str(self.sharing_mode),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DeviceEnvironment:
        missing = [f for f in ("gpu_product", "addressable_mib", "power_limit_w", "p2p_topology",
                               "engine_image", "sharing_mode") if d.get(f) in (None, "")]
        if missing:
            raise ValueError(
                f"this device environment is missing {', '.join(missing)}. It was written before "
                "those fields existed, so what the sheet under it was measured on is not recorded, "
                "and there is no honest way to fill them in after the fact.")
        return cls(gpu_product=str(d["gpu_product"]), addressable_mib=int(d["addressable_mib"]),
                   power_limit_w=int(d["power_limit_w"]), p2p_topology=str(d["p2p_topology"]),
                   engine_image=str(d["engine_image"]), sharing_mode=str(d["sharing_mode"]))


@dataclass(frozen=True)
class Environment:
    """What a stored number is conditioned on, other than the workload that produced it.

    Frozen, like every key in this store: an environment that can be mutated after rows have been
    filed under its hash is an environment that can orphan them.

    Nothing here is defaulted from a guess. Every field arrives from a node observation
    (`agent/capture.py`), from the fleet file's own `image:`, or from a constant in this module --
    and a field that cannot be obtained is a refusal rather than a fallback, because a fabricated
    field would file rows under a machine nobody measured.
    """

    gpu_product: str
    gpu_count: int
    addressable_mib: int
    power_limit_w: int
    p2p_topology: str
    engine_image: str
    sharing_mode: str = SHARING_MODE_MPS
    metrics_version: str = METRICS_VERSION

    def __post_init__(self) -> None:
        # The six device fields are validated by constructing the projection rather than by a second
        # copy of the same checks, so the two types cannot drift into disagreeing about what a legal
        # environment is -- which would let a profile row be filed under a machine its own device
        # sheet refuses.
        self.device()
        if int(self.gpu_count) < 1:
            raise ValueError(f"gpu_count must be >= 1; got {self.gpu_count!r}")
        if not str(self.metrics_version).strip():
            raise ValueError("metrics_version is required")

    def device(self) -> DeviceEnvironment:
        """The projection a device measurement is conditioned on. See `device_canonical`."""
        return DeviceEnvironment(
            gpu_product=self.gpu_product, addressable_mib=int(self.addressable_mib),
            power_limit_w=int(self.power_limit_w), p2p_topology=self.p2p_topology,
            engine_image=self.engine_image, sharing_mode=self.sharing_mode)

    def device_canonical(self) -> dict:
        """The part that conditions a measurement of one *device*, for `hwfacts.HardwareKey`.

        `gpu_count` and `metrics_version` are left out, and both omissions are the same argument the
        hardware key has always made: a card's memory law is valid for every model that will ever run
        on that device, so keying it to how many cards the box has would re-measure the same card per
        SKU, and keying it to the metric definitions would expire it when a latency percentile
        changed. What conditions it is the card, what the card is allowed to draw, the links it
        speaks over, the runtime that opens a context on it, and how it is shared.
        """
        return self.device().canonical()

    def canonical(self) -> dict:
        """Every field, which is what a profile row and an observation are both conditioned on."""
        return {**self.device_canonical(),
                "gpu_count": int(self.gpu_count),
                "metrics_version": str(self.metrics_version)}

    @property
    def hash(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:TOPOLOGY_HASH_CHARS]

    @classmethod
    def from_dict(cls, d: dict) -> Environment:
        """Rebuild from an `environment` block in a `key.json` or an instance observation.

        Every field is required. A missing one is refused rather than defaulted, for the reason the
        three `ValidityKey.from_dict` refusals give: a guessed field produces a hash that names a
        directory whose rows are not what it claims.
        """
        missing = [f for f in ("gpu_product", "gpu_count", "addressable_mib", "power_limit_w",
                              "p2p_topology", "engine_image", "sharing_mode", "metrics_version")
                   if d.get(f) in (None, "")]
        if missing:
            raise ValueError(
                f"this environment block is missing {', '.join(missing)}. It was written before "
                "those fields existed, so what the rows under it were conditioned on is not "
                "recorded, and there is no honest way to fill them in after the fact.")
        return cls(gpu_product=str(d["gpu_product"]), gpu_count=int(d["gpu_count"]),
                   addressable_mib=int(d["addressable_mib"]),
                   power_limit_w=int(d["power_limit_w"]),
                   p2p_topology=str(d["p2p_topology"]), engine_image=str(d["engine_image"]),
                   sharing_mode=str(d["sharing_mode"]),
                   metrics_version=str(d["metrics_version"]))
