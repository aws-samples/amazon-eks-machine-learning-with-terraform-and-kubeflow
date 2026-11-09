"""Hardware fact sheets: what a GPU is, measured rather than tabulated.

A deleted `facts.py` did this for models, and its argument transfers unchanged even though the sheets
it filed did not survive it (impl §14.4). What this module replaces is smaller and worse than they
were: not a person reading `config.json`, but a dict in `agent/hardware.py` mapping a GPU product to
two capacity figures, one of which had to be obtained by running a one-off feasibility probe and
typing the result into source. That table had exactly one entry. Any GPU it did not name was refused
until somebody edited a source file, which is the definition of a solution that does not generalize -- and worse, of the two numbers in it the planner divided by the one the engine
does not use, so every tenant was deployed with slightly more memory than its plan priced.

**What is in here and where each number comes from.**

    total_gib        the number the engine multiplies `--gpu-memory-utilization` by. MEASURED, as
                     the driver's own `memory.total` minus `memory.reserved` -- what a CUDA context
                     can address. Sheets already on EFS may carry a `slope` source instead, fitted
                     from an engine's own KV report by the deleted `solver/devicelaw.py` (impl
                     §14.4); they are still read, and nothing fits one now. No default, ever: a card
                     assumed smaller than it is refuses fleets that fit, and one assumed larger
                     over-subscribes every GPU while reporting that it had not. There is no safe
                     direction to guess in, so an unmeasured device is refused instead of estimated.
    overhead_gib     per TP degree: what an engine costs the device OUTSIDE that budget -- CUDA
                     context, cuBLAS/cuDNN scratch, graph pools, NCCL buffers. MEASURED, from the
                     driver's resident memory before and during a boot. This one does have a safe
                     direction -- overstating it makes an unmeasured candidate rank worse than it
                     deserves -- so `memory.MPS_RESERVE_TP1_GIB` survives as a conservative fallback
                     for hardware nobody has measured. A sheet supersedes it; nothing tunes it.

**Only the degrees that were measured.** `overhead_gib` is a dict with holes in it on purpose. The
standing assumption in `memory.py` is one figure at TP=1 and a larger one at every wider degree,
which is a guess about how NCCL buffers scale that has never been tested. A consumer asking for TP=4
and finding no TP=4 key has learned something true, and interpolating on its behalf would convert
that into a number nobody measured.

**Identity: what the device reports about itself, not the instance type and not a driver version.**
What conditions these numbers is the card, how much of it a context can address, what it is allowed to
draw, the links it speaks over, how it is shared, and the CUDA runtime inside the container -- that is
`store/environment.DeviceEnvironment`, and every field of it is observed. The driver version used to be
half of this key and is now recorded in the sheet: it was a proxy for the addressable total, which is
now measured directly, and as a key it discarded the store on a bump that changed nothing while sitting
still through a power cap that changed everything. The instance type is not in the key either: two
instance types carrying the same GPU in the same environment have the same device law, and keying on
the SKU would spend a measurement each to rediscover it. What the instance type does determine is how
many GPUs there are and what environment a fleet will meet on one, which is an observation of a node
rather than a measurement of a device, needs no engine to obtain, and is filed separately as an
`InstanceShape` -- the record that lets `plan` compute an address offline.

**The schema version is inside the key**, for the same reason the deleted model-fact key carried
one: an old sheet under new field
semantics is the dangerous case, and a version in the key means it simply does not match.

Layout:

    /efs/colocation/hwfacts/
      instances/<instance-type>.json          observed at capture; no engine involved
      <hardware-key-hash>/
        key.json
        devices/<gpu-product>.json

**Two readers, deliberately.** `HardwareStore` reads one environment, which is what a writer needs
and what any consumer of `overhead_gib` needs. `HardwareIndex` reads the device *total* across all of
them, because the total is a property of the card rather than of the environment -- and it refuses
when two environments disagree about it, which is how that claim gets tested rather than assumed.
`agent/hardware.hardware_for` is built on the second.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..tiers import TIER_MEASURED, TIERS
from .environment import DeviceEnvironment, Environment, validate_device_facts
from .validity_key import HASH_CHARS, ValidityKey

log = logging.getLogger(__name__)

# Bumped when the meaning of a field changes. Adding `overhead_gib` degrees is not a bump; changing
# what `total_gib` is the total *of* would be.
HWFACTS_SCHEMA_VERSION = 1


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(name).strip().lower())


@dataclass(frozen=True)
class HardwareKey:
    """The environment a device law is valid in, plus the schema it was written under.

    Two fields. `env` is hashed through `Environment.device_canonical()` -- the card, what it
    addresses, what it may draw, the links it speaks over, the runtime that opens a context on it and
    how it is shared -- and that projection is the content: `gpu_count` and `metrics_version` are
    deliberately left out, because a card's memory law is valid for every model that will ever run on
    that device and does not expire when the box has a different number of them or when a latency
    percentile is redefined. `schema_version` so that a build reading a different schema looks in a
    different directory and finds nothing, rather than parsing an old sheet under new meanings.

    The driver version used to be half of this key. It is now recorded in the sheet instead: what it
    was a proxy for -- how much of the board a CUDA context can address -- is a field of the
    environment, measured (`store/environment.py`).

    Deliberately NOT composed from `ProfileKey`, the way the deleted model-fact key was (impl §14.4).
    A model fact is valid for one model under one serving configuration on one machine; a device fact
    is valid for every model that will ever run on that device. Borrowing the narrower key would
    expire the card's capacity whenever a checkpoint's flags changed, and re-measure hardware for no
    reason.
    """

    env: DeviceEnvironment
    schema_version: int = HWFACTS_SCHEMA_VERSION

    @classmethod
    def from_validity_key(cls, key: ValidityKey,
                          schema_version: int = HWFACTS_SCHEMA_VERSION) -> HardwareKey:
        return cls(env=key.env.device(), schema_version=schema_version)

    def canonical(self) -> dict:
        return {"environment": self.env.canonical(),
                "schema_version": int(self.schema_version)}

    @property
    def hash(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:HASH_CHARS]

    def key_json(self) -> str:
        return json.dumps({"hash": self.hash, "kind": "hardware-facts", **self.canonical()},
                          indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, d: dict) -> HardwareKey:
        if "environment" not in d:
            raise ValueError(
                "this hardware key has no environment block. It was written when the key was a "
                "driver version and an image, so the addressable total and the power limit the sheet "
                "under it was measured at are not recorded in its address. Re-measure the card.")
        return cls(env=DeviceEnvironment.from_dict(d["environment"]),
                   schema_version=int(d.get("schema_version", HWFACTS_SCHEMA_VERSION)))


@dataclass(frozen=True)
class InstanceShape:
    """What a node of this instance type presents. An observation, not a measurement.

    Filed outside the hardware key because it is what the key is *computed from*, and obtained by
    reading a node rather than by running an engine -- `capture._observe` asks `nvidia-smi` and the
    Karpenter labels for all of it and cross-checks the two where both answer. Recording it here is
    what lets `plan` work offline on an instance type somebody has leased once, with no table in the
    source to edit and nothing hardware-specific written by hand.

    It carries the whole device half of an `Environment` and not just the GPU count, and that is the
    field list that matters: `plan` is offline, it must name the directory its rows live in, and the
    only honest way to do that without asking an author to type a hardware fact is to replay what the
    last node of this type actually reported. `engine_image` is the one term it cannot supply, because
    that is the fleet's own choice rather than the node's.

    Nothing here is defaulted. A shape filed before these fields existed is refused by `parse` rather
    than half-read, because a shape missing the addressable total would compute an address for rows
    that were measured somewhere else.
    """

    instance_type: str
    gpu_count: int
    gpu_product: str
    addressable_mib: int
    power_limit_w: int
    p2p_topology: str
    observed_at: str = ""
    node: str = ""
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if not str(self.instance_type).strip():
            raise ValueError("instance_type is required")
        if int(self.gpu_count) < 1:
            raise ValueError(f"gpu_count must be >= 1, got {self.gpu_count!r}")
        if not str(self.gpu_product).strip():
            raise ValueError("gpu_product is required: a shape that does not say which GPU cannot "
                             "be joined to the device law that describes it")
        # Shared with `DeviceEnvironment`, so a shape cannot be filed that the key computed from it
        # would refuse.
        validate_device_facts(self.gpu_product, self.addressable_mib, self.power_limit_w,
                              self.p2p_topology)

    @property
    def id(self) -> str:
        return _safe(self.instance_type)

    @property
    def gpu_ids(self) -> tuple:
        """Host ordinals, which are always 0..n-1."""
        return tuple(range(int(self.gpu_count)))

    def device_environment(self, engine_image: str, **kwargs) -> DeviceEnvironment:
        return DeviceEnvironment(gpu_product=str(self.gpu_product),
                                 addressable_mib=int(self.addressable_mib),
                                 power_limit_w=int(self.power_limit_w),
                                 p2p_topology=str(self.p2p_topology),
                                 engine_image=str(engine_image), **kwargs)

    def environment(self, engine_image: str, *, gpu_count: int | None = None,
                    **kwargs) -> Environment:
        """The full `Environment` a fleet served on a node of this type is conditioned by.

        `gpu_count` may be overridden for the case a caller is describing fewer devices than the node
        has; it defaults to what the node presented, which is what a fleet gets.
        """
        return Environment(gpu_product=str(self.gpu_product),
                           gpu_count=int(self.gpu_count if gpu_count is None else gpu_count),
                           addressable_mib=int(self.addressable_mib),
                           power_limit_w=int(self.power_limit_w),
                           p2p_topology=str(self.p2p_topology),
                           engine_image=str(engine_image), **kwargs)

    def to_dict(self) -> dict:
        return {"schema_version": HWFACTS_SCHEMA_VERSION, "instance_type": self.instance_type,
                "gpu_count": int(self.gpu_count), "gpu_product": self.gpu_product,
                "addressable_mib": int(self.addressable_mib),
                "power_limit_w": int(self.power_limit_w), "p2p_topology": str(self.p2p_topology),
                "observed_at": self.observed_at, "node": self.node,
                "notes": dict(self.notes or {})}

    @classmethod
    def parse(cls, d: dict) -> InstanceShape:
        _check_version(d, "instance shape")
        missing = [f for f in ("addressable_mib", "power_limit_w", "p2p_topology")
                   if d.get(f) in (None, "")]
        if missing:
            raise ValueError(
                f"this instance shape is missing {', '.join(missing)}. It was observed when a shape "
                "was a GPU count and a product name, so it does not say what one of those cards "
                "addressed, what it was allowed to draw, or how the devices were linked -- and those "
                "are terms in the address every row on this instance type is filed under. Capture a "
                "node of this type again.")
        return cls(instance_type=str(d["instance_type"]), gpu_count=int(d["gpu_count"]),
                   gpu_product=str(d["gpu_product"]), addressable_mib=int(d["addressable_mib"]),
                   power_limit_w=int(d["power_limit_w"]), p2p_topology=str(d["p2p_topology"]),
                   observed_at=str(d.get("observed_at") or ""),
                   node=str(d.get("node") or ""), notes=dict(d.get("notes") or {}))


def _check_version(d: dict, what: str) -> None:
    got = int(d.get("schema_version", -1))
    if got != HWFACTS_SCHEMA_VERSION:
        raise ValueError(
            f"{what} schema_version={got}, this build reads {HWFACTS_SCHEMA_VERSION}. The version "
            "is inside the key precisely so this cannot happen by accident.")


@dataclass(frozen=True)
class HardwareFacts:
    """One GPU product's memory law under one driver and one image.

    `total_source` says which of the two measurements the total came from -- the affine slope across
    two boots, or the driver's `memory.total`. Both read a real device, so both are `measured`; what
    differs is what they are measurements *of*. The slope measures the number the engine's own
    arithmetic used, which is the number this sheet exists to supply, and the driver figure is the
    independent one that can contradict it. A sheet resting on the driver alone says so, because it
    means nothing has tested that the engine agrees.

    `total_falsifiable` is the same distinction one level up: two boots fit a slope exactly and leave
    nothing over to disagree, so a total from two points and no cross-check has been fitted rather
    than tested. It reads `False` on everything filed today -- a driver reading is one instrument and
    has nothing to disagree with -- and the sheets it reads `True` on are the older fitted ones.
    """

    gpu_product: str
    total_gib: float
    total_tier: str = TIER_MEASURED
    total_source: str = ""
    total_uncertainty_gib: float | None = None
    total_falsifiable: bool = False
    driver_total_gib: float | None = None
    overhead_gib: dict = field(default_factory=dict)
    overhead_uncertainty_gib: dict = field(default_factory=dict)
    overhead_tier: str = TIER_MEASURED
    boots: tuple = ()
    checks: tuple = ()
    measured_with: dict = field(default_factory=dict)
    driver: str = ""
    image: str = ""
    gpu_node: str = ""
    measured_at: str = ""
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if not str(self.gpu_product).strip():
            raise ValueError("gpu_product is required")
        if not float(self.total_gib) > 0.0:
            raise ValueError("total_gib must be positive; a sheet without a device total states "
                             "nothing, and there is no default to fall back on")
        for name in ("total_tier", "overhead_tier"):
            if getattr(self, name) not in TIERS:
                raise ValueError(f"{name}={getattr(self, name)!r} is not one of {TIERS}")

    @property
    def id(self) -> str:
        return _safe(self.gpu_product)

    @property
    def checks_passed(self) -> bool:
        """A sheet with no checks is not passing; it is untested."""
        return bool(self.checks) and all(c.get("passed") for c in self.checks)

    @property
    def failed_checks(self) -> tuple:
        return tuple(c for c in self.checks if not c.get("passed"))

    @property
    def degrees_measured(self) -> tuple:
        return tuple(sorted(int(k) for k in self.overhead_gib))

    def overhead_for(self, tp: int) -> float | None:
        """The overhead measured at exactly this degree, or None. Never a neighbour's.

        None is the honest answer and callers are expected to fall back to
        `memory.MPS_RESERVE_*` -- which is over-safe rather than accurate, and is the right thing to
        be wrong with. Returning the nearest measured degree instead would look more helpful and be a
        number nobody took.
        """
        return self.overhead_gib.get(int(tp), self.overhead_gib.get(str(int(tp))))

    def committable(self) -> tuple:
        """`(ok, reason)` -- whether this sheet is strong enough to deploy a real tenant against.

        Falsifiability is tested before the checks, and the order is the message: a sheet with no
        checks at all usually has none *because* nothing here could have disagreed, and "no checks
        recorded" states the symptom while "fitted and never tested" states the cause and names the
        remedy. A reader who is told the cause knows which boot to spend.
        """
        if not self.total_falsifiable:
            return False, ("the device total was fitted and never tested: no third utilisation, no "
                           "second degree, and no driver figure beside it to disagree. One more boot "
                           "at another utilisation, or one `nvidia-smi --query-gpu=memory.total` on "
                           "the node, settles it")
        if not self.checks_passed:
            if not self.checks:
                return False, "no checks recorded: nothing about this sheet has been falsified"
            return False, ("failed checks: "
                           + "; ".join(str(c.get("name")) for c in self.failed_checks))
        return True, "measured, cross-checked, and every recorded check passed"

    def to_dict(self) -> dict:
        return {
            "schema_version": HWFACTS_SCHEMA_VERSION,
            "gpu_product": self.gpu_product,
            "total_gib": self.total_gib,
            "total_tier": self.total_tier,
            "total_source": self.total_source,
            "total_uncertainty_gib": self.total_uncertainty_gib,
            "total_falsifiable": bool(self.total_falsifiable),
            "driver_total_gib": self.driver_total_gib,
            "overhead_gib": {int(k): v for k, v in self.overhead_gib.items()},
            "overhead_uncertainty_gib": {int(k): v
                                         for k, v in self.overhead_uncertainty_gib.items()},
            "overhead_tier": self.overhead_tier,
            "boots": [dict(b) for b in self.boots],
            "checks": [dict(c) for c in self.checks],
            "measured_with": dict(self.measured_with or {}),
            "driver": self.driver,
            "image": self.image,
            "gpu_node": self.gpu_node,
            "measured_at": self.measured_at,
            "notes": dict(self.notes or {}),
        }

    @classmethod
    def parse(cls, d: dict) -> HardwareFacts:
        _check_version(d, "hardware fact sheet")
        return cls(
            gpu_product=str(d["gpu_product"]),
            total_gib=float(d["total_gib"]),
            total_tier=str(d.get("total_tier", TIER_MEASURED)),
            total_source=str(d.get("total_source") or ""),
            total_uncertainty_gib=(None if d.get("total_uncertainty_gib") is None
                                   else float(d["total_uncertainty_gib"])),
            total_falsifiable=bool(d.get("total_falsifiable")),
            driver_total_gib=(None if d.get("driver_total_gib") is None
                              else float(d["driver_total_gib"])),
            overhead_gib={int(k): float(v) for k, v in (d.get("overhead_gib") or {}).items()},
            overhead_uncertainty_gib={int(k): float(v) for k, v
                                      in (d.get("overhead_uncertainty_gib") or {}).items()},
            overhead_tier=str(d.get("overhead_tier", TIER_MEASURED)),
            boots=tuple(dict(b) for b in (d.get("boots") or ())),
            checks=tuple(dict(c) for c in (d.get("checks") or ())),
            measured_with=dict(d.get("measured_with") or {}),
            driver=str(d.get("driver") or ""),
            image=str(d.get("image") or ""),
            gpu_node=str(d.get("gpu_node") or ""),
            measured_at=str(d.get("measured_at") or ""),
            notes=dict(d.get("notes") or {}))


class HardwareStore:
    """Device laws under one hardware key, plus the instance shapes that are keyed by nothing."""

    def __init__(self, root: str | Path, key: HardwareKey | str):
        self.root = Path(root)
        if isinstance(key, HardwareKey):
            self.key: HardwareKey | None = key
            self.key_hash = key.hash
        else:
            self.key = None
            self.key_hash = key

    @property
    def base_dir(self) -> Path:
        return self.root / "hwfacts"

    @property
    def key_dir(self) -> Path:
        return self.base_dir / self.key_hash

    @property
    def devices_dir(self) -> Path:
        return self.key_dir / "devices"

    @property
    def instances_dir(self) -> Path:
        """Outside the key directory: an instance's GPU count is not conditioned by driver or image."""
        return self.base_dir / "instances"

    def write_key(self) -> Path:
        if self.key is None:
            raise ValueError("no HardwareKey: this store was opened by hash only")
        path = self.key_dir / "key.json"
        payload = self.key.key_json()
        if path.exists() and path.read_text() != payload:
            raise ValueError(
                f"{path} exists and does not match this key. Same hash, different contents means "
                "two environments have been filed together, and every device law in here would be "
                "attributed to the wrong driver or image.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
        return path

    # --- device laws ---------------------------------------------------------

    def path_for(self, facts_or_product) -> Path:
        pid = (_safe(facts_or_product) if isinstance(facts_or_product, str)
               else facts_or_product.id)
        return self.devices_dir / f"{pid}.json"

    def write(self, facts: HardwareFacts) -> Path:
        path = self.path_for(facts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(facts.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    def read(self, gpu_product: str) -> HardwareFacts | None:
        path = self.path_for(gpu_product)
        if not path.exists():
            return None
        return HardwareFacts.parse(json.loads(path.read_text()))

    def all(self) -> list:
        """Every device law under this key. An unreadable one is a hard error, unlike an unreadable
        profile row: the fallback for a missing device total is a refusal, and silently taking the
        refusal because a file had a stray byte in it would read as "this hardware was never
        measured"."""
        if not self.devices_dir.exists():
            return []
        out = []
        for path in sorted(self.devices_dir.glob("*.json")):
            try:
                out.append(HardwareFacts.parse(json.loads(path.read_text())))
            except Exception as exc:
                raise ValueError(f"{path} is not a readable hardware fact sheet: {exc}") from exc
        return out

    # --- instance shapes -----------------------------------------------------

    def shape_path(self, instance_type: str) -> Path:
        return self.instances_dir / f"{_safe(instance_type)}.json"

    def write_shape(self, shape: InstanceShape) -> Path:
        """File one instance shape, refusing to contradict the two fields that cannot change.

        **The count and the product are refused; the device facts are replaced.** How many GPUs of what
        kind a `g6e.48xlarge` has cannot change without the instance type changing meaning, so two
        captures disagreeing about it means one of the two nodes was not what it said it was -- most
        likely a daemon that could see fewer devices than the node had, which `capture._observe`
        already treats as a hard error -- and overwriting would erase the evidence.

        The addressable total, the power limit and the link matrix are not like that. An operator can
        cap a card, and a driver upgrade can change what it reserves, so a newer reading is the better
        one rather than a contradiction. Replacing it is also the only way the store can heal: what a
        change in any of those three means is that this instance type now addresses a different
        directory, and a run that could not update the shape would keep computing the old address
        forever. `capture.reconcile_hardware` is what makes the change visible -- it files the new
        observation and stops the run rather than mixing the two.

        An existing file that cannot be read as a current shape is replaced for the same reason and
        not treated as a contradiction: it is a shape from before those fields were recorded, so it
        states nothing this one disagrees with, and refusing would leave the type unreadable and
        unwritable at once.
        """
        path = self.shape_path(shape.instance_type)
        if path.exists():
            try:
                old = InstanceShape.parse(json.loads(path.read_text()))
            except Exception as exc:        # noqa: BLE001 -- see the docstring
                log.warning("%s held a shape this build cannot read (%s); replacing it with what "
                            "%s observed", path, exc, shape.node or "this capture")
                old = None
            if old is not None and (old.gpu_count, old.gpu_product) != (shape.gpu_count,
                                                                       shape.gpu_product):
                raise ValueError(
                    f"{shape.instance_type} is already filed as {old.gpu_count}x"
                    f"{old.gpu_product} (from node {old.node or 'unknown'}) and this capture says "
                    f"{shape.gpu_count}x{shape.gpu_product}. An instance type has one shape, so one "
                    "of the two nodes was not what it claimed. Nothing overwritten.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(shape.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    def read_shape(self, instance_type: str) -> InstanceShape | None:
        path = self.shape_path(instance_type)
        if not path.exists():
            return None
        return InstanceShape.parse(json.loads(path.read_text()))

    def shapes(self) -> list:
        if not self.instances_dir.exists():
            return []
        out = []
        for path in sorted(self.instances_dir.glob("*.json")):
            try:
                out.append(InstanceShape.parse(json.loads(path.read_text())))
            except Exception as exc:
                raise ValueError(f"{path} is not a readable instance shape: {exc}") from exc
        return out


class Disagreement(Exception):
    """Two environments filed different sizes for the same card. Never resolved by picking one."""


@dataclass(frozen=True)
class DeviceTotal:
    """A device total found in the store, and everything needed to say what it is worth.

    Separate from `HardwareFacts` because it is an answer assembled across sheets rather than one
    sheet: `keys` is every hardware key whose sheet agrees on this figure, and a consumer that
    prints "44.99 GiB, measured under 2 environments" is saying something a single sheet cannot.
    """

    gpu_product: str
    total_gib: float
    uncertainty_gib: float | None
    source: str
    falsifiable: bool
    keys: tuple

    def provenance(self) -> str:
        """One line, for a report. Says which measurement this is and how exposed it is."""
        unc = "" if self.uncertainty_gib is None else f" +/- {self.uncertainty_gib:.3f}"
        tested = "cross-checked" if self.falsifiable else "not yet contradicted by anything"
        under = f"{len(self.keys)} environment(s)" if self.keys else "no recorded environment"
        return (f"{self.total_gib:.3f}{unc} GiB per device, measured ({self.source}) under "
                f"{under}, {tested}")


class HardwareIndex:
    """Every device law under one store root, whatever environment measured it.

    **Why this reads across hardware keys when `HardwareStore` does not.** A `HardwareKey` names the
    driver and the image, and it is the right key for the sheet as a whole because the *overhead*
    half of it is a property of the CUDA context the image creates. The device total is not: it is
    how much memory the card has, and the container that talks to it does not change that. So a
    consumer that needs only the total is asking a question the key is narrower than, and making it
    supply an image it has no reason to know would mean re-measuring a card because a container tag
    moved.

    `plan` is the caller that forces the issue. It is offline and free, it needs the total to size
    envelopes, and it runs before anything has been leased -- so it can state which image a fleet
    declares, but not that the sheet it wants was measured under that image. Reading across keys is
    what lets it work at all.

    **What makes that safe is the check, not the argument.** If the total really is a property of
    the card, then every environment that measured it must have got the same number, and
    `device_total` refuses when two of them did not. That refusal is the claim being tested: a
    disagreement means something in the driver or the runtime *does* condition the capacity, this
    class's premise is wrong for that hardware, and the last thing to do is average two figures or
    take the newer one. Overhead is never read this way and there is no method here that offers it.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def base_dir(self) -> Path:
        return self.root / "hwfacts"

    @property
    def instances_dir(self) -> Path:
        return self.base_dir / "instances"

    def key_hashes(self) -> list:
        """Every hardware key directory present. `instances/` is not one: it is keyed by nothing."""
        if not self.base_dir.exists():
            return []
        return sorted(d.name for d in self.base_dir.iterdir()
                      if d.is_dir() and d.name != "instances")

    def store_for(self, key_hash: str) -> HardwareStore:
        return HardwareStore(self.root, str(key_hash))

    def read_key(self, key_hash: str) -> HardwareKey | None:
        path = self.base_dir / str(key_hash) / "key.json"
        if not path.exists():
            return None
        return HardwareKey.from_dict(json.loads(path.read_text()))

    # --- instance shapes (keyed by nothing, so they are read here) ------------

    def read_shape(self, instance_type: str) -> InstanceShape | None:
        path = self.instances_dir / f"{_safe(instance_type)}.json"
        if not path.exists():
            return None
        return InstanceShape.parse(json.loads(path.read_text()))

    # Delegated to a store opened with no key, rather than reimplemented, because the shape
    # directory sits outside every key directory and there is one set of rules for writing it --
    # notably `write_shape`'s refusal to contradict a filed shape, which must hold whichever of the
    # two types a caller happens to be holding.
    def shapes(self) -> list:
        return HardwareStore(self.root, "").shapes()

    def write_shape(self, shape: InstanceShape) -> Path:
        return HardwareStore(self.root, "").write_shape(shape)

    # --- device totals across every environment ------------------------------

    def sheets_for(self, gpu_product: str) -> list:
        """`(key_hash, HardwareFacts)` for every environment that measured this product."""
        out = []
        for key_hash in self.key_hashes():
            sheet = self.store_for(key_hash).read(gpu_product)
            if sheet is not None:
                out.append((key_hash, sheet))
        return out

    def device_total(self, gpu_product: str) -> DeviceTotal | None:
        """The one figure the store holds for this card's size, or None if nobody has measured it.

        Raises `Disagreement` when two environments filed sizes further apart than their own
        uncertainties allow. The tolerance comes from the sheets rather than from a constant here,
        because a total taken across a narrow utilisation spread is honestly fuzzier than one taken
        across a wide one, and holding both to the same absolute gap would fail an honest pair of
        measurements. The fitted sheets that carry the wider uncertainties are older than impl §14.4;
        a driver reading carries the query's own quantisation and no more.

        When several agree, the reported figure is the best-resolved one -- smallest uncertainty --
        and not their mean. Same rule and same reason as `identify_device` picking the widest
        utilisation spread over averaging: a fuzzy reading has no business moving a sharp one, and a
        mean is a number no instrument produced.
        """
        found = self.sheets_for(gpu_product)
        if not found:
            return None

        def unc(sheet) -> float:
            # A sheet that recorded no uncertainty is not thereby exact. It is treated as the
            # loosest of the group so that it can never be the one chosen, and can never tighten a
            # comparison it has no standing to tighten.
            u = sheet.total_uncertainty_gib
            return float("inf") if u is None else abs(float(u))

        totals = [(float(s.total_gib), unc(s), h, s) for h, s in found]
        lo = min(t for t, *_ in totals)
        hi = max(t for t, *_ in totals)
        if hi - lo > 0.0:
            # Each pair is allowed the sum of what the two sheets admit to, floored at the print
            # quantum of `nvidia-smi memory.total` -- whole MiB -- so that two driver readings of one
            # card differing in the last place are not a disagreement.
            slack = sum(sorted(u for _, u, _, _ in totals)[:2])
            if slack == float("inf") or hi - lo > max(slack, 1.0 / 1024.0):
                raise Disagreement(
                    f"the store holds {len(totals)} different sizes for {gpu_product!r}: "
                    + "; ".join(f"{h[:8]}={t:.3f} GiB" for t, _, h, _ in sorted(totals))
                    + ". A card's capacity is not conditioned by the driver or the image, so these "
                    "cannot all be right, and nothing here will pick one: either two different "
                    "devices have been filed under one product name, or something in those "
                    "environments does change the capacity and the assumption this lookup rests on "
                    "is wrong for this hardware. Re-measure on one node and delete the sheet that "
                    "does not describe it.")
        best = min(totals, key=lambda row: (row[1], row[0]))
        _, _, _, sheet = best
        return DeviceTotal(
            gpu_product=sheet.gpu_product,
            total_gib=float(sheet.total_gib),
            uncertainty_gib=sheet.total_uncertainty_gib,
            source=sheet.total_source or "unrecorded",
            # Falsifiable if ANY sheet tested it, and the several-sheet case is itself a test: two
            # environments agreeing on a figure is exactly the independent confirmation a two-point
            # slope lacks.
            falsifiable=(any(s.total_falsifiable for _, s in found) or len(found) > 1),
            keys=tuple(h for _, _, h, _ in sorted(totals, key=lambda row: row[2])))
