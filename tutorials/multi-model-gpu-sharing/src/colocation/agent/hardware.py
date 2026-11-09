"""What a machine is, before anything is running on it.

Hardware selection is by **EC2 instance type** (decided 2026-09-04 by the user; AWS only). A
fleet file names a type and nothing else about the machine: how many GPUs it has, which GPU, and
how big one of them is all follow from the type. That matters because those three used to be
restated per fleet file as `node.gpu_ids` and `node.usable_gib` beside a hand-written node name,
and a fleet file is the wrong place for a fact about hardware -- it is copied, and a copied
hardware fact goes stale silently.

**There are no hardware numbers in this module.** There used to be two tables: instance type ->
(GPU count, GPU product), and GPU product -> device memory. Both are gone, and what replaced them is
the store (`store/hwfacts.py`):

    how many GPUs, which GPU   an `InstanceShape`, OBSERVED on a node that was leased -- the GPU
                               count from `nvidia-smi` and the Karpenter label, cross-checked
                               against each other by `capture._observe`.
    how big one GPU is         a `HardwareFacts` total, MEASURED -- the driver's own addressable
                               total, `memory.total` minus `memory.reserved`, read off the leased
                               node by `capture._observe`. The subtraction is the whole point: the
                               reserved region is a part of the board no CUDA context can address,
                               and it is the figure the engine multiplies
                               `--gpu-memory-utilization` by.

The point of the change is generality, not tidiness. A table has to be edited before this stack can
be pointed at hardware nobody here has run on, so every entry in it is a claim somebody typed about a
machine they may not have seen; and a capacity typed from a spec sheet is wrong in whichever
direction the spec sheet rounds, with the consequence arriving hours later as an OOM in one tenant's
sampler warmup, where it reads as a bad placement rather than as a bad constant. Read from the store
instead, the same code works on a card that did not exist when it was written, and the price of new
hardware is a run rather than a patch.

**The one figure that has no safe default, and therefore no default.** A device total assumed too
small refuses fleets that would have fitted; assumed too large it over-subscribes every GPU on the
node while reporting that it had not. There is no direction to be conservative in, so an unmeasured
card is refused rather than estimated, and `hardware_for` says which of the two facts is missing and
what obtains it.

**Bootstrapping, and why it is a flag rather than a table.** The store starts empty, and the shape is
what `capture` needs *before* it can lease anything -- it has to request every GPU on the node to hold
it exclusively. So the first run against an instance type nobody has leased may state the count with
`--gpu-count` (and `--gpu-product` if it wants a plan priced against a card measured elsewhere). That
is the same nameplate fact the deleted table held, with three differences that are the whole
argument: it is stated by the person running the mode rather than inherited from whoever wrote the
source, it is checked against the leased node before it can affect any measurement
(`Hardware.verify`), and it is filed as an observation afterwards so nobody states it twice. A
`Hardware` carries `shape_source` so that a plan made on an unverified count says so in its own
output.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..store.environment import Environment
from ..store.hwfacts import HardwareIndex, InstanceShape

# Where a `Hardware`'s GPU count and product came from. Printed, not branched on -- except by
# `verify`, which is stricter about a figure the store already holds than about one just stated.
SHAPE_OBSERVED = "observed on a leased node"
SHAPE_STATED = "stated by the operator (--gpu-count), never verified against a node"

# `nvidia-smi --query-gpu=memory.total` reports whole MiB, so a driver-sourced total carries that
# much quantisation and no more. It is a property of the query this module's callers make.
DRIVER_TOTAL_QUANTUM_GIB = 1.0 / 1024.0

# Floor for the relative agreement check in `verify` below. The quantisation above dominates it on any
# card this runs on; it exists so that allocator rounding cannot fail an honest reading. It used to
# live in `solver/devicelaw.py`, which fitted a card's total from an engine's own KV report and was
# deleted with `onboard` (impl §14.4) -- the check that survived it is this module's, over two figures
# the driver states, so the constant belongs here.
DEVICE_TOL_FLOOR = 0.005


class UnknownHardware(Exception):
    """The store holds no shape for this instance type, or no measured total for its GPU."""


@dataclass(frozen=True)
class Hardware:
    """One instance type, resolved against the store. What the solver needs to size a placement.

    `device_total_gib` is **optional here and nowhere else**, and the asymmetry is the design. Every
    consumer that multiplies a utilisation by it needs a real number and gets one by calling
    `require_total_gib`, which refuses with the remedy. But the modes that lease a node do not need it
    to lease -- `capture` requests GPUs by count -- and by the time they do need it they have read the
    driver's own figure off the node they are holding (`with_driver_total`). Requiring it up front
    would mean no run could ever measure a card the store does not already describe.
    """

    instance_type: str
    gpu_count: int
    # The Karpenter product token (`l40s`, `a10g`, `h100`), or "" when nobody has observed this
    # instance type and the operator stated only a count. Empty means no device total can be looked
    # up: the total is filed per card, not per instance type.
    gpu_product: str
    # What the driver reports as the size of ONE device, or None when no environment has measured
    # this card. The only capacity figure: it is both the denominator of every tenant's
    # `gpu_memory_utilization` and the capacity the envelopes on one GPU must fit inside. There used
    # to be a second, smaller "usable" figure beside it; the difference between them was memory the
    # engine handed out and the packing never priced.
    device_total_gib: float | None = None
    # Provenance of that figure, for reports and for `verify`'s tolerance. `total_source` is `driver`
    # on everything filed today; `slope` appears in sheets already on EFS, filed by the deleted
    # `solver/devicelaw.py` (impl §14.4), which fitted it from an engine's own KV report.
    # `total_falsifiable` is whether anything could have contradicted it; `measured_under` is the
    # hardware keys whose sheets agree on it.
    total_source: str = ""
    total_uncertainty_gib: float | None = None
    total_falsifiable: bool = False
    measured_under: tuple = ()
    shape_source: str = SHAPE_OBSERVED
    shape_node: str = ""
    # The rest of what a node of this type reported about its devices, from the filed `InstanceShape`:
    # the addressable total in whole MiB, the enforced power limit in whole watts, and a digest of the
    # link matrix. They are here for one purpose -- with the fleet's own image they are the address
    # every stored row is filed under (`store/environment.py`) -- and that is why they are on this
    # object rather than read at each use: `plan` is offline and still has to name the directory it
    # reads from, and the only honest way to do that is to replay what the last node of this type
    # actually reported.
    #
    # None/"" means the shape was never observed, which today is only the `--gpu-count` bootstrap. Then
    # no address can be computed, and `environment` refuses instead of inventing one. That refusal
    # costs nothing real: a type nobody has leased has no rows to look up.
    addressable_mib: int | None = None
    power_limit_w: int | None = None
    p2p_topology: str = ""

    @property
    def gpu_ids(self) -> tuple:
        """Host ordinals, which are always 0..n-1.

        Nothing here distinguishes one ordinal from another, and that is an assumption, not an
        observation -- see the `No adjacency constraint` note in solver/cpsat.py, and the
        g6e.48xlarge measurement that violates it. Kept because the alternative is a
        per-instance-type interconnect table that would have to be right about machines nobody
        here has run on; the run records which ordinals it used instead.
        """
        return tuple(range(self.gpu_count))

    @property
    def total_known(self) -> bool:
        return self.device_total_gib is not None

    @property
    def environment_known(self) -> bool:
        """Whether this machine can name the store directory its rows live in."""
        return (self.addressable_mib is not None and self.power_limit_w is not None
                and bool(self.p2p_topology) and bool(self.gpu_product))

    def environment(self, engine_image: str, **kwargs) -> Environment:
        """The address every row taken on this machine is filed under.

        Refused rather than defaulted when the device facts are not in hand, for the reason the whole
        module is built on: a fabricated field produces a hash that names a directory whose rows are
        not what it claims, and unlike a wrong capacity that failure is silent forever -- the rows in
        it are each individually valid.
        """
        if not self.environment_known:
            missing = [n for n in ("gpu_product", "addressable_mib", "power_limit_w", "p2p_topology")
                       if not getattr(self, n)]
            raise UnknownHardware(
                f"nobody has observed what a {self.instance_type} presents ({', '.join(missing)} "
                "unknown), so there is no address to file measurements under or read them from. It is "
                "not a table to add a row to: lease one node of this type -- `profile` or `search` "
                "does it, with `--gpu-count N` on a first run -- and the capture reads the addressable "
                "total, the enforced power limit and the link matrix off it and files them. Every run "
                "after that resolves the address offline and for free.")
        return Environment(gpu_product=self.gpu_product, gpu_count=int(self.gpu_count),
                           addressable_mib=int(self.addressable_mib),
                           power_limit_w=int(self.power_limit_w),
                           p2p_topology=str(self.p2p_topology),
                           engine_image=str(engine_image), **kwargs)

    def require_total_gib(self, what: str = "") -> float:
        """The device total, or a refusal naming what obtains it.

        Called at every site that multiplies a utilisation by it or fits envelopes inside it, rather
        than checked once at construction, so that the refusal names the thing that needed the number
        instead of arriving before the run knows what it is doing.
        """
        if self.device_total_gib is None:
            need = f" {what} needs it." if what else ""
            raise UnknownHardware(
                f"no environment has measured how much memory a {self.gpu_product or 'GPU'} has, "
                f"and there is no default for it.{need} It is the figure the engine multiplies "
                "gpu_memory_utilization by, so every tenant's whole memory budget rests on it, and a "
                "spec sheet does not state it: assumed too small it refuses fleets that fit, "
                "assumed too large it over-subscribes every GPU while reporting that it had not. "
                f"Lease a {self.instance_type} -- `profile` or `search` does it, or `--gpu-count` "
                "on a first run -- and the capture reads the driver's addressable total off the node "
                "and files it in the hardware store. Every run after that resolves it offline.")
        return float(self.device_total_gib)

    def provenance(self) -> str:
        """One line describing where this machine's three facts came from. For reports."""
        shape = f"{self.gpu_count}x{self.gpu_product or 'unknown GPU'} ({self.shape_source}"
        shape += f", {self.shape_node})" if self.shape_node else ")"
        if self.device_total_gib is None:
            return f"{shape}; device total UNMEASURED"
        unc = ("" if self.total_uncertainty_gib is None
               else f" +/- {self.total_uncertainty_gib:.3f}")
        tested = "cross-checked" if self.total_falsifiable else "nothing has contradicted it yet"
        return (f"{shape}; {self.device_total_gib:.3f}{unc} GiB per device, measured "
                f"({self.total_source or 'source unrecorded'}, {tested})")

    def verify(self, *, gpu_count: int | None = None, gpu_product: str | None = None,
               driver_total_gib: float | None = None) -> None:
        """Check what was observed on a leased node against what this `Hardware` already claims.

        Raises rather than adopting the observation, and that direction is deliberate. These numbers
        have already been used -- `plan` printed feasible placements from them and the store
        directory is named by a key built beside them -- so silently switching to the observed value
        mid-run would mean the placements reported and the placements deployed were sized
        differently, with nothing saying so.

        The exception is a total nobody has measured, which cannot have been used for anything:
        `with_driver_total` adopts in that one case, and this method has nothing to compare.
        """
        if gpu_count is not None and int(gpu_count) != self.gpu_count:
            source = ("was stated on the command line as --gpu-count"
                      if self.shape_source == SHAPE_STATED
                      else f"is filed in the hardware store as {self.gpu_count}")
            raise UnknownHardware(
                f"{self.instance_type} {source} but the leased node reports {gpu_count} GPU(s). "
                "The fleet was already planned against "
                f"{self.gpu_count}, so continuing would deploy a placement sized for a machine this "
                "is not. If the node is right, delete the filed shape and capture again; if the "
                "count was stated, state the right one.")
        if gpu_product is not None and self.gpu_product:
            observed = normalize_gpu(gpu_product)
            if observed != self.gpu_product:
                raise UnknownHardware(
                    f"{self.instance_type} is recorded here as {self.gpu_product} but the leased "
                    f"node reports {observed!r}. The device total is measured per GPU product, so "
                    "the figure this run is pricing with describes a different device.")
        if driver_total_gib is not None and self.device_total_gib is not None:
            # A filed total may be a slope rather than a driver reading, and a slope legitimately
            # differs from the driver by its own uncertainty -- so the tolerance comes from the
            # figure's own error bar, floored, rather than from a constant chosen here.
            observed = float(driver_total_gib)
            unc = self.total_uncertainty_gib
            tol = max(DEVICE_TOL_FLOOR * observed, 0.0 if unc is None else 2.0 * float(unc))
            if abs(observed - float(self.device_total_gib)) > tol:
                raise UnknownHardware(
                    f"this run is pricing {self.gpu_product} at {self.device_total_gib:.3f} GiB per "
                    f"device (measured, {self.total_source or 'source unrecorded'}) and the leased "
                    f"node's driver reports {observed:.3f} GiB, which is further apart than "
                    f"{tol:.3f}. One of the two is not this card: either the sheet was measured on "
                    "different hardware filed under the same product name, or this node is not what "
                    "it claims. Every envelope on the node is a fraction of this number, so it is "
                    "not something to proceed past.")

    def with_driver_total(self, driver_total_gib: float, *, node: str = "") -> Hardware:
        """This machine with the driver's own addressable total adopted as the device total.

        Legal only when no measured total was known. A run that already had one calls `verify`
        instead: adopting there would silently re-size placements that have already been printed,
        and disagreement is a finding rather than an update.

        The figure is `measured` -- the driver read the device -- and its source is recorded as the
        driver. **Nothing cross-checks that the engine's own arithmetic uses it**, and after impl
        §14.4 nothing can: the cross-check was `solver/devicelaw.py` fitting `d(kv_gib)/d(util)` from
        the engine's KV report, which is the engine's account of its own internals and so is not
        evidence this design accepts. What replaced it is not a second reading but a different shape
        of answer -- a degree that does not boot is a witness, and the walk moves (impl §14.2).
        """
        if self.device_total_gib is not None:
            raise UnknownHardware(
                f"{self.instance_type} already carries a measured device total of "
                f"{self.device_total_gib:.3f} GiB. A second figure does not replace it silently; "
                "`verify` is what compares them.")
        # `replace` rather than a field-by-field rebuild, in all three of these. The rebuild was the
        # shape of a bug waiting to happen: a field added to this class is a field one of these
        # constructors silently drops back to its default, and the ones that matter most here -- the
        # three device facts that name the store directory -- would drop to "unknown" and re-key.
        return replace(self, device_total_gib=float(driver_total_gib), total_source="driver",
                       total_uncertainty_gib=DRIVER_TOTAL_QUANTUM_GIB, total_falsifiable=False,
                       shape_node=node or self.shape_node)

    def with_shape(self, *, gpu_count: int, gpu_product: str, node: str = "") -> Hardware:
        """This machine with an observed shape in place of a stated one. `verify` first."""
        return replace(self, gpu_count=int(gpu_count), gpu_product=normalize_gpu(gpu_product),
                       shape_source=SHAPE_OBSERVED, shape_node=node)

    def with_device_facts(self, *, addressable_mib: int, power_limit_w: int, p2p_topology: str,
                          node: str = "") -> Hardware:
        """This machine with the device facts a leased node reported.

        Adopted rather than checked, and that is the opposite of `verify`'s rule on the count and the
        total -- deliberately, because these facts are not claims this run made. They came out of the
        store's own record of the last node of this type, they were never used to size anything, and
        the node in hand is the better authority on all three. What they do decide is which directory
        this run's rows go in, so the *caller* compares them: `capture` files the new observation and
        stops the run when the address moved, so the next one reads and writes in one place rather than
        reading history from one directory and filing into another.
        """
        return replace(self, addressable_mib=int(addressable_mib), power_limit_w=int(power_limit_w),
                       p2p_topology=str(p2p_topology), shape_node=node or self.shape_node)

    def shape(self, *, observed_at: str = "", node: str = "", notes: dict | None = None
              ) -> InstanceShape:
        """This machine as a filable shape. Only meaningful once the shape has been observed."""
        if self.shape_source != SHAPE_OBSERVED:
            raise UnknownHardware(
                f"{self.instance_type}'s shape was {self.shape_source} and filing it would record a "
                "claim as an observation. `with_shape` after `verify` is what makes it filable.")
        if not self.environment_known:
            raise UnknownHardware(
                f"{self.instance_type} carries no device facts to file: a shape is the addressable "
                "total, the power limit and the link matrix as well as the count and the product, "
                "because it is what an offline `plan` computes this store's address from. "
                "`with_device_facts` is what puts them on it, from a capture's own observation.")
        return InstanceShape(instance_type=self.instance_type, gpu_count=self.gpu_count,
                             gpu_product=self.gpu_product,
                             addressable_mib=int(self.addressable_mib),
                             power_limit_w=int(self.power_limit_w),
                             p2p_topology=str(self.p2p_topology), observed_at=observed_at,
                             node=node or self.shape_node, notes=dict(notes or {}))


def normalize_gpu(name: str) -> str:
    """A GPU name to the store's key form: lowercase, vendor prefix off, spaces to dashes.

    The Karpenter label (`karpenter.k8s.aws/instance-gpu-name`) is the authority and this is the
    fallback for a node without one, because the two agree only by luck: "NVIDIA L40S" does reduce to
    `l40s`, but "Tesla T4" reduces to `tesla-t4` against a label of `t4`, and "NVIDIA
    A100-SXM4-40GB" carries a form factor and a memory size the label does not. It is applied to both
    sides of a comparison so that a match is still a match; a mismatch on a product this cannot
    normalize is reported with both strings rather than resolved here.
    """
    token = str(name).strip().lower()
    for prefix in ("nvidia ", "nvidia-"):
        if token.startswith(prefix):
            token = token[len(prefix):]
    return token.replace(" ", "-")


def hardware_for(instance_type: str, index: HardwareIndex, *,
                 gpu_count: int | None = None, gpu_product: str | None = None) -> Hardware:
    """Resolve an instance type against the hardware store, or say exactly what is missing.

    Two facts, and they fail differently, which is why they are reported differently. A missing
    *shape* is a machine nobody here has leased: anybody can supply it with `--gpu-count`, and the
    next capture verifies and files it. A missing *total* is a measurement nobody has taken, and
    nobody can supply it without booting an engine on the card -- so it is not refused here at all.
    It is carried as unknown, and refused at the point something tries to multiply a utilisation by
    it (`Hardware.require_total_gib`), which is the first moment a refusal can name what wanted it.

    `gpu_count` and `gpu_product` are the bootstrap, and they may not contradict the store. Once a
    shape has been observed, a stated one disagreeing with it means the operator or the store is
    wrong about the machine, and that is a finding rather than an override.
    """
    it = str(instance_type).strip()
    if not it:
        raise UnknownHardware("no instance type given")

    try:
        shape = index.read_shape(it)
    except ValueError as exc:
        # A shape is filed for this type, but from before the device facts were part of it, so it
        # cannot say what the card addressed, what it was allowed to draw or how the devices were
        # linked. Reported as an unknown shape rather than let out as a `ValueError`, because that is
        # what it is -- and because `UnknownHardware` is already routed to the remedy: the modes that
        # can lease a node go to `_bootstrap_environment`, which reads those three facts off a real
        # machine and files them, and the offline modes get a refusal that says so. Letting the
        # parse error through instead produced a traceback out of `main`, which reads as a crash
        # rather than as a store that needs one node held once.
        raise UnknownHardware(
            f"the shape filed for {it!r} cannot be read by this build: {exc} Run `profile` or "
            "`search`, which lease a node of this type, read those facts off it and re-file the "
            "shape; the offline modes cannot, because nothing offline can observe a machine.") \
            from exc
    stated_product = normalize_gpu(gpu_product) if gpu_product else ""

    if shape is not None:
        if gpu_count is not None and int(gpu_count) != shape.gpu_count:
            raise UnknownHardware(
                f"--gpu-count {int(gpu_count)} contradicts the hardware store, which has "
                f"{it} filed as {shape.gpu_count}x{shape.gpu_product} observed on node "
                f"{shape.node or 'unknown'}. The store's figure was read off a real machine, so it "
                "is not overridden from the command line: if it is wrong, the node it was taken "
                "from was not what it claimed and the shape should be deleted, not shadowed.")
        if stated_product and stated_product != shape.gpu_product:
            raise UnknownHardware(
                f"--gpu-product {stated_product!r} contradicts the hardware store, which has {it} "
                f"filed as carrying {shape.gpu_product!r}.")
        count, product = shape.gpu_count, shape.gpu_product
        shape_source, shape_node = SHAPE_OBSERVED, shape.node
        facts = (shape.addressable_mib, shape.power_limit_w, shape.p2p_topology)
    elif gpu_count is not None:
        count, product = int(gpu_count), stated_product
        shape_source, shape_node = SHAPE_STATED, ""
        # Not offered on the command line, and that is the point: these are the fields a fleet author
        # cannot know, so a bootstrap run has no address until it has held a node. See
        # `Hardware.environment`.
        facts = (None, None, "")
    else:
        known = sorted(s.instance_type for s in index.shapes())
        raise UnknownHardware(
            f"the hardware store holds no shape for {it!r}: nobody has leased one and read how many "
            "GPUs it has. This is not a table to add a row to -- it is an observation, and there are "
            "two ways to get it. Run a mode that leases a node of a type that IS filed, which files "
            "its shape on the way past; or, for this type, state the count once with "
            "`--gpu-count N` (and `--gpu-product` to price it against a card measured elsewhere), "
            "which is checked against the leased node before it can affect any measurement and "
            f"filed afterwards. Filed: {known or 'nothing yet'}")

    total = index.device_total(product) if product else None
    return Hardware(
        instance_type=it, gpu_count=count, gpu_product=product,
        device_total_gib=(None if total is None else total.total_gib),
        total_source=("" if total is None else total.source),
        total_uncertainty_gib=(None if total is None else total.uncertainty_gib),
        total_falsifiable=bool(total is not None and total.falsifiable),
        measured_under=(() if total is None else total.keys),
        shape_source=shape_source, shape_node=shape_node,
        addressable_mib=facts[0], power_limit_w=facts[1], p2p_topology=facts[2])
