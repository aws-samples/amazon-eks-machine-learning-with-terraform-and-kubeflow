"""What the driver reports about a device, and the one subtraction that makes it the right number.

**`memory.total` is the board, not the budget.** The driver keeps a region of every device for itself
-- page-retirement tables, ECC structures, firmware -- and `nvidia-smi` reports it separately as
`memory.reserved`. No CUDA context can address it, so vLLM's `gpu_memory_utilization` multiplies
`memory.total - memory.reserved`, and anything that multiplies `memory.total` hands every engine on
the node slightly more memory than the packing priced. The error has no symptom: nothing fails, the
placement simply is not the placement that was scored, and the first sign is an OOM in some later
co-resident run.

This module exists because that distinction was got wrong in **two** places independently, and fixing
one of them left the other to contradict it (2026-09-14). `capture` filed the device total
for the planner; the deleted `agent/booter.py` wrote it into every boot document, and with the
subtraction in only the first the two disagreed by the reserve on the same node in the same run.

**There is one caller today** -- `capture._reserved_gib` -- and the module stays separate anyway. The
defect was never that two callers existed; it was that the subtraction was written twice. Anything
that reaches a node's GPUs and needs the addressable figure calls this, and the next such caller is
the second engine's control adapter.
"""

from __future__ import annotations

import logging

from .deploy import DeployError, _run

log = logging.getLogger(__name__)

# `nounits` so the parse is integers rather than "46068 MiB", and no header so every line is a device.
SMI_FORMAT = "--format=csv,noheader,nounits"

# Asked on its own, never appended to another query, and that is a decision rather than a style: a
# driver too old to know the field fails the WHOLE query it appears in. Bundling it into the
# used/total read would cost a run that has already leased a node its overhead measurement -- and
# would do it in the one place that cannot be retried, since the two samples bracket a boot.
RESERVED_QUERY = "--query-gpu=memory.reserved"


def reserved_gib(exec_prefix, *, where: str, timeout_s: float = 120.0) -> float:
    """The largest per-device reserve on the node, in GiB. Zero when the driver will not say.

    `exec_prefix` is the argv up to and including `--`, i.e. whatever gets a command into a container
    that can see the GPUs; the caller owns that because the callers reach the node differently.

    Measured 2026-09-14, and it is why this function exists: a fit of a card's total from three
    engine start-ups, as `d(kv_gib)/d(util)`, came out tens of times its own uncertainty BELOW the
    driver's `memory.total` -- and exactly, to under a thousandth of a GiB, `memory.total -
    memory.reserved` on the same node (611 MiB on an L40S). Two independent instruments, one of them
    the engine itself, agreed on the addressable figure; only the board figure disagreed with both.
    The fit is gone (impl §14.4); the reading it corroborated is what this returns.

    **Degrades rather than refuses.** An unreadable or unsupported field returns 0.0 with a warning,
    which reverts to the previous behaviour: the board total, which is generous. That is worth saying
    plainly -- a run that reverts is over-subscribing every GPU by whatever the driver reserves, and
    nothing downstream will notice. It is still the better failure: stranding a lease over a soft
    metadata field costs the same GPU time and measures nothing.

    **When devices disagree the LARGEST reserve wins**, because the alternative is an envelope some
    device on the node cannot honour.
    """
    try:
        # `_run` looked up on the module at call time, not captured as a default argument. A default
        # binds at import and cannot be monkeypatched, which is not a testing nicety here: the only
        # thing standing between this function and the defect it exists to fix is a self-test that
        # replaces the exec, and with a bound default that test passed while reading a board total.
        out = _run([*exec_prefix, "nvidia-smi", RESERVED_QUERY, SMI_FORMAT], timeout_s=timeout_s)
    except DeployError as exc:
        log.warning("nvidia-smi on %s cannot report memory.reserved (%s), so the device total is the "
                    "board total and over-subscribes every GPU by however much the driver reserves",
                    where, exc)
        return 0.0
    reserves = []
    for row in out.splitlines():
        text = row.strip()
        if not text:
            continue
        try:
            reserves.append(float(text) / 1024.0)
        except ValueError:
            # `[N/A]` is what an unsupported field looks like on a driver new enough to accept the
            # query and old enough not to answer it. Treated as unknown, not as zero reserve.
            log.warning("nvidia-smi reported memory.reserved as %r on %s, which is what an "
                        "unsupported field looks like; treating the reserve as unknown and using "
                        "the board total", text, where)
            return 0.0
    if not reserves:
        return 0.0
    if len(set(reserves)) != 1:
        log.warning("the GPUs of %s reserve different amounts (%s GiB); taking the largest, so the "
                    "device total is one every device on the node can honour",
                    where, sorted(set(reserves)))
    return max(reserves)


def reserved_mib(exec_prefix, *, where: str, timeout_s: float = 120.0) -> int:
    """`reserved_gib` in the units `nvidia-smi` prints, for callers that work in MiB.

    Rounded UP, so a total derived by subtracting it is never larger than the addressable one. A MiB
    in the generous direction is what the whole finding above was about.
    """
    import math

    return int(math.ceil(reserved_gib(exec_prefix, where=where, timeout_s=timeout_s) * 1024.0))
