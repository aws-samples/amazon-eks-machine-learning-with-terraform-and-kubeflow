"""Direct scraping of the DCGM exporter (impl §6.5).

Deliberately not via the cluster Prometheus. That instance stored its TSDB on an
emptyDir, so a pod restart destroyed every sample and its nominal retention was
fiction. Independently of whether that gets fixed, a measurement whose cause signals
live only in someone else's database is not reproducible -- so the harness scrapes
and writes its own artifacts.

**One endpoint, and it is not a serving engine's.** `scrape_tenant` was here until impl
§14.6, reading each tenant's `/metrics` for its queue depth, KV occupancy, preemption
counter and KV pool size. It is gone, with the engine metric vocabulary it needed: the
harness does not read a server's account of its own internals, for any purpose including
provenance, and the question those gauges answered is answered instead from the driver's
own in-flight level (`controller.backlog_series`). DCGM stays because it is the *driver's*
account of the device -- published identically whatever process holds the GPU, so it
cannot make one engine's rows judgeable and another's not.
"""

from __future__ import annotations

import logging

import httpx
from prometheus_client.parser import text_string_to_metric_families

log = logging.getLogger(__name__)

# 17 fields. The list that decides what is PUBLISHED lives in the mps-daemon chart
# (charts/.../mps-daemon/values.yaml, `dcgmExporter.customMetrics`) and this one decides
# what is READ; a self-test compares them, because they had already drifted and nothing
# said so. `scrape_dcgm` filters by name, so a field named here and not published is
# simply absent from every record's per_gpu block, silently and forever. It had happened
# to DCGM_FI_DEV_GPU_UTIL and DCGM_FI_DEV_MEM_COPY_UTIL, while the exporter published
# DCGM_FI_DEV_FB_TOTAL and DCGM_FI_DEV_GPU_TEMP that nothing here read. Both lists were
# 15 long, which is how it stayed invisible. Fixed 2026-09-04 by taking the union.
#
# The throttle group is mandatory: a clock drop caused by a power cap is indistinguishable
# from one caused by co-location unless the cap is recorded, and this project has already
# paid for one misattribution of exactly that shape (impl §6.5, §4.2.1).
DCGM_FIELDS = (
    "DCGM_FI_PROF_SM_ACTIVE",
    "DCGM_FI_PROF_SM_OCCUPANCY",
    "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "DCGM_FI_PROF_DRAM_ACTIVE",
    "DCGM_FI_PROF_PCIE_TX_BYTES",
    "DCGM_FI_PROF_PCIE_RX_BYTES",
    "DCGM_FI_DEV_SM_CLOCK",
    "DCGM_FI_DEV_POWER_USAGE",
    "DCGM_FI_DEV_POWER_VIOLATION",
    "DCGM_FI_DEV_THERMAL_VIOLATION",
    "DCGM_FI_DEV_GPU_UTIL",
    "DCGM_FI_DEV_MEM_COPY_UTIL",
    "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_FB_USED",
    "DCGM_FI_DEV_FB_FREE",
    "DCGM_FI_DEV_FB_TOTAL",
)


async def scrape_dcgm(client: httpx.AsyncClient, exporter_url: str) -> dict:
    """Return {gpu_uuid: {field: value}} for the pinned fields.

    Keyed by UUID, never by pod. The exporter's pod attribution works by asking the
    device plugin who owns each GPU, and tenants deliberately bypass the plugin
    (impl §4.3). Measured, that lookup does not fail cleanly -- it returns whichever
    pod holds the plugin allocation, so 64 series were attributed with total
    confidence to a pod doing no GPU work. A harness joining on `pod` would have
    credited every co-location effect to it. The exporter therefore runs with
    --kubernetes=false, and the harness joins on UUID, which it knows from each
    tenant's own values file and is correct by construction.

    The limitation this leaves: a per-GPU signal answers "did this device
    saturate", never "which co-tenant saturated it".
    """
    by_uuid: dict[str, dict] = {}
    try:
        resp = await client.get(exporter_url, timeout=15.0)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.warning("DCGM scrape failed: %s: %s", type(exc).__name__, exc)
        return by_uuid

    wanted = set(DCGM_FIELDS)
    for fam in text_string_to_metric_families(resp.text):
        for s in fam.samples:
            if s.name not in wanted:
                continue
            uuid = s.labels.get("UUID") or s.labels.get("uuid")
            if not uuid:
                continue
            by_uuid.setdefault(uuid, {})[s.name] = s.value
    return by_uuid
