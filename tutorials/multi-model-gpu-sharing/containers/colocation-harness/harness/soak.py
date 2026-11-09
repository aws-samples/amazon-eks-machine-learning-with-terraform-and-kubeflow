"""Soak: drive a deployment for whole clock hours and report what the client saw, hour by hour.

Not a measurement. `python -m harness` measures one placement for one window and writes an
observation record, and that is the only thing the search reads. This drives a deployment that is
already serving, from outside it, so its own Prometheus metrics -- which `colocation.kai drift`
reads -- describe traffic that arrived through the Service like real traffic does. It reuses the
run's instrument rather than a second one: the same calibrated prompts (`Workload`) and the same
open-loop driver (`LoadDriver`), so a difference from what the search certified is the
deployment's, not the client's.

It writes nothing to the store. Each whole clock hour it logs one `SOAK HOUR` JSON line per tenant
with the client's own figures, which set beside `drift`'s show the network hop between them:

  python -m harness.soak --config run.yaml --hours 2

runs until the end of the second whole clock hour after it starts, plus `TAIL_S`, so the last
hour's requests land inside the hour that `drift` records.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time

import httpx

from .__main__ import build_workloads, calibrate_workloads, check_ready
from .config import RunConfig
from .load import LoadDriver

log = logging.getLogger("harness.soak")

HOUR_S = 3600
# Past the last whole hour, so the requests issued in its final seconds complete inside it.
TAIL_S = 120


def _p95(xs: list) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]


def _mean(xs: list) -> float | None:
    return sum(xs) / len(xs) if xs else None


def hour_summary(name: str, driver: LoadDriver, start_t: float, end_t: float,
                 wall_start: float) -> dict:
    """The client's view of one tenant over one hour, selected on dispatch time as a window is."""
    ok = driver.completed_in(start_t, end_t)
    launched = driver.launched_in(start_t, end_t)

    def ms(v):
        return None if v is None else round(v * 1000, 3)

    return {
        "tenant": name, "hour_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wall_start)),
        "launched": launched, "completed": len(ok), "not_completed": launched - len(ok),
        "rate_hz": round(launched / HOUR_S, 4),
        "isl_tokens": _mean([r.prompt_tokens for r in ok if r.prompt_tokens is not None]),
        "osl_tokens": _mean([r.tokens for r in ok]),
        "itl_mean_ms": ms(_mean([r.itl_mean_s for r in ok if r.itl_mean_s is not None])),
        "ttft_p95_ms": ms(_p95([r.ttft_s for r in ok if r.ttft_s is not None])),
        "schedule_lag_max_ms": ms(max((r.schedule_lag_s for r in ok), default=None)),
    }


async def soak(cfg: RunConfig, hours: int) -> int:
    workloads = build_workloads(cfg)
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=None)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        bad = await check_ready(client, cfg)
        if bad:
            log.error("tenants not serving: %s", bad)
            return 2
        unsettled = await calibrate_workloads(client, cfg, workloads)
        if unsettled:
            log.error("ISL could not be settled: %s", unsettled)
            return 2

        now = time.time()
        first = math.ceil(now / HOUR_S) * HOUR_S
        end = first + hours * HOUR_S + TAIL_S
        # perf_counter is what the driver stamps; wall-clock hours are what `drift` records.
        offset = time.time() - time.perf_counter()
        drivers = {t.name: LoadDriver(
            t.name, f"{t.base_url}{workloads[t.name].endpoint}",
            (lambda w, served: (lambda i: w.payload(served, i)))(workloads[t.name], t.served_name),
            t.rate_hz, max_requests=int(t.rate_hz * (end - now) * 1.1) + 100)
            for t in cfg.tenants}
        tasks = [asyncio.create_task(d.run(client)) for d in drivers.values()]
        log.info("driving %d tenants until %s (%d whole hour(s) from %s): %s", len(drivers),
                 time.strftime("%H:%M:%SZ", time.gmtime(end)), hours,
                 time.strftime("%H:%M:%SZ", time.gmtime(first)),
                 {t.name: t.rate_hz for t in cfg.tenants})
        try:
            for h in range(hours):
                # Each hour is reported once its requests have had TAIL_S to finish.
                hour_end = first + (h + 1) * HOUR_S
                await asyncio.sleep(max(0.0, hour_end + TAIL_S - time.time()))
                for name, d in drivers.items():
                    s = hour_summary(name, d, hour_end - HOUR_S - offset, hour_end - offset,
                                     hour_end - HOUR_S)
                    print("SOAK HOUR " + json.dumps(s, sort_keys=True), flush=True)
        finally:
            for d in drivers.values():
                d.stop()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(d.drain(cfg.drain_timeout_s) for d in drivers.values()))
        failed = {n: sum(1 for r in d.results if not r.ok) for n, d in drivers.items()}
        log.info("done; failed requests per tenant: %s", failed)
        return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="harness.soak", description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="run config YAML, as for `python -m harness`")
    p.add_argument("--hours", type=int, required=True, help="whole clock hours to drive")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    if args.hours < 1:
        p.error("--hours must be at least 1")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return asyncio.run(soak(RunConfig.load(args.config), args.hours))


if __name__ == "__main__":
    raise SystemExit(main())
