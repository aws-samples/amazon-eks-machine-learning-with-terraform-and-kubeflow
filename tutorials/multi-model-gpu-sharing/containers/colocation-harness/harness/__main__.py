"""Harness entry point: one run, one observation record.

Sequence, in order, because the order is the design:

  1. Check every tenant is Ready. Failure here is `infeasible_deploy` -- distinguishing
     "the placement did not come up" from "the placement came up and could not serve the
     load" matters to the agent, and after load starts the two are no longer separable.
     Readiness used to require a 200 from the tenant's `/metrics` as well; that went with
     everything else that read a server's account of itself (impl §14.6).
  2. Start the 5s sampler and all load drivers.
  3. Wait for steady state, observing four signals rather than a timer. Settling time
     is a function of occupancy: a quiet placement settles in seconds, one near the
     knee takes far longer.
  4. Mark the window start, then extend the window until the ITL half-width meets the
     precision target.
  5. Stop issuing, drain in-flight requests, write the record.

Both waits have ceilings, and hitting one is a feasibility verdict, not a failed
measurement (design §8.2).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

import httpx

from . import engines
from .config import RunConfig
from .controller import Sampler, await_steady_state, await_window
from .load import LoadDriver, price_prompt
from .observation import (VERDICT_DEPLOY, VERDICT_MEASURED, VERDICT_NO_STEADY_STATE,
                          build_record, write_record)
from .workload import Workload

log = logging.getLogger("harness")


async def check_ready(client: httpx.AsyncClient, cfg: RunConfig) -> list[str]:
    """Return the names of tenants that are not serving. Empty means all good.

    One request per tenant, to the engine's health path. A second request to `/metrics` was
    part of this check until impl §14.6 -- it was a readiness proxy for a scrape that no
    longer happens, so requiring it would now refuse to start a run over an endpoint nothing
    reads.
    """
    bad = []
    for t in cfg.tenants:
        eng = engines.get(t.engine)
        try:
            h = await client.get(f"{t.base_url}{eng.health_path}", timeout=10.0)
            ok = h.status_code == 200
            if ok:
                # Some stacks answer 200 on a liveness path while still loading weights, so the
                # engine gets to say whether its own 200 means serving. vLLM's does, and its
                # `ready_ok` is the identity -- this costs nothing until an engine needs it.
                try:
                    ok = eng.ready_ok(h.json())
                except ValueError:
                    ok = eng.ready_ok(None)
            if not ok:
                bad.append(f"{t.name}: health={h.status_code}")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{t.name}: {type(exc).__name__}: {exc}")
    return bad


def build_workloads(cfg: RunConfig) -> dict:
    """One `Workload` per tenant, keyed by tenant name (impl §6.2).

    Built per *shape, tokenizer and image payload*, and shared by name, so a fleet whose tenants
    happen to agree on all three pays for one tokenizer load, one prompt construction and one set
    of rendered images. The sharing is by value, not an assumption: two tenants sharing an entry
    are driven with the byte-identical request, which is what holds prefill cost constant between
    them.

    **The tokenizer is part of the key, not a constant.** It used to be `cfg.tokenizer_path`, one
    per run, and then two tenants asking for 512 tokens got the same prompt whether or not they
    shared a vocabulary -- exact for one model, approximate for the other, and identical-looking
    in the record. Keyed this way, tenants that share a tokenizer still share the work, and
    tenants that do not each get a prompt that is exact for themselves.

    **The image payload is in the key for a sharper version of the same reason.** Two tenants at
    one ISL, one sending a megapixel image and one sending none, are not the same request and
    cannot share a calibration: the image one spends most of its prompt on visual tokens. Sharing
    them would calibrate once and report the result twice.

    **And the serving engine, for the same reason again.** Two stacks take the request at different
    paths with different fields; a shared workload would build one body and report it as both.
    """
    by_prompt: dict = {}
    out: dict = {}
    for t in cfg.tenants:
        # **The engine is in the key** for the same reason the tokenizer and the image payload are:
        # two tenants on two serving stacks are not sending the same request -- different path,
        # different fields, possibly no output pin at all -- so sharing one workload between them
        # would build one body and report it as both.
        key = (t.isl_tokens, t.osl_tokens, t.tokenizer_path, t.image_count, t.image_pixels,
               t.engine)
        if key not in by_prompt:
            by_prompt[key] = Workload(t.isl_tokens, t.osl_tokens, t.tokenizer_path,
                                      image_count=t.image_count, image_pixels=t.image_pixels,
                                      engine=t.engine)
        out[t.name] = by_prompt[key]
    return out


async def calibrate_workloads(client: httpx.AsyncClient, cfg: RunConfig,
                              workloads: dict) -> list[str]:
    """Probe each distinct workload until the server agrees with its stated ISL.

    Runs after the readiness check and before any load, because it needs a live engine and
    because its requests must not land inside a measurement window. Distinct workloads are
    calibrated once each -- `build_workloads` already shares them by value -- so a fleet of
    five tenants with three distinct shapes pays for three calibrations.

    Returns the tenants whose ISL could not be settled. That is a run-blocking condition and
    not a warning: an approximate input length makes prefill cost a hidden variable, which is
    the one thing this module exists to prevent (impl §6.2).
    """
    done: dict[int, str] = {}
    bad: list[str] = []
    for t in cfg.tenants:
        w = workloads[t.name]
        if id(w) in done:
            log.info("workload %s: calibrated with %s", t.name, done[id(w)])
            continue

        async def price(payload, url=f"{t.base_url}{w.endpoint}"):
            return await price_prompt(client, url, payload)

        try:
            await w.calibrate(price, t.served_name)
        except Exception as exc:  # noqa: BLE001 -- a failure here is a verdict, not a crash
            bad.append(f"{t.name}: {type(exc).__name__}: {exc}")
            continue
        done[id(w)] = t.name
        log.info("workload %s: %s", t.name, w.describe())
        if not w.isl_exact:
            bad.append(f"{t.name}: isl_tokens={w.isl_tokens} not achieved "
                       f"(server said {w.isl_achieved} after {len(w.probes)} probe(s))")
    return bad


async def run(cfg: RunConfig) -> int:
    # Built offline (tokenizer, images), calibrated against the live engine below. The two
    # phases are separate because only the second needs a server, and it must not run until
    # the readiness check has passed.
    workloads = build_workloads(cfg)

    # No connection-pool ceiling: a pool limit would silently reintroduce closed-loop
    # behaviour by queueing requests client-side, which is the one failure mode this
    # harness cannot tolerate (impl §6.2).
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=None)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        bad = await check_ready(client, cfg)
        if bad:
            log.error("tenants not serving: %s", bad)
            record = _empty_record(cfg, notes={"deploy_failures": bad})
            write_record(record, cfg.out_dir, cfg.validity_key_hash, cfg.run_id)
            return 2

        # ISL is settled here, against the engine that will serve the run, before any load
        # exists to perturb it. A failure is filed as a deploy failure rather than as a new
        # verdict: the four verdicts are the agent's vocabulary, and what has happened is
        # that this placement cannot be driven at the shape that was asked for -- which is
        # the same class of answer as "it did not come up", and is emphatically not a
        # latency measurement of anything.
        unsettled = await calibrate_workloads(client, cfg, workloads)
        if unsettled:
            log.error("ISL could not be settled: %s", unsettled)
            record = _empty_record(cfg, notes={"calibration_failures": unsettled})
            write_record(record, cfg.out_dir, cfg.validity_key_hash, cfg.run_id)
            return 2

        drivers = {t.name: LoadDriver(
            t.name, f"{t.base_url}{workloads[t.name].endpoint}",
            # Bound per tenant, and the request index is passed through so successive
            # requests rotate the image variant (`workload.ImagePool`).
            (lambda w, served: (lambda i: w.payload(served, i)))(
                workloads[t.name], t.served_name),
            t.rate_hz) for t in cfg.tenants}
        # No tenant URLs and no per-tenant engine: the sampler reads this process's own drivers and
        # the DCGM exporter, and nothing else (impl §14.6). The engine still decides how each request
        # is shaped and how readiness is checked -- that is `workloads` and the probe above.
        sampler = Sampler(cfg.exporter_url, drivers)

        tasks = [asyncio.create_task(sampler.run(client))]
        tasks += [asyncio.create_task(d.run(client)) for d in drivers.values()]
        log.info("driving %d tenants at %s req/s", len(drivers),
                 {t.name: t.rate_hz for t in cfg.tenants})

        verdict = VERDICT_MEASURED
        notes: dict = {}
        # Bound before the try only so the cleanup path is well defined. An
        # unexpected exception here propagates and the Job fails with no record,
        # deliberately: a harness bug is not one of the four verdicts, and filing it
        # as `infeasible_deploy` would put a fiction in the agent's store.
        settled, ss_s, ss_detail = False, None, {}
        converged, win_detail = False, {}
        w0 = w1 = time.perf_counter()
        try:
            settled, ss_s, ss_detail = await await_steady_state(
                sampler, cfg.stability, cfg.steady_state_ceiling_s)
            if not settled:
                verdict = VERDICT_NO_STEADY_STATE
                notes["ceiling"] = "steady_state"

            # The window is measured either way. A placement that never settled still
            # produced metrics, and throwing them away loses the evidence for *why*
            # it is infeasible while the verdict already records that it is.
            w0 = time.perf_counter()
            converged, win_s, win_detail = await await_window(
                drivers, w0, cfg.itl_target_halfwidth_s,
                cfg.window_min_s, cfg.window_ceiling_s)
            if not converged and verdict == VERDICT_MEASURED:
                verdict = VERDICT_NO_STEADY_STATE
                notes["ceiling"] = "window"
            notes.setdefault("ceiling", "none")
            w1 = time.perf_counter()
            log.info("window closed after %.1fs (converged=%s)", win_s, converged)
        finally:
            for d in drivers.values():
                d.stop()
            sampler.stop()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        await asyncio.gather(*(d.drain(cfg.drain_timeout_s) for d in drivers.values()))

        record = build_record(cfg, drivers, sampler, w0, w1, verdict=verdict,
                             steady_state_s=ss_s if settled else None,
                             steady_state_detail=ss_detail,
                             window_detail=win_detail, notes=notes)
        # What each tenant's prompt actually came out as. Per tenant, because whether ISL was hit
        # exactly is a property of the prompt that was built, and two tenants can differ.
        for name, w in workloads.items():
            record["workload"]["tenants"][name].update(w.describe())
        path = write_record(record, cfg.out_dir, cfg.validity_key_hash, cfg.run_id)

        log.info("verdict=%s -> %s", record["verdict"], path)
        for name, r in record["per_tenant"].items():
            log.info("  %s: delivered=%s itl_mean_ms=%s in_flight=%s->%s", name,
                     _fmt(r.get("delivered_fraction")), _fmt(r.get("itl_mean_ms")),
                     _fmt(r.get("in_flight_first_quarter_mean")),
                     _fmt(r.get("in_flight_last_quarter_mean")))
        # Exit 0 for any recorded verdict including an infeasible one: the Job
        # succeeded in producing an observation. Non-zero means no record.
        return 0


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _empty_record(cfg: RunConfig, notes: dict) -> dict:
    return {
        "run_id": cfg.run_id,
        "validity_key_hash": cfg.validity_key_hash,
        "node": cfg.node,
        "verdict": VERDICT_DEPLOY,
        "steady_state_reached_s": None,
        "window_s": 0.0,
        "configuration": {"tenants": [
            {"name": t.name, "model_id": t.model_id, "tp": t.tp,
             "envelope_gib": t.envelope_gib, "rate_hz": t.rate_hz,
             "isl_tokens": t.isl_tokens, "osl_tokens": t.osl_tokens,
             "image_count": t.image_count, "image_pixels": t.image_pixels,
             "gpu_uuids": t.gpu_uuids} for t in cfg.tenants]},
        "per_tenant": {}, "per_gpu": {},
        "provenance": cfg.provenance, "notes": notes,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="harness", description="Co-location measurement harness")
    p.add_argument("--config", required=True, help="run config YAML")
    p.add_argument("--run-id", help="override run_id from the config")
    p.add_argument("--out-dir", help="override the observation store root")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout)
    # httpx logs a line per request at INFO. At 8 req/s over a five-minute run that is
    # ~2400 lines burying the four that matter, and it grows with the load being
    # applied -- so the noise is worst exactly when the run is most interesting.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    cfg = RunConfig.load(args.config)
    if args.run_id:
        cfg.run_id = args.run_id
    if args.out_dir:
        cfg.out_dir = args.out_dir

    return asyncio.run(run(cfg))


if __name__ == "__main__":
    raise SystemExit(main())
