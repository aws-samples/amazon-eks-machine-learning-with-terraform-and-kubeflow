"""Open-loop load generation (impl §6.2).

Requests are placed on the wire on a fixed-rate schedule computed from t0, with no
coupling to whether earlier requests have completed. This is the whole design:

A closed-loop generator ("send, await, send") silently becomes a throughput
measurement when the server slows down -- it stops offering the rate it was asked
to offer, so overload is invisible in the load it applies. Backlog is the signal
this harness exists to see, so the schedule must not bend.

Two consequences of that, both deliberate:
  - Schedule times come from `t0 + i * interval`, never from "sleep(interval)" in a
    loop, which would accumulate the dispatch cost of every prior request as drift.
  - `schedule_lag` is recorded per request. If the driver itself cannot keep up, the
    measurement is invalid for reasons that have nothing to do with the server, and
    that must be visible rather than inferred.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)


@dataclass
class RequestResult:
    tenant: str
    dispatch_t: float          # perf_counter at the moment the request was issued
    scheduled_t: float         # when it was *supposed* to be issued
    ok: bool
    chunks: int = 0            # SSE chunks the client observed carrying text
    usage_tokens: int | None = None   # the server's own completion_tokens count
    prompt_tokens: int | None = None  # the server's own prompt_tokens count
    ttft_s: float | None = None
    itl_mean_s: float | None = None
    itl_from_gaps_s: float | None = None
    complete_t: float | None = None
    error: str | None = None
    itl_gaps: list[float] = field(default_factory=list)

    @property
    def schedule_lag_s(self) -> float:
        return self.dispatch_t - self.scheduled_t

    @property
    def tokens(self) -> int:
        """Delivered tokens. The server's count when available, else observed chunks.

        The distinction is not academic. The first corrected run had 98 of 120
        requests report 1-5 fewer chunks than the OSL that `min_tokens` +
        `ignore_eos` pins, which dragged the delivered fraction to 0.9857 and tripped
        the feasibility gate on a placement that was not remotely overloaded. Chunk
        coalescing was ruled out by measurement (128 chunks even with a 20ms stall per
        chunk), so the loss is somewhere in the client's own observation path -- which
        is exactly why the gate must not depend on it. The server's count is ground
        truth; the chunk count is kept alongside it so the divergence stays visible.
        """
        return self.chunks if self.usage_tokens is None else self.usage_tokens


def _chunk_carries_text(obj: dict) -> bool:
    """Whether this SSE chunk delivered a token.

    Two endpoints, two shapes. `/v1/completions` puts the token in `choices[0].text`;
    `/v1/chat/completions` -- the only route that accepts an image -- puts it in
    `choices[0].delta.content`. Counting only the first made every multimodal request
    look like it delivered nothing, which reads as a delivery failure rather than as a
    parser that was never taught the other shape.
    """
    choices = obj.get("choices") or []
    if not choices:
        return False
    c = choices[0]
    if c.get("text"):
        return True
    delta = c.get("delta") or {}
    return bool(delta.get("content"))


async def price_prompt(client: httpx.AsyncClient, url: str, payload: dict) -> int | None:
    """One non-streaming request, returning the server's `usage.prompt_tokens`.

    This is the whole of `workload.calibrate`'s instrument. It is deliberately not a
    streaming request and deliberately not counted anywhere: it runs before the load
    drivers start, so it is not part of any window, and its only output is a length.
    """
    resp = await client.post(url, json=payload, timeout=120.0)
    if resp.status_code != 200:
        raise RuntimeError(f"probe HTTP {resp.status_code}: {resp.text[:300]}")
    usage = (resp.json() or {}).get("usage") or {}
    got = usage.get("prompt_tokens")
    return None if got is None else int(got)


async def issue(client: httpx.AsyncClient, url: str, payload: dict, tenant: str,
                scheduled_t: float) -> RequestResult:
    """Issue one streaming completion and time every chunk arrival.

    Chunks are counted as tokens. vLLM emits one token per SSE chunk for
    non-speculative decoding, which is the configuration under test, but the count
    is reported as `chunks` rather than `tokens` so the assumption stays visible if
    that ever changes.
    """
    dispatch_t = time.perf_counter()
    res = RequestResult(tenant=tenant, dispatch_t=dispatch_t, scheduled_t=scheduled_t, ok=False)
    arrivals: list[float] = []

    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread())[:200].decode("utf-8", "replace")
                res.error = f"HTTP {resp.status_code}: {body}"
                return res
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    res.usage_tokens = obj["usage"].get("completion_tokens")
                    # The server's own prompt length, for the request that was actually
                    # driven. `workload.calibrate` settles this before the run against a
                    # probe; carrying it per request is the check that the driven payload
                    # is the one that was calibrated, not merely one built the same way.
                    res.prompt_tokens = obj["usage"].get("prompt_tokens")
                if _chunk_carries_text(obj):
                    arrivals.append(time.perf_counter())
        res.complete_t = time.perf_counter()
        res.chunks = len(arrivals)
        if arrivals:
            res.ttft_s = arrivals[0] - dispatch_t
            gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
            res.itl_gaps = gaps
            # Mean, not p95 -- design §1.8. A tail statistic over a single request's
            # gaps measures scheduler jitter, not the quantity being modelled.
            #
            # Computed as elapsed / (tokens - 1) using the SERVER's token count, not as
            # the mean of observed gaps. If the client misses a chunk, two gaps merge
            # into one and the mean-of-gaps rises -- so that estimator degrades exactly
            # when the run is under load. Span over server-counted intervals does not:
            # a missed chunk leaves the span unchanged and the divisor correct.
            span = arrivals[-1] - arrivals[0]
            n = res.tokens
            res.itl_mean_s = span / (n - 1) if n and n > 1 else None
            res.itl_from_gaps_s = sum(gaps) / len(gaps) if gaps else None
        res.ok = True
    except Exception as exc:  # noqa: BLE001 -- any failure is a datum, not a crash
        res.error = f"{type(exc).__name__}: {exc}"
    return res


class LoadDriver:
    """Drives one tenant at a fixed rate until stopped."""

    def __init__(self, tenant: str, url: str, payload_for, rate_hz: float,
                 max_requests: int = 200_000):
        self.tenant = tenant
        self.url = url
        # A factory, not a body. Successive requests must be allowed to differ in their
        # image bytes while costing an identical amount of work: a fixed body would send
        # one byte-identical image forever, and any engine-side hash of multimodal input
        # would then serve the vision tower's result out of a cache. The request index is
        # passed in so what varies is chosen here rather than by a counter hidden inside.
        self.payload_for = payload_for
        self.rate_hz = rate_hz
        self.max_requests = max_requests
        self.results: list[RequestResult] = []
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self.t0: float | None = None
        self.issued = 0
        self.hit_cap = False
        # Maintained by a done-callback rather than recomputed by walking _tasks. The
        # sampler reads these many times a second and the task list grows with the
        # run, so an O(n) read would make the sampler miss its cadence at exactly the
        # rates where accurate sampling matters most.
        self._in_flight = 0
        self.delivered_chunks = 0
        self.completed_ok = 0
        # Scheduled dispatch time of every request this driver started, appended at launch and
        # never removed. `dispatched_in` cannot serve as the count of offered load, because it
        # walks the task list and skips anything not `done()` -- so a window in which every
        # request was still streaming when the drain timed out reports **zero requests
        # dispatched**, which is the one case where the count matters most. That is not a
        # cosmetic miscount: it is also the null denominator behind a null `delivered_fraction`.
        # This list is the driver's own books, so it counts a request that was sent and never
        # came back, which is the only counter an overload argument can rest on.
        self._launched: list[float] = []

    async def run(self, client: httpx.AsyncClient) -> None:
        interval = 1.0 / self.rate_hz
        self.t0 = time.perf_counter()
        i = 0
        while not self._stop.is_set():
            scheduled = self.t0 + i * interval
            delay = scheduled - time.perf_counter()
            if delay > 0:
                try:
                    # Waiting on the stop event rather than sleeping keeps shutdown
                    # prompt at low rates, where a plain sleep would overshoot the
                    # window by up to one interval.
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    break
                except asyncio.TimeoutError:
                    pass
            if self._stop.is_set():
                break
            if i >= self.max_requests:
                # A safety stop, not a design limit. Recorded so a truncated run is
                # never mistaken for a complete one.
                self.hit_cap = True
                log.error("%s: request cap %d reached, stopping issue",
                          self.tenant, self.max_requests)
                break
            task = asyncio.create_task(
                issue(client, self.url, self.payload_for(i), self.tenant, scheduled))
            self._in_flight += 1
            task.add_done_callback(self._on_done)
            self._tasks.append(task)
            self._launched.append(scheduled)
            self.issued += 1
            i += 1

    def _on_done(self, task: asyncio.Task) -> None:
        self._in_flight -= 1
        try:
            r = task.result()
        except Exception:  # noqa: BLE001 -- cancelled or raised; nothing to count
            return
        if r.ok:
            self.delivered_chunks += r.tokens
            self.completed_ok += 1

    def stop(self) -> None:
        self._stop.set()

    async def drain(self, timeout_s: float) -> None:
        """Await in-flight requests after the window closes.

        Requests dispatched inside the window but still streaming carry real
        latency data; abandoning them would bias the sample toward fast requests,
        which under overload is exactly the wrong direction.
        """
        if not self._tasks:
            return
        done, pending = await asyncio.wait(self._tasks, timeout=timeout_s)
        for t in done:
            try:
                self.results.append(t.result())
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: task raised: %s", self.tenant, exc)
        if pending:
            log.warning("%s: %d requests still in flight after %.0fs drain; cancelling",
                        self.tenant, len(pending), timeout_s)
            for t in pending:
                t.cancel()

    def in_flight(self) -> int:
        return self._in_flight

    def completed_in(self, start_t: float, end_t: float) -> list[RequestResult]:
        """Requests *dispatched* inside the window that finished successfully.

        Selecting on dispatch time rather than completion time is what keeps the
        offered load and the measured sample describing the same interval.
        """
        out = []
        for t in self._tasks:
            if not t.done() or t.cancelled():
                continue
            try:
                r = t.result()
            except Exception:  # noqa: BLE001
                continue
            if r.ok and start_t <= r.dispatch_t < end_t:
                out.append(r)
        return sorted(out, key=lambda r: r.dispatch_t)

    def launched_in(self, start_t: float, end_t: float) -> int:
        """Requests this driver started in the window, whatever became of them.

        Selected on *scheduled* dispatch time rather than actual, because a request that never
        returned has no actual one to select on -- and this counter exists precisely to count
        those. The two differ by the driver's schedule lag, which is recorded alongside so a
        reader can see how far apart they could be; at the rates in play it is milliseconds.

        Never a substitute for `dispatched_in` in the delivery gate. That denominator is
        deliberately "requests we hold a result for", and widening it here would move a
        calibrated threshold as a side effect of adding an observation.
        """
        return sum(1 for t in self._launched if start_t <= t < end_t)

    def dispatched_in(self, start_t: float, end_t: float) -> int:
        n = 0
        for t in self._tasks:
            if t.done() and not t.cancelled():
                try:
                    r = t.result()
                except Exception:  # noqa: BLE001
                    continue
                if start_t <= r.dispatch_t < end_t:
                    n += 1
        return n
