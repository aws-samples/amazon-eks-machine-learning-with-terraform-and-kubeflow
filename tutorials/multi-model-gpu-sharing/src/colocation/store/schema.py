"""Typed views over what the harness wrote (impl §8).

The harness produces JSON; the agent and the solver read these objects.
This module is the seam design §4.1 calls "the schema": everything else is written against
it, so it is the right place to enforce the rules that must not be re-litigated downstream.

Three such rules, each one earned:

  - **A record whose delivery gate failed has no latency, and asking for it raises.** The
    harness deliberately omits latency in that case (survivorship bias, impl §6.3). If this
    layer returned `None` instead, the value would flow into arithmetic and become a zero,
    and an overloaded placement would score as the fastest one in the store.
  - **A record sampled without sub-sampling is quarantined, not consumed.** Live run 2 read
    occupancy of exactly 0.0 on every sample while DCGM reported the GPU 47% busy, because
    the 5s sampler was phase-locked to the arrival schedule. That record looked perfectly
    healthy. `sampling.subsamples_per_sample_mean` is what distinguishes it, and checking it
    here is the only reason the harness records it.

    This survived the cut to a one-signal steady-state gate, and it is the *client-side*
    series it protects: the count is the driver loop's own iterations (`controller.Sampler`),
    and the in-flight mean the gate judges is an average over exactly those iterations. One
    read per bucket can sit at a fixed phase against a fixed-rate arrival schedule and come
    out constant, and constant reads as settled. Nothing here reads an engine's own numbers,
    and since impl §14.6 there is nothing to read: the block used to carry an
    `engine_reads_per_sample_mean` alongside this count, recorded and never judged, and it went
    with the scrape it counted. An engine that publishes nothing is measured like any other
    because every engine is now treated as publishing nothing (impl §6.4).
  - **An unknown verdict is an error, not an unknown.** The four verdicts are a closed set.
    A fifth string means the record was written by something this code does not understand,
    and guessing is worse than stopping.

Pure standard library, deliberately. This runs on the ML Ops desktop and on the solver's
critical path, both CPU-only and deterministic (impl §9), so there is nothing here that
needs numpy and nothing that should drag it in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VERDICT_MEASURED = "measured"
VERDICT_DELIVERY = "infeasible_delivery"
VERDICT_NO_STEADY_STATE = "infeasible_no_steady_state"
VERDICT_DEPLOY = "infeasible_deploy"
VERDICTS = frozenset({VERDICT_MEASURED, VERDICT_DELIVERY,
                      VERDICT_NO_STEADY_STATE, VERDICT_DEPLOY})

# Below this many DRIVER reads per 5s sample, the record's client-side series are point reads
# and may be phase-locked to the arrival schedule. 2 rather than 1 because a single extra read
# proves the fast loop ran at all; the harness normally lands ~36. Driver reads, not engine
# scrapes: a server's silence must not disqualify a row, so the two are counted separately.
MIN_SUBSAMPLES = 2


class SuppressedMetric(Exception):
    """Raised when a caller asks for a metric the harness deliberately did not emit."""


@dataclass(frozen=True)
class TenantObservation:
    name: str
    requests_dispatched: int = 0
    # Requests the driver sent, whatever became of them. Differs from `requests_dispatched`, which
    # only counts requests a result was received for and therefore reads 0 under total overload.
    # Defaults to 0 for records written before it existed, so a reader must treat 0 as "unknown or
    # none" rather than as "none" -- `SoloProfile.too_narrow_reason` does.
    requests_launched: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    delivered_fraction: float | None = None
    delivered_fraction_threshold: float | None = None
    # How the driver's own in-flight count moved across the window, which is what "did the backlog
    # drain" is now decided from (impl §14.6). Load is offered open-loop, so a rising count *is*
    # completion below arrival. The two quarter means are the direction; `in_flight_samples` is what
    # stops an empty or two-sample series from reading as a flat one.
    in_flight_mean: float | None = None
    in_flight_max: float | None = None
    in_flight_first_quarter_mean: float | None = None
    in_flight_last_quarter_mean: float | None = None
    in_flight_samples: int | None = None
    # **Parse-only, and nothing writes them any more (impl §14.6).** Each was a serving engine's
    # account of its own internals, scraped from the tenant's `/metrics`. Records already in the
    # store carry them and must keep reading back -- a reader that dropped the fields would fail to
    # parse its own history -- so they are kept, unset on every new record, and judged by nothing.
    # `num_gpu_blocks` was already provenance-only before that (impl §6.0): the solver takes KV
    # capacity from the feasibility probe's own arithmetic, which is a fact about the envelope rather
    # than a number the engine happened to report after a profiling race (§4.2.2).
    preemptions: int | None = None
    kv_usage_mean: float | None = None
    kv_usage_max: float | None = None
    num_requests_running_mean: float | None = None
    num_requests_waiting_mean: float | None = None
    num_gpu_blocks: int | None = None
    schedule_lag_max_ms: float | None = None
    # None when the delivery gate failed. Access through the properties, which raise.
    _ttft_mean_ms: float | None = None
    _ttft_p95_ms: float | None = None
    _itl_mean_ms: float | None = None
    _itl_ci_halfwidth_ms: float | None = None
    latency_suppressed: str | None = None
    # Client-observed chunk counts vs the server's token counts. A divergence here is a
    # defect in the harness's observation path, not in the placement (see load.py).
    tokens_min: int | None = None
    chunks_min: int | None = None

    @property
    def latency_available(self) -> bool:
        return self.latency_suppressed is None and self._itl_mean_ms is not None

    def _require(self, value: float | None, what: str) -> float:
        if value is None:
            raise SuppressedMetric(
                f"{self.name}: {what} is not in this record. "
                f"{self.latency_suppressed or 'the harness did not emit it'}. "
                "Latency over the surviving requests of an under-delivering run is "
                "survivorship bias (impl §6.3); there is no safe default to substitute.")
        return value

    @property
    def itl_mean_ms(self) -> float:
        return self._require(self._itl_mean_ms, "itl_mean_ms")

    @property
    def itl_ci_halfwidth_ms(self) -> float:
        return self._require(self._itl_ci_halfwidth_ms, "itl_ci_halfwidth_ms")

    @property
    def ttft_mean_ms(self) -> float:
        return self._require(self._ttft_mean_ms, "ttft_mean_ms")

    @property
    def ttft_p95_ms(self) -> float:
        return self._require(self._ttft_p95_ms, "ttft_p95_ms")

    @property
    def delivery_passed(self) -> bool:
        if self.delivered_fraction is None or self.delivered_fraction_threshold is None:
            return False
        return self.delivered_fraction >= self.delivered_fraction_threshold

    @property
    def chunk_divergence(self) -> int:
        """Tokens the server sent that the client never observed as chunks.

        Kept as a first-class field because it is a live, unexplained divergence: with the
        istio sidecar removed from the path it persisted and moved in both directions, so it
        is neither the proxy nor a monotone leak. It affects no reported number -- the gate
        and ITL both read the server's count -- but it must stay visible.
        """
        if self.tokens_min is None or self.chunks_min is None:
            return 0
        return max(0, self.tokens_min - self.chunks_min)

    @classmethod
    def parse(cls, name: str, d: dict) -> TenantObservation:
        return cls(
            name=name,
            requests_dispatched=d.get("requests_dispatched", 0),
            requests_launched=d.get("requests_launched", 0),
            requests_completed=d.get("requests_completed", 0),
            requests_failed=d.get("requests_failed", 0),
            delivered_fraction=d.get("delivered_fraction"),
            delivered_fraction_threshold=d.get("delivered_fraction_threshold"),
            in_flight_mean=d.get("in_flight_mean"),
            in_flight_max=d.get("in_flight_max"),
            in_flight_first_quarter_mean=d.get("in_flight_first_quarter_mean"),
            in_flight_last_quarter_mean=d.get("in_flight_last_quarter_mean"),
            in_flight_samples=d.get("in_flight_samples"),
            # Parse-only: absent from every record written since impl §14.6, present in older ones.
            preemptions=d.get("preemptions"),
            kv_usage_mean=d.get("kv_usage_mean"),
            kv_usage_max=d.get("kv_usage_max"),
            num_requests_running_mean=d.get("num_requests_running_mean"),
            num_requests_waiting_mean=d.get("num_requests_waiting_mean"),
            num_gpu_blocks=d.get("num_gpu_blocks"),
            schedule_lag_max_ms=d.get("schedule_lag_max_ms"),
            _ttft_mean_ms=d.get("ttft_mean_ms"),
            _ttft_p95_ms=d.get("ttft_p95_ms"),
            _itl_mean_ms=d.get("itl_mean_ms"),
            _itl_ci_halfwidth_ms=d.get("itl_ci_halfwidth_ms"),
            latency_suppressed=d.get("latency_suppressed"),
            tokens_min=d.get("tokens_min"),
            chunks_min=d.get("chunks_min"),
        )


@dataclass(frozen=True)
class GpuObservation:
    """Per-GPU cause signals, keyed by UUID.

    Fields are left as a dict because the DCGM field set is a configuration choice, not a
    schema constant, and enumerating them here would mean this file needs editing every
    time a field is added to the exporter.

    A per-GPU signal answers "did this device saturate", never "which co-tenant saturated
    it" -- pod attribution is unavailable by construction for MPS tenants, since they bypass
    the device plugin that would otherwise own the mapping.
    """

    uuid: str
    fields: dict = field(default_factory=dict)

    def get(self, name: str, default=None):
        return self.fields.get(name, default)

    @property
    def sm_active_mean(self) -> float | None:
        return self.fields.get("sm_active_mean")

    @property
    def throttled(self) -> bool:
        """True if a power or thermal cap was seen.

        Recorded on every GPU whether or not it tripped, because a clock drop caused by a
        cap is indistinguishable from one caused by co-location unless the cap is in the
        record. `None` means the field was collected but never non-zero.
        """
        for k in ("power_violation_max", "thermal_violation_max"):
            v = self.fields.get(k)
            if v:
                return True
        return False


@dataclass(frozen=True)
class Observation:
    run_id: str
    validity_key_hash: str
    verdict: str
    node: str | None = None
    window_s: float = 0.0
    steady_state_reached_s: float | None = None
    per_tenant: dict = field(default_factory=dict)   # name -> TenantObservation
    per_gpu: dict = field(default_factory=dict)      # uuid -> GpuObservation
    workload: dict = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    source_path: str | None = None

    @property
    def measured(self) -> bool:
        return self.verdict == VERDICT_MEASURED

    @property
    def subsamples_per_sample(self) -> float:
        return float(self.sampling.get("subsamples_per_sample_mean") or 0.0)

    @property
    def sampling_suspect(self) -> bool:
        """True if this record's sampled series may be phase-locked to the arrival schedule.

        See the module docstring. A record from before the two-cadence sampler existed has
        no `sampling` block at all, which is exactly as suspect as one reporting a single
        sub-sample -- so an absent block is suspect, not assumed innocent.

        Engine-neutral by construction: the number counted is the driver loop's, so this says
        nothing about what the server published and cannot be tripped by an engine that
        published nothing.
        """
        return self.subsamples_per_sample < MIN_SUBSAMPLES

    @property
    def usable_as_evidence(self) -> bool:
        """Can this record be cited as evidence about latency?

        Requires a `measured` verdict AND trustworthy sampling. An infeasible record is
        still valuable -- it stops the agent re-proposing a placement that already failed
        (impl §8) -- but it is evidence about feasibility, not about latency.

        Named for what it is used for now. It was `usable_for_model`, and the model it named
        was the deleted interference predictor's fit set; the predicate itself is unchanged,
        because "a real measurement whose gauges can be trusted" is the same question whether
        the consumer is a fit or a reasoner reading the run (impl §10).
        """
        return self.measured and not self.sampling_suspect

    def rate_hz(self, tenant: str) -> float | None:
        for t in self.configuration.get("tenants", []):
            if t.get("name") == tenant:
                return t.get("rate_hz")
        return None

    def tenant_config(self, tenant: str) -> dict:
        for t in self.configuration.get("tenants", []):
            if t.get("name") == tenant:
                return t
        return {}

    def sequence_shape(self, tenant: str) -> tuple | None:
        """`(isl, osl, image_count, image_pixels)` this tenant was driven at, or None.

        Read from `configuration.tenants` rather than from the fleet, because the question a
        reader asks is what *this run* offered -- and the fleet may have been edited since. None
        rather than a default: a record written when the shape was fleet-wide cannot be read as
        a per-tenant one, and substituting the old fleet-wide pair would invent agreement.

        The image pair, unlike the token pair, reads as zeros when absent. That is not a default
        standing in for a missing fact: a record without it was written by a harness that had no
        route for an image, so it was a text run. The form matches `ValidityKey.sequence_shapes`
        so a reader can compare a record against the key it was filed under directly.
        """
        cfg = self.tenant_config(tenant)
        isl, osl = cfg.get("isl_tokens"), cfg.get("osl_tokens")
        if isl is None or osl is None:
            return None
        return (int(isl), int(osl),
                int(cfg.get("image_count") or 0), int(cfg.get("image_pixels") or 0))

    @classmethod
    def parse(cls, d: dict, source_path: str | None = None) -> Observation:
        verdict = d.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(
                f"unknown verdict {verdict!r} in {source_path or 'record'}; "
                f"expected one of {sorted(VERDICTS)}. This record was written by a version "
                "of the harness this code does not understand -- guessing what it meant "
                "would put a fiction in the store.")
        return cls(
            run_id=d.get("run_id", ""),
            validity_key_hash=d.get("validity_key_hash", ""),
            verdict=verdict,
            node=d.get("node"),
            window_s=d.get("window_s", 0.0) or 0.0,
            steady_state_reached_s=d.get("steady_state_reached_s"),
            per_tenant={n: TenantObservation.parse(n, v)
                        for n, v in (d.get("per_tenant") or {}).items()
                        if isinstance(v, dict)},
            per_gpu={u: GpuObservation(uuid=u, fields=v)
                     for u, v in (d.get("per_gpu") or {}).items()},
            workload=d.get("workload") or {},
            configuration=d.get("configuration") or {},
            sampling=d.get("sampling") or {},
            provenance=d.get("provenance") or {},
            notes=d.get("notes") or {},
            source_path=source_path,
        )


@dataclass(frozen=True)
class Calibration:
    """Outputs of design §8.1, written once per validity key.

    `between_run_itl_halfwidth_ms` is keyed by the placement's co-residency bucket, because a
    placement that shares GPUs is not expected to reproduce as well as one that shares none, and
    a single number would make the agent either chase noise in one case or fail to discriminate
    in the other. `agent.calibrate.bucket` is what produces the key; the value is a measured
    half-width and nothing here invents one.

    The agent reads its discrimination threshold from here, and the harness reads its window
    rule from here. Absent means "run calibration" (impl §11), which is why the store
    returns None rather than a default: a default would let the whole search run at a
    resolution nobody chose.
    """

    between_run_itl_halfwidth_ms: dict = field(default_factory=dict)   # regime -> ms
    solo_saturation_rate_hz: dict = field(default_factory=dict)        # model-tp -> Hz
    repeats: int = 0
    notes: dict = field(default_factory=dict)

    def threshold_ms(self, bucket: str) -> float | None:
        return self.between_run_itl_halfwidth_ms.get(bucket)

    @property
    def tightest_ms(self) -> float | None:
        """The smallest half-width measured, over every bucket. None when uncalibrated.

        The conservative reading, for a caller with no bucket to look up: it demands the longest
        measurement window, and a window that was too short is the one error that cannot be
        repaired after the run.
        """
        vals = [v for v in self.between_run_itl_halfwidth_ms.values() if v is not None]
        return min(vals) if vals else None

    def to_dict(self) -> dict:
        return {
            "between_run_itl_halfwidth_ms": self.between_run_itl_halfwidth_ms,
            "solo_saturation_rate_hz": self.solo_saturation_rate_hz,
            "repeats": self.repeats,
            "notes": self.notes,
        }

    @classmethod
    def parse(cls, d: dict) -> Calibration:
        return cls(
            between_run_itl_halfwidth_ms=d.get("between_run_itl_halfwidth_ms") or {},
            solo_saturation_rate_hz=d.get("solo_saturation_rate_hz") or {},
            repeats=d.get("repeats", 0),
            notes=d.get("notes") or {},
        )
