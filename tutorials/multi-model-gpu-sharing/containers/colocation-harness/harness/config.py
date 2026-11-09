"""Run configuration: what to drive, how hard, and when to stop.

One YAML file per run. Every field that changes what the numbers mean is here rather
than in code, because the validity key (design §4.1, §8.4) has to be able to name the
conditions an observation was taken under -- including the metric definitions and the
workload, not only the hardware.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from . import engines
from .controller import StabilityThresholds

log = logging.getLogger(__name__)

# Stability thresholds that existed in a shipped image and are now refused by name rather than by
# TypeError. Kept as data so the self-test can assert the refusal without restating the list.
RETIRED_STABILITY_KEYS = frozenset({"delivered_rate_rel", "kv_usage_abs", "preemption_rate_abs"})


@dataclass
class TenantConfig:
    """One tenant to drive.

    `gpu_uuids` is supplied, not discovered. DCGM cannot tell the harness which pod
    owns a GPU under MPS -- see scrape.scrape_dcgm -- so the mapping comes from the
    tenant's own Helm values, where it is correct by construction.
    """
    name: str
    base_url: str
    served_name: str
    rate_hz: float
    # This tenant's request shape. Per tenant, not per run: design §8.3 makes the mix the
    # problem, and a fleet-wide pair would drive four tenants at one shape while the record named
    # the shapes each of them asked for. Required, with no default, for the same reason `rate_hz`
    # has none -- an invented shape is a measurement of something nobody requested.
    isl_tokens: int
    osl_tokens: int
    # Which tokenizer builds this tenant's prompt. Per tenant, not per run: it decides what
    # `isl_tokens` above actually delivers, so one tokenizer for a mixed fleet would make the
    # same number mean a different prompt for each tenant. None means the prompt is built without
    # one and `isl_exact` comes back false -- recorded, never assumed.
    tokenizer_path: str | None = None
    # This tenant's multimodal payload. Zero images is a text tenant and is the default,
    # because most tenants are; a tenant that states images is driven through the chat
    # endpoint instead, since there is no route for an image through `/v1/completions`.
    #
    # `image_pixels` is per image and must sit inside the engine's own `min_pixels` /
    # `max_pixels`, which the fleet-file loader checks. Outside them the server's
    # preprocessor silently resizes, and the driver would then be sending one resolution
    # while the engine costs another -- the same class of error as an approximate ISL,
    # and less visible.
    #
    # Neither field feeds an arithmetic prediction of visual tokens. What they do is fix
    # the payload; what it costs is measured by the start-up probe (`workload.calibrate`).
    image_count: int = 0
    image_pixels: int = 0
    gpu_uuids: list[str] = field(default_factory=list)
    tp: int = 1
    envelope_gib: float | None = None
    model_id: str | None = None
    # Fraction of offered tokens that must arrive for latency figures to mean
    # anything. Below it the run is infeasible_delivery and no latency is emitted
    # (impl §6.3, design §8.5).
    delivered_fraction_threshold: float = 0.99
    # Which serving stack this tenant runs, resolved through `harness.engines`. It decides the request
    # path, how output length is pinned, the readiness path and the names of the runtime gauges.
    # Defaults to vLLM's OpenAI server, which is what a run config written before this field existed
    # was driving -- the only engine there was.
    engine: str = engines.DEFAULT_ENGINE
    # The MPS SM ceiling the search's reasoner set on this tenant (impl §16), 1-99. The harness does
    # not apply it -- the chart sets it on the pod -- it only carries it into the record, because a
    # capped tenant is a different placement from the same tenant uncapped. None is uncapped.
    active_thread_percentage: int | None = None


@dataclass
class RunConfig:
    run_id: str
    tenants: list[TenantConfig]

    # The sequence shape is on the tenant, and so is the tokenizer. Neither is fleet-wide: a
    # tokenizer builds a prompt of an exact token count for the model it belongs to and an
    # approximate one for any other, so a single one here made ISL exact for one tenant of a
    # mixed fleet and silently approximate for the rest.
    exporter_url: str | None = None
    node: str | None = None

    # Window length. `itl_target_halfwidth_ms` is half the between-run
    # reproducibility figure for this regime, from the calibration record. It does
    # not exist before calibration has run -- calibration is what produces it -- so
    # leaving it unset selects a fixed window and the record says which rule applied.
    itl_target_halfwidth_ms: float | None = None
    window_min_s: float = 60.0
    window_ceiling_s: float = 900.0
    # 900s = 180 samples at the 5s cadence, raised from 600s. Not a loosened threshold --
    # the criterion is untouched -- but the time it needs to be *readable*. `delivered`
    # counts tokens at request completion, so at 1 request/s the series is a lumped
    # Poisson count with a ~45% per-sample spread, and its batch-means half-width only
    # crosses the 0.10 gate at ~140 samples. A slow model spends the first ~11 on its
    # warm-up (`controller.warmup_samples`), so 600s left a still machine a coin flip
    # away from `infeasible_no_steady_state`. Runs that do settle return early and pay
    # none of this. The deeper fix is to count chunks as they stream instead of at
    # completion, which would make the rate smooth and the warm-up a ramp rather than a
    # gap; that changes what the signal *is*, so it is a separate change.
    steady_state_ceiling_s: float = 900.0
    drain_timeout_s: float = 120.0

    stability: StabilityThresholds = field(default_factory=StabilityThresholds)

    validity_key_hash: str = "unset"
    out_dir: str = "/efs/colocation"
    # Extra provenance passed straight through to the record: image digest, driver
    # version, α, traffic mix. Whatever identifies this run's conditions.
    provenance: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        legacy = sorted({"isl_tokens", "osl_tokens"} & set(raw))
        if legacy:
            # Named explicitly rather than left to a TypeError on an unexpected keyword: a run
            # config written against the fleet-wide shape is a config for a different experiment.
            raise ValueError(
                f"{path}: {', '.join(legacy)} is no longer a run-level field. Move the sequence "
                "shape onto each tenant (impl §6.2). It is not applied fleet-wide here, because "
                "driving every tenant at one shape is a different measurement than the one a "
                "mixed fleet asks for.")
        if "tokenizer_path" in raw:
            # Same treatment, same reason. Silently lifting it onto every tenant would make ISL
            # exact for whichever tenant owns that tokenizer and approximate for the others,
            # which is a fleet nobody configured.
            raise ValueError(
                f"{path}: tokenizer_path is no longer a run-level field. State it on each tenant "
                "(normally its own model's directory). Not applied fleet-wide here, because one "
                "tokenizer builds a prompt of an exact length for one model only, and the rest "
                "of the fleet would be driven at a shape the record names but never delivered.")
        tenants = [TenantConfig(**t) for t in raw.pop("tenants")]
        stab_raw = dict(raw.pop("stability", {}) or {})
        retired = sorted(set(stab_raw) & RETIRED_STABILITY_KEYS)
        if retired:
            # A run config written against an earlier harness image can name these, and the bare
            # TypeError from the dataclass would read as a typo rather than as a deliberate removal.
            raise ValueError(
                f"{path}: stability threshold(s) {', '.join(retired)} no longer exist. Steady state "
                "is judged on one signal -- in-flight holding flat on the driver's own books -- and "
                "the rest were removed on purpose: the engine's KV and preemption gauges because no "
                "decision here may depend on what a backend chooses to publish about its own "
                "internals, and delivered_rate because with output length pinned it is in-flight's "
                "flatness restated through a noisier estimator. See controller.StabilityThresholds. "
                "Drop the key; there is no replacement to set.")
        stability = StabilityThresholds(**stab_raw)
        cfg = cls(tenants=tenants, stability=stability, **raw)
        cfg._apply_env()
        cfg.validate()
        return cfg

    def _apply_env(self) -> None:
        # Only for values the Job manifest knows and the config file should not
        # hardcode: which node the Job landed on, and the run id when a controller
        # generates one per placement.
        if v := os.environ.get("HARNESS_NODE"):
            self.node = v
        if v := os.environ.get("HARNESS_RUN_ID"):
            self.run_id = v

    def validate(self) -> None:
        if not self.tenants:
            raise ValueError("no tenants configured")
        names = [t.name for t in self.tenants]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tenant names: {names}")
        for t in self.tenants:
            # Resolved here rather than at first request, because the failure it prevents is a run
            # that has already deployed, warmed up and started driving before discovering it does not
            # know how to talk to the server. The refusal names the engines this image has.
            engines.get(t.engine)
            if t.rate_hz <= 0:
                raise ValueError(f"{t.name}: rate_hz must be > 0")
            if t.isl_tokens < 1 or t.osl_tokens < 1:
                raise ValueError(
                    f"{t.name}: isl_tokens and osl_tokens must both be >= 1; got "
                    f"{t.isl_tokens} and {t.osl_tokens}")
            if t.image_count < 0 or t.image_pixels < 0:
                raise ValueError(
                    f"{t.name}: image_count and image_pixels must both be >= 0; got "
                    f"{t.image_count} and {t.image_pixels}")
            if t.image_count and t.image_pixels < 1:
                raise ValueError(
                    f"{t.name}: image_count={t.image_count} but no image_pixels. The pixel "
                    "count is the whole specification of a synthetic image -- it decides the "
                    "visual token count and the vision tower's work -- so there is no default "
                    "that would not be inventing a workload.")
            if not t.gpu_uuids:
                # Not fatal: the run still measures effect signals. But per-GPU cause
                # signals cannot be attributed without it, so say so loudly.
                log.warning("%s: no gpu_uuids -- per-GPU cause signals will be "
                            "unattributed for this tenant", t.name)
        if self.itl_target_halfwidth_ms is None:
            log.warning("no itl_target_halfwidth_ms: using a fixed %.0fs window. "
                        "Correct during calibration, wrong afterwards.", self.window_min_s)
        if not self.exporter_url:
            log.warning("no exporter_url: no DCGM cause signals. Every causal claim "
                        "from this run is inference from latency (design §8.7).")

    @property
    def itl_target_halfwidth_s(self) -> float | None:
        if self.itl_target_halfwidth_ms is None:
            return None
        return self.itl_target_halfwidth_ms / 1000.0

    def describe(self) -> dict:
        """The config as it will be stored, so a record is self-contained."""
        d = asdict(self)
        d.pop("out_dir", None)
        return json.loads(json.dumps(d))
