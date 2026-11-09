"""Solo profiles: what one model does on its own, per TP degree (impl §0, §6).

This is the store that establishes **minimal TP**, which §0 makes part of the baseline rather than
part of the search. It answers one question per row: *this model, sharded this many ways, driven at
this rate and this request shape, alone on its own GPUs with the whole of each one to itself — what
latency did it deliver?* A tenant's minimal TP is then the smallest degree whose answer meets that
tenant's target.

Three things about the shape of this store are decisions, not conveniences.

**It is keyed by the environment, not by a fleet.** `ValidityKey.tensor_parallel` is per tenant and
in the hash (`validity_key.py`), which is right for observations of a fleet and fatal for a profile:
a profile run at TP=1 and the same model's run at TP=2 land under *different* validity keys, so no
single key can hold the sweep that establishes minimal TP. What conditions a solo measurement is the
part of the key that has nothing to do with the fleet — the environment: what the card is, what it
addresses, what it may draw, how its devices are linked and shared, the engine image and the metric
definitions (`store/environment.py`) — and that is `ProfileKey`. TP, shape and rate move into the *record*, where varying them
is the point.

**A profile outlives the fleet that provoked it.** A model profiled for one fleet is profiled for
every later fleet on the same hardware, which is what makes this the one artifact in the system that
amortizes: the second fleet on a g6e is cheaper to place than the first. That only works because
targets are *not* in here. A target is a requirement a reader applies (`meets`), not a condition of
measurement — the same argument `TenantSlo` makes for keeping targets out of the validity key.

**Solo means best case, and that is what makes it a bound.** The profile is taken with the tenant
alone on its GPUs and the envelope spent down to what one GPU holds, so it is the fastest that
(model, TP, rate, shape) can go on this hardware. If a degree misses the target *there*, no
co-resident placement can rescue it — sharing only ever takes SM time away. So a failed solo profile
rules a degree out for that target permanently, and a passing one is the reference every co-resident
measurement is compared against ("vs solo").

Layout, beside `observations/` rather than inside it, because these are not observations of a fleet:

    /efs/colocation/profiles/
      <profile-key-hash>/
        key.json
        solo/<model>-tp<N>-isl<I>-osl<O>-<rate>hz-cfg<digest>.json

`cfg<digest>` is the engine configuration the row was measured under (`serving.py`). It is in the
filename rather than the key because the key is the environment and this is the tenant: raising a
batch cap and re-measuring has to write a new row, not overwrite the one taken at the old cap.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .environment import Environment
from .schema import MIN_SUBSAMPLES, VERDICT_MEASURED
from .serving import ServingConfig
from .validity_key import HASH_CHARS, RATE_DECIMALS, ValidityKey

log = logging.getLogger(__name__)

# What a profile could not establish, in the words the reports use. `minimal_tp` returns one of
# these in `status` and nothing infers a TP degree from any of them.
RESOLVED = "resolved"
UNPROFILED = "unprofiled"
NONE_MEETS = "no_profiled_degree_meets_the_target"
NO_TARGET = "no_itl_target_stated"

# Why a degree below the answer did not settle it. `minimal_tp` files one of these beside each
# degree in `TpResolution.below`, and both `certain` and `missing` are computed from the CODE
# rather than from the prose next to it.
#
# The split that matters is MISSED against everything else. A degree that was measured and missed
# is *checked*: it rules that degree out permanently, because solo is the best case and sharing
# only ever takes SM time away. Every other code means the degree is **unchecked** -- nobody knows
# whether it would have met the target -- and an unchecked degree below the answer is what makes
# the answer an upper bound rather than the minimum. Conflating the two in either direction is a
# §0 error: calling unchecked "missed" pushes a tenant to a wider shard count on a measurement that
# never happened, and calling it "checked" hands the baseline GPUs that re-sharding alone would free.
MISSED = "missed"
# Ruled out by arithmetic rather than by measurement: the shard does not fit at that degree, so
# there is nothing to measure and no gap in the sweep. Checked, therefore, and it must not make the
# answer an upper bound -- a degree that cannot hold the weights is not a degree that might have
# been narrower.
DOES_NOT_FIT = "does_not_fit"
NOT_PROFILED = "not_profiled"
# The row exists but cannot be cited about latency. The first four are the grounds
# `ObservationRecord.usable_as_evidence` applies, kept in step with it deliberately; the fifth is
# this store's own and is about the instrument rather than the server -- see `unusable_reason`.
UNUSABLE_NO_LATENCY = "latency_suppressed"
UNUSABLE_DELIVERY = "delivery_failed"
UNUSABLE_VERDICT = "harness_verdict"
UNUSABLE_SAMPLING = "sampling_suspect"
UNUSABLE_SCHEDULE_LAG = "schedule_lag"
# Every unusable ground belongs here, `UNUSABLE_SCHEDULE_LAG` included: a row disqualified because
# the driver missed its schedule says nothing about whether that degree meets the target, so the
# degree is unchecked and the answer above it is an upper bound. Leaving it out of both sets would
# have kept it out of `missing` and reported `NONE_MEETS` -- "no profiled degree meets the target" --
# about a degree that was never validly driven.
UNCHECKED = frozenset({NOT_PROFILED, UNUSABLE_NO_LATENCY, UNUSABLE_DELIVERY,
                       UNUSABLE_VERDICT, UNUSABLE_SAMPLING, UNUSABLE_SCHEDULE_LAG})
# The row cannot be cited about latency AND rules its degree out anyway, on evidence that the server
# was working and behind (`SoloProfile.too_narrow_reason`). CHECKED, because a degree ruled out is
# ruled out however -- the answer above it is a measured minimum and not an upper bound. It stays out
# of UNCHECKED for the same reason it is not in the unusable set: those name a row that says nothing,
# and this row says something.
TOO_NARROW = "too_narrow"
CHECKED = frozenset({MISSED, DOES_NOT_FIT, TOO_NARROW})

# What "the device was busy" and "the backlog did not drain" mean, for `too_narrow_reason`. Both are
# dimensionless fractions of a whole, so neither names a card, a cluster or a model -- 0.85 is a
# statement about what counts as saturated, not about any hardware. They are deliberately loose:
# this predicate's job is to separate a server working flat out from a server that never got going,
# and that is a gross distinction. Tightening them would not make the answer more right, it would
# make a genuinely saturated round fall through to "not measured" and stall the walk.
BUSY_FRACTION = 0.85
# `KV_FULL_FRACTION = 0.90` was here, with `num_requests_waiting_mean >= 1` and `preemptions > 0`
# alongside it, and between them they decided "the backlog did not drain" before the driver was
# consulted at all. All three read a serving engine's account of its own internals, which is the one
# thing no judgement here may do, so they are gone (impl §14.6) and the two below replace them.
#
# How far the in-flight count must climb across the window before the backlog counts as not draining.
# A fraction of the level it started at, so it says nothing about a rate, a model or a card. Load is
# offered open-loop, so in-flight rising *is* completion below arrival -- there is no other way for it
# to rise -- and the only question a threshold answers is how much rise is more than sampling noise.
# 25% of the opening level over three quarters of a window is well outside it, and the population this
# is asked about makes a false positive unlikely from the other side too: a settled row is flat by
# construction, so a row where this fires either never settled or settled and then came apart.
IN_FLIGHT_RISE_FRACTION = 0.25
# Below this many 5s samples the two quarter means are each an average of one or two numbers, and the
# comparison is noise. A full window is ~30; this refuses the stub, not the short run.
MIN_IN_FLIGHT_SAMPLES = 8
# How far the driver may fall behind its own arrival schedule before the offered rate stops being
# the rate the fleet file states. One second: the rates in play are single-digit Hz, so a lag of
# this size means whole requests were never sent, and the row cannot be read as the server failing
# to keep up with a load it was never offered.
MAX_SCHEDULE_LAG_MS = 1000.0


@dataclass(frozen=True)
class ProfileKey:
    """The fleet-independent part of a validity key: what conditions a *solo* measurement.

    Exactly `ValidityKey`'s environment and nothing else. The omissions are the content:
    `tensor_parallel`, `sequence_shapes` and `offered_rates_hz` are per tenant and are what a
    profile sweeps, so a key naming them could hold one row of the sweep and not the sweep.
    """

    env: Environment

    @classmethod
    def from_validity_key(cls, key: ValidityKey) -> ProfileKey:
        """The environment a fleet's key describes, with the fleet dropped.

        So a `search` and the `profile` runs that justified its TP degrees agree about the machine
        by construction, rather than by two places being edited consistently.
        """
        return cls(env=key.env)

    def canonical(self) -> dict:
        return {"environment": self.env.canonical()}

    @property
    def hash(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:HASH_CHARS]

    def key_json(self) -> str:
        return json.dumps({"hash": self.hash, "kind": "solo-profile", **self.canonical()},
                          indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, d: dict) -> ProfileKey:
        if "environment" not in d:
            raise ValueError(
                "this profile key has no environment block. It was written when the environment was "
                "a driver version and an instance type, so what the device reported about itself is "
                "not recorded -- see store/validity_key.py for why that cannot be filled in later.")
        return cls(env=Environment.from_dict(d["environment"]))


def profile_id(model_id: str, tp: int, isl_tokens: int, osl_tokens: int,
               rate_hz: float, serving: ServingConfig,
               image_count: int = 0, image_pixels: int = 0) -> str:
    """The identity of one profiled row, as a filename-safe string.

    The rate is quantized exactly as `ValidityKey` quantizes `offered_rates_hz`, for the same
    reason: a rate arrived at as a fraction of a measured saturation rate carries float noise, and
    `2.0000000000000004` must not be a different row from `2.0`.

    **`serving` is part of the identity and is required.** A row says what one model did at one
    degree, shape and rate -- and it is only about that if the engine flags are pinned too. The
    `cfg` suffix is `ServingConfig.digest`; the flags themselves are inside the row, so nothing
    ever has to reverse it. Before 2026-09-09 this function ignored them, which meant raising
    `--max-num-seqs` and re-measuring overwrote the row taken at the old cap with nothing anywhere
    recording that the two were measured under different engines (`serving.py`).

    **The image payload is part of the identity for that same reason**, and it is the sharpest case
    of it: a row measured at ISL 1088 with one megapixel image and a row measured at ISL 1088 with
    no image are the same model at the same nominal shape doing two different amounts of work, and
    the first one ran a vision tower the second never touched. Sharing an id, the multimodal
    re-measurement would silently supersede the text one -- `write` overwrites by design -- and
    `minimal_tp` would then set a text tenant's degree from an image measurement.
    The suffix appears only for a row that sends images, so a text row's id is unchanged and the
    text rows already in the store stay findable; there is nothing to conflate, because a row
    written before this suffix existed was measured by a harness with no route for an image.
    """
    safe = str(model_id).replace("/", "_")
    rate = f"{round(float(rate_hz), RATE_DECIMALS):g}".replace(".", "p")
    img = f"-img{int(image_count)}x{int(image_pixels)}" if int(image_count) else ""
    return (f"{safe}-tp{int(tp)}-isl{int(isl_tokens)}-osl{int(osl_tokens)}-{rate}hz"
            f"-cfg{serving.digest}{img}")


@dataclass(frozen=True)
class SoloProfile:
    """One model at one TP degree, alone, at one rate and shape.

    `itl_mean_ms` and `ttft_p95_ms` are None when the harness suppressed latency -- a failed
    delivery gate (impl §6.3). None is not slow and not fast: it is a row that cannot certify or
    rule out anything, and `meets()` says so by returning None rather than False.

    `reserve_gib` is the per-shard memory fact this run measured, and it is here because a solo run
    is the only place it can be measured cleanly -- with a co-tenant on the card, resident memory is
    the sum of both and MPS gives no attribution. It is what `ModelMemory.reserve_gib` would
    otherwise fall back to a safe default for, and the fleet loader merges it
    (`ProfileStore.reserve_by_tp`).

    `activation_gib` is declared beside it and is always None. It is the other half of the same
    budget and cannot be measured the same way: the reserve is resident memory MINUS the utilisation
    budget, which is visible from outside, while the activation term is a slice of the inside of that
    budget and the engine fills whatever is left of it with KV cache. Reading it needs the engine's
    own start-up report, so this design does not read it at all and the safe fallback stands
    (`solver/memory.ACTIVATION_GIB_DEFAULT`). The field is kept rather than deleted because rows
    already filed carry it, and a `None` here is the honest statement that nothing measured it.
    """

    model_id: str
    tp: int
    isl_tokens: int
    osl_tokens: int
    rate_hz: float
    envelope_gib: float
    # The engine flags this row was measured under. Part of the row's identity, not decoration:
    # see `profile_id`.
    serving: ServingConfig
    # The multimodal payload this row was driven with, per request. Also part of the identity, and
    # defaulted to none because a text row is the common case -- see `profile_id` for why sharing
    # an id with the text row would let one supersede the other.
    image_count: int = 0
    image_pixels: int = 0
    itl_mean_ms: float | None = None
    ttft_p95_ms: float | None = None
    delivered_fraction: float | None = None
    delivery_passed: bool | None = None
    activation_gib: float | None = None
    reserve_gib: float | None = None
    # The KV pool this row was DRIVEN WITH, and where the reserve that sized it came from. Both are
    # inputs, not results, and they are here because `reserve_gib` above makes the row's own
    # conditions move: a solo round is the only place the reserve can be measured, so the FIRST row
    # for a (model, degree, environment) is necessarily measured on the default reserve, and every
    # run afterwards -- including every co-resident run this row is the `vs solo` reference for --
    # gets the measured one the fleet loader has since merged in (`ProfileStore.reserve_by_tp`).
    # Nothing is wrong with either number. What is wrong is subtracting them: the reference and the
    # thing compared to it were driven with different pools, so the difference includes an
    # uncontrolled input and cannot be attributed to sharing. Recorded rather than inferred, because
    # the row outlives the record and the condition is not recoverable from the row's identity --
    # the reserve is not part of the profile key, deliberately, since it is measured BY these rows.
    # `_vs_solo` is the reader: it compares these against the co-resident run's own and declines to
    # quote a bare ratio across a boundary. Blank on a row written before this existed, which reads
    # as "not known" and not as "the same".
    kv_tokens: int | None = None
    reserve_source: str = ""
    # Provenance. `observation_key_hash` is the validity key the underlying record was filed
    # under -- which is a *different* hash per TP degree, and is exactly why this store exists.
    run_id: str = ""
    run_token: str = ""
    gpu_node: str = ""
    gpu_ids: tuple = ()
    observation_key_hash: str = ""
    measured_at: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return profile_id(self.model_id, self.tp, self.isl_tokens, self.osl_tokens, self.rate_hz,
                          self.serving, self.image_count, self.image_pixels)

    @property
    def verdict(self) -> str | None:
        """The harness's own verdict on the run behind this row, or None for a row without one."""
        return (self.notes or {}).get("verdict")

    @property
    def unusable_reason(self) -> tuple | None:
        """`(code, prose)` if this row cannot be cited about latency, else None.

        **This is the same question `ObservationRecord.usable_as_evidence` answers, and it was
        answered inconsistently until 2026-09-09.** That property requires a `measured` verdict and
        trustworthy sampling, on the argument that an infeasible record is evidence about
        *feasibility* and not about latency. This one checked only that a latency number was present
        and the delivery gate had not failed -- so a row the harness declined to vouch for set
        minimal TP, and therefore the §0 baseline, with nothing printed. That is the worst place in
        the system for a missing guard: the baseline is the number a search is scored against, so an
        unvouched row does not merely add noise, it moves the yardstick.

        It surfaced on GLM-4.5-Air at tp4, whose run came back `infeasible_no_steady_state` and was
        nonetheless reported as `BASELINE at minimal TP: 4 GPU(s)`. Its latency was in fact sound
        (queue flat, nothing preempted, everything delivered), which is exactly why the gap survived
        this long -- the number looked fine, so nobody asked whether it was allowed to be used.

        The verdict is not a judgement this store may second-guess. `infeasible_no_steady_state`
        means the harness could not establish that the system had settled, so the average it went on
        to compute is an average over an unknown mixture of transient and steady state. It may be
        close to the truth; nothing in the row says how close.

        On sampling: the check is a *positive* suspect flag, not an absent one. That is the opposite
        of `sampling_suspect`, where an absent block is suspect rather than assumed innocent, and the
        difference is deliberate -- there, absence means the two-cadence sampler did not exist yet;
        here it means this NOTE did not exist yet, and rows written before it carry no claim either
        way. `profiler.py` records the flag from the record at write time so newer rows do.

        **On the driver's own schedule, added when the steady-state gate was cut to one signal.**
        `too_narrow_reason` has always refused to call a degree insufficient when the driver fell
        behind its arrival schedule (criterion 4), and `profiler.py` says why when it files the
        number: a driver that fell behind offered less load than the fleet file states, so the server
        was never asked for the rate the row is about to be judged against. That guard was missing
        *here*, in the direction that matters more. Late requests are still launched, so
        `delivered_fraction` shrinks its numerator and denominator together and stays near 1.0 -- the
        row passes every other check and gets cited as this model meeting its target at this rate,
        which sets minimal TP and therefore the §0 baseline. Same class of defect as the 2026-09-09
        one above: not noise, a moved yardstick. Absent means no claim, as everywhere else here,
        because `notes.saturation` postdates the earliest rows.
        """
        if self.itl_mean_ms is None:
            # The harness records WHY it suppressed the metric, so quote it rather than assert a
            # cause. Asserting one was wrong on the anchor's tp8 row: the gate did fail, but the
            # row's own `delivered_fraction` is null -- not one request completed inside the
            # window, so the fraction had no denominator -- and a reader chasing "failed delivery
            # gate" finds nothing in the row that says it failed.
            said = (self.notes or {}).get("latency_suppressed")
            return (UNUSABLE_NO_LATENCY,
                    f"no latency in the row: {said}" if said else
                    "no latency in the row, and it does not say why the harness suppressed it")
        if self.delivery_passed is False:
            return (UNUSABLE_DELIVERY,
                    f"delivery gate failed at {self.delivered_fraction!r} of offered load")
        v = self.verdict
        if v is not None and v != VERDICT_MEASURED:
            return (UNUSABLE_VERDICT,
                    f"the harness did not vouch for this run: verdict {v}")
        if (self.notes or {}).get("sampling_suspect") is True:
            return (UNUSABLE_SAMPLING,
                    f"fewer than {MIN_SUBSAMPLES} driver reads per sample, so this row's "
                    "in-flight series is a point read per sample and may be phase-locked to the "
                    "arrival schedule, which reads as flat whether or not it was")
        lag = ((self.notes or {}).get("saturation") or {}).get("schedule_lag_max_ms")
        if lag is not None and float(lag) > MAX_SCHEDULE_LAG_MS:
            return (UNUSABLE_SCHEDULE_LAG,
                    f"the driver fell {float(lag):.0f} ms behind its own arrival schedule "
                    f"(limit {MAX_SCHEDULE_LAG_MS:.0f} ms), so the offered rate was below the "
                    f"{self.rate_hz} Hz this row is about to be cited at")
        return None

    @property
    def usable(self) -> bool:
        """Whether this row can be cited about latency, and so accept or reject a degree on time.

        Not the same as "this row says nothing": see `too_narrow_reason`, which can rule a degree out
        on evidence that has no latency number in it at all.
        """
        return self.unusable_reason is None

    @property
    def too_narrow_reason(self) -> str | None:
        """Why this row establishes its degree as **too narrow**, or None if it does not.

        This is a second question, not a softening of the first. `unusable_reason` asks *can this row
        be cited about latency* -- and everything it says, including the 2026-09-09 rule that an
        unvouched verdict may not set minimal TP, stands exactly as written. This asks *does this row
        settle that the degree is insufficient*, and a row can fail the first and pass the second: a
        server driven flat out that completed nothing has no latency to quote and is strong evidence
        that one shard cannot hold the work. Under the old single question that round widened
        nothing, which had the walk conclude "not measured" about the most conclusive round it had.

        **Only positive evidence counts.** The failure mode being guarded against is a broken round
        -- wrong flags, unreachable server, a driver that never sent anything -- reading as overload
        and buying the baseline a wider shard nothing ruled the narrower one out at. That is the same
        error §0 deleted `tp:` for, arrived at from the other direction, so the bar is a server that
        can be *shown* to have been working and *shown* to have been behind:

        1. the server priced this exact payload itself (the ISL probe landed), so the request shape
           is one it accepts and "nothing came back" cannot be a rejection at the door;
        2. load was actually offered -- the driver's own count of what it sent;
        3. none of them failed in transport -- an error is the server saying no, not saying slow;
        4. the driver kept its own arrival schedule, so the offered rate is the stated one;
        5. the device was busy -- engine-blind, read off DCGM rather than off the server;
        6. the backlog did not drain, by the driver's own books: the in-flight count climbed across
           the window, or far fewer requests completed than were sent.

        **6 used to ask the server first (impl §14.6).** It tried `num_requests_waiting_mean >= 1`,
        then `kv_usage_max`, then a preemption count, and only reached the client's books if an engine
        published none of the three -- so the authority to widen a degree rested on gauges one engine
        publishes and another does not, which is the failure the whole seam exists to prevent. 2 had
        the same shape: a missing client count fell back to the server's report that it was holding
        requests. Both now read one series, the driver's own in-flight level, which is the *same*
        series the steady-state gate reads -- so the walk's two questions come from one measurement,
        and neither can be defeated by a server that stops answering a scrape.

        Why the in-flight count answers it: load is offered open-loop, from a fixed schedule with no
        back-pressure, so in-flight rising is completion below arrival by arithmetic rather than by
        inference. And the threshold cannot be generous towards a broken round, because a row that
        settled is flat by construction -- rising can only fire on the population this is asked about.

        Returns prose rather than a code because, unlike `unusable_reason`, nothing branches on
        *which* evidence carried it -- the caller needs the answer and a line to log.

        **A boot that never came up is the second ground, and it is not a driven round at all.**
        Everything above is about a server that answered; `notes.boot_failed` records the opposite
        case -- the tenant was deployed alone with the whole card at this degree and the engine could
        not start in it. That is evidence about the DEGREE only because of the control the profiler
        attaches beside it: `booted_at_tp` is a wider degree at which the same tenant, image, chart
        and model path came up in the same run, which is what separates "this shard is too small" from
        "this fleet cannot start" (a bad image or a wrong flag fails at every degree). The profiler
        writes the note only when it holds that control, so the predicate reports it rather than
        re-deciding it -- and no engine log is read to reach it, only the pod (impl §14.2, §14.5).
        """
        boot = (self.notes or {}).get("boot_failed") or {}
        if boot.get("gate"):
            return (f"the engine was given the whole card at tp{self.tp} "
                    f"({self.envelope_gib:.2f} GiB per shard) and did not come up "
                    f"({boot['gate']}: {boot.get('detail') or 'no detail recorded'}), while the same "
                    f"tenant at tp{boot.get('booted_at_tp')} came up in the same run")
        ev = (self.notes or {}).get("saturation") or {}
        if not ev:
            return None

        def num(key):
            v = ev.get(key)
            return None if v is None else float(v)

        if ev.get("isl_exact") is not True:
            return None
        failed = num("requests_failed")
        if failed is None or failed > 0:
            return None
        lag = num("schedule_lag_max_ms")
        if lag is not None and lag > MAX_SCHEDULE_LAG_MS:
            return None

        # Was load actually offered? Two ways to know, both the client's own books.
        # `requests_dispatched` is second and is the weaker, because it counts only requests a result
        # came back for and so reads 0 in exactly this situation -- it is here for records written
        # before `requests_launched` existed. A third way used to follow: the same fact seen from the
        # far end, a server reporting it was holding requests nobody but this driver could have sent
        # it. That salvaged a record with no client-side count, and it did so by reading the server
        # (impl §14.6). A record that cannot say what it sent now settles nothing, which is the
        # correct answer for a row whose own books are missing.
        offered = num("requests_launched") or num("requests_dispatched")
        if not offered:
            return None

        busy = None
        for f in ("gr_engine_active_mean", "sm_active_mean"):
            v = num(f)
            if v is not None and v >= BUSY_FRACTION:
                busy = f"{f}={v:.4g}"
                break
        if busy is None:
            return None

        completed = num("requests_completed")
        first, last = (num("in_flight_first_quarter_mean"), num("in_flight_last_quarter_mean"))
        n_samples = num("in_flight_samples")
        behind = None
        if (first is not None and last is not None and n_samples is not None
                and n_samples >= MIN_IN_FLIGHT_SAMPLES and first > 0.0
                and last >= first * (1.0 + IN_FLIGHT_RISE_FRACTION)):
            # The primary signal, and the sharper of the two: a degree that is merely falling behind
            # shows up here while it is still completing most of what it is sent, which is the case
            # the shortfall test below cannot see.
            behind = (f"requests in flight climbed from {first:.3g} to {last:.3g} across the "
                      f"window, so completions ran below arrivals")
        elif completed is not None and completed < offered / 2.0:
            # Total collapse, for the run that produced too few samples for the series above to be
            # read, or that was behind so far it never got going. Half is not a tuned threshold: it
            # is the point past which the shortfall cannot be the window's tail cutting off requests
            # still in flight, which is the one benign explanation for completions trailing what was
            # sent.
            behind = (f"only {completed:.0f} of {offered:.0f} requests completed, with no "
                      "failures")
        if behind is None:
            return None
        return f"the device was working ({busy}) and did not keep up: {behind}"

    @property
    def boot_failed_at(self) -> int | None:
        """The wider degree that came up, if this row is a FAILED BOOT rather than a driven round.

        Written only by a descent round (`SoloProfiler.measure`, impl §14.2), and returned here for
        the one inference a failed boot licenses about degrees other than its own: a shard at a
        NARROWER degree is strictly larger against the same whole card, so an engine that could not
        start in a whole card here cannot start in one there either. That direction is arithmetic, and
        it is the reason a descent may stop at the first boot failure instead of spending a round per
        remaining degree. Nothing analogous holds for the *overload* ground above -- a wider degree
        has measured WORSE than a narrower one on this hardware (the MoE anchor at tp8 against tp4) --
        so the two grounds are reported separately rather than through one predicate.
        """
        boot = (self.notes or {}).get("boot_failed") or {}
        if not boot.get("gate") or boot.get("booted_at_tp") is None:
            return None
        return int(boot["booted_at_tp"])

    @property
    def pool_below_stated_floor(self) -> tuple[int, int] | None:
        """`(offered, stated)` KV tokens when this row was driven on a pool the fleet did not ask for.

        A descent round relaxes the KV floors so the degree can be reached at all, so a band degree
        whose refusal came FROM the floor is necessarily driven on a smaller pool than the fleet file
        states (`SoloProfiler.measure`, impl §14.2). The row still means what it measured -- the
        target was met and the offered rate was delivered on that pool -- but the fleet file's floor
        is a statement of demand, and only whoever wrote it can say whether the smaller pool is
        acceptable. Reported so the mode can say that out loud, because the solver will keep refusing
        this degree while the file states the larger floor.
        """
        d = (self.notes or {}).get("descent") or {}
        if not d or d.get("clears_stated_floor") is not False:
            return None
        return int(d.get("offered_tokens") or 0), int(d.get("stated_floor_tokens") or 0)

    def meets(self, slo) -> bool | None:
        """Whether this solo run met a target. None when the row cannot answer.

        Both ceilings are checked when both are stated, and a target with neither is not a target
        (`TenantSlo` refuses to be constructed empty). None propagates: a row with a suppressed
        ITL cannot rule a degree out, and treating it as a failure would push a tenant to a wider
        shard count on the strength of a measurement that did not happen.
        """
        if slo is None:
            return None
        if not self.usable:
            return None
        want_itl = getattr(slo, "itl_mean_ms", None)
        want_ttft = getattr(slo, "ttft_p95_ms", None)
        if want_itl is not None and self.itl_mean_ms > want_itl:
            return False
        if want_ttft is not None:
            if self.ttft_p95_ms is None:
                return None
            if self.ttft_p95_ms > want_ttft:
                return False
        return True

    def to_dict(self) -> dict:
        return {"model_id": self.model_id, "tp": self.tp, "isl_tokens": self.isl_tokens,
                "osl_tokens": self.osl_tokens, "rate_hz": self.rate_hz,
                "envelope_gib": self.envelope_gib, "serving": self.serving.canonical(),
                "image_count": self.image_count, "image_pixels": self.image_pixels,
                "itl_mean_ms": self.itl_mean_ms,
                "ttft_p95_ms": self.ttft_p95_ms, "delivered_fraction": self.delivered_fraction,
                "delivery_passed": self.delivery_passed, "activation_gib": self.activation_gib,
                "reserve_gib": self.reserve_gib,
                "kv_tokens": self.kv_tokens, "reserve_source": self.reserve_source,
                "run_id": self.run_id,
                "run_token": self.run_token, "gpu_node": self.gpu_node,
                "gpu_ids": list(self.gpu_ids), "observation_key_hash": self.observation_key_hash,
                "measured_at": self.measured_at, "notes": dict(self.notes)}

    @classmethod
    def parse(cls, d: dict) -> SoloProfile:
        if "serving" not in d:
            # Refused rather than defaulted, and this is the refusal that matters most in the whole
            # store: a row read back with guessed engine flags sets minimal TP, and minimal TP is
            # the §0 baseline. The chart defaults that would be guessed are exactly the values that
            # were found to be load-bearing (`serving.py`), so guessing them would assert that a row
            # measured at some other batch cap is a row measured at this one.
            raise ValueError(
                "this profile row does not say what engine flags it was measured under. It was "
                "written before max_num_seqs, max_num_batched_tokens, max_model_len, dtype and "
                "extra_args were part of a row's identity, and they decide what its latency means. "
                "Re-measure it.")
        return cls(model_id=d["model_id"], tp=int(d["tp"]), isl_tokens=int(d["isl_tokens"]),
                   osl_tokens=int(d["osl_tokens"]), rate_hz=float(d["rate_hz"]),
                   envelope_gib=float(d["envelope_gib"]),
                   serving=ServingConfig.from_dict(d["serving"]),
                   # Absent reads as no images, unlike `serving` above, and the asymmetry is the
                   # point: a row written before this field existed was measured by a harness that
                   # could only send text, so reading it as text is reading it correctly rather
                   # than guessing.
                   image_count=int(d.get("image_count", 0)),
                   image_pixels=int(d.get("image_pixels", 0)),
                   itl_mean_ms=d.get("itl_mean_ms"), ttft_p95_ms=d.get("ttft_p95_ms"),
                   delivered_fraction=d.get("delivered_fraction"),
                   delivery_passed=d.get("delivery_passed"),
                   activation_gib=d.get("activation_gib"), reserve_gib=d.get("reserve_gib"),
                   # Absent stays absent in both, and for the same reason the field exists: a row
                   # written before this was recorded was driven under *some* pool and *some*
                   # reserve, and nobody can say which. `None` and `""` are what `_comparable`
                   # reads as "this row does not say", which makes it withhold the ratio. Reading
                   # a missing key as the compared run's own conditions would be asserting the
                   # very equality the gate exists to check.
                   kv_tokens=(None if d.get("kv_tokens") is None else int(d["kv_tokens"])),
                   reserve_source=d.get("reserve_source") or "",
                   run_id=d.get("run_id", ""), run_token=d.get("run_token", ""),
                   gpu_node=d.get("gpu_node", ""), gpu_ids=tuple(d.get("gpu_ids") or ()),
                   observation_key_hash=d.get("observation_key_hash", ""),
                   measured_at=d.get("measured_at", ""), notes=dict(d.get("notes") or {}))


@dataclass(frozen=True)
class TpResolution:
    """What the profile store can and cannot say about one tenant's minimal TP.

    `certain` is the field to read before quoting `tp`. It is true only when every candidate degree
    from `floor_tp` up to `tp` was profiled and *measured to miss* the target. With a gap in
    the sweep, `tp` is the smallest degree known to work and some narrower degree might also have --
    which would mean a baseline still carrying GPUs it does not need.

    `floor_tp` is the narrowest degree the caller was entitled to consider, and since 2026-09-18
    that is the WEIGHT floor rather than the memory floor (impl §14.2): the weights are the only term
    of the envelope that cannot come back smaller than the arithmetic states, so they are the only
    term a degree may be *eliminated* on. Everything under it is `Infeasible` by arithmetic and was
    never a candidate; everything between it and the memory floor is a degree the arithmetic doubts
    and nobody measured, which is why those show up in `below` as unchecked instead of not existing.
    """

    tenant: str
    model_id: str
    status: str
    tp: int | None = None
    floor_tp: int | None = None
    certain: bool = False
    evidence: SoloProfile | None = None
    # (tp, code, why) for degrees below `tp`, in ascending order. `code` is MISSED or one of the
    # UNCHECKED codes; `why` is prose for a person. Reading the code rather than the prose is what
    # keeps `certain` and `missing` from being decided by a wording change: they were computed by
    # comparing `why` against the literal "not profiled" until 2026-09-09, so giving an unchecked
    # degree a more honest message would silently have promoted an upper bound to a minimum.
    below: tuple = ()

    @property
    def solo_itl_ms(self) -> float | None:
        """The reference a co-resident measurement of this tenant is compared against."""
        return None if self.evidence is None else self.evidence.itl_mean_ms

    @property
    def missing(self) -> tuple:
        """Degrees below the answer that nobody checked, which is what makes `tp` an upper bound."""
        return tuple(tp for tp, code, _ in self.below if code in UNCHECKED)

    def as_dict(self) -> dict:
        return {"tenant": self.tenant, "model_id": self.model_id, "status": self.status,
                # `floor_tp` and not `memory_floor_tp`, which is what this key was called until
                # 2026-09-18: the figure in it is now the weight floor, and a key that names the
                # wrong floor is how a reader concludes the band between the two was ruled out.
                "minimal_tp": self.tp, "floor_tp": self.floor_tp, "certain": self.certain,
                "solo_itl_ms": self.solo_itl_ms,
                "below": [{"tp": tp, "code": code, "why": why} for tp, code, why in self.below]}


# Paths already reported as unreadable, so `ProfileStore.all` says it once per process. Module
# scope rather than instance scope because the store is re-opened, not passed around.
_UNREADABLE_REPORTED: set = set()


class ProfileStore:
    """One environment's worth of solo profiles, as files under a local root."""

    def __init__(self, root: str | Path, key: ProfileKey | str):
        self.root = Path(root)
        if isinstance(key, ProfileKey):
            self.key: ProfileKey | None = key
            self.key_hash = key.hash
        else:
            self.key = None
            self.key_hash = key

    @property
    def key_dir(self) -> Path:
        return self.root / "profiles" / self.key_hash

    @property
    def solo_dir(self) -> Path:
        return self.key_dir / "solo"

    def write_key(self) -> Path:
        if self.key is None:
            raise ValueError("no ProfileKey: this store was opened by hash only")
        path = self.key_dir / "key.json"
        payload = self.key.key_json()
        if path.exists() and path.read_text() != payload:
            raise ValueError(
                f"{path} exists and does not match this key. Same hash, different contents means "
                "two environments have been filed together, and every profile in here would be "
                "attributed to the wrong machine.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
        return path

    # --- rows ---------------------------------------------------------------

    def path_for(self, profile_or_id) -> Path:
        pid = profile_or_id if isinstance(profile_or_id, str) else profile_or_id.id
        return self.solo_dir / f"{pid}.json"

    def write(self, profile: SoloProfile) -> Path:
        """Write one row, replacing any existing one for the same identity.

        Overwriting, unlike `ObservationStore.append_observation`, and the asymmetry is the same
        one `write_feasibility` draws: an observation is a historical fact about a fleet, while a
        profile row is the *current best statement* about a model on this hardware. A re-profile
        supersedes. The observation behind it is untouched and append-only in its own store, and
        `run_id` here points at it, so nothing is lost by superseding -- the history stays where
        history belongs.
        """
        path = self.path_for(profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    def read(self, model_id: str, tp: int, isl_tokens: int, osl_tokens: int,
             rate_hz: float, serving: ServingConfig,
             image_count: int = 0, image_pixels: int = 0) -> SoloProfile | None:
        path = self.path_for(profile_id(model_id, tp, isl_tokens, osl_tokens, rate_hz, serving,
                                        image_count, image_pixels))
        if not path.exists():
            return None
        return SoloProfile.parse(json.loads(path.read_text()))

    def all(self) -> list:
        """Every readable row. Unreadable ones are logged once and skipped.

        Skipping is safe here and is not safe for observations: a missing observation makes the
        agent re-propose a placement that already failed, while a missing profile row makes a
        degree look unprofiled -- which is reported as a gap and measured again, at worst costing
        a run it did not strictly need.

        **Said once per path per process, not once per call.** One `plan` opens the store and walks
        it half a dozen times -- once to resolve each tenant's degree, again for the baseline, again
        for the placements -- so a per-call message turns a handful of superseded rows into pages of
        identical errors and buries whatever else was logged. The condition is a property of the file
        and does not change while the process runs, so repeating it adds nothing.
        """
        if not self.solo_dir.exists():
            return []
        out = []
        for path in sorted(self.solo_dir.glob("*.json")):
            try:
                out.append(SoloProfile.parse(json.loads(path.read_text())))
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                if str(path) not in _UNREADABLE_REPORTED:
                    _UNREADABLE_REPORTED.add(str(path))
                    log.error("profile %s could not be read (%s); treating that degree as "
                              "unprofiled", path, exc)
        return out

    def for_model(self, model_id: str, isl_tokens: int, osl_tokens: int,
                  rate_hz: float, serving: ServingConfig,
                  image_count: int = 0, image_pixels: int = 0) -> dict:
        """`{tp: SoloProfile}` for one model at one shape, payload, rate and engine configuration.

        `serving` is matched on, not merely recorded. This is the lookup `minimal_tp` walks, so a
        row admitted here sets a tenant's minimal degree and therefore the §0 baseline; admitting
        one measured under a different batch cap would hand the baseline a number taken on a
        different engine. The image payload is matched on for the same reason and with more force:
        a text row admitted for a multimodal tenant would certify a degree at a fraction of the
        prefill work that tenant does.
        """
        want = (str(model_id), int(isl_tokens), int(osl_tokens),
                round(float(rate_hz), RATE_DECIMALS), serving.digest,
                int(image_count), int(image_pixels))
        return {p.tp: p for p in self.all()
                if (p.model_id, p.isl_tokens, p.osl_tokens,
                    round(p.rate_hz, RATE_DECIMALS), p.serving.digest,
                    p.image_count, p.image_pixels) == want}

    def reserve_by_tp(self, model_id: str) -> dict:
        """`{tp: measured reserve GiB per shard}` for one model, for the loader to price with.

        **Merged into the fleet, not offered to a person to paste.** This method used to be
        `memory_facts` and returned the activation term beside this one, under an argument that has
        since been overturned twice. The argument was that a store the reader of the fleet file
        cannot see, silently changing how a candidate is scored, is worse than an over-safe default,
        so the number belonged in the file with a `provenance:` pointer beside it. What overturned
        it: the same objection applies with more force to the *shard count*, which is resolved from
        this very store and reported rather than stated (`fleetfile.resolve_tp`) -- so "the reader
        cannot see it" is answered by printing what was merged, which is what the loader now does.
        And a number a person pastes is a number a person can paste wrongly, into a file whose
        engine may not be the one that measured it, which is the hazard a second engine makes real.

        The activation half is gone rather than merged: nothing writes `SoloProfile.activation_gib`
        and nothing can, because separating activation from reserve needs the engine's own account of
        its start-up memory and a driven round cannot see inside the utilisation budget
        (`agent/profiler.profile_from_record`). The safe fallback stands for it instead.

        **The largest row wins when a degree has several.** Rows differ by rate, shape and engine
        configuration, and the reserve genuinely depends on the last of those -- CUDA-graph pools
        scale with the batch cap. Matching on the serving digest instead would be sharper and would
        also find nothing for most tenants, since a fleet's rider is rarely profiled under the
        anchor's flags. The maximum is the safe reading: a larger reserve buys a smaller KV pool,
        which over-states occupancy, which is the direction an error should point.
        """
        out = {}
        for p in self.all():
            if p.model_id != model_id or p.reserve_gib is None:
                continue
            prev = out.get(p.tp)
            if prev is None or float(p.reserve_gib) > prev:
                out[p.tp] = float(p.reserve_gib)
        return out


def minimal_tp(store: ProfileStore, tenant, *, floor_tp: int | None,
               candidates) -> TpResolution:
    """The smallest profiled TP degree at which this tenant meets its own target.

    Walks the candidate degrees upward from `floor_tp` -- the WEIGHT floor, below which a shard's
    weights alone are over the card, so there is nothing to measure and nothing a measurement could
    change (`TpResolution`, impl §14.2). Returns the first degree
    whose solo profile meets the target, and says whether the walk was complete: a gap below the
    answer means a narrower degree was never measured, so the answer is an upper bound on minimal
    TP rather than minimal TP. One kind of gap closes itself -- a degree under a FAILED BOOT is ruled
    out by the same arithmetic the boot failure settles (`boot_failed_at`), and is reported as not
    fitting rather than as unmeasured.

    Deliberately does not fall back to the stated `tp`, and does not fall back to the floor. Both
    would return an integer that reads like a measured minimum, which is the mistake §0 exists to
    prevent. `status` says which question could not be answered and the caller decides.
    """
    if tenant.slo is None or tenant.slo.itl_mean_ms is None:
        # No ITL ceiling: nothing distinguishes one degree from another for this tenant, so there
        # is no minimum to find. A tenant like this cannot be certified either (`TenantSlo`).
        return TpResolution(tenant=tenant.name, model_id=tenant.model_id, status=NO_TARGET,
                            floor_tp=floor_tp)
    profiles = store.for_model(tenant.model_id, tenant.isl_tokens, tenant.osl_tokens,
                               tenant.rate_hz, tenant.serving,
                               tenant.image_count, tenant.image_pixels)
    below, first_ok, evidence = [], None, None
    boot_out = None          # the widest degree a failed boot ruled out; see after the loop
    # `candidates` is walked in the order given, not re-sorted. Every caller passes
    # `candidate_tp_degrees`, which is already narrowest-first, so the sort was redundant -- and it
    # was the one line here that requires a degree to be a NUMBER. A degree is on its way to being a
    # shape rather than an integer (a tensor-parallel width and a pipeline depth, so a walk can widen
    # into a second node), and a shape has no total order for `sorted` to find: only the list itself
    # knows which candidate is the next-cheapest. Taking the order from the list is what lets that
    # change be made in one function instead of in every walk that reads it.
    for k in candidates:
        if floor_tp is not None and k < floor_tp:
            continue
        p = profiles.get(k)
        if p is None:
            below.append((k, NOT_PROFILED, "not profiled"))
            continue
        # A row that exists but cannot be cited says so in its own words. Reporting it as "not
        # profiled" was actively misleading: it sends somebody to re-run a degree that already ran,
        # without telling them the run needs *fixing* rather than repeating. Both are unchecked, so
        # the arithmetic below treats them alike -- it is only the message that differs.
        bad = p.unusable_reason
        if bad is not None:
            # ...unless it rules the degree out without a latency number. Then it is checked, and
            # saying so is what keeps the answer above it a measured minimum rather than an upper
            # bound: the walk widened past this degree on evidence, so the report must not turn
            # round and call the degree unmeasured. The row is still never cited for latency -- it
            # cannot be `first_ok`, having no latency to meet a target with.
            narrow = p.too_narrow_reason
            if narrow is not None:
                if p.boot_failed_at is not None:
                    boot_out = k if boot_out is None else max(boot_out, k)
                below.append((k, TOO_NARROW, f"too narrow -- {narrow}"))
                continue
            below.append((k, bad[0], f"profiled but unusable -- {bad[1]}"))
            continue
        if p.meets(tenant.slo):
            first_ok, evidence = k, p
            break
        # Name the ceiling that was actually breached, not just the ITL one. `meets` checks both,
        # so a row can miss on TTFT with its ITL comfortably inside target -- and the first version
        # of this message reported that as "missed: 73.536 ms ITL solo" against an 80 ms ceiling,
        # which reads as a passing number labelled a failure. Whoever sees this line is deciding
        # whether to change the fleet file, and they cannot if it points at the wrong quantity.
        breached = []
        want_itl = getattr(tenant.slo, "itl_mean_ms", None)
        want_ttft = getattr(tenant.slo, "ttft_p95_ms", None)
        if want_itl is not None and p.itl_mean_ms > want_itl:
            breached.append(f"{p.itl_mean_ms:.3f} ms ITL solo over a {want_itl:.3f} ms ceiling")
        if want_ttft is not None and p.ttft_p95_ms is not None and p.ttft_p95_ms > want_ttft:
            breached.append(
                f"{p.ttft_p95_ms:.0f} ms TTFT p95 over a {want_ttft:.0f} ms ceiling")
        below.append((k, MISSED, "missed: " + " and ".join(breached)))
    if boot_out is not None:
        # A boot that could not start in the whole card at `boot_out` settles every narrower degree
        # too, because a narrower shard is a bigger one against the same card (`boot_failed_at`).
        # Carrying that through matters for exactly the reason the descent exists: without it, a
        # descent that stopped at the first failed boot would leave the degrees under it reading
        # "not profiled", and the answer above would be an upper bound forever -- the only way to
        # clear it being GPU rounds spent to re-derive arithmetic.
        below = [(k, DOES_NOT_FIT,
                  f"does not fit -- the engine could not start in a whole card at tp{boot_out}, and "
                  f"a tp{k} shard is larger still") if k < boot_out and c == NOT_PROFILED
                 else (k, c, why) for k, c, why in below]
    if first_ok is None:
        status = UNPROFILED if any(c in UNCHECKED for _, c, _ in below) else NONE_MEETS
        return TpResolution(tenant=tenant.name, model_id=tenant.model_id, status=status,
                            floor_tp=floor_tp, below=tuple(below))
    return TpResolution(tenant=tenant.name, model_id=tenant.model_id, status=RESOLVED,
                        tp=first_ok, floor_tp=floor_tp,
                        certain=all(c in CHECKED for _, c, _ in below),
                        evidence=evidence, below=tuple(below))
