"""A fleet as data: YAML in, `FleetSpec` and `ValidityKey` out.

The agent searches whatever fleet it is handed, so the fleet itself must not be Python. A YAML
file means a new fleet is a new file rather than a new code path, and it means the thing that
was searched is a checkable artifact sitting next to the observations it produced.

Three things are deliberately *not* in the fleet file:

  - **The GPU UUID map.** It is read from the `<lease>-gpu-uuids-<node>` ConfigMap the daemon
    publishes on the node the run captured. Writing UUIDs by hand is the one error on this stack
    with no symptom: a mis-keyed MPS memory limit references nothing and is silently ignored, so
    the tenant runs with no envelope at all and the record looks fine (finding 9).
  - **The node.** A `node:` block used to name one, and since 2026-09-04 a run *captures* a node
    instead: it installs its own daemon, and whichever node that pod lands on is the node
    (agent/capture.py). What the file states is `hardware.instance_type`, and everything else
    about the machine -- GPU count, GPU product, device memory per GPU, addressable MiB, the
    enforced power limit, the link matrix -- is looked up from the hardware store, where it was
    observed or measured on a real node (`hardware.py`, `store/hwfacts.py`), and then checked
    against the node that was actually leased. Those last three are also the store's own address
    (`store/environment.py`), which is why a `validity_key:` block is refused rather than read:
    the environment is observed, never typed.
  - **Anything the solver decides.** No file here names a GPU assignment or a memory envelope.
    If a fleet file could pin those, it would stop being a search problem.

**A `models:` entry is an id.** It was five fields and a nested block, and everything that has been
taken out of it came out for one reason: the format's correctness rested on a paragraph a person
wrote. Five such fields are gone now.

Three went with `onboard` (impl §14.4) because a *GPU* produced them: `activation_gib`,
`measured_reserve_gib` and the `provenance:` block that pointed at the fact sheet the first two were
pasted from. The rule they violated is the one that makes a second serving engine possible at all --
a fleet file a person fills in by transcribing figures out of one engine's start-up report is a
fleet file that cannot be written for another engine without another such report.

Two more went with Fix 2 (impl §14.8) because a *person* produced them, by hand, from a file they
could have been read from: `weight_gib` and the `kv:` block of `{num_layers, num_kv_heads, head_dim}`.
Both now come from the checkpoint itself, fetched and cached by `store/hubfacts.py` -- `config.json`
for the geometry, the safetensors headers by HTTP range request for the bytes, no weight downloaded.
That is a better source than a careful person for a reason worth stating: the transcription was not
merely laborious, it was *wrong in ways the file could not express*. The `kv:` product assumes every
layer holds a KV page, which a hybrid family breaks by 3.76x, and `config.json` has always carried
the `layer_types` field that says which layers do. A hand-written `weight_gib` has to pick between
three quantities that differ -- file bytes, checkpoint tensor bytes, and what an engine resides -- and
the answer §14.2 needs is the third, as a lower bound, which means finding the speculative-decoding
head in a 36,000-tensor checkpoint and leaving it out. GLM-4.5-Air's is 3.98 GiB.

What is left about the page size is one override: `kv_bytes_per_token`, still the `derived` tier
(nothing in a file a person edits is anything else, which is the same rule that deleted `tp:`), and
now logged against the derived figure whenever the two disagree. It exists because a hybrid stack also
keeps a per-sequence recurrent state, so no single byte count is right at two context lengths;
`store/hubfacts.py` derives that state and nothing prices it yet, so the fleet that runs that family
states the number its own context needs and the log says it overrode a derivation.

Each tenant states three things about its own load and none of them is defaulted: the model, the
offered rate and sequence shape, and the SLO (design §2). The first two are in the validity key
and the third deliberately is not, and that split is a separate decision from where they are
declared: a target does not change what the hardware does, so it cannot change which measurements
may be compared -- see `spec.TenantSlo`. A rate and a shape do.

**The shard count is the fourth thing every tenant has and the one thing it may not state.** Since
2026-09-08 a `tp:` line is *refused* rather than honoured, and the degree is resolved instead: the
memory floor from arithmetic (`solver/baseline.py`), then the smallest degree at or above that
floor whose *solo* measurement meets the tenant's own target (`store/profile.py`). Three reasons,
in the order they were argued:

  - **A shard count is a latency claim, and this file cannot settle a latency claim.** Every other
    field here is checkable against something outside it: weights against the safetensors on disk,
    KV geometry against the model's own `config.json`, a rate and a shape against what the harness
    sent. `tp: 4` asserts that tp2 would miss this tenant's target, and only a measurement knows
    that. It is the one input whose truth this file cannot carry.
  - **It is the number the result is scored against.** impl §0's baseline is `sum(tp)` with no GPU
    shared, so a degree wider than the target requires puts GPUs into the *baseline*, and a search
    scored against that baseline is credited with GPUs that re-sharding alone would have freed.
    That failure is silent, and it fails *upward* -- in the direction that flatters the result.
  - **Checking a stated degree costs the same measurements as computing it.** To show `tp: 4` is
    minimal you have to show tp2 misses, which is a solo run at tp2; resolution walks up from the
    memory floor and stops at the first degree that passes, which is the same runs in the same
    order. So accepting-and-checking buys no GPU time over computing. All it buys is a code path
    where the check can later be skipped.

Resolution therefore requires a profile store with the relevant rows in it, and the loader refuses
with the exact `profile` command to run rather than inventing a degree. `tensor_parallel` stays in
the validity key, because a record must say which degree it was taken at -- that is provenance, not
input. What is deliberately *not* here is an escape hatch for a fleet that wants a degree pinned
for reasons outside the model ("pack around what I already run"). That is a placement constraint,
not a shard count, and it belongs in the report as a cost against the measured baseline rather than
as an input to it. Nobody has asked for it, and an escape hatch for a hypothetical is how a
hand-written degree gets back in.

A `workload:` block survives for the one thing that really is fleet-wide, the delivery gate. A
file that still puts `isl_tokens` or `tokenizer_path` there is refused rather than lifted: the
first would drive every tenant at one shape, the second would build every tenant's prompt with
one model's tokenizer, and both would then be named per tenant in the validity key as though
somebody had asked for them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from .. import engines
from ..solver.baseline import candidate_tp_degrees, memory_floor_tp, weight_floor_tp
from ..solver.memory import ModelMemory, kv_bytes_per_token
from ..store.hubfacts import HubUnavailable
from ..store.profile import (DOES_NOT_FIT, NO_TARGET, NOT_PROFILED, RESOLVED, UNPROFILED,
                             ProfileKey, TpResolution, minimal_tp)
from ..store.validity_key import ValidityKey
from ..tiers import TIER_DERIVED, TIERS
from .deploy import _run
from .hardware import Hardware, hardware_for
from .render import DEFAULT_DAEMON_RELEASE, DEFAULT_NAMESPACE
from .spec import (AGENT_ENVELOPE_STEP_MIB, FleetSpec, NodeInventory, TenantSlo,
                   TenantTemplate, WorkloadSpec)

log = logging.getLogger(__name__)

# The only keys an `slo:` block may carry. Checked rather than ignored: a target the loader
# silently dropped would read downstream as "this tenant stated no target", which is a
# different and much worse thing than an error -- the search would decline to certify a tenant
# whose target was sitting right there in the file, misspelt.
SLO_KEYS = ("ttft_p95_ms", "itl_mean_ms", "tolerance_pct")


class FleetFileError(Exception):
    pass


def load_slo(tenant_name: str, d) -> TenantSlo | None:
    """`slo:` for one tenant, or None when the block is absent.

    Absent means absent. There is no default target and no inferred one -- see `TenantSlo`.

    `tolerance_pct` is the exception, and only because it is not a target: absent means the
    fleet did not state a policy, so `spec.DEFAULT_TOLERANCE_PCT` applies and the report says
    which of the two it used. An absent *ceiling* cannot be defaulted that way, because there is
    nothing to be tolerant of.
    """
    if d is None:
        return None
    if not isinstance(d, dict):
        raise FleetFileError(
            f"tenant {tenant_name!r}: slo must be a mapping of "
            f"{{{', '.join(SLO_KEYS)}}}; got a {type(d).__name__}")
    unknown = sorted(set(d) - set(SLO_KEYS))
    if unknown:
        raise FleetFileError(
            f"tenant {tenant_name!r}: slo has unknown key(s) {unknown}. Known keys are "
            f"{list(SLO_KEYS)}. Refused rather than ignored, because a dropped target looks "
            "exactly like a tenant that never stated one.")
    try:
        return TenantSlo(
            ttft_p95_ms=None if d.get("ttft_p95_ms") is None else float(d["ttft_p95_ms"]),
            itl_mean_ms=None if d.get("itl_mean_ms") is None else float(d["itl_mean_ms"]),
            tolerance_pct=(None if d.get("tolerance_pct") is None
                           else float(d["tolerance_pct"])))
    except (TypeError, ValueError) as exc:
        raise FleetFileError(f"tenant {tenant_name!r}: bad slo -- {exc}") from None


def _engine_flag_json(extra_args, flag: str):
    """The JSON value of `--flag=<json>` or `--flag <json>` in a tenant's `extra_args`.

    Both spellings, because both reach vLLM identically and a file may use either. None means the
    flag is absent; a value that will not parse is an error here rather than at boot, since a
    malformed one costs a deploy to discover.
    """
    args = [str(a) for a in extra_args]
    raw = None
    for i, a in enumerate(args):
        if a.startswith(f"{flag}="):
            raw = a[len(flag) + 1:]
        elif a == flag and i + 1 < len(args):
            raw = args[i + 1]
        else:
            continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FleetFileError(f"{flag} is not valid JSON: {raw!r} ({exc})") from exc
    return None


def _load_image_spec(name: str, t: dict, extra_args, engine: str) -> tuple[int, int]:
    """A tenant's per-request image payload, checked against the engine flags beside it.

    Returns `(image_count, image_pixels)`; `(0, 0)` is a text tenant. Stated as an `images:` block
    rather than inferred from `extra_args`, because the flags are a *capacity* and this is a
    *workload*: `--limit-mm-per-prompt` says what the engine will accept and reserves an encoder
    budget for, while these two numbers say what the harness sends. Inferring one from the other
    would make the request shape a side effect of a memory decision.

    Both directions of disagreement are refused, and each one is a defect this repo has actually
    filed:

      - **More images, or bigger ones, than the engine admits.** The server clamps or rejects, so
        the run measures a shape nobody chose. The pixel bound is the sharper half: outside
        `min_pixels`/`max_pixels` the preprocessor silently *resizes*, and then the driver sends
        one resolution while the engine costs another -- invisible in every record.
      - **An engine configured for images driven with text.** This is the defect that prompted the
        whole change: `qwen35-multimodal.yaml` passed `--limit-mm-per-prompt={"image": 24}`, paid
        for the encoder budget in its KV arithmetic, and the harness sent a wall of text. Every
        latency in that run would be a text latency filed under a multimodal fleet. Refused rather
        than warned, because a warning is exactly what nobody read the first time.
      - **A tenant that sends images on a stack that cannot serve them.** Checked here because it is
        checkable here: the engine is named in this file and its capability is a fact about the
        adapter, so the file can be refused before `plan` prints a line. The alternative is finding
        out from a run that leased a node, installed a daemon, pulled weights, and then collected
        400s.

    Nothing here predicts the visual token count. That needs the patch size and the spatial merge,
    which are the model family's business; the harness measures it against the live engine
    (`containers/colocation-harness/harness/workload.py`).
    """
    block = t.get("images")
    if block is not None and not isinstance(block, dict):
        raise FleetFileError(
            f"tenant {name!r}: images: must be a mapping with `count` and `pixels`, got "
            f"{type(block).__name__}. One number cannot say both how many images a request "
            "carries and how big each one is, and the two have different consequences -- count "
            "against the engine's per-prompt limit, pixels against its encoder budget.")
    block = block or {}
    unknown = sorted(set(block) - {"count", "pixels"})
    if unknown:
        raise FleetFileError(
            f"tenant {name!r}: images: has unknown key(s) {unknown}. Only `count` and `pixels` "
            "are read, so a misspelling here would silently leave the payload at its default of "
            "no images -- which is the failure this block exists to prevent.")
    count = int(block.get("count", 0) or 0)
    pixels = int(block.get("pixels", 0) or 0)
    if count < 0 or pixels < 0:
        raise FleetFileError(
            f"tenant {name!r}: images.count and images.pixels must both be >= 0; got "
            f"{count} and {pixels}")
    if bool(count) != bool(pixels):
        raise FleetFileError(
            f"tenant {name!r}: images.count={count} and images.pixels={pixels} -- state both or "
            "neither. Images of no size and a size nothing is sent at are each half a workload.")

    limit = _engine_flag_json(extra_args, "--limit-mm-per-prompt") or {}
    if not isinstance(limit, dict):
        raise FleetFileError(
            f"tenant {name!r}: --limit-mm-per-prompt must be a JSON object like "
            f'\'{{"image": 4, "video": 0}}\', got {limit!r}')
    allowed_images = int(limit.get("image", 0) or 0)
    allowed_video = int(limit.get("video", 0) or 0)
    if allowed_video:
        raise FleetFileError(
            f"tenant {name!r}: --limit-mm-per-prompt allows {allowed_video} video item(s), and "
            "the harness has no route for video. The engine would reserve an encoder budget for "
            "frames nothing sends, and the KV pool that budget came out of is what every memory "
            'number in this file is derived against. State "video": 0.')

    if count and not engines.get(engine).serves_images:
        # The alternatives are asked of the registry rather than named here, for the reason every
        # other engine fact is: this module is not allowed to know one stack's name.
        can = [n for n in engines.names() if engines.get(n).serves_images]
        raise FleetFileError(
            f"tenant {name!r} sends {count} image(s) per request and is served by {engine!r}, which "
            "cannot serve images. This is the serving stack's frontend, not the model, so no flag "
            f"and no smaller image turns it on: state `engine:` on this tenant naming one of "
            f"{', '.join(can)}. A tenant may name its own engine, so the rest of the fleet can stay "
            "where it is and a mixed-engine fleet is a normal arrangement rather than a workaround. "
            "Say why in the file, because the reason is not visible from the outside.")

    if count:
        if allowed_images < count:
            raise FleetFileError(
                f"tenant {name!r} sends {count} image(s) per request but "
                f'--limit-mm-per-prompt allows {allowed_images}. The engine rejects or clamps the '
                "extras, so the run would measure a shape nobody asked for. Raise the limit -- and "
                "recompute the envelope, because the encoder budget it reserves is part of it.")
        kwargs = _engine_flag_json(extra_args, "--mm-processor-kwargs") or {}
        if not isinstance(kwargs, dict):
            raise FleetFileError(
                f"tenant {name!r}: --mm-processor-kwargs must be a JSON object, got {kwargs!r}")
        lo, hi = kwargs.get("min_pixels"), kwargs.get("max_pixels")
        if hi is not None and pixels > int(hi):
            raise FleetFileError(
                f"tenant {name!r} sends {pixels}-pixel images but max_pixels is {int(hi)}. The "
                "preprocessor resizes them down without saying so, and then the driver's stated "
                "resolution and the engine's actual one differ with nothing in the record able to "
                "tell them apart -- the same class of error as an approximate ISL, and less "
                "visible.")
        if lo is not None and pixels < int(lo):
            raise FleetFileError(
                f"tenant {name!r} sends {pixels}-pixel images but min_pixels is {int(lo)}. The "
                "preprocessor scales them UP, so the request costs more visual tokens than its "
                "size suggests and the extra cost is attributed to nothing.")
    elif allowed_images:
        raise FleetFileError(
            f"tenant {name!r} is configured for {allowed_images} image(s) per prompt but states no "
            "images: block, so the harness would drive this multimodal engine with pure text. "
            "Every latency in the run would be a text latency filed under a multimodal fleet, and "
            "the encoder budget the limit reserves would be paid for and unused. State "
            "`images: {count: N, pixels: P}`, or drop --limit-mm-per-prompt and the "
            "--mm-processor-kwargs beside it and recompute the envelope.")
    return count, pixels


def _load_kv_page(name: str, d: dict, manifest=None) -> tuple[int, str]:
    """A model's KV bytes per token at TP=1, and which tier the number is.

    Derived from the checkpoint's own `config.json`, fetched and cached by `store/hubfacts.py`, and
    the one thing a fleet file may still say about it is an override:

      - nothing stated: the page is `attention layers x KV heads x head_dim x 2 x dtype`, where
        "attention layers" is the count `layer_types` marks as caching K and V rather than
        `num_hidden_layers`. That distinction is the whole reason this moved out of the file: the
        all-layers product a person used to write is 3.76x too large on a hybrid family, and the
        config has always said which layers are which.
      - `kv_bytes_per_token: <int>`: the page size itself, as one number, overriding the derivation.
        Architecture-blind, so a family whose geometry nothing here models correctly costs one line
        rather than a code change. It is the WHOLE page: a hybrid's state term is not added on top of
        it (`load_model_memory`), because the number a person has to state one of these from is a page
        an engine delivered, which already has the state in it.

    **The override is logged with the derived figure beside it, and it is the one number in this file
    that may still disagree with the checkpoint.** It used to be refused alongside a `kv:` block --
    two hand-written claims about one quantity, and a loader picking a winner between them would make
    the file stop meaning what it says. Against a *derived* figure there is no such ambiguity: one is
    arithmetic on the published config and the other is a person overruling it, which is a legible
    thing to do and is reported as what it is. Both are still the `derived` tier (impl §14.4).

    **What it existed for is now derived, and what remains is narrower.** It was here because a hybrid
    stack keeps a per-sequence recurrent state as well as per-token pages, nothing priced the state,
    and the attention-only product is 4x too small on such a family. Impl §14.9 prices it: the page is
    the attention geometry plus the state divided by the tenant's own `max_model_len`, which lands
    within 6% of every page the engine delivered to `qwen35-multimodal.yaml` and is *low*, so it
    may choose a degree and not refuse one (§14.2). The four overrides that fleet carried are gone.
    What the field is still for is a family whose geometry this arithmetic does not model at all -- and
    for that it is a stopgap with a cost, since one flat number per model cannot be right for two
    tenants at two contexts.
    """
    if "kv" in d:
        raise FleetFileError(
            f"model {name!r} states a `kv:` block of dimensions, which this format no longer accepts: "
            "layer count, KV heads and head dim are read from the checkpoint's own config.json "
            "(store/hubfacts.py), keyed by the model id and commit already in this file. Delete the "
            "block. Transcribing them was how the all-layers product got written for a hybrid model "
            "whose config says only a quarter of its layers cache K and V -- the file could not "
            "contradict the assumption, and the config could. State `kv_bytes_per_token` only to "
            "override the derivation.")
    if "provenance" in d:
        raise FleetFileError(
            f"model {name!r} states a `provenance:` block. It named the fact sheet a person pasted "
            "`kv_bytes_per_token` and `activation_gib` out of, and both the sheets and the mode that "
            "wrote them are gone (impl §14.4): the numbers came from an engine's own account of its "
            "start-up memory, which is not evidence this design reads. Delete the block. The byte "
            "count may stay and is now the `derived` tier whatever the deleted block claimed, "
            "because a number in a file a person edits is a number a person could have computed.")
    derived = None
    if manifest is not None:
        geometry = manifest.kv_geometry()
        derived = kv_bytes_per_token(num_layers=geometry.kv_layers,
                                     num_kv_heads=geometry.num_kv_heads,
                                     head_dim=geometry.head_dim)
        log.info("kv geometry for %s from config.json: %d of %d layer(s) cache K and V, "
                 "%d KV head(s) x %d head_dim -> %d B/token at tp1%s", name, geometry.kv_layers,
                 geometry.kv_layers + geometry.state_layers, geometry.num_kv_heads,
                 geometry.head_dim, derived,
                 (f"; and {geometry.state_layers} layer(s) keep a recurrent state estimated at "
                  f"{geometry.state_bytes_per_seq / (1 << 20):.1f} MiB per SEQUENCE, which each "
                  "tenant's own max_model_len amortises into its page (§14.9)")
                 if geometry.is_hybrid else "")
    explicit = d.get("kv_bytes_per_token")
    if explicit is not None:
        try:
            page = int(explicit)
        except (TypeError, ValueError):
            raise FleetFileError(
                f"model {name!r}: kv_bytes_per_token must be an integer number of bytes at TP=1; "
                f"got {explicit!r}") from None
        if page <= 0:
            raise FleetFileError(
                f"model {name!r}: kv_bytes_per_token must be positive; got {page}")
        if derived is not None and page != derived:
            # Said out loud every time, because an override that nobody notices is indistinguishable
            # from a derivation that nobody checked. Which way it points matters: above the derived
            # figure is conservative for occupancy, below it prices a pool the engine will not fit in.
            log.info("model %s overrides the derived %d B/token with a stated %d (%.2fx): the fleet "
                     "file is asserting a page size the checkpoint's config does not imply",
                     name, derived, page, page / derived)
        # `derived`, with no way for the file to say otherwise. The file used to be able to claim
        # `measured` through a `provenance:` block, and the claim was even true of the numbers that
        # motivated the field -- an engine had reported them. That is the problem: a measurement is
        # something a GPU did under an instrument this design owns, and an engine's start-up log is
        # neither. A byte count here is a number a person wrote, and the weakest tier is the honest
        # one for it; what it may do is choose which degree to boot first, never refuse one (§14.2).
        return page, TIER_DERIVED
    if derived is None:
        raise FleetFileError(
            f"model {name!r} has no KV page size: nothing read its config.json (no hub store was "
            "passed to `load`) and the entry states no `kv_bytes_per_token` to stand in. There is no "
            "default: a guessed page size sets the scale of every KV floor in the fleet.")
    return derived, TIER_DERIVED


def load_model_memory(name: str, d: dict, reserves: dict | None = None,
                      manifest=None) -> ModelMemory:
    """One `models:` entry as a `ModelMemory`: an id, and everything else read from somewhere else.

    `reserves` is `{tp: GiB per shard}` from `ProfileStore.reserve_by_tp`, and None means no store
    was available -- `plan` with no profiles yet, or a caller that only wants the geometry. None and
    an empty dict mean the same thing to the solver and both leave the safe fallback in force; the
    difference is only whether anything was asked.

    `manifest` is a `store/hubfacts.HubManifest` for this entry's checkpoint, and it supplies the two
    things the file stated until impl §14.8's Fix 2: the KV geometry (`_load_kv_page`) and the weight
    bytes. `weight_gib` becomes a LOWER BOUND on resident weights -- checkpoint tensor bytes less a
    duplicated tied head and less any speculative-decoding draft block -- which is what §14.2 requires
    of the one term allowed to refuse a degree. On the eight checkpoints in `fleets/` it reproduces
    three hand-written figures to 0.003 GiB and moves five: GLM-4.5-Air-FP8 falls 3.99 GiB because its
    47th layer is a multi-token-prediction head no boot without speculative decoding loads, and the
    four Qwen3.5 checkpoints fall 0.28 to 0.57 for the same reason at smaller scale.

    **Neither the reserve nor the activation term may be stated in the file, and the two are refused
    for different reasons.** The reserve is measurable and measured -- resident memory minus the
    utilisation budget, on a solo round, off the driver's own books -- so it comes from the store,
    keyed by the model and the degree, and the loader logs what it merged. The activation term is not
    measurable that way at all: it is a slice of the inside of the utilisation budget and the engine
    fills what it leaves with KV cache, so from outside the budget activation and KV are one number.
    Reading it needs the engine's own start-up report. So it has no replacement, the documented safe
    fallback carries it, and that costs the shipped fleets 0.23 to 0.81 GiB per shard of priced KV
    pool while moving no fleet's GPU count -- but it does cost one arrangement, and the reserve merged
    here pays part of it back. Netted per tenant against the fleets as they shipped: the 0.6B riders
    rise 0.45 GiB/shard, `glm-air-anchor-riders`'s largest rider rises far enough to stop pairing with
    any other on an anchor shard (808 enumerated placements down to 600), `qwen-anchor-riders`'s anchor
    FALLS 0.59 on a reserve measured at tp2, and `qwen35-multimodal`'s `caption` falls 0.33. Read the
    `measured reserve offered` line a run logs for which half applied where.
    """
    page, tier = _load_kv_page(name, d, manifest)
    for gone, why in (
            ("weight_gib",
             "it is arithmetic on the checkpoint's own safetensors headers, read without downloading "
             "a weight (store/hubfacts.py) and cached under the commit this file names. A hand-written "
             "figure had to pick between three quantities that differ -- file bytes, checkpoint tensor "
             "bytes, and the bytes an engine actually resides -- and every shipped fleet needed a "
             "paragraph to say which. What is derived now is the third, as a lower bound, which is "
             "what §14.2 requires of the only term allowed to refuse a degree"),
            ("activation_gib",
             "the only source for it was an engine's own report of its start-up memory, fitted by "
             "the deleted `onboard` mode (impl §14.4). A driven round cannot separate it from the KV "
             "pool, so there is nothing to merge from the store either: the documented safe fallback "
             "in `solver/memory.ACTIVATION_GIB_DEFAULT` carries it, and a plan priced on the fallback "
             "is conservative rather than wrong -- an over-stated activation under-states KV tokens, "
             "which over-states occupancy"),
            ("measured_reserve_gib",
             "it is measured, and now merged from the profile store instead of pasted here "
             "(`ProfileStore.reserve_by_tp`). Keying it on the model and the degree in one place "
             "means a second fleet on the same hardware gets it for free, and a stale paste cannot "
             "outlive the row it came from. `profile` files it; this loader reads it and logs it")):
        if gone in d:
            raise FleetFileError(
                f"model {name!r} states `{gone}`, which this format no longer accepts: {why}. "
                "Delete the field.")
    if "model_id" not in d:
        raise FleetFileError(f"model {name!r} is missing 'model_id'")
    if manifest is None:
        raise FleetFileError(
            f"model {name!r} has no checkpoint manifest, so its weight bytes are unknown. Nothing "
            "read its config.json or its safetensors headers -- `load` was given no hub store. There "
            "is no default and there is no field to state one in: a guessed weight figure decides how "
            "many GPUs this fleet needs.")
    # Already net of the tied head, so `tied_embedding_gib` stays 0.0 and nothing subtracts it twice.
    # The field survives for a caller that builds a `ModelMemory` from a gross figure by hand; this
    # loader no longer has a gross figure to hand anybody.
    return ModelMemory(
        model_id=d["model_id"],
        weight_gib=manifest.resident_bytes() / float(1 << 30),
        kv_bytes_per_token_tp1=page,
        # The other half of a hybrid page, priced per sequence and amortised over the context the
        # tenant serves at -- so it is deliberately NOT folded into `page` here: this loader does not
        # know the context, and two tenants of one checkpoint need not share it. `TenantTemplate`
        # binds it (`spec.py`), and until it does nothing can price this model's KV.
        # Zero when the entry overrides the page, because an override is the WHOLE page -- the figure
        # a person states is one an engine delivered, state included -- and adding the state term to
        # it again would charge it twice. Which is also the override's remaining cost: a stated page
        # is flat across contexts, so a checkpoint served at two `max_model_len`s is priced right for
        # at most one of them (`_load_kv_page`).
        state_bytes_per_seq=(0 if d.get("kv_bytes_per_token") is not None
                             else manifest.kv_geometry().state_bytes_per_seq),
        # Not to size anything -- to know at which degree the attention half stops sharding
        # (`ModelMemory.kv_bytes_per_token`). Carried even for an overridden page, because the cliff
        # is a fact about the architecture and an override does not change where it is.
        num_kv_heads=manifest.kv_geometry().num_kv_heads,
        measured_reserve_gib=dict(reserves) if reserves else None,
        kv_bytes_per_token_tier=tier)


def uuid_configmap_prefix(release: str = DEFAULT_DAEMON_RELEASE) -> str:
    """The base name the daemon's publisher sidecar builds its ConfigMap from.

    Derived from the release name, not a constant. It was
    `"mps-daemon-gpu-uuids-"` while there was exactly one hand-installed daemon, and the chart's
    own default is `<release>-gpu-uuids` -- so the constant was only ever right for one release
    name. Once each run installs its own daemon (agent/capture.py) it is wrong for every run but
    the hand-installed one, and wrong here means reading another node's UUIDs.
    """
    return f"{release}-gpu-uuids-"


def node_name_to_configmap(node_name: str, release: str = DEFAULT_DAEMON_RELEASE) -> str:
    """The daemon builds the ConfigMap name by replacing dots with dashes."""
    return uuid_configmap_prefix(release) + node_name.replace(".", "-")


def read_gpu_uuids(node_name: str, namespace: str = DEFAULT_NAMESPACE,
                   kubectl: str = "kubectl",
                   release: str = DEFAULT_DAEMON_RELEASE) -> dict:
    """The ordinal -> UUID map off the cluster, and nothing else.

    Keys are `gpu<N>`, so the ordinal is recovered from the key rather than from iteration
    order: a dict's order is not a promise, and pairing the two lists by position is exactly
    where an off-by-one becomes a silently lifted memory limit.

    Split out from `read_node_inventory` because a caller that only wants to label UUIDs in a
    report (`inspect --ordinals`) should not have to name a device capacity to get them. A
    `NodeInventory` carries one and it is measured, not defaulted, so inventing one for a display
    path would put a fabricated capacity one refactor away from a placement.
    """
    cm = node_name_to_configmap(node_name, release)
    out = _run([kubectl, "get", "configmap", cm, "-n", namespace, "-o",
                "jsonpath={.data}"], timeout_s=60.0)
    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        raise FleetFileError(f"ConfigMap {cm} has no readable .data: {exc}") from None
    by_ordinal = {}
    for k, v in data.items():
        if not k.startswith("gpu"):
            raise FleetFileError(f"{cm} has an unexpected key {k!r}; expected gpu<N>")
        by_ordinal[int(k[3:])] = v
    return by_ordinal


def read_node_inventory(node_name: str, *, device_total_gib: float, gpu_ids=None,
                        namespace: str = DEFAULT_NAMESPACE,
                        kubectl: str = "kubectl",
                        release: str = DEFAULT_DAEMON_RELEASE) -> NodeInventory:
    """The ordinal -> UUID map, as an inventory sized for the card it was read from.

    `device_total_gib` is required and comes from a measurement (`store/hwfacts.py`, by way of
    `hardware_for`). The ConfigMap says nothing about how big the GPUs are, so this function
    cannot supply it and must not appear to.

    Reading this is necessary but not sufficient, and `capture._read_inventory` is where the
    rest happens: the ConfigMap is written at runtime by a sidecar, so its existence proves
    only that *some* daemon published it. What proves it describes the node this run holds is
    its `ownerReference` pointing at this run's daemon pod.
    """
    by_ordinal = read_gpu_uuids(node_name, namespace=namespace, kubectl=kubectl, release=release)
    wanted = sorted(by_ordinal) if gpu_ids is None else [int(g) for g in gpu_ids]
    missing = [g for g in wanted if g not in by_ordinal]
    if missing:
        cm = node_name_to_configmap(node_name, release)
        raise FleetFileError(f"{cm} has no UUID for GPU ordinal(s) {missing}")
    return NodeInventory(node_name=node_name, gpu_ids=tuple(wanted),
                         uuids=tuple(by_ordinal[g] for g in wanted),
                         device_total_gib=float(device_total_gib))


def planned_node(hardware: Hardware) -> NodeInventory:
    """The shape of a machine nobody has captured yet.

    Real GPU count and real device capacity, and **no UUIDs** -- so the solver can settle
    feasibility offline while anything that would deploy a tenant fails loudly
    (`NodeInventory.uuid_for`). The node name says what it is rather than pretending to be one,
    because it gets printed and a plausible-looking hostname here would invite somebody to
    believe a plan was made against a specific machine.
    """
    return NodeInventory(node_name=f"any {hardware.instance_type}", gpu_ids=hardware.gpu_ids,
                         uuids=(),
                         device_total_gib=hardware.require_total_gib("planning against this shape"))


def fleet_instance_type(doc: dict) -> str:
    """The instance type a fleet file names, with the whole of the `hardware:` block's validation.

    Split from `load_hardware` because its callers want only the string. `load` with a lease already
    holds a `Hardware` that was verified against the node, so re-resolving one from the store would
    create a second answer to a question the lease has already settled; and the refusals below are
    about the shape of the file, which is worth reading before anything is resolved at all.
    """
    if "node" in doc:
        raise FleetFileError(
            "`node:` is no longer a fleet-file block. A run does not name its node, it CAPTURES "
            "one: it installs its own mps-daemon release, and whichever node that pod lands on "
            "is the node (agent/capture.py). Replace it with\n"
            "    hardware:\n      instance_type: g6e.48xlarge\n"
            "The size of one GPU is looked up per card in the hardware store, where it was "
            "measured (store/hwfacts.py, agent/hardware.py), rather than being a figure copied into "
            "each fleet file. `node.gpu_ids` is gone with no replacement: a run holds every GPU "
            "on the node it leases, and leaving some empty is a placement the solver may choose "
            "-- not a restriction the fleet declares.")
    hw = doc.get("hardware") or {}
    if not hw.get("instance_type"):
        raise FleetFileError(
            "hardware.instance_type is required. It is the whole of what a fleet file says about "
            "the machine: GPU count, GPU product and device memory per GPU are all looked up from it "
            "(agent/hardware.py), and the node itself is captured rather than named.")
    unknown = sorted(set(hw) - {"instance_type", "max_nodes"})
    if unknown:
        raise FleetFileError(
            f"hardware: has unknown key(s) {unknown}. Only instance_type and max_nodes belong "
            "here -- everything else about the hardware is an observation or a measurement, so "
            "another key would be a place to write a hardware fact by hand, which is the thing the "
            "store exists to replace.")
    fleet_max_nodes(doc)
    return str(hw["instance_type"])


def fleet_max_nodes(doc: dict) -> int:
    """How many nodes of the instance type the fleet may be spread over, 1 when it does not say.

    A budget the fleet's owner sets, not a hardware fact, which is why it may sit beside
    `instance_type`. Each tenant still runs on one node, so the fleet is split into groups and each
    group is an ordinary single-node fleet (`solver/partition.py`). It is not part of any validity
    key: a group is measured, keyed and recalled exactly as a fleet of those tenants on one node is,
    whatever budget it was split under.
    """
    n = (doc.get("hardware") or {}).get("max_nodes", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise FleetFileError(
            f"hardware.max_nodes is a whole number of nodes, at least 1; got {n!r}. Leave it out "
            "for a fleet on one node.")
    return n


def load_hardware(doc: dict, index, *, gpu_count: int | None = None,
                  gpu_product: str | None = None) -> Hardware:
    """The `hardware:` block resolved against the hardware store.

    `index` is a `store/hwfacts.HardwareIndex` and is not optional. The fleet file names an instance
    type and nothing else about the machine, so without the store there is nothing to resolve it
    against -- and the alternative, a table of GPU counts and capacities in source, is what this
    signature exists to have deleted.
    """
    return hardware_for(fleet_instance_type(doc), index,
                        gpu_count=gpu_count, gpu_product=gpu_product)


def _tenant_engine(doc: dict, t: dict) -> str:
    """Which serving stack one tenant is served by: its own `engine:`, else the fleet's, else vLLM.

    Resolved through the registry rather than merely copied, so a misspelled engine is refused while
    the fleet file is being read -- before anything is installed, and with the legal names listed.
    The alternative is a `helm upgrade` against a chart directory that does not exist, which arrives
    as a Helm path error naming no fleet and no tenant.
    """
    name = str(t.get("engine", doc.get("engine", engines.DEFAULT_ENGINE)))
    try:
        engines.get(name)
    except KeyError as exc:
        raise FleetFileError(
            f"tenant {t.get('name')!r}: {exc.args[0]}") from None
    return name


def _tenant_image(doc: dict, t: dict) -> str:
    """One tenant's image: its own, else the fleet's, else **its engine's** default.

    The last term is why this is a function. An image belongs to an engine -- a vLLM image is not a
    default a Triton fleet should silently inherit -- so a fleet that names an engine and no image
    gets that engine's image, and a fleet that names neither is unchanged from before engines
    existed.
    """
    return str(t.get("image", doc.get("image",
                                     engines.get(_tenant_engine(doc, t)).default_image)))


def fleet_engines(doc: dict) -> frozenset:
    """Every engine this fleet names. Plural on purpose: nothing here requires it to be one.

    The engine is keyed *per tenant* (`ServingConfig.engine`), exactly as TP and the batch caps are,
    so a fleet with two engines in it keys honestly. Whether such a fleet is a sensible experiment is
    a different question and not this loader's to answer -- `qwen35-multimodal-2.yaml` is one, and it
    is mixed because one of its tenants sends images and one of its stacks cannot serve them.
    """
    return frozenset(_tenant_engine(doc, t) for t in (doc.get("tenants") or []))


def fleet_engine_images(doc: dict) -> dict:
    """engine name -> the one image that engine is served from in this fleet.

    Refuses an engine served from two images, which is the one arrangement that makes a fleet's key
    unreadable. See `fleet_image` for why that particular refusal and not a broader one.
    """
    if not (doc.get("tenants") or []):
        raise FleetFileError("no tenants: section")
    out: dict = {}
    for t in doc["tenants"]:
        engine, image = _tenant_engine(doc, t), _tenant_image(doc, t)
        prior = out.setdefault(engine, image)
        if prior != image:
            raise FleetFileError(
                f"engine {engine!r} is served from two images in this fleet ({prior!r} and "
                f"{image!r}, the second on tenant {t.get('name')!r}). The key names the images a run "
                "used and names each tenant's engine, so one image per engine is what makes a "
                "tenant's image recoverable from its key; two images behind one engine is a "
                "difference the key records the existence of and cannot attribute. If both images "
                "are wanted, they are two runs.")
    return out


def fleet_image(doc: dict) -> str:
    """The engine image (or images) this fleet's rows are keyed by, from the tenants up.

    One implementation because the validity key names this and the *profile* key names the same one:
    two derivations of "which image is this fleet" would let a fleet be profiled under one and
    searched under another, which is exactly the comparison a key exists to prevent. It is only ever
    a key: the image a tenant is actually deployed from is its own (`_tenant_image`, carried on the
    tenant and written into its chart values), so nothing pulls this string.

    **A fleet may name more than one image, and then this names all of them,** sorted and comma-joined
    -- because a mixed-engine fleet is necessarily a mixed-image fleet, and refusing the second image
    refused the arrangement. What used to be here refused it: the key held one image, so a fleet with
    a vLLM tenant and a Triton tenant "could not be keyed honestly". The correct reading is the
    opposite -- a key that names both images and, per tenant, which engine served it (`serving`) says
    exactly what ran, and each tenant's image follows from its engine because `fleet_engine_images`
    refuses an engine served from two.

    Why joining is honest rather than a fudge, in the two places the string lands:

      - **The validity key.** It says "these rows were taken on a node where both of these images
        ran". That is what happened. A single-image fleet joins one element and so hashes exactly as
        it always did, which is the property that matters most here: re-keying the existing history
        would lose it silently.
      - **The hardware key**, whose directory a device sheet is written under. A composite never
        equals a single image, so a sheet measured under a mixed fleet is never reused *as* a
        single-image sheet. That is the conservative direction -- it can cost a re-measurement, and it
        cannot let a number describe a condition it was not taken in. `HardwareIndex.device_total`
        reads across every key anyway, deliberately: a card's capacity is not conditioned by the
        image.
    """
    return ",".join(sorted(set(fleet_engine_images(doc).values())))


def _refuse_validity_block(doc: dict) -> None:
    """A fleet file may no longer say anything about its own key. Says where every field went.

    The block was the last place in this format where a person typed a fact about a machine, and it
    was the one fact a fleet author is in the worst position to know: `driver` is a property of a
    node nobody has leased yet. Its two remaining fields left for different reasons, and both are in
    `store/environment.py`:

      - `driver` is not in the key at all any more. The environment is named by what the device
        reports about itself -- addressable MiB, enforced power limit, link matrix, product, count,
        engine image, sharing mode -- all read off the leased node, which is both finer than a
        driver string (it catches a capped card) and coarser in the right direction (a driver bump
        that changed nothing measurable no longer discards the store). The version is still recorded
        on every row as provenance.
      - `metrics_version` is a constant in code (`store/environment.METRICS_VERSION`), bumped
        deliberately when a definition of ITL, TTFT or delivered fraction changes. Nobody writing a
        fleet could answer it, and every fleet that ever stated it stated the same value.

    Refused rather than ignored. A block that is read by nothing still reads as the thing that keys
    the store, so the first driver upgrade after this change would have somebody editing a string
    and believing they had re-keyed their measurements.
    """
    if "validity_key" not in doc:
        return
    stated = sorted(doc.get("validity_key") or {})
    raise FleetFileError(
        f"validity_key: is no longer part of a fleet file (this one states {stated or 'nothing'}). "
        "The key is now derived, and every field it used to hold has a source that is not a person: "
        "`driver` is gone from the key entirely -- the environment is what the leased node reports "
        "about itself (addressable MiB, enforced power limit, link matrix, product, count, engine "
        "image, sharing mode), and the driver version is recorded on each row as provenance; "
        "`metrics_version` is a constant in store/environment.py; `instance_type` comes from "
        "`hardware.instance_type`, which is also what selects the machine; `tensor_parallel`, "
        "`sequence_shapes`, `offered_rates_hz` and `serving` are read off the tenants below, so the "
        "key cannot name a shape, a load, a shard count or an engine flag the fleet was not served "
        "at. Delete the block: nothing in it is read, and nothing it used to say is lost.")


def check_doc(doc: dict) -> None:
    """Every refusal `load` can make from the fleet file alone: no hardware store, no hub, no profiles.

    `load` calls this first, so a run refuses these exactly as before. Split out so that a check with
    no store -- the self-test, over example fleets for instance types nobody here has leased -- can
    still hold such a file to the rules a file can break, rather than skipping it whole or inventing
    a shape for it. What it cannot check without the store (the instance shape, the checkpoint, a
    hostable placement) it leaves to `load`.
    """
    # A file that still carries the block is a file whose author believes they control the key, and
    # every message below reads differently to somebody holding that belief.
    _refuse_validity_block(doc)
    fleet_instance_type(doc)
    w = doc.get("workload") or {}
    legacy = sorted(set(w) & {"isl_tokens", "osl_tokens"})
    if legacy:
        raise FleetFileError(
            f"workload.{' and workload.'.join(legacy)} is no longer a fleet-wide setting. The "
            "sequence shape is per tenant (design §8.3), so state isl_tokens and osl_tokens on "
            "each tenant. Refused rather than applied to every tenant, because a fleet-wide "
            "shape is not a simplification of a mixed fleet -- it is a different problem, and "
            "the validity key would then name a shape nobody chose per tenant.")
    if "tokenizer_path" in w:
        raise FleetFileError(
            "workload.tokenizer_path is no longer a fleet-wide setting. Each tenant's tokenizer "
            "is its own model's, which is now the default -- so in the common case delete this "
            "line and state nothing. Set tenants[].tokenizer_path only to override it. Refused "
            "rather than applied to every tenant, because one tokenizer building every prompt "
            "makes ISL exact for one model and approximate for the rest, while the validity key "
            "names the requested ISL for all of them.")
    models = doc.get("models") or {}
    if not models:
        raise FleetFileError("no models: section")
    tenants = doc.get("tenants") or []
    for t in tenants:
        key = t.get("model")
        if key not in models:
            raise FleetFileError(
                f"tenant {t.get('name')!r} refers to model {key!r}, which is not in models: "
                f"({sorted(models)})")
        for required in ("isl_tokens", "osl_tokens"):
            if t.get(required) is None:
                raise FleetFileError(
                    f"tenant {t.get('name')!r} does not state {required}. Every tenant states "
                    "its own request shape and nothing defaults it (design §2): a shape the "
                    "loader invented would be reported back as though somebody had asked for "
                    "it, and it decides both the KV floor and what the harness sends.")
        for gone, why in (
                ("min_kv_tokens",
                 "the minimum viable KV pool is now derived: `MIN_POOL_REQUESTS` requests at this "
                 "tenant's own max_model_len (spec.MIN_POOL_REQUESTS, impl §14.9). Every fleet that "
                 "stated it stated exactly 2x or 4x its own max_model_len -- the same arithmetic, "
                 "transcribed, and then obliged to move by hand whenever the context did. The floor "
                 "the rate and the target imply is separate and also derived "
                 "(`kv_tokens_at_target`); what a fleet may still state is the arrival burst both "
                 "are sized for, as `burst_quantile` or `peak_rate_hz`"),
                ("max_num_seqs",
                 "the batch cap is now derived from the burst occupancy of a slot -- "
                 "`rate x (ttft_p95 + osl x itl)`, at the arrival quantile -- floored at the serving "
                 "engine's own default (`engines/*.default_max_num_seqs`, impl §14.9). A "
                 "hand-written cap is a claim about queueing that this file cannot check and that "
                 "fails as a diverging queue rather than as a missed target: `search-r2hk3z` was "
                 "refused for a cap of 64 against an offered 82, and the GLM-4.5-Air anchor was "
                 "driven at 96 while its own 56 s TTFT p95 meant the slots, not the tokens, were "
                 "what ran out. State the load (`rate_hz`, `burst_quantile`, `peak_rate_hz`) and "
                 "the target (`slo:`); the cap follows from both")):
            if gone in t:
                raise FleetFileError(
                    f"tenant {t.get('name')!r} states `{gone}`, which this format no longer "
                    f"accepts: {why}. Delete the field.")
        if t.get("tp") is not None:
            raise FleetFileError(
                f"tenant {t['name']!r} states tp: {t['tp']}, which this file may no longer do. A "
                "shard count is a latency claim -- it asserts that a narrower degree would miss "
                "this tenant's target -- and a latency claim is the one thing nothing in this file "
                "can check. It is also the number the whole result is scored against: impl §0's "
                "baseline is sum(tp) with no GPU shared, so a degree one wider than the target "
                "needs hands the search a GPU it did not free, silently and in the flattering "
                "direction. Delete the line. The degree is then resolved as the smallest one whose "
                "SOLO measurement meets this tenant's slo.itl_mean_ms, which costs the same "
                "measurements that checking a stated degree would have cost -- run `profile` to "
                "take them. If the degree is pinned for a reason outside the model, that is a "
                "placement constraint and not a shard count; nothing here implements one yet, and "
                "an inflated baseline is not a way to express it.")
        engine = _tenant_engine(doc, t)
        _load_image_spec(t.get("name"), t, tuple(t.get("extra_args") or ()), engine)
        load_slo(t.get("name"), t.get("slo"))
        _tenant_image(doc, t)
    if not tenants:
        raise FleetFileError("no tenants: section")


def read_doc(path: str | Path) -> dict:
    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise FleetFileError(f"{path} does not contain a mapping")
    return doc


def profile_key(path: str | Path, index, *, gpu_count: int | None = None,
                gpu_product: str | None = None) -> ProfileKey:
    """The profile key a fleet file implies, without loading its tenants.

    Needed because of an ordering that looks circular and is not: every tenant's degree is resolved
    from the profile store, so the store has to be open before the tenants can be built -- while the
    *validity* key is only knowable after they are, since it names their shard counts. The profile
    key is exactly the part that does not depend on them (`store/profile.py`), so it can be read from
    the file's `hardware:` and `image:` blocks and the hardware store behind them.

    `index` is a `store/hwfacts.HardwareIndex` and is **not** optional, which is the change from when
    the key named the instance type as a string. The address is now what a node of that type reports
    about itself, so a type nobody has ever leased has no address and this refuses -- by way of
    `Hardware.environment`, whose message names the one run that fixes it. That refusal is the point:
    the alternative is opening a directory named after a machine nobody measured and filing rows in
    it.
    """
    doc = read_doc(path)
    _refuse_validity_block(doc)
    hardware = load_hardware(doc, index, gpu_count=gpu_count, gpu_product=gpu_product)
    return ProfileKey(env=hardware.environment(fleet_image(doc)))


def unresolved_tp(loaded: dict) -> list:
    """The tenants in a loaded fleet whose TP degree is a memory floor, not a measured minimum.

    Only ever non-empty under `allow_unresolved_tp`, and the one thing a caller that passes that
    flag has to ask: those tenants' degrees are arithmetic about what *fits*, and every number
    derived from them -- the baseline, every placement, every envelope -- inherits that.

    The rule lives here rather than in the caller because of what it must NOT catch. A tenant whose
    memory floor is already the widest degree the node offers is resolved without a measurement, and
    saying otherwise would report an established baseline as unestablished. `resolve_tp` records
    that case as RESOLVED for this reason; the belt-and-braces `getattr` covers a tenant carrying no
    resolution at all, which no path in `load` now produces.
    """
    resolutions = loaded["tp_resolutions"]
    return sorted(t.name for t in loaded["fleet"].tenants
                  if getattr(resolutions.get(t.name), "status", None) != RESOLVED)


def upper_bound_tp(loaded: dict) -> list:
    """The tenants whose degree is RESOLVED but is an upper bound on their minimal TP.

    The successor to the "stated more shards than the weights require" warning, which went away with
    `tp:`. A resolved degree above the memory floor is now the *expected* outcome for a tenant whose
    floor degree was measured to miss its target -- that is not an inflated baseline, it is a
    measured one. What still inflates a baseline is a degree that passed solo while some narrower
    degree was never measured at all: nothing rules the narrower one out, so `sum(tp)` may still
    carry GPUs this fleet does not need. `TpResolution.certain` is exactly that distinction
    (`store/profile.py`), so the flag is read from the resolution rather than re-derived from
    arithmetic, which cannot see it.
    """
    return sorted(n for n, r in loaded["tp_resolutions"].items()
                  if r.status == RESOLVED and not r.certain)


def resolve_tp(provisional: TenantTemplate, *, hardware: Hardware, profiles=None,
               allow_unresolved: bool = False) -> tuple:
    """The TP degree a tenant gets, since it may not state one. Returns `(tp, TpResolution | None)`.

    Two steps, in this order, because they answer different questions and only the first is free:

      1. **The memory floor**, from the tenant's weights and the KV pool its own target requires
         (`solver/baseline.py`). Below it a shard does not fit on one GPU at any envelope, so
         nothing below it is worth measuring.
      2. **The smallest degree at or above the floor whose solo measurement meets the target**
         (`store/profile.py`). Solo at the full envelope is the best case, so a degree that misses
         there cannot be rescued by any placement.

    `provisional` is the tenant as written with `tp` set to 1, which is legal for anything the
    floor scan does: the scan re-shards the request itself, and `kv_tokens_at_target` does not
    depend on TP (at TP=N each shard holds all the tokens and 1/N of each token's KV, so the token
    count is the TP-invariant).

    Raises rather than falling back. The whole point of resolving is that the degree is derived
    from a measurement; a fallback to the floor, to the widest degree, or to 1 would return an
    integer that every report downstream reads as a measured minimum -- impl §0's mistake, with
    better provenance and the same consequence.

    `allow_unresolved=True` is the exception, and belongs to the two modes whose job is the missing
    measurement itself: `profile`, which takes it -- refusing it for want of one would make the loop
    unenterable -- and `plan`, which says which degrees to take it at. It then yields the memory
    floor -- where profiling starts anyway, so a starting point rather than an answer -- together
    with the `TpResolution` saying it is not established. Both callers print that; `search` does not
    pass the flag, so no certified result reads an unmeasured degree as a minimum.
    """
    name = provisional.name
    total = hardware.require_total_gib(f"resolving tenant {provisional.name!r}'s TP degree")
    floor = memory_floor_tp(provisional.request, device_total_gib=total,
                            envelope_step_mib=AGENT_ENVELOPE_STEP_MIB,
                            gpu_count=hardware.gpu_count)
    candidates = candidate_tp_degrees(hardware.gpu_count)
    # The weight floor, separately from the memory floor, because the two answer different questions
    # (impl §14.2): the memory floor is where a boot is most likely to come up, and the weight floor
    # is the only one of the two a degree may be *eliminated* on. Both are needed below -- once to
    # decide whether "does not fit at any degree" is a refusal at all, and once to decide whether a
    # floor at the widest degree really leaves a measurement nothing to choose between.
    wf = weight_floor_tp(provisional.request.mem, device_total_gib=total, candidates=candidates)
    if floor.floor_tp is None:
        needed = floor.ruled_out[-1][1] if floor.ruled_out else floor.envelope_at_resolved_gib
        # Which half of that figure put it over decides whether this is a refusal at all (impl
        # §14.2). This is the FIRST refusal a fleet meets -- earlier than anything in `solver/cpsat`
        # -- so it was the one most able to close a fleet on a number nobody checked, and it said so
        # in as many words: "no placement and no measurement changes it".
        common = (f"tenant {name!r} does not fit on a {hardware.instance_type} at any TP degree: "
                  f"{needed:.2f} GiB per shard is needed even at tp{candidates[-1]} against "
                  f"{total:.2f} GiB per GPU ({floor.floor_detail}).")
        if wf.floor_tp is None:
            raise FleetFileError(
                f"{common} Its WEIGHTS alone are {wf.ruled_out[-1][1]:.2f} GiB per shard at "
                f"tp{candidates[-1]}, so no placement and no measurement changes it -- the fleet "
                "needs a wider machine or a smaller model.")
        if not allow_unresolved:
            raise FleetFileError(
                f"{common} BUT ITS WEIGHTS FIT: "
                f"{provisional.request.mem.weight_lower_bound_gib(candidates[-1]):.2f} GiB "
                f"per shard at tp{candidates[-1]} is under the card, so what does not fit is the KV "
                f"pool the target implies, plus the activation and reserve above it -- all computed, "
                f"and the KV page size behind them was 3.27x too large the one time it met a booted "
                f"engine (impl §14.2). Nothing here has ruled this tenant out, so this refusal is "
                f"THIS mode's -- a result may not be certified at a degree no measurement settled. "
                f"`plan` and `profile` load this fleet: run `profile` and let a boot at "
                f"tp{candidates[-1]} say whether it comes up, or lower its rate so the pool the "
                f"target implies is smaller. Neither needs a wider machine. What this refusal may "
                f"NOT be read as is 'this tenant cannot be served'.")
        # `plan` and `profile` load it instead, at the widest degree the node offers (impl §14.2).
        # Refusing them here was the exit code disagreeing with the prose: the message said nothing
        # had ruled the tenant out and then made it impossible to find out, since the two modes that
        # exist to resolve an open degree could not construct the fleet to resolve it in. The
        # arithmetic doubts every degree, so there is no floor to start from and the widest degree is
        # the one a boot is likeliest to come up at -- and if it does, the descent has somewhere to
        # go from there. `search` still refuses above: an unmeasured degree may not reach a result.
        log.warning("tenant %s's computed envelope fits at no TP degree on a %s (%.2f GiB/shard "
                    "needed even at tp%d against %.2f GiB), but its WEIGHTS fit from tp%d up, and "
                    "every term above them has measured smaller than stated at least once. Taking "
                    "tp%d to get it into the run: nothing here has ruled this tenant out, and only "
                    "a boot settles it.", name, hardware.instance_type, needed, candidates[-1],
                    total, wf.floor_tp, candidates[-1])
    # Where a walk should START, which is a different question from where a degree may be ELIMINATED
    # (impl §14.2). The memory floor when there is one; the widest degree the node offers when the
    # computed envelope doubts every degree, since then the roomiest is the best guess. Only prose
    # and starting points below use it -- `minimal_tp` is given the weight floor.
    start_tp = floor.floor_tp if floor.floor_tp is not None else candidates[-1]
    if floor.floor_tp == candidates[-1] and wf.floor_tp == candidates[-1]:
        # The one case where no measurement is needed, because no alternative exists: the floor is
        # already the widest degree the node offers, so every legal degree is this one. Resolving it
        # from arithmetic here is not the same as defaulting to the floor -- there is nothing for a
        # profile to choose between.
        #
        # Both floors have to agree for that to hold (impl §14.2). A memory floor at the widest
        # degree while the WEIGHTS fit somewhere narrower leaves degrees that only the computed terms
        # rule out -- a KV pool sized from a page size measured 3.27x too large, an activation
        # fallback measured 4.4x too large -- and calling those "forced by memory alone" closed the
        # question at the one place in the pipeline that spends no round to do it.
        log.info("tenant %s: tp%d, forced by memory alone (%s). No profile needed: it is the only "
                 "legal degree on a %s.", name, floor.floor_tp, floor.why, hardware.instance_type)
        # RESOLVED, and `certain`, with no evidence behind it -- the one resolution that is honestly
        # arithmetic. Every narrower degree was ruled out by what fits, and there is no wider one on
        # this machine, so a measurement could not change the answer. Recorded rather than returned
        # as None so a caller asking which degrees are floors gets told this one is not.
        return floor.floor_tp, TpResolution(
            tenant=name, model_id=provisional.model_id, status=RESOLVED, tp=floor.floor_tp,
            floor_tp=floor.floor_tp, certain=True,
            below=tuple((k, DOES_NOT_FIT, "does not fit") for k in candidates if k < floor.floor_tp))
    if provisional.slo is None or provisional.slo.itl_mean_ms is None:
        if allow_unresolved:
            log.warning("tenant %s states no itl_mean_ms target, so nothing can pick a minimal "
                        "degree for it. Taking tp%d to get it into the run; state a target for the "
                        "degree to mean anything.", name, start_tp)
            return start_tp, TpResolution(
                tenant=name, model_id=provisional.model_id, status=NO_TARGET,
                floor_tp=floor.floor_tp)
        raise FleetFileError(
            f"tenant {name!r} states no itl_mean_ms target, and a shard count cannot be stated "
            "instead. Resolving TP means finding the smallest shard count that meets a latency "
            "ceiling, so with no ceiling there is no minimum to find and nothing here will pick "
            "one. State slo.itl_mean_ms and let the degree be measured against it. (Memory alone "
            f"allows tp{floor.floor_tp} and up for this tenant.)")
    if profiles is None:
        if allow_unresolved:
            log.info("tenant %s: no profile store, so starting from tp%d (%s)",
                     name, start_tp, floor.why)
            return start_tp, TpResolution(
                tenant=name, model_id=provisional.model_id, status=UNPROFILED,
                floor_tp=floor.floor_tp,
                # Unprofiled from the WEIGHT floor up, not from the memory floor up (impl §14.2):
                # a degree only the computed terms exclude is a degree nobody measured, and listing
                # it here is what keeps the eventual answer an upper bound rather than a minimum.
                below=tuple((k, NOT_PROFILED, "not profiled")
                            for k in candidates if k >= wf.floor_tp))
        raise FleetFileError(
            f"tenant {name!r} has no profile store to resolve its TP degree from, and the fleet "
            "file cannot state one. Run `profile` against this fleet file first. (Memory allows "
            f"tp{floor.floor_tp} and up: {floor.why}.)")
    # Against the WEIGHT floor, not the memory floor (impl §14.2): `minimal_tp` treats everything
    # below the floor it is given as ruled out on a checked ground, and only the weights are a
    # checked ground. A degree the memory floor excludes and the weights allow is unmeasured, so it
    # belongs in `missing` -- where it makes the answer an upper bound and names the round that would
    # settle it -- rather than in the "does not fit" list, where it silently disappeared.
    res = minimal_tp(profiles, provisional, floor_tp=wf.floor_tp, candidates=candidates)
    if res.status == RESOLVED:
        if not res.certain:
            # Reported here and again in the search summary. tp is an *upper* bound on minimal TP
            # while a narrower degree is unmeasured, which means the baseline may still carry GPUs
            # this fleet does not need.
            log.warning("tenant %s: tp%d meets its target solo, but tp%s were never profiled, so "
                        "tp%d is an upper bound on its minimal TP rather than the minimum. The "
                        "baseline may still be inflated.", name, res.tp,
                        ", tp".join(str(k) for k in res.missing), res.tp)
        return res.tp, res
    if res.status == NO_TARGET:      # unreachable: the ITL ceiling was checked above.
        raise FleetFileError(f"tenant {name!r}: no ITL target to resolve TP against")
    detail = "; ".join(f"tp{k}: {why}" for k, _code, why in res.below) or "nothing profiled"
    if allow_unresolved:
        log.info("tenant %s: minimal TP is not established yet (%s), so starting from tp%d",
                 name, detail, start_tp)
        return start_tp, res
    if res.missing:
        raise FleetFileError(
            f"tenant {name!r}: its minimal TP is not established yet -- {detail}. Run `profile` to "
            f"measure tp{', tp'.join(str(k) for k in res.missing)} solo on a "
            f"{hardware.instance_type}. There is no way to assert the degree instead: that "
            "assertion is what the solo run checks.")
    raise FleetFileError(
        f"tenant {name!r} met its target at no measured TP degree -- {detail}. Every degree from "
        f"tp{wf.floor_tp}, the narrowest its weights allow, to tp{candidates[-1]} was measured "
        "ALONE, with the "
        "whole of each GPU to itself, which is the best case for this model at this rate and shape. "
        "So this target is not achievable on a "
        f"{hardware.instance_type} at all, let alone co-resident. Loosen the target, lower the "
        "rate, or take the tenant to different hardware.")


def load(path: str | Path, *, index=None, namespace: str = DEFAULT_NAMESPACE,
         kubectl: str = "kubectl", lease=None, profiles=None, hub=None,
         gpu_count: int | None = None, gpu_product: str | None = None,
         allow_unresolved_tp: bool = False) -> dict:
    """Load a fleet file. Returns `{"fleet", "key", "hardware", "raw", "tp_resolutions"}`.

    With no `lease` this is entirely offline and touches no cluster: the fleet is built against
    the *shape* of the instance type, resolved from the hardware store (`index`, required in that
    case), which is what `plan` and `calibrate` need and is why they cost nothing. With a `NodeLease`
    (agent/capture.py) the fleet is bound to the node that was actually captured, and two things the
    file merely *asserts* are checked against it.

    **A leased run prices against the lease's own `Hardware`, not against a freshly resolved one.**
    That machine was verified against the node (`Hardware.verify`) and may carry a device total the
    store did not have -- the driver's own `memory.total`, read off the node this run holds. Resolving
    a second `Hardware` here would produce a different answer to a question the lease has already
    settled, and the difference would be the size of every envelope.

    `profiles` also carries the measured per-shard reserve into the fleet (`reserve_by_tp`), which is
    the second thing the file used to state and no longer may. What was merged comes back in
    `reserves`, keyed by the local model name, so a mode can print what it priced with -- and is
    logged here for the same reason.

    `hub` is a `store/hubfacts.HubFactsStore` and is what makes a `models:` entry an id rather than a
    transcription: the weight bytes and the KV geometry are read from the checkpoint the entry names
    and cached under its commit. Required, in the sense that a fleet cannot load without one -- there
    is no default weight figure and no field left to state one in -- and injected rather than
    constructed here so that a test never reaches the network and a caller can point at another hub.
    An entry may pin `model_revision`; with none, the store resolves the repository's default branch
    once per run and falls back to its newest cached manifest if the hub cannot be reached.

    `profiles` is a `ProfileStore`, and since `tp:` was deleted from the file every tenant needs it:
    the shard counts are resolved from its solo rows, so without it the only fleet that loads is one
    whose memory floor is the widest degree the machine has. It is still typed optional, and
    omitting it is still a refusal rather than a default -- the two are not the same thing, and
    which one applies is `allow_unresolved_tp`.

    `allow_unresolved_tp` belongs to the two modes that exist to find out what to do next --
    `profile`, which is about to measure the degrees, and `plan`, which reports them as floors and
    says so. Every other mode leaves it False, so an unmeasured degree cannot reach a report as
    though it were a measured minimum. Callers that pass it True must read `tp_resolutions` and
    label those tenants; `load` warns, but a warning is not a report.
    """
    doc = read_doc(path)
    # First, before a node is looked at or a checkpoint is fetched: everything the file alone can
    # get wrong is refused while nothing else is known.
    check_doc(doc)

    wanted = fleet_instance_type(doc)
    if lease is None:
        if index is None:
            raise FleetFileError(
                "loading a fleet without a lease needs a hardware store to resolve "
                f"{wanted!r} against: how many GPUs it has was observed on a node, and how big one "
                "of them is was measured on one (store/hwfacts.py). Pass `index=HardwareIndex(root)`.")
        hardware = load_hardware(doc, index, gpu_count=gpu_count, gpu_product=gpu_product)
        node_inventory = planned_node(hardware)
    else:
        if lease.hardware.instance_type != wanted:
            raise FleetFileError(
                f"this fleet asks for {wanted} and the lease holds "
                f"{lease.hardware.instance_type} ({lease.node_name}). The instance type is what this "
                "run resolved its device facts from before any node existed, so the envelopes, the "
                "placements and the store directory already in play all describe a machine that is "
                "not the one in hand -- and the type is recorded on every row it files, so the "
                "records would name a machine they were not taken on.")
        hardware = lease.hardware
        node_inventory = lease.inventory

    w = doc.get("workload") or {}
    workload = WorkloadSpec(delivered_fraction_threshold=float(
        w.get("delivered_fraction_threshold", 0.99)))

    # The measured reserve, merged from the store rather than pasted into the file. Asked per model
    # id and not per tenant: two tenants may share a checkpoint under different engine flags (the
    # multimodal fleet's `caption` and `thumb` do), and one `ModelMemory` has to serve both, which is
    # also why `reserve_by_tp` takes the largest row at a degree rather than matching the flags.
    reserves = {}
    for name, d in (doc.get("models") or {}).items():
        model_id = d.get("model_id")
        if profiles is not None and model_id:
            found = profiles.reserve_by_tp(str(model_id))
            if found:
                reserves[name] = found

    # The checkpoint manifests, one per entry, fetched-or-cached before any envelope is priced. Keyed
    # by `(model_id, model_revision)` and not by the local name: two entries may point at one
    # checkpoint under different engine flags, and re-reading 36,000 tensor headers for the second
    # would be the same work for the same answer.
    manifests = {}
    for name, d in (doc.get("models") or {}).items():
        model_id, revision = d.get("model_id"), d.get("model_revision")
        if hub is None or not model_id:
            continue
        try:
            manifests[name] = hub.manifest(str(model_id), str(revision) if revision else None)
        except HubUnavailable as exc:
            raise FleetFileError(
                f"model {name!r} ({model_id}): {exc}") from None
    models = {name: load_model_memory(name, d, reserves.get(name), manifests.get(name))
              for name, d in (doc.get("models") or {}).items()}
    for name, manifest in sorted(manifests.items()):
        # The whole accounting, every run, because the number that reaches an envelope is a
        # subtraction and a reader who can only see the result cannot check the subtraction. The
        # cross-check against the hub's own dtype census is here rather than in a refusal: it is one
        # field of a response already fetched, it has agreed to the byte on every checkpoint in
        # `fleets/`, and a disagreement has no third opinion to break the tie with.
        drafts = manifest.draft_bytes()
        census = manifest.hub_parameter_bytes()
        log.info("weights for %s from %s@%s: %.4f GiB of checkpoint tensors, less %.4f tied head and "
                 "%.4f draft (%s) = %.4f GiB resident lower bound%s", name, manifest.model_id,
                 manifest.revision[:12], manifest.checkpoint_bytes / (1 << 30),
                 manifest.tied_head_bytes() / (1 << 30), sum(drafts.values()) / (1 << 30),
                 ", ".join(sorted(drafts)) or "none", manifest.resident_bytes() / (1 << 30),
                 "" if census in (None, manifest.checkpoint_bytes) else
                 f" -- NOTE the hub's own parameter census says {census} bytes against this file's "
                 f"{manifest.checkpoint_bytes}")
    # Said out loud because it is priced into every envelope below and the fleet file does not show
    # it. That is the objection the old `provenance:` paste answered, and this is the answer that does
    # not need a person: the same shape as the `TP resolved from solo profiles` line.
    #
    # Both halves, and logged even when nothing merged, because the half that is missing is the one
    # worth seeing: a model absent from the merged list prices the 1.0/1.5 GiB default, and so does a
    # model whose merged degrees do not include the degree it resolves to -- the degrees resolve below
    # this point, so this line names what was OFFERED, and `reserve_gib` marks what was taken
    # `measured@tpN` or `default(tpN)` per shard in the reports.
    _merged = "; ".join(f"{n}: " + ", ".join(f"tp{k}={v:.4g} GiB" for k, v in sorted(r.items()))
                        for n, r in sorted(reserves.items()))
    _defaulted = sorted(set(models) - set(reserves))
    log.info("measured reserve offered by solo profiles for %s; on the default: %s",
             _merged or "no model in this fleet", ", ".join(_defaulted) or "none")

    tenants, resolutions = [], {}
    for t in doc.get("tenants") or []:
        mem = models[t.get("model")]
        extra_args = tuple(t.get("extra_args") or ())
        engine = _tenant_engine(doc, t)
        image_count, image_pixels = _load_image_spec(t["name"], t, extra_args, engine)
        built = dict(
            name=t["name"], model_id=mem.model_id, model_path=t["model_path"],
            served_name=t["served_name"], rate_hz=float(t["rate_hz"]),
            isl_tokens=int(t["isl_tokens"]), osl_tokens=int(t["osl_tokens"]),
            # Checked against this tenant's own engine flags above, not merely read.
            image_count=image_count, image_pixels=image_pixels,
            mem=mem,
            slo=load_slo(t.get("name"), t.get("slo")),
            # The only thing this file may still say about a floor or a cap: the shape of the offered
            # load. Both optional, both `None` meaning "the fleet default quantile"; `spec.py` owns
            # what they do.
            peak_rate_hz=(None if t.get("peak_rate_hz") is None
                          else float(t["peak_rate_hz"])),
            burst_quantile=(None if t.get("burst_quantile") is None
                            else float(t["burst_quantile"])),
            # Absent means "this tenant's own weights", resolved in `TenantTemplate`. Not
            # defaulted to `model_path` here: one place decides it, and it is the one every
            # caller that builds a tenant goes through.
            tokenizer_path=t.get("tokenizer_path"),
            max_model_len=int(t.get("max_model_len", 4096)),
            max_num_batched_tokens=int(t.get("max_num_batched_tokens", 2048)),
            dtype=t.get("dtype", "bfloat16"),
            engine=engine,
            image=_tenant_image(doc, t),
            shm_size_gi=int(t.get("shm_size_gi", 16)),
            model_revision=str(t.get("model_revision", "")),
            extra_args=extra_args,
            extra_env=tuple(dict(e) for e in (t.get("extra_env") or ())))
        # Resolved, not defaulted. The provisional tp=1 is a placeholder the floor scan re-shards
        # immediately; nothing downstream ever sees it, because either `resolve_tp` returns a
        # measured degree or it raises.
        tp, res = resolve_tp(TenantTemplate(tp=1, **built), hardware=hardware,
                             profiles=profiles, allow_unresolved=allow_unresolved_tp)
        if res is not None:
            resolutions[t["name"]] = res
        tenants.append(TenantTemplate(tp=tp, **built))
    fleet = FleetSpec(tenants=tuple(tenants), node=node_inventory, workload=workload)

    key = ValidityKey(
        # The environment half, and not one field of it comes from this file. Offline it is the
        # instance shape a previous capture filed; with a lease it is what THIS node reported, because
        # `lease.hardware` is the reconciled one (`capture.reconcile_hardware`) -- and a node whose
        # device facts differ from the filed shape does not reach here at all, since the capture files
        # the new observation and stops the run rather than re-keying underneath a plan that has
        # already been printed (`capture.EnvironmentMoved`).
        env=hardware.environment(fleet_image(doc)),
        # All four always recorded, and all four taken from the tenants, so the key cannot claim a
        # load, a shape, a shard count or an engine flag the fleet was not served at.
        sequence_shapes=fleet.sequence_shapes,
        tensor_parallel=fleet.tp_degrees,
        serving=fleet.serving_configs,
        offered_rates_hz={t.name: t.rate_hz for t in tenants},
        model_revisions={t.served_name: t.model_revision for t in tenants
                         if t.model_revision})
    log.info("loaded fleet %s: %d tenants on %s, validity key %s",
             path, len(tenants), node_inventory.node_name, key.hash)
    # Printed for every fleet, always, with no threshold deciding whether it is interesting. A
    # tenant's target requires a batch size (`TenantTemplate.concurrency_at_target`) and
    # `max_num_seqs` caps it; over the cap the load refuses, but *at* 96% of it a fleet is feasible
    # and one small change from not being, and that is a fact about the fleet a reader should have
    # before a run rather than after one. Descending, so the tenant nearest the cliff reads first.
    loads = sorted(((t.concurrency_load(), t) for t in tenants if t.slo
                    and t.slo.itl_mean_ms is not None), key=lambda p: -p[0])
    if loads:
        log.info("batch cap headroom: %s", "; ".join(
            f"{t.name} needs {t.concurrency_at_target():.1f} of {t.max_num_seqs} concurrent "
            f"({100.0 * frac:.0f}%)" for frac, t in loads))
    if not node_inventory.leased:
        log.info("no node captured, so this fleet is planned against the shape of a %s: %s. "
                 "Nothing can be deployed from it.",
                 hardware.instance_type, hardware.provenance())
    if fleet.untargeted:
        # Said once, at load, rather than discovered when a report cannot certify a tenant.
        log.warning("no SLO stated for %s. Nothing will be invented for them: a placement can "
                    "be measured for these tenants but not certified against a target.",
                    ", ".join(fleet.untargeted))
    resolved = {n: r for n, r in resolutions.items() if r.status == RESOLVED}
    if resolved:
        log.info("TP resolved from solo profiles for %s",
                 ", ".join(f"{n}=tp{r.tp}{'' if r.certain else ' (upper bound)'}"
                           for n, r in resolved.items()))
    unresolved = sorted(set(resolutions) - set(resolved))
    if unresolved:
        # Only reachable under `allow_unresolved_tp`, i.e. from `profile`. Said at load so the
        # degrees printed below are not read as minima.
        log.warning("the TP degree shown for %s is a memory floor, not a measured minimum: %s",
                    ", ".join(unresolved),
                    "; ".join(f"{n}: {resolutions[n].status}" for n in unresolved))
    return {"fleet": fleet, "key": key, "hardware": hardware, "raw": doc,
            "tp_resolutions": resolutions, "reserves": reserves, "manifests": manifests,
            "max_nodes": fleet_max_nodes(doc)}
