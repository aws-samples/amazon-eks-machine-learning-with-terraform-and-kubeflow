"""The canonical form of a result: what to deploy, with this cluster's identifiers taken out.

A search answers a question about one node: these tenants, on *those* devices, addressed by the UUIDs
a particular g6e reported on a particular afternoon. That answer is what the store holds, and it is
the right thing to hold -- a measurement is only ever a measurement of the machine it ran on. But it
is the wrong thing to hand to whatever deploys the fleet. A UUID names hardware that Karpenter can
reclaim in ten minutes; an ordinal names a position in one driver's enumeration order. Neither
survives the node, and a deployment built on either is a deployment that silently addresses the wrong
card.

So this module rewrites the same answer in terms that outlive the node:

  * a **slot** is one device with no identity of its own, carrying the (tenant, shard) pairs that
    sit on it. Two shards of one tenant on one device is impossible; a tenant with four shards
    occupies four slots; three tenants on one device is one slot with three residents. That is the
    whole of what a placement says, and it says it without an ordinal or a UUID.
  * memory is **absolute GiB**, per shard, because a utilisation fraction is a fraction of a card
    nobody named. `required_gib` is the figure a deployment sizes against.
  * the **conditions** -- the rates, the sequence shapes, the ceilings, the checkpoints -- are
    carried by reference: the fleet file's path and the sha256 of its bytes, plus the validity key
    they hash to. The key's expansion is already filed beside the observations as `key.json`, and
    copying it here would create a second copy of the one thing that must not be able to disagree
    with itself.

Three results, not one, and this is deliberate. The **baseline** is always exported, because it is
the only one that exists unconditionally: it is arithmetic on measured TP degrees, so a fleet whose
search found nothing still has a deployable answer and a number to beat. The **strict** result is
the placement certified against the ceilings as written. The **relaxed** result is what a stated
tolerance would buy and is `null` more often than not -- with `relaxed_why_not` saying why, because a
null with no reason reads as a missing feature rather than as an answer. And the placements that were
measured and **rejected** are exported too: a consumer that knows 2 GPUs was tried and missed by
4.2 s of TTFT will not propose it again, and that is worth more than the space it takes.

Free and offline. It reads the store and the fleet file, spends no GPU time and touches no cluster
beyond the mirror push, which is how it can run at the end of every pipeline run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..store.schema import SuppressedMetric

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class ExportError(Exception):
    pass


# --------------------------------------------------------------------------------------------
# the layout: YAML, because a result needs comments
# --------------------------------------------------------------------------------------------

class Flow(dict):
    """A mapping short enough to read on one line. Affects layout and nothing else."""


def _needs_quotes(text: str) -> bool:
    """Whether a string has to be quoted to come back as the same string.

    The structural cases first, then the one that is not worth enumerating: YAML resolves several
    unquoted shapes to something other than a string, and the list is longer than it looks --
    `true`, `null`, `1.0`, and a timestamp like `2026-09-20T13:55:53+00:00`, which `generated_at`
    is. Rather than keep a table of them, ask the reader that will parse this file.
    """
    if text == "" or text.strip() != text:
        return True
    if text[0] in "-?:,[]{}#&*!|>'\"%@`":
        return True
    if ": " in text or " #" in text:
        return True
    try:
        return yaml.safe_load(text) != text
    except yaml.YAMLError:
        return True


def _scalar(value) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        # Rounded by whoever built the document, so this only has to avoid exponent notation and
        # keep a whole number looking like a measurement rather than a count.
        return f"{value:.6f}".rstrip("0").rstrip(".") if value != int(value) else f"{value:.1f}"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    return json.dumps(text) if _needs_quotes(text) else text


def _key(name) -> str:
    """A mapping key, quoted by the same rule as a value.

    A key is a scalar too, and YAML resolves one the same way: a tenant called `no` becomes the
    boolean False as a key, which is a document that parses into something else entirely rather than
    one that fails to parse.
    """
    return _scalar(name) if isinstance(name, str) else str(name)


def _flow(mapping) -> str:
    return "{" + ", ".join(f"{_key(k)}: {_scalar(v)}" for k, v in mapping.items()) + "}"


def _inline(seq) -> bool:
    return all(isinstance(i, Flow) or not isinstance(i, (dict, list, tuple)) for i in seq)


def _dump_seq(seq, path, depth, out, comments) -> None:
    pad = "  " * depth
    for item in seq:
        if isinstance(item, Flow):
            out.append(f"{pad}- {_flow(item)}")
        elif isinstance(item, dict):
            sub = []
            _dump_map(item, path, depth + 1, sub, comments)
            sub[0] = f"{pad}- {sub[0].lstrip()}"
            out.extend(sub)
        elif isinstance(item, (list, tuple)):
            out.append(f"{pad}-")
            _dump_seq(item, path, depth + 1, out, comments)
        else:
            out.append(f"{pad}- {_scalar(item)}")


def _dump_map(mapping, path, depth, out, comments) -> None:
    pad = "  " * depth
    for key, value in mapping.items():
        here = f"{path}.{key}" if path else str(key)
        block = comments.get(here)
        if block:
            if out and out[-1].strip():
                out.append("")
            out.extend(f"{pad}# {line}".rstrip() for line in block)
        if isinstance(value, Flow):
            out.append(f"{pad}{_key(key)}: {_flow(value)}")
        elif isinstance(value, dict):
            if not value:
                out.append(f"{pad}{_key(key)}: {{}}")
            else:
                out.append(f"{pad}{_key(key)}:")
                _dump_map(value, here, depth + 1, out, comments)
        elif isinstance(value, (list, tuple)):
            if not value:
                out.append(f"{pad}{_key(key)}: []")
            elif _inline(value):
                out.append(f"{pad}{_key(key)}: ["
                           + ", ".join(_flow(i) if isinstance(i, Flow) else _scalar(i)
                                       for i in value) + "]")
            else:
                out.append(f"{pad}{_key(key)}:")
                _dump_seq(value, here, depth + 1, out, comments)
        else:
            out.append(f"{pad}{_key(key)}: {_scalar(value)}")


def dump(document, comments=None) -> str:
    """The document as YAML, with the exporter's own comments in it.

    Hand-rolled rather than PyYAML for one reason: the comments. They are most of why this is YAML
    and not JSON, and a round-trip through a serialiser that discards them would leave a file that
    only a program can read -- which is the format this replaces. The shape being emitted is narrow
    (mappings, sequences, scalars, no anchors, no multi-line strings), and YAML is a JSON superset,
    so what comes out still parses with `yaml.safe_load` and still diffs line by line.
    """
    comments = COMMENTS if comments is None else comments
    out = ["# " + line if line else "#" for line in comments.get("", ())]
    _dump_map(document, "", 0, out, comments)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------
# slots
# --------------------------------------------------------------------------------------------

def slots_for(assignment: dict) -> list:
    """A placement as slots: devices with their identity removed, carrying their residents.

    `assignment` is what the store holds -- tenant name to the GPU ordinals its shards ran on. The
    ordinals are dropped and only their grouping survives, which is exactly the part that transfers:
    a slot is "one device", a resident is "this tenant's shard n", and two residents in one slot were
    sharing a device.

    The shard index is positional within the tenant, assigned in ordinal order. Nothing downstream
    may read meaning into which shard sits where -- tensor-parallel ranks are interchangeable at
    placement time -- but a *stable* index is what lets two exports of the same placement be
    compared, and what says four slots hold four different shards rather than one shard four times.

    Slots are sorted by their residents so the same placement always emits the same document, and a
    diff between two exports is a difference in the answer rather than in a dict's iteration order.
    """
    by_device: dict = {}
    for tenant, gpus in assignment.items():
        for shard, gpu in enumerate(sorted(gpus)):
            by_device.setdefault(gpu, []).append(Flow({"tenant": tenant, "shard": shard}))
    slots = [{"residents": sorted(by_device[gpu], key=lambda r: (r["tenant"], r["shard"]))}
             for gpu in sorted(by_device)]
    slots.sort(key=lambda s: [(r["tenant"], r["shard"]) for r in s["residents"]])
    return slots


def disjoint_slots(shards: dict) -> list:
    """The baseline as slots: every shard alone on a device, which is what §0 prices.

    Built from the degrees rather than from an assignment, because the baseline is not a placement
    anybody measured -- it is the arrangement with overlap forbidden, and it has no ordinals to drop.
    """
    return [{"residents": [Flow({"tenant": name, "shard": shard})]}
            for name, count in sorted(shards.items()) for shard in range(count)]


# --------------------------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------------------------

def _ms(value):
    return None if value is None else round(float(value), 4)


def _gib(value):
    return None if value is None else round(float(value), 3)


def _ratio(value):
    return None if value is None else round(float(value), 4)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tenant_catalogue(fleet, base) -> dict:
    """What to deploy, per tenant: the model, the engine, the shards and the memory per shard.

    The required envelope comes from the solver at the *resolved* degree and is the same number in
    the baseline and in every placement, because it is a property of the tenant and its demand rather
    than of where it landed. It is the figure a deployment sizes against.
    """
    envelope = {t.name: t.envelope_at_resolved_gib for t in base.tenants}
    floors = {t.name: t.floor_tp for t in base.tenants}
    out = {}
    for tenant in sorted(fleet.tenants, key=lambda t: t.name):
        slo = tenant.slo
        out[tenant.name] = {
            "model_id": tenant.model_id,
            "model_path": tenant.model_path,
            "model_revision": tenant.model_revision or None,
            "served_name": tenant.served_name,
            "engine": tenant.engine,
            "dtype": tenant.dtype,
            "shards": tenant.tp,
            "required_gib": _gib(envelope.get(tenant.name)),
            "memory_floor_shards": floors.get(tenant.name),
            "max_model_len": tenant.max_model_len,
            "max_num_batched_tokens": tenant.max_num_batched_tokens,
            "max_num_seqs": tenant.max_num_seqs,
            "demand": Flow({"rate_hz": tenant.rate_hz, "isl_tokens": tenant.isl_tokens,
                            "osl_tokens": tenant.osl_tokens}),
            "images": (None if not tenant.image_count
                       else Flow({"count": tenant.image_count, "pixels": tenant.image_pixels})),
            # `tolerance_pct` is the tolerance that was *in effect*, with `tolerance_declared`
            # saying whether the fleet file stated it or it is the default. The number alone
            # supports the wrong claim either way: "this tenant accepts 5%" and "nobody said, so
            # 5%" are different statements, and only one of them is somebody's requirement.
            "targets": Flow({
                "itl_mean_ms": None if slo is None else slo.itl_mean_ms,
                "ttft_p95_ms": None if slo is None else slo.ttft_p95_ms,
                "tolerance_pct": None if slo is None else slo.tolerance(),
                "tolerance_declared": None if slo is None else slo.tolerance_is_declared,
            }),
        }
    return out


def _baseline_block(base, upper_bound, resolutions) -> dict:
    """The disjoint arrangement at the measured degrees -- the answer that always exists."""
    shards = {t.name: t.resolved_tp for t in base.tenants}
    rows = {}
    for name in sorted(shards):
        resolution = (resolutions or {}).get(name)
        evidence = None if resolution is None else resolution.evidence
        rows[name] = Flow({
            "shards": shards[name],
            "solo_itl_ms": _ms(None if evidence is None else evidence.itl_mean_ms),
            "solo_run_id": None if evidence is None else evidence.run_id,
            "narrower_degree_unmeasured": name in set(upper_bound or ()),
        })
    return {
        "devices": base.resolved_gpus,
        "upper_bound": bool(upper_bound),
        "fits_on_one_node": base.fits_disjoint,
        "devices_at_memory_floor": base.floor_gpus,
        "slots": disjoint_slots(shards),
        "per_tenant": rows,
    }


def _from_record(record, tenant: str, field: str):
    """One measured figure for one tenant, or None if the record does not carry it.

    The search summary keeps ITL per tenant and nothing else, because ITL is what a placement is
    ranked by. TTFT and the delivered fraction are in the record, and they matter here for the same
    reason a rejected placement is exported at all: the run that missed by 4.2 s of TTFT met every
    ITL ceiling it was given, so a document that showed only ITL would show a placement that looks
    like it passed.

    A record may be absent -- superseded and deleted, or measured on a store this desktop has not
    pulled -- and a figure may be suppressed, which is the harness declining to report a latency it
    could not measure properly. Both come back as None rather than as a gap in the document.
    """
    tenants = {} if record is None else (record.per_tenant or {})
    observed = tenants.get(tenant)
    if observed is None:
        return None
    try:
        return getattr(observed, field)
    except (SuppressedMetric, AttributeError):
        return None


def _placement_block(block, summary, catalogue, records=None) -> dict | None:
    """One measured placement -- strict or relaxed -- as slots plus what each tenant measured.

    The latency each tenant reached comes from the *attempt* rather than from the summary's `vs_solo`
    table, which describes one placement only (the one the run reported against) and would attribute
    strict's numbers to relaxed. What is taken from `vs_solo` is the pair of facts that belong to the
    tenant and not to the placement: what it measured alone, and the ceiling it was judged against.
    The ratio is then computed here, so it always describes the placement it is printed under.
    """
    if not block or not block.get("assignment") or block.get("gpus") is None:
        return None
    run_id = block.get("run_id")
    attempts = {a.get("run_id"): a for a in summary.get("attempts") or ()}
    attempt = attempts.get(run_id) or {}
    vs_solo = summary.get("vs_solo") or {}
    granted = attempt.get("envelope_gib") or {}
    caps = attempt.get("active_thread_percentage") or {}
    itl = attempt.get("per_tenant_itl_ms") or {}
    met = attempt.get("targets_met") or {}
    coresidency = attempt.get("coresidency") or {}
    # A relaxed offer is scored against the baseline *the same concession* would leave standing,
    # which is not always the strict baseline: relaxing a ceiling can free a GPU by itself, and
    # crediting that to co-location is the overstatement the whole report is built to avoid.
    against = block.get("scored_against_baseline_gpus") or summary.get("baseline_gpus")
    record = (records or {}).get(run_id)
    rows = {}
    for name in sorted(catalogue):
        solo = vs_solo.get(name) or {}
        measured, alone = itl.get(name), solo.get("solo_itl_ms")
        rows[name] = Flow({
            "shards": catalogue[name]["shards"],
            "required_gib": catalogue[name]["required_gib"],
            "granted_gib": _gib(granted.get(name)),
            "coresidents": coresidency.get(name),
            "itl_mean_ms": _ms(measured),
            "target_itl_ms": _ms(solo.get("target_itl_ms")),
            "ttft_p95_ms": _ms(_from_record(record, name, "ttft_p95_ms")),
            "delivered_fraction": _ratio(_from_record(record, name, "delivered_fraction")),
            "solo_itl_ms": _ms(alone),
            "solo_run_id": solo.get("solo_run_id"),
            "ratio_vs_solo": (None if not measured or not alone else _ratio(measured / alone)),
            "targets_met": met.get(name),
        })
        if name in caps:
            # The SM ceiling it was measured under (impl §16), and only on a tenant that had one: a
            # deployment that drops it serves a different placement from the one measured.
            rows[name]["active_thread_percentage"] = int(caps[name])
    concession = {
        name: Flow({"metric": row.get("metric"), "ceiling_ms": _ms(row.get("ceiling")),
                    "measured_ms": _ms(row.get("measured")),
                    "over_by_factor": _ratio(row.get("factor")),
                    "tolerance_pct": row.get("tolerance_pct"),
                    "tolerance_declared": row.get("declared_tolerance")})
        for name, row in sorted((block.get("concession") or {}).items())}
    document = {
        "certified": bool(block.get("certified")),
        "devices": block["gpus"],
        "devices_freed": block.get("freed"),
        "scored_against_devices": against,
        "fraction_freed": (None if not against or block.get("freed") is None
                           else _ratio(block["freed"] / against)),
        "run_id": run_id,
        "slots": slots_for(block["assignment"]),
        "per_tenant": rows,
    }
    if concession:
        # Only a relaxed offer has these, and a strict result carries no empty copy of them: a
        # `concession: null` under a certified placement invites the reading that a concession was
        # made and came to nothing.
        document["concession"] = concession
        document["baseline_moves_under_the_concession"] = \
            block.get("baseline_moves_under_the_concession")
        document["baseline_movement_checked"] = block.get("baseline_movement_checked")
    return document


def _rejected_block(summary, strict, relaxed, records=None) -> list:
    """The placements that were measured and missed, with the overshoot that ruled each one out.

    Exported because it is the expensive half of what a search learned and the half nothing else
    records in transferable terms. Every entry here cost a deploy-and-drive round, and each one
    closes off a denser arrangement that a reader looking only at the certified answer would
    reasonably propose again.

    Only measured attempts appear. A placement that never reached readiness says nothing about the
    fleet -- it says something about that afternoon -- and the count of those is in the provenance
    instead.
    """
    claimed = {block.get("run_id") for block in (strict, relaxed) if block}
    relaxation = summary.get("required_relaxation") or {}
    out = []
    for attempt in summary.get("attempts") or ():
        run_id = attempt.get("run_id")
        met = attempt.get("targets_met") or {}
        if attempt.get("outcome") != "measured" or run_id in claimed or all(met.values()):
            continue
        detail = relaxation.get(run_id) or {}
        misses = {}
        for name, row in sorted((detail.get("per_tenant") or {}).items()):
            if met.get(name):
                continue
            misses[name] = Flow({
                "metric": row.get("metric"),
                "ceiling_ms": _ms(row.get("ceiling")),
                "measured_ms": _ms(row.get("measured")),
                "over_by_ms": _ms(row.get("over_by")),
                "over_by_factor": _ratio(row.get("factor")),
                "within_stated_tolerance": row.get("within_tolerance"),
            })
        for name, ok in sorted(met.items()):
            if ok or name in misses:
                continue
            # A tenant that missed on something the relaxation table does not price: the delivery
            # gate, or a verdict it never got. Reported as a miss with no overshoot rather than
            # omitted, because a tenant absent from both `met` and `missed` reads as an oversight.
            misses[name] = Flow({
                "metric": None,
                "delivered_fraction": _ratio(_from_record((records or {}).get(run_id), name,
                                                          "delivered_fraction")),
                "note": "this tenant did not meet its targets and the summary prices no overshoot "
                        "for it, which is a missing verdict or a failed delivery gate rather than a "
                        "latency it missed by. The record under this run id says which",
            })
        out.append({
            "run_id": run_id,
            "devices": attempt.get("gpus_used"),
            "slots": slots_for(attempt.get("assignment") or {}),
            "met": sorted(n for n, ok in met.items() if ok),
            "missed": misses,
            # A tenant no relaxation can reach: the metric it missed on has no stated tolerance, so
            # the miss is not a question of how far a ceiling may move.
            "unreachable_by_any_tolerance": sorted(detail.get("blocked") or ()),
        })
    return out


def _provenance(summary, summary_path, store_key_hash, profile_key_hash, resolutions,
                calibration) -> dict:
    """Where each half of this document came from. Two halves, because they are keyed differently.

    The placements are keyed by the **validity key**: this workload, at these degrees, on this
    hardware and this image. The baseline is keyed by the **environment**, which is why a solo row
    measured for one fleet prices another -- and why the two blocks cannot be collapsed into one
    address even though they describe the same run.
    """
    # A key with no search has no placements half at all, and that is a normal export: the baseline
    # exists on solo rows alone.
    attempts = (summary or {}).get("attempts") or ()
    rows = {}
    for name, resolution in sorted((resolutions or {}).items()):
        evidence = getattr(resolution, "evidence", None)
        rows[name] = Flow({
            "run_id": None if evidence is None else evidence.run_id,
            "shards": getattr(resolution, "tp", None),
            "minimum_certain": bool(getattr(resolution, "certain", False)),
        })
    placements = {
        "validity_key_hash": store_key_hash,
        "key_expansion": f"observations/{store_key_hash}/key.json",
        "search_summary": summary_path,
        "run_token": summary.get("run_token"),
        "verdict": summary.get("verdict"),
        "stopped_because": summary.get("stopped_because"),
        "reasoner": summary.get("reasoner"),
        "node": summary.get("gpu_node"),
        "attempts": len(attempts),
        "measured": sum(1 for a in attempts if a.get("outcome") == "measured"),
        "closest_to_targets": (summary.get("closest_to_targets") or {}).get("run_id"),
        "elapsed_s": summary.get("elapsed_s"),
    } if summary else None
    return {
        "placements": placements,
        "baseline": {
            "profile_key_hash": profile_key_hash,
            "solo_rows": f"profiles/{profile_key_hash}/solo",
            "per_tenant": rows,
        },
        "calibration": (None if calibration is None else {
            "between_run_itl_halfwidth_ms": {
                bucket: _ms(value) for bucket, value
                in sorted(getattr(calibration, "between_run_itl_halfwidth_ms", {}).items())},
            "repeats": getattr(calibration, "repeats", None),
        }),
    }


def build(*, fleet, base, fleet_path, validity_key_hash, profile_key_hash, hardware,
          summary=None, summary_path=None, upper_bound=(), resolutions=None,
          calibration=None, records=None, now=None) -> dict:
    """The whole document, in the order it is written out.

    `summary` is the stored search summary, or None when no search has run under this key. The
    baseline does not depend on it: that is the point of exporting the baseline unconditionally, and
    it is why this mode is worth running before any GPU time is spent at all.

    `records` is the store's observations by run id, and everything read from them is optional: a
    document that is missing a TTFT because a record was superseded is still the same answer.
    """
    catalogue = _tenant_catalogue(fleet, base)
    strict = _placement_block((summary or {}).get("strict"), summary or {}, catalogue, records)
    relaxed = _placement_block((summary or {}).get("relaxed"), summary or {}, catalogue, records)
    why_not = ((summary or {}).get("relaxed") or {}).get("why_not")
    stamp = (now or datetime.now(timezone.utc)).replace(microsecond=0).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": stamp,
        "fleet": {
            "path": str(fleet_path),
            "sha256": _sha256(fleet_path),
            "validity_key_hash": validity_key_hash,
            "instance_type": hardware.instance_type,
            "device_model": hardware.gpu_product or None,
            "devices_per_node": hardware.gpu_count,
        },
        "tenants": catalogue,
        "baseline": _baseline_block(base, upper_bound, resolutions),
        "strict": strict,
        "relaxed": relaxed,
        "relaxed_why_not": (None if relaxed else (
            why_not or ("no search has been run under this validity key, so no placement has been "
                        "measured and nothing has been relaxed"))),
        "rejected": _rejected_block(summary or {}, strict, relaxed, records),
        "provenance": _provenance(summary, summary_path, validity_key_hash, profile_key_hash,
                                  resolutions, calibration),
    }


COMMENTS = {
    "": [
        "A co-location result in canonical form: what to deploy, without this cluster in it.",
        "",
        "Written by `colocation.agent export`. Every number here was measured or derived from a",
        "measurement -- nothing in this file is a prediction, and nothing in it was chosen by hand.",
        "",
        "A SLOT is one device with no identity: no UUID, no ordinal. Its residents are (tenant,",
        "shard) pairs, so two residents in one slot shared a device and a tenant with four shards",
        "occupies four slots. That is the part of a placement that survives the node it was",
        "measured on, and the part a deployment can be built from.",
        "",
        "Memory is absolute GiB per shard. `required_gib` is what a deployment sizes against; to",
        "turn it into a fraction of a device, divide by the capacity of the device you are",
        "deploying on -- this file deliberately states no card's capacity.",
        "",
        "The conditions this result holds under -- rates, sequence shapes, ceilings, checkpoints --",
        "are the fleet file's, referenced below by path and content hash. The result is only a",
        "result for that file: change a rate and the validity key moves, which is what stops a",
        "placement being reused for a workload it was never measured under.",
        "",
        "YAML is a JSON superset, so this parses with any YAML reader while keeping these comments.",
    ],
    "fleet": [
        "The question. `sha256` is of the fleet file's bytes, so a consumer can tell whether the",
        "file it is holding is the file this was measured from.",
    ],
    "tenants": [
        "What to deploy, per tenant, at the shard count a solo measurement settled. `shards` is the",
        "tensor-parallel degree: the number of devices one copy of this model occupies. It is not a",
        "replica count -- nothing here proposes replicas.",
        "`memory_floor_shards` is the narrowest degree the weights alone allow. When it is below",
        "`shards`, a measurement ruled the narrower degree out on latency rather than on memory.",
    ],
    "baseline": [
        "THE BASELINE: every shard alone on its own device, nothing shared. This is what",
        "co-location has to beat and what a result is scored against, and it is exported whether or",
        "not a search found anything -- it is arithmetic on measured degrees, so it always exists.",
        "`upper_bound: true` means some tenant met its target at the degree shown while a narrower",
        "degree was never measured; the baseline is then real but possibly one device too wide.",
    ],
    "strict": [
        "THE CERTIFIED RESULT: the densest placement measured to meet every ceiling as written, with",
        "no ceiling moved. `null` means no measured placement met them all -- deploy the baseline.",
        "`granted_gib` is what the run gave that tenant on its device, which for a tenant alone on a",
        "card is the whole card. Size deployments from `required_gib`, not from this.",
    ],
    "relaxed": [
        "THE RELAXED RESULT: what a tenant's own stated tolerance would buy past its ceiling.",
        "Never a certification -- `certified` is false by construction -- and `null` whenever no",
        "denser placement fits inside the tolerances, with the reason on the next line.",
    ],
    "rejected": [
        "MEASURED AND MISSED: placements that were driven on real hardware and failed. Each one cost",
        "a measurement round, and each one rules out an arrangement that looks reasonable from the",
        "certified answer alone. `over_by_ms` is against the ceiling as written.",
    ],
    "provenance": [
        "Where this came from. The two blocks are addressed differently on purpose: placements are",
        "keyed by the validity key (this workload, these degrees, this hardware, this image), while",
        "the solo rows behind the baseline are keyed by the environment, which is why they are",
        "reusable across fleets.",
        "`calibration` is the resolution of a measurement on this node -- two runs of one placement",
        "that differ by less than this differ by nothing. `null` means no placement was measured",
        "twice, and then no difference between two placements here is established as real.",
    ],
}


# --------------------------------------------------------------------------------------------
# reading the stored search
# --------------------------------------------------------------------------------------------

def newest_search(store, run_token: str | None = None) -> tuple:
    """The stored search summary to export, as `(document, relative path)`.

    `(None, None)` when the key holds no search at all, which is not an error: the baseline is still
    exportable, and saying so is more useful than refusing. With several searches under one key --
    two answers, taken at different times, neither superseding the other -- the newest is exported
    unless `run_token` names one.
    """
    directory = store.key_dir / "searches"
    if run_token:
        path = directory / f"{run_token}.json"
        if not path.is_file():
            have = sorted(p.stem for p in directory.glob("*.json")) if directory.is_dir() else []
            raise ExportError(
                f"no search summary for run token {run_token!r} under {store.key_hash}. "
                + (f"This key holds: {', '.join(have)}." if have
                   else "This key holds no search at all."))
    else:
        found = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime) \
            if directory.is_dir() else []
        if not found:
            return None, None
        path = found[-1]
    try:
        document = json.loads(path.read_text())
    except ValueError as exc:
        raise ExportError(f"{path} is not readable as JSON: {exc}") from exc
    return document, f"observations/{store.key_hash}/searches/{path.name}"


# --------------------------------------------------------------------------------------------
# more than one node: schema_version 2
# --------------------------------------------------------------------------------------------

SPLIT_SCHEMA_VERSION = 2


def nodes_of(doc: dict) -> list:
    """The per-node documents of an export: a version 1 export is one node, and is its own.

    A node of a version 2 export *is* a version 1 body -- its group's own export, read back -- so a
    consumer that handles one node handles every node of a split the same way.
    """
    version = doc.get("schema_version") if isinstance(doc, dict) else None
    if version == SCHEMA_VERSION:
        return [doc]
    if version == SPLIT_SCHEMA_VERSION:
        return list(doc.get("nodes") or ())
    raise ExportError(f"schema_version {version!r} is not one this build reads "
                      f"({SCHEMA_VERSION} or {SPLIT_SCHEMA_VERSION})")


def _reflow(value, depth=0):
    """A version 1 body read back from YAML, with its one-line rows one line again. Layout only:
    every mapping of scalars below a node's own blocks is a row, which is where `build` puts Flow."""
    if isinstance(value, dict):
        if depth >= 2 and value and all(not isinstance(v, (dict, list)) for v in value.values()):
            return Flow(value)
        return {k: _reflow(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_reflow(v, depth + 1) for v in value]
    return value


def build_split(*, fleet_path, max_nodes, verdict, chooser, stopped_because, nodes,
                now=None) -> dict:
    """A split's result: the parent fleet, and each node's group export under `nodes`.

    `nodes` is `(export path, version 1 document)` per group of the split. Each group was searched,
    stored and exported as a fleet of its own, under its own validity key, and its export is copied
    here whole rather than re-derived: there is one canonical form of a group's result, and it is
    the one beside its records.
    """
    if not nodes:
        raise ExportError("a split has at least one node")
    docs = [doc for _, doc in nodes]
    for doc in docs:
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise ExportError("a node of a split is a schema_version 1 export")
    types = {d["fleet"]["instance_type"] for d in docs}
    if len(types) != 1:
        raise ExportError(f"a split is over one instance type; its groups were measured on "
                          f"{', '.join(sorted(types))}")
    seen: dict = {}
    for d in docs:
        for name in d["tenants"]:
            if name in seen:
                raise ExportError(f"{name} is on two nodes of the split")
            seen[name] = True
    first = docs[0]["fleet"]
    strict = [d.get("strict") for d in docs]
    stamp = (now or datetime.now(timezone.utc)).replace(microsecond=0).isoformat()
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "generated_at": stamp,
        "fleet": {
            "path": str(fleet_path),
            "sha256": _sha256(fleet_path),
            "instance_type": first["instance_type"],
            "device_model": first.get("device_model"),
            "devices_per_node": first["devices_per_node"],
            "max_nodes": int(max_nodes),
        },
        "split": {
            "verdict": verdict,
            "chosen_by": chooser,
            "stopped_because": stopped_because,
            "nodes": len(docs),
            "devices": (None if any(s is None for s in strict)
                        else sum(s["devices"] for s in strict)),
            "baseline_devices": sum(d["baseline"]["devices"] for d in docs),
        },
        "nodes": [_reflow({"export": str(path),
                           **{k: v for k, v in doc.items()
                              if k not in ("schema_version", "generated_at")}})
                  for path, doc in nodes],
    }


SPLIT_COMMENTS = {
    "": [
        "A co-location result over more than one node, in canonical form.",
        "",
        "Written by `colocation.agent search` on a fleet with `max_nodes` above 1. Each tenant is",
        "on exactly one node, and the tenants of one node are a group: a fleet of their own, which",
        "was searched, stored and exported on its own node under its own validity key.",
        "",
        "Each entry of `nodes` is that group's own export (schema_version 1), copied whole, with",
        "`export` saying where it is filed. Nothing in one node's entry describes another node:",
        "nodes share no device, so a slot never spans two of them. Nothing here names a node",
        "either -- which node a group goes on is the deployment's to choose.",
    ],
    "fleet": [
        "The parent fleet file, the one that was split. Each node's own `fleet` is its group's",
        "file, generated from this one.",
    ],
    "split": [
        "`devices` is every node's certified placement added up, and null unless every node has",
        "one. `baseline_devices` is the same sum over each node's baseline.",
    ],
}
