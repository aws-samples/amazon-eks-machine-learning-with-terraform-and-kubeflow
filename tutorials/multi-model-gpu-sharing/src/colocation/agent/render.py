"""Placement -> deployable values, and Placement -> harness run config (impl §5, §11).

This is the *actuation* step, and it is a rendering step rather than a translation step: the
serving chart's values schema is the solver's output schema by construction (see that chart's
values.yaml header), so nothing here decides anything. The one block a serving engine gets a say in
is rendered by its adapter (`colocation.engines`); everything else here is Kubernetes, MPS and FSx. Every number written below was
already decided by the solver or fixed by the `FleetSpec`. If this file ever has to compute a
serving parameter, either the chart or the solver is missing a value.

Two outputs, from the same `Placement`, and they must agree:

  - `render_releases()`  -> one values dict per tenant, for `helm upgrade --install`
  - `render_run_config()` -> the §6 harness's run YAML, naming the same tenants and the same
    GPU UUIDs

They are rendered together from one placement for a reason. The harness attributes per-GPU
DCGM series by UUID, because DCGM cannot see through MPS to tell which pod owns a GPU
(`scrape.scrape_dcgm`). If the two ever disagreed, every cause signal in the record would be
filed against the wrong tenant, and nothing would flag it.
"""

from __future__ import annotations

import re

from .. import engines
from ..solver.memory import HEADROOM_DEFAULT, gpu_memory_utilization
from .spec import FleetSpec, NodeInventory, TenantTemplate

# RFC-1123 label: Kubernetes rejects anything else, and the failure arrives from the API
# server as a rejected Deployment rather than as a bad tenant name.
_RFC1123 = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

DEFAULT_NAMESPACE = "kubeflow-user-example-com"
DEFAULT_SERVICE_PORT = 8000
DEFAULT_MPS_PIPE_HOST_PATH = "/var/run/nvidia-mps"
DEFAULT_MPS_PIPE_MOUNT_PATH = "/mps"
DEFAULT_FSX_CLAIM = "pv-fsx"
DEFAULT_FSX_MOUNT = "/fsx"
DEFAULT_STORE_ROOT = "/efs/colocation"
DEFAULT_EXPORTER_PORT = 9400

# The Helm release that owns the MPS control daemon AND, since chart 0.2.0, the
# dcgm-exporter that instruments its node. One release, because the exporter's useful
# lifetime is exactly the node lease's.
DEFAULT_DAEMON_RELEASE = "mps-daemon"


def exporter_service_url(daemon_release: str = DEFAULT_DAEMON_RELEASE,
                         namespace: str = DEFAULT_NAMESPACE,
                         port: int = DEFAULT_EXPORTER_PORT) -> str:
    """Where the harness scrapes DCGM, derived from the daemon's release name.

    Derived rather than configured, for the same reason `fullname()` above is: the harness's
    run config is rendered before anything is installed, so the name has to be computable.
    The chart holds up its end by naming every exporter resource `<release>-dcgm-exporter`
    (mps-daemon/templates/_helpers.tpl).

    This used to be the frozen string `dcgm-prototype-dcgm-exporter...`, the name of a
    release installed once by hand. That is fine while there is exactly one of everything
    and wrong the moment a run installs its own daemon, which is now the plan: the release
    name is per-run, so the scrape target has to be too.
    """
    return f"http://{daemon_release}-dcgm-exporter.{namespace}.svc.cluster.local:{port}/metrics"


DEFAULT_EXPORTER_URL = exporter_service_url()


class RenderError(Exception):
    """The placement cannot be expressed in the chart's values schema."""


def release_name(t: TenantTemplate, prefix: str = "") -> str:
    """`<prefix>-<tenant>`, or just the tenant name when there is no prefix.

    The prefix is the run token (`capture.new_run_token`), and it exists because a fixed tenant
    release name is not safe once more than one run can be in flight. Two runs on two different
    nodes both install `m1` into one namespace, and `helm upgrade --install` does not treat that
    as a conflict -- it ADOPTS the existing release and re-pins its pod to the second run's node.
    The first run then keeps measuring a tenant that was moved out from under it, and no step
    anywhere reports an error.

    Empty by default so a hand-run keeps the short names, and so every name in this repo's
    examples still resolves.
    """
    return f"{prefix}-{t.name}" if prefix else t.name


def fullname(t: TenantTemplate, prefix: str = "") -> str:
    """What the chart's `fullname` helper will produce: `<release>-<values.name>`.

    Derived here rather than discovered, because the harness needs the Service DNS name
    *before* the release exists -- the run config is rendered in the same step as the values.
    """
    return f"{release_name(t, prefix)}-{t.served_name}"


def service_url(t: TenantTemplate, namespace: str = DEFAULT_NAMESPACE,
                port: int = DEFAULT_SERVICE_PORT, prefix: str = "") -> str:
    return f"http://{fullname(t, prefix)}.{namespace}.svc.cluster.local:{port}"


def _check_names(t: TenantTemplate, prefix: str = "") -> None:
    for label, value in (("name", t.name), ("served_name", t.served_name)):
        if not _RFC1123.match(value):
            raise RenderError(
                f"tenant {t.name!r}: {label}={value!r} is not an RFC-1123 label "
                "(lowercase alphanumerics and '-', starting and ending alphanumeric)")
    full = fullname(t, prefix)
    if len(full) > 63:
        raise RenderError(
            f"tenant {t.name!r}: the chart's fullname {full!r} is {len(full)} characters and "
            "Kubernetes truncates at 63. The chart truncates silently, which would collide "
            "two tenants' Services onto one name -- shorten `name` or `served_name`."
            + (f" The run prefix {prefix!r} costs {len(prefix) + 1} of the 63." if prefix
               else ""))


def _envelope_gib_int(name: str, envelope_gib: float) -> int:
    """The MPS limit is an integer GiB. A fractional envelope is refused, not rounded.

    Rounding would deploy something other than what was scored: the chart's `mpsMemLimit`
    helper writes `<int>G` while `gpu_memory_utilization` was computed from the unrounded
    figure, so the engine would ask for memory the envelope does not contain and die in
    sampler warmup (impl §4.2, second correction). Solve with
    `envelope_step_mib=AGENT_ENVELOPE_STEP_MIB` and this cannot arise.
    """
    as_int = int(round(envelope_gib))
    if abs(envelope_gib - as_int) > 1e-9:
        raise RenderError(
            f"tenant {name!r} was given a {envelope_gib} GiB envelope, but the chart enforces "
            f"the MPS limit as an integer GiB ({as_int}G). Deploying it would enforce "
            f"{as_int} GiB while gpu_memory_utilization was computed for {envelope_gib} -- the "
            "engine over-asks and dies in sampler warmup. Solve with a 1024 MiB envelope "
            "step (spec.AGENT_ENVELOPE_STEP_MIB).")
    return as_int


def render_values(*, name: str, served_name: str, image: str, model_path: str,
                  model_revision: str, tp: int, node: NodeInventory, gpu_ids,
                  limit_gib: int, util: float, max_num_seqs: int, max_num_batched_tokens: int,
                  max_model_len: int, dtype: str, extra_args=(), extra_env=(),
                  engine: str = engines.DEFAULT_ENGINE,
                  shm_size_gi: int = 16, service_port: int = DEFAULT_SERVICE_PORT,
                  fsx_claim: str = DEFAULT_FSX_CLAIM, fsx_mount: str = DEFAULT_FSX_MOUNT,
                  pipe_host_path: str = DEFAULT_MPS_PIPE_HOST_PATH,
                  pipe_mount_path: str = DEFAULT_MPS_PIPE_MOUNT_PATH,
                  active_thread_percentage: int | None = None) -> dict:
    """The chart's value schema, from primitives. THE one place that knows the schema.

    It takes primitives rather than a `TenantTemplate`, and that outlived the reason: there were two
    callers, and the second was the deleted `agent/booter.py` (impl §14.4), which deployed one engine
    with no workload at all in order to ask it how much memory it took. Rendering that through the
    tenant path would have meant inventing a `rate_hz`, an `isl_tokens`, an `osl_tokens` and a
    `ModelMemory` to satisfy a type built to refuse exactly those inventions.

    The signature stays primitive because the argument for it does: this is the only place that knows
    the chart's value schema, and the chart is where a silently-ignored MPS memory limit comes from.
    A second caller wanting a different shape of deployment -- a second engine's adapter, say -- must
    come through here rather than keep its own copy of the schema.
    """
    eng = engines.get(engine)
    gpu_ids = [int(g) for g in gpu_ids]
    if len(gpu_ids) != int(tp):
        raise RenderError(
            f"{name!r} has tp={tp} but was assigned {len(gpu_ids)} GPUs "
            f"{gpu_ids}. One shard per GPU, on distinct GPUs.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise RenderError(
            f"{name!r} was assigned GPU {gpu_ids} with a repeat. Two shards of one "
            "TP group on one GPU would each claim the full envelope on it.")
    # Order matters and is not incidental: gpu_uuids[i] must be the UUID of gpu_indices[i],
    # because the chart pairs them positionally to build the memory-limit string.
    uuids = [node.uuid_for(g) for g in gpu_ids]

    return {
        "name": served_name,
        "image": image,
        # Which serving stack these values describe. No chart template reads it -- the chart already
        # knows what it is -- and it is here because the values file is what `deploy.py` has in hand
        # when it has to decide which chart to install a tenant from. Deriving that from the presence
        # of an engine's block would work today and would stop working the first time two stacks
        # spelled their block the same, whereas a fleet that mixes engines is the case this exists
        # for. It is also provenance: `helm get values` on a live release says which stack it is,
        # which nothing else in the values makes plain.
        "engine": eng.name,
        "model": {
            "path": model_path,
            "served_name": served_name,
            "revision": model_revision,
        },
        "placement": {
            "node": node.node_name,
            "gpu_indices": gpu_ids,
            "gpu_uuids": uuids,
        },
        "mps": {
            "pipeHostPath": pipe_host_path,
            "pipeMountPath": pipe_mount_path,
            "memory_limit_gib": int(limit_gib),
            # Unset unless the search's reasoner capped this tenant on this measurement (impl §16).
            # A cap changes what the measurement means, so it is part of the placement's identity
            # and travels into the record through the run config, never set here by default.
            "active_thread_percentage": (None if active_thread_percentage is None
                                         else int(active_thread_percentage)),
        },
        # The engine's own block, under the key the engine's chart uses. This is the one part of
        # the schema that is not shared: everything above and below it is Kubernetes, MPS and FSx,
        # which no serving stack gets a say in.
        eng.values_key: eng.values(
            tp=int(tp), util=util, max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens, max_model_len=max_model_len,
            dtype=dtype, extra_args=extra_args),
        "service": {"port": service_port},
        "shmSizeGi": int(shm_size_gi),
        "storage": {"fsxClaimName": fsx_claim, "fsxMountPath": fsx_mount},
        "extraEnv": [dict(e) for e in extra_env],
    }


def render_tenant_values(t: TenantTemplate, node: NodeInventory, gpu_ids, envelope_gib: float,
                         util: float, *, service_port: int = DEFAULT_SERVICE_PORT,
                         fsx_claim: str = DEFAULT_FSX_CLAIM,
                         fsx_mount: str = DEFAULT_FSX_MOUNT,
                         pipe_host_path: str = DEFAULT_MPS_PIPE_HOST_PATH,
                         pipe_mount_path: str = DEFAULT_MPS_PIPE_MOUNT_PATH,
                         prefix: str = "", active_thread_percentage: int | None = None) -> dict:
    """One tenant's chart values. `gpu_ids` are host ordinals in shard order."""
    _check_names(t, prefix)
    return render_values(
        name=t.name, served_name=t.served_name, image=t.image, model_path=t.model_path,
        model_revision=t.model_revision, tp=t.tp, node=node, gpu_ids=gpu_ids,
        limit_gib=_envelope_gib_int(t.name, envelope_gib), util=util,
        max_num_seqs=t.max_num_seqs, max_num_batched_tokens=t.max_num_batched_tokens,
        max_model_len=t.max_model_len, dtype=t.dtype, extra_args=t.extra_args,
        extra_env=t.extra_env, engine=t.engine, shm_size_gi=t.shm_size_gi,
        service_port=service_port,
        fsx_claim=fsx_claim, fsx_mount=fsx_mount, pipe_host_path=pipe_host_path,
        pipe_mount_path=pipe_mount_path, active_thread_percentage=active_thread_percentage)


def render_releases(placement, fleet: FleetSpec, *, prefix: str = "", sm_caps=None,
                    **kwargs) -> list:
    """`[(release_name, values), ...]`, one per tenant, in fleet order.

    `sm_caps` is `{tenant: percent}` (impl §16); a tenant not in it is uncapped.
    """
    caps = dict(sm_caps or {})
    out = []
    for t in fleet.tenants:
        gpu_ids = placement.assignment[t.name]
        out.append((release_name(t, prefix),
                    render_tenant_values(t, fleet.node, gpu_ids,
                                         placement.envelope_gib[t.name],
                                         placement.util[t.name], prefix=prefix,
                                         active_thread_percentage=caps.get(t.name), **kwargs)))
    return out


def render_run_config(placement, fleet: FleetSpec, run_id: str, validity_key_hash: str, *,
                      namespace: str = DEFAULT_NAMESPACE,
                      service_port: int = DEFAULT_SERVICE_PORT,
                      prefix: str = "",
                      exporter_url: str | None = DEFAULT_EXPORTER_URL,
                      out_dir: str = DEFAULT_STORE_ROOT,
                      itl_target_halfwidth_ms: float | None = None,
                      window_min_s: float = 60.0,
                      window_ceiling_s: float = 900.0,
                      # Kept in step with the harness's own default, and it is this value that
                      # takes effect: whatever is written here overrides it. 900s was set when the
                      # gate included a lumped per-completion delivered-rate signal that needed
                      # ~140 samples at 1 Hz to cross its threshold at all. That signal is gone and
                      # the surviving one (in-flight, a level) settles far sooner, so this is now a
                      # conservative ceiling rather than a required one -- left at 900s because
                      # shortening a measurement window is a change to make against data, not
                      # against an argument (harness `controller.StabilityThresholds`). A run that
                      # settles returns early and pays none of it either way.
                      steady_state_ceiling_s: float = 900.0,
                      drain_timeout_s: float = 120.0,
                      provenance: dict | None = None,
                      sm_caps: dict | None = None) -> dict:
    """The §6 harness's run config for this placement. Consumed as `run.yaml`.

    `itl_target_halfwidth_ms` is None before calibration exists and the harness then uses a
    fixed window and says so in the record. That is correct during calibration and wrong
    afterwards, so the agent passes the calibrated figure once the store has one.

    `sm_caps` (impl §16) reaches the record from here. Only a capped tenant carries the field, so
    an uncapped run config is the one written before caps existed.
    """
    caps = dict(sm_caps or {})
    tenants = []
    for t in fleet.tenants:
        gpu_ids = placement.assignment[t.name]
        tenants.append({
            "name": t.name,
            "base_url": service_url(t, namespace, service_port, prefix),
            "served_name": t.served_name,
            "model_id": t.model_id,
            "tp": t.tp,
            "envelope_gib": _envelope_gib_int(t.name, placement.envelope_gib[t.name]),
            "rate_hz": t.rate_hz,
            # The tenant's own shape, driven per tenant by the harness (impl §6.2). Not a
            # fleet-wide field: a single ISL/OSL for the whole run would measure a fleet nobody
            # asked for while the record named the one that was asked for.
            "isl_tokens": t.isl_tokens,
            "osl_tokens": t.osl_tokens,
            # Per tenant for the same reason the shape is: it decides what "isl_tokens" above
            # actually means for this tenant, so a fleet-wide one made the field mean different
            # things for different tenants while reading identically.
            "tokenizer_path": t.tokenizer_path,
            # The rest of the request shape: how many images each one carries and at what size.
            # Emitted for text tenants too, as zeros, because "this tenant sends no images" is a
            # statement the record should carry rather than an absence a reader has to interpret.
            # The loader has already checked both against this tenant's `--limit-mm-per-prompt` and
            # `--mm-processor-kwargs` (`fleetfile._load_image_spec`), so the payload the harness
            # builds and the capacity the engine was started with cannot disagree.
            "image_count": t.image_count,
            "image_pixels": t.image_pixels,
            # Which serving stack the harness is driving. It decides the request format, how output
            # length is pinned, the readiness path and the names of the runtime gauges -- all four
            # of which the harness would otherwise have to guess from a response it has not had yet.
            # Emitted always, including for the default engine, because an absent field here would
            # make an old run config and a `vllm-openai` one indistinguishable while the whole
            # purpose of the field is to distinguish engines.
            "engine": t.engine,
            "gpu_uuids": [fleet.node.uuid_for(g) for g in gpu_ids],
            "delivered_fraction_threshold": fleet.workload.delivered_fraction_threshold,
            **({"active_thread_percentage": int(caps[t.name])} if t.name in caps else {}),
        })
    cfg: dict = {
        "run_id": run_id,
        "validity_key_hash": validity_key_hash,
        "exporter_url": exporter_url,
        "itl_target_halfwidth_ms": itl_target_halfwidth_ms,
        "window_min_s": window_min_s,
        "window_ceiling_s": window_ceiling_s,
        "steady_state_ceiling_s": steady_state_ceiling_s,
        "drain_timeout_s": drain_timeout_s,
        "out_dir": out_dir,
        "tenants": tenants,
        "provenance": dict(provenance or {}),
    }
    return cfg


def check_agreement(releases, run_config, prefix: str = "", *, device_total_gib: float,
                    mps_reserve_gib: dict, headroom: float = HEADROOM_DEFAULT) -> None:
    """Refuse to deploy if the values and the run config disagree about the GPUs.

    Cheap, and the failure it prevents is not detectable afterwards: the harness joins DCGM
    series to tenants by UUID, so a disagreement files every per-GPU cause signal against the
    wrong tenant and the record looks entirely healthy.

    `prefix` has to be passed because releases are keyed by release name and the run config by
    tenant name, and those stopped being the same string when release names became per-run.
    Getting it wrong does not weaken the check: a prefix that does not match makes every lookup
    miss and every release fail loudly.

    `device_total_gib` and `mps_reserve_gib` are required, and they are what makes this a check on
    the *denominator* rather than only on rounding. Every `gpu_memory_utilization` in `releases` is
    a fraction of one number -- what the driver reports as the size of one device -- and the engine
    multiplies it by that number on the machine, whatever the number the fraction was computed
    against was. So the fraction is recomputed here from the envelope actually being deployed and
    the device total the hardware store measured for this node, and any disagreement is refused
    with the denominator the chart's fraction implies. A utilisation computed against a nameplate
    figure, a remembered figure, or another card's figure all arrive here as that disagreement.
    They cannot be caught later: the engine reports the utilisation it was handed and its KV pool
    comes out the size the wrong denominator bought, so the run reads as healthy and the packing
    was priced against a pool that does not exist (§4.2.2 -- two tenants both Ready and serving
    with a 41% KV deficit between them, which then reads as co-location interference).

    Pass `mps_reserve_gib` and `headroom` straight from the placement and the solver that produced
    it (`Placement.mps_reserve_gib`, `PlacementSolver(headroom=...)`); the whole point is that
    nothing here invents a term.
    """
    by_tenant = {c["name"]: c for c in run_config["tenants"]}
    head = f"{prefix}-" if prefix else ""
    device_total = float(device_total_gib)
    per_gpu: dict = {}
    for release, values in releases:
        tenant = release[len(head):] if head and release.startswith(head) else release
        cfg = by_tenant.get(tenant)
        if cfg is None:
            raise RenderError(
                f"release {release!r} is being deployed but the harness config does not drive "
                "it. An undriven tenant still holds its envelope, so the run would measure a "
                "different placement than the one deployed.")
        if list(cfg["gpu_uuids"]) != list(values["placement"]["gpu_uuids"]):
            raise RenderError(
                f"{release}: chart UUIDs {values['placement']['gpu_uuids']} != harness UUIDs "
                f"{cfg['gpu_uuids']}")
        if cfg["envelope_gib"] != values["mps"]["memory_limit_gib"]:
            raise RenderError(
                f"{release}: chart envelope {values['mps']['memory_limit_gib']} GiB != "
                f"harness envelope {cfg['envelope_gib']} GiB")
        # Resolved from the run config rather than from a default, so a disagreement about which
        # engine is being deployed is itself refused: reading the chart's block under the wrong
        # engine's key raises `KeyError` below, which is the correct outcome but a poor message.
        eng = engines.get(cfg.get("engine") or engines.DEFAULT_ENGINE)
        stated = values.get("engine")
        if stated and stated != eng.name:
            raise RenderError(
                f"{release}: the chart values say engine {stated!r} and the harness config says "
                f"{eng.name!r}. One of the two decides which chart is installed and the other decides "
                "what request format is sent at it, so a disagreement here deploys one stack and "
                "drives another.")
        if eng.values_key not in values:
            raise RenderError(
                f"{release}: the harness is told to drive it as {eng.name!r}, but its chart values "
                f"carry no {eng.values_key!r} block. The two artifacts name different serving "
                "engines, so the run would drive one stack's request format at another's server.")
        block = values[eng.values_key]
        if cfg["tp"] != eng.read_tp(block):
            raise RenderError(f"{release}: TP degree disagrees")
        # The record says what the cap was from the run config; the pod gets it from the chart. A
        # disagreement would record one experiment and run another (impl §16).
        if cfg.get("active_thread_percentage") != values["mps"].get("active_thread_percentage"):
            raise RenderError(
                f"{release}: chart SM cap {values['mps'].get('active_thread_percentage')} != "
                f"harness SM cap {cfg.get('active_thread_percentage')}")

        # The denominator. `limit_gib` is the envelope: `_envelope_gib_int` has already refused a
        # fractional one, and the two artifacts have just been checked to agree about it.
        limit_gib = float(values["mps"]["memory_limit_gib"])
        if tenant not in mps_reserve_gib:
            raise RenderError(
                f"{release}: no MPS reserve was passed for tenant {tenant!r}, so its utilisation "
                "cannot be re-derived. This means the placement being deployed is not the "
                "placement these reserves came from.")
        reserve = float(mps_reserve_gib[tenant])
        util = eng.read_util(block)
        want = gpu_memory_utilization(limit_gib, reserve, device_total, headroom)
        # The chart's figure is rounded to 6 decimals by `render_values`, which is worth ~2e-5 GiB
        # on a device of this size. Anything past that is a different denominator, not rounding.
        if abs(util - want) > 1e-6:
            implied = (limit_gib - reserve) * headroom / util if util > 0 else float("inf")
            raise RenderError(
                f"{release}: gpu_memory_utilization is {util}, but a {limit_gib:.0f} GiB envelope "
                f"less a {reserve} GiB MPS reserve at {headroom} headroom is {want:.6f} of the "
                f"{device_total:.2f} GiB this device measures. The chart's fraction implies a "
                f"device total of {implied:.2f} GiB instead. {eng.name} multiplies the fraction by what "
                f"the driver reports, so deploying this asks the engine for "
                f"{util * device_total:.2f} GiB inside a {limit_gib:.0f} GiB envelope: over-asking "
                "dies in sampler warmup, and under-asking strands memory the packing already "
                "spent. The device total is measured and read from the hardware store (impl "
                "§8.1); it is not a nameplate figure and there is no table of cards.")

        for uuid in values["placement"]["gpu_uuids"]:
            per_gpu[uuid] = per_gpu.get(uuid, 0.0) + limit_gib

    # impl §9's invariant, re-checked where the numbers become a deployment. The solver already
    # proved it (`Placement.verify`), but it proved it about the total it was built with, and this
    # is the last point before helm at which the total can be compared to the node being deployed
    # to. Over-subscription need not crash, which is why it is refused rather than warned about.
    for uuid, promised in sorted(per_gpu.items()):
        if promised > device_total + 1e-6:
            raise RenderError(
                f"GPU {uuid} is promised {promised:.0f} GiB of envelope across the tenants placed "
                f"on it, against a measured device total of {device_total:.2f} GiB. Both tenants "
                "can still reach Ready and serve, sharing a KV deficit that reads afterwards as "
                "co-location interference (impl §9, §4.2.2).")

    deployed = {(r[len(head):] if head and r.startswith(head) else r) for r, _ in releases}
    extra = set(by_tenant) - deployed
    if extra:
        raise RenderError(
            f"the harness config drives {sorted(extra)}, which are not being deployed. Those "
            "requests would fail and be recorded as a delivery failure of this placement.")
