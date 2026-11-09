"""CLI for the search agent.

    # which placements are feasible, and what each tenant's target requires, without the cluster
    python -m colocation.agent plan --fleet fleets/my-fleet.yaml

    # the between-run spread implied by repeats already measured
    python -m colocation.agent calibrate --fleet fleets/my-fleet.yaml

    # each tenant's minimal TP, measured alone: what the search's baseline has to be built on
    python -m colocation.agent profile --fleet fleets/my-fleet.yaml --budget 8

    # the search: the reasoner drives it, with six tools and a measurement budget (impl §11)
    python -m colocation.agent search --fleet fleets/my-fleet.yaml --budget 6

**There were five modes. `onboard` was the fifth, and it is gone (impl §14.4).** It booted an engine,
read that engine's own account of its memory, fitted a law to it and printed a `models:` block for a
human to paste back into the fleet file. Every step of that is something this design has since ruled
out: it took the engine's internal report as evidence, which does not generalise past the one engine
whose log it parsed; it ended in a human transcribing numbers; and the numbers it produced were
inputs to arithmetic that may now only *choose* experiments, never refuse them (§14.2). What it was
built to fix -- that multiplying out a fleet file's KV geometry is 3.27x too large on Qwen3.5-9B and
1.06x too small on Qwen3.5-0.8B -- is fixed instead by never letting that arithmetic refuse a degree.
Only the weight floor may, and `profile`'s walk boots the rest.

`plan` names no node and captures none. It solves against the *shape* of `hardware.instance_type`
-- GPU count and measured device memory per GPU, from `hardware.py` -- so it costs nothing and
can be run before any node exists. Nothing can be deployed from what it loads: the fleet it builds
has no GPU UUIDs, and `NodeInventory.uuid_for` refuses rather than inventing one.

`calibrate` deploys nothing either: it summarises repeats the store already holds. It is a
measured quantity the reasoner is shown, not a stop condition applied above it -- the reasoner
decides when to stop, and the budget bounds it either way.

`profile` is what makes the baseline honest, and it is the mode to run first on new hardware. It
measures each tenant **alone**, with the whole of each GPU to itself, ascending from the memory floor
and stopping at the first TP degree that meets the tenant's own ITL target -- so the answer is the
smallest shard count that works, rather than the one somebody typed. Two consequences worth knowing:

  - **It is why `tp:` is not in the fleet file at all.** Every tenant's degree comes from these rows
    (agent/fleetfile.py), and a `tp:` line is refused rather than honoured. A hand-written degree
    wider than the target needs hands the *baseline* GPUs that re-sharding alone would free, and
    `search` then reports them as co-location's saving -- impl §0's mistake, in the flattering
    direction and silently. Checking a typed-in degree would have cost the same solo runs as
    computing it, so there was nothing to buy by keeping the field.
  - **Its rows are keyed by the environment, not by the fleet** (store/profile.py), so they are
    reused by every later fleet on the same environment -- the same card presenting the same
    addressable memory under the same power limit and links, behind the same engine image, however
    the instance type is spelled (store/environment.py). Running it a second time
    re-measures nothing it already holds; `--reprofile` is how you force a re-measurement.

`profile` and `search` are the modes that touch a GPU, and the first thing either does is **capture a
node**: it installs its own `mps-<token>` daemon release requesting every GPU on one machine, and whichever
node that pod lands on is the node (agent/capture.py). Three things follow that are worth knowing
before running it:

  - **It may buy a node.** A pod requesting 8 GPUs that nothing can schedule is, under Karpenter, an
    order for a g6e.48xlarge. If a node is already held by a hand-installed `mps-daemon` at default
    priority, this run cannot preempt it and will provision a *second* one. To hand the existing
    node over instead, `helm uninstall mps-daemon` and immediately
    `kubectl apply -f manifests/gpu-node-holder.yaml`: the holder keeps the node (and its GPU UUIDs)
    at priority -1000, which this run's daemon does preempt.
  - **It does not give the node back unless asked.** Without `--release-node` the lease is left
    standing at the end, which keeps the node -- releasing it is what risks losing the machine, not
    keeping it. With `--release-node` the lease is uninstalled and `gpu-node-holder` is applied in
    its place, so the node and its UUIDs survive an idle period. `--release-node` implies
    `--teardown`, and it has to: the tenants are default priority and the holder is -1000, so a
    handoff that removes only the daemon leaves the holder Pending and the node effectively unheld.
    The closing message names every release the run installed, in the order they have to go.
  - **It refuses a node the cluster has already decided to replace.** The holder is what blocks a
    drifted node's removal, and preempting it is what unblocks it, so `capture` checks the node's
    claim the moment it knows which node it got (`capture._refuse_if_disrupted`).

`search` is one conversation, six tools, and a budget counted in measurements rather than in
iterations. `--budget` is the number of times it may touch the GPUs; `--max-turns` bounds the
conversation around them, since asking the solver and the store costs nothing and a loop that only
stopped on GPU spend would never stop on a run that never measured.

Whatever it measures, `search` ends on **three numbers in one unit: BASELINE, STRICT, RELAXED**, all
in GPUs and all three always printed. The baseline is what `plan` printed before any GPU was touched
-- this fleet with no GPU shared. STRICT is the fewest-GPU placement that met every ceiling exactly
as written, which is why a tenant left without a verdict blocks it rather than passing quietly, and
it is the only one of the three that certifies anything. RELAXED is the fewest-GPU placement that
every tenant's *own declared* tolerance would admit (`slo.tolerance_pct` in the fleet file), stated
as the concession it requires in milliseconds of ceiling and labelled NOT CERTIFIED wherever it
appears. Each carries how many GPUs it frees as a field rather than as a line of its own.

Two lines and not three was the earlier mistake. A run can find nothing at the targets as written
*and* a placement two GPUs denser at a third of a percent on one tenant, and those are both results;
a report that headlines only the first makes the reader go looking for the second. Beside whichever
placement is reported, per tenant, is the co-resident latency against that tenant's own solo latency
at the same degree, read from the `profile` rows: the SM time sharing cost, which is what impl §0
says the freed GPUs were bought with. A run that met everything and freed nothing prints that too, in
those words.

Every placement denser than the claim that was measured and refused is named, and beneath each one,
the smallest relaxation of the stated targets that would have let it certify -- always, not only when
the search consolidated nothing. The bound on what the report will *offer* is per tenant and lives in
the fleet file; `--relax-targets-pct` can only lower it, never raise it, so a concession cannot be
invented at the command line once the result is known. Nothing is judged against a relaxed target and
no run is ever claimed on one: relaxing a ceiling can narrow a tenant's minimal TP and so move the
baseline too, which is checked from the `profile` rows and reported on screen (`_relaxed_baseline`),
and a tenant whose latency the harness suppressed is beyond any relaxation's reach because moving a
ceiling cannot supply a measurement that was never taken.

`--driver` chooses the runtime, and `sdk` is the default because it is the one that ships: the
Claude Agent SDK, real `tool_use` blocks, and a deny-by-default gate that makes `measure` the only
reachable route to the GPUs. `converse` drives the same six tools over a JSON-in-text protocol; it
is what the self-tests can substitute a `ReplayReasoner` into, and it is kept for that. The two
differ in transport only -- the tools, the invariants, the fallback and the derived verdict are the
same code -- so a run's records do not say which drove it beyond naming the reasoner.

There is no deterministic search mode. The designed co-residency sweep and the explore/exploit
policy both had their own beliefs about what to measure next, which is the decision this agent
exists to make; they were removed rather than kept as a control arm.

**There is no `--seed-hash` either, and its absence is the point.** It existed because the old
solver needed a predicted ITL per tenant to size a KV pool, refused to guess one, and so could not
run at all against a new validity key whose store is empty. The prediction-free solver sizes pools
from the fleet's own stated targets (impl §9), so an empty store is no longer an obstacle: a brand
new key plans and searches on day one. Borrowing measurements across keys was always in tension
with what a validity key is for, and nothing needs it now. If retrieval later wants evidence from a
neighbouring key, that has to be argued for on its own terms rather than inherited from a flag.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

from ..retrieval import required_relaxation, targets_met
from ..store.hubfacts import DEFAULT_HUB_ENDPOINT, HubFactsStore
from ..store.hwfacts import Disagreement, HardwareIndex
from ..store.profile import RESOLVED, ProfileStore
from ..store.schema import Observation
from ..store.store import ObservationStore
from . import export as canonical
from . import fleetfile
from .calibrate import between_run_halfwidth
from .capture import (DEFAULT_DAEMON_CHART, DEFAULT_HOLDER_MANIFEST, DO_NOT_DISRUPT, CaptureError,
                      EnvironmentMoved, HelmCapture, new_run_token)
from .deploy import DeployError, HelmDeployer, check_prerequisites
from .hardware import UnknownHardware, hardware_for
from .loop import OUTCOME_MEASURED, SearchAgent, _worst_itl_ms, closest_to_targets
from .measure import (HARNESS_IMAGE_ENV, DEFAULT_JOB_TEMPLATE, JobHarness,
                      default_harness_image)
from .mirror import EfsMirror, HardwareMirror, ProfileMirror
from .packer import (DEFAULT_MAX_TURNS, PACKER_FALLBACK, ModelPackerAgent,
                     expectation_report)
from .profiler import SoloProfiler
from .render import DEFAULT_NAMESPACE, DEFAULT_STORE_ROOT
from .sdkpacker import SdkPackerAgent
from .. import engines
from .spec import AGENT_ENVELOPE_STEP_MIB, DEFAULT_TOLERANCE_PCT, MIN_POOL_REQUESTS, TenantSlo
from ..solver.baseline import fleet_baseline
from ..solver.cpsat import Infeasible, PlacementSolver, Unproven
from ..solver.partition import disjoint_nodes, split
from .splitpacker import SplitPackerAgent, SplitTools
from .splits import GroupRun, drive, recall_group, text_sha256, write_group

log = logging.getLogger("colocation.agent")

# tutorials/multi-model-gpu-sharing/ -- the directory the chart and Job template paths are relative to.
REPO = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_STORE = Path.home() / ".colocation" / "store"

# How far a target may be stretched before the report stops offering the placement behind it as a
# RELAXED reading. The bound belongs to the fleet file -- `slo.tolerance_pct`, per tenant -- and this
# is only what applies when a tenant does not state one, which is why it is `spec`'s constant and not
# a second number here. Nothing is ever judged against a relaxed target: STRICT is always computed
# and printed against the ceilings as written. The reading exists because "3 GPUs missed" and "3 GPUs
# missed by 2.9% on one tenant of five" are different findings, and only the second can be acted on
# by the person who wrote the requirement. Beyond a few percent the question stops being "is this
# target exactly right" and becomes "is this the right fleet", which no report should answer for
# anybody -- so a tenant that will not negotiate says `tolerance_pct: 0`.
DEFAULT_RELAX_TARGETS_PCT = DEFAULT_TOLERANCE_PCT


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m colocation.agent",
                                description="propose, deploy, measure, record")
    p.add_argument("mode", choices=("plan", "calibrate", "profile", "search", "export"))
    p.add_argument("--fleet", required=True, help="fleet YAML (see agent/fleetfile.py)")
    p.add_argument("--store", default=str(DEFAULT_LOCAL_STORE),
                   help="local mirror of the observation store")
    p.add_argument("--efs-store", default=DEFAULT_STORE_ROOT,
                   help="the canonical store root, inside the cluster")
    p.add_argument("--local-only", action="store_true",
                   help="`plan` and `export` only: never reach the cluster. The EFS store is "
                        "neither read nor written, no kubectl command is run, and --store is the "
                        "whole of what the run sees. For a laptop with no cluster, and for a "
                        "store that must stay apart from the real one. Refused by the modes that "
                        "measure or calibrate, which need the full history on EFS")
    p.add_argument("--hub-endpoint", default=DEFAULT_HUB_ENDPOINT,
                   help="where a checkpoint's config.json and safetensors headers are read from. "
                        "Every `models:` entry is an id and a commit, and the weight bytes and KV "
                        "page size behind it are derived from the checkpoint itself rather than "
                        "stated (store/hubfacts.py). Point this at a mirror when the hub is not "
                        "reachable; manifests already cached under --store are used without asking "
                        "it anything")
    p.add_argument("--gpu-count", type=int, default=None,
                   help="how many GPUs one node of this instance type has. Needed only to bootstrap "
                        "an instance type nobody has leased yet: the count is normally an "
                        "OBSERVATION, read off a captured node and filed in the hardware store "
                        "(store/hwfacts.py), and there is no table of it in the source. Stating it "
                        "here is checked against the leased node before it can affect any "
                        "measurement and filed afterwards, so it is needed exactly once per type. It "
                        "may not contradict a shape already filed")
    p.add_argument("--gpu-product", default=None,
                   help="which GPU that instance type carries, as the Karpenter token (l40s, a10g, "
                        "h100). Only useful beside --gpu-count, and only to `plan`: it names the "
                        "card whose measured device total should price the plan. A mode that leases "
                        "reads the product off the node instead")
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    # Unset means every tenant is installed from its own engine's chart, which is the only thing that
    # works for a fleet whose tenants do not share a stack -- `engine:` is a per-tenant field, so one
    # chart for the whole run can be the wrong chart for some of it. Passing a path overrides that for
    # a chart under development, and `deploy.HelmDeployer._chart_for` refuses the override for a
    # tenant it does not belong to: rendering one stack's values into another stack's templates is
    # silent, because Helm ignores a values block a chart does not recognise.
    p.add_argument("--chart", default="",
                   help="install every tenant from this chart instead of from its engine's. Only "
                        "accepted where it is the same chart the engine names, so it is for working "
                        "on a copy of one, not for changing which stack a fleet runs on")
    p.add_argument("--daemon-chart", default=str(REPO / DEFAULT_DAEMON_CHART),
                   help="the mps-daemon chart this run installs to capture its node")
    p.add_argument("--holder-manifest", default=str(REPO / DEFAULT_HOLDER_MANIFEST),
                   help="applied in the lease's place by --release-node, so the node and its "
                        "GPU UUIDs survive")
    p.add_argument("--no-holder", action="store_true",
                   help="with --release-node, let the node go instead of handing it to the "
                        "holder. Karpenter reclaims it in ~600s and the replacement has "
                        "different GPU UUIDs, which no stored placement can be re-used against")
    p.add_argument("--capture-timeout", type=float, default=1800.0,
                   help="how long to wait for a node. Long by default: this includes Karpenter "
                        "provisioning one from scratch")
    p.add_argument("--run-token", default=None,
                   help="the run's identity, which names its daemon release and prefixes its "
                        "tenant releases. Random by default; set it only to reproduce a name. "
                        "`export` reads it instead of writing it: it names which stored search to "
                        "put in canonical form, and defaults to the newest under the key")
    p.add_argument("--export-to", default=None,
                   help="an extra copy of the canonical form, for `export` and for the one a "
                        "`search` writes as it finishes. Both always write "
                        "exports/<run token>.yaml under the key in the local store and push the "
                        "same file to EFS; a path here is written as well, not instead")
    p.add_argument("--job-template", default=str(REPO / DEFAULT_JOB_TEMPLATE))
    p.add_argument("--harness-image", default=default_harness_image(),
                   help=f"the §6 harness image. Defaults to ${HARNESS_IMAGE_ENV} if that is "
                        "exported, and has no default beyond it: the image has to be built from "
                        "containers/colocation-harness/ and pushed to a registry this cluster can "
                        "pull from (USER-GUIDE.md, step 3). Both modes that drive a load need it -- "
                        "`profile` and `search` are refused without one before the cluster is "
                        "touched at all; `plan` and `calibrate` never look at it")
    p.add_argument("--budget", type=int, default=6,
                   help="measurements the search may spend. The only GPU cost there is. `profile` "
                        "counts its solo rounds against the same budget")
    p.add_argument("--reprofile", action="store_true",
                   help="`profile` only: re-measure degrees the profile store already holds a "
                        "usable row for. Off by default -- a row is the current statement about a "
                        "model at a degree on this hardware, so re-running the mode after adding a "
                        "tenant costs GPU time only for what is missing. Pass this when the thing "
                        "that changed is not in the key: a chart change, a model's weights on disk, "
                        "or a deliberate replicate to see the noise floor")
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                   help="reasoner turns, including the free ones. Bounds the conversation, "
                        "not the search")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--reasoner-model", default=None,
                   help="Bedrock model id for the reasoner. Defaults to reason.DEFAULT_MODEL_ID")
    p.add_argument("--driver", choices=("sdk", "converse"), default="sdk",
                   help="how the reasoner is driven. 'sdk' is the Claude Agent SDK, which is what "
                        "impl §11 names and what a live run should use: real tool calls, and a "
                        "deny-by-default gate so `measure` is the only route to the GPUs. "
                        "'converse' is the JSON-in-text protocol the self-tests drive; it reaches "
                        "the same tools, but its parsing is not the one that ships")
    p.add_argument("--ready-timeout", type=float, default=1800.0)
    p.add_argument("--measure-timeout", type=float, default=5400.0)
    p.add_argument("--solver-seconds", type=float, default=10.0)
    p.add_argument("--teardown", action="store_true",
                   help="uninstall the tenant releases when the run finishes")
    p.add_argument("--release-node", action="store_true",
                   help="also give the node lease up at the end, applying the holder manifest in "
                        "its place. Implies --teardown, since uninstalling the daemon takes every "
                        "tenant's MPS server with it anyway")
    p.add_argument("--allow-disjoint-only", action="store_true",
                   help="search even when arithmetic already shows no co-resident placement is "
                        "feasible -- no two tenants' minimum envelopes fit on one GPU. Refused by "
                        "default because such a run can only re-derive the disjoint baseline at the "
                        "cost of a node-hour. Right when the baseline itself is what you are "
                        "measuring")
    p.add_argument("--relax-targets-pct", type=float, default=None,
                   help="reporting only, and a ceiling on the fleet file rather than a substitute "
                        "for it: each tenant's tolerance is `slo.tolerance_pct` (default "
                        f"{DEFAULT_TOLERANCE_PCT:g}%%), and this can only lower it, never raise it. "
                        "Use it to ask a narrower question than the fleet allows -- 'what holds "
                        "within 1%%?'. Nothing is judged against a relaxed target and no run is "
                        "ever claimed on one: the report states what relaxing would cost, per "
                        "tenant, and whether the miss is larger than the calibrated noise. 0 turns "
                        "the offer off for every tenant and still prints the required relaxation, "
                        "because that number is the finding")
    p.add_argument("--summary", default=None, help="write the run summary here as JSON")
    p.add_argument("--group-budget", type=int, default=4,
                   help="for a fleet with hardware.max_nodes above 1: how many groups `search` "
                        "may search, each its own node and its own --budget of measurements")
    p.add_argument("--split-chooser", choices=("reasoner", "drive"), default="reasoner",
                   help="for a fleet with hardware.max_nodes above 1: who chooses which group to "
                        "search next. 'reasoner' is the --reasoner-model, over `converse`, with "
                        "the deterministic driver as its recorded fallback; 'drive' is that "
                        "driver alone")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _stores(args, key) -> tuple:
    """The local mirror for this key. One key, one store: nothing is borrowed from another."""
    if args.local_only:
        # No mirror at all, so nothing below can reach the cluster. See `--local-only`.
        store = ObservationStore(args.store, key)
        store.write_key()
        return store, None
    mirror = EfsMirror(key.hash, args.store, efs_root=args.efs_store,
                       namespace=args.namespace)
    mirror.pull()
    store = ObservationStore(args.store, key)
    store.write_key()
    mirror.push_key(key.key_json())
    return store, mirror


def _hardware(args) -> tuple:
    """The local hardware store, brought up to date with EFS first, and its mirror.

    Pulled before anything resolves an instance type, because that is what the resolution reads: how
    many GPUs the type has and how big one of its GPUs is are both filed rather than tabulated
    (`agent/hardware.py`), and a desktop that has never captured holds neither. Every environment's
    sheets come down, not this run's, for the reason `HardwareIndex` reads across them.

    A cluster that cannot be reached is a warning and not a refusal. The local store may well be
    enough -- `plan` is meant to work on a laptop -- and if it is not, the refusal that follows
    already says what to run. Failing here instead would turn "nothing has measured this card" into
    "kubectl exec failed", which is a worse message for the same situation.
    """
    if args.local_only:
        return HardwareIndex(args.store), None
    mirror = HardwareMirror(args.store, efs_root=args.efs_store, namespace=args.namespace)
    try:
        pulled = mirror.pull()
        log.info("hardware store: %d device law(s) under %d environment(s) and %d instance shape(s) "
                 "pulled from EFS", pulled["devices"], len(pulled["keys"]), pulled["shapes"])
    except Exception as exc:                                          # noqa: BLE001 -- see above
        log.warning("could not pull the hardware store from EFS (%s). Continuing against whatever "
                    "is in %s; if nothing has measured this hardware, the refusal will say so.",
                    exc, args.store)
    return HardwareIndex(args.store), mirror


def _push_hardware(mirror, index) -> list:
    """Copy everything the local hardware store now holds up to EFS. Logged, never fatal.

    Everything rather than just what this run filed, because the operation is idempotent and the
    files are a handful of small JSON documents -- while working out which sheet was new would mean
    threading that through `capture.reconcile_hardware`, which does not know a cluster exists.

    Never fatal for `mirror`'s own reason: the measurement is on local disk either way, and losing a
    node lease over a failed copy costs the GPU time that produced it.
    """
    pushed = []
    for shape in index.shapes():
        try:
            pushed.append(mirror.push_shape(shape.id, shape.to_dict()))
        except Exception as exc:                                      # noqa: BLE001
            log.warning("could not push %s's shape to EFS: %s", shape.instance_type, exc)
    for key_hash in index.key_hashes():
        key = index.read_key(key_hash)
        try:
            if key is not None:
                mirror.push_key(key_hash, key.key_json())
            for sheet in index.store_for(key_hash).all():
                pushed.append(mirror.push_sheet(key_hash, sheet.id, sheet.to_dict()))
        except Exception as exc:                                      # noqa: BLE001
            log.warning("could not push the device law(s) under %s to EFS: %s", key_hash, exc)
    if pushed:
        log.info("hardware store: pushed %d file(s) to EFS", len(pushed))
    return pushed


def _bootstrap_environment(args, index, mirror, why: UnknownHardware) -> int:
    """Hold one node of an instance type nobody has observed, file what it reports, and stop.

    The one run this stack cannot do offline, and the reason it exists is the ordering: the profile
    store's directory is named by what the card presents -- addressable MiB, enforced power limit,
    link matrix -- so it cannot be opened before a node has been held, while every tenant's TP degree
    is resolved out of that store before the fleet can be built. So the first run on a new instance
    type cannot both observe the machine and measure on it.

    It stops rather than carrying on with the address it just learned, which is `EnvironmentMoved`'s
    argument in the other direction: a run that switched store directories half way through would
    have resolved its degrees and printed its baseline out of one directory and filed its rows in
    another, with nothing anywhere saying so. Stopping costs one node lease of a few minutes and the
    re-run needs no flag, no edit and no decision from anybody -- which is the property worth paying
    for.

    The node is given back at the end, and that is not the usual asymmetry (`_release`): this run
    measured nothing, so there is nothing on the machine to keep, and a *held* node is exactly what
    would make the re-run buy a second one -- its own daemon release cannot preempt this one's, which
    is default priority. Handed to the holder instead, which the re-run preempts.
    """
    doc = fleetfile.read_doc(args.fleet)
    # Resolved again here rather than passed in, so that an instance type with no shape AND no
    # `--gpu-count` raises `hardware_for`'s own refusal, which is the one that names the flag.
    hardware = hardware_for(fleetfile.fleet_instance_type(doc), index,
                            gpu_count=args.gpu_count, gpu_product=args.gpu_product)
    image = fleetfile.fleet_image(doc)
    capture = HelmCapture(args.daemon_chart, namespace=args.namespace,
                          holder_manifest=None if args.no_holder else args.holder_manifest)
    token = args.run_token or new_run_token()
    print(f"\n{why}\n\nSo this run observes the machine and nothing else. Capturing a "
          f"{hardware.instance_type} as run {token}: installing daemon release mps-{token}, which "
          f"requests all {hardware.gpu_count} GPUs on one node. If no held node can be preempted, "
          "Karpenter provisions one -- this is where the GPU spend starts, and it is the only spend "
          "this run makes: no engine is booted and no measurement is taken.")
    lease = capture.capture(token, hardware, timeout_s=args.capture_timeout,
                            index=index, image=image)
    print(f"  {lease.summary()}")
    _push_hardware(mirror, index)
    # Read back off disk rather than off the lease, because what the re-run will resolve the address
    # from is the file, and a shape the capture could not write is the one failure this run has to
    # report -- it would otherwise print success and leave the next run refusing for the same reason
    # as this one.
    shape = index.read_shape(hardware.instance_type)
    capture.release(lease)
    if shape is None or not shape.p2p_topology:
        raise SystemExit(
            f"held {lease.node_name} and read it, but the hardware store has no usable shape for "
            f"{hardware.instance_type} afterwards, so nothing was learned and a re-run would refuse "
            "for the same reason. The capture logs a warning naming why the write failed; the node "
            "has been given back either way.")
    print(f"\nfiled {shape.instance_type}: {shape.gpu_count}x{shape.gpu_product}, "
          f"{shape.addressable_mib} MiB addressable per device, {shape.power_limit_w} W enforced, "
          f"links {shape.p2p_topology} -- read off {lease.node_name}.")
    print(f"\n{lease.node_name} is back under the holder and nothing is deployed on it. Re-run this "
          "exact command: the address is now derived offline from the shape above, so every run "
          "after this one resolves it for free.")
    return 0


def _hub(args) -> HubFactsStore:
    """The checkpoint-manifest store: what makes a `models:` entry an id and a commit.

    Built for every mode including `plan`, because every mode loads the fleet and no fleet can be
    priced without it: `weight_gib` and the KV page size are derived from the checkpoint's own
    `config.json` and safetensors headers rather than stated (`store/hubfacts.py`), and a model whose
    manifest cannot be had is refused by name rather than defaulted.

    Nothing is pulled from EFS and nothing is pushed to it, unlike every other store here. A manifest
    is refetchable from a public URL, so mirroring it would be caching a cache -- and it is not a
    measurement, so losing it costs HTTP requests rather than GPU time. Cached under `--store` so a
    laptop that has loaded a fleet once can load it again on a plane.

    `HF_TOKEN` is read if it is exported and passed through unlogged, which is the only way a gated
    checkpoint can be priced at all. None of the checkpoints in `fleets/` is gated, so it is normally
    absent; the store never prints it and never puts it in a cached manifest.
    """
    return HubFactsStore(args.store, endpoint=args.hub_endpoint,
                         token=os.environ.get("HF_TOKEN") or None)


def _profiles(args, index) -> tuple:
    """The solo-profile store for this fleet's *environment*, and its EFS mirror.

    Opened from the fleet file's instance type and image resolved against the hardware store
    (`fleetfile.profile_key`), because the tenants cannot be built until it is open: every tenant's
    TP degree is resolved from it. Keyed by the environment and not by the fleet, so rows a previous
    fleet paid for on this hardware are already here -- which is the whole reason minimal TP is
    affordable at all.

    `index` is why `_hardware` runs before this: the address is what a node of this type reported
    about itself, so an instance type nobody has ever leased has no directory and this refuses
    (`hardware.Hardware.environment`). `main` turns that refusal into the one run that fixes it.
    """
    key = fleetfile.profile_key(args.fleet, index, gpu_count=args.gpu_count,
                                gpu_product=args.gpu_product)
    if args.local_only:
        store = ProfileStore(args.store, key)
        store.write_key()
        return store, None
    mirror = ProfileMirror(key.hash, args.store, efs_root=args.efs_store,
                           namespace=args.namespace)
    pulled = mirror.pull()
    store = ProfileStore(args.store, key)
    store.write_key()
    mirror.push_key(key.key_json())
    log.info("solo profiles under %s: %d row(s) pulled from EFS, %d local",
             key.hash, pulled.get("solo", 0), len(store.all()))
    return store, mirror


def _print_placement(label: str, candidate, fleet) -> None:
    """One feasible placement, as constraints met. No predicted latency, because there is none."""
    print(f"\n  {label}")
    for t in fleet.tenants:
        gpus = candidate.placement.assignment[t.name]
        floor = candidate.floor_tokens.get(t.name, 0)
        margin = candidate.margin.get(t.name)
        print(f"    {t.name:<6} tp{t.tp} gpus={list(gpus)!s:<14} "
              f"env={candidate.placement.envelope_gib[t.name]:>5.1f}GiB "
              f"util={candidate.placement.util[t.name]:.4f} "
              f"coresident={candidate.coresidency[t.name]} "
              f"kv={candidate.placement.kv_tokens[t.name]:>7} "
              f"floor={floor:>7} "
              + (f"({margin:.2f}x)" if margin is not None else "(none stated)"))
    if candidate.min_margin is None:
        print(f"    -> {candidate.gpus_used} GPU(s); no tenant stated a floor")
    else:
        print(f"    -> {candidate.gpus_used} GPU(s); tightest margin "
              f"{candidate.min_margin:.2f}x at {candidate.tightest_tenant}")


def _baseline(fleet):
    """The offline baseline for this fleet. No node, no store, no GPU time."""
    return fleet_baseline(fleet.node.node_spec, fleet.requests,
                          envelope_step_mib=AGENT_ENVELOPE_STEP_MIB)


def pairs_fitting_on_weights(base, fleet) -> tuple:
    """Tenant pairs whose two shards fit one GPU counting *weights alone* (impl §14.2).

    The witness question for "no co-resident placement is feasible", which is a conclusion about
    whether co-location has anything to sell this fleet and therefore whether to spend a node-hour.
    `FleetBaseline.can_coreside` answers it on full minimum envelopes -- weights plus a KV pool sized
    from a derived page size, plus an activation fallback, plus a reserve constant. This answers it
    on the one term that cannot come back smaller than stated.

    A non-empty result does not mean any pair fits: an engine with no KV cache serves nothing. It
    means arithmetic is not entitled to say the pair does not, so the conclusion is owed to a
    measurement. Empty means the weights alone settle it, and then the refusal is a fact.

    Returns `()` when `fleet` is None, which is how a caller with no requests to hand says it cannot
    ask -- silence, not a claim either way.
    """
    if fleet is None:
        return ()
    mem = {r.name: r.mem for r in fleet.requests}
    tp = {r.name: r.tp for r in fleet.requests}
    total = fleet.node.node_spec.device_total_gib
    return tuple(p for p in base.pairs
                 if p.a in mem and p.b in mem
                 and mem[p.a].weight_lower_bound_gib(tp[p.a])
                 + mem[p.b].weight_lower_bound_gib(tp[p.b]) <= total)


def _print_baseline(base, unresolved=(), upper_bound=(), fleet=None) -> None:
    """What the search has to beat, and what arithmetic already rules out (impl §0).

    Printed before any placement, because both halves are decisions about whether to run at all: a
    baseline nobody has measured is not a baseline, and a fleet where no two shards fit on one GPU
    has no co-resident placement for a search to find.

    `unresolved` names the tenants whose degree is a memory floor standing in for a measurement
    nobody has taken. Their arithmetic is worth printing -- it is what says which degrees to profile
    -- but the *sum* is then a lower bound on the baseline and not the baseline, and the difference
    is the whole of §0. So it is printed under a different name.

    `upper_bound` names the other half of the same problem, and the half that survives now that the
    fleet file cannot state a degree: a tenant that met its target solo at the degree shown while
    some *narrower* degree was never measured. Its degree is real, and it may still be one wider
    than necessary, so the baseline is an upper bound (`fleetfile.upper_bound_tp`). A degree above
    the memory floor is otherwise the ordinary case rather than a warning sign -- it means the floor
    degree was measured and missed.
    """
    print("\n--- the baseline co-location has to beat (impl §0) ---")
    for t in base.tenants:
        floor = "none fits" if t.floor_tp is None else f"tp{t.floor_tp}"
        if t.name in unresolved:
            mark = "  <-- FLOOR, not a measured minimum"
        elif t.name in upper_bound:
            mark = "  <-- UPPER BOUND: a narrower degree was never measured"
        elif t.above_floor:
            mark = "  <-- above the floor, and measured to need it"
        else:
            mark = ""
        print(f"  {t.name:<6} tp{t.resolved_tp:<3} memory floor {floor:<9} "
              f"env {t.envelope_at_resolved_gib:>5.2f} GiB/shard at tp{t.resolved_tp}{mark}")
        print(f"  {'':<6} {t.why}")
    if unresolved:
        # "no solo profile" would be the same mislabelling `TpResolution` carries reason codes to
        # avoid: a row may exist and still be unable to settle a degree -- an unvouched verdict, a
        # failed delivery gate, sampling too thin to trust. Which one it is, per tenant and per
        # degree, is on the `minimal TP is not established` line logged when the fleet was loaded,
        # so this says what is missing (a usable measurement) without claiming nothing ran.
        print(f"\n  BASELINE NOT ESTABLISHED. {', '.join(sorted(unresolved))} have no solo "
              "measurement that can settle a degree, so the degrees above are memory floors -- "
              "and a fleet file cannot state a degree "
              f"instead. {base.resolved_gpus} GPU(s) of {base.gpu_count} is therefore a LOWER BOUND "
              "on the baseline, not the baseline -- a degree that fits may still miss its target, "
              "and only a solo measurement settles which. Run `profile`, then read this line again.")
        return
    print(f"\n  BASELINE {base.resolved_gpus} GPU(s) of {base.gpu_count} with no GPU shared, at the "
          "TP degrees these tenants resolved to.")
    if not base.fits_disjoint:
        # The sharpest case for co-location: the fleet cannot be served at all without sharing, so
        # any feasible placement is a win rather than an improvement on one.
        print(f"  This fleet CANNOT be served disjointly: {base.resolved_gpus} shards do not fit on "
              f"{base.gpu_count} GPUs. Every feasible placement below shares at least one GPU.")
    if base.unplaceable:
        print(f"  {', '.join(base.unplaceable)} fit at no TP degree this node can offer -- their "
              "KV floor exceeds one GPU however the weights are split.")
    if base.floor_gpus != base.resolved_gpus:
        gap = base.resolved_gpus - base.floor_gpus
        print(f"  At the memory floor the same baseline would be {base.floor_gpus} GPU(s), {gap} "
              f"fewer: {', '.join(base.above_floor)} carry more shards than the weights and their "
              "own KV floor require.")
        if not upper_bound:
            print("  Which is established rather than inflated. Each of those degrees is the "
                  "narrowest that met the tenant's own target in a SOLO run, so the narrower "
                  "degree was measured to miss -- re-sharding cannot reach the memory floor here, "
                  f"and those {gap} GPU(s) are not co-location's to sell either.")
    if upper_bound:
        print(f"\n  THE BASELINE ABOVE IS AN UPPER BOUND. {', '.join(sorted(upper_bound))} met "
              "their target solo at the degree shown, but a narrower degree was never measured, so "
              "nothing rules the narrower one out. Run `profile` for the missing degrees; until "
              "then a saving scored against this number may include GPUs that re-sharding alone "
              "would free (impl §0).")
    if not base.can_coreside:
        tight = base.tightest_pair
        closest = ("" if tight is None else
                   f" The closest pair is {tight.a}+{tight.b} at {tight.total_gib:.2f} GiB of "
                   f"{tight.device_total_gib:.2f}, over by {tight.deficit_gib:.2f}.")
        print("\n  NO CO-RESIDENT PLACEMENT IS FEASIBLE. No two of these tenants' minimum "
              f"envelopes fit on one GPU, and the solver may raise an envelope above its floor but "
              f"never below -- so this holds at every envelope, not just the tight ones.{closest}")
        print("  A search on this fleet can only reproduce the disjoint baseline. That is a valid "
              "answer and it costs nothing to have here: co-location has nothing to sell to this "
              "fleet (impl §0), and `search` refuses it unless --allow-disjoint-only says so.")
        # ...unless the pairs failed on the computed terms rather than on the weights, in which case
        # the paragraph above is arithmetic talking about a fleet nobody booted (impl §14.2).
        by_weight = pairs_fitting_on_weights(base, fleet)
        if by_weight:
            names = ", ".join(f"{p.a}+{p.b}" for p in by_weight[:3])
            print(f"  BUT NOT ON THE WEIGHTS: {len(by_weight)} of {len(base.pairs)} pairs fit one "
                  f"GPU counting weights alone ({names}{', ...' if len(by_weight) > 3 else ''}). "
                  "What does not fit is the KV pool, activation and reserve above them -- computed "
                  "terms, and the KV page size behind them was 3.27x too large the one time it met "
                  "a booted engine. So the paragraph above is what arithmetic expects, not what "
                  "this fleet does: `profile` the pair's tenants before accepting it.")
    else:
        print(f"  {len(base.shareable)} of {len(base.pairs)} tenant pairs fit on one GPU, so a "
              "co-resident placement exists to be searched for.")


def _plan(args, fleet, store, unresolved=(), upper_bound=(), max_nodes=1) -> int:
    print(f"\nfleet {args.fleet}: {len(fleet.tenants)} tenants on {fleet.node.node_name} "
          f"({len(fleet.node.gpu_ids)} GPUs, {fleet.node.device_total_gib:.2f} GiB each)"
          + (f", up to {max_nodes} nodes of it" if max_nodes > 1 else "")
          + f", store {store.key_hash}, {len(store.load().observations)} observations")
    if not fleet.node.leased:
        # Said plainly, because the placements below name GPU ordinals and look deployable.
        print("  No node is captured, so these are placements on the shape of the instance type. "
              "`search` captures one and re-checks this fleet against it.")

    print("\n--- what each tenant's own target requires ---")
    for t in fleet.tenants:
        derived = t.kv_tokens_at_target()
        if derived is None:
            print(f"  {t.name:<6} no ITL ceiling stated: no floor follows, and no measurement "
                  "can certify it")
        else:
            tokens, how = derived
            print(f"  {t.name:<6} {tokens:>7} KV tokens   ({how})")
        if t.min_kv_tokens:
            print(f"  {'':<6} {t.min_kv_tokens:>7} KV tokens   minimum viable pool, "
                  f"{MIN_POOL_REQUESTS} x max_model_len {t.max_model_len}; the larger binds")
        slots = t.concurrency_at_slot_occupancy()
        how = "no ITL ceiling, so the engine's own default" if slots is None else \
            f"{slots:.1f} slots held at the arrival burst, floored at the engine's own default"
        print(f"  {'':<6} {t.max_num_seqs:>7} max_num_seqs ({how})")

    base = _baseline(fleet)
    _print_baseline(base, unresolved, upper_bound, fleet=fleet)
    if max_nodes > 1:
        return _plan_splits(args, fleet, base, max_nodes, unresolved)

    solver = PlacementSolver(fleet.node.node_spec, fleet.requests,
                             envelope_step_mib=AGENT_ENVELOPE_STEP_MIB)
    print("\n--- feasible placements, densest first ---")
    if unresolved:
        # The solver was handed floor degrees, so these placements are the ones that exist IF the
        # floor turns out to be enough. If `profile` finds a tenant needs a wider degree, its shard
        # count changes and so does every placement below. Said here rather than only at the
        # baseline, because this is the list that looks deployable.
        print(f"  At the memory floors above, not measured minima: {', '.join(sorted(unresolved))} "
              "may yet need a wider degree, and every placement here changes if one does.")
    try:
        feasible = solver.solve(cap=args.top_k, max_seconds=args.solver_seconds)
    except Unproven as exc:
        # Not an answer about the fleet: an answer about the arithmetic (impl §14.2). Zero, and the
        # same zero `BASELINE NOT ESTABLISHED` exits with, because the remedy is identical -- go and
        # measure. Exiting 1 here would tell a script the fleet file has to change, which is the one
        # conclusion nothing above supports.
        print(f"  THIS FLEET IS NOT PROVEN INFEASIBLE, only unplaced by arithmetic: {exc}")
        return 0
    except Infeasible as exc:
        # A real answer about the fleet, not a crash: these tenants cannot be hosted on this node
        # at these rates and targets. Non-zero, so `plan` fails a script that assumed otherwise.
        print(f"  this fleet cannot be placed on this node: {exc}")
        return 1
    for i, candidate in enumerate(feasible):
        _print_placement(f"#{i + 1}  {_compact(candidate.placement.assignment)}",
                         candidate, fleet)
    # Said outright rather than implied by the length of the list: impl §9 requires a caller to be
    # able to tell "there are no others" from "the search stopped looking".
    print(f"\n  complete: {feasible.exhausted}   ({feasible.stopped_because})")
    if feasible.truncated:
        print("  Other feasible placements exist that are not listed above.")
    return 0


def _plan_splits(args, fleet, base, max_nodes, unresolved=()) -> int:
    """`plan` for a fleet that may use more than one node: the splits into one group per node.

    Each group is placed exactly as a single-node fleet is, so the best split's groups are printed
    with `_print_placement`, and the rest as one line each. Exit codes are `_plan`'s: 1 only when
    the weights alone rule every split out, 0 when what closed them is computed terms.
    """
    gpus = len(fleet.node.gpu_ids)
    need = disjoint_nodes([t.tp for t in fleet.tenants], gpus)
    print(f"  Disjoint over nodes of {gpus} GPUs, with each tenant on one node: {need} node(s)"
          + ("" if need <= max_nodes else f", more than the {max_nodes} this fleet allows") + ".")
    print(f"\n--- splits over up to {max_nodes} nodes, fewest nodes first ---")
    if unresolved:
        print(f"  At the memory floors above, not measured minima: {', '.join(sorted(unresolved))} "
              "may yet need a wider degree, and every split here changes if one does.")
    result = split(fleet.node.node_spec, fleet.requests, max_nodes, cap=args.top_k,
                   max_seconds=args.solver_seconds, envelope_step_mib=AGENT_ENVELOPE_STEP_MIB)
    for name, why in result.unplaced.items():
        print(f"  {name} fits no node even alone: {why}")
    if not result.splits:
        if result.proven_none:
            print(f"  this fleet cannot be split over {max_nodes} node(s): the weights alone rule "
                  "every split out.")
            return 1
        if not result.exhausted:
            print(f"  no split found within --solver-seconds {args.solver_seconds} per solve, "
                  "which proves nothing about the fleet. Raise it and run `plan` again.")
            return 0
        print(f"  NO SPLIT OVER {max_nodes} NODE(S) IS PROVEN INFEASIBLE, only unplaced by "
              "arithmetic: the weights alone fit, so what closed it is the KV pool, activation "
              "and reserve, which are computed terms (impl §14.2). `profile` can settle them.")
        return 0
    for i, s in enumerate(result.splits):
        margin = "no floor stated" if s.min_margin is None else f"tightest margin {s.min_margin:.2f}x"
        print(f"\n  #{i + 1}  {s.nodes} node(s), {s.gpus} GPU(s), {margin}")
        for j, g in enumerate(s.groups):
            print(f"    node {j + 1}: {' '.join(g.tenants)}  ({g.gpus} GPU(s))  "
                  f"{_compact(g.candidate.placement.assignment)}")
    best = result.splits[0]
    print("\n--- the first split, node by node ---")
    for j, g in enumerate(best.groups):
        _print_placement(f"node {j + 1}", g.candidate,
                         replace(fleet, tenants=tuple(fleet.tenant(n) for n in g.tenants)))
    print(f"\n  complete: {result.exhausted}   "
          + ("(no other split exists)" if result.exhausted else
             f"(stopped at the cap of {args.top_k}, or a solve ran out of time; there may be "
             "more)"))
    print("  Which split meets its targets is a measurement; nothing here predicts it.")
    return 0


def _compact(assignment: dict) -> str:
    return " ".join(f"{n}:{','.join(str(g) for g in sorted(gpus))}"
                    for n, gpus in sorted(assignment.items()))


def _caps_text(caps) -> str:
    """`, SM cap a 40% b 60%` for a capped attempt, "" for an uncapped one (impl §16).

    Printed beside every placement the report names, because a capped placement holds only with
    its caps: the same assignment uncapped is a different placement nobody measured.
    """
    return ("" if not caps else
            ", SM cap " + " ".join(f"{n} {int(c)}%" for n, c in sorted(caps.items())))


def _calibrate(args, store, mirror) -> int:
    """Write the noise floor implied by repeats already in the store. No GPU time."""
    report = store.load()
    cal = between_run_halfwidth(report.observations,
                               notes={"validity_key_hash": store.key_hash})
    if cal is None:
        print(f"no placement under {store.key_hash} has been measured more than once, so there "
              "is no between-run spread to report. A repeat is what produces one: the reasoner "
              "may choose a placement the store already holds, and this is what that buys.")
        return 1
    path = store.write_calibration(cal)
    mirror.push_calibration(json.dumps(cal.to_dict(), indent=2, sort_keys=True) + "\n")
    print(f"\ncalibration from {cal.notes['repeated_placements']} repeated placement(s), "
          f"{cal.repeats} passes each -- no new measurement:")
    for bucket, hw in sorted(cal.between_run_itl_halfwidth_ms.items()):
        print(f"  {bucket:<16} between-run ITL half-width {hw:.4f} ms")
        print(f"                   {cal.notes['widest_pair_per_bucket'][bucket]}")
    print(f"\nwritten to {path} and pushed to EFS. This is the resolution of a measurement on "
          "this node -- two runs of one placement that differ by less than this differ by "
          "nothing. It is shown to the reasoner, not applied above it.")
    return 0


def _export(args, fleet, loaded, hardware, store, mirror, profiles) -> int:
    """This key's result in canonical form: slots, absolute GiB, no UUID and no ordinal. No GPU time.

    Runs at the end of every pipeline run, and is worth running before one: the baseline is exported
    unconditionally, so a fleet whose search has not happened yet still gets a deployable document and
    a number to beat. What a search adds to it is the strict placement, the relaxed offer, and the
    denser arrangements that were measured and missed.

    The one refusal is not here: a fleet whose degrees are memory floors rather than measurements
    cannot load in this mode at all (`allow_unresolved_tp` excludes it), and the loader's message
    already names `profile` as the remedy. That is deliberate rather than convenient -- an export is
    the document somebody deploys from, and a floor exported as a baseline is the one overstatement
    §0 exists to prevent, arriving in a file that outlives the conversation about where it came from.
    """
    try:
        summary, summary_path = canonical.newest_search(store, args.run_token)
    except canonical.ExportError as exc:
        # A named run token that is not there, or a summary that will not parse. Reported rather
        # than raised: the message already lists what the key does hold.
        print(f"  {exc}")
        return 1
    _write_canonical(fleet=fleet, loaded=loaded, hardware=hardware, store=store, mirror=mirror,
                     profiles=profiles, fleet_path=args.fleet, summary=summary,
                     summary_path=summary_path, export_to=args.export_to)
    return 0


def _write_canonical(*, fleet, loaded, hardware, store, mirror, profiles, fleet_path, summary,
                     summary_path, export_to=None) -> dict:
    """Build the document, write it everywhere it belongs, print what it says. Returns it.

    Shared by the `export` mode and the end of a `search`, so a run that exports itself and a run
    exported afterwards produce the same file from the same code -- two copies of this would be two
    answers, and the whole point of a canonical form is that there is one.
    """
    base = _baseline(fleet)
    document = canonical.build(
        fleet=fleet, base=base, fleet_path=fleet_path, validity_key_hash=store.key_hash,
        profile_key_hash=profiles.key_hash, hardware=hardware,
        summary=summary, summary_path=summary_path,
        upper_bound=fleetfile.upper_bound_tp(loaded),
        resolutions=loaded.get("tp_resolutions"), calibration=store.read_calibration(),
        records={o.run_id: o for o in store.load().observations})
    text = canonical.dump(document)

    name = (summary or {}).get("run_token") or "baseline"
    path = store.exports_dir / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    written = [str(path)]
    if export_to:
        Path(export_to).write_text(text)
        written.append(export_to)
    if mirror is not None:          # None only under --local-only, which never reaches EFS
        try:
            written.append(mirror.push_export(name, text))
        except Exception as exc:                                      # noqa: BLE001
            # The document is already on disk, and this mode is meant to work on a desktop that
            # cannot reach the cluster. A failed push loses nothing that cannot be re-derived.
            log.warning("could not push the export to EFS (%s); it is local only", exc)

    print(f"\n--- canonical form of {store.key_hash} ---")
    if summary is None:
        print("  no search has run under this key, so this is the BASELINE alone: every shard on "
              "its own device, at the degrees solo profiling settled.")
    else:
        print(f"  from search {summary.get('run_token')} ({summary_path}), verdict "
              f"{summary.get('verdict')}")
    baseline = document["baseline"]
    print(f"  BASELINE  {baseline['devices']} device(s)"
          + ("  <-- an upper bound: a narrower degree was never measured for "
             f"{', '.join(n for n, r in baseline['per_tenant'].items() if r['narrower_degree_unmeasured'])}"
             if baseline["upper_bound"] else ""))
    strict = document["strict"]
    if strict is None:
        print("  STRICT    none. No measured placement met every ceiling as written, so the "
              "baseline is what this document offers.")
    else:
        print(f"  STRICT    {strict['devices']} device(s) in "
              f"{len(strict['slots'])} slot(s), {strict['devices_freed']} freed "
              f"({strict['fraction_freed']:.0%} of the baseline), certified")
    relaxed = document["relaxed"]
    if relaxed is None:
        print(f"  RELAXED   none: {document['relaxed_why_not']}")
    else:
        print(f"  RELAXED   {relaxed['devices']} device(s), {relaxed['devices_freed']} freed, "
              "NOT CERTIFIED -- within tenants' own stated tolerances")
    print(f"  REJECTED  {len(document['rejected'])} measured placement(s) that missed, with the "
          "overshoot that ruled each one out")
    for line in written:
        print(f"  written to {line}")
    print("  The conditions it holds under are the fleet file's, by reference: "
          f"{document['fleet']['path']} at sha256 {document['fleet']['sha256'][:12]}. Slots carry "
          "no device identity, so nothing in here goes stale when the node does.")
    return document


def _print_profile_plan(profiler, hardware) -> int:
    """What `profile` is about to spend, per tenant, before it spends any of it.

    Printed because the cost is not obvious from the command: it is one deploy-and-drive cycle per
    (tenant, degree) round, and the number of rounds depends on what the store already holds -- which
    is a fact about this hardware's history rather than about this fleet.
    """
    print(f"\n--- solo profiling on a {hardware.instance_type}, one tenant at a time ---")
    print("Each round runs ONE tenant alone with the whole of each GPU to itself, which is the best "
          "case for that model at that rate and shape. A degree that misses its target there cannot "
          "be rescued by any placement, and a degree that meets it is the 'vs solo' reference every "
          "co-resident measurement is read against.")
    worst = 0
    for tenant, floor, degrees, have, band in profiler.plan():
        target = ("no itl_mean_ms target" if tenant.slo is None or tenant.slo.itl_mean_ms is None
                  else f"target {tenant.slo.itl_mean_ms:.3f} ms ITL")
        floor_txt = "none fits" if floor.floor_tp is None else f"tp{floor.floor_tp}"
        print(f"\n  {tenant.name:<6} {tenant.model_id}  {tenant.isl_tokens}/{tenant.osl_tokens} "
              f"tokens at {tenant.rate_hz:g} Hz, {target}")
        print(f"  {'':<6} memory floor {floor_txt}: {floor.why}")
        if not degrees and not band:
            print(f"  {'':<6} nothing to profile -- its weights fit at no TP degree this node offers")
            continue
        if len(degrees) + len(band) == 1:
            print(f"  {'':<6} tp{(degrees or band)[0]} is forced by memory alone, so no round is "
                  "spent: it is the only degree this model's weights leave")
            continue
        todo = [k for k in degrees if k not in have] if not profiler.reprofile else list(degrees)
        band = [k for k in band if k not in have] if not profiler.reprofile else list(band)
        print(f"  {'':<6} degrees to consider: {', '.join(f'tp{k}' for k in degrees)}"
              + (f"; already measured {', '.join(f'tp{k}' for k in have)}" if have else ""))
        print(f"  {'':<6} at most {len(todo)} round(s) upward, narrowest first, stopping at the "
              "first degree that meets the target")
        worst += len(todo)
        if band:
            # The band is what the arithmetic doubts and nothing has measured. Priced separately
            # because it is only reached if a degree above it passes, and because it stops at the
            # first degree that cannot hold the engine -- so this is a ceiling, not a count.
            print(f"  {'':<6} then up to {len(band)} round(s) DOWNWARD into "
                  f"{', '.join(f'tp{k}' for k in band)}: degrees the computed envelope refuses but "
                  "the weights allow, so only a boot can rule them out. Without them the minimum "
                  "above is an upper bound")
            worst += len(band)
    print(f"\n  at most {worst} measurement(s) in total, against a budget of "
          f"{'unbounded' if profiler.budget is None else profiler.budget}")
    return worst


def _profile_report(out, fleet, hardware) -> dict:
    """What the rounds established, and what is still unmeasured. The mode's whole output."""
    rounds, resolutions = out["rounds"], out["resolutions"]
    print("\n=== solo rounds ===")
    if not rounds:
        print("  none. Nothing needed measuring, or nothing could be.")
    print(f"{'tenant':<8} {'tp':>4} {'status':<18} {'ITL':>10} {'TTFT p95':>10}  {'delivered':>9}"
          "  run")
    for r in rounds:
        p = r.profile
        itl = "-" if p is None or p.itl_mean_ms is None else f"{p.itl_mean_ms:.3f}ms"
        ttft = "-" if p is None or p.ttft_p95_ms is None else f"{p.ttft_p95_ms:.1f}ms"
        deliv = "-" if p is None or p.delivered_fraction is None else f"{p.delivered_fraction:.4f}"
        # A descent round is marked, because it is not the same experiment as the one above it: its
        # KV floors were relaxed to nothing so the degree could be reached at all (impl §14.2), and
        # its latency is therefore measured against a pool the fleet file did not ask for.
        print(f"{r.tenant:<8} {('v' if r.descent else '') + str(r.tp):>4} {r.status:<18} "
              f"{itl:>10} {ttft:>10}  {deliv:>9}  {r.run_id or r.detail}")
        if p is not None and not p.usable:
            # Said out loud, because this row is neither a pass nor a miss and the table above
            # cannot show that difference: a suppressed latency reads as unmeasured everywhere
            # downstream, and treating it as a miss would widen the tenant's shard count.
            #
            # The reason comes from the row, not from a guess here. Guessing got it wrong on the
            # anchor's tp8 round: `delivery_passed` is None rather than False (the fraction had no
            # denominator), so this printed "the harness emitted no ITL" for a round the harness
            # had in fact stopped over delivery -- and `minimal_tp`, reading the row properly, said
            # something different about the same round three lines further down.
            why = p.unusable_reason[1]
            print(f"{'':<8} {'':>4} its latency is not usable -- {why} -- so this degree counts "
                  "as unmeasured rather than as a miss")
    if any(r.descent for r in rounds):
        print("  a 'v' before the degree marks a DESCENT round: a degree the computed envelope "
              "refused and the weights allow, reached with its KV floor relaxed to nothing, because "
              "nothing but a boot can settle it")

    print("\n=== minimal TP ===")
    minimal, unresolved = 0, []
    for t in fleet.tenants:
        res = resolutions.get(t.name)
        if res is None or res.status != RESOLVED:
            unresolved.append(t.name)
            detail = "no resolution" if res is None else res.status
            # Three fields per entry, not two: (degree, reason code, human text). The code is what
            # `minimal_tp` sorts CHECKED from UNCHECKED by. Unpacking two here crashed the report
            # AFTER every row had been written and the store was correct -- a whole profiling
            # round's conclusion reachable only by reading the store by hand.
            below = "" if res is None or not res.below else \
                "  (" + "; ".join(f"tp{k}: {why}" for k, _code, why in res.below) + ")"
            missing = "" if res is None or not res.missing else \
                f"  still unmeasured: tp{', tp'.join(str(k) for k in res.missing)}"
            print(f"  {t.name:<8} UNRESOLVED  {detail}{below}{missing}")
            continue
        certainty = "minimum" if res.certain else "UPPER BOUND"
        solo = "" if res.solo_itl_ms is None else f", {res.solo_itl_ms:.3f} ms ITL solo"
        target = "" if t.slo is None or t.slo.itl_mean_ms is None else \
            f" against a {t.slo.itl_mean_ms:.3f} ms target"
        print(f"  {t.name:<8} tp{res.tp:<4} {certainty:<12}{solo}{target}")
        if not res.certain:
            print(f"  {'':<8} tp{', tp'.join(str(k) for k in res.missing)} were never profiled, so "
                  "a narrower degree may also pass -- the baseline may still be inflated")
        short = None if res.evidence is None else res.evidence.pool_below_stated_floor
        if short is not None:
            # The one outcome of a descent round that nothing here can settle. The round is real --
            # it met the target and delivered at the offered rate -- but it did so on a pool this
            # fleet file says is too small, and the file's floor is a statement of demand rather than
            # a computed term. So the degree is reported as measured and the contradiction is handed
            # over whole, with the lever named: quietly keeping the wider degree would inflate the
            # baseline (impl §0), and quietly editing the demand is not this mode's to do.
            offered, stated = short
            print(f"  {'':<8} this rests on a DESCENT round driven on a KV pool of {offered:,} "
                  f"tokens, against the {stated:,} this fleet file states as {t.name}'s floor. It "
                  f"met its target and delivered at the offered rate on that pool -- which is the "
                  f"only evidence there is about whether the larger floor was needed -- but the "
                  f"file still states it, so `plan` and `search` will go on refusing tp{res.tp} "
                  f"while it does. Lowering it changes what this fleet is asking for, so it is "
                  f"yours to decide; leaving it means paying the wider degree.")
        minimal += res.tp

    if not unresolved:
        at_load = sum(t.tp for t in fleet.tenants)
        print(f"\n  BASELINE at minimal TP: {minimal} GPU(s) of {hardware.gpu_count} with no GPU "
              f"shared. That is what a search on this fleet has to beat (impl §0).")
        if at_load != minimal:
            print(f"  The degrees this run was loaded with sum to {at_load} GPU(s), so re-sharding "
                  f"alone accounts for {at_load - minimal} of them.")
        # Named separately from the resolution lines above, because it is the difference between "go
        # and search" and "the file and the measurement disagree, and the file wins until you change
        # it". A degree measured on a relaxed pool is settled as a degree and still unplaceable.
        blocked = [t.name for t in fleet.tenants
                   if (resolutions[t.name].evidence is not None
                       and resolutions[t.name].evidence.pool_below_stated_floor is not None)]
        if blocked:
            print(f"  But {', '.join(blocked)} reached that degree on a relaxed KV pool, as said "
                  "above, so `search` will NOT load this fleet as written: the solver refuses the "
                  "degree the measurement settled. The fleet file's floor is the thing to change, "
                  "and it is a change to what the fleet is asking for.")
        else:
            print("  Every degree above is now stated by measurement, so `search` will load this "
                  "fleet without needing this mode again.")
    else:
        _has = "has" if len(unresolved) == 1 else "have"
        _them = "it" if len(unresolved) == 1 else "them"
        print(f"\n  {', '.join(unresolved)} still {_has} no measured minimal TP, so there is no "
              "honest baseline yet. Re-run this mode (it re-measures nothing it already has), or "
              f"state tp: for {_them} in the fleet file and own the choice.")
    return {
        "rounds": [r.as_dict() for r in rounds],
        "resolutions": {n: r.as_dict() for n, r in resolutions.items()},
        "measurements_spent": out["spent"],
        "minimal_tp_baseline_gpus": None if unresolved else minimal,
        "unresolved": unresolved,
        "tenants": _tenant_definitions(fleet),
    }


def _profile(args, loaded, profiles, profile_mirror, lease, hub=None) -> int:
    """Measure each tenant alone, ascending from its memory floor, and file what it measured.

    The fleet is re-loaded against the leased node for the same reason `search` does it: the
    instance type stops being an assertion in a file and becomes a check against the machine, the
    environment the rows will be filed under is the one this node reported rather than the one a
    previous capture filed, and the GPU UUIDs the rounds will deploy against only exist once a node
    is held.
    """
    leased = fleetfile.load(args.fleet, namespace=args.namespace, lease=lease,
                           profiles=profiles, hub=hub, allow_unresolved_tp=True)
    fleet, key, hardware = leased["fleet"], leased["key"], leased["hardware"]

    def make_agent(solo, store, solo_key):
        """A `SearchAgent` for one solo round, on this run's lease. Same actuator as `search`.

        `push_key` runs here, before the round rather than after it, and the ordering is the
        whole point. The harness Job runs as uid 1000 and writes its record straight to EFS,
        while `observations/` is root-owned 755 -- so the key directory has to exist and carry
        the sticky group-write bit BEFORE the Job starts, and `push_key` is the only thing that
        creates it (as root, through `attach-pvc`) and chmods it. `search` gets this for free
        because `_stores` builds its mirror; a solo round's key is one per tenant per degree and
        was built here, bare.

        Skipped, every solo round of 2026-09-14 ended as a Failed Job with
        `PermissionError: /efs/colocation/observations/<key>`. Nothing was lost -- `write_record`
        dumps the record into the log for exactly this reason and the agent recovered every one
        of them -- but EFS is the store every other operator reads, and it got nothing.
        """
        mirror = EfsMirror(solo_key.hash, args.store, efs_root=args.efs_store,
                           namespace=args.namespace)
        mirror.push_key(solo_key.key_json())
        return SearchAgent(
            solo, store,
            deployer=HelmDeployer(args.chart, namespace=args.namespace,
                                  lease_release=lease.release, tenant_prefix=lease.tenant_prefix),
            harness=JobHarness(args.job_template, namespace=args.namespace,
                               image=args.harness_image, store_root=args.efs_store,
                               node_name=lease.node_name),
            lease=lease,
            run_prefix=args.mode, top_k=args.top_k, solver_seconds=args.solver_seconds,
            ready_timeout_s=args.ready_timeout, measure_timeout_s=args.measure_timeout,
            mirror=mirror,
            run_config_kwargs={"out_dir": args.efs_store, "exporter_url": lease.exporter_url})

    profiler = SoloProfiler(fleet, key, hardware, profiles, make_agent=make_agent,
                            store_root=args.store, profile_mirror=profile_mirror,
                            envelope_step_mib=AGENT_ENVELOPE_STEP_MIB,
                            solver_seconds=args.solver_seconds, reprofile=args.reprofile,
                            budget=args.budget)
    t0 = time.time()
    _print_profile_plan(profiler, hardware)
    out = profiler.run()
    summary = _profile_report(out, fleet, hardware)
    summary.update(fleet_file=str(args.fleet), mode=args.mode, profile_key_hash=profiles.key_hash,
                   validity_key_hash=key.hash, budget=args.budget, reprofile=args.reprofile,
                   elapsed_s=round(time.time() - t0, 1), run_token=lease.token,
                   gpu_node=lease.node_name, gpu_node_observed=lease.observed)
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"\nsummary written to {args.summary}")
    print(f"\nthe rows themselves are on EFS under {args.efs_store}/profiles/"
          f"{profiles.key_hash}/solo/, one file per (model, degree, shape, rate), and each names "
          "the run and the validity key of the measurement behind it.")
    # Non-zero when nothing was resolved: a mode whose whole purpose is to establish minimal TP and
    # which established none must not read as a success in a log.
    return 0 if not summary["unresolved"] else 1


def _packer(args, fleet, store, agent):
    """The driver, and the one line of output that says which one is about to spend the budget."""
    from .reason import DEFAULT_MODEL_ID
    model_id = args.reasoner_model or DEFAULT_MODEL_ID
    print(f"the reasoner is {model_id} over the {args.driver} driver, with six tools and "
          f"{args.budget} measurement(s) to spend. If it stops without a usable answer, the loop "
          "measures the solver's densest candidate instead and records that it did.")
    if args.driver == "sdk":
        return SdkPackerAgent(fleet, store, agent, model_id=model_id, budget=args.budget,
                              top_k=args.top_k, solver_seconds=args.solver_seconds,
                              max_turns=args.max_turns)
    from .reason import BedrockReasoner
    return ModelPackerAgent(fleet, store, agent, BedrockReasoner(model_id), budget=args.budget,
                            top_k=args.top_k, solver_seconds=args.solver_seconds,
                            max_turns=args.max_turns)


def _tenant_definitions(fleet) -> dict:
    """The tenants as resolved, written into the summary beside the results they produced.

    Without this the summary's `targets_met` booleans are unreadable six months later: they are
    judged at read time against whatever the fleet file says *then* (design §2, `_targets_met`), so
    a summary that records the verdict but not the threshold records an opinion with no premise. The
    records themselves are deliberately unstamped -- that is the right call for evidence, which is
    re-judged whenever a target moves -- and it is exactly why the *document* has to carry the
    targets that were in force when it was written.

    TP, the shapes and the rates are in the validity key as well, and the duplication is wanted: the
    key is a hash input and hard to read, while this is the fleet a person would have to reconstruct
    to reproduce the run.
    """
    out = {}
    for t in fleet.tenants:
        out[t.name] = {
            "model": t.model_id, "served_name": t.served_name,
            "tp": t.tp, "rate_hz": t.rate_hz,
            "isl_tokens": t.isl_tokens, "osl_tokens": t.osl_tokens,
            "max_model_len": t.max_model_len, "max_num_seqs": t.max_num_seqs,
            "max_num_batched_tokens": t.max_num_batched_tokens,
            "min_kv_tokens": t.min_kv_tokens,
            # None, not an empty dict: a tenant with no SLO stated none, and that is a fact about
            # the fleet rather than a gap in the summary.
            "slo": None if t.slo is None else {"itl_mean_ms": t.slo.itl_mean_ms,
                                               "ttft_p95_ms": t.slo.ttft_p95_ms},
        }
    return out


def _closest_row(result, fleet) -> dict | None:
    """The best placement the run found, whether or not it met anything (`closest_to_targets`).

    Recorded unconditionally, because "nothing met the targets" is not the whole of what a failed
    search learned. The one thing this must not do is read as a pass: `targets_met` travels with it
    and `is_certification` is flatly false, so a reader picking this out of the summary cannot take
    it for the verdict.
    """
    found = closest_to_targets(result.attempts, {t.name: t.slo for t in fleet.tenants})
    if found is None:
        return None
    attempt, worst_ratio = found
    met = _targets_met(attempt.record, fleet)
    return {
        "run_id": attempt.run_id, "label": attempt.label,
        "assignment": {n: list(g) for n, g in attempt.candidate.placement.assignment.items()},
        "gpus_used": attempt.candidate.gpus_used,
        **({"active_thread_percentage": dict(attempt.sm_caps)} if attempt.sm_caps else {}),
        "worst_margin_ratio": round(worst_ratio, 4),
        # No rolled-up boolean here on purpose. Whether a run passes is `result.verdict`, derived
        # once from the records; a second implementation of the same judgement is how a summary
        # comes to call a run met that retrieval shows as missed.
        "targets_met": met,
        "is_certification": False,
        "ranked_by": ("worst tenant's measured ITL over its target, then GPUs used; verdict "
                      "'measured' only. Not a certification -- see targets_met."),
    }


def _certified(result, fleet) -> tuple:
    """The densest placement this run may actually claim, and why the others do not count.

    Claimable means every tenant's stated target was met on a measured record: no miss, and no
    tenant left without a verdict. A None is not a pass -- either nobody stated a target for that
    tenant or the harness suppressed the latency it would be judged on (impl §6.3) -- and counting
    one as a pass is how a fleet gets certified on a measurement that does not exist. Fewest GPUs
    first, because that is the objective; the KV margin breaks ties, as in the solver.
    """
    claimable, rejected = [], []
    for a in result.attempts:
        if a.outcome != OUTCOME_MEASURED:
            continue
        met = _targets_met(a.record, fleet)
        if any(v is False for v in met.values()):
            rejected.append((a, f"missed {','.join(sorted(n for n, v in met.items() if v is False))}"))
        elif any(v is None for v in met.values()):
            rejected.append((a, "no verdict for "
                             f"{','.join(sorted(n for n, v in met.items() if v is None))}"))
        else:
            claimable.append(a)
    if not claimable:
        return None, rejected
    best = min(claimable, key=lambda a: (a.candidate.gpus_used,
                                         -(a.candidate.min_margin or 0.0)))
    return best, rejected


# How far the two KV pools may differ before the comparison stops being controlled. A pool sets the
# page count and the concurrency ceiling, so it is an input to latency, not a consequence of it: two
# runs on pools this far apart are two experiments, and the difference between them is not
# attributable to sharing. Small but not zero, because the pool is derived from a measured reserve in
# GiB and rounded to whole pages, so bit-identical is not a condition any real pair of runs meets.
KV_POOL_SAME_FRAC = 0.005


def _comparable(row, co_kv, co_src) -> tuple:
    """Were the solo reference and the co-resident run driven under the same conditions?

    Returns `(comparable, reason)`; `reason` is None when they were. Two facts are checked and they
    are not the same check. The pool is the physical input -- pages, concurrency ceiling -- and the
    reserve source is where it came from. Either one differing is enough to stop a bare ratio: the
    first because the experiments differ, the second because it says the difference is *structural*
    rather than incidental, and so will recur on the next fleet rather than being this run's bad luck.

    **Why this is not a rare case.** A reserve can only be measured on a solo round, so the first row
    for a (model, degree, environment) is measured on `default(tpN)` and writes the measured figure
    the fleet loader merges in. Every run after it -- co-resident runs included -- is driven on
    `measured@tpN`. The reference and the thing compared to it therefore differ *by construction* the
    first time any fleet is taken through the pipeline, which is why this is a code check and not a
    thing to notice: it is the normal state of a freshly profiled fleet, not an anomaly in one.

    Blank fields read as "not known", never as "the same". A row written before these were recorded
    cannot support the claim that the conditions matched, and the honest answer for it is the same as
    for a row that genuinely differs: say what is unknown and do not quote the ratio as if it were.
    """
    if row is None:
        return False, None                      # no reference at all; the caller says so already
    if not row.reserve_source or not row.kv_tokens:
        return False, ("the solo row does not record the KV pool or the reserve it was driven "
                       "under, so nothing here can say the two runs were the same experiment")
    if co_kv and row.kv_tokens:
        rel = abs(float(co_kv) - float(row.kv_tokens)) / float(row.kv_tokens)
        if rel > KV_POOL_SAME_FRAC:
            return False, (f"the solo row was driven with a {int(row.kv_tokens):,}-token KV pool "
                           f"and this run with {int(co_kv):,} ({rel * 100:.2f}% apart)"
                           + (f", because the reserve went from {row.reserve_source} to {co_src}"
                              if co_src and co_src != row.reserve_source else ""))
    if co_src and row.reserve_source and co_src != row.reserve_source:
        return False, (f"the solo row was driven on reserve {row.reserve_source} and this run on "
                       f"{co_src}; the pools came out within {KV_POOL_SAME_FRAC * 100:.1f}%, but "
                       "the two runs were still given different inputs")
    return True, None


def _vs_solo(attempt, fleet, profiles) -> dict:
    """Each tenant's co-resident latency beside its own solo latency at the same degree.

    This is the price §0 says co-location pays: stranded memory is converted into GPUs the fleet
    does not need, and SM time is what buys it. The solo row is that tenant alone with the whole of
    each GPU to itself -- the best case -- so the ratio is the whole of what sharing cost it, and it
    is the number that says whether the target was met with room to spare or by luck.

    Read from the profile store at the tenant's *deployed* degree, so it works whether that degree
    was resolved or hand-stated. A tenant with no row gets None rather than a comparison against
    another degree's measurement.

    **Two things can make the ratio not mean what it says, and both are checked here rather than
    left to a reader.** `_comparable` is one: the reference and the run may have been driven with
    different KV pools, which happens by construction the first time a fleet is profiled. The other
    is arithmetic on the ratio itself -- solo is defined as the best case, whole GPUs and no
    co-tenant, so a ratio below 1.00 is not a saving that sharing produced. It is the bound failing
    to bound, and something uncontrolled is in the comparison whether or not this function can name
    it. Both are recorded on the row so the summary JSON carries them, not only the printed line.
    """
    out = {}
    per = (attempt.record or {}).get("per_tenant") or {}
    prov = (attempt.record or {}).get("provenance") or {}
    co_kvs = prov.get("kv_tokens") or {}
    co_srcs = prov.get("mps_reserve_source") or {}
    for t in fleet.tenants:
        co = (per.get(t.name) or {}).get("itl_mean_ms")
        row = None if profiles is None else profiles.read(t.model_id, t.tp, t.isl_tokens,
                                                          t.osl_tokens, t.rate_hz, t.serving,
                                                          t.image_count, t.image_pixels)
        solo = None if row is None or not row.usable else row.itl_mean_ms
        ratio = None if not co or not solo else round(co / solo, 4)
        comparable, why = _comparable(row, co_kvs.get(t.name), co_srcs.get(t.name))
        out[t.name] = {
            "tp": t.tp, "coresident_itl_ms": co, "solo_itl_ms": solo,
            "ratio": ratio,
            "target_itl_ms": None if t.slo is None else t.slo.itl_mean_ms,
            "solo_run_id": None if row is None else row.run_id,
            # The conditions each side was driven under, so the summary can be re-judged without
            # the store and the record both to hand.
            "solo_kv_tokens": None if row is None else row.kv_tokens,
            "coresident_kv_tokens": co_kvs.get(t.name),
            "solo_reserve_source": None if row is None else (row.reserve_source or None),
            "coresident_reserve_source": co_srcs.get(t.name),
            "comparable": comparable,
            "incomparable_reason": why,
            # A bound that is exceeded is not a bound. Stated as its own field rather than left for a
            # reader to derive from the ratio, because the whole failure mode is a ratio being read
            # as a saving.
            "bound_violated": None if ratio is None else ratio < 1.0,
        }
    return out


def _relaxed_slos(fleet, factors) -> dict:
    """Each stated target multiplied by that tenant's own factor. For reading, never for judging.

    `factors` is a per-tenant mapping, defaulting to 1.0 for a tenant it does not name, because the
    relaxation a reading proposes is per tenant: the ones that met their targets are not being asked
    to concede anything, and relaxing them too would overstate how easily the baseline moves.
    """
    out = {}
    for t in fleet.tenants:
        if t.slo is None:
            out[t.name] = None
            continue
        factor = float(factors.get(t.name, 1.0)) if isinstance(factors, dict) else float(factors)
        out[t.name] = TenantSlo(
            itl_mean_ms=None if t.slo.itl_mean_ms is None else t.slo.itl_mean_ms * factor,
            ttft_p95_ms=None if t.slo.ttft_p95_ms is None else t.slo.ttft_p95_ms * factor,
            tolerance_pct=t.slo.tolerance_pct)
    return out


def _relaxed_baseline(fleet, base, profiles, factors) -> dict:
    """Whether the same relaxation would also narrow a tenant's minimal TP, and so move the baseline.

    **The trap this exists to close.** Minimal TP is the narrowest degree measured to *meet its
    target* (impl §0), so a looser target can resolve a tenant one degree narrower and `sum(tp)`
    falls with it. Scoring a relaxed placement against the strict baseline would then credit
    co-location with GPUs the relaxation alone freed -- and §0 is a comparison of like with like or
    it is nothing.

    Costs no GPU time when `profile` resolved the degrees: a tenant sitting at degree k was measured
    at every narrower degree down to its memory floor -- that is why it is at k and not below -- so
    those rows are already in the store and the relaxed answer is arithmetic over them. A tenant
    already at its memory floor cannot move at all: nothing narrower fits, whatever the target says,
    and that is the case in which this answer is certain rather than merely unrefuted.

    Which is why an absent row is reported as `unchecked` and not as "does not move". A degree with
    nothing measured below it leaves the question open, and answering it "no" would be the same trap
    wearing the opposite sign -- a saving scored against a baseline nobody checked.

    The walk is run twice, relaxed and strict, because only the *difference* is relaxation's doing. A
    tenant that would narrow at its *stated* target too is an upper-bound baseline
    (`fleetfile.upper_bound_tp`, reported above this on its own line) and crediting that to the
    relaxation would name the wrong cause for GPUs that were never co-location's to sell either
    way.

    `factors` is per tenant and not one number, because the reading being checked is per tenant: only
    the tenants that missed are conceding anything. Relaxing every ceiling by the worst tenant's
    factor -- which is what this did before tolerances were declared per tenant -- hands the other
    tenants slack nobody asked for, and slack is exactly what lets a degree narrow, so it reports the
    baseline as moving more readily than the reading would actually move it.
    """
    floors = {t.name: t.floor_tp for t in base.tenants}

    def walk(slos: dict) -> tuple:
        moved, unchecked = {}, {}
        for t in fleet.tenants:
            if slos[t.name] is None:
                continue
            # Only degrees the tenant's own memory floor permits. Below the floor the shard plus its
            # KV floor does not fit, so no target however loose can put it there, and listing those
            # degrees as unchecked would invent an open question out of closed arithmetic.
            floor = floors.get(t.name) or t.tp
            for tp in sorted(k for k in (1, 2, 4, 8) if floor <= k < t.tp):
                row = None if profiles is None else profiles.read(t.model_id, tp, t.isl_tokens,
                                                                 t.osl_tokens, t.rate_hz,
                                                                 t.serving, t.image_count,
                                                                 t.image_pixels)
                # `meets` returns None for a row that cannot answer -- suppressed, or missing the
                # metric a stated ceiling names -- and None is not "no". It goes with the absent rows.
                verdict = None if row is None else row.meets(slos[t.name])
                if verdict is None:
                    unchecked.setdefault(t.name, []).append(tp)
                    continue
                if verdict:
                    moved[t.name] = {"from_tp": t.tp, "to_tp": tp, "solo_itl_ms": row.itl_mean_ms,
                                     "ceiling_ms": getattr(slos[t.name], "itl_mean_ms", None),
                                     "solo_run_id": row.run_id}
                    break
        return moved, unchecked

    relaxed, _strict_slos = _relaxed_slos(fleet, factors), _relaxed_slos(fleet, {})
    moved, unchecked = walk(relaxed)
    strict, _ = walk(_strict_slos)

    def gpus(m: dict) -> int:
        return sum((m[t.name]["to_tp"] if t.name in m else t.tp) for t in fleet.tenants)

    # Relaxation's own doing: the tenants it narrows that the stated target does not, and the ones it
    # narrows further than the stated target does.
    because = {n: m for n, m in moved.items()
               if n not in strict or m["to_tp"] < strict[n]["to_tp"]}
    return {"tenants_that_would_narrow": moved, "relaxed_baseline_gpus": gpus(moved),
            "measured_minimal_baseline_gpus": gpus(strict),
            "narrowed_by_the_relaxation": because, "moves_the_baseline": bool(because),
            "unchecked": unchecked, "checked": not unchecked}


def _pct(fraction: float) -> str:
    """A relaxation as a percentage, with enough digits to stay a number rather than a rounding.

    `0.1f` turned a 0.34% concession into "0.3%" and a 0.04% one into "0.0%", which reads as nothing
    at all -- and a fraction of a percent on one tenant is precisely the case where the reader has a
    real decision to make, so it is the case that must not round to zero.
    """
    p = fraction * 100.0
    if p >= 10.0:
        return f"{p:.1f}%"
    return f"{p:.2f}%" if p >= 1.0 else f"{p:.3f}%"


def _tolerance_clause(w: dict) -> str:
    """`within chat's declared 1.0% tolerance`, or `beyond` it. Names whose number it is.

    "within the 5% cap" said nothing about who agreed to it. Whether the bound is the tenant's own
    statement or a default nobody wrote decides how much weight the RELAXED reading carries, so both
    the number and its provenance are on the line.
    """
    cap_pct, declared = w.get("tolerance_pct"), w.get("declared_tolerance")
    if cap_pct is None:
        return "no tolerance applies"
    side = "within" if w.get("within_tolerance") else "beyond"
    whose = "declared" if declared else "default"
    return f"{side} its {whose} {cap_pct:g}% tolerance"


def _relaxation_lines(attempt, fleet, base, profiles, pct, calibration) -> tuple:
    """What it would take, in requirements rather than in GPUs, for one rejected placement to pass.

    Printed for every placement denser than the certified one, always -- not only when the search
    freed nothing. A run that freed two GPUs and stopped 1.45 ms short of freeing a third has
    measured something the person who wrote that target needs to see, and "missed chat" does not
    say it. Nothing here is a claim: the placement stays rejected, the record stays unstamped, and
    the relaxed reading is labelled as one wherever it appears.

    **The cap is per tenant and comes out of the fleet file** (`slo.tolerance_pct`, defaulting to
    `spec.DEFAULT_TOLERANCE_PCT`), with `pct` able only to lower it. So `within_cap` is not one
    comparison any more: every tenant that missed has to be inside its *own* declared tolerance, and
    a fleet where one tenant said 0% cannot be offered a relaxed reading at all. That is the point --
    the tenant that will not negotiate says so in the file, in advance, rather than being negotiated
    for afterwards by whoever reads the report.
    """
    slos = {t.name: t.slo for t in fleet.tenants}
    req = required_relaxation(Observation.parse(attempt.record), slos)
    factor = req["factor"]
    out = dict(req)
    # Each tenant's own tolerance, recorded whether or not that tenant missed, so a reader can see
    # what the fleet declared without having to infer it from which lines got printed.
    out["tolerance"] = {
        name: {"pct": slo.tolerance(pct), "declared": slo.tolerance_is_declared,
               "cap_factor": round(slo.cap_factor(pct), 6)}
        for name, slo in sorted(slos.items()) if slo is not None}
    over_caps = []
    for name, w in out["per_tenant"].items():
        tol = out["tolerance"].get(name)
        cap = 1.0 if tol is None else tol["cap_factor"]
        w["tolerance_pct"] = None if tol is None else tol["pct"]
        w["declared_tolerance"] = bool(tol is not None and tol["declared"])
        w["cap_factor"] = cap
        w["within_tolerance"] = bool(w["factor"] <= cap)
        if not w["within_tolerance"]:
            over_caps.append(name)
    out["tenants_beyond_their_tolerance"] = sorted(over_caps)
    out["within_cap"] = bool(not req["blocked"] and 1.0 < factor and not over_caps)
    lines = []
    if req["blocked"]:
        # Two different reasons land in `blocked` and they are not the same news. A tenant nobody set
        # a target for has no ceiling to move; a tenant whose latency the delivery gate suppressed
        # has a ceiling but no measurement, and that one is the reason the delivery gate is not
        # relaxable (impl §6.3) -- relaxing it would change the workload in the direction that
        # flatters co-location.
        unstated = sorted(n for n in req["blocked"]
                          if next((t.slo for t in fleet.tenants if t.name == n), None) is None)
        unmeasured = [n for n in req["blocked"] if n not in set(unstated)]
        if unstated:
            lines.append(f"              no relaxation reaches it: {', '.join(unstated)} state no "
                         "target, so there is no ceiling to move -- state one and this placement "
                         "becomes judgeable")
        if unmeasured:
            lines.append(f"              no relaxation reaches it: {', '.join(unmeasured)} was not "
                         "measured on this run (the harness suppressed the latency), and moving a "
                         "ceiling cannot supply a measurement that was not taken")
        if factor <= 1.0:
            return out, lines
    if factor <= 1.0 and not req["blocked"]:
        # It met every stated target and was rejected for another reason, which `_certified` names.
        return out, lines
    worst = max(req["per_tenant"].items(), key=lambda kv: kv[1]["factor"], default=None)
    if worst is not None:
        name, w = worst
        others = sorted(n for n in req["per_tenant"] if n != name)
        # How many targets have to move, said in the same breath as how far. One tenant of five is a
        # negotiation; four of five is a different fleet, and the count is what distinguishes them.
        scope = ("one target and nothing else moves" if not others
                 else f"{len(req['per_tenant'])} targets, the others being "
                      + ", ".join(others))
        # "certify" only when relaxing the stated ceilings is the whole of what stands in the way.
        # With a blocked tenant it is not, and saying otherwise would promise something no relaxation
        # can deliver -- so the tolerance is not mentioned either, because it is not the binding
        # limit.
        cap_note = "" if req["blocked"] else ", " + _tolerance_clause(w)
        head = "it would still not certify, but it is" if req["blocked"] else "it would certify"
        lines.append(f"              {head} at {name} {w['metric']} {w['measured']:.3f} ms against "
                     f"its {w['ceiling']:.3f} ms ceiling -- a {_pct(factor - 1.0)} "
                     f"relaxation of {scope}{cap_note}")
        lines.append(f"              the miss is {w['over_by']:.3f} ms"
                     + _noise_clause(attempt, w["over_by"], calibration))
        # Every other tenant that is over, each against its own declared tolerance. Without this a
        # multi-tenant relaxation reads as one negotiation at the worst tenant's number, when in fact
        # each tenant's owner declared a different willingness and one of them may have said 0%.
        for other in sorted(n for n in req["per_tenant"] if n != name):
            o = req["per_tenant"][other]
            lines.append(f"                and {other} {o['metric']} {o['measured']:.3f} ms against "
                         f"{o['ceiling']:.3f} ms -- {_pct(o['factor'] - 1.0)}, "
                         + _tolerance_clause(o))
    rb = _relaxed_baseline(fleet, base, profiles,
                           {n: w["factor"] for n, w in req["per_tenant"].items()})
    out["relaxed_baseline"] = rb
    if rb["moves_the_baseline"]:
        who = ", ".join(f"{n} tp{m['from_tp']}->tp{m['to_tp']}"
                        for n, m in sorted(rb["narrowed_by_the_relaxation"].items()))
        at_most = "at most " if not rb["checked"] else ""
        lines.append(f"              and the same relaxation moves the BASELINE to {at_most}"
                     f"{rb['relaxed_baseline_gpus']} GPU(s) from {at_most}"
                     f"{rb['measured_minimal_baseline_gpus']} ({who} narrows once the target moves),"
                     " so the saving would have to be scored against that -- otherwise co-location "
                     "is credited with GPUs the relaxation alone freed")
    elif not rb["checked"]:
        # Not "no". Nobody measured the narrower degree, so the honest report is which measurement
        # is missing -- and it is a `profile` run, not another search.
        gaps = ", ".join(f"{n} at tp{'/tp'.join(str(k) for k in sorted(v))}"
                         for n, v in sorted(rb["unchecked"].items()))
        lines.append("              whether it also moves the baseline is unchecked: no solo row "
                     f"for {gaps}, so `profile` is what settles it, not another search")
    else:
        lines.append("              and it does not move the baseline: every tenant is already at "
                     "the narrowest degree its memory floor allows, or misses the relaxed target "
                     "there, so the comparison above still holds")
    return out, lines


def _noise_clause(attempt, over_by: float, calibration) -> str:
    """Whether a miss is larger than the resolution of the measurement that found it.

    Two ways it can be less than an answer, and both are said rather than smoothed over. Nothing
    repeated on this key means there is no resolution to compare against at all. And a half-width
    borrowed from another co-residency bucket is not this arrangement's own: `tightest_ms` is the
    right fallback for *sizing a window*, which is what `loop.py` uses it for, but here it is the
    flattering direction -- the tightest number measured anywhere makes any miss look like a larger
    multiple of noise than this arrangement's own spread might. So the borrowing is named.
    """
    if calibration is None:
        return (", and nothing has been repeated on this key, so there is no calibrated half-width "
                "to say whether that is resolvable -- one replicate settles it")
    degrees = attempt.candidate.coresidency.values()
    bucket = f"coresident={max(degrees)}" if degrees else None
    own = calibration.threshold_ms(bucket) if bucket else None
    half = own or calibration.tightest_ms
    if not half:
        return (", and the calibration holds no half-width for this arrangement, so whether that is "
                "resolvable is unmeasured")
    whose = (f"{bucket}'s own" if own is not None else
             f"tightest measured (nothing repeated at {bucket}, so this is borrowed from another "
             "arrangement and is the smallest spread in the store)")
    return (f", which is {over_by / half:.0f}x the calibrated {half:.4f} ms half-width, {whose}"
            + (" -- so it is a real miss, and relaxing the target is a decision about the "
               "requirement rather than about noise" if over_by > half else
               " -- so it is INSIDE the noise, and a replicate is the honest next step, not a "
               "looser target"))


def _relaxed_offer(rejected, relax: dict, base, strict_gpus: int):
    """The fewest-GPU rejected placement every tenant's *own* declared tolerance would admit.

    Returned as `(attempt, block, scored_against_gpus)`, or None. Three things disqualify a
    placement, and each is a different sentence in the report rather than a silent skip:

    - it needs more than some tenant said it would accept, or a tenant it cannot judge at all
      (`within_cap`, which `_relaxation_lines` computes per tenant);
    - the same concession would narrow the baseline, and it does not beat the *narrowed* baseline --
      then the GPUs belong to the relaxation, not to co-location;
    - it is no denser than what STRICT already certified, so there is nothing on offer.

    Ties go to the smaller concession, which is why a deliberate replicate of the same placement can
    displace the original: the one that asks for less is the one to put in front of a reader.
    """
    offers = []
    for a, _why in rejected:
        block = relax.get(a.run_id) or {}
        if not block.get("within_cap"):
            continue
        rb = block.get("relaxed_baseline") or {}
        against = base.resolved_gpus
        if rb.get("moves_the_baseline") and rb.get("relaxed_baseline_gpus") is not None:
            against = rb["relaxed_baseline_gpus"]
        if a.candidate.gpus_used >= min(strict_gpus, against):
            continue
        offers.append((a, block, against))
    if not offers:
        return None
    return min(offers, key=lambda o: (o[0].candidate.gpus_used, o[1].get("factor", 1.0)))


def _beyond_tolerance_lines(rejected, relax, offered_run_id) -> list:
    """The placements a tolerance refused, each with the concession it would have needed.

    Kept to one line each and kept apart from the offer, because 16.7% and 0.34% must never format
    alike -- the whole reason the old report was hard to act on is that a near miss and a different
    fleet appeared under the same heading.
    """
    lines = []
    for a, _why in sorted(rejected, key=lambda p: p[0].candidate.gpus_used):
        block = relax.get(a.run_id) or {}
        if a.run_id == offered_run_id or block.get("within_cap"):
            continue
        if not lines:
            lines.append("            beyond what a tenant's own tolerance allows, so not offered:")
        over = block.get("tenants_beyond_their_tolerance") or []
        if over:
            who = ", ".join(f"{n} needs {_pct(block['per_tenant'][n]['factor'] - 1.0)} against a "
                            f"{block['per_tenant'][n]['tolerance_pct']:g}% tolerance" for n in over)
        elif block.get("blocked"):
            who = f"no tolerance reaches {', '.join(block['blocked'])}"
        else:
            continue
        lines.append(f"              {a.run_id} on {a.candidate.gpus_used} GPU(s)"
                     f"{_caps_text(a.sm_caps)}: {who}")
    return lines


def _print_relaxed(offer, rejected, relax: dict, base, strict_gpus: int, out: dict) -> None:
    """The `RELAXED` slot: what a concession the fleet file already permits would buy, if anything.

    Always printed, including when there is nothing on offer, and always labelled `NOT CERTIFIED`.
    The slot exists so that a run which certified nothing still reports its most decision-relevant
    number in a place a reader will find it, and so that a run which *did* certify still says what
    one more percent would have bought.
    """
    if offer is None:
        denser = [a for a, _ in rejected if a.candidate.gpus_used < strict_gpus]
        if not rejected:
            why = ("nothing was rejected, so there is no placement a concession could add to what "
                   "STRICT already reports")
        elif not denser:
            why = ("no rejected placement is denser than what STRICT reports, so a concession would "
                   "buy no GPU")
        else:
            why = ("every denser placement needs more than some tenant's own tolerance allows, or "
                   "names a tenant no tolerance can reach")
        print(f"\n  RELAXED   nothing on offer: {why}.")
        for line in _beyond_tolerance_lines(rejected, relax, None):
            print(line)
        out["relaxed"]["why_not"] = why
        return

    a, block, against = offer
    freed = against - a.candidate.gpus_used
    pctg = "" if not against else f" ({freed / against:.0%} of the baseline)"
    per, tol = block.get("per_tenant", {}), block.get("tolerance", {})
    print(f"\n  RELAXED   {a.candidate.gpus_used} GPU(s), {freed} freed{pctg} -- within every "
          "tenant's own tolerance. NOT CERTIFIED.")
    print(f"            {a.run_id} ({_compact(a.candidate.placement.assignment)}"
          f"{_caps_text(a.sm_caps)})")
    verb = "concedes" if len(per) == 1 else "concede"
    print(f"            the concession: {len(per)} of {len(tol)} tenants {verb}; the rest met "
          "their ceilings as written")
    for name, w in sorted(per.items()):
        print(f"              {name} {w['metric']} ceiling {w['ceiling']:.3f} -> "
              f"{w['measured']:.3f} ms (+{_pct(w['factor'] - 1.0)}), "
              + _tolerance_clause(w))
    rb = block.get("relaxed_baseline") or {}
    if rb.get("moves_the_baseline"):
        print(f"            the same concession narrows the BASELINE to {against} GPU(s), so the "
              f"{freed} above is scored against that and not against "
              f"{base.resolved_gpus} -- otherwise co-location is credited with GPUs the concession "
              "alone freed")
    elif not rb.get("checked", True):
        print("            whether the same concession also narrows the BASELINE is UNCHECKED, so "
              f"the {freed} above may include GPUs the concession alone would free. `profile` "
              "settles it, not another search")
    else:
        print(f"            the BASELINE does not move under the same concession (still "
              f"{base.resolved_gpus}), so the {freed} is co-location's and not the concession's")
    print("            NOT CERTIFIED: this is what a concession the fleet file permits would buy. "
          "No record is stamped with it and STRICT above is unaffected.")
    for line in _beyond_tolerance_lines(rejected, relax, a.run_id):
        print(line)
    out["relaxed"] = {
        "offered": True, "gpus": a.candidate.gpus_used, "freed": freed, "run_id": a.run_id,
        "certified": False, "scored_against_baseline_gpus": against,
        "baseline_moves_under_the_concession": bool(rb.get("moves_the_baseline")),
        "baseline_movement_checked": bool(rb.get("checked", False)),
        "concession": {n: {k: w[k] for k in ("metric", "ceiling", "measured", "factor",
                                            "tolerance_pct", "declared_tolerance")}
                       for n, w in sorted(per.items())},
        "assignment": {n: list(g) for n, g in a.candidate.placement.assignment.items()}}


def _print_outcome(result, fleet, base, profiles, pct=None,
                   calibration=None, upper_bound=()) -> dict:
    """`BASELINE`, then `STRICT`, then `RELAXED` -- all three in GPUs, all three always printed.

    Printed last and stated in one unit, because everything above it is a measurement and this is the
    conclusion: how many GPUs the fleet occupies with sharing, against how many it occupies without.

    **Why three lines and not two.** This used to print `ACHIEVED` and `FREED`. Both were poor.
    `ACHIEVED` named a state rather than a quantity -- it was in GPUs, like `BASELINE`, but the label
    did not say so, and it silently carried a predicate ("every stated target met, with a verdict for
    every tenant") that was the whole of its content. `FREED` was `BASELINE - ACHIEVED`: arithmetic on
    two numbers directly above it, given a slot as prominent as the measurements it derived from, and
    null in exactly the runs that mattered. Worse, the pair was binary where the result is not. A run
    can perfectly well find nothing at the targets as written *and* four GPUs at a third of a percent
    on one tenant, and under the old vocabulary the first got a headline while the second was an
    indented sub-bullet under a rejected candidate. A reader had to know to go looking for the number
    that was actually decision-relevant. So:

    - `STRICT` is the fewest-GPU placement that met every ceiling exactly as written. It is the only
      one of the two that is a certification.
    - `RELAXED` is the fewest-GPU placement that every tenant's *declared* tolerance would admit
      (`slo.tolerance_pct`), labelled as not certified every time it is printed.

    Both are printed even when empty, so the reader compares two numbers instead of accepting a
    verdict, and `freed` is a field of each rather than a line of its own -- it belongs beside the
    number it is subtracted from. A relaxed reading also states the concession in the units the
    operator sets it in (milliseconds of ceiling), never as a bare factor, and carries the
    baseline-movement guard on screen rather than only in the JSON.

    `upper_bound` is `fleetfile.upper_bound_tp`: the tenants whose degree met the target solo while a
    narrower degree went unmeasured. They are the only remaining way the BASELINE line can be too
    high -- the fleet file cannot state a degree, so a degree above the memory floor otherwise means
    the floor degree was measured and missed.
    """
    best, rejected = _certified(result, fleet)
    print("\n=== BASELINE vs STRICT vs RELAXED (impl §0) ===")
    print(f"  BASELINE  {base.resolved_gpus} GPU(s) of {base.gpu_count}, no GPU shared, at "
          f"tp{'+tp'.join(str(t.resolved_tp) for t in base.tenants)}")
    if upper_bound:
        print(f"            and possibly as low as {base.floor_gpus}: "
              f"{', '.join(sorted(upper_bound))} met their target solo at the degree above, but a "
              "narrower degree was never profiled. Until it is, the saving below may include GPUs "
              "re-sharding alone would free.")
    out = {"baseline_gpus": base.resolved_gpus, "baseline_at_memory_floor_gpus": base.floor_gpus,
           "baseline_is_possibly_inflated": bool(upper_bound),
           "baseline_upper_bound_tenants": sorted(upper_bound),
           "baseline_tenants_above_memory_floor": list(base.above_floor),
           "strict": {"gpus": None, "freed": None, "run_id": None, "certified": False},
           "relaxed": {"offered": False, "gpus": None, "freed": None, "run_id": None,
                       "certified": False},
           "not_claimable": [{"run_id": a.run_id, "gpus_used": a.candidate.gpus_used,
                              "why": why} for a, why in rejected]}
    relax = {}
    # Computed for every rejected placement before anything is printed, because RELAXED is chosen
    # from these blocks and the choice cannot be made while walking them.
    for a, _why in rejected:
        relax[a.run_id], _ = _relaxation_lines(a, fleet, base, profiles, pct, calibration)
    out["required_relaxation"] = relax
    for entry in out["not_claimable"]:
        entry["required_relaxation"] = relax.get(entry["run_id"])

    def _relax(a, indent: str) -> None:
        """Print the smallest relaxation that would let one rejected placement pass."""
        _block, lines = _relaxation_lines(a, fleet, base, profiles, pct, calibration)
        for line in lines:
            print(indent + line.lstrip())

    # --- STRICT -------------------------------------------------------------------------------
    if best is None:
        print("\n  STRICT    no placement. All "
              f"{base.resolved_gpus} GPU(s) stand, 0 freed. Every ceiling exactly as written; no "
              "measured placement met all of them with a verdict for every tenant.")
        for a, why in sorted(rejected, key=lambda p: p[0].candidate.gpus_used):
            print(f"            {a.run_id} on {a.candidate.gpus_used} GPU(s)"
                  f"{_caps_text(a.sm_caps)}: {why}")
            _relax(a, "              ")
    else:
        freed = base.resolved_gpus - best.candidate.gpus_used
        pctg = "" if not base.resolved_gpus else f" ({freed / base.resolved_gpus:.0%} of the baseline)"
        print(f"\n  STRICT    {best.candidate.gpus_used} GPU(s), {freed} freed{pctg} -- every "
              f"ceiling exactly as written. CERTIFIED.")
        print(f"            {best.run_id} ({_compact(best.candidate.placement.assignment)}"
              f"{_caps_text(best.sm_caps)})")
        if freed <= 0:
            print("            The saving is nothing. Co-location had nothing to sell this fleet: "
                  "it strands no memory the tenants' targets leave the SM headroom to reclaim "
                  "(impl §0).")
        # A denser placement that did not hold is the most informative thing a search finds, and it
        # is the one a reader would otherwise have to reconstruct from the table above: it says the
        # saving stopped here because the next step was measured and refused, not because nothing
        # tried it.
        for a, why in sorted((p for p in rejected
                              if p[0].candidate.gpus_used < best.candidate.gpus_used),
                             key=lambda p: p[0].candidate.gpus_used):
            print(f"            denser but not certified: {a.run_id} on "
                  f"{a.candidate.gpus_used} GPU(s){_caps_text(a.sm_caps)}, {why}")
            _relax(a, "              ")
        out["strict"] = {
            "gpus": best.candidate.gpus_used, "freed": freed, "run_id": best.run_id,
            "certified": True,
            "assignment": {n: list(g) for n, g in best.candidate.placement.assignment.items()}}

    # --- RELAXED ------------------------------------------------------------------------------
    strict_gpus = base.resolved_gpus if best is None else best.candidate.gpus_used
    offer = _relaxed_offer(rejected, relax, base, strict_gpus)
    _print_relaxed(offer, rejected, relax, base, strict_gpus, out)

    reported = best if best is not None else (None if offer is None else offer[0])
    if reported is None:
        print("\n  No placement to compare against solo: nothing was certified and nothing is on "
              "offer, so there is no arrangement whose cost of sharing there would be to state.")
        return out

    vs = _vs_solo(reported, fleet, profiles)
    print("\n  vs solo -- what sharing cost each tenant, against the same tenant alone on whole "
          f"GPUs ({reported.run_id}):")
    for name, row in vs.items():
        co = "-" if row["coresident_itl_ms"] is None else f"{row['coresident_itl_ms']:.3f} ms"
        if row["solo_itl_ms"] is None:
            print(f"    {name:<8} tp{row['tp']:<2} {co:>10} co-resident, no solo row to compare "
                  "against -- run `profile` and this line becomes the cost of sharing")
            continue
        target = ("" if row["target_itl_ms"] is None
                  else f", {row['coresident_itl_ms'] / row['target_itl_ms']:.2f}x of its "
                       f"{row['target_itl_ms']:.3f} ms target")
        # The ratio is quoted only when the two runs were the same experiment. Where they were not,
        # the two latencies still print -- they are both measurements and both true -- but the number
        # that would be read as "what sharing cost" does not, because that is not what it would be.
        if row["comparable"]:
            ratio = f", {row['ratio']:.2f}x"
        else:
            ratio = ", ratio withheld"
        print(f"    {name:<8} tp{row['tp']:<2} {co:>10} co-resident against "
              f"{row['solo_itl_ms']:.3f} ms solo{ratio}{target}")
        if not row["comparable"] and row["incomparable_reason"]:
            print(f"             not a controlled comparison: {row['incomparable_reason']}. Both "
                  "numbers stand; their difference is not the cost of sharing, because the two runs "
                  "were given different inputs. One `profile --reprofile` round at this degree "
                  "re-measures the solo reference under the conditions the fleet now uses, and the "
                  "ratio becomes quotable.")
        if row["bound_violated"]:
            print(f"             and the co-resident run came out FASTER than solo "
                  f"({(1.0 - row['ratio']) * 100:.1f}%), which cannot be a saving sharing "
                  "produced: solo is the same tenant alone on whole GPUs, so it is the best case by "
                  "construction and the ratio's floor is 1.00. Something uncontrolled is in the "
                  "comparison; treat the solo figure as not being a bound on this run.")
    out["vs_solo"] = vs
    return out


def _report(result, fleet, base=None, profiles=None, pct=None,
            calibration=None, upper_bound=()) -> dict:
    """What each iteration measured, beside what was required of it.

    The column that used to sit next to the measured ITL was the prediction, and the report was
    largely a scorecard for the interference model. There is no prediction to score now. What takes
    its place is the tightest KV margin the placement was solved to -- a fact about the
    configuration, not a forecast -- and whether each tenant's stated target was met, which is
    computed here at read time rather than stamped into the record (design §2).
    """
    rows = []
    print("\n=== results ===")
    print(f"{'run':<30} {'label':<14} {'outcome':<22} {'worst ITL':>10}  {'margin':>7}  targets")
    for a in result.attempts:
        measured = _worst_itl_ms(a.record) if a.record else float("inf")
        shown = "-" if measured == float("inf") else f"{measured:.3f}ms"
        margin = a.candidate.min_margin
        met = _targets_met(a.record, fleet)
        print(f"{a.run_id:<30} {a.label:<14} {a.outcome:<22} {shown:>10}  "
              f"{'-' if margin is None else f'{margin:.2f}x':>7}  {_targets_text(met)}"
              f"{_caps_text(a.sm_caps)}")
        rows.append({
            "run_id": a.run_id, "label": a.label, "outcome": a.outcome,
            "assignment": {n: list(g) for n, g in a.candidate.placement.assignment.items()},
            "coresidency": dict(a.candidate.coresidency),
            "envelope_gib": dict(a.candidate.placement.envelope_gib),
            # Only when capped (impl §16), so an uncapped row reads as it did before caps existed.
            **({"active_thread_percentage": dict(a.sm_caps)} if a.sm_caps else {}),
            "gpus_used": a.candidate.gpus_used,
            "kv_floor_tokens": dict(a.candidate.floor_tokens),
            "kv_tokens": dict(a.candidate.placement.kv_tokens),
            "tightest_kv_margin": margin,
            "measured_worst_itl_ms": None if measured == float("inf") else measured,
            "per_tenant_itl_ms": {n: (a.record or {}).get("per_tenant", {})
                                  .get(n, {}).get("itl_mean_ms")
                                  for n in a.candidate.placement.assignment},
            "targets_met": met,
            "verdict": (a.record or {}).get("verdict"),
        })
    print(f"\nstopped: {result.stopped_because}")

    # A summary over whatever the run happened to measure, not a designed comparison. Grouping by
    # co-residency degree is the coarsest cut that separates a packed placement from a spread one,
    # and it is worth printing even when nothing arranged for the groups to be comparable -- but
    # with unequal group sizes and no controlled contrast, a ratio here is a description of these
    # runs and not a measured effect.
    by_degree: dict = {}
    for a in result.attempts:
        if a.outcome != OUTCOME_MEASURED:
            continue
        degree = max(a.candidate.coresidency.values())
        by_degree.setdefault(degree, []).append(_worst_itl_ms(a.record))
    if len(by_degree) > 1:
        print("\nworst-tenant ITL by co-residency degree:")
        least = None
        for degree in sorted(by_degree):
            vals = [v for v in by_degree[degree] if v != float("inf")]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            least = mean if least is None else least
            print(f"  coresident={degree}: {mean:.3f} ms  (n={len(vals)}, "
                  f"{mean / least:.2f}x the least-contended)")
    out = {"attempts": rows, "stopped_because": result.stopped_because,
           "by_coresidency_degree": {str(k): v for k, v in by_degree.items()},
           "tenants": _tenant_definitions(fleet),
           "closest_to_targets": _closest_row(result, fleet)}

    fell_back = [a.run_id for a in result.attempts
                 if (a.record or {}).get("provenance", {}).get("policy_kind") == PACKER_FALLBACK]
    if fell_back:
        # Surfaced at the top level, not left in the records. A run where the reasoner silently
        # stopped being consulted is a run in which nothing chose those placements at all -- the
        # solver's ranking was simply taken -- and that must not be something a reader has to go
        # looking for.
        print(f"\nthe reasoner was unusable on {len(fell_back)} of {len(result.attempts)} "
              f"iteration(s), which took the solver's densest candidate instead: "
              f"{', '.join(fell_back)}")
        out["reasoner_fallback_runs"] = fell_back

    # What it believed, beside what happened (impl §11.2). A placement that missed as expected and
    # one that missed unexpectedly are different findings, and printing only the miss loses that.
    expectations = expectation_report(result, {t.name: t.slo for t in fleet.tenants})
    if expectations:
        print("\n=== what the reasoner expected ===")
        for line in expectations:
            print(line)

    closest = out["closest_to_targets"]
    if closest:
        print(f"\nclosest to targets: {closest['run_id']} ({closest['label']}), worst tenant at "
              f"{closest['worst_margin_ratio']:.3f}x its target on {closest['gpus_used']} GPUs"
              f"{_caps_text(closest.get('active_thread_percentage'))}, "
              f"{_targets_text(closest['targets_met'])}")
        print("  ranked, not certified. Reported whether or not it met anything, so a search that "
              "failed still says which direction was closest.")

    print(f"\nverdict: {result.verdict}")
    if result.submitted:
        print(f"  the reasoner submitted {result.submitted['verdict']}: "
              f"{result.submitted['rationale']}")
        if result.submitted["verdict"] != result.verdict:
            # Kept as a difference rather than resolved into one number. `met` is derived from a
            # record here and cannot be claimed; anything else is the reasoner's reading, and where
            # the two disagree that disagreement is the finding.
            print(f"  the loop derived {result.verdict} instead, from the records themselves.")
        if result.submitted["choice"]:
            print(f"  it recommends {_compact(result.submitted['choice'])}")
    else:
        print("  the reasoner never submitted a conclusion; this verdict is the loop's.")
    out.update(verdict=result.verdict, submitted=result.submitted, turns=result.turns)

    # Last, because it is the conclusion and everything above it is the evidence. `base` is passed
    # in rather than recomputed so the number printed here is the same one `plan` printed before any
    # GPU was touched -- a baseline that moved between the two would make the saving unfalsifiable.
    if base is not None:
        out.update(_print_outcome(result, fleet, base, profiles, pct, calibration, upper_bound))
    return out


def _targets_met(record, fleet) -> dict:
    """tenant -> True, False, or None, judged at read time against the targets in force now.

    The judgement itself is `retrieval.targets_met`, deliberately not re-implemented here. It is
    the same question retrieval answers for every prior run it returns, and two implementations of
    it would drift -- the report would call a run met that retrieval showed the reasoner as missed.

    None is not a failure and not a pass: it means this run cannot answer for that tenant, either
    because no target was stated or because the harness suppressed the latency it would be judged
    on (impl §6.3). Collapsing None into False would report a fleet as missing targets nobody set;
    collapsing it into True would certify a fleet on a measurement that does not exist. An attempt
    that never produced a record is that case for every tenant.
    """
    targets = {t.name: t.slo for t in fleet.tenants}
    if not record:
        return {name: None for name in targets}
    return targets_met(Observation.parse(record), targets)


def _targets_text(met: dict) -> str:
    passed = sorted(n for n, v in met.items() if v is True)
    failed = sorted(n for n, v in met.items() if v is False)
    unknown = sorted(n for n, v in met.items() if v is None)
    parts = []
    if passed:
        parts.append(f"met {','.join(passed)}")
    if failed:
        parts.append(f"MISSED {','.join(failed)}")
    if unknown:
        parts.append(f"no verdict for {','.join(unknown)}")
    return "; ".join(parts) or "no targets stated"


def sdk_available() -> bool:
    """Whether `claude_agent_sdk` could be imported, WITHOUT importing it.

    `find_spec` locates the module and does not execute it, which matters twice: the self-tests
    assert this package pulls in no SDK, and a run that only wants `--driver converse` must not pay
    for one it will never call.
    """
    import importlib.util

    return importlib.util.find_spec("claude_agent_sdk") is not None


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    # `--local-only` is for the two modes that read the store and measure nothing. The others are
    # refused it: `profile` and `search` lease a node, and `calibrate` writes a noise floor every
    # later run is judged against, so all three must see the whole history on EFS, not a local part.
    if args.local_only:
        if args.mode not in ("plan", "export"):
            raise SystemExit(f"--local-only is for `plan` and `export`; `{args.mode}` needs the "
                             "EFS store, so run it without the flag.")
        log.warning("--local-only: the cluster is not contacted and the EFS store is neither read "
                    "nor written. This run sees %s and nothing else.", args.store)

    # The same rule applied to the driver, and it is the cheapest check in this file. Both SDK
    # drivers import `claude_agent_sdk` inside the method that needs it, deliberately -- that is what
    # keeps this package importable, and the whole self-test suite runnable, with nothing installed.
    # The cost of that is where the ImportError lands: inside `run()`, which is called with a node
    # already leased and, in `search`, tenants already deployed. So it is asked here, by name and
    # without importing anything, before a cluster is touched.
    if args.mode == "search" and args.driver == "sdk" and not sdk_available():
        raise SystemExit(
            f"{args.mode} --driver sdk needs the Claude Agent SDK and it is not importable: "
            "pip install claude-agent-sdk (it brings its own copy of the CLI). Refused here rather "
            "than inside the run, because the driver is not reached until a node is leased. "
            "--driver converse needs only boto3 and reaches the same tools under the same refusals, "
            "but its JSON-in-text parsing is not what ships.")

    # Same rule as the fleet file below, and cheaper still: a run with no harness image cannot take
    # a measurement, so it is refused before the cluster is touched at all. `plan` and `calibrate`
    # never run the harness and do not need one.
    if args.mode in ("profile", "search") and not args.harness_image:
        raise SystemExit(
            f"{args.mode} needs a harness image and none is set. Export {HARNESS_IMAGE_ENV} or pass "
            "--harness-image. It has no default: build it from containers/colocation-harness/ and "
            "push it to a registry this cluster can pull from (USER-GUIDE.md, step 3).")

    # The top of `fleets/` ships EMPTY on purpose (impl §0), so "no such file" is the first thing a
    # new reader hits -- the guide's commands name a file they have to write. A traceback here reads
    # as a broken tool rather than as the one step they have not done yet.
    if not Path(args.fleet).is_file():
        raise SystemExit(
            f"no fleet file at {args.fleet}. The top of fleets/ is where yours goes and ships empty; "
            "the fleets shipped here are under fleets/examples/<instance type>/, and they describe "
            "workloads that exist rather than templates to copy. Write yours following USER-GUIDE.md "
            "step 7, which carries the annotated skeleton, and read USER-GUIDE.md step 8 first: it is "
            "what decides whether the search has anything to find.")

    # The hardware store first, pulled before anything else because everything else is resolved
    # against it (`_hardware`): nothing about a card is in the source any more, so a desktop that has
    # never captured has to be told what a colleague's capture measured. It is also what names the
    # profile directory below -- the store's address is what a node of this type reported about
    # itself, not a string from the fleet file -- which is why this line moved above that one.
    hardware_index, hardware_mirror = _hardware(args)
    # Then the profile store, before the fleet is loaded, in every mode, because every tenant's TP
    # degree is resolved from these rows and so they have to be in hand before the tenants can be
    # built -- `_profiles` keys the store off the file's environment for exactly that reason. This
    # used to be conditional on some tenant omitting `tp:`; with `tp:` gone from the file the
    # condition is always true, and a fleet that opens no store is one that can only be loaded as
    # memory floors (`plan`, `profile`).
    try:
        profiles, profile_mirror = _profiles(args, hardware_index)
    except UnknownHardware as exc:
        # No address, because nobody has ever held a node of this type: there is no directory to read
        # rows from and none to write them to. For the two modes that lease a node, this run IS the
        # run that fixes it -- so it holds one, files what the machine reports, gives it back and
        # stops. See `_bootstrap_environment`.
        if args.mode in ("profile", "search"):
            return _bootstrap_environment(args, hardware_index, hardware_mirror, exc)
        # `plan` and `calibrate` cannot lease, so the refusal is the answer. Reported rather than
        # raised for the same reason the loader's own errors are, one step further down: pointing
        # this at hardware nobody has held yet is the first thing a new reader does, and the message
        # already names the run that fixes it.
        raise SystemExit(f"{args.fleet}: {exc}") from exc
    except (fleetfile.FleetFileError, Disagreement) as exc:
        # The fleet file is opened here, before `load` below, because the store's address comes out of
        # it -- so its own refusals can arrive from this line instead. A `validity_key:` block is the
        # one that will, and its remedy is to delete four lines, which a stack trace buries.
        raise SystemExit(f"{args.fleet}: {exc}") from exc
    # And the checkpoint manifests, for the same reason one step further out: loading the fleet reads
    # every model's `config.json` and safetensors headers through this, and a `models:` entry that
    # states only an id and a commit cannot be priced without one (`_hub`).
    hub = _hub(args)

    # Loaded offline first, whatever the mode. Every fleet-file error -- a missing SLO, a mixed
    # image, an unknown instance type -- is found here, before anything can buy a node. A run that
    # captures a g6e and then fails to parse its own fleet file has spent money to learn nothing.
    # `plan` joins `profile` in tolerating an unresolved degree, and for the same reason: it is the
    # mode you run to find out what to do next, and refusing to load a fleet whose degrees are not
    # measured yet would refuse to answer the question it exists for. It reports them as floors and
    # says so; what it must not do, and does not, is print `sum(floor)` as the baseline.
    #
    # A fleet-file error becomes a refusal rather than a traceback. With `fleets/` shipped empty,
    # the first thing a new reader sees is a fleet they just wrote -- and every message raised in
    # here says what to change, which a stack trace buries.
    try:
        loaded = fleetfile.load(args.fleet, index=hardware_index,
                                namespace=args.namespace, profiles=profiles, hub=hub,
                                gpu_count=args.gpu_count, gpu_product=args.gpu_product,
                                allow_unresolved_tp=args.mode in ("plan", "profile"))
    except (fleetfile.FleetFileError, UnknownHardware, Disagreement) as exc:
        # `UnknownHardware` and `Disagreement` are here for the same reason and are not fleet-file
        # errors: the file is fine and the STORE cannot answer it -- nobody has leased this instance
        # type, or two environments filed different sizes for its card. Both messages already name
        # the remedy, and both would otherwise arrive as a stack trace from three frames down.
        raise SystemExit(f"{args.fleet}: {exc}")
    fleet, key, hardware = loaded["fleet"], loaded["key"], loaded["hardware"]
    if loaded["max_nodes"] > 1 and args.mode == "search":
        # Before the fleet's own store is opened: the fleet over several nodes is never measured as
        # one, so its key would be a store folder for an experiment nobody runs. Each group is.
        return _search_splits(args, argv, loaded)
    if loaded["max_nodes"] > 1 and args.mode == "export":
        raise SystemExit(
            f"{args.fleet}: hardware.max_nodes is {loaded['max_nodes']}, so there is nothing for "
            "`export` to read: a split is exported by `search` itself, once every group of it is "
            "met, as <store>/groups/<fleet>/export.<stamp>.yaml (schema_version 2). Each group's "
            "own search also exports that group.")
    if args.mode == "profile":
        # No fleet-level observation store, and deliberately none: this mode never measures the
        # fleet. Every round is one tenant at one degree, which is a different validity key each
        # time (`profiler.solo_key`), and each round opens and pushes its own. Writing this fleet's
        # key here would create a store directory for an experiment nobody ran.
        store = mirror = None
    else:
        store, mirror = _stores(args, key)
    # The workload and the shard counts, which are the two parts a person changes by editing the
    # fleet file. The environment half of the key is stated elsewhere in the same run's output
    # (`hardware.provenance`, and the capture's own lines), and printing a key whose hash moved
    # without showing the field that moved it is how a re-key looks like a bug.
    # `serving` is here for that reason and not for symmetry: adding it moved every fleet's hash,
    # and the first run after it landed printed a moved hash with nothing on the line to explain it.
    _canon = key.canonical()
    log.info("validity key %s: %s", key.hash,
             json.dumps({**_canon["workload"], "tensor_parallel": _canon["tensor_parallel"],
                         "serving": _canon["serving"]}, sort_keys=True))

    if args.mode == "plan":
        # Which tenants' degrees are floors standing in for a measurement, so `plan` can print them
        # as floors. Empty for every fleet that states `tp:`, and empty as well for one whose
        # degrees the profile store already settles.
        return _plan(args, fleet, store, fleetfile.unresolved_tp(loaded),
                     fleetfile.upper_bound_tp(loaded), max_nodes=loaded["max_nodes"])
    if args.mode == "calibrate":
        return _calibrate(args, store, mirror)
    if args.mode == "export":
        return _export(args, fleet, loaded, hardware, store, mirror, profiles)

    # The last free check, and the one with the largest saving attached: if no two of these tenants'
    # minimum envelopes fit on one GPU then no co-resident placement exists, and the search below
    # would spend a node-hour re-deriving the disjoint baseline.
    #
    # Applied to `search` alone. `profile` measures one tenant at a time with nothing beside it, so
    # neither check bears on it: a fleet whose pairs do not fit still has a minimal TP per tenant,
    # and establishing it is what turns the refusal above from arithmetic on stated degrees into
    # arithmetic on measured ones.
    base = _baseline(fleet) if args.mode == "search" else None
    if (base is not None and not base.can_coreside and not args.allow_disjoint_only
            and pairs_fitting_on_weights(base, fleet)):
        # Refused above until impl §14.2. The pairs miss on the computed terms -- KV pool from a
        # derived page size, an activation fallback, a reserve constant -- and not on the weights, so
        # closing the fleet here would be arithmetic ending a search nobody ran. Warn at the volume
        # the refusal had and carry on: a search that finds nothing has measured that, and a search
        # that finds a placement has just corrected the arithmetic that would have forbidden it.
        pairs = pairs_fitting_on_weights(base, fleet)
        log.warning("no co-resident placement is feasible AT THE COMPUTED ENVELOPES, but %d of %d "
                    "tenant pairs fit one GPU on weights alone (%s). Searching anyway, because the "
                    "terms that rule the pairs out are the ones this stack has caught being wrong "
                    "-- the KV page size by 3.27x on the 9B (impl §14.2). Expect either a placement "
                    "the arithmetic did not predict, or a measured confirmation of the disjoint "
                    "baseline. Both are results; the refusal was not.",
                    len(pairs), len(base.pairs),
                    ", ".join(f"{p.a}+{p.b}" for p in pairs[:4]))
    elif base is not None and not base.can_coreside and not args.allow_disjoint_only:
        tight = base.tightest_pair
        closest = ("" if tight is None else
                   f" The closest pair is {tight.a}+{tight.b}, needing "
                   f"{tight.total_gib:.2f} GiB on a {tight.device_total_gib:.2f} GiB GPU -- over by "
                   f"{tight.deficit_gib:.2f}.")
        raise SystemExit(
            "refusing to search: no co-resident placement is feasible for this fleet, so the only "
            f"placement to find is the disjoint baseline of {base.resolved_gpus} GPU(s).{closest} The "
            "solver may raise an envelope above a tenant's KV floor but never below it, so this is "
            "not a tight-envelope problem -- it holds at every envelope. Nor is it a computed-terms "
            "problem: no two of these tenants' shards fit one GPU counting WEIGHTS alone, which is "
            "the one figure here that cannot come back smaller than stated (impl §14.2), so a "
            "measurement has nothing to correct. Run `plan` for the whole "
            "picture, and see USER-GUIDE.md step 8 for what makes a fleet worth searching. Pass "
            "--allow-disjoint-only to search anyway, which is the right flag when the point is to "
            "measure the baseline itself.")
    upper_bound = fleetfile.upper_bound_tp(loaded)
    if base is not None and upper_bound:
        # Not refused, and no longer keyed off arithmetic. A degree above the memory floor is the
        # ordinary case here -- it means the floor degree was measured to miss -- so what is worth
        # warning about is narrower than that: a degree that passed solo while some narrower degree
        # was never measured at all. Then and only then is `sum(tp)` an upper bound, and the saving
        # below may include GPUs re-sharding alone would free. Said loudly now and carried into the
        # summary rather than left implicit.
        log.warning("baseline may be inflated: %s met their target solo at the degree loaded, but a "
                    "narrower degree was never profiled, so nothing rules the narrower one out. The "
                    "%d-GPU baseline could be as low as %d. Run `profile` for the missing degrees "
                    "to settle it.", ", ".join(upper_bound), base.resolved_gpus, base.floor_gpus)

    # The cluster, last of the free checks: a stack that needs an operator the cluster does not run
    # would be leased a node, installed, and never come up.
    try:
        check_prerequisites(fleetfile.fleet_engines(fleetfile.read_doc(args.fleet)))
    except DeployError as exc:
        raise SystemExit(f"{args.fleet}: {exc}") from exc

    capture = HelmCapture(args.daemon_chart, namespace=args.namespace,
                          holder_manifest=None if args.no_holder else args.holder_manifest)
    token = args.run_token or new_run_token()
    print(f"\ncapturing a {hardware.instance_type} as run {token}: installing daemon release "
          f"mps-{token}, which requests all {hardware.gpu_count} GPUs on one node. If no held "
          "node can be preempted, Karpenter provisions one -- this is where the GPU spend starts.")
    try:
        lease = capture.capture(token, hardware, timeout_s=args.capture_timeout,
                                index=hardware_index, image=key.env.engine_image)
    except EnvironmentMoved as exc:
        # This node is not the machine the store was keyed to, and the capture has already done
        # everything that can be done about it: filed what the node actually reports and taken its own
        # daemon back off it, so the machine is held by the holder again. Shared here rather than in
        # the capture, because the capture does not know a cluster exists.
        _push_hardware(hardware_mirror, hardware_index)
        raise SystemExit(
            f"{exc}\n\nNothing was measured and the node is back under the holder, so re-run this "
            "exact command. No flag and no edit: every store address is derived from the shape now "
            "filed, which is this machine's own, and the baseline the run prints will be read from "
            "the rows that belong to it.") from exc
    print(f"  {lease.summary()}")
    # Shared as soon as it is known, for `_push_hardware`'s reason: this is the only run holding this
    # node, and what it observed about the machine should not be lost if the search below fails.
    _push_hardware(hardware_mirror, hardware_index)
    try:
        rc = (_profile(args, loaded, profiles, profile_mirror, lease, hub=hub)
              if args.mode == "profile"
              else _search(args, loaded, store, mirror, lease, profiles=profiles, hub=hub))
    except BaseException:
        _release(args, capture, lease, failed=True)
        raise
    _release(args, capture, lease, failed=False)
    return rc


_GROUP_OWN = ("--fleet", "--summary", "--run-token", "--export-to")


def _group_argv(argv, fleet_path, summary_path) -> list:
    """The same command for one group: its own fleet file and its own summary. A run token and an
    export target each name one run, so neither is handed to a group's.

    `--allow-disjoint-only`, because a group no two of whose tenants share a GPU -- one tenant,
    say -- is still a node of the split, and it is deployed from its own measured record and
    export like any other. Refusing it would leave the split with a node nothing can deploy."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in _GROUP_OWN:
            skip = True
            continue
        if a.split("=", 1)[0] in _GROUP_OWN:
            continue
        out.append(a)
    return out + ["--fleet", str(fleet_path), "--summary", str(summary_path)] + (
        [] if "--allow-disjoint-only" in out else ["--allow-disjoint-only"])


def _split_reasoner(args):
    """The reasoner choosing groups. A function of its own so the self-tests can replace it."""
    from .reason import DEFAULT_MODEL_ID, BedrockReasoner
    return BedrockReasoner(args.reasoner_model or DEFAULT_MODEL_ID)


def _search_splits(args, argv, loaded) -> int:
    """`search` for a fleet that may use more than one node: an ordinary search per group.

    The splits are `plan`'s, and which groups are searched in which order is `splits.drive`'s. Each
    group is this same command on the group's own fleet file (`splits.write_group`), so it leases its
    own node, keeps or releases it exactly as the flags say, and stores, summarizes and exports
    itself. What this adds is the record of which split was searched and why it stopped.
    """
    fleet, max_nodes = loaded["fleet"], loaded["max_nodes"]
    solved: dict = {}

    def solve_splits(apart=(), together=()):
        # Cached, because the reasoner's `partition` asks again for what the order below printed.
        key = (tuple(apart), tuple(together))
        if key not in solved:
            solved[key] = split(fleet.node.node_spec, fleet.requests, max_nodes, cap=args.top_k,
                                max_seconds=args.solver_seconds,
                                envelope_step_mib=AGENT_ENVELOPE_STEP_MIB, apart=apart,
                                together=together)
        return solved[key]

    found = solve_splits()
    if not found.splits:
        raise SystemExit(f"{args.fleet}: no split over {max_nodes} node(s) to search. `plan` says "
                         "why, and whether the weights alone rule it out.")
    print(f"\n{len(found.splits)} split(s) over up to {max_nodes} node(s), best first; at most "
          f"{args.group_budget} group(s) will be searched, chosen by the {args.split_chooser}:")
    for i, s in enumerate(found.splits, 1):
        print(f"  #{i}  " + " | ".join(", ".join(g.tenants) for g in s.groups))
    doc = fleetfile.read_doc(args.fleet)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    def run_group(tenants) -> GroupRun:
        path = write_group(args.store, args.fleet, doc, tenants)
        summary_path = path.with_name(f"{path.stem}.{stamp}.summary.json")
        print(f"\n=== group of {len(tenants)}: {', '.join(tenants)}\n    {path}")
        try:
            main(_group_argv(argv, path, summary_path))
        except SystemExit as exc:
            # A refusal of the group's own, before or after its lease: the group did not meet its
            # targets, and the message is the reason. Anything else is not caught, so a failure
            # that may have left a node held stops the walk rather than leasing another.
            if exc.code not in (0, 1, None):
                print(f"  the group's search refused: {exc.code}")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        if summary is not None:
            # What `recall_group` matches a later run on: this exact group file, not its name.
            summary["group_fleet_sha256"] = text_sha256(path.read_text())
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return GroupRun(tuple(tenants), str(path), (summary or {}).get("verdict"), summary)

    def recall(tenants):
        run = recall_group(args.store, args.fleet, doc, tenants)
        if run is not None:
            print(f"\n=== group of {len(tenants)}: {', '.join(tenants)}\n    searched by an "
                  f"earlier run ({run.summary.get('run_token')}): {run.verdict}; not searched again")
        return run

    if args.split_chooser == "drive":
        result = drive(found.splits, run_group, args.group_budget, recall=recall)
        chosen_by = ["store" if r.recalled else "drive" for r in result.runs]
        verdict = "met" if result.split is not None else (
            "budget" if sum(not r.recalled for r in result.runs) >= args.group_budget
            else "unresolvable")
        submitted = None
    else:
        tools = SplitTools(fleet, solve_splits, run_group, args.group_budget, recall=recall)
        result = SplitPackerAgent(tools, _split_reasoner(args), max_nodes=max_nodes,
                                  instance_type=fleetfile.fleet_instance_type(doc)).run()
        chosen_by, verdict, submitted = result.chosen_by, result.verdict, result.submitted
    print(f"\n=== splits ===\nstopped: {result.stopped_because}\nverdict: {verdict}")
    for run, by in zip(result.runs, chosen_by):
        print(f"  {run.verdict or 'no verdict':<14} {', '.join(run.tenants)}  "
              f"(run {(run.summary or {}).get('run_token', '-')}, chosen by the {by})")
    if submitted:
        print(f"  the reasoner submitted {submitted['verdict']}: {submitted['rationale']}")
    if "store" in chosen_by:
        print(f"  {chosen_by.count('store')} group(s) were an earlier run's, recalled at no cost.")
    if "fallback" in chosen_by:
        print(f"  {chosen_by.count('fallback')} group(s) were searched by the fallback, not "
              "chosen by the reasoner.")
    out = {"fleet_file": str(args.fleet), "max_nodes": max_nodes, "mode": args.mode,
           "chooser": args.split_chooser, "verdict": verdict, "submitted": submitted,
           "stopped_because": result.stopped_because,
           "split": None if result.split is None else [
               {"tenants": list(g.tenants)} for g in result.split.groups],
           "groups": [{"tenants": list(r.tenants), "fleet_file": r.fleet_file,
                       "verdict": r.verdict, "chosen_by": by,
                       "run_token": (r.summary or {}).get("run_token"),
                       "validity_key_hash": (r.summary or {}).get("validity_key_hash")}
                      for r, by in zip(result.runs, chosen_by)]}
    out["export"] = (None if result.split is None
                     else _export_split(args, max_nodes, result, chosen_by, verdict, stamp))
    path = Path(args.store) / "groups" / Path(args.fleet).stem / f"split.{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"split summary written to {path}")
    if args.summary:
        Path(args.summary).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    if result.split is not None:
        print("Every group of that split met its targets on its own node. Each group's run says "
              "what is still standing on its node; nothing here releases one.")
    return 0 if result.split is not None else 1


def _export_split(args, max_nodes, result, chosen_by, verdict, stamp) -> str | None:
    """The met split as one schema_version 2 export: each group's own export, one per node.

    Each group exported itself at the end of its search, under its own key; this reads those back
    rather than re-deriving them. A group export that is not there is reported and nothing is
    written, because a split with a node missing is not a split anybody can deploy.
    """
    runs = {frozenset(r.tenants): r for r in result.runs}
    nodes = []
    for g in result.split.groups:
        summary = runs[frozenset(g.tenants)].summary or {}
        path = (ObservationStore(args.store, summary.get("validity_key_hash")).exports_dir
                / f"{summary.get('run_token')}.yaml")
        if not path.is_file():
            print(f"  no split export: the group {', '.join(g.tenants)} has no export at {path}")
            return None
        nodes.append((path, yaml.safe_load(path.read_text())))
    try:
        document = canonical.build_split(
            fleet_path=args.fleet, max_nodes=max_nodes, verdict=verdict,
            chooser=args.split_chooser, stopped_because=result.stopped_because, nodes=nodes)
    except canonical.ExportError as exc:
        print(f"  no split export: {exc}")
        return None
    text = canonical.dump(document, canonical.SPLIT_COMMENTS)
    path = Path(args.store) / "groups" / Path(args.fleet).stem / f"export.{stamp}.yaml"
    path.write_text(text)
    if args.export_to:
        Path(args.export_to).write_text(text)
    split = document["split"]
    print(f"split export (schema_version 2, {split['nodes']} node(s), {split['devices']} device(s) "
          f"against a baseline of {split['baseline_devices']}) written to {path}"
          + (f" and {args.export_to}" if args.export_to else ""))
    return str(path)


def _search(args, loaded, store, mirror, lease, profiles=None, hub=None) -> int:
    """The search itself, with a node in hand. Everything here is bound to the lease."""
    # Re-loaded against the captured node, which turns the file's one assertion about the machine
    # into a check -- the instance type must be the one leased -- and re-derives the environment from
    # what this node reported rather than from the shape a previous capture filed. It is also what
    # puts real GPU UUIDs in the fleet: the offline load has none.
    leased = fleetfile.load(args.fleet, namespace=args.namespace, lease=lease, profiles=profiles,
                            hub=hub)
    fleet, key = leased["fleet"], leased["key"]
    if key.hash != store.key_hash:
        # The backstop, not the mechanism: the whole environment half of this key is read off the
        # node, and a node whose device facts differ from the filed shape is caught in the capture
        # itself, which files the new observation and raises `EnvironmentMoved` (agent/capture.py).
        # Reaching here means something moved that the capture does not reconcile -- so the store was
        # already pulled and written under the other hash, and these records would be filed apart
        # from the history they were chosen against.
        raise CaptureError(
            f"the fleet keys to {key.hash} against {lease.node_name} and keyed to "
            f"{store.key_hash} offline. The store has already been read under the latter, so "
            "these records would be filed apart from the history they were chosen against.")

    agent = SearchAgent(
        fleet, store,
        deployer=HelmDeployer(args.chart, namespace=args.namespace,
                              lease_release=lease.release, tenant_prefix=lease.tenant_prefix),
        harness=JobHarness(args.job_template, namespace=args.namespace,
                           image=args.harness_image, store_root=args.efs_store,
                           node_name=lease.node_name),
        lease=lease,
        run_prefix=args.mode, top_k=args.top_k, solver_seconds=args.solver_seconds,
        ready_timeout_s=args.ready_timeout, measure_timeout_s=args.measure_timeout,
        mirror=mirror,
        # The exporter is this lease's own, pinned to this lease's node by the mps-daemon chart.
        # Scraping any other one would attribute another node's DCGM series to these tenants.
        run_config_kwargs={"out_dir": args.efs_store, "exporter_url": lease.exporter_url})

    t0 = time.time()
    packer = _packer(args, fleet, store, agent)
    result = packer.run(teardown_at_end=args.teardown or args.release_node)
    # The half-width this run's misses are judged against, and it is the one the `calibrate` mode
    # last wrote -- not one derived here. A repeat measured during this run is in the store but does
    # not move this number: nothing but `calibrate` writes `calibration.json`, so widening the floor
    # to account for a repeat this run produced takes another `calibrate` and a re-read of the
    # report. Deliberate rather than an oversight -- recomputing it here would judge a run against a
    # threshold its own results moved, and the direction is not the safe one, since a repeat that
    # disagreed more than the stored floor would retroactively excuse the miss that prompted it.
    # `read_calibration` returns None when nothing has been repeated, and `_noise_clause` says so in
    # words rather than substituting a default floor.
    summary = _report(result, fleet, _baseline(fleet), profiles,
                      args.relax_targets_pct, store.read_calibration(),
                      # From the leased load, not the offline one: same file and same store, but the
                      # baseline printed here is `leased`'s, and one function's numbers should come
                      # from one load.
                      fleetfile.upper_bound_tp(leased))
    summary.update(fleet_file=str(args.fleet), validity_key_hash=key.hash,
                   validity_key=key.canonical(), mode=args.mode, driver=args.driver,
                   reasoner=getattr(packer, "name", None) or packer.reasoner.name,
                   budget=args.budget, elapsed_s=round(time.time() - t0, 1),
                   run_token=lease.token, lease_release=lease.release,
                   gpu_node=lease.node_name, gpu_node_observed=lease.observed)
    summary_json = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary:
        Path(args.summary).write_text(summary_json)
        print(f"summary written to {args.summary}")
    # Pushed whether or not --summary was passed. The conclusion belongs beside the evidence it was
    # drawn from, and making that conditional on a local flag means the runs nobody remembered to
    # flag are the ones with no record of what they decided.
    try:
        print(f"summary pushed to {mirror.push_summary(lease.token, summary_json)}")
    except Exception as exc:                                   # noqa: BLE001
        # Loud, but not fatal. The records are already in EFS and the run may have cost an hour of
        # GPU time; raising here would throw away a result over a failed write of its description.
        local = (f"a local copy is at {args.summary}" if args.summary
                 else "and there is no local copy -- pass --summary to keep one")
        print(f"WARNING: could not push the summary to EFS ({exc}). The records are there; this "
              f"run's conclusion is not, {local}.")
    # And the canonical form, written by the run that produced it rather than left to a later
    # invocation of `export`. Everything the store now holds about this result addresses GPUs by UUID
    # and ordinal, and both expire with the node -- so the one document somebody can deploy from is
    # the one nobody was made to remember to ask for. It is free and offline, which is why it belongs
    # here: no part of it costs the lease anything.
    try:
        _write_canonical(fleet=fleet, loaded=leased, hardware=leased["hardware"], store=store,
                         mirror=mirror, profiles=profiles, fleet_path=args.fleet, summary=summary,
                         summary_path=f"observations/{key.hash}/searches/{lease.token}.json",
                         export_to=args.export_to)
    except Exception as exc:                                   # noqa: BLE001
        # Same reasoning as the summary push, more so: this run may have cost an hour of GPU time and
        # every measurement behind the document is already filed. The remedy is one free command.
        log.exception("could not write the canonical form (%s)", exc)
        print("WARNING: the result is stored but was not exported. Nothing is lost -- run "
              f"`export --fleet {args.fleet}` to write it; it reads the store and no cluster.")
    # Non-zero when nothing was measured: an unattended sweep that deployed nothing must not
    # look like a success in a log.
    return 0 if result.measured else 1


def _release(args, capture, lease, *, failed: bool) -> None:
    """Give the lease up, or say exactly what is still running and what it costs.

    Not released by default, and the asymmetry is deliberate: keeping a lease costs an idle
    g6e.48xlarge, while releasing one risks the node itself -- the holder has to be applied inside
    the ~600s window or Karpenter takes the machine, and the replacement's GPU UUIDs make every
    stored placement un-reusable. Money is the recoverable mistake.

    A release failure never masks a failing run. It is logged rather than raised on that path,
    because the run's own exception is the one worth reading, and the node is still held either way.

    **The message names every release the run installed, not just the lease.** A search ends with its
    tenants still up, deliberately (`profiler.run`) -- somebody may want to look at the placement,
    keep serving it or measure it again, which is why teardown is opt-in. But the tenant pods are
    *default* priority, so a handoff that uninstalls only the daemon leaves them holding the GPUs and
    `gpu-node-holder` cannot preempt them: it goes Pending, and a Pending holder is not on the node,
    so it carries neither the GPUs nor its own `do-not-disrupt`. Both runs on 2026-09-10 ended that
    way. The defect was never the default; it was a closing message which, followed exactly, left the
    node unheld. So the tenant releases are listed and ordered before the lease.

    **And the steps are not joined with `&&`.** Ordering the uninstalls correctly is necessary and
    not sufficient: `helm uninstall` returns when the API server accepts the deletion of the objects
    it created, and the pod those objects made keeps running until its container exits. `--wait` does
    not close the gap either -- it waits on the objects, which are already gone. Measured in
    `deploy._await_departed`: an uninstall returned in 0.9s with the outgoing pod still holding 43 GiB
    half a minute later. So `uninstall && apply` can put the holder in while the GPUs are still held,
    which is the Pending-holder-buys-a-second-node failure above arrived at from the other side. The
    code path does not have this bug -- `--release-node` waits for the tenant pods in
    `deploy._await_departed` and for the daemon pod in `capture.release._daemon_gone`, and only then
    applies the holder -- so what is printed here has to wait the same way, by polling the pods rather
    than the releases. The poll is deliberately unbounded: until it clears, the node is still held by
    the very pods it is waiting on, so waiting forever is the safe side and a stuck pod is a thing to
    look at rather than to race.
    """
    if not args.release_node:
        # Read back rather than remembered: what is still installed is the thing the reader has to
        # deal with, and the same ownership rule that decides what may be removed decides what is
        # named here (`deploy.HelmDeployer.live_tenant_releases`).
        tenants: list = []
        try:
            tenants = HelmDeployer(args.chart, namespace=args.namespace,
                                   lease_release=lease.release,
                                   tenant_prefix=lease.tenant_prefix).live_tenant_releases()
        except Exception as exc:                                   # noqa: BLE001
            print(f"\nWARNING: could not list this run's tenant releases ({exc}), so the handoff "
                  f"below names only the lease. Check for releases prefixed "
                  f"{lease.tenant_prefix}- before applying the holder: it is priority -1000 and "
                  "cannot preempt them.")
        held = f"{lease.release} holds {lease.node_name}"
        # `capture.release` takes this off itself, but a handoff is followed by hand -- so anything
        # the run put on the node has to appear in the commands, after the holder apply and never
        # before it, or the reader unprotects a node whose holder is still Pending. Omitted entirely
        # when the run annotated nothing, which is the case on a cluster with no such autoscaler.
        unprotect = "".join(
            f"\n    kubectl annotate node {n} {DO_NOT_DISRUPT}-"
            for n in getattr(lease, "protected_nodes", ()))
        why_unprotect = (
            f"\n  The {DO_NOT_DISRUPT} above is what this run put on the node in place "
            "of the holder's own, so the last step gives that job back to the holder. Left on, it "
            "keeps the machine indefinitely -- the safe side, but not a state anybody chose."
            if unprotect else "")
        # Polls the pods, not the releases, and on its own line rather than `&&`-joined to the
        # apply: see this function's docstring for why an uninstall returning is not a GPU
        # released. A set-based selector covers every release in one call, and `grep -q .` is what
        # makes "no pods at all" the exit condition instead of an error.
        def _wait_for(releases) -> str:
            sel = f"app.kubernetes.io/instance in ({','.join(releases)})"
            return (f"    while kubectl get pod -n {args.namespace} -l '{sel}' "
                    "-o name 2>/dev/null | grep -q .; do sleep 5; done\n")
        why_wait = (
            "\n  The wait between the uninstalls and the apply is load-bearing, not caution: helm "
            "returns when the API server accepts the deletion, while the pod goes on holding its "
            "GPUs until the container exits (0.9s vs ~30s, measured). Join those two steps with "
            "`&&` and the holder can land while the devices are still held, which is the Pending "
            "holder that buys a second node. It has no timeout on purpose -- whatever it is waiting "
            "on is still holding the node, so there is nothing at risk while it waits.")
        if tenants:
            print(f"\nthe lease is still standing: {held}, and {len(tenants)} tenant release(s) "
                  f"from this run are still serving on it: {', '.join(tenants)}. Nothing will "
                  "reclaim the node while any of them is up.\n"
                  "  To keep the placement up, do nothing -- that is what the default is for.\n"
                  "  To give the node back to the holder, ALL of them have to go first: the holder "
                  "is priority -1000 and cannot preempt a tenant pod, so uninstalling only the "
                  "daemon leaves it Pending and the node effectively unheld. Run these in order, "
                  "and do not collapse them onto one line:\n"
                  f"    helm uninstall {' '.join(tenants)} -n {args.namespace} --wait\n"
                  f"    helm uninstall {lease.release} -n {args.namespace} --wait\n"
                  f"{_wait_for(list(tenants) + [lease.release])}"
                  f"    kubectl apply -f {args.holder_manifest}{unprotect}\n"
                  "  or re-run with --release-node, which implies --teardown and does exactly "
                  f"this, waits included.{why_wait}{why_unprotect}")
        else:
            print(f"\nthe lease is still standing: {held}, and no tenant release of this run is "
                  "left. Nothing will reclaim the node while the lease does. To give it back, in "
                  "this order and not collapsed onto one line:\n"
                  f"    helm uninstall {lease.release} -n {args.namespace} --wait\n"
                  f"{_wait_for([lease.release])}"
                  f"    kubectl apply -f {args.holder_manifest}{unprotect}\n"
                  f"  or re-run with --release-node.{why_wait}{why_unprotect}")
        print("To search on this same node again, capture again after handing it to the holder -- "
              "a new run's daemon preempts it.")
        return
    try:
        capture.release(lease)
    except Exception:                       # noqa: BLE001
        if not failed:
            raise
        log.exception("could not release %s after a failed run; the node is still held, which is "
                      "the safe side of this failure", lease.release)


if __name__ == "__main__":
    sys.exit(main())
