"""An exported result, sized for KAI NvFractions and written as Helm values. Pure: no cluster.

**What changes between an MPS run and a KAI one is who enforces the wall, not where it is.** Under
MPS the envelope is a pinned device memory limit and vLLM is handed `util = ((E - r_mps) h) / T` of
a card that still reports its whole size `T`. Under NvFractions the fraction itself is the device:
CUDA reports the fraction size `L` as the total (measured: an 8192Mi fraction reads 8.000 GiB). So
the one quantity worth carrying across is the engine's own budget, `(E - r_mps) h`, and a shard is
given

    L    = (E - r_mps) + r_kai
    util = ((L - r_kai) h) / L

which hands vLLM exactly the budget it was measured with. `r_kai` is what a fraction costs on top
of the engine's budget -- context, workspaces, and the MPS server sm-sharing runs underneath -- and
it was measured a little higher than the MPS reserve for the same model (proof test 1, Qwen3-0.6B
TP=1: 0.642 GiB against 0.537 under MPS), so it is the MPS reserve plus `KAI_RESERVE_MARGIN_GIB`.

**Every slot's card is filled.** KAI offers no say in which card a fraction lands on; its binpack
puts a fraction on the fullest card it fits. Submitted in slot order with each slot summing to the
card, a slot's later residents can only fit where its first one went, and the next slot's first
resident only on an empty card -- so binpack reproduces the slots. The fill goes to the slot's
single-shard residents, as extra KV, in proportion to what each requires. A TP tenant's fraction is
one size on all its devices, so it is padded only as far as the emptiest of its slots allows, and
only where nothing else on the card could take the slack instead -- which can leave one of its
cards part-empty, and is why the order is not simply slot order (`submission_order`). Whatever binpack actually does is
checked afterwards (`verify`); this is what makes the check pass, not a substitute for it.

The card size is the store's: an instance shape's `addressable_mib`, read off a node (impl §7),
less `CARD_MARGIN_MIB`. No capacity is stated here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..engines import vllm_openai
from ..solver.memory import HEADROOM_DEFAULT, gpu_memory_utilization

# r_kai - r_mps. Measured +0.105 GiB at TP=1 (proof test 1); rounded up to an eighth. At TP=2 the
# engine was measured using a little more than this per card, inside the fraction's headroom over the
# engine's budget -- it is that headroom, not this margin, that keeps a TP>1 shard in its fraction.
KAI_RESERVE_MARGIN_GIB = 0.125

# Left free on every card. KAI's view of a card is a little larger than what the driver lets a
# process allocate, so a card filled to `addressable_mib` would be over-promised; this keeps the
# fill inside what CUDA can actually hand out, beside the MPS server's own context.
CARD_MARGIN_MIB = 256

CONTAINER = "vllm"
PORT = 8000
MANAGED_BY = "colocation-kai"
LABEL_DEPLOYMENT = "colocation.kai/deployment"
LABEL_TENANT = "colocation.kai/tenant"
LABEL_QUEUE = "kai.scheduler/queue"
# KAI gives a Deployment's pods the `inference` priority, which is non-preemptible, and a
# non-preemptible pod is scheduled only within its queue's GPU quota -- the Helm-installed
# `default-queue` has none. The blueprint's kai.tf creates this queue with a quota.
DEFAULT_QUEUE = "colocation-serving"
NODE_POOL = "kai-gpu"
NODE_LABEL = "kai.scheduler/node"
ZONE_LABEL = "topology.kubernetes.io/zone"
TAINT = "kai.scheduler/gpu"
FSX_CLAIM = "pv-fsx"
# What the binder writes once it has chosen cards: `k8s.device-plugin.nvidia.com/gpu=GPU-<uuid>,...`.
DEVICES_ANNOTATION = f"nvidia.com/container.{CONTAINER}.gpus.devices"
REQUEST_ANNOTATION = f"nvidia.com/container.{CONTAINER}.gpu-memory.request"

# Relative to the repository. Every release of a deployment is this chart with values from here.
CHART = "charts/machine-learning/model-serving/kai-vllm"

RESULTS = ("strict", "relaxed", "baseline")


class PlanError(Exception):
    """A refusal before anything is applied. The message says what to change."""


# --------------------------------------------------------------------------------------------
# the export
# --------------------------------------------------------------------------------------------

def resolve_export(arg: str, store_root) -> Path:
    """A path, or a run token naming `observations/<key>/exports/<token>.yaml` in the store."""
    p = Path(arg)
    if p.is_file():
        return p
    hits = sorted(Path(store_root).glob(f"observations/*/exports/{arg}.yaml"))
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise PlanError(f"run token {arg} names {len(hits)} exports; pass the path: "
                        + ", ".join(str(h) for h in hits))
    raise PlanError(f"no export at {arg}, and no exports/{arg}.yaml under {store_root}/observations. "
                    "`colocation.agent export --fleet <fleet>` writes one")


def load_export(path, versions=(1, 2)) -> dict:
    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict) or doc.get("schema_version") not in versions or (
            "tenants" not in doc and "nodes" not in doc):
        raise PlanError(f"{path} is not a schema_version {' or '.join(map(str, versions))} export")
    return doc


def node_exports(doc: dict, path, repo: Path) -> list:
    """`(path, version 1 document)` per node: a version 1 export is one node, and a version 2
    export's nodes are its groups' own exports, read from where each is filed. Each is checked
    against the copy the split export carries, so the file deployed from is the one it names."""
    if doc["schema_version"] == 1:
        return [(Path(path), doc)]
    out = []
    for i, node in enumerate(doc["nodes"]):
        p = Path(node["export"])
        if not p.is_absolute() and not p.is_file():
            p = repo / p
        if not p.is_file():
            raise PlanError(f"node {i + 1} of the split names its export {node['export']}, and it "
                            f"is not at {p}")
        own = load_export(p, versions=(1,))
        body = {k: v for k, v in node.items() if k != "export"}
        if {k: v for k, v in own.items() if k not in ("schema_version", "generated_at")} != body:
            raise PlanError(f"{p} is not the export node {i + 1} of the split was written from: "
                            "the group was exported again since. Search the split again")
        out.append((p, own))
    return out


def choose_slots(doc: dict, result: str) -> list:
    """The slots of one of the export's three results, as lists of tenant names per card."""
    if result not in RESULTS:
        raise PlanError(f"--result must be one of {RESULTS}")
    block = doc.get(result)
    if block is None:
        why = doc.get("relaxed_why_not") if result == "relaxed" else (
            "no measured placement met every ceiling")
        raise PlanError(f"the export has no {result} result ({why}). `--result baseline` deploys "
                        "every shard alone on its own card, which always exists")
    return [[r["tenant"] for r in s["residents"]] for s in block["slots"]]


def fleet_path(doc: dict, repo: Path) -> Path:
    """The fleet file the export was measured from, checked against the hash it carries."""
    p = Path(doc["fleet"]["path"])
    if not p.is_absolute():
        p = repo / p
    if not p.is_file():
        raise PlanError(f"the export names fleet {doc['fleet']['path']}, and it is not at {p}")
    have = hashlib.sha256(p.read_bytes()).hexdigest()
    if have != doc["fleet"]["sha256"]:
        raise PlanError(f"{p} has changed since the export was written (sha256 {have[:12]}, the "
                        f"export says {doc['fleet']['sha256'][:12]}). The result is only a result "
                        "for the file it was measured from: restore it, or profile and search again")
    return p


def envelope_mismatches(doc: dict, fleet, base) -> list:
    """Where re-deriving the envelope offline disagrees with the export's `required_gib`.

    Empty is the only acceptable answer. The export rounds to 3 places and so does this; anything
    else means the reserves, the checkpoint facts or the code moved since the export, and sizing a
    fraction from either number would be sizing it from a guess about which one is right.
    """
    derived = {t.name: t.envelope_at_resolved_gib for t in base.tenants}
    out = []
    for name, row in sorted(doc["tenants"].items()):
        t = fleet.tenant(name)
        got = None if derived.get(name) is None else round(float(derived[name]), 3)
        if got != row["required_gib"]:
            out.append(f"{name}: export requires {row['required_gib']} GiB, re-derived {got}")
        if t.tp != row["shards"]:
            out.append(f"{name}: export says {row['shards']} shard(s), the fleet resolves {t.tp}")
    return out


# --------------------------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------------------------

@dataclass
class TenantPlan:
    name: str
    tp: int
    slots: list
    required_gib: float
    reserve_mps_gib: float
    reserve_source: str
    reserve_kai_gib: float
    min_mib: int          # the fraction with no fill: L at the required envelope
    mib: int              # what is requested, fill included
    util: float
    served_name: str = ""
    model_path: str = ""
    image: str = ""
    args: list = field(default_factory=list)
    env: list = field(default_factory=list)
    shm_size_gi: int = 16


@dataclass
class Plan:
    name: str
    result: str
    export_path: str
    export_sha256: str
    instance_type: str
    devices_per_node: int
    card_mib: int
    headroom: float
    slots: list
    tenants: dict
    order: list
    warnings: list = field(default_factory=list)

    def leftover_mib(self) -> list:
        return [self.card_mib - sum(self.tenants[t].mib for t in slot) for slot in self.slots]

    def to_dict(self) -> dict:
        return {**{k: v for k, v in asdict(self).items() if k != "tenants"},
                "tenants": {n: asdict(t) for n, t in self.tenants.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        return cls(**{**d, "tenants": {n: TenantPlan(**t) for n, t in d["tenants"].items()}})


def card_mib(addressable_mib: int) -> int:
    return int(addressable_mib) - CARD_MARGIN_MIB


def size(slots: list, tenants: dict, card: int, headroom: float = HEADROOM_DEFAULT) -> dict:
    """Per tenant `(min_mib, mib)`: the bare fraction and the one that fills its slots' cards.

    `tenants` maps a name to `(tp, envelope_gib, reserve_mps_gib)`. Pure arithmetic, so it is what
    the self-test drives.
    """
    where = {}
    for i, slot in enumerate(slots):
        for name in slot:
            where.setdefault(name, []).append(i)
    for name, (tp, _, _) in tenants.items():
        if len(where.get(name, ())) != tp:
            raise PlanError(f"{name} has {tp} shard(s) but sits in {len(where.get(name, ()))} slot(s)")
    base = {}
    for name, (tp, envelope, r_mps) in tenants.items():
        r_kai = float(r_mps) + KAI_RESERVE_MARGIN_GIB
        base[name] = math.ceil(((float(envelope) - float(r_mps)) + r_kai) * 1024)
    used = [sum(base[n] for n in slot) for slot in slots]
    for i, u in enumerate(used):
        if u > card:
            raise PlanError(f"slot {i} ({', '.join(slots[i])}) needs {u} MiB as fractions and the "
                            f"card has {card}: the KAI reserve does not fit where the MPS one did")
    mib = dict(base)
    # TP tenants first: one size on every device, so only as much as every one of its cards has
    # spare, and none at all where a single-shard mate can take the slack instead.
    multi = sorted((n for n in tenants if tenants[n][0] > 1), key=lambda n: (-tenants[n][0], n))
    for name in multi:
        spare = min(0 if any(tenants[m][0] == 1 for m in slots[i]) else card - used[i]
                    for i in where[name])
        mib[name] += spare
        for i in where[name]:
            used[i] += spare
    for i, slot in enumerate(slots):
        single = [n for n in slot if tenants[n][0] == 1]
        spare = card - used[i]
        if not single or spare <= 0:
            continue
        weight = sum(base[n] for n in single)
        given = {n: spare * base[n] // weight for n in single}
        given[max(single, key=lambda n: (base[n], n))] += spare - sum(given.values())
        for n in single:
            mib[n] += given[n]
        used[i] = card
    return {n: (base[n], mib[n]) for n in tenants}


def submission_order(slots: list, tps: dict) -> list:
    """Cards with no TP tenant first, then TP tenants widest first, then their single-shard mates.

    Binpack puts a fraction on the fullest card it fits, so the order is what keeps a card that is
    not full from attracting the wrong tenant. A TP tenant's card can be left part-empty (its
    fraction is one size on every device), and anything submitted after it would prefer that card
    to an empty one -- so every slot that can fill a card exactly goes first, while every card not
    yet touched is still empty. A TP tenant then has only empty cards to take, and its mates go
    last, where the fullest card they fit is one of its own -- and those are interchangeable.
    """
    has_multi = [any(tps[n] > 1 for n in slot) for slot in slots]
    first = {}
    for i, slot in enumerate(slots):
        for n in slot:
            first.setdefault(n, i)
    alone = [n for i, slot in enumerate(slots) if not has_multi[i] for n in slot]
    multi = sorted((n for n in first if tps[n] > 1), key=lambda n: (-tps[n], first[n], n))
    mates = [n for i, slot in enumerate(slots) if has_multi[i] for n in slot if tps[n] == 1]
    return alone + multi + mates


def build(*, name: str, result: str, export_path, doc: dict, fleet, base, addressable_mib: int,
          image: str | None = None, headroom: float = HEADROOM_DEFAULT) -> Plan:
    slots = choose_slots(doc, result)
    if len(slots) > int(doc["fleet"]["devices_per_node"]):
        raise PlanError(f"{len(slots)} slots do not fit one {doc['fleet']['instance_type']} "
                        f"({doc['fleet']['devices_per_node']} devices)")
    envelope = {t.name: float(t.envelope_at_resolved_gib) for t in base.tenants}
    inputs, reserves, warnings, swapped = {}, {}, [], {}
    for n in doc["tenants"]:
        t = fleet.tenant(n)
        reserves[n] = t.mem.reserve_gib(t.tp)
        inputs[n] = (t.tp, envelope[n], reserves[n][0])
        if t.tp > 1:
            warnings.append(f"{n}: TP={t.tp}; at TP>1 the engine can use a little more than its "
                            f"planned KAI reserve on each card, which the fraction's "
                            f"{(1 - headroom) * 100:g}% headroom covers")
    card = card_mib(addressable_mib)
    sized = size(slots, inputs, card, headroom)
    # impl §16. KAI has no SM annotation, and its injection passes a pod's own MPS variables through,
    # so a cap the result was measured with is carried the way the search's charts carry it.
    caps = {n: row["active_thread_percentage"] for n, row in (doc[result].get("per_tenant") or {}).items()
            if (row or {}).get("active_thread_percentage") is not None}
    tenants = {}
    for n, row in doc["tenants"].items():
        t = fleet.tenant(n)
        r_mps, source = reserves[n]
        r_kai = r_mps + KAI_RESERVE_MARGIN_GIB
        lo, mib = sized[n]
        util = round(gpu_memory_utilization(mib / 1024, r_kai, mib / 1024, headroom), 6)
        img = image or (t.image if t.engine == vllm_openai.ENGINE.name else vllm_openai.DEFAULT_IMAGE)
        if img != t.image:
            swapped.setdefault((t.engine, t.image, img), []).append(n)
        tenants[n] = TenantPlan(
            name=n, tp=t.tp, slots=[i for i, s in enumerate(slots) if n in s],
            required_gib=row["required_gib"], reserve_mps_gib=r_mps, reserve_source=source,
            reserve_kai_gib=round(r_kai, 4), min_mib=lo, mib=mib, util=util,
            served_name=row["served_name"], model_path=row["model_path"], image=img,
            args=vllm_args(row, t.tp, util, t.extra_args), env=list(t.extra_env) + (
                [{"name": "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "value": str(caps[n])}]
                if n in caps else []),
            shm_size_gi=t.shm_size_gi)
    for (engine, measured, served), names in sorted(swapped.items()):
        warnings.append(f"{', '.join(sorted(names))}: measured on {engine} ({measured}); KAI serves "
                        f"vLLM's OpenAI server from {served}, whose vLLM may not be the measured "
                        "one's (--image to choose)")
    text = Path(export_path).read_bytes()
    return Plan(name=name, result=result, export_path=str(export_path),
                export_sha256=hashlib.sha256(text).hexdigest(),
                instance_type=doc["fleet"]["instance_type"],
                devices_per_node=int(doc["fleet"]["devices_per_node"]), card_mib=card,
                headroom=headroom, slots=slots, tenants=tenants,
                order=submission_order(slots, {n: t.tp for n, t in tenants.items()}),
                warnings=sorted(set(warnings)))


def vllm_args(row: dict, tp: int, util: float, extra_args=()) -> list:
    """The chart's argv, from the export's serving flags (`vllm-openai/templates/deployment.yaml`)."""
    return ["-m", "vllm.entrypoints.openai.api_server", "--host=0.0.0.0", f"--port={PORT}",
            f"--model={row['model_path']}", f"--served-model-name={row['served_name']}",
            f"--tensor-parallel-size={tp}", f"--gpu-memory-utilization={util}",
            f"--max-num-seqs={row['max_num_seqs']}",
            f"--max-num-batched-tokens={row['max_num_batched_tokens']}",
            f"--max-model-len={row['max_model_len']}", f"--dtype={row['dtype']}", *extra_args]


# --------------------------------------------------------------------------------------------
# Helm values
# --------------------------------------------------------------------------------------------

def dns(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")


def object_name(plan_name: str, tenant: str) -> str:
    return dns(f"kai-{plan_name}-{tenant}")[:63].rstrip("-")


def labels(plan_name: str, tenant: str | None = None) -> dict:
    out = {"app.kubernetes.io/managed-by": MANAGED_BY, LABEL_DEPLOYMENT: dns(plan_name)}
    if tenant is not None:
        out[LABEL_TENANT] = dns(tenant)
    return out


def plan_values(plan: Plan, namespace: str, node: str | None = None) -> dict:
    """The plan release: the plan kept beside what it deployed, so `verify` and `teardown` need
    only its name, and the drift recording rules. `node` is the first tenant's, once bound."""
    from .drift import recording_rules
    return {"plan": {"name": object_name(plan.name, "plan"), "labels": labels(plan.name),
                     "planJson": json.dumps(plan.to_dict(), sort_keys=True), "node": node or "",
                     "rules": recording_rules(namespace, plan.name)}}


def tenant_values(plan: Plan, tenant: str, queue: str = DEFAULT_QUEUE,
                  node: str | None = None, avoid=(), zone: str | None = None) -> dict:
    """One tenant's release: a one-replica Deployment on a KAI fraction, its Service, and the
    ServiceMonitor that has Prometheus scrape it. `node` pins it beside the rest. `avoid` and `zone`
    place the first tenant of a split's later node: on no host running a pod of the deployments
    named in `avoid` (the split's earlier nodes), and in the first one's zone. That is pod
    anti-affinity, not `kubernetes.io/hostname NotIn` the nodes: Karpenter refuses to provision for
    a node affinity on the hostname label (seen live, 2026-10-10), and this pod is the one that
    has to bring a new node up. The chart (`kai-vllm`) lays these out; every value is decided
    here."""
    from .drift import SCRAPE_INTERVAL
    t = plan.tenants[tenant]
    selector = {NODE_LABEL: "true", "node.kubernetes.io/instance-type": plan.instance_type}
    if node:
        selector["kubernetes.io/hostname"] = node
    affinity = {}
    if zone:
        affinity["nodeAffinity"] = {"requiredDuringSchedulingIgnoredDuringExecution": {
            "nodeSelectorTerms": [{"matchExpressions": [
                {"key": ZONE_LABEL, "operator": "In", "values": [zone]}]}]}}
    if avoid:
        affinity["podAntiAffinity"] = {"requiredDuringSchedulingIgnoredDuringExecution": [{
            "labelSelector": {"matchExpressions": [
                {"key": LABEL_DEPLOYMENT, "operator": "In", "values": sorted(avoid)}]},
            "topologyKey": "kubernetes.io/hostname"}]}
    env = [{"name": k, "value": v} for k, v in (("HF_HUB_OFFLINE", "1"),
                                                ("TRANSFORMERS_OFFLINE", "1"),
                                                ("HF_HOME", "/tmp/hf"),
                                                ("VLLM_NO_USAGE_STATS", "1"),
                                                ("DO_NOT_TRACK", "1"))] + list(t.env)
    return {"tenant": {
        "name": object_name(plan.name, tenant), "labels": labels(plan.name, tenant),
        "podLabels": {**labels(plan.name, tenant), LABEL_QUEUE: queue},
        "container": CONTAINER, "memory": f"{t.mib}Mi", "devices": t.tp,
        "nodeSelector": selector, "taint": TAINT, "fsxClaim": FSX_CLAIM,
        **({"affinity": affinity} if affinity else {}),
        "shmSizeGi": t.shm_size_gi, "image": t.image, "args": list(t.args), "env": env,
        "port": PORT,
        # Helm rewrites app.kubernetes.io/managed-by to "Helm" on what it installs, so the
        # Service is selected without it.
        "monitor": {"selector": {k: v for k, v in labels(plan.name, tenant).items()
                                 if k != "app.kubernetes.io/managed-by"},
                    "targetLabels": [LABEL_DEPLOYMENT, LABEL_TENANT], "interval": SCRAPE_INTERVAL}}}


def releases(plan: Plan, namespace: str, queue: str = DEFAULT_QUEUE) -> list:
    """(release, values) in install order: the plan, then each tenant in submission order.
    `deploy` installs them one tenant at a time; the release is named as its objects are."""
    return ([(object_name(plan.name, "plan"), plan_values(plan, namespace))]
            + [(object_name(plan.name, n), tenant_values(plan, n, queue)) for n in plan.order])


# --------------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------------

def devices_of(annotations: dict) -> list:
    raw = (annotations or {}).get(DEVICES_ANNOTATION) or ""
    return [d.split("=", 1)[-1] for d in raw.split(",") if d.strip()]


def compare(plan: Plan, pods: dict, others_on_cards: dict | None = None) -> list:
    """What differs between the plan's slots and where the binder put each tenant. Empty = match.

    `pods` maps tenant to `{"node", "annotations"}`. A card's identity is irrelevant, and so is
    which shard of a tenant is which, so the comparison is between multisets of per-card resident
    sets. `others_on_cards` maps a card to fraction pods not in this plan, which share its SMs and
    so void the measurement even when the plan's own residents match.
    """
    problems = []
    missing = [n for n in plan.tenants if n not in pods]
    if missing:
        return [f"no pod for {', '.join(sorted(missing))}"]
    nodes = {p["node"] for p in pods.values()}
    if len(nodes) != 1:
        problems.append(f"the tenants are spread over {len(nodes)} nodes: {sorted(nodes)}")
    cards = {}
    for n, p in pods.items():
        devices = devices_of(p["annotations"])
        if not devices:
            problems.append(f"{n} has not been bound to a card yet")
            continue
        if len(devices) != plan.tenants[n].tp:
            problems.append(f"{n} is on {len(devices)} card(s), planned {plan.tenants[n].tp}")
        want = f"{plan.tenants[n].mib}Mi"
        if (p["annotations"] or {}).get(REQUEST_ANNOTATION) != want:
            problems.append(f"{n} requests {p['annotations'].get(REQUEST_ANNOTATION)}, planned {want}")
        for d in devices:
            cards.setdefault(d, []).append(n)
    if problems:
        return problems
    have = Counter(tuple(sorted(v)) for v in cards.values())
    want = Counter(tuple(sorted(s)) for s in plan.slots)
    if have != want:
        problems.append("the cards hold " + "; ".join("+".join(k) for k in sorted(have.elements()))
                        + " where the slots say " + "; ".join("+".join(k)
                                                              for k in sorted(want.elements())))
    for card, foreign in sorted((others_on_cards or {}).items()):
        if card in cards and foreign:
            problems.append(f"card {card} also holds {', '.join(sorted(foreign))}, which this plan "
                            "does not")
    return problems


def queue_shortfall(queue: dict, name: str, gpus: int | None = None) -> str | None:
    """Why `queue` cannot hold `gpus` whole devices of non-preemptible pods, or None if it can.

    A plan's fractions never exceed its cards, so `len(plan.slots)` is enough quota. With no plan
    (`check` alone) any quota at all is enough. A quota of -1 is unlimited.
    """
    quota = float(((queue.get("spec") or {}).get("resources") or {}).get("gpu", {}).get("quota", 0))
    if quota < 0 or (quota > 0 and (gpus is None or quota >= gpus)):
        return None
    need = "any GPUs" if gpus is None else f"{gpus} GPU(s)"
    return (f"KAI queue {name} has a GPU quota of {quota:g}, and a Deployment's pods are "
            f"non-preemptible, so they are scheduled only within it: it needs {need}")


def describe(plan: Plan) -> str:
    """The sizing, one line per slot, for a person to read before anything is applied."""
    lines = [f"{plan.name}: {plan.result} result, {len(plan.slots)} card(s) of "
             f"{plan.card_mib} MiB on one {plan.instance_type}"]
    for i, (slot, left) in enumerate(zip(plan.slots, plan.leftover_mib())):
        cells = []
        for n in slot:
            t = plan.tenants[n]
            cap = next((e["value"] for e in t.env
                        if e["name"] == "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"), None)
            cells.append(f"{n}{'' if t.tp == 1 else f'(tp{t.tp})'} {t.mib}Mi "
                         f"(needs {t.min_mib}) util {t.util}"
                         + ("" if cap is None else f" SM cap {cap}%"))
        lines.append(f"  slot {i}: " + ", ".join(cells) + f"  -- {left} MiB unfilled")
    for n in plan.order:
        t = plan.tenants[n]
        lines.append(f"  {n}: reserve {t.reserve_mps_gib} GiB ({t.reserve_source}) + "
                     f"{KAI_RESERVE_MARGIN_GIB} on KAI; image {t.image}")
    lines.append("  submission order: " + " -> ".join(plan.order))
    lines += [f"  warning: {w}" for w in plan.warnings]
    return "\n".join(lines)


def dump_all(docs: list) -> str:
    return yaml.safe_dump_all(docs, sort_keys=False)
