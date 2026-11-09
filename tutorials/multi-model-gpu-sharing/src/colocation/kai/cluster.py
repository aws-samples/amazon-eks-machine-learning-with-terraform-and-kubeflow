"""The cluster half of `colocation.kai`: helm and kubectl, and nothing that decides anything.

Every decision is in `plan.py`, where the self-test can reach it. What is here installs, waits and
reads -- and the one thing it does that `plan.py` cannot is wait for the binder between tenants,
because KAI's binpack only sees a fraction once it is bound.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import urllib.parse
from datetime import datetime

from . import plan as kp

log = logging.getLogger(__name__)


class ClusterError(Exception):
    pass


def kubectl(args: list, stdin: str | None = None, check: bool = True) -> str:
    done = subprocess.run(["kubectl", *args], input=stdin, capture_output=True, text=True,
                          timeout=120)
    if check and done.returncode != 0:
        raise ClusterError(f"kubectl {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def get_json(args: list, check: bool = True):
    out = kubectl([*args, "-o", "json"], check=check)
    return json.loads(out) if out.strip() else None


def helm(args: list, stdin: str | None = None) -> str:
    try:
        done = subprocess.run(["helm", *args], input=stdin, capture_output=True, text=True,
                              timeout=300)
    except FileNotFoundError:
        raise ClusterError("helm is not on PATH; every release of a KAI deployment is a Helm one")
    if done.returncode != 0:
        raise ClusterError(f"helm {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def install(namespace: str, name: str, release: str, values: dict, chart) -> None:
    """One release, not waited on: Ready is minutes of engine start, and the next tenant only
    needs this one bound. The release is labelled with its deployment, which is how teardown
    finds every release of it and no other's."""
    helm(["upgrade", "--install", release, str(chart), "-n", namespace, "-f", "-",
          "--labels", f"{kp.LABEL_DEPLOYMENT}={kp.dns(name)}"], stdin=kp.dump_all([values]))


def releases_of(namespace: str, name: str) -> list:
    out = helm(["list", "-n", namespace, "-q", "-l", f"{kp.LABEL_DEPLOYMENT}={kp.dns(name)}"])
    return out.split()


def selector(name: str, tenant: str | None = None) -> str:
    return ",".join(f"{k}={v}" for k, v in kp.labels(name, tenant).items())


def live_pods(namespace: str, name: str, tenant: str | None = None) -> list:
    items = (get_json(["get", "pods", "-n", namespace, "-l", selector(name, tenant)]) or {})
    return [p for p in items.get("items", []) if not p["metadata"].get("deletionTimestamp")]


def by_tenant(pods: list) -> dict:
    out = {}
    for p in pods:
        out.setdefault(p["metadata"]["labels"].get(kp.LABEL_TENANT), []).append(p)
    return out


def ready(pod: dict) -> bool:
    return any(c["type"] == "Ready" and c["status"] == "True"
               for c in pod.get("status", {}).get("conditions", []))


def last_event(namespace: str, pod: dict) -> str:
    uid = pod["metadata"]["uid"]
    ev = get_json(["get", "events", "-n", namespace,
                   "--field-selector", f"involvedObject.uid={uid}"], check=False) or {}
    items = sorted(ev.get("items", []), key=lambda e: e.get("lastTimestamp") or "")
    return "" if not items else f"{items[-1]['reason']}: {items[-1]['message']}"[:300]


def wait_bound(namespace: str, name: str, tenant: str, timeout_s: float) -> dict:
    """The tenant's pod once the binder has written its cards. Raises on timeout."""
    deadline, said = time.monotonic() + timeout_s, ""
    while time.monotonic() < deadline:
        pods = live_pods(namespace, name, kp.dns(tenant))
        if len(pods) == 1:
            pod = pods[0]
            if kp.devices_of(pod["metadata"].get("annotations")) and pod["spec"].get("nodeName"):
                return pod
            if pod.get("status", {}).get("phase") == "Failed":
                raise ClusterError(f"{tenant}: pod failed before binding")
            msg = last_event(namespace, pod)
            if msg and msg != said:
                log.info("%s: %s", tenant, msg)
                said = msg
        time.sleep(10)
    raise ClusterError(f"{tenant} was not bound within {timeout_s:.0f}s"
                       + (f"; last event {said}" if said else ""))


def read_plan(namespace: str, name: str) -> kp.Plan:
    cm = get_json(["get", "configmap", "-n", namespace, kp.object_name(name, "plan")], check=False)
    if not cm:
        raise ClusterError(f"no deployment named {name} in {namespace} (no ConfigMap "
                           f"{kp.object_name(name, 'plan')})")
    return kp.Plan.from_dict(json.loads(cm["data"]["plan.json"]))


def plan_created(namespace: str, name: str) -> datetime:
    """When the deployment's plan was first installed: the earliest hour drift can record."""
    cm = get_json(["get", "configmap", "-n", namespace, kp.object_name(name, "plan")])
    return datetime.fromisoformat(cm["metadata"]["creationTimestamp"].replace("Z", "+00:00"))


def prometheus_range(service: str, expr: str, start: datetime, end: datetime, step_s: int) -> dict:
    """A query_range answer, asked through the API server's service proxy: no port-forward, and
    the same kubectl credentials as everything else here. `service` is `<namespace>/<name>:<port>`."""
    ns, svc = service.split("/", 1)
    query = urllib.parse.urlencode({"query": expr, "start": int(start.timestamp()),
                                    "end": int(end.timestamp()), "step": step_s})
    answer = json.loads(kubectl(["get", "--raw",
                                 f"/api/v1/namespaces/{ns}/services/{svc}/proxy/api/v1/query_range?"
                                 f"{query}"]))
    if answer.get("status") != "success":
        raise ClusterError(f"Prometheus refused {expr}: {answer.get('error')}")
    return answer


def observed(namespace: str, name: str) -> tuple:
    """`(pods by tenant for plan.compare, other fraction pods per card, the raw pods)`."""
    pods = live_pods(namespace, name)
    grouped = by_tenant(pods)
    view = {}
    for tenant_dns, ps in grouped.items():
        if len(ps) != 1:
            continue
        p = ps[0]
        view[tenant_dns] = {"node": p["spec"].get("nodeName"),
                            "annotations": p["metadata"].get("annotations") or {}}
    nodes = {v["node"] for v in view.values() if v["node"]}
    others = {}
    mine = {p["metadata"]["uid"] for p in pods}
    for node in nodes:
        allp = get_json(["get", "pods", "-A", "--field-selector", f"spec.nodeName={node}"]) or {}
        for p in allp.get("items", []):
            if p["metadata"]["uid"] in mine or p.get("status", {}).get("phase") in ("Succeeded",
                                                                                     "Failed"):
                continue
            for d in kp.devices_of(p["metadata"].get("annotations")):
                others.setdefault(d, []).append(
                    f"{p['metadata']['namespace']}/{p['metadata']['name']}")
    return view, others, pods


def problems_for(namespace: str, plan: kp.Plan) -> tuple:
    """`plan.compare` against the cluster, plus pods duplicated by a restart. Returns (problems, pods)."""
    view, others, pods = observed(namespace, plan.name)
    dup = [t for t, ps in by_tenant(pods).items() if len(ps) > 1]
    keyed = {n: view[kp.dns(n)] for n in plan.tenants if kp.dns(n) in view}
    problems = [f"{t} has {len(by_tenant(pods)[t])} pods" for t in dup]
    return problems + kp.compare(plan, keyed, others), pods


def prerequisites(namespace: str, queue: str, instance_type: str | None = None,
                  gpus: int | None = None) -> list:
    """What has to exist before a deploy can bind anything. Empty means ready. Read-only."""
    out = []
    pods = get_json(["get", "pods", "-n", "kai-scheduler"], check=False) or {}
    running = {p["metadata"]["name"].rsplit("-", 2)[0] for p in pods.get("items", [])
               if p.get("status", {}).get("phase") == "Running"}
    for want in ("kai-scheduler-default", "binder", "admission", "pod-grouper",
                 "kai-scheduler-gpu-fractioning"):
        if want not in running:
            out.append(f"kai-scheduler/{want} is not Running")
    q = get_json(["get", "queues.scheduling.run.ai", queue], check=False)
    if not q:
        out.append(f"KAI queue {queue} does not exist")
    elif kp.queue_shortfall(q, queue, gpus):
        out.append(kp.queue_shortfall(q, queue, gpus))
    pool = get_json(["get", "nodepool", kp.NODE_POOL], check=False)
    if not pool:
        out.append(f"Karpenter NodePool {kp.NODE_POOL} does not exist")
    elif instance_type:
        reqs = pool["spec"]["template"]["spec"].get("requirements", [])
        types = [r for r in reqs if r["key"] == "node.kubernetes.io/instance-type"]
        if types and types[0]["operator"] == "In" and instance_type not in types[0]["values"]:
            out.append(f"NodePool {kp.NODE_POOL} does not offer {instance_type} "
                       f"(it offers {', '.join(types[0]['values'])})")
    if not get_json(["get", "pvc", "-n", namespace, kp.FSX_CLAIM], check=False):
        out.append(f"PVC {kp.FSX_CLAIM} does not exist in {namespace}")
    return out


def teardown(namespace: str, name: str, timeout_s: float = 900.0) -> int:
    """Uninstall every release of this deployment, the plan's last, and wait until its pods have
    departed. Anything left labelled with it -- an object no release owns, a load Job and its
    ConfigMap -- is deleted too."""
    plan_release = kp.object_name(name, "plan")
    for release in sorted(releases_of(namespace, name), key=lambda r: r == plan_release):
        helm(["uninstall", release, "-n", namespace])
    kubectl(["delete", "deployment,service,configmap,job", "-n", namespace, "-l", selector(name),
             "--ignore-not-found", "--wait=false"])
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pods = get_json(["get", "pods", "-n", namespace, "-l", selector(name)]) or {}
        left = len(pods.get("items", []))
        if not left:
            return 0
        time.sleep(5)
    raise ClusterError(f"{left} pod(s) of {name} still present after {timeout_s:.0f}s")
