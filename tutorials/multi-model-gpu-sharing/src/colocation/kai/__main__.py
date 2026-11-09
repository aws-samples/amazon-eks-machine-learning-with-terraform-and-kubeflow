"""python -m colocation.kai {check,render,deploy,verify,drift,load,teardown}

Deploys an exported result onto KAI NvFractions, as a separate step that nothing else runs.

  check     the KAI stack, the queue, the node pool and the FSx claim are there. Read-only.
  render    size the export's slots as fractions and print each Helm release's values. Offline.
  deploy    check, render, then install the plan's release and one release per tenant, one at
            a time, waiting for each to be bound and pinning every tenant after the first to
            the first one's node; then verify. A split's export (schema_version 2) deploys
            each node as a deployment of its own, `<name>-n<i>`, the first tenant of each later
            node kept off the hosts of the earlier nodes' deployments, in the first one's zone.
  verify    compare where the binder put each tenant with the slots. `--wait-ready` also waits
            for every engine to report healthy.
  drift     record, per tenant and per whole hour since the last record, the real traffic against
            the fleet file's demand and its ITL and TTFT against the targets and the certified
            figures, from Prometheus. Beside the export, and on EFS. Read-only on the cluster.
  load      start a load test of a deployment: a scenario's traffic, through each tenant's
            Service, from a harness Job for whole clock hours. Returns once the Job is applied.
  teardown  uninstall every release the deployment made, delete its load Jobs, and wait until its
            pods have gone.

Exit codes: 0 as planned; 1 failed or timed out; 2 refused before the cluster was touched;
3 placed, but not as the slots say -- tear down and deploy again, the binder's choice of card is
not ours to steer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import cluster
from . import drift
from . import load as kl
from . import plan as kp

log = logging.getLogger("colocation.kai")

REPO = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_STORE = Path.home() / ".colocation" / "store"
DEFAULT_NAMESPACE = "kubeflow-user-example-com"
DEFAULT_PROMETHEUS = "kube-system/prometheus-kube-prometheus-prometheus:http-web"
MISMATCH = 3
REFUSED = 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m colocation.kai", description=__doc__.split("\n")[2])
    p.add_argument("mode", choices=("check", "render", "deploy", "verify", "drift", "load",
                                       "teardown"))
    p.add_argument("target", nargs="?",
                   help="render/deploy: an export path, or a run token found under --store. "
                        "verify/drift/load/teardown: the deployment name (the export's run token by "
                        "default)")
    p.add_argument("scenario", nargs="?", help="load: the scenario file (load-tests/examples/...)")
    p.add_argument("--result", default="strict", choices=kp.RESULTS,
                   help="which of the export's placements to deploy")
    p.add_argument("--name", default=None, help="the deployment's name; the export's stem by default")
    p.add_argument("--image", default=None,
                   help="the vLLM image for every tenant. Default: the tenant's own image when it "
                        "was measured on vllm-openai, else vllm-openai's default. load: the harness "
                        "image, $COLOCATION_HARNESS_IMAGE by default")
    p.add_argument("--store", default=str(DEFAULT_LOCAL_STORE),
                   help="the local store the export was written from. Read only; nothing is pulled "
                        "from EFS -- run `colocation.agent export` here first if it came from elsewhere")
    p.add_argument("--hub-endpoint", default=None)
    p.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    p.add_argument("--queue", default=kp.DEFAULT_QUEUE)
    p.add_argument("-o", "--output", default=None, help="render: write the values here")
    p.add_argument("--chart", default=str(REPO / kp.CHART), help="the kai-vllm chart")
    p.add_argument("--bind-timeout", type=float, default=1800.0,
                   help="how long the FIRST tenant may wait for a card: this includes Karpenter "
                        "launching a node and the GPU Operator building its driver")
    p.add_argument("--next-bind-timeout", type=float, default=600.0)
    p.add_argument("--wait-ready", type=float, default=0.0, metavar="SECONDS",
                   help="verify: also wait this long for every engine to be Ready")
    p.add_argument("--prometheus", default=DEFAULT_PROMETHEUS,
                   help="drift: the Prometheus service, <namespace>/<name>:<port>")
    p.add_argument("--efs-store", default=None,
                   help="drift: the store root on EFS; the agent's default when unset")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _load_plans(args) -> list:
    """One plan per node: a version 1 export is one, and each node of a split is its group's own
    export, deployed as a deployment of its own named `<name>-n<i>`."""
    path = kp.resolve_export(args.target, args.store)
    parts = kp.node_exports(kp.load_export(path), path, REPO)
    name = args.name or kp.dns(path.stem)
    return [_load_plan(args, p, doc, name if len(parts) == 1 else f"{name}-n{i}")
            for i, (p, doc) in enumerate(parts, 1)]


def _load_plan(args, path, doc, name) -> kp.Plan:
    """The export, re-derived offline exactly as `export` derived it, then sized for KAI."""
    from ..agent import fleetfile
    from ..agent.__main__ import _baseline
    from ..store.hubfacts import DEFAULT_HUB_ENDPOINT, HubFactsStore
    from ..store.hwfacts import HardwareIndex
    from ..store.profile import ProfileStore

    fleet_file = kp.fleet_path(doc, REPO)
    index = HardwareIndex(args.store)
    profiles = ProfileStore(args.store, fleetfile.profile_key(fleet_file, index))
    hub = HubFactsStore(args.store, endpoint=args.hub_endpoint or DEFAULT_HUB_ENDPOINT)
    loaded = fleetfile.load(fleet_file, index=index, namespace=args.namespace, profiles=profiles,
                            hub=hub, allow_unresolved_tp=False)
    if loaded["key"].hash != doc["fleet"]["validity_key_hash"]:
        raise kp.PlanError(f"the fleet now keys to {loaded['key'].hash}, the export to "
                           f"{doc['fleet']['validity_key_hash']}: the environment or the code moved")
    fleet = loaded["fleet"]
    base = _baseline(fleet)
    bad = kp.envelope_mismatches(doc, fleet, base)
    if bad:
        raise kp.PlanError("re-deriving the envelopes offline disagrees with the export:\n  "
                           + "\n  ".join(bad))
    shape = index.read_shape(doc["fleet"]["instance_type"])
    if shape is None:
        raise kp.PlanError(f"the store has no instance shape for {doc['fleet']['instance_type']}, "
                           "so there is no card size to fill")
    return kp.build(name=name, result=args.result, export_path=path,
                    doc=doc, fleet=fleet, base=base, addressable_mib=shape.addressable_mib,
                    image=args.image)


def _verify(args, plan: kp.Plan) -> int:
    problems, pods = cluster.problems_for(args.namespace, plan)
    if problems:
        print("NOT AS PLANNED:\n  " + "\n  ".join(problems))
        unbound = any("not been bound" in p or "no pod" in p for p in problems)
        return 1 if unbound else MISMATCH
    node = pods[0]["spec"]["nodeName"]
    print(f"placed as planned: {len(plan.slots)} card(s) on {node}, every slot's residents together")
    deadline = time.monotonic() + args.wait_ready
    while True:
        pods = cluster.live_pods(args.namespace, plan.name)
        waiting = sorted(p["metadata"]["labels"][kp.LABEL_TENANT] for p in pods
                         if not cluster.ready(p))
        if not waiting:
            print("every engine is Ready")
            return 0
        if time.monotonic() >= deadline:
            print("not Ready yet: " + ", ".join(waiting))
            return 1 if args.wait_ready else 0
        time.sleep(15)


def _drift(args, name: str) -> int:
    """Append the hours not yet recorded, beside the export the deployment was made from, and
    push the file to EFS. The export has to be the one deployed: the records are read against it."""
    from ..agent.deploy import DeployError
    from ..agent.mirror import EfsMirror, MirrorError
    from ..agent.render import DEFAULT_STORE_ROOT

    plan = cluster.read_plan(args.namespace, name)
    path = Path(plan.export_path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != plan.export_sha256:
        raise kp.PlanError(f"{path} is not here as it was when {name} was deployed. The records are "
                           "kept beside it, so run drift where the export is")
    doc = kp.load_export(path)
    out = path.parent.parent / "drift" / f"{plan.name}.jsonl"
    lines = [line for line in (out.read_text().splitlines() if out.exists() else []) if line.strip()]
    since = cluster.plan_created(args.namespace, name)
    ends = drift.window_ends(since, drift.last_end(lines), datetime.now(timezone.utc))
    if not ends:
        print(f"{name}: no whole hour to record yet (deployed {since.isoformat()}, "
              f"{len(lines)} record(s) so far)")
        return 0
    figures = {f: drift.series(cluster.prometheus_range(args.prometheus, expr, ends[0], ends[-1],
                                                        drift.WINDOW_S))
               for f, expr in drift.expressions(args.namespace, plan.name).items()}
    recs = drift.records(plan, drift.references(doc, plan.result), figures, ends)
    print(drift.describe(recs))
    text = "".join(f"{line}\n" for line in lines) + "".join(
        json.dumps(r, sort_keys=True) + "\n" for r in recs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"written to {out}")
    mirror = EfsMirror(doc["fleet"]["validity_key_hash"], args.store,
                       efs_root=args.efs_store or DEFAULT_STORE_ROOT, namespace=args.namespace)
    try:
        print(f"written to {mirror.push_drift(plan.name, text)}")
    except (MirrorError, DeployError) as exc:
        print(f"recorded here but not pushed to EFS: {exc}")
        return 1
    return 0


def _load(args, name: str) -> int:
    """Apply a scenario's ConfigMap and Job and return: the Job runs for hours, and its log and
    `drift` are where it is read."""
    import os
    from ..agent.measure import DEFAULT_JOB_TEMPLATE

    if not args.scenario:
        raise kp.PlanError("load needs the deployment name and a scenario file")
    image = args.image or os.environ.get("COLOCATION_HARNESS_IMAGE")
    if not image:
        raise kp.PlanError("no harness image: pass --image or set $COLOCATION_HARNESS_IMAGE "
                           "(plugins/colocation-pipeline/scripts/harness-image.sh --ensure)")
    scenario = kl.load_scenario(args.scenario)
    plan = cluster.read_plan(args.namespace, name)
    path = Path(plan.export_path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != plan.export_sha256:
        raise kp.PlanError(f"{path} is not here as it was when {name} was deployed, and the "
                           "scenario's demand is read from it")
    doc = kp.load_export(path)
    pods = cluster.live_pods(args.namespace, plan.name)
    waiting = sorted(set(kp.dns(n) for n in plan.tenants)
                     - {p["metadata"]["labels"][kp.LABEL_TENANT] for p in pods if cluster.ready(p)})
    if waiting:
        raise kp.PlanError(f"not every engine of {name} is Ready ({', '.join(waiting)}); "
                           f"`verify {name} --wait-ready 900` first")
    run_id = kl.job_name(plan.name, args.scenario, time.strftime("%m%d%H%M", time.gmtime()))
    config = kl.run_config(plan, doc, scenario, args.namespace, run_id)
    node = cluster.get_json(["get", "node", pods[0]["spec"]["nodeName"]])
    zone = node["metadata"].get("labels", {}).get(kl.ZONE_LABEL)
    template = yaml.safe_load((REPO / DEFAULT_JOB_TEMPLATE).read_text())
    cm, job = kl.manifests(template, plan, config, scenario["hours"], args.namespace, image, zone,
                           args.scenario)
    print(kl.describe(config, scenario["hours"]))
    cluster.kubectl(["apply", "-f", "-"], stdin=kp.dump_all([cm, job]))
    print(f"applied Job {run_id}{f' in {zone}' if zone else ''}. Follow it with\n"
          f"  kubectl logs -n {args.namespace} -f job/{run_id}\n"
          f"and once its hours have passed, `drift {name}` records what the deployment saw.")
    return 0


def _deploy(args, plans: list) -> int:
    """Each node's plan in turn, then verify each. A later node's first tenant goes on no host
    running an earlier node's deployment, in the first one's zone; the rest of its tenants follow
    it."""
    held = [p.name for p in plans if cluster.live_pods(args.namespace, p.name)]
    if held:
        print(f"{', '.join(held)} already has pods in {args.namespace}; `teardown` it first")
        return REFUSED
    used, zone = [], None
    for i, plan in enumerate(plans):
        used.append(_deploy_node(args, plan, [p.name for p in plans[:i]], zone))
        if zone is None and len(plans) > 1:
            node = cluster.get_json(["get", "node", used[0]]) or {}
            zone = node.get("metadata", {}).get("labels", {}).get(kp.ZONE_LABEL)
    rcs = [_verify(args, plan) for plan in plans]
    if len(plans) > 1:
        print("deployed as " + ", ".join(f"{p.name} on {n}" for p, n in zip(plans, used))
              + ". Each is a deployment of its own: verify, drift, load and teardown take its name")
    return next((rc for rc in rcs if rc), 0)


def _deploy_node(args, plan: kp.Plan, avoid=(), zone=None) -> str:
    plan_release = kp.object_name(plan.name, "plan")
    cluster.install(args.namespace, plan.name, plan_release, kp.plan_values(plan, args.namespace),
                    args.chart)
    node = None
    for i, tenant in enumerate(plan.order):
        cluster.install(args.namespace, plan.name, kp.object_name(plan.name, tenant),
                        kp.tenant_values(plan, tenant, args.queue, node=node,
                                         avoid=() if node else avoid, zone=None if node else zone),
                        args.chart)
        pod = cluster.wait_bound(args.namespace, plan.name, tenant,
                                 args.bind_timeout if i == 0 else args.next_bind_timeout)
        if node is None:
            node = pod["spec"]["nodeName"]
            cluster.install(args.namespace, plan.name, plan_release,
                            kp.plan_values(plan, args.namespace, node), args.chart)
        print(f"  {tenant}: bound on {node} -> "
              f"{', '.join(kp.devices_of(pod['metadata'].get('annotations')))}")
    return node


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    # The fleet loader explains itself at INFO, for the pipeline; here only this package's progress
    # is news, and the loader's "nothing can be deployed from it" would read as a refusal.
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    try:
        if args.mode == "check":
            problems = cluster.prerequisites(args.namespace, args.queue)
            print("\n".join(problems) if problems else "KAI stack, queue, node pool and FSx claim: ok")
            return 1 if problems else 0
        if args.mode in ("verify", "drift", "load", "teardown"):
            if not args.target:
                raise kp.PlanError(f"{args.mode} needs the deployment name")
            name = kp.dns(args.target)
            if args.mode == "teardown":
                cluster.teardown(args.namespace, name)
                print(f"{name}: deleted, and every pod has departed")
                return 0
            if args.mode == "drift":
                return _drift(args, name)
            if args.mode == "load":
                return _load(args, name)
            return _verify(args, cluster.read_plan(args.namespace, name))
        if not args.target:
            raise kp.PlanError(f"{args.mode} needs an export path or run token")
        plans = _load_plans(args)
        for plan in plans:
            print(kp.describe(plan), file=sys.stderr if args.mode == "render" and not args.output
                  else sys.stdout)
        if args.mode == "render":
            text = "".join(f"# helm upgrade --install {r} {args.chart} -n {args.namespace} -f -\n"
                           f"---\n{kp.dump_all([v])}"
                           for plan in plans
                           for r, v in kp.releases(plan, args.namespace, args.queue))
            if args.output:
                Path(args.output).write_text(text)
                print(f"wrote {args.output}")
            else:
                print(text)
            return 0
        problems = cluster.prerequisites(args.namespace, args.queue, plans[0].instance_type,
                                         sum(len(p.slots) for p in plans))
        if problems:
            print("refused, the cluster is not ready for it:\n  " + "\n  ".join(problems))
            return REFUSED
        return _deploy(args, plans)
    except (kp.PlanError, kl.LoadError) as exc:
        print(f"refused: {exc}")
        return REFUSED
    except cluster.ClusterError as exc:
        print(f"failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
