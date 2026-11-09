"""A load test of a KAI deployment: traffic through each tenant's Service, from a pod in the cluster.

A scenario names the fleet it is for and how its traffic differs from that fleet's `demand`:

  fleet: fleets/examples/g6e.48xlarge/qwen-anchor-riders-3.yaml
  hours: 2
  tenants:
    guard: {rate: 2.0}      # multipliers on the fleet file's demand
    route: {isl: 1.9}
    title: {per_hour: 10}   # a fixed count instead of a rate
    chat: off               # not driven at all

A tenant it does not name is driven at its fleet demand, so `tenants: {}` is the fleet as written.
The demand itself is read from the export the deployment was made from, never copied into the
scenario, so a scenario cannot drift from its fleet file except by the multipliers it states.

What drives it is the harness image's `harness.soak` (the run's own calibrated prompts and open-loop
driver) as a Job on the CPU pool, in the zone of the deployment's node, as `search` places its Jobs.
It writes nothing to the store: `colocation.kai drift` then records what the deployment saw, hour by
hour, and the Job's log has the client's figures for the same hours.

Pure: the cluster half is in `__main__`.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from . import plan as kp

TENANT_KEYS = ("rate", "isl", "osl", "per_hour")
HARNESS_CONFIGMAP = "colocation-harness-run"
ZONE_LABEL = kp.ZONE_LABEL
LABEL_LOAD = "colocation.kai/load"
# Past the scenario's hours: up to an hour to the first whole clock hour, and the calibration.
DEADLINE_SLACK_S = 2 * 3600


class LoadError(Exception):
    pass


def load_scenario(path) -> dict:
    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict):
        raise LoadError(f"{path} is not a scenario")
    extra = sorted(set(doc) - {"fleet", "hours", "tenants"})
    if extra:
        raise LoadError(f"{path}: unknown key(s) {', '.join(extra)}; a scenario has fleet, hours "
                        "and tenants")
    if not isinstance(doc.get("fleet"), str):
        raise LoadError(f"{path}: `fleet` names the fleet file the scenario is for")
    hours = doc.get("hours")
    if not isinstance(hours, int) or isinstance(hours, bool) or hours < 1:
        raise LoadError(f"{path}: `hours` is a whole number of clock hours, at least 1")
    tenants = doc.get("tenants") or {}
    if not isinstance(tenants, dict):
        raise LoadError(f"{path}: `tenants` maps a tenant name to how its traffic differs")
    for n, t in tenants.items():
        # YAML reads a bare `off` as False.
        if t is False:
            continue
        if not isinstance(t, dict) or not t:
            raise LoadError(f"{path}: tenant {n} is `off` or a map of {', '.join(TENANT_KEYS)}")
        bad = sorted(set(t) - set(TENANT_KEYS))
        if bad:
            raise LoadError(f"{path}: tenant {n}: unknown key(s) {', '.join(bad)}")
        if "rate" in t and "per_hour" in t:
            raise LoadError(f"{path}: tenant {n}: `rate` scales the fleet's rate and `per_hour` "
                            "replaces it; give one")
        for k, v in t.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                raise LoadError(f"{path}: tenant {n}: {k} must be a number above 0")
    return {"fleet": doc["fleet"], "hours": hours, "tenants": tenants}


def demand(scenario: dict, doc: dict) -> dict:
    """{tenant: (rate_hz, isl_tokens, osl_tokens)} for every tenant driven, after the scenario's
    changes. Refuses a scenario for another fleet, a tenant the export does not have, and a
    request longer than the tenant's context, which the engine would refuse one request at a time."""
    if scenario["fleet"] != doc["fleet"]["path"]:
        raise LoadError(f"the scenario is for {scenario['fleet']}, the deployment was exported "
                        f"from {doc['fleet']['path']}")
    unknown = sorted(set(scenario["tenants"]) - set(doc["tenants"]))
    if unknown:
        raise LoadError(f"the fleet has no tenant {', '.join(unknown)} "
                        f"(it has {', '.join(sorted(doc['tenants']))})")
    out = {}
    for n, t in doc["tenants"].items():
        change = scenario["tenants"].get(n, {})
        if change is False:
            continue
        d = t["demand"]
        rate = change["per_hour"] / 3600 if "per_hour" in change else d["rate_hz"] * change.get(
            "rate", 1.0)
        isl = max(1, round(d["isl_tokens"] * change.get("isl", 1.0)))
        osl = max(1, round(d["osl_tokens"] * change.get("osl", 1.0)))
        limit = t.get("max_model_len")
        if limit and isl + osl > limit:
            raise LoadError(f"{n}: {isl} prompt + {osl} output tokens is past its max_model_len "
                            f"{limit}")
        out[n] = (rate, isl, osl)
    if not out:
        raise LoadError("the scenario turns every tenant off")
    return out


def run_config(plan: kp.Plan, doc: dict, scenario: dict, namespace: str, run_id: str) -> dict:
    """The harness's run config, each tenant addressed at its Service. The engine is vLLM's own
    OpenAI server whatever the fleet was measured on, because that is what the chart serves."""
    tenants = []
    for n, (rate, isl, osl) in demand(scenario, doc).items():
        t = doc["tenants"][n]
        images = t.get("images") or {}
        tenants.append({
            "name": n, "base_url": f"http://{kp.object_name(plan.name, n)}.{namespace}.svc:{kp.PORT}",
            "served_name": plan.tenants[n].served_name, "rate_hz": rate, "isl_tokens": isl,
            "osl_tokens": osl, "tokenizer_path": t.get("tokenizer_path") or t["model_path"],
            "image_count": images.get("count", 0), "image_pixels": images.get("pixels", 0),
            "engine": "vllm-openai"})
    return {"run_id": run_id, "validity_key_hash": doc["fleet"]["validity_key_hash"],
            "tenants": tenants}


def job_name(plan_name: str, scenario_path, stamp: str) -> str:
    return kp.object_name(plan_name, f"load-{Path(scenario_path).stem}-{stamp}")


def manifests(template: dict, plan: kp.Plan, config: dict, hours: int, namespace: str,
              image: str, zone: str | None, scenario_path) -> tuple:
    """(ConfigMap, Job) from the harness Job template, running `harness.soak` instead of a
    measurement. The Job carries the deployment's labels so `teardown` takes it too; its pod does
    not, or it would be counted among the deployment's tenants."""
    name = config["run_id"]
    meta = {**kp.labels(plan.name), LABEL_LOAD: kp.dns(Path(scenario_path).stem)}
    cm = {"apiVersion": "v1", "kind": "ConfigMap",
          "metadata": {"name": name, "namespace": namespace, "labels": meta},
          "data": {"run.yaml": yaml.safe_dump(config, sort_keys=False)}}
    job = copy.deepcopy(template)
    if job.get("kind") != "Job":
        raise LoadError("the harness template is not a Job manifest")
    job["metadata"] = {"name": name, "namespace": namespace, "labels": meta}
    job["spec"]["activeDeadlineSeconds"] = hours * 3600 + DEADLINE_SLACK_S
    spec = job["spec"]["template"]["spec"]
    if len(spec["containers"]) != 1:
        raise LoadError("the harness template must have exactly one container")
    c = spec["containers"][0]
    c["image"] = image
    c["command"] = ["python", "-m", "harness.soak"]
    c["args"] = ["--config=/config/run.yaml", f"--hours={hours}"]
    for vol in spec.get("volumes", []):
        if vol.get("configMap", {}).get("name") == HARNESS_CONFIGMAP:
            vol["configMap"]["name"] = name
            break
    else:
        raise LoadError(f"the harness template mounts no `{HARNESS_CONFIGMAP}` ConfigMap")
    if zone:
        spec.setdefault("nodeSelector", {})[ZONE_LABEL] = zone
    return cm, job


def describe(config: dict, hours: int) -> str:
    rows = [f"  {t['name']}: {t['rate_hz'] * 3600:.0f}/h ({t['rate_hz']:.4g}/s), "
            f"{t['isl_tokens']} in, {t['osl_tokens']} out" + (
                f", {t['image_count']} image(s)" if t["image_count"] else "")
            for t in config["tenants"]]
    return f"{hours} whole clock hour(s), through each tenant's Service:\n" + "\n".join(rows)
