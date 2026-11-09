---
name: kai-deploy
description: Deploy an exported co-location result onto the KAI scheduler's GPU fractions (vLLM OpenAI server, one Helm release per tenant), check the binder put every tenant where the export's slots say, and retry when it did not; then record its drift from the fleet file and load-test it. Use only when the user asks to deploy, serve, or roll out a result on KAI, to check on a KAI deployment, to measure its drift, to load-test it, or to take one down. Never as a step of running a fleet.
---

# Deploy an export on KAI

You are handed an **export** — `observations/<key>/exports/<token>.yaml`, or just its run token
(`rt65y1`). Your job is to serve its placement on the `kai-gpu` node pool and leave it either placed
exactly as its slots say, or torn down with the reason stated.

**This is a separate step and the user starts it.** Nothing in the pipeline deploys a result, and
`run-fleet` never chains into this. The guard hook asks before any `colocation.kai deploy`, `load` or
`teardown` unless the user's own prompt was `/fleet-kai-deploy`, `/fleet-kai-load` or
`/fleet-kai-teardown`; an ask is
the rule working, not something to route around.

Everything runs from the repository's `tutorials/multi-model-gpu-sharing` directory with `PYTHONPATH=src`.

**A split's export is one deployment per node.** An export with `schema_version: 2` (written by
`search` on a fleet with `max_nodes` above 1, `<store>/groups/<fleet>/export.<stamp>.yaml`) holds
each node's own group export. `render` and `deploy` take it like any export, and each node becomes
a deployment of its own, `<name>-n1`, `<name>-n2`, ..., deployed one after the other exactly as a
single node is. The first tenant of each later node is kept off the earlier nodes' hosts and in the
first node's zone. It therefore needs that many `kai-gpu` nodes of the instance type in one zone,
and capacity for that is not guaranteed, so say so before deploying. Both `render` and `deploy`
refuse (exit 2) if a node's own export file no longer matches the copy in the split, because that
means the group was exported again since. Every per-deployment step below (retry, verify, drift,
teardown) takes a node's `<name>-n<i>`, and a mismatch on one node is retried by tearing down
**every** node of the split and deploying again, since later nodes were placed relative to earlier ones.

## What the code decides, so you do not

- **Sizing.** `render` re-derives each tenant's envelope offline, exactly as `export` did, and
  refuses (exit 2) if any differs from the export's `required_gib` or if the fleet file's sha256
  moved. Each shard becomes a fraction `L = (E - r_mps) + r_kai` with `r_kai = r_mps + 0.125 GiB`,
  and `util` is computed against `L`, which hands vLLM the budget it was measured with.
- **Filling.** Every slot's card is filled to the store's `addressable_mib` less 256 MiB, the extra
  going to the slot's single-shard tenants as KV. That is what makes KAI's binpack reproduce the
  slots; it is not a choice to revisit per deploy.
- **Order.** Cards without a TP tenant first, then TP tenants, then their mates — each tenant
  waited on until the binder has written its cards, and every tenant after the first pinned to the
  first one's node. Each tenant is a Helm release of the `kai-vllm` chart, `kai-<name>-<tenant>`,
  installed without `--wait` (bound is what the next one needs, not Ready), and the plan is a
  release of its own, `kai-<name>-plan`. Helm orders nothing between releases; `deploy` does.
- **SM caps.** A tenant the search measured under an SM cap carries `active_thread_percentage` in
  the export, and is served with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` set to it in its pod env, which
  KAI's shared MPS server honours. `render` prints `SM cap N%` beside it. An uncapped tenant gets no
  variable.

Do not edit the values or the chart, change a fraction size, or pass `--image` unless the user asked
for it, and do not `helm install`, `upgrade` or `uninstall` these releases by hand: out of order,
binpack puts a fraction wherever it fits.

## Procedure

1. **Read the plan first. Free and offline:**
   ```bash
   PYTHONPATH=src python3 -m colocation.kai render <export> -o /tmp/<token>-kai.yaml
   ```
   Report the slot lines and every `warning:` line. Two are expected today and must be passed on,
   not dropped:
   - the image swap — results measured on `dynamo-vllm` are served by vLLM's OpenAI server from
     the NGC image, whose vLLM version differs from the measured one;
   - TP>1 — the engine can use a little more than its planned KAI reserve on each card, which
     the fraction's headroom covers.
   `--result strict` is the default; if the export has no strict result the refusal names
   `--result baseline`. Use another result only when the user chose it.

   `render` also refuses (exit 2), before anything is applied, when the fleet now keys to a
   different validity key than the export (the environment or the code moved), when the slots do
   not fit one node of the instance type, when a slot's fractions with the KAI reserve overflow the
   card where the MPS reserve did not, or when the store has no instance shape to size cards from.
   Report the refusal; none of these is fixed by editing the export.

2. **Deploy, in the background** (`run_in_background: true`): the first tenant may wait up to 30
   minutes for Karpenter to launch a node and the GPU Operator to build its driver.
   ```bash
   PYTHONPATH=src python3 -m colocation.kai deploy <export>
   ```
   It needs `helm` on PATH. It runs `check` first and refuses (exit 2) if any of the five `kai-scheduler` components
   (scheduler, binder, admission, pod-grouper, gpu-fractioning) is not Running, if the
   `colocation-serving` queue, the `kai-gpu` NodePool or the `pv-fsx` claim is missing, if the
   NodePool does not offer the export's instance type, if the queue's GPU quota is short of the
   plan's cards (a Deployment's pods are non-preemptible, so KAI schedules them only within it), or
   if a deployment of that name already has pods.

3. **Read the exit code.**
   - `0` — every tenant is bound where its slot says. Go to 4.
   - `3` — bound, but not as the slots say (the binder chooses cards; we cannot steer it). Retry:
     ```bash
     PYTHONPATH=src python3 -m colocation.kai teardown <name> && \
       PYTHONPATH=src python3 -m colocation.kai deploy <export>
     ```
     **At most 3 attempts in all.** After the third mismatch, tear it down and report the last
     `NOT AS PLANNED` lines verbatim. Do not try to fix a placement by hand — deleting one pod lets
     its Deployment re-create it wherever binpack likes.
   - `1` — something failed or timed out (a tenant never bound, a pod failed). Report the message
     and the last event it printed. Do not retry blindly: a pod that cannot bind will not bind on
     the second try either. Leave it up for the user to look at unless they say to tear it down.
   - `2` — refused before anything was applied. Report the refusal; it says what to change.

4. **Wait for the engines**, in the background:
   ```bash
   PYTHONPATH=src python3 -m colocation.kai verify <name> --wait-ready 1800
   ```
   `0` means placed as planned and every engine Ready. A tenant that is placed but never Ready
   (CrashLoopBackOff, OOM at startup) is a finding about the fraction size: report it with
   `kubectl logs -n kubeflow-user-example-com deploy/kai-<name>-<tenant> --tail=40` and do not
   retry.

5. **Report**: the node, each tenant's fraction and util, any SM cap, the cards it is on (`deploy`
   prints them), Ready or not, the warnings from step 1, and how many attempts it took. Each tenant is
   served at `http://kai-<name>-<tenant>.kubeflow-user-example-com.svc:8000/v1` under its export
   `served_name`.

## Status and teardown

- `PYTHONPATH=src python3 -m colocation.kai verify <name>` — read-only; reads the plan from the
  `kai-<name>-plan` ConfigMap, so it needs only the name. `0` placed as planned, `3` placed
  differently (also two pods for one tenant, or a foreign pod on a planned card), `1` a tenant with
  no pod or unbound, no such deployment, or `--wait-ready` running out.
- `PYTHONPATH=src python3 -m colocation.kai teardown <name>` — uninstalls every release labelled
  `colocation.kai/deployment=<name>`, the plan's last, deletes its load Jobs, and returns only once
  every pod has departed; after 900 s with pods left it exits 1. The
  node stays until Karpenter consolidates it (about an hour empty); say so rather than deleting the
  node claim yourself.

## Drift

`PYTHONPATH=src python3 -m colocation.kai drift <name>` is read-only. Run it whenever the user asks
how a deployment is doing under real traffic. Every tenant's release is scraped by the cluster
Prometheus. Per tenant and per whole hour since the deploy, the command records the request rate
and the prompt and output lengths against the fleet file's `demand`, and ITL mean and TTFT p95
against the targets and the certified figures. It appends the records to `drift/<name>.jsonl`
beside the export's directory and pushes them to EFS. Hours already recorded are skipped. Hours
older than Prometheus's 30 days are lost, so it has to run at least that often.

Report the drift and nothing more. `no traffic` is not zero drift, and `too few requests` has no
p95. TTFT p95 comes from vLLM's coarse histogram, so it is a trend. Do not edit the fleet file or
start a pipeline run because of what it says: that is the user's decision.

## Load test

Only when the user asks: it puts a harness Job on the CPU pool for hours. Use `/fleet-kai-load`'s
command, which ensures the harness image and runs

```bash
COLOCATION_HARNESS_IMAGE="$IMG" PYTHONPATH=src python3 -m colocation.kai load <name> <scenario>
```

The scenarios are `load-tests/examples/<instance type>/<fleet>/{as-fleet,demand-drift}.yaml`, one
folder per example fleet. Each states only changes to the export's demand. `load` refuses (exit 2)
when the scenario is for another fleet, names a tenant the fleet does not have, or asks for a
request past a tenant's `max_model_len`, and when any engine is not Ready. On a node of a split
it refuses a fleet scenario for now: the node was exported from its group's fleet file, while the
scenario names the parent and all of its tenants. Otherwise it applies the
Job and returns. The Job drives each tenant through its Service from the next whole clock hour, and
logs a `SOAK HOUR` line per tenant per hour with the client's figures. `drift` after the hours
gives the server's figures for the same hours; report both.
