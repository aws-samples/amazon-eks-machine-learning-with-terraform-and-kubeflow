---
name: run-fleet
description: Run the GPU co-location pipeline end to end on a model fleet file — harness image, plan, demand sweep, profile, plan again, search, read the result, export it — in the right order, and read every refusal rather than working around it. Use when asked to "run this fleet", "find a placement for these models", "co-locate this fleet", or to continue a fleet that is part-way through the pipeline.
---

# Run a model fleet through the co-location pipeline

You are handed one thing: a **fleet file** (`fleets/*.yaml`). Your job is to take it through
the pipeline to either a deployable placement or a well-evidenced answer that it does not fit.

**No agent and no state machine owns this sequence.** The only enforcement in the code is the
fleet-file loader, which refuses to load a fleet for `search` until `profile` has settled every
tenant's TP degree. Everything else — running `plan` at all, running the demand sweep, re-running
`plan` after `profile` — is convention. **You are that convention.** That is why this skill exists.

Each step also has a command of its own, and the command holds the same instructions for that step
alone: `/fleet-preflight`, `/fleet-image`, `/fleet-plan`, `/fleet-sweep`, `/fleet-profile`,
`/fleet-search`, `/fleet-calibrate`, `/fleet-inspect`, `/fleet-export`. This skill is what holds
them in order.

**Serving the result is not part of this sequence.** Deploying an export on KAI GPU fractions is the
`kai-deploy` skill and the `/fleet-kai-deploy`, `/fleet-kai-status`, `/fleet-kai-drift`,
`/fleet-kai-load` and `/fleet-kai-teardown` commands, and the user starts it. Never chain into it from here; say the export is ready for it.

The plugin's hooks refuse a handful of commands outright — a `helm upgrade` of the daemon under a
live fleet, a pattern-kill of the MPS processes, any command that reads the Hugging Face token,
`--no-holder`, an uninstall that does not say what it removes or says a namespace other than the
working one, and a `profile` or `search` whose harness image is not built from the current source.
They put to the user `--reprofile`, `--allow-disjoint-only`, `--relax-targets-pct`, a KAI deploy or
teardown the user did not start, and an edit to a fleet file's `slo:` block, `tolerance_pct` or
`kv_bytes_per_token`. After a `profile` or `search` they report whether the node is still held, that
`plan` must run again after a `profile`, and any search with no export beside it. A refusal or an
ask from a hook is the rule below being applied, not an obstacle to get past.

## Non-negotiables — read before running anything

1. **`search` and `profile` act on the cluster from their first step, so never start one you are
   not going to let finish.** Past the image check, `search` immediately `helm install`s a daemon pod
   requesting every GPU on the node. A killed process leaves that pod Pending, and in a Karpenter
   cluster a Pending pod is a purchase order: that orders a **second** node of the fleet's instance
   type. `profile` captures a node the same way, and each of its solo rounds holds the fleet's GPUs
   for minutes.
2. **The GPU node must always be held.** `manifests/gpu-node-holder.yaml` (a bare Pod at priority
   `-1000`) or a live `mps-daemon` must hold all its GPUs at all times. An unheld GPU node is
   reclaimed in ~600 s and the replacement's GPU UUIDs make every stored placement un-reusable.
   `gpu-node-holder` is a bare Pod — preemption deletes it, so it does **not** survive a run.
   After `--release-node`, confirm the holder is back **Running**, not merely created: it needs every
   GPU, so any tenant pod still up leaves it Pending, and a Pending holder holds nothing and carries
   none of its own `do-not-disrupt` protection either.
3. **Never `helm upgrade mps-daemon` under a live fleet.** It restarts the control daemon, which
   tears down every MPS server and therefore every running tenant.
4. **Never `pkill -f nvidia-cuda-mps-server`.** `-f` matches the `bash -c` wrapper's own command
   line and kills the shell (exit 143). Kill MPS strays by PID; `ps -eo comm` truncates at 15 chars.
5. **Never read, echo, or interpolate the HF token.** It lives at `/home/ubuntu/.hf_token`
   (mode 0600). Pass it to Kubernetes with
   `kubectl create secret generic ... --from-file=token=/home/ubuntu/.hf_token` and reference it
   with `hf_secret_name` / `hf_secret_key`. Never via `helm --set` / `--set-json`.
6. **Uninstalling is yours to do, blind uninstalling is not.** Putting the daemon in and taking it
   out again is ordinary operation of this pipeline, so run the uninstall yourself: name every
   release, pass `-n kubeflow-user-example-com`, and never let a substitution or a glob choose the
   names for you — one expansion removes every release in the namespace, including a lease another
   run is measuring under. An uninstall is never the last command: see 2.
7. **Never write a GPU's memory capacity anywhere** — not into a fleet file, a spec, a constant, a
   default argument, or a comment that a later reader would copy. It is read off a leased node when
   one is captured, and it lives only in the hardware store. The self-test suite scans every
   shipped module for capacity-shaped constants and fails if one reappears, so this is not a style
   preference: a solution that names a number for one card is not a solution for the next one.

## Invocation basics

Everything runs from the repository's `tutorials/multi-model-gpu-sharing` directory, which is where this plugin lives
(`plugins/colocation-pipeline/`):

```bash
PYTHONPATH=src python3 -m colocation.agent <mode> --fleet fleets/<name>.yaml
```

- `PYTHONPATH=src` is required. `--fleet` paths are relative to `tutorials/multi-model-gpu-sharing/`.
- **Where fleet files live.** The fleets shipped with the repo are under
  `fleets/examples/<instance type>/`, one directory per machine, named as the `instance_type:` they
  state. The top of `fleets/` is where a user writes their own. If you are handed a bare file name,
  look in both before saying it is not there.
- Modes: `plan`, `calibrate`, `profile`, `search`, `export`.
- Defaults you rarely change: namespace `kubeflow-user-example-com`, local store
  `~/.colocation/store`, canonical store `/efs/colocation`.
- **`profile` and `search`** need `COLOCATION_HARNESS_IMAGE`; `main()` refuses them without it,
  before the cluster is touched at all. `plan`, `calibrate` and `export` never look at it. Each Bash
  call is a fresh shell, so an `export` in an earlier call is not there any more: set it in the same
  command, from the script in step 0.
- Long-running invocations need an explicit `timeout` above 120000 ms on the Bash call.
- **Engines.** A fleet file may say `engine:`, fleet-wide or per tenant: `vllm-openai` (the default,
  and what most stored rows were taken on), `vllm-triton-inference-server`, or `dynamo-vllm`, which
  needs the Dynamo operator in the cluster. Each tenant is installed from its own engine's chart;
  `--chart` overrides that for a chart under development and is refused for a tenant it does not
  belong to. The engine is part of the key, so rows taken on one do not settle another.
- `/fsx` is readable only from inside a cluster pod. To read EFS or FSx, `kubectl exec` into the
  standing `attach-pvc` pod in `kubeflow-user-example-com` — never build an ad-hoc reader pod.

## The sequence

```
harness image -> plan -> demand sweep -> profile -> plan again -> search -> read -> export
   (step 0)       ^                        |          ^            |
                  +------------------------+          +------------+
```

**The run is not finished at `read`.** The canonical form (step 6) is the only form of the result
that outlives the node it was measured on, and `search` writes it itself as it finishes — the `export`
mode is for re-exporting, not for remembering. Check that it was written, and say where it is.

Work the steps in order. **The back-edges are the common case, not the exception** — treat being
sent backwards as a normal outcome, not a failure.

**Prerequisite A — the hardware itself must be filed, and `plan` refuses rather than assume it.**
There is no table of GPU capacities anywhere in the source, deliberately. Two facts have to exist
before any mode can price anything, and both are observations:

- **how many GPUs a node of this instance type has, and what its cards present** — read off a
  captured node and filed automatically. The profile store is addressed by what the cards present
  (addressable memory, power limit, links), so on a type nobody has ever leased there is no address
  and `plan` stops with `the hardware store holds no shape for '<instance-type>'` (or, given
  `--gpu-count`, `nobody has observed what a <type> presents`). **`plan` cannot bootstrap it.** Run
  `profile` (or `search`) with `--gpu-count N`: it holds one node, files the shape and the device
  facts, gives the node back to the holder, measures nothing and exits 0. Run the same command again
  for the real round. The stated count is checked against the real node and may not contradict a
  filed shape. Needed exactly once per type.
- **how much memory one of those GPUs has** — this is the number the engine multiplies
  `gpu_memory_utilization` by, so every tenant's whole memory budget rests on it. If it is missing,
  `plan` stops with `no environment has measured how much memory a <product> has, and there is no
  default for it`. Do not go looking for a spec-sheet figure: assumed too small it refuses fleets
  that fit, assumed too large it over-subscribes every GPU **while reporting that it had not** — both
  tenants reach Ready and serve, and the KV deficit reads afterwards as co-location interference. It
  is filed by the node capture `profile` and `search` already do: `agent/capture.py` reads the
  driver's own addressable total off the leased node (`memory.total` minus `memory.reserved`, not
  `memory.total`) and files it in the hardware store. So on a cluster whose node type has been run
  once it is already there, and on a new instance type the first GPU round settles it as a side
  effect. Every run after that resolves it offline.

`tools/demand_sweep.py` refuses the same way on a type nobody has leased, with the same remedy.
Never answer either refusal with a number you found: the count comes from the flag, and the total
comes from a leased node or not at all.

**A model needs no prerequisite of its own: the checkpoint states itself.** Its weight bytes and
its KV page size come from `config.json` and the safetensors headers, fetched with range requests so
no weight is downloaded, cached under the commit sha (`store/hubfacts.py`, impl §14.8). That is why a
`models:` entry is an id and nothing else, and why `plan` can price a model nobody has ever run. It
also has to be derived rather than typed: a hybrid-attention checkpoint's page cannot be guessed from
a layer count, and guessing it that way gave Qwen3.5-9B 131,072 B/token against 40,128 measured. Two
consequences for a run:

- **On a hybrid checkpoint the derived page is a slight under-statement** — 1.6–5.9% on the ones
  measured so far, because it prices the per-sequence recurrent state at `max_model_len` and a real
  allocator rounds. Where `plan` puts a fleet within a few percent of a card, say that rather than
  reporting the printed envelope as exact.
- **A measured page overrides it**, per model, as `kv_bytes_per_token` (`agent/fleetfile`). It is the
  one thing about a checkpoint a fleet file may still state, and it belongs there only when a
  measurement produced it.

Note on `--driver`: leave it at the default. It selects how the `search` reasoner is driven — `sdk`
is the Claude Agent SDK (`agent/sdkpacker.py`): real tool calls, and a deny-by-default gate that
makes `measure` the only route to a GPU. `--driver converse` is the JSON-in-text protocol the
self-tests drive; it reaches the same tools under the same refusals, but its parsing is not the one
that ships, so a run driven that way is worth saying so about in the report.

**Prerequisite C — exactly one DCGM exporter may be on the cluster, and it must be ours.** A GPU
allows one DCGM host engine to own its profiling (DCP) fields. The second to ask exits with `error
watching fields: The third-party Profiling module returned an unrecoverable error` and crash-loops, so
every record of the run it was instrumenting comes back with **no `per_gpu` block** — the run reports
success and has lost every cause signal. On a fresh EKS cluster the other engine is the
`amazon-cloudwatch-observability` addon's own dcgm-exporter DaemonSet, which is **on by default**.

`capture`'s pre-flight refuses while another exporter is present, before the install that orders a
node, and prints the fix. `/fleet-preflight` checks it too, and so does this:

```bash
kubectl get pods -A -o wide | grep -i dcgm      # anything outside kubeflow-user-example-com is a conflict
kubectl get ds -A | grep -i dcgm                # a DaemonSet counts even at zero pods: it follows the lease
```

The remedy is the addon's own switch — deleting its DaemonSet or its `DcgmExporter` CR is undone by
`amazon-cloudwatch-agent-operator` within seconds:

```bash
CLUSTER=$(kubectl config current-context | cut -d/ -f2)
REGION=$(kubectl config current-context | cut -d: -f4)
aws eks update-addon --cluster-name $CLUSTER --region $REGION \
  --addon-name amazon-cloudwatch-observability \
  --configuration-values '{"dcgmExporter":{"enabled":false}}' --resolve-conflicts OVERWRITE
```

Container Insights loses its GPU dashboards; logs and node/pod metrics are unaffected. Confirm with
`kubectl get dcgmexporters -A` (none) and then that the lease's exporter is `1/1 Running` with the
seven `DCGM_FI_PROF_*` fields present. **Do this on every new cluster.** The switch lives in the
cluster repo's gitignored `terraform.tfvars`, so a rebuilt cluster brings the addon's exporter back
with no sign of it. Background: `USER-GUIDE.md` §0, "Only one DCGM exporter per GPU".

---

### Step 0 — the harness image (`/fleet-image`; build and push it, do not assume it)

`profile` and `search` are both refused without an image (`__main__.py`, the `not args.harness_image`
check in `main()`; the variable is `COLOCATION_HARNESS_IMAGE`, `measure.HARNESS_IMAGE_ENV`), before the cluster is
touched. Do not assume `COLOCATION_HARNESS_IMAGE` is exported, and **do not trust a tag just because
it pulls.** `colocation-harness:v5` sat in ECR pulling perfectly while predating the per-tenant
tokenizer, so it could not parse what `render.py` writes — a pullable image is not a usable one.

One command does all of it, and it is the one the hooks expect:

```bash
bash plugins/colocation-pipeline/scripts/harness-image.sh --ensure
```

It prints the image reference on stdout and nothing else, so it can be used as `$(...)`. The rest of
this step is what it does and why, which you need in order to read its output.

**The image identity is derived from the source, never chosen by hand.** The tag is a digest of the
build context, so a changed harness gets a new tag automatically and "pullable but stale" cannot
happen:

```bash
TAG=src-$(find Dockerfile requirements.txt harness tests -type f -name '*.py' \
            -o -type f -name 'Dockerfile' -o -type f -name 'requirements.txt' \
          | grep -v __pycache__ | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-12)
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=$(kubectl config current-context | cut -d: -f4)   # the cluster's own region, not a guess
REPO=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/colocation-harness
```

The region is read out of the EKS context ARN (`arn:aws:eks:<region>:<account>:cluster/<name>`, so
field 4) rather than written down, because a region written down is the one thing here that goes
stale silently: the push succeeds against the wrong region's ECR and the failure surfaces much later
as a pod that cannot pull. This cluster is currently in `ap-northeast-2`; do not hard-code that
either.

`__pycache__` is excluded deliberately — there is no `.dockerignore`, so running the harness locally
leaves `.pyc` files inside the build context, and hashing them would move the tag without the source
having changed.

**If that tag is already in ECR, there is nothing to do.** Checking first is what makes the step
idempotent, and costs nothing on every run after the first:

```bash
aws ecr describe-images --repository-name colocation-harness --region $REGION \
  --image-ids imageTag=$TAG >/dev/null 2>&1 && echo PRESENT || echo ABSENT
```

**If absent, build, verify, then push — in that order.** The verification is offline, and doing it
before the push rather than after is the whole point:

```bash
aws ecr create-repository --repository-name colocation-harness --region $REGION 2>/dev/null || true
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

docker build -t colocation-harness:$TAG .
docker run --rm --entrypoint python colocation-harness:$TAG -m tests.selftest   # must pass
docker tag colocation-harness:$TAG $REPO:$TAG
docker push $REPO:$TAG
```

If the in-image selftest fails, **stop and report it. Do not push.** A failing harness image measures
wrongly rather than not at all, and the last time this repo shipped a measuring defect it voided two
complete search results.

Then pass it to the mode **in the same command**, because each Bash call is a fresh shell and an
`export` from an earlier one is gone:

```bash
IMG=$(bash plugins/colocation-pipeline/scripts/harness-image.sh --ensure) || exit 1
COLOCATION_HARNESS_IMAGE="$IMG" PYTHONPATH=src python3 -m colocation.agent <mode> --fleet ...
```

State the account, region and tag in your report. `harness_image` is **not** recorded in the
observation's provenance, so the run's own record will not say which instrument produced it — until
that is fixed, your report is the only place it is written down.

**Building and pushing is authorized; retagging is not.** Never push over an existing tag and never
invent a `vN` tag by hand. The `v5`/`v6`/`v7` tags in ECR are history and stay readable, but a
hand-chosen number is exactly how a stale image gets trusted.

**Check the cluster-specific fields in `containers/colocation-harness/deploy/job.yaml` — and do it
even when step 0 had nothing to build.** They are `metadata.namespace` (and the namespace in the ConfigMap
command in its header), `karpenter.sh/nodepool`, and the `compute-tier` toleration. Each one fails
as a Job that sits **Pending with no event explaining why**. Report anything that looks wrong for
the cluster rather than editing it silently.

**An image already in ECR is not a reason to skip the rest of step 0**, because this check has
nothing to do with the image. Skipping it on that reasoning cost a run on 2026-09-14: the template
still pinned `topology.kubernetes.io/zone: us-west-2b` on a cluster in ap-northeast-2, the driver
could not be scheduled, and `measure` polled a Job that could never start while an engine held two
GPUs. The zone is not in the file at all now — `agent/measure.py` reads it off the node under test —
but the remaining three are typed in by hand, so `/fleet-preflight` prints them and you read them.

### Step 1 — `plan` (`/fleet-plan`; offline, no cluster)

```bash
PYTHONPATH=src python3 -m colocation.agent plan --fleet fleets/<name>.yaml
```

Run this **every time anything changes**. It is not a one-time step.

Read the output in this order, and stop at the first thing that refuses:

| what you see | what it means | what to do |
|---|---|---|
| `the hardware store holds no shape for '<type>'`, or `nobody has observed what a <type> presents` | nobody has leased this instance type, so the store has no address for it | prerequisite A: `plan` cannot fix it, so run `profile --gpu-count N` once, which observes a node and exits, then re-run it. Do not edit the fleet file to work around it |
| `no environment has measured how much memory a <product> has` | the card's memory total has never been measured, and there is no default | on an instance type nobody has leased this is prerequisite A: the first `profile` or `search` files it off the node, so state that and go. On a type that HAS been run, the store or the `--gpu-product` is wrong — **stop** and report it rather than supplying a number |
| `this fleet cannot be placed on this node` (exit 1) | a tenant's KV floor exceeds one GPU at every degree | **stop.** Report which tenant. The fleet file has to change — lower the rate, or drop the tenant |
| `NO CO-RESIDENT PLACEMENT IS FEASIBLE` | no two tenants' minimum envelopes fit on one GPU | **stop**, unless the next line is `BUT NOT ON THE WEIGHTS`. Co-location has nothing to sell this fleet. Report the closest pair and how much it is over by. Do not reach for `--allow-disjoint-only` unless the user says the baseline itself is what they want measured |
| `BUT NOT ON THE WEIGHTS` (after the line above) | some pairs fit on weights alone; only the computed KV, activation and reserve terms rule them out | go on. `search` warns and runs anyway, because those terms have been measured wrong before; profile the pair's tenants first |
| `THE BASELINE ABOVE IS AN UPPER BOUND` | a tenant met its target at the degree shown, but a narrower degree was never measured | note it. It means a later `profile` round can still shrink the baseline |
| `BASELINE NOT ESTABLISHED` | normal on a fresh fleet — every degree is a memory floor | continue to step 2 |
| `<-- FLOOR, not a measured minimum` per tenant | which tenants still need a solo round | this is your profile list |

**Never quote `sum(floor)` as the baseline or compute a saving against it.** `plan` prints it as a
*lower bound* precisely because a floor that fits may still miss its target, in which case the
tenant needs more shards and the baseline goes **up**. A saving quoted against the floor sum is an
overstatement, and this is the single error the whole design is built to prevent.

On a fleet with `hardware.max_nodes` above 1, `plan` also prints `--- splits over up to N nodes ---`:
each tenant on one node, ranked by fewest nodes, then GPUs, then margin, with the first split placed
node by node. The disjoint node count printed above it gates nothing. `this fleet cannot be split over
N node(s): the weights alone rule every split out` (exit 1) is a **stop**. `NO SPLIT OVER N NODE(S) IS
PROVEN INFEASIBLE` is not a proof: profile first. `<tenant> fits no node even alone` names what to
change.

Note also: no node is captured, so the placements are on the *shape* of the instance type. They
name GPU ordinals and look deployable. They are not.

### Step 2 — the demand sweep (`/fleet-sweep`; offline, no cluster)

```bash
python3 tools/demand_sweep.py --fleet fleets/<name>.yaml
```

(It refuses like `plan` on an instance type nobody has leased. `--tenant`, `--rates` and
`--set TENANT.FIELD=VALUE` choose what it sweeps; see `/fleet-sweep`.)

Run it whenever a rate is new or a target looks tight — i.e. before *every* first `profile` of a
fleet. It answers the question `plan` does not: **is the demand written in this file serveable at
all?** Skipping it means learning arithmetic from a measurement round.

Its memory half is exactly as trustworthy as `plan`'s — same `solver/` functions. Its latency half
is a **prediction**, marked `~` in every row, fitted per model in the `FIT` table. Treat it as a
ranking of candidate demands, never as a result. Specifically:

- **`THE BASELINE DOES NOT FIT`** — the disjoint arrangement needs more GPUs than the node has.
  Co-location cannot rescue this. Demand must come down further or a tenant leaves. **Stop and
  report.**
- **`no placement solved` / a tenant listed as unpredicted** — an MoE off its measured degree, or a
  model not in `FIT`. Report it as "unpredictable, not infeasible"; the sweep is declining to guess.
- **`cap N vs M decoding at target`** — the batch cap against the decoders the target implies. The
  cap is derived from the rate, the SLO and the engine's own default (impl §14.9), and it is sized on
  a whole *slot* — prefill queue included — so it cannot come out below this number. There is no
  `max_num_seqs` to set: if a row ever shows the cap losing, the derivation is wrong, so report it
  rather than editing the fleet.
- A row that lands only *without* the co-residency factor (1.27× anchor, 2.04× stacked rider) is a
  predicted co-resident miss. Say so.

**When a demand has to change, change the rate — never the target.** Relaxing an ITL ceiling moves
the goalposts *and* raises the KV floor, which silently changes the memory geometry too. Lowering
the rate lowers both honestly. If a target genuinely has to move, that is the user's call, not
yours: ask.

If you change a fleet file, read the comments around the change before moving on: a rate, a shape or a
target that moves usually makes the sentence next to it wrong, and a fleet file's comments are the
author's reasons rather than a record of arithmetic. Report the geometry `plan` prints — do not write it
back into the file. Then go back to step 1.

### Step 3 — `profile` (`/fleet-profile`; captures a node and deploys engines on it)

**Step 0 must have run** — `profile` is refused without an image, and an image nobody verified is
worse than none. One deploy-and-drive round runs per (tenant, degree) that the store does not already
hold and arithmetic has not settled.

```bash
IMG=$(bash plugins/colocation-pipeline/scripts/harness-image.sh --ensure) || exit 1
COLOCATION_HARNESS_IMAGE="$IMG" PYTHONPATH=src python3 -m colocation.agent profile \
  --fleet fleets/<name>.yaml --budget <n>
```

`--budget` caps the measurements. `--reprofile` re-measures degrees the store already holds a
row for — pass it **only** when the thing that changed is not in the key (a chart change, a model's
weights on disk) or when a deliberate replicate is wanted. The guard hook asks the user first.

What it does: measures each tenant **alone, at the full envelope**, walking degrees upward from the
memory floor and stopping at the first pass. That establishes **minimal TP**, which is what turns
`plan`'s lower bound into a real baseline.

Three outcomes take no round at all and are normal:
- `forced by memory alone` — the floor is already the widest degree the node offers.
- `already profiled` — a usable row exists. This is why profiles amortize across fleets.
- a tenant with no `itl_mean_ms` is skipped: there is no criterion to judge a degree against.

The walk widens on evidence about the degree, and only on that: a measured miss, a degree the
arithmetic would not place, an engine that failed to boot twice at the same degree (one retry is
automatic, because one failed boot cannot tell a too-narrow shard from a fault in the run), or a row
showing a server driven flat out that is too narrow.

**Any other unusable round stops the walk rather than widening it — do not fight this.** An
unusable row (no ITL, `delivery_passed: false`, verdict not `measured`, sampling too thin) did not
say the degree is too slow; it said the degree *was not measured*. Fix whatever broke and run the
same command again: on the way up a degree with no usable row is measured again, with no
`--reprofile`. The exception is a walk *down* from a degree that passed, where a stored unusable row
stops the descent until `--reprofile`; the log says which. Never widen the degree to get past it:
that buys the baseline a wider shard nothing ruled the narrower one out at. `profile` exits 1 while
any tenant is unresolved.

After it finishes:
- If `sum(minimal tp)` now exceeds the node's GPU count, **there is nothing to search** — the
  disjoint arrangement being scored against does not fit. Report and go back to step 2.
- Otherwise **re-run `plan`** (step 1) before searching. The degrees moved, so the baseline and
  every placement moved with them.
- The per-shard MPS reserve the round measured needs no paste. `profile` files it in the profile
  store and the loader merges it into every later load of that model at that degree (impl §14.4), so
  re-running `plan` is what picks it up. Report the number and which envelopes it moved — it changes
  the KV pool and therefore the geometry.

### Step 4 — `search` (`/fleet-search`; installs a daemon holding every GPU, and may order a node)

**State what it will do to the cluster, then run it** — the three confirmations below are yours to
check, not the user's to authorize:

```bash
IMG=$(bash plugins/colocation-pipeline/scripts/harness-image.sh --ensure) || exit 1
COLOCATION_HARNESS_IMAGE="$IMG" PYTHONPATH=src python3 -m colocation.agent search \
  --fleet fleets/<name>.yaml --budget 6 --summary /tmp/<name>-search.json
```

Before running, confirm all three:
- The harness image comes from step 0's script — built from the current source, selftest passed,
  present in ECR. Not merely "set to something that pulls".
- The GPU node is currently held by `gpu-node-holder` (priority `-1000`), so this run's daemon can
  preempt it instead of ordering a second machine. If a *default-priority* pod holds the node, this
  run **cannot** preempt it and will provision a second one — stop and say so.
- `plan` has been re-run since the last `profile` and reports a co-resident placement exists, or
  prints `BUT NOT ON THE WEIGHTS`.

If the fleet-file load refuses here, **read the refusal rather than working around it.** The four
`resolve_tp` refusals all mean the same thing: no usable solo measurement settles a degree. The
remedy is always `profile`, never asserting the degree — the fleet file cannot state a `tp:` at
all, deliberately, and that escape route was deleted because a hand-written `tp: 4` inflated a
baseline once.

If a deploy is refused with `gpu_memory_utilization is X, but ... implies a device total of Y GiB
instead`, or with `GPU <uuid> is promised N GiB of envelope`, that is the render boundary re-checking
the placement against the device this fleet is actually being deployed to. **Report it as a refusal,
never route around it** — the fraction it rejected would have been accepted by the engine and echoed
back unchanged, leaving only an undersized KV pool as evidence.

`--budget` is the number of measurement rounds. `--max-turns` bounds the conversation, which reaches
only the solver and the store. `--relax-targets-pct` is reporting only — nothing is ever
judged against a relaxed target. **Do not pass it to widen anything: it can only lower.** How far a
tenant may be reported past its own ceilings is declared in the fleet file (`slo.tolerance_pct`, 5%
where a tenant does not say), before the run and by whoever owns the requirement; the flag takes the
lower of the two. Pass it only if the user wants the offer tightened — `0` turns the offer off
entirely and still prints the relaxation as a finding.

Keeping the lease costs an idle node; `--release-node` gives it up and risks not getting the machine
back, so default to keeping it and pass the flag when the node is genuinely finished with. Either
way the run prints the exact sequence that hands the node back — **run it whole and do not shorten
it.** It is usually three commands, not two: a search ends with its tenants still serving on
purpose, they are default priority, and the holder is −1000, so uninstalling only the daemon leaves
the holder Pending and the node effectively unheld. That is what happened on both runs on
2026-09-10. If the run could not list its own releases it says so — then check for releases prefixed
with the run token yourself before applying the holder, and confirm the holder reaches Running.

**`capture` can also refuse the node it just got**, with `is marked Drifted on its claim`. That is not
a transient to retry: the holder carries `do-not-disrupt`, so it was the only thing keeping the node,
and the capture removing it is what lets the cluster act on a drift it has been waiting to act on.
Report the refusal, let the replacement node register, confirm the holder is on it, and capture again.
The replacement arrives on a **new AMI**. A different driver version on it changes nothing by itself.
What would is a change in what the node reports about its GPUs — usable memory, enforced power limit,
or how the devices are wired — and then the run stops with `EnvironmentMoved` and asks you to run the
same command again. That is not a fault and needs no flag or edit: the new machine has already been
filed, and the re-run starts a fresh store directory because the old rows were taken on a different
machine. Report it and re-run.

`search` exits 1 when it measured nothing at all.

**A fleet with `hardware.max_nodes` above 1** (impl §17) is never measured as one fleet. `search` splits
it into groups, one per node, and runs this same search on each group's own fleet file under
`<store>/groups/<fleet>/`, with its own lease, `--budget`, summary and export. The reasoner chooses
which group to search and reacts to a group that missed. `--split-chooser drive` uses the deterministic
driver, which is also the reasoner's recorded fallback. `--group-budget` (default 4) caps how many groups
are searched. Leases are taken one node at a time, so add `--teardown --release-node`: the run never
holds two nodes, and every confirmation above applies to each group's lease. A group an earlier run
searched against the same fleet text is recalled and costs no budget. The split is `met` only when
every group of one listed split met, whatever the reasoner submitted. It exits 1 otherwise. Report
the `=== splits ===` block, including any `searched by the fallback` line.

**SM caps.** The reasoner may cap a co-resident tenant's share of the SMs on a measurement
(`sm_caps`, an MPS active-thread percentage from 1 to 99, impl §16). A capped placement is a different
placement from the same one uncapped: it is recalled, calibrated and exported as its own, and the
summary rows and the export carry `active_thread_percentage` for each capped tenant, and the printed
report shows `SM cap <tenant> N%` beside every capped placement it names.

### Step 5 — `calibrate` (`/fleet-calibrate`) and reading the result

```bash
PYTHONPATH=src python3 -m colocation.agent calibrate --fleet fleets/<name>.yaml
```

It reads repeats already in the store and writes the between-run ITL half-width per co-residency
bucket, and it touches no cluster. It exits 1 and writes nothing when no placement has been measured
twice —
which is the normal outcome after a first search, because a repeat is a placement the reasoner
*chose* to re-measure and nothing forces one.

Run it after a search that produced a repeat, and before quoting any delta between two placements
as real. Without it the report says "uncalibrated" in words rather than substituting a floor —
never fill that gap with a number of your own.

If the user wants a calibration and none exists, the honest answer is that it needs one more
`search` round spent on a placement the store already holds.

Then read the result. **The result is what `search` printed at the end of its own run** — the
`BASELINE / STRICT / RELAXED` block, and the same three numbers in the `--summary` JSON. `inspect`
does not reproduce that block, so do not go looking for it there:

```bash
PYTHONPATH=src python3 -m colocation.inspect record <candidate-record-id> --key <validity-key-hash>
```

(`/fleet-inspect` is the same thing.)

`inspect record` prints **one placement's** record, and the id it takes is per-candidate
(`search-rqolmi-001-chat-solo-4gpu`), not the run id — a run id gets you `no record ... locally or on
EFS`. Use it to check a single measurement, not to read the outcome.

`inspect` has no view of the hardware store yet, so filed shapes and device totals can only be read
as raw JSON under `~/.colocation/store/hwfacts/` (and on EFS at `/efs/colocation/hwfacts/`). You do
not need to: `plan`'s header line prints the total it planned against (`N GPUs, X.XX GiB each`), and
on an uncaptured node the load log names the provenance of that figure. Quote those rather than a
number dug out of the store.

Report, in this order, and report **all three** — they are all in GPUs, so no arithmetic is needed to
compare them:

- **`BASELINE`** — `sum(tp)` with no GPU shared, and whether it is an upper bound.
- **`STRICT`** — the fewest-GPU placement on which every tenant met the ceiling it stated, and the GPUs
  that frees. **This is the only certification.** Empty means nothing beat the baseline at the
  ceilings as written; say that plainly rather than reaching for the relaxed number to fill the slot.
  A tenant with no verdict blocks it rather than passing quietly — name that tenant.
- **`RELAXED`** — the fewest-GPU placement that every tenant's own declared tolerance admits, and the
  GPUs *that* frees. Report it whenever there is one, **with the concession in the conceding tenant's
  own units** (`chat` ITL mean ceiling 50.000 -> 50.168 ms, +0.336%) and with `NOT CERTIFIED` beside
  it. This is usually the line with a decision in it — "2 of 6 GPUs free for a third of a percent on
  one tenant" is a question for whoever wrote the ceiling. Omitting it has already forced the user to
  ask for it once, and a certification alone did not answer them.
- the **`vs solo`** block — what sharing cost each tenant against the same tenant alone on whole GPUs.
  Say which placement it is reported against: `STRICT` if there is one, otherwise the relaxed offer.
- any **SM cap** on a reported placement, by tenant: the result holds only with it.

Two things not to do with these numbers. Never compute a saving yourself as `BASELINE - RELAXED` — a
looser ceiling can resolve a tenant to a narrower TP and move the baseline down, and when it does the
report scores against the moved number and prints which it used; quote that. And placements needing
more than a tenant declared appear under their own heading, *not offered* — they are findings, not
results, so report them as how far out of reach a denser placement was.

### Step 6 — the canonical form (`/fleet-export`; offline, and where the result actually goes)

**A `search` exports itself.** Its last act is to write the canonical form, through the same code the
`export` mode runs, so the ordinary case needs no command at all: read the `written to` lines at the
end of the run and report the path. Run the mode yourself when there is nothing to read them from —
an older result, another `--run-token` under the same key, a key whose search predates this, or a run
that printed `the result is stored but was not exported`:

```bash
PYTHONPATH=src python3 -m colocation.agent export --fleet fleets/<name>.yaml
```

It is free — the store and the fleet file, no GPU time, no cluster beyond the mirror write — and it is
what a result has to be handed on in. The stored record addresses GPUs by UUID and ordinal, and both
expire with the node: the UUID names hardware Karpenter reclaims in about 600 s, the ordinal names a
position in one driver's enumeration order. A result handed on in either is a result that silently
addresses the wrong card later. It exports the baseline whether or not a search found anything, so
there is no run it has nothing to say about.

It writes `observations/<validity-key-hash>/exports/<run-token>.yaml`, locally and on EFS, or
`exports/baseline.yaml` for a key with no search. `--export-to <path>` writes a third copy anywhere,
which is what to use when the user wants to see the file. `--run-token` picks an older search under
the same key; the newest is the default, and a token the key does not hold exits 1 listing the ones
it does.

What it re-states, and why:

- **slots instead of devices** — a slot is one device with no identity, carrying the `(tenant, shard)`
  pairs on it. Two residents in one slot shared a card; a tp4 tenant occupies four slots.
- **absolute GiB per shard** — `required_gib` is what a deployment sizes against. The `granted_gib`
  beside it is what that run gave the tenant, which for a tenant alone on a card is the whole card;
  it is measured data about the run and not a recommendation.
- **the conditions by reference** — the fleet file's path and the sha256 of its bytes, plus the
  validity key. Change a rate and the key moves, which is what stops a placement being reused for a
  workload nobody measured it under.
- **all three results and the rejects** — baseline always, `strict`, and `relaxed` (usually `null`,
  with `relaxed_why_not` stating why). The measured-and-missed placements are exported too, with the
  overshoot that ruled each one out: each cost a measurement round, and each rules out an arrangement
  that looks reasonable from the certified answer alone.
- **`active_thread_percentage`** on a capped tenant of a result, and nowhere else — the cap is part
  of what was certified, and a KAI deploy serves it.

**A split is exported by `search`, not by this mode.** On a fleet with `max_nodes` above 1, `export`
refuses. A met split writes `<store>/groups/<fleet>/export.<stamp>.yaml`, schema_version 2, with
each node's group export copied whole under `nodes:`, plus `split.<stamp>.json` beside it. Report
that path, the node count, and `devices` against `baseline_devices`. A split that is not met has no
split export, only its groups' own.

If it refuses, read the refusal — `export` is under the same loader rule as `search`, so an unsettled
TP degree stops it and the message names `profile`. Do not work around it.

Report the path, the three device counts, and whether `relaxed` is null. The YAML is the artifact; do
not paraphrase it back in prose, and never quote a number from it that `search` did not produce.

---

## When to stop and ask rather than proceed

- A failing in-image selftest at step 0 — report it and stop. Do not push, and do not fall back to an
  older tag that happens to pull.
- Any change to a `slo:` block in a fleet file, `tolerance_pct` included — a target decides what the
  run certifies and a tolerance decides what the report will offer, and neither is yours to move.
- `--allow-disjoint-only`, `--reprofile`, `--relax-targets-pct` (the hook asks for each).
  `--no-holder` is not on this list because the hook refuses it outright.
- Anything that would leave the GPU node unheld with nothing lined up to take it back.
- Stating a `kv_bytes_per_token` for a model: it overrides what the checkpoint says about itself.
- When arithmetic says the fleet cannot work: report it and stop. Do not quietly relax a target to
  produce a result.

## What to report at the end

State plainly which steps ran, which refused and why, what the numbers were, and what was **not**
done. Name the file step 6 wrote, since that is the form of the result anything downstream should be
given. If a step was skipped or blocked, say so explicitly rather than letting the summary imply the
pipeline completed. A directional result on one fleet is the expected outcome — the method is the
deliverable, not closure on a single run.

## Reference

- `USER-GUIDE.md` — the full guide; §7 is the fleet-file reference, §8–§13 are the modes, §14 is
  serving on KAI.
- `skills/kai-deploy/SKILL.md` in this plugin — serving an export, which this skill never does.
- `fleets/examples/<instance type>/*.yaml` — each shipped fleet file's header comment says what it is for.
- `USER-GUIDE.md` §8 — the baseline rule every refusal above traces back to.
- `USER-GUIDE.md` §15 — what this plugin installs, and which rules its hooks hold.
- `PYTHONPATH=src python3 -m colocation.tests.selftest` — run after any code change.
