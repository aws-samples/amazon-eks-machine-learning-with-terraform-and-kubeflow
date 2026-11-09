# User guide — running this on your own cluster

Hand it a **model fleet** and a **GPU instance type**, and it
searches for a placement that co-locates the fleet on one node — or, with `hardware.max_nodes`, splits
it over a few nodes of that type, each tenant on one of them — while meeting every tenant's stated
latency target. A reasoner drives the search, a CP-SAT solver enumerates memory-feasible
placements, and every candidate it picks is *measured under load* rather than predicted. The output
is a placement you can deploy, or a well-evidenced answer that the fleet does not fit.

It is a research harness, not a product. It touches a live cluster, provisions GPU nodes, and
costs real money per measurement. Read [What it will do to your cluster](#what-it-will-do-to-your-cluster)
before the first `search`.

---

## The sequence, and what each step costs

The section numbering below is a reading order, not a run order. Two of the six modes spend GPU time
and four are free, and the free ones are not preliminaries you graduate from — you return to them.

| step | cost | when you run it |
|---|---|---|
| write the fleet file (§7) | free | first, and again whenever a measurement changes your mind |
| `plan` (§8) | free, offline | **every time anything changes.** Not a one-time step |
| `tools/demand_sweep.py` (§8) | free, offline | before spending, whenever a rate is new or a target looks tight |
| `profile` (§10) | **GPU** | first of the two GPU modes; once per (tenant, serving config, hardware) |
| `calibrate` (§9) | free | only says anything once repeats exist — so in practice, before trusting a search's deltas |
| `search` (§11) | **GPU** | after `profile` has established the baseline |
| read the result (§12) | free | — |
| `export` (§13) | free, offline | at the end of every run — and `search` does it for you |
| serve it on KAI (§14) | **GPU**, optional | only when you decide to serve a result; nothing runs it for you |

**`plan` and `profile` do not depend on each other in the way the numbering suggests.** `plan` runs
happily on a fleet with nothing profiled — that is what it is for — and `profile` never reads `plan`'s
output. What `profile` does is change what `plan` can *say*: before it, degrees are memory floors and
the baseline is a lower bound; after it, both are measured. So the real shape is:

```
fleet file -> plan -> demand sweep -> profile -> plan again -> search -> read -> export
                ^                        |          ^            |
                +------------------------+          +------------+
                  a floor that does not fit          a search whose baseline
                  sends you back to the file         moved re-plans first
```

**And the back-edges are the common case, not the exception.** `profile` can invalidate the fleet
rather than feed `search` — if `sum(minimal tp)` exceeds the node's GPU count, the disjoint
arrangement being scored against does not fit and there is nothing to search. Any change to a rate,
a sequence shape, or a serving-config field (`dtype`, `max_model_len`, `extra_args`, and the derived
`max_num_seqs` that follows from the rate and the target) re-keys the profile rows, so it sends you
back through `profile`.

**Nothing in the code enforces this order except one thing**, and it is worth knowing which: the
fleet-file loader refuses to load a fleet for `search` until `profile` has settled every tenant's TP
degree. That protects the baseline. It does not make you run `plan`, or the demand sweep, or re-run
`plan` after `profile` — those are convention. If you would rather not be the convention, hand the
whole thing to Claude Code: **[§15](#15-driving-all-of-this-with-claude-code)**.

## 0. What this assumes

**It is built on the platform in this repository.** It runs on the EKS cluster this repository
builds, and it depends on that platform's conventions by name, in code, not merely by resemblance.
Build the cluster first and get a model serving normally on it before coming back here. Specifically, this
code expects:

| What | Named as | Used for |
|---|---|---|
| Karpenter GPU nodepool | `cuda` | `search` installs a daemon requesting every GPU on one node; this selects the pool it comes from (`agent/capture.py`) |
| Karpenter CPU nodepool | `cpu-compute`, taint `compute-tier=heavy-cpu:NoSchedule` | the load driver runs here, never on the node under test (`containers/colocation-harness/deploy/job.yaml`) |
| Namespace | `kubeflow-user-example-com` | everything is installed here |
| EFS claim | `pv-efs`, mounted `/efs` | the stores: `/efs/colocation/observations/<validity-key>/`, `/efs/colocation/profiles/<profile-key>/` and `/efs/colocation/hwfacts/` |
| FSx for Lustre claim | `pv-fsx`, mounted `/fsx` | model weights, read by the engines |
| Weight layout | `/fsx/pretrained-models/<org>/<model>` | what the platform's `hf-snapshot` chart writes (`SNAPSHOT_ROOT`) |
| Utility pod | `attach-pvc`, from `eks-cluster/utils/attach-pvc.yaml` | **the only way the agent reaches EFS.** It runs off-cluster, so every store read and write is a `kubectl exec` into this pod (`agent/mirror.py`) |
| Service mesh | istio, `istio-injection=enabled` | why every pod template here carries `sidecar.istio.io/inject: "false"` — an Envoy hop costs ~0.4 ms of TTFT, measured |

Beyond the platform you need:

- **Your fleet's GPU instance type available in some AZ** (`g6e.48xlarge` in our own fleets). The FSx filesystem does not have to be in that AZ:
  Lustre is mounted by IP inside the VPC, so single-AZ describes where the data lives and its
  failure domain, not who may mount it. Measured on 2026-09-14 — an FS in `ap-northeast-2a`, both
  the engines and the load driver on nodes in `ap-northeast-2b`, weights and tokenizer read from it.
  The load driver is pinned to the **leased node's** zone, which the agent reads off the node, so
  the driver and the engines it drives always share an AZ and no AZ is written down anywhere.
- **`kubectl` and `helm` on PATH**, authenticated to the cluster. The agent shells out to both.
- **Python 3.11+** with `ortools`, `pyyaml`, `boto3`, and `claude-agent-sdk`. Developed and run on
  3.14. There is no `pyproject.toml` for the agent — it is run as a module from `tutorials/multi-model-gpu-sharing/src`.
- **Bedrock access** to a Claude model in your region (§5).
- **An OCI registry your cluster can pull from**, for the harness image (§3).
- **Only to serve a result on KAI (§14):** the platform's KAI option (`kai_enabled`), which adds
  the `kai-gpu` node pool, the KAI scheduler and the `colocation-serving` queue. The search does
  not need it.

### Hardware actually supported

**There is no table of GPUs, and no capacity constant anywhere in this repo.** `agent/hardware.py`
resolves an instance type out of the *hardware store* (`hwfacts/` under `--store`, shared through EFS
like every other record), and what it needs is two things: the **shape** of the instance type — GPU
count and GPU product — and the **memory total of one device**. Both are recorded from real nodes:

- **Shape.** Filed the first time a node of that type is leased, naming the node it was read off.
  Before that first lease there is nothing to read, so a type nobody has leased has to be
  introduced once, by `profile` or `search` with `--gpu-count N`. That run leases one node, files
  what it presents (shape, usable memory, power limit, how the GPUs are wired), hands the node back
  to the holder, measures nothing and exits `0`; run the same command again and it measures. `plan`
  and the demand sweep cannot do this even with `--gpu-count`, because what they price against is
  the observed device, not the count: they refuse until a node of the type has been observed. The
  stated count is checked against the node and may not contradict a shape already filed.
- **Total.** `capture` reads `nvidia-smi --query-gpu=memory.total` on the leased node. That is the
  driver's word, and it is now the only word: a second source used to exist — boot one engine at two
  `gpu_memory_utilization` values and take the slope of the KV pool the engine reports — and it was
  deleted with everything else that read a server's account of itself
  (impl §14.4). A store that already holds a slope-sourced total still answers
  from it, labelled as such; nothing produces a new one. Until something has filed a total, anything
  that needs one refuses by name rather than substituting a guess.

Totals are kept per environment — the card, its usable memory, its power limit, how the GPUs are
wired, and the engine image (§7). If two environments disagree by more than their own error bars, the
store refuses to answer instead of picking one. So supporting a new GPU means one `profile` or `search` with
`--gpu-count N` to observe a node, then the same command again — not editing any code.

**And the total is re-checked at the moment of deployment, not just at planning time.** Every
`gpu_memory_utilization` a run deploys is a fraction of this number, and vLLM multiplies it by
whatever the driver reports on the machine it lands on — so a fraction computed against any other
figure is memory the engine takes and the packing never priced. Just before `helm`, the agent
recomputes each fraction from the envelope it is about to enforce and the total the store measured,
and refuses the deployment if they differ, printing the total the fraction *implies*. Nothing later
can catch this: the engine reports back the utilisation it was handed, so the run reads as healthy
and only the KV pool is the wrong size.

### Only one DCGM exporter per GPU

**A GPU allows exactly one DCGM host engine to own its profiling fields.** The second exporter to ask
for them fails with

```
error watching fields: The third-party Profiling module returned an unrecoverable error
```

and crash-loops. Nothing else on the node notices: the run keeps measuring latency and writes records
with **no `per_gpu` block at all**, so it reports success with half its evidence missing. Every run
deploys its own exporter alongside the MPS daemon, pinned to the leased node, so **no other DCGM
exporter may be on the cluster**. On this platform the usual one is the
`amazon-cloudwatch-observability` addon's, which the cluster's Terraform turns on by default
(`cloudwatch_dcgm_exporter_enabled`). Turning it off costs the Container Insights GPU dashboards and
nothing else the addon collects.

`capture` refuses before leasing a node if it finds another exporter, and prints the command that
turns the addon's copy off. Use the addon's own switch rather than `kubectl delete`: the exporter is
reconciled from a `DcgmExporter` custom resource, and deleting the DaemonSet or the resource gets
both back within seconds. `/fleet-preflight` (§15) checks this too.

---

## 1. What in here is this cluster's, and not yours

Everything below was ours and has to become yours. Most is a flag or a YAML field; **two are
constants in code with no flag**, which is called out where it applies.

| Value | Ours | Where you change it |
|---|---|---|
| Harness image | *(no default — deliberately)* | `$COLOCATION_HARNESS_IMAGE` or `--harness-image` (§3) |
| Namespace | `kubeflow-user-example-com` | `--namespace`; also `metadata.namespace` in `deploy/job.yaml` |
| EFS store root | `/efs/colocation` | `--efs-store`; default in `agent/render.py` |
| FSx claim / mount | `pv-fsx` → `/fsx` | `storage.*` in the `vllm-openai` chart values |
| Model paths | `/fsx/pretrained-models/Qwen/...` | `model_path` per tenant in your fleet file |
| Load-driver AZ | *(none — the leased node's, read off the node)* | **code only**, `agent/measure.py`. No flag, and nothing to edit |
| CPU nodepool + taint | `cpu-compute`, `compute-tier=heavy-cpu` | `deploy/job.yaml` |
| Bedrock region | `us-west-2` | `DEFAULT_REGION` in `agent/reason.py` |
| Reasoner model | `us.anthropic.claude-opus-5` | `--reasoner-model` |
| NVIDIA driver version | `580.178.04` | *nothing to change.* The agent reads it off the node and records it alongside each measurement (§7) |
| GPU nodepool | `cuda` | **code only**, `agent/capture.py`. No flag. |
| Attach pod name | `attach-pvc` | **code only**, `DEFAULT_ATTACH_POD` in `agent/mirror.py` and `agent/measure.py`. No flag. |

Two things that look substitutable and are not:

- **GPU UUIDs are never written by hand.** They are read from the `<lease>-gpu-uuids-<node>`
  ConfigMap the daemon publishes. A mis-keyed MPS memory limit references nothing and is *silently
  ignored* — the tenant then runs with no envelope at all and the record looks healthy. This is the
  one error on this stack with no symptom.
- **Placements are not portable.** The GPU-ordinal→UUID mapping belongs to one physical machine, so
  a placement names that machine. `placement.node`/`gpu_indices`/`gpu_uuids` in the chart values are
  intentionally empty and the chart **fails the render** if you install it by hand without them.

---

## 2. Verify the platform before spending anything

```bash
NS=kubeflow-user-example-com

kubectl get ns $NS
kubectl get pvc -n $NS                       # expect a bound pv-fsx (and your EFS claim)
kubectl get pod attach-pvc -n $NS            # must be Running; the store is unreachable without it
kubectl get nodepool                         # expect `cuda` and `cpu-compute`
kubectl exec -n $NS attach-pvc -c attach-pvc -- ls /efs /fsx
```

The `attach-pvc` pod is a standing pod on purpose. Keep it running; do not build ad-hoc reader
pods — every store operation in this codebase goes through it.

---

## 3. Build and push the harness image

The harness is the instrument: it drives load at each tenant's stated rate and shape, measures ITL
and TTFT per tenant, scrapes the run's own DCGM exporter, and writes one observation record to EFS.
It runs as a Job on a CPU node.

**There is no default image, and that is the fix for a real defect.** This used to default to a
private ECR repository in one AWS account. Anyone else got the pull refused — and an
`ImagePullBackOff` against a private ECR repo reports `no basic auth credentials`, not "this is not
your registry", so the cause is invisible in the pod's events. `search` now refuses before touching
the cluster if no image is set.

```bash
cd tutorials/multi-model-gpu-sharing/containers/colocation-harness

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-west-2                       # yours
TAG=v6

aws ecr create-repository --repository-name colocation-harness --region $REGION || true
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

docker build -t colocation-harness:$TAG .
docker tag colocation-harness:$TAG $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/colocation-harness:$TAG
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/colocation-harness:$TAG

export COLOCATION_HARNESS_IMAGE=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/colocation-harness:$TAG
```

Then edit `deploy/job.yaml` for your cluster — its header lists exactly which fields, and each one
of them fails as a Job that sits **Pending with no event explaining why**, which is the most
expensive way to learn it:

- `metadata.namespace`
- `karpenter.sh/nodepool` and the `compute-tier` toleration, if your CPU pool differs

There is no zone to set. The template carried a literal `us-west-2b` until 2026-09-14, and the
first time the pipeline ran on a cluster in another region the driver could not be scheduled at
all — a Job Pending for six minutes with `no instance type met the scheduling requirements`, an
engine loaded behind it, and the GPUs billing the whole time. The zone is now read off the node
under test.

Sanity-check the image without a cluster:

```bash
docker run --rm --entrypoint python $COLOCATION_HARNESS_IMAGE -m tests.selftest
```

**`TAG=v6` above is a hand-chosen number, and that is the weak part of these instructions** — a tag
says nothing about what is inside it, which is how `colocation-harness:v5` came to sit in ECR pulling
perfectly while being unable to parse what `render.py` writes. The `colocation-pipeline` plugin
([§15](#15-driving-all-of-this-with-claude-code)) does all of the above for you and tags the image
with a digest of the build context instead, so a changed harness always gets a new tag and a stale
one can never be mistaken for current.

---

## 4. Stage the weights on FSx

Each tenant reads its weights from a local path, not from HuggingFace at runtime. Use the
platform's `hf-snapshot` chart, which writes `/fsx/pretrained-models/<org>/<model>`.

Your HF token goes in as a **Kubernetes secret from a file** — never `helm --set`, which bakes it
into the release secret and your shell history:

```bash
kubectl create secret generic hf-token -n $NS \
  --from-file=token=$HOME/.hf_token --dry-run=client -o yaml | kubectl apply -f -
```

and reference it by `hf_secret_name`/`hf_secret_key`. Confirm the layout before running anything —
a wrong `model_path` costs a node lease to discover:

```bash
kubectl exec -n $NS attach-pvc -c attach-pvc -- ls /fsx/pretrained-models/Qwen/Qwen3-14B
```

First read of a file on FSx for Lustre is an HSM pull from S3 and can dominate startup. That is why
the chart's readiness probe allows a 30-minute ceiling.

---

## 5. Give the reasoner a model

The search is driven by a Claude model over Bedrock. Default is the cross-region inference profile
`us.anthropic.claude-opus-5` in `us-west-2`; `--reasoner-model` overrides it, and `DEFAULT_REGION`
in `agent/reason.py` is the region.

Whatever principal runs the agent needs `bedrock:InvokeModel` and
`bedrock:InvokeModelWithResponseStream` on that profile. The default `--driver sdk` runs the Claude
Agent SDK with `CLAUDE_CODE_USE_BEDROCK=1`; `--driver converse` uses `boto3` against the Converse
API instead. The two differ in transport only — same six tools, same invariants, same derived
verdict — so a record does not say which drove it beyond naming the model.

Quick check:

```bash
aws bedrock-runtime converse --region us-west-2 \
  --model-id us.anthropic.claude-opus-5 \
  --messages '[{"role":"user","content":[{"text":"ok"}]}]' >/dev/null && echo "bedrock reachable"
```

---

## 6. Install the agent's dependencies

```bash
pip install ortools pyyaml boto3 claude-agent-sdk
cd tutorials/multi-model-gpu-sharing/src
python -m colocation.tests.selftest          # no GPU; changes nothing on any cluster
```

Every command below runs from `tutorials/multi-model-gpu-sharing/src`.

---

## 7. Write a fleet file for your fleet

This is the only file you author, and it decides whether the search has anything to find. Read
[What `plan` is, conceptually](#what-plan-is-conceptually) in §8 first — it is the
premise the rest of this depends on. The short form, because there is one choice here you can get
badly wrong, and one that used to be a choice and is now refused for the same reason:

**State each `slo:` from what your application needs, not from what the model benchmarks at.**
Co-location spends latency headroom to reclaim GPU memory. A target set to the model's own
dedicated-GPU number leaves no headroom, so the only placement that can pass is one GPU per
tenant, and the search will spend an hour of measured runs confirming that. A target that reflects
a real requirement — a batch summarizer does not need 14 ms per token — is what gives it something
to trade.

**You do not set `tp`, and since 2026-09-08 you may not.** A file that states a shard count is
refused, with the reason in the message. Three reasons, and they are the same reason from three
sides. A degree is a *latency* claim — `tp: 4` asserts that tp2 would miss this tenant's target — and
a latency claim is the one thing nothing else in this file can be checked against: the weight bytes
and the KV geometry are derived from the checkpoint and logged on every load, and the rate and shape
are checkable against what the harness actually sent. It is also the number your whole result is scored
against, since the baseline is `sum(tp)` with no GPU shared, so a degree one wider than the target
needs hands the search a GPU it never freed — silently, and in the flattering direction. And checking
a degree you had stated would cost exactly the measurements that computing it costs: to show `tp: 4`
is minimal, somebody has to show tp2 misses, which is a solo run at tp2. Accepting the line and
checking it buys no GPU time, only a code path where the check can later be skipped.

**The degree is resolved from measurement instead** (§10). Each tenant gets the smallest degree whose
*solo* run met that tenant's own `slo.itl_mean_ms`, read out of the profile store. What does not
happen anywhere is a guess: with nothing profiled, `search` refuses the fleet outright and names the
degrees to measure, while `plan` and `profile` — the two modes whose job is that measurement — load it
at the **memory floors**, label every degree as a floor and print `BASELINE NOT ESTABLISHED` in place
of a number to beat. Two things follow that are worth knowing before you write the file. A tenant
needs `slo.itl_mean_ms` or it has no target to resolve a degree against, and the loader refuses rather
than inventing one. And the store is keyed by *environment* rather than by fleet, so rows an earlier
fleet paid for on the same hardware and the same engine image are already there for yours. The
serving stack is part of that identity: changing a fleet's `engine:` line does not re-use the rows it
had before, and it is not meant to — whether the server matters is the question such a change asks.

If a degree is fixed for some reason outside the model, that is a placement constraint and not a
shard count. Nothing here implements one, and an inflated baseline is not a way to express it.

Also worth knowing before you spend anything: if `sum(tp)` is less than or equal to the GPUs on
your node, disjoint placement already fits on one machine, and co-location's win is idle GPUs
inside it rather than a machine you never provision. That is still a win — it is just a smaller
headline, and `plan` will tell you which case you are in for free.

**Write your own file at the top of `fleets/`, which ships empty for exactly that.** The fleets shipped
with the repo sit under `fleets/examples/<instance type>/` — one directory per machine, named as the
`instance_type:` they state — and they describe workloads that exist rather than templates. Read them
for how a fleet argues its own choices; the annotated skeleton below is what you copy. Each
shipped fleet file opens with a header comment saying what it is for.

Four blocks, and one optional line above them:

```yaml
hardware:
  instance_type: g6e.48xlarge     # the only thing stated about the machine
  # max_nodes: 2                  # optional, a whole number, default 1: how many nodes of that type
  #   the fleet may be split over, each tenant on exactly one of them. Above 1, `plan` lists the
  #   splits and `search` searches them group by group (§11). Nothing else belongs under hardware.

# engine: vllm-openai             # optional, fleet-wide, and overridable per tenant. Which
#   serving stack the tenants are served by. Omitted means vLLM's own OpenAI server, which is
#   what every measurement in the store was taken on. The other names available today are
#   `vllm-triton-inference-server` — Triton Inference Server with the vLLM backend, driven
#   through Triton's OpenAI-compatible frontend — and `dynamo-vllm` — NVIDIA Dynamo with the
#   vLLM backend, one frontend and one aggregated worker per tenant, which needs the Dynamo
#   operator installed in the cluster. A name no adapter is registered for is refused
#   while the file is read, with the legal ones listed, rather than at `helm install`.

workload:
  delivered_fraction_threshold: 0.99   # the fleet-wide delivery gate, and nothing else

models:                           # one entry per model, keyed by a short local name
  qwen3-14b:
    model_id: Qwen/Qwen3-14B      # and that is all a model entry states
    # model_revision: main        # optional. Resolved to a commit sha and cached under it,
    #   so a moved tag is a different row rather than a quietly different fleet.
    #
    # Nothing else about a checkpoint is stated or may be. The weight bytes come from the
    # safetensors headers (fetched by HTTP range request, no weight downloaded) and the KV
    # geometry from config.json, both keyed by the commit sha; every mode logs the
    # derivation. `weight_gib`, a `kv:` block and `activation_gib` were fields until
    # impl §14.8 and are now refused, with the derivation named in the message.
    #
    # kv_bytes_per_token: 163840  # the one override left, for a family the derivation does
    #   not model. Logged against the derived page every time the file loads.
    # The activation term takes a documented 1.0 GiB/shard fallback — nothing outside an
    # engine's own utilisation budget can see it — and the per-shard MPS reserve is merged
    # from the profile store where `profile` filed it, or falls back to 1.0/1.5 GiB.

tenants:                          # one per served endpoint
  - name: chat
    model: qwen3-14b              # must match a key in models:
    served_name: qwen3-14b-chat
    model_path: /fsx/pretrained-models/Qwen/Qwen3-14B
    # engine: vllm-openai         # optional, and takes precedence over the fleet's line. State it
    #   when one tenant needs something the fleet's stack cannot do — a tenant that sends images,
    #   served by a stack whose frontend accepts text only, is refused while the file is read. A
    #   fleet with two stacks in it is a normal arrangement, not a workaround.
    # no tp: — there is no line for it. It is resolved from the profile store, and
    # writing one is refused rather than honoured.
    rate_hz: 3.0                  # offered request rate
    isl_tokens: 1024              # required, never defaulted
    osl_tokens: 256               # required, never defaulted
    max_model_len: 4096
    # no min_kv_tokens and no max_num_seqs — both were fields until impl §14.9 and both are
    # refused now. Each was arithmetic on the three lines above it: the pool floor is the
    # larger of what this rate and target need at the arrival burst and two full contexts,
    # and the batch cap is the burst occupancy of a SLOT — rate x (ttft_p95 + osl x itl) —
    # floored at the serving engine's own default. `plan` prints both with their derivations.
    # burst_quantile: 0.99        # optional, and the only new input either derivation takes:
    #   the arrival quantile the KV pool is sized for. Omitted means 0.99. Arrivals at a
    #   stated mean rate are not evenly spaced, and a pool sized for the mean is short about
    #   half the time — a short pool preempts, and the engine serves a preemption as a worse
    #   ITL, so the tenant misses its target for a reason nothing in its record names.
    # peak_rate_hz: 3.0           # optional, mutually exclusive with burst_quantile. State
    #   it when you know your peak: the pool is then sized for rate_hz x (peak/mean) in
    #   flight instead of for a Poisson quantile.
    slo:
      itl_mean_ms: 40.0
      ttft_p95_ms: 1000.0
      # tolerance_pct: 1.0          # optional. How far past the two ceilings above this tenant
      #   is willing to be *reported* as a RELAXED reading (§12). Omitted means 5%. `0` means
      #   this ceiling does not move, and no reading will ever offer to move it.
      #   It is not a target: STRICT is always computed against the ceilings as written, and
      #   nothing is ever judged against a relaxed one. Declaring it here rather than passing
      #   `--relax-targets-pct` at the end is the whole point -- a concession stated in advance
      #   by whoever owns the requirement is a policy; the same concession discovered after the
      #   run, by reading how far the measurement missed, is a rationalisation.
```

That is the whole file. There is no section describing the hardware or the software versions, because
nothing in a fleet file needs to: see below.

**What the loader refuses, and why it matters when you write your first file:**

- No `models:` or no `tenants:` — refused.
- A tenant with a `tp:` line — refused, per above, and the message says what to do instead. Deleting
  the line is not the whole fix: the tenant then needs `slo.itl_mean_ms`, and the degrees need
  measuring with `profile`.
- A tenant without `isl_tokens` or `osl_tokens` — refused rather than defaulted. A shape the loader
  invented would be reported back as if you had asked for it, and it decides both the KV floor and
  what the harness actually sends.
- `isl_tokens` or `tokenizer_path` under `workload:` — refused. The first would drive every tenant
  at one shape; the second would build every tenant's prompts with one model's tokenizer.
- One engine served from two different `image:` values — refused. Each tenant's engine is in the
  key, so one image per engine is what makes a tenant's image recoverable from the key; two images
  behind one engine is a difference the key records the existence of and cannot attribute. Two
  *engines* with an image each is fine, and the key names both.
- **`weight_gib`, a `kv:` block, `activation_gib`, `min_kv_tokens`, `max_num_seqs` — all refused**,
  each with its derivation named. The first three were facts about a checkpoint a person transcribed
  (impl §14.8); the last two were arithmetic on the rate, the shape and the target in the same tenant
  block (impl §14.9). The refusal is the point: a field that is still accepted "for compatibility" is
  a field the next engine's adapter has to be told to honour.
  **This replaced a refusal that used to live here**, and the history is worth keeping. A hand-written
  cap that could not admit its own SLO — `rate_hz x osl_tokens x itl_mean_ms/1000` resident requests
  against `max_num_seqs` — surfaced as a *placement* being reported infeasible, on 2026-09-09, with
  142 requests waiting and 0 completed. A derived cap cannot contradict the SLO it is derived from, so
  what the check guarded against is now unreachable rather than caught. What a cap still has to avoid
  is binding *at* the target — a cap of N stops being satisfiable exactly at `ITL = N/(rate x osl)`,
  which turns a co-resident miss into a diverging queue — and that is why the derivation counts slots
  (a request holds one through all of its prefill) and puts the tenant's own TTFT ceiling inside the
  occupancy, rather than counting decoders.
- A `validity_key:` block of any kind — refused. Everything it used to hold now comes from somewhere
  other than a person, which the next section explains. The refusal names each field and where it
  comes from now, so you can delete the block and move on.
- A node, a GPU assignment, a memory envelope, or a GPU UUID — not expressible. If a fleet file
  could pin those it would stop being a search problem.

### Where measurements are filed, and why your store starts empty

Every measurement is filed under a directory named by a hash of the things that have to match before
two measurements can be compared. Those things come in two halves.

**The machine.** Not a name for the machine — the facts it reports about itself:

| Fact | Read from |
|---|---|
| Which GPU, and how many | the node's labels, or `nvidia-smi` if there are none |
| How much memory one GPU can actually use | `memory.total` minus `memory.reserved`, in whole MiB |
| The power limit being enforced | `enforced.power.limit`, in whole watts |
| How the GPUs are wired to each other | `nvidia-smi topo -m`, stored as a short hash of the table |
| The engine container image | your fleet file's `image:` |
| How the GPUs are shared | today always MPS, since that is what the agent installs |
| Which definitions of ITL and TTFT are in force | a constant in `store/environment.py` |

**The workload.** Per tenant: the TP degree, the request shape, the offered rate, the engine flags,
and the model revision.

You do not type any of this. The machine half is read off the node the run leases; the workload half
is read off the tenant blocks you already wrote. Nothing here is a value you can get wrong, and
nothing here is a value you can keep up to date by editing a file.

**Your store starts empty, and that is fine.** Your node will almost certainly report something
different from ours — a different card, a different amount of usable memory, a different power limit —
so your directory name will differ from ours. Nothing is lost by that: the solver sizes KV pools from
each tenant's own stated target, not from anything we measured, so a brand-new store plans and
searches on day one.

**Why these facts and not a version string.** The store used to be keyed partly by the NVIDIA driver
version, typed into the fleet file by hand. That was wrong in both directions. A driver upgrade that
changed no measurable thing would still give you a new, empty directory — and an empty directory looks
exactly like a fresh start, so you would lose your history with nothing telling you. Meanwhile a
power cap, a clock limit, or a change in how much memory the driver reserves would move every number
you measure while the version string sat still. The facts above move when the measurements move.

The driver version is still recorded, on every row, so you can see months later which driver a number
was taken under. It just does not decide where the number is filed, so upgrading it costs you only
whatever it actually changed.

**One consequence worth knowing.** Two different instance types that present the same GPUs, wired the
same way, with the same usable memory and power limit, now share one directory. They used to need two
sets of measurements. The instance type is still recorded on each row, so you can always tell which
machine a number came from.

---

## 8. `plan` — free, offline, no cluster

```bash
python -m colocation.agent plan --fleet ../fleets/my-fleet.yaml
```

`plan` (and `export`) first bring the local store up to date with the shared store on EFS, through
the `attach-pvc` pod. Pass `--local-only` to skip that: no kubectl command is run and `--store` is
all the run sees. That is how both run on a laptop with no cluster, and how a store that must stay
apart from the real one (TUTORIAL.md's synthetic store) is kept apart. The modes that measure or
calibrate refuse the flag.

Solves against the *shape* of the instance type — GPU count, and the device memory total the
hardware store measured for that GPU —
and names no node. It reports which placements are memory-feasible and what each tenant's stated
target requires of its KV pool. Nothing can be deployed from what it loads: the fleet it builds has
no GPU UUIDs and refuses to invent one.

Run this until the output makes sense. It costs nothing, and it is where an impossible fleet shows up
— along with every derivation it made: the weight sum and what it excluded, each tenant's page, its
KV floor and its batch cap.

**With `max_nodes` above 1**, `plan` also prints how many nodes the disjoint baseline would need,
and then up to `--top-k` splits — which tenants share a node — ranked by fewest nodes, then fewest
GPUs, then the largest smallest KV margin, with the first split placed node by node. Every tenant is
at its KV floor here, so tenants may share GPUs: a fleet whose disjoint placement needs more nodes
than `max_nodes` is still split, if sharing makes it fit. As everywhere else (impl §14.2),
"no split" is called proven only when the weights alone rule one out; otherwise
it says none was found and that this is not a proof.

### What `plan` is, conceptually

**A memory bin-packing problem with latency-derived floors.** That is the whole of it, and the two
things people assume it also is are worth ruling out first: it has no model of compute, and the
placements it prints are the **co-located** ones, not the solo ones.

The solo arrangement — one shard per GPU, no sharing — is in the same output, but as the *baseline*
the packing gets scored against. Both numbers appear together and they are different claims. Reading
the placement list as "where my models go" and the baseline as "what that saves" is the correct
reading; treating either as the other is the error §0 is about.

**The four memory terms, in the order they compose.** Each tenant's minimum per-shard envelope is

```
envelope_gib = (weight_gib/tp + kv_floor_gib/tp + activation_gib) / 0.95  +  mps_reserve_gib
```

Weights, cache and activation all live *inside* vLLM's `gpu_memory_utilization` fraction, so they are
divided by it. The MPS reserve sits *outside* that fraction (it is the MPS server's own context and allocator
rounding, which vLLM does not count) and is added afterward. Both of those are easy to drop, and dropping either invents room that
does not exist: getting it wrong on a GLM-4.5-Air shard was worth 1.8 GiB per shard, in the
flattering direction.

**The KV term is not something you set. It is derived from your latency target:**

```
concurrency     = rate_hz × osl_tokens × (itl_mean_ms / 1000)          # the mean
burst           = concurrency + 2.3263 × √concurrency                  # the q0.99 arrival burst
kv_floor_tokens = ceil(burst × (isl_tokens + osl_tokens))
floor_tokens    = max(kv_floor_tokens, 2 × max_model_len)              # page-rounded
```

Worked, for a tenant at 0.5 Hz with 1024 in / 1024 out and an 80 ms ITL ceiling:
`0.5 × 1024 × 0.080 = 40.96` resident requests in the mean, `55.85` at the burst the pool is sized
for, `× 2048 = 114,378` KV tokens. At 46 KiB/token/shard that is 5.02 GiB of cache the solver must
reserve on every shard — a number nobody typed anywhere.

The second term is the minimum viable pool: a pool that cannot hold two full contexts cannot overlap
one request's prefill with another's decode at all, whatever the rate says. Every fleet file that used
to state `min_kv_tokens` stated exactly two or four times its own `max_model_len`, which is this
arithmetic transcribed — and then obliged to be re-transcribed whenever the context moved.

`rate × osl × itl` is how many requests are concurrently resident if the tenant is to hold that ITL,
and each is priced as holding its full `isl + osl` span. So **a latency requirement becomes a memory
requirement**, which is the mechanism that makes this a co-location problem at all.

Note the direction, because it is counterintuitive: a *tighter* ITL ceiling produces a *smaller*
floor, since requests that finish sooner occupy the cache for less time. **It is the rate that demands
memory, not the target.** Which is one more reason targets are not a knob here — relaxing one to make
a tenant pass raises its KV floor and can push it off its memory floor entirely, making the baseline
worse.

**Envelopes are decision variables, not constants.** The formula above is only the lower bound. The
solver may hand a tenant more than its minimum, and its objectives run in strict priority order:

1. minimize the number of GPUs the fleet occupies;
2. maximize the *worst* tenant's KV margin (`allocated / floor`);
3. allocate whatever is left over, because unused envelope buys nothing.

**What it does not know: anything about compute.** No interference table, no throughput model, no
expected latency in the objective — latency enters *only* as that KV floor. So a placement `plan`
prints means exactly one thing: **not ruled out by memory.** Whether those tenants hold their targets
while sharing SMs is what `search` (§11) measures and what `plan` cannot guess.

**And two refutations it gives away free**, which is most of why it is worth running before anything
is leased. If no *pair* of tenants' minimum envelopes fits together on one GPU, then `can_coreside` is
False and no co-resident placement exists at any envelope the solver could choose — a search would
spend a node-hour confirming the disjoint answer. And if `sum(floor)` already exceeds the node's GPU
count, the baseline itself does not fit, so there is nothing for co-location to beat.

The first of those holds only when the pairs miss **on their weights**. When `plan` adds
`BUT NOT ON THE WEIGHTS`, some pair fits one GPU counting weights alone and misses only on the
computed terms above them (KV pool, activation, reserve) — the terms this stack has caught being
wrong — so `search` warns and runs anyway rather than refusing. When no pair fits even on weights,
`search` refuses unless you pass `--allow-disjoint-only`.

### TP: computed, but only as a lower bound

`plan` does resolve a shard count for every tenant, and it is worth being exact about which one.
Where measured solo rows exist, `fleetfile.resolve_tp` uses the degree those rows established. Where
they do not, `plan` walks the legal degrees — powers of two up to the node's GPU count — and takes the
first whose minimum envelope fits one GPU. That is the **memory floor**, and it is a lower bound on
the minimal TP, never the answer: a degree that *fits* can still miss the tenant's target.

A worked pair from this repo's two fleets: the GLM-4.5-Air anchor has floor 4 and its minimal degree
is also 4. The Qwen3-14B `chat` tenant has floor **1** — 27.5 GiB of weights fits one L40S — and a
minimal degree of **2**, because reading those weights out of HBM spends 34 ms of a 50 ms ITL budget
before the tenant does any work. That one-degree gap is the difference between a baseline of 6 and 7,
which is the denominator every saving gets divided by.

**The solver never searches over TP.** Once resolved, the degree is fixed input. The solver chooses
which GPUs each shard lands on and how large each envelope is; it does not choose how many shards a
tenant gets. That is deliberate: TP is a latency claim *and* the number the whole result is scored
against, so an optimizer allowed to pick it would pick the degrees that make packing look good and
manufacture its own baseline. A deleted fleet file did exactly that by hand — `tp: 4` with the comment
*"TP=2 leaves no room to share"*, when TP=2 fits — which is why the loader now refuses the line at all
(§7).

One special case worth knowing: where a tenant's memory floor is already the widest degree the node
offers, the floor *is* its minimal TP, with no measurement needed — there is nowhere to escalate to.
That is why `profile` sometimes reports a tenant resolved by arithmetic alone.

**It runs on a fleet whose degrees are not measured yet** — that is what it is for, and every fleet
starts there. A tenant with nothing profiled is loaded at its memory floor, its line is marked
`<-- FLOOR, not a measured minimum`, the
placement list carries the same caveat, and the baseline is printed as `BASELINE NOT ESTABLISHED`
with `sum(floor)` given as a **lower bound**. That distinction is the whole of §0: a degree that
*fits* is not a degree measured to *meet its target*, and only `profile` (§10) settles which. Every
other fleet-file error is still a refusal here, before a GPU is leased.

### Before `profile`: is the demand you wrote actually serveable?

`plan` answers "what placements fit?". It does not answer "can this hardware serve this `rate_hz` at
all?", and that question is worth asking on a laptop, because the way you find out on a GPU is by
spending 25 minutes of a leased node to learn arithmetic. That happened here: fleet 2 was measured
three times at 1 Hz, and 1 Hz was never serveable at any shard count or batch cap.

```bash
python tools/demand_sweep.py --fleet fleets/my-fleet.yaml --tenant anchor
```

It prints, per tenant and per candidate rate: token work, the memory-floor TP degree, the narrowest
degree **predicted** to meet the tenant's ITL target, the envelope and pocket at that degree, how much
of the derived batch cap the resident sequences the target implies consume, and the densest placement the
solver finds at the predicted degrees. Half of that is arithmetic straight out of `solver/` — the same
functions `plan` calls. The latency half is a **prediction** from a fitted `a + b × work` model, every
number marked `~`, and §0 forbids treating it as a result. Its only job is to tell you which demands
are worth measuring.

**And the knob it sweeps is demand, never a target.** If a fitted prediction says a tenant misses,
the legitimate responses are: a wider shard count, fewer tenants, or a lower `rate_hz` — a different
workload, declared before measuring, with the baseline recomputed. Editing `slo:` so the measurement
clears it is fitting the requirement to the answer, and it is the one move this whole method exists to
prevent. Note also that relaxing an ITL target *raises* the KV concurrency the solver reserves
(`rate × osl × itl`), so it can push a tenant off its memory floor and make the baseline worse.

One thing the sweep cannot tell you: whether lowering demand lowers the *baseline* as well as the
achieved count, which would shrink the saving you are trying to demonstrate. Check that by hand.
Where the baseline is set purely by weights — a model too big for one GPU — it cannot move, because
weights do not shrink with rate. Where it is set by latency, it can move in either direction.

## 9. `calibrate` — what the store already knows

**Position warning:** this is numbered before `profile` and `search` because it is free and reads only
what is already stored, but it has nothing to say until those have run at least once, and nothing
*useful* to say until something has been measured twice. On a first fleet, skip to §10 and come back.

```bash
python -m colocation.agent calibrate --fleet ../fleets/my-fleet.yaml
```

Summarises the between-run spread implied by **repeats already in the store**, which is what tells
the reasoner whether a difference between two placements is resolvable or noise. Deploys nothing.

On an empty store it has nothing to say. So **plan a deliberate replicate** in your first search —
measure one placement twice. Without it, nothing establishes a noise floor and the harness falls
back to a fixed 60-second window, which is correct during calibration and wrong afterwards.

## 10. `profile` — each tenant alone, which is what the baseline rests on

```bash
export COLOCATION_HARNESS_IMAGE=...        # same image as `search`; profile refuses without it
python -m colocation.agent profile --fleet ../fleets/my-fleet.yaml \
  --budget 8 --summary /tmp/profile.json
```

**Run this first among the modes that spend GPU time** — before `search`, and on new hardware before
anything else that costs money. It is numbered after `plan` and `calibrate` because those are free and
you will have run them already, not because they are prerequisites. It measures each tenant **alone**, with the whole of every GPU
it uses to itself, starting at the TP degree the tenant's memory forces and going up one power of
two at a time until the measured ITL meets that tenant's own target. The first degree that passes is
the minimum. Alone at the full GPU envelope is the tenant's best case, so a degree that misses there
misses everywhere and the answer is permanent for that target.

It costs at most one measurement per degree tried, and usually fewer: a tenant whose memory floor is
already the widest degree the node can offer is resolved by arithmetic with no measurement at all,
and `--budget` caps the rest. The plan it prints before spending anything says exactly how many
rounds it is about to run and why.

### What `profile` is, conceptually

**It is the actuator for one number: minimal TP.** Not a benchmark, not a characterization sweep — it
answers one question per row and then stops. *This model, sharded this many ways, driven at this rate
and this request shape, alone on its own GPUs with the whole of each one to itself — what latency did
it deliver?*

**Why that number has to be measured outside the search.** `search` decides which GPUs tenants share;
it never decides how many GPUs a tenant is sharded across (§8). So a fleet whose degrees are wider
than they need to be has *already conceded* the GPUs co-location would otherwise be credited with
saving — and nothing inside the search can notice, because the baseline it scores against was inflated
before it started. `sum(minimal tp)` is the yardstick, so it is established by a measurement the
optimizer has no hand in.

**Solo means best case, and both halves of "solo" matter.** *Alone* means no other tenant on those
GPUs, so no SM contention: this is the fastest the model can go on this hardware at this rate and
shape. *Full envelope* means the KV pool is as large as one GPU allows, so the tenant is not being
starved of cache either. Those two together are what make the result a **bound**:

- a degree that **misses** here cannot be rescued by any placement — sharing only ever takes SM time
  away — so the miss is permanent for that target;
- a degree that **passes** becomes the reference every co-resident measurement is compared against,
  which is the "vs solo" column in a search report.

**The escalation, and what it costs.** It starts at the memory floor — below that a shard does not fit
on one GPU at any envelope, so there is nothing to measure — and walks upward one power of two at a
time, stopping at the first degree that passes. So establishing minimal TP costs *(degrees ruled out)
+ 1* measurements per tenant, not a sweep. Two cases cost nothing at all: a tenant whose floor is
already the widest degree the node offers is resolved by arithmetic (there is no alternative to
compare against), and a tenant whose rows are already in the store is not re-measured.

### What "meets its target" means, and the three-valued answer

Both ceilings are checked, not just the interesting one: `itl_mean_ms` **and** `ttft_p95_ms`. A row can
sit comfortably inside its ITL target and still fail on TTFT — that is exactly what the GLM-4.5-Air
anchor did at a cap of 96, passing at 73.5 ms against 80 while its TTFT p95 came in at 56,365 ms
against 4,000. The report names the ceiling that was actually breached, because whoever reads it is
deciding whether to change the fleet file and cannot if it points at the wrong quantity.

**And the answer is three-valued: pass, miss, or *cannot say*.** That third state is the important
one. A row is refused as evidence about latency when:

- there is no ITL in it (the harness's own suppression reason is quoted rather than a cause asserted);
- the delivery gate failed — the offered load was not actually delivered, so the latency describes a
  lighter workload than the one you asked about;
- the harness's verdict was not `measured` — e.g. `infeasible_no_steady_state`, meaning it could not
  establish that the system had settled, so the average is over an unknown mixture of transient and
  steady state;
- the run's sampling was flagged suspect.

**None of those count as a miss, and the walk stops rather than widening.** Ascending a degree is
earned only by evidence that the degree is too slow or too small: a *measured* miss; a shard that is
not placeable at that degree; an engine that could not come up alone at the full envelope twice at
the same degree (one failed boot is retried once automatically, since one failure does not separate
a shard too narrow from a fault in the run); or a row that is unusable for latency but itself shows
the shard too narrow. A round that came back unusable for any other reason did not say the degree is
too slow; it said the degree was not measured — and whatever broke is a property of the run, not of
the shard count, so a wider degree would most likely break the same way with the budget spent to find
out. So `profile` stops there, reports the tenant unresolved, and tells you to fix the run and re-run
the same command. No `--reprofile` is needed: the walk up re-measures any degree that has no usable
row.

Treating "could not measure" as "missed" would push a tenant to a
wider shard count on the strength of a measurement that never happened — and a wider degree inflates
the baseline, which flatters the result. So an unusable row leaves the degree *unchecked*, `profile`
exits non-zero, and the answer is reported as an **upper bound** rather than as a number that reads
like a measured minimum. This guard was missing until 2026-09-09, and the run that exposed it reported
`BASELINE at minimal TP: 4 GPU(s)` from a row the harness had declined to vouch for.

### Why the rows live in their own store

**TP cannot be part of the store's key, and that is what forces the design.** A tenant's TP degree is
inside `ValidityKey`, so a round at TP=2 and the same tenant's round at TP=4 are two *different* keys
with two different `observations/` directories — correctly, since they are different experiments. No
single observation key can therefore hold the sweep that establishes minimal TP. So the comparison
lives in `store/profile.py` under a key naming **only the environment** (§7), with TP, rate and shape
moved into the record where varying them is the point.

**Targets are deliberately not stored.** A target is a requirement a *reader* applies to a row, not a
condition of measurement. That is what lets one profile serve every later fleet on the same hardware
at a different target — and it is why `profile` is the one artifact here that amortizes: the second
fleet on a g6e is cheaper to place than the first.

**The engine configuration is in the filename**, as `-cfg<digest>`. So raising a batch cap and
re-measuring writes a *new* row beside the old one rather than overwriting it, and two fleets that
state the same tenant identically share a row (which is worth engineering for — it is a free
profiling round). A row written before that digest existed is refused rather than read with defaults
guessed in, because those defaults are precisely the values that turned out to matter.

### The one thing only a solo round can measure

Per-shard memory the engine used that `gpu_memory_utilization` did not account for —
`nvidia-smi resident − util × device total`, the *measured reserve*. This is measurable **only** solo: with a
co-tenant, `fb_used` is the sum of both processes and MPS gives no per-process attribution. It is
filed in the profile store, keyed on the model and the degree, and every later load of a fleet on the
same environment merges it (the run logs a `measured reserve offered` line saying where it applied).
There is no field to paste it into: a fleet file that states `measured_reserve_gib` is refused, so a
stale paste cannot outlive the row it came from, and a second fleet on the same hardware gets the
figure for free.

If the derivation comes out at or below zero it is recorded as **unmeasured**, not as zero. The engine
claimed less than its budget, so the run put no lower bound on the overhead — and storing the negative
number would *widen* the KV pool the solver prices, pushing a later placement past what the GPU holds.

### What `profile` cannot tell you

Nothing about co-residency. It measures one tenant at a time by construction, so it has no view of
interference, and a fleet whose every tenant passes solo can still have no viable co-resident
placement. That is the whole of what `search` is for.

- **Rows are keyed by the environment, not by the fleet** (§7) — so every later fleet on the same
  hardware and engine image reuses them. A second `profile` run re-measures nothing it already holds
  a usable row for; `--reprofile` forces a re-measurement.
- **A first run on an instance type nobody has observed measures nothing.** With `--gpu-count N` it
  leases one node, files what the node presents, hands it back to the holder and exits `0` (§0); run
  the same command again to measure.
- **Everything about the tenant is in the row's filename instead**, including the engine flags it was
  served with, as `-cfg<digest>`. So changing a batch cap or a dtype writes a *new* row rather than
  overwriting the one measured at the old value, and a run under the new flags does not silently read
  back a latency produced by the old ones. Rows written before that was true are refused with one
  line each saying which field is missing — they are not read with the old defaults guessed in,
  because those defaults are exactly the values that turned out to matter.
- **It captures a node exactly as `search` does**, with the same consequences (see the next section
  in full before running either).
- **A row that cannot answer is not a failure.** A round whose delivery gate failed, or whose
  harness reported no ITL, counts as *unmeasured* rather than as a miss — treating it as a miss
  would widen a shard count on a measurement that never happened. The report says which, by name.
- The exit status is 0 only when every tenant resolved. An unresolved tenant is reported as an
  **upper bound**, never as a number that reads like a measured minimum.
- Rows land on EFS at `/efs/colocation/profiles/<profile-key>/solo/`, and are pulled back at the
  start of the next run.
- **It uninstalls its last tenant when it finishes**, unlike `search`, which leaves its final
  placement up unless you pass `--teardown`. There is nothing to inspect on a finished solo round —
  the row is already written — and a tenant left serving holds its whole envelope on gpu0 upward,
  which is the memory the next round, or the next search, would then be measured against. The node
  itself is *not* released; that is still `--release-node`.

Nothing measured here is written into your fleet file. The one memory fact a solo round measures,
the reserve above, is filed in the profile store and merged from there at load time.

## 11. `search` — the co-location search itself

```bash
export COLOCATION_HARNESS_IMAGE=...        # §3; search refuses without it
python -m colocation.agent search --fleet ../fleets/my-fleet.yaml \
  --budget 6 --summary /tmp/run.json
```

`--budget` is the number of *measurements* it may spend — the only GPU cost there is. `--max-turns`
bounds the conversation around them, because asking the solver and the store is free and a loop
that only stopped on GPU spend would never stop on a run that never measured.
`--relax-targets-pct` affects reporting only and costs nothing. It is a **ceiling on** each tenant's
own `slo.tolerance_pct` (§7, default 5%), able to lower it and never to raise it: see §12.

**When `plan` says no co-resident placement is feasible**, `search` follows the two cases in §8: it
warns and runs when `plan` added `BUT NOT ON THE WEIGHTS`, and otherwise refuses unless you pass
`--allow-disjoint-only`. It exits `1` when it measured nothing at all, and with `--gpu-count N` on an
instance type nobody has observed it only observes the node and exits `0` (§0).

**SM caps.** Besides choosing placements, the reasoner may cap a co-resident tenant's share of its
cards' SMs on a measurement (`sm_caps`, an MPS active-thread percentage from 1 to 99). A cap is a
ceiling, not a reservation: it takes SMs from that tenant and leaves them to the others on its cards,
whereas uncapped MPS gives idle SMs to whichever tenant can use them. A capped placement is a
different placement from the same one uncapped — it is recalled, calibrated and exported as its own —
and the summary rows carry `active_thread_percentage` for each capped tenant.

### A fleet over more than one node

With `hardware.max_nodes` above 1, `search` never measures the fleet as one. Each split is a set of
**groups**, one per node, and a group is the fleet file cut down to its tenants, written to
`<store>/groups/<fleet>/` and searched by this same command — its own lease, validity key, summary
and export, with `--budget` measurements of its own. Leases are taken **one node at a time**: with
`--teardown --release-node` each group gives its node back before the next group starts, so a
two-node split never holds two nodes during the search.

```bash
python -m colocation.agent search --fleet fleets/examples/g6e.48xlarge/three-teams.yaml \
  --budget 6 --group-budget 4 --teardown --release-node --summary /tmp/split.json
```

- `--group-budget` (default 4) is how many groups may be searched — each one a whole search, and a
  node while it runs.
- `--split-chooser reasoner` (the default) has the reasoner choose which group to search next and
  react to a group that missed, by asking for splits that keep its tenants apart. If the reasoner
  cannot be reached, or its replies cannot be used, the group the deterministic driver would search
  next is searched instead and recorded as a fallback. `--split-chooser drive` runs that driver
  alone: the ranked splits in order, the tightest group of each first.
- **A group an earlier run already searched is not searched again.** If its group file is
  byte-identical to the one that run used, its verdict is reused at no cost. Edit a target, a rate or
  a model in the parent and the group's file changes, so it is searched afresh.
- Every group is searched with `--allow-disjoint-only`, so a group whose tenants cannot share a GPU
  — one tenant alone, say — is measured rather than refused: it is still a node to deploy.

It writes the split summary to `<store>/groups/<fleet>/split.<stamp>.json`, and to `--summary`. When
a split is met it also writes the split's export there, `export.<stamp>.yaml` (§13), and to
`--export-to` if given. It exits `0` only when a split was met.

### What it will do to your cluster

- **It may buy a node.** The first thing `search` does is capture one: it installs its own
  `mps-<token>` daemon release whose pod requests *every* GPU on a machine, and whichever node that
  pod lands on is the node. Under Karpenter, an unschedulable pod requesting every GPU is an order
  for a whole instance of the fleet's type. If a node is already held by something at default priority, this run cannot
  preempt it and will provision a **second** one.
- **To hand an existing node over instead:** `helm uninstall <holder-release>` and *immediately*
  `kubectl apply -f manifests/gpu-node-holder.yaml`. The holder keeps the node — and its GPU UUIDs
  — at priority `-1000`, which a run's own daemon does preempt. Do not leave the node unheld: an
  idle GPU node is reclaimed in ~600 s and its UUIDs go with it.
- **It does not give the node back unless asked.** Without `--release-node` the lease is left
  standing, which *keeps* the node. Releasing is what risks losing the machine, not keeping it.
  With `--release-node` the lease is uninstalled and `gpu-node-holder` is applied in its place;
  it implies `--teardown`, and it has to — see the next point.
- **The handover it prints is usually three commands, not two, and the order matters.** A search ends
  with its tenants still serving on purpose, they are default priority, and the holder is `-1000`, so
  uninstalling only the daemon leaves the holder `Pending` — holding no GPUs and carrying none of its
  own `do-not-disrupt`. The closing message lists every release the run still has installed, tenants
  first; run it whole. If it could not list them, it says so, and you should check for releases
  prefixed with the run token before applying the holder.
- **It may refuse the node it just captured**, saying the node `is marked Drifted on its claim`. The
  holder carries `do-not-disrupt`, so while it sits there nothing can replace the node — which means
  the capture preempting it is what lets a pending replacement go ahead. Rather than measure on a
  machine that will be taken away mid-run, `capture` refuses. Let the replacement register, confirm
  the holder is on it, and capture again. The replacement comes up on a new AMI. If it reports the
  same usable memory, power limit and wiring as the old one, nothing changes and your rows still
  apply — a different driver version on its own does not move them. If one of those three *has*
  changed, the run stops with `EnvironmentMoved` and tells you to run the same command again: the new
  machine has been filed, so the re-run needs no flag and no edit, and it will start filling a fresh
  directory because the old rows describe a different machine.
- **It installs, per run:** one `mps-<token>` daemon + DCGM exporter pinned to the leased node, one
  engine release per tenant per candidate placement (`vllm-openai` unless the fleet or the tenant
  states another `engine:`, §7; `dynamo-vllm` also needs the Dynamo operator on the cluster), and one Job per measurement.
- Do not `helm upgrade` the daemon release under a live fleet. It restarts the control daemon,
  which tears down every MPS server and therefore every running tenant.

Records land on EFS within seconds of each measurement, at
`/efs/colocation/observations/<key>/observations/<run-id>.json`. The run's own summary — verdict,
reasoner turns, placements as ordinals, the targets they were judged against — is pushed to
`/efs/colocation/observations/<key>/searches/<run-token>.json` unconditionally, and to
`--summary` locally if you pass it.

## 12. Read the result

The run ends on three numbers in one unit — GPUs — and all three are always printed:

```
=== BASELINE vs STRICT vs RELAXED (impl §0) ===
  BASELINE  6 GPU(s) of 8, no GPU shared, at tp1+tp1+tp2+tp2

  STRICT    no placement. All 6 GPU(s) stand, 0 freed. Every ceiling exactly as written; no
            measured placement met all of them with a verdict for every tenant.
            search-003-dense-3gpu on 3 GPU(s): missed chat
              ...

  RELAXED   4 GPU(s), 2 freed (33% of the baseline) -- within every tenant's own tolerance. NOT CERTIFIED.
            search-002-chat-solo (chat:2 guard:3 reason:0,1 route:1 title:3)
            the concession: 1 of 5 tenants concedes; the rest met their ceilings as written
              chat ITL mean ceiling 50.000 -> 50.168 ms (+0.336%), within its default 5% tolerance
            the BASELINE does not move under the same concession (still 6), so the 2 is
            co-location's and not the concession's
            NOT CERTIFIED: this is what a concession the fleet file permits would buy. No record
            is stamped with it and STRICT above is unaffected.
            beyond what a tenant's own tolerance allows, so not offered:
              search-003-dense-3gpu on 3 GPU(s): chat needs 36.2% against a 5% tolerance

  vs solo -- what sharing cost each tenant, against the same tenant alone on whole GPUs (search-002-chat-solo):
    chat     tp1     8.412 ms co-resident against 6.104 ms solo, 1.38x, 0.84x of its 10.000 ms target
```

**Why three lines and not two.** This block used to print `ACHIEVED` and `FREED`. `ACHIEVED` named a
state rather than a quantity — it was in GPUs like `BASELINE`, but the label did not say so, and it
silently carried the predicate "every stated target met, with a verdict for every tenant", which was
the whole of its content. `FREED` was `BASELINE - ACHIEVED`: arithmetic on two numbers directly above
it, given a slot as prominent as the measurements it derived from, and `null` in exactly the runs that
mattered. And the pair was binary where results are not — a run can find nothing at the targets as
written *and* a placement two GPUs denser at a third of a percent on one tenant. Under the old
vocabulary the first got a headline and the second was an indented sub-bullet beneath a rejected
candidate, so the number a reader could actually act on was the one they had to go looking for.

Five things to read carefully there:

- **BASELINE is `sum(tp)`** at the resolved degrees, the same number `plan` printed before anything
  was captured. Where a resolved degree sits above its memory floor the block says so and prints what
  the baseline would be at the floor. That is normally not a warning: a degree lands above the floor
  precisely because the floor degree was measured and missed, and then those GPUs really are the
  disjoint fleet's cost. It becomes a warning when some *narrower* degree was never profiled at all —
  then the block says `THE BASELINE ABOVE IS AN UPPER BOUND`, names the tenants, and gives the lower
  figure it could fall to. Run `profile` for the missing degrees to settle it.
- **STRICT is the only certification, and it is deliberately hard to earn.** It is the fewest-GPU
  placement on which *every* tenant met its stated ceiling exactly as written. A tenant with no
  verdict — no target stated, or a latency the harness suppressed — blocks it rather than passing
  quietly, so a run can measure well and certify nothing. That is a result, not a malfunction.
- **RELAXED is what a concession your fleet file already permits would buy, and it is never a
  certification.** Each tenant declares how far past its ceilings it will be reported —
  `slo.tolerance_pct` (§7), defaulting to 5% — and RELAXED is the fewest-GPU placement every tenant's
  own tolerance admits. It states the concession in milliseconds of ceiling, because that is what you
  edit, names how many tenants concede, and says `NOT CERTIFIED` twice. It prints even when there is
  nothing on offer, with the reason: an empty slot is what tells you the question was asked. A
  placement past someone's tolerance appears in a separate one-line list, never as an offer, so a
  0.3% miss and a 36% miss can never be mistaken for each other. **Deciding to take a RELAXED
  reading is a change to a `slo:` block, which is yours to make, not the report's.**
- **A denser placement that was measured and refused is named under STRICT**, so the saving is
  not mistaken for the limit of what was attempted — and beneath it, what it would have taken:

  ```
              denser but not certified: search-003-dense-3gpu on 3 GPU(s), missed chat
                it would certify at chat ITL mean 51.451 ms against its 50.000 ms ceiling -- a 2.90%
                relaxation of one target and nothing else moves, within its default 5% tolerance
                the miss is 1.451 ms, which is 66x the calibrated 0.0221 ms half-width -- so it is a
                real miss, and relaxing the target is a decision about the requirement rather than
                about noise
                and it does not move the baseline: every tenant is already at the narrowest degree
                its memory floor allows, ...
  ```

  Your targets are inputs (§7), so `missed chat` is half the finding; how far it missed by is the
  half you can act on. **Nothing is judged against a relaxed target and no run is ever claimed on
  one** — the placement stays rejected and STRICT is computed against the ceilings as written. The
  bound on what gets *offered* is each tenant's own `slo.tolerance_pct`; `--relax-targets-pct` can
  only **lower** it, never raise it, so a concession cannot be invented at the command line once the
  result is known. `--relax-targets-pct 0` turns every offer off without hiding a single number.
  Three things to check before acting on one:
  - **the half-width**, from `calibrate` (§9). A miss smaller than the measured between-run spread is
    a reason to run a replicate, not to loosen a requirement, and the report says which it is. With
    nothing repeated on your key it says that instead of assuming a floor. **Read this line
    sceptically when your calibration rests on few passes.** Half the range of two draws is a lower
    bound on the true spread — the calibration record says so itself — and the same tenant's own
    `itl_ci_halfwidth_ms` in the record, a batch-means interval, is usually several times wider. On
    our own records the two disagreed by ~15x for one tenant in one bucket. So the printed multiple is
    an upper bound on how resolved a miss is; if it is not large, treat the miss as unsettled and run
    a replicate.
  - **whether it moves the baseline.** A looser ceiling can let a tenant meet its target at a
    narrower TP, which lowers `sum(tp)` — so the saving would have to be re-scored against the lower
    baseline, or co-location gets credit for GPUs the relaxation freed. The report works this out
    from the `profile` rows you already have, at no GPU cost, and reports `unchecked` rather than
    guessing when a narrower degree was never measured.
  - **that nothing is blocked.** A tenant whose latency was suppressed, or that stated no target, is
    named as unreachable by any relaxation; moving a ceiling cannot supply a measurement nobody took.
    The delivery threshold itself is never relaxed.
- **`vs solo` is the price.** It needs `profile` rows (§10) to have something to compare against; a
  tenant with none prints as such instead of borrowing another degree's number.

Above that block the run prints a `closest to targets:` line and a verdict — *ranked*, not certified,
and reported whether or not anything met its target, so a failed search still says which direction
was closest. Then:

```bash
python -m colocation.inspect record <run-id> --key <validity-key-hash>
python -m colocation.inspect record <run-id> --key <hash> --ordinals   # GPUs by host ordinal
```

Read the qualifiers next to the numbers, not just the numbers:

- **`verdict`** — a record that is not `measured` is not a latency measurement, whatever ITL it
  carries.
- **`sampling`** — sub-samples per sample. A phase-locked gauge read occupancy of exactly 0.0 while
  DCGM saw the GPU 47% busy, and it looked healthy in every field a summary would normally show.
- **`THROTTLED`** — a power or thermal cap is indistinguishable from co-location contention. Any
  latency difference in a throttled record may be the cap.
- **`suppressed`** — a latency that could not be measured prints as `suppressed`, never as `0.000`.
- The **`gpus`** column prints identities, not a count, because two placements sharing *both* GPUs
  versus *one* both show "2 GPUs, degree 1" and measured 5.76 ms against 10.88 ms.

A KV floor cleared is **not** a target met. The floors refuse placements that cannot hold a
request; they are not a latency model. In our own 14B/30B run every failing placement cleared its
floor with margin — the failures were SM contention, which no memory constraint expresses.

**A placement measured under an SM cap (§11) holds only with that cap.** Wherever the report names a
capped placement — the results table, STRICT, RELAXED, a denser miss, the closest-to-targets line — it
prints `SM cap <tenant> N%` beside it; the summary and the export (§13) carry the same caps as
`active_thread_percentage`. Report the cap beside the placement, and deploy it with the cap or not at
all.

Everything above addresses this node's GPUs, so none of it is what you deploy from. The run's last act
is to write the same result in a form that does not name the machine at all — **§13**.

---

### Reading a split

A search over several nodes prints each group's own report, as above, and then the split's:

```
=== splits ===
stopped: the reasoner submitted
verdict: met
  unresolvable   q-reason, q-chat, ... m-thumb  (run rcbuwu, chosen by the store)
  met            q-reason, q-chat, g-route, ...  (run rqu0bs, chosen by the store)
  met            q-route, q-guard, ... m-thumb  (run r26ljz, chosen by the reasoner)
```

The verdict is derived from the groups, not taken from the reasoner: `met` only when every group of
one split was searched and met, and the reasoner's own submission is printed beside it. Each group
says who chose it — `reasoner`, `fallback` (the driver's choice, made because the reasoner's could
not be used), `drive` (`--split-chooser drive`), or `store` (an earlier run's, reused at no cost).
To read a group's placements, open its own summary, named in `split.<stamp>.json`.

## 13. `export` — the result in a form that outlives the node

A search answers a question about one node: these tenants, on *those* devices, addressed by the UUIDs
a particular machine reported on a particular afternoon. That is the right thing for the store to
hold — a measurement is only ever a measurement of the machine it ran on — and the wrong thing to hand
to whatever deploys the fleet. A UUID names hardware Karpenter can reclaim in ten minutes; an ordinal
names a position in one driver's enumeration order. Neither survives the node, and a deployment built
on either silently addresses the wrong card.

So the same answer is written out a second time, in terms that do survive:

```bash
python -m colocation.agent export --fleet fleets/<name>.yaml
```

Free and offline: the store and the fleet file, no GPU time, no cluster beyond the write to the
canonical store. **A `search` runs this itself as it finishes**, through the same code, so in the
ordinary case there is nothing to do but read the `written to` lines. Run the mode yourself for an
older result, for another `--run-token` under the same key, for a fleet that has been profiled but
never searched, or if a run ever prints `the result is stored but was not exported`.

**A fleet with `max_nodes` above 1 is exported by `search` itself**, and `export` on it refuses. The
split's export is `schema_version: 2`: the parent fleet, the split's verdict and device total, and
under `nodes` each group's own export copied whole, with the path it was copied from. Each group's
own export is also written as below, under the group's key.

It writes `observations/<validity-key-hash>/exports/<run-token>.yaml` (or `exports/baseline.yaml`
for a fleet with no search yet), locally and on the canonical store. `--run-token` names an older
search under the key; one the key does not hold exits `1` and lists the tokens it does hold.
`--export-to <path>` writes an extra copy anywhere you like. YAML rather than JSON for one
reason — the file explains itself in comments — and YAML is a JSON superset, so any reader still
parses it.

What changes in the retelling:

- **A device becomes a slot.** A slot is one device with no identity, carrying the `(tenant, shard)`
  pairs that sit on it. Two residents in one slot shared a card; a tp4 tenant occupies four slots, one
  shard each. That is the whole of what a placement says, and it says it with no ordinal and no UUID:

  ```yaml
  strict:
    certified: true
    devices: 3
    devices_freed: 2
    scored_against_devices: 5
    fraction_freed: 0.4
    run_id: search-r70mmd-002-three-light-3gpu
    slots:
      - residents: [{tenant: alt, shard: 0}, {tenant: tag, shard: 0}, {tenant: thumb, shard: 0}]
      - residents: [{tenant: caption, shard: 0}]
      - residents: [{tenant: docqa, shard: 0}]
  ```

- **Memory becomes absolute GiB per shard.** `required_gib` is the figure to size a deployment
  against — a utilisation fraction is a fraction of a card nobody named, and this file deliberately
  states no card's capacity. The `granted_gib` beside it is what that run happened to give the tenant,
  which for a tenant alone on a card is the whole card; it is measured data about the run and not a
  recommendation. To turn `required_gib` into a fraction, divide by the capacity of the device you are
  deploying on.
- **An SM cap travels with its tenant.** A tenant measured under a cap (§11) carries
  `active_thread_percentage` beside its memory; an uncapped tenant carries no such field. A
  deployment that drops the cap serves a different placement from the one measured.
- **The conditions are carried by reference.** The rates, sequence shapes, ceilings and checkpoints
  are the fleet file's, named by path and by the sha256 of its bytes, plus the validity key they hash
  to. The key's expansion is already filed beside the observations as `key.json`, and a second copy
  here is a second thing that can disagree. Change a rate and the key moves — which is exactly what
  stops a placement being reused for a workload nobody measured it under.

**Three results are exported, not one, and the baseline always.** The baseline exists
unconditionally — it is arithmetic on measured degrees — so a fleet whose search found nothing still
has a deployable arrangement and a number to beat. `strict` is the certified placement, `relaxed` is
what a stated tolerance would buy and is `null` more often than not, with `relaxed_why_not` saying
why in a sentence: a null with no reason reads as a missing feature rather than as an answer. Then the
placements that were **measured and missed**, with the overshoot that ruled each one out:

```yaml
rejected:
  - run_id: search-r70mmd-001-dense-4share-2gpu
    devices: 2
    met: [docqa]
    missed:
      caption: {metric: TTFT p95, ceiling_ms: 2000.0, measured_ms: 6177.3183, over_by_ms: 4177.3183,
                over_by_factor: 3.0887, within_stated_tolerance: false}
```

Each of those cost a measurement round, and each rules out an arrangement that looks perfectly
reasonable from the certified answer alone — a consumer that knows 2 devices was tried and missed by
4.2 s of TTFT will not propose it again.

Last, `provenance`, in two halves that are addressed differently on purpose: the placements by the
**validity key** (this workload, these degrees, this hardware, this image), and the solo rows behind
the baseline by the **environment**, which is why one solo row prices several fleets (§10). The
`calibration` block is the resolution of a measurement on that node; `null` means nothing was measured
twice, and then no difference between two placements there is established as real (§9).

**The one refusal.** `export` will not load a fleet whose TP degrees are memory floors rather than
measurements — the same loader rule `search` is under, and the message names `profile`. An export is
the document somebody deploys from, and a floor exported as a baseline is the one overstatement the
whole design exists to prevent (impl §0), arriving in a file that outlives the conversation about
where it came from.

---

## 14. Serve the result on KAI — optional, and only when you ask

The search measures a placement on a node it holds under MPS, and the export (§13) states that
placement with no node in it. Serving it for real is a different job: the tenants have to stay up,
survive the node being replaced, and share their cards without the harness holding anything. This
step does that on the [KAI scheduler](https://github.com/kai-scheduler/KAI-Scheduler)'s GPU
fractions. **Nothing in the pipeline runs it.** A search ends at the export; deploying is a decision
about money and production that belongs to whoever owns the fleet.

```bash
python -m colocation.kai render <export or run token>     # free, offline: the plan and the Helm values
python -m colocation.kai deploy <export or run token>     # GPU: launches a kai-gpu node if none fits
python -m colocation.kai verify <name> [--wait-ready S]   # read-only: still placed as planned? Ready?
python -m colocation.kai drift <name>                     # read-only: real traffic vs the fleet file, per hour
python -m colocation.kai load <name> <scenario>           # CPU pod: a load test through each tenant's Service
python -m colocation.kai teardown <name>                  # returns once every pod has gone
```

It needs the platform's KAI option (§0): a separate `kai-gpu` Karpenter pool on Ubuntu, where the
GPU Operator installs a driver new enough for KAI's GPU fractions, tainted so nothing else lands on
it and so the `cuda` pool the search runs on is untouched. `check` (and `deploy`, which runs it
first) refuses when any of it is missing: the five KAI components (`kai-scheduler-default`, the
binder, admission, the pod-grouper and `kai-scheduler-gpu-fractioning`), the serving queue and its
quota, a `kai-gpu` NodePool that offers the result's instance type, and the `pv-fsx` volume claim. A
`deploy` whose name already has pods refuses too: `teardown` it first.

**What one deploy is.** Helm releases of one chart,
[`kai-vllm`](charts/machine-learning/model-serving/kai-vllm/): `kai-<name>-plan`, a ConfigMap holding
the plan plus the drift recording rules, and one release per tenant, `kai-<name>-<tenant>`, holding
a Kubernetes Deployment, a Service and the ServiceMonitor that has Prometheus scrape it. `helm list` shows them, labelled `colocation.kai/deployment=<name>`; `teardown` is still how
they come down, because it also waits for the pods to go. Each tenant serves vLLM's
OpenAI server, at `http://kai-<name>-<tenant>.<namespace>.svc:8000/v1` under the tenant's
`served_name`. Each shard gets a KAI GPU-memory fraction with SM sharing; a TP>1 tenant asks for that
fraction on each of its TP cards. All of them go to the `colocation-serving` queue: KAI treats a
Deployment's pods as non-preemptible serving work and schedules those only within their queue's GPU
quota, so the queue must hold at least the plan's cards, and `check` says so when it does not.

**The fractions are derived, not chosen.** `render` re-derives every envelope offline exactly as
`export` did and refuses (exit `2`) if any disagrees with the export, if the fleet file has changed
since, if the validity key has moved, if the slots do not fit one node, if the KAI reserve would
overflow a card, if no instance shape is filed for the type, or if there is no such result.
Then, per shard:

- the fraction is the envelope the search measured, less the MPS reserve it included, plus the
  reserve a KAI fraction costs instead;
- `--gpu-memory-utilization` is recomputed against that fraction, so vLLM is handed the same budget
  it was measured with, not a fraction of a card nobody named;
- every slot's card is then **filled** to the card's usable memory, the remainder going to the
  slot's single-shard tenants as KV cache.

**An SM cap is served too.** A tenant the export carries an `active_thread_percentage` for (§13) gets
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` set to it in its pod environment, which KAI's shared MPS server
honours, and `render` prints `SM cap N%` beside it on each of its slots. An uncapped tenant gets no
such variable.

**Why fill, and why one at a time.** KAI picks the cards, not us: its binpack strategy puts a
fraction on the fullest card it still fits on. A full card cannot take another tenant, so with every
slot filled there is only one way to place the next one. The tenants are applied in an order that
makes binpack reproduce the slots — cards without a TP tenant first, then the TP tenants, then their
mates — each waited on until KAI has bound it, and every tenant after the first pinned to the first
one's node. Helm has no order between releases, so `deploy` supplies it: it installs one release,
does not wait for it to be Ready, only bound, and installs the next; the engines then start together.
`deploy` then reads back the cards KAI chose and compares them with the slots.

**Reading the outcome.** `deploy` exits `0` when every tenant sits where its slot says, `3` when they
are bound but not as the slots say, `1` when something failed or timed out, and `2` when it refused
before applying anything. A `3` is retried by tearing the deployment down and deploying again — not
by moving one pod, because deleting a pod lets its Deployment re-create it wherever binpack likes.
`verify --wait-ready` then waits for every engine to be Ready. A tenant that is placed but never Ready
is a finding about its fraction, and is reported, not retried. `verify` exits `0` when placed as
planned (and, with `--wait-ready`, Ready), `3` on a mismatch — tenants spread over nodes, a wrong card
count or request, duplicate pods, or a pod that is not ours on one of the cards — `1` when a pod is
missing or unbound, the stored plan is gone, or the wait timed out, and `2` when given no name.

**What carries over from the search, and what does not.** The placement and every tenant's memory
budget carry over exactly. The latencies do not, by construction: the serving stack is vLLM's own
OpenAI server whatever engine the search measured on, and sharing is KAI's rather than the
harness's MPS daemon. `render` prints a warning for every such difference; read them before you
deploy. The result's instance type is the one the deploy asks Karpenter for, so it must be one the
`kai-gpu` pool allows.

**Measuring drift.** The search was run on assumptions: each tenant's `demand` in the fleet file,
and the targets it had to hold under that demand. Real traffic is whatever arrives. `drift <name>`
records how far the two have moved apart, per tenant and per whole clock hour since the deploy:

- **demand drift**: the request rate, mean prompt tokens and mean output tokens, each beside the
  fleet file's figure and as a percentage of it. Usually the cause, and what to correct in the fleet
  file before running the pipeline again;
- **latency drift**: ITL mean and TTFT p95, each beside the tenant's target and what the search
  certified for the result being served, with the tenant's `tolerance_pct` alongside. The effect,
  and whether it matters yet.

An hour with no requests is `no traffic`, not zero drift. An hour with fewer than 20 gets its rates,
lengths and ITL but no p95, marked `too few requests`. Hours already recorded are skipped, the
current hour waits until five minutes after it ends, and nothing older than Prometheus's 30 days can
be recorded, so run it at least monthly; daily is plenty. The records are JSON lines in
`drift/<name>.jsonl` beside the export's directory, and the same file is pushed to EFS under the
fleet's validity key. It decides nothing: no alert and no rerun. Whether the pipeline should run
again is for you to read off the records.

The figures are vLLM's own, scraped by the cluster Prometheus: each tenant's release carries a
ServiceMonitor, and the plan's release carries recording rules (`colocation_kai:<figure>:1h`) with
the same queries, for graphing. Where the Prometheus Operator's CRDs are absent, neither is
installed and the deploy goes ahead unmonitored. Two caveats: ITL is each request's own mean gap
between tokens, the same quantity the harness reports, but TTFT p95 is read off vLLM's histogram,
whose buckets are coarse above 250 ms (0.25, 0.5, 0.75, 1, 2.5 s), so it is a trend rather than the
harness's figure. And both are timed in the server, not a client, so they miss the network hop.

**A split's export deploys as one deployment per node.** Given a `schema_version: 2` export,
`deploy` deploys node `i` as `<name>-n<i>`, one node after the other, each exactly as above. The
first tenant of each later node is kept off the earlier nodes' hosts (pod anti-affinity) and in
the first node's zone, so the `kai-gpu` pool has to supply that many nodes of the type in one zone —
two g6e.48xlarge for `three-teams`, two g6e.12xlarge for `g6e.12xlarge/two-reasoners`. `deploy` refuses if any of them already has pods, and verifies every node at the end.
From then on each node is a deployment of its own: `verify`, `drift` and `teardown` take
`<name>-n<i>`.

**Load-testing a deployment.** `load <name> <scenario>` drives the deployment with a scenario's
traffic, through each tenant's Service, for whole clock hours. The traffic comes from a harness Job
on the CPU pool, running the harness image's `harness.soak` (`$COLOCATION_HARNESS_IMAGE`, or
`--image`). It applies the Job and returns. `drift` then records what the deployment saw in those
hours. The Job's log has the client's own figures for the same hours, which include the network
hop. A scenario is the fleet's demand with changes stated on top: `as-fleet.yaml` runs every tenant
as its fleet file says, and `demand-drift.yaml` moves each of a few tenants away from it in one way.
Every example fleet has both, in [`load-tests/examples/`](load-tests/examples/). `load` refuses
until every engine is Ready. It also refuses, for now, a fleet's scenario on one node of a split:
the scenario names the parent fleet and all its tenants, and the node's deployment holds one group.

**Writing a scenario.** A scenario is a short YAML file under
`load-tests/examples/<instance type>/<fleet>/`, one folder per fleet under `fleets/examples/`, named as
the fleet file. It names its fleet, how many whole clock hours to run, and the changes to make to
each tenant's demand:

```yaml
fleet: fleets/examples/g6e.48xlarge/qwen-anchor-riders-3.yaml
hours: 2
tenants:
  guard: {rate: 2.0}      # rate, isl and osl multiply the fleet file's demand
  route: {isl: 1.9}
  title: {per_hour: 10}   # a fixed count instead of a rate
  chat: off               # not driven
```

A tenant that is not named runs at its fleet demand. That demand is read from the export the
deployment was made from, never copied into the scenario, so a scenario cannot drift from its fleet
except by the changes it states. Unknown keys are refused, as are a tenant the fleet does not have
and a request longer than the tenant's context. The self-test checks that every example fleet has a
scenario folder, that every folder belongs to a fleet, and that each scenario fits its fleet.

**Teardown means gone.** `teardown` uninstalls every release of the deployment, the plan's last,
deletes its load Jobs, and returns only once every pod has departed, so the fractions are free when it says so. It gives up
after 900 s and exits `1`. The node is left
for Karpenter to consolidate, about an hour after it empties.

With Claude Code (§15), `/fleet-kai-deploy <export>` does all of the above, including the retry, and
`/fleet-kai-status`, `/fleet-kai-drift`, `/fleet-kai-load` and `/fleet-kai-teardown` are the others.

---

## 15. Driving all of this with Claude Code

Sections 8–13 are six modes, four back-edges and about a dozen refusals, and **nothing in the code
walks you through them.** The loader protects the baseline; the order and the responses to each
refusal are yours to remember. That is a lot to hold, and it is exactly what a skill is for.

This repo ships one, inside a plugin: **[`plugins/colocation-pipeline/`](plugins/colocation-pipeline/)**.
The plugin holds three things — two skills (`run-fleet`, which owns the sequence, and `kai-deploy`,
which serves a result on KAI only when you ask, §14), a command per step, and hooks that refuse the
handful of commands which destroy a live run.

### Using it

Open Claude Code in `tutorials/multi-model-gpu-sharing/`:

```
cd /path/to/amazon-eks-machine-learning-with-terraform-and-kubeflow/tutorials/multi-model-gpu-sharing
claude
```

Install the plugin once, from inside that session:

```
/plugin marketplace add ./plugins
/plugin install colocation-pipeline@colocation-plugins
```

Then say what you want in your own words:

```
> Run fleets/my-fleet.yaml through the pipeline.
```

or invoke it by name:

```
> /run-fleet fleets/my-fleet.yaml
```

Either way the skill loads and Claude drives `plan` → the demand sweep → `profile` → `plan` again →
`search` → the report → the canonical form, in that order, reading each refusal and doing what it says.

A step on its own has a command of its own: `/fleet-preflight`, `/fleet-image`, `/fleet-plan`,
`/fleet-sweep`, `/fleet-profile`, `/fleet-search`, `/fleet-calibrate`, `/fleet-inspect`,
`/fleet-export`, and the five that serve a result on KAI (§14): `/fleet-kai-deploy`,
`/fleet-kai-status`, `/fleet-kai-drift`, `/fleet-kai-load`, `/fleet-kai-teardown`. They carry the same instructions for that one step, so picking up a fleet part-way
through does not mean re-reading the whole sequence.

`/fleet-preflight` is worth running first on a cluster you have not measured on today. It is
read-only, and it checks the four things that otherwise fail silently: that only our own DCGM
exporter is on the cluster, that something is holding the GPU node, which harness image the source
hashes to, and the hand-typed cluster fields in the harness Job template.

### What it does for you

- **Builds and pushes the harness image for you** (§3), rather than assuming one is exported. It tags
  the image with a **digest of the build context**, so a changed harness gets a new tag automatically
  and an image cannot be pullable-but-stale — the failure mode that `colocation-harness:v5` was. If
  that tag is already in ECR the step is free; if not, it builds, runs the shipped in-image selftest,
  and pushes only if that passes.
- **Runs the offline steps first, and again after every change.** `plan` and
  `tools/demand_sweep.py` touch no cluster, so there is never a reason to skip them.
- **Reads the refusals as instructions.** `NO CO-RESIDENT PLACEMENT IS FEASIBLE`,
  `THE BASELINE DOES NOT FIT`, `BASELINE NOT ESTABLISHED`, the four `resolve_tp` refusals — each has
  one correct response and the skill encodes which.
- **Takes the back-edges.** A profile round that widens a degree sends it back to `plan` before it
  will search. A demand that cannot work sends it back to the fleet file — and when it edits one, it
  re-reads the comments around the change, because a rate or a target that moves usually makes the
  sentence next to it wrong.
- **Refuses the shortcuts.** It will not quote a saving against `sum(floor)`. It will not widen a TP
  degree past an unusable profile row. It will not relax a `slo:` to make a fleet pass — it changes
  the *rate* and says so, or asks.

### What the hooks refuse, and what they put to you

The skill states these in prose. The hooks are what makes them hold when the prose is not being read,
and a hook decision cannot be talked around.

**Refused outright:**

| refused | because |
|---|---|
| a `helm uninstall` that names no release, lets a substitution or glob choose the names, or names another namespace | an uninstall that says exactly what it removes is ordinary operation and is allowed, with a reminder that removing the daemon leaves the node held by nothing until the holder is Running |
| `helm upgrade` naming the mps-daemon | it restarts the control daemon, tearing down every MPS server and every running tenant |
| `pkill`/`killall` matching the MPS or NVIDIA processes | `-f` matches the `bash -c` wrapper's own command line and kills the shell |
| reading, echoing or interpolating the HF token, or passing it via `helm --set` | it reaches Kubernetes as a file, from `--from-file`, and nothing else |
| `--no-holder` | an unheld node is reclaimed in ~600 s and its GPU UUIDs go with it |
| `profile` or `search` on an image that is not the digest of the current harness source | a tag that pulls is not a tag that measures |

**Put to you instead:** `--reprofile`, `--allow-disjoint-only`, `--relax-targets-pct`, and any edit
to a fleet file's `slo:` block, its `tolerance_pct`, or a stated `kv_bytes_per_token`. Each changes what a run measures or what it risks,
which makes it a decision for whoever owns the requirement. So is a `colocation.kai deploy`, `load`
or `teardown`, unless your own last prompt was `/fleet-kai-deploy`, `/fleet-kai-load` or
`/fleet-kai-teardown`: serving a result is a step you start (§14). A load test sends synthetic
traffic to a serving deployment, and its drift records count that traffic as real demand.

**Noted, without a decision:** deleting the node holder comes with the reminder that something has to
be holding the node within about 600 s, and `--release-node` with the reminder that the release
re-applies the holder and that a created holder is not yet a Running one.

**Checked after a measurement round:** whether anything is still holding the GPU node; after a
`profile`, that `plan` has to run again, because the degrees it settled moved the baseline and every
placement; and after a `search`, whether the result was written in canonical form (§13) — a result
addressed by GPU UUID and ordinal expires with the node, and re-stating it is free.

None of this is about what a run costs. GPU time is not the plugin's concern.

### What it will not do

It will not invent a noise floor when `calibrate` has nothing to report, it will not certify a
placement no measurement supports, and it will not let a summary imply the pipeline completed when a
step was skipped or blocked. If the arithmetic says your fleet cannot be co-located, that is the
answer it gives you.

### Adapting it

Everything in the plugin is plain text: Markdown with YAML frontmatter for the skill and the
commands, one JSON file for the hooks, and two shell scripts plus two Python ones under `scripts/`
and `hooks/`. Read them and edit them for your cluster.

The parts most likely to need changing are the namespace (`COLOCATION_NAMESPACE` overrides it in the
scripts), the holder-manifest name, and the node-holding rules in the skill's **Non-negotiables**,
which encode our cluster's Karpenter behaviour. Paths are derived from where the plugin sits, so
they need no editing. The sequence and the refusal table are cluster-independent.

---

## A note on references in comments and messages

Some code comments, and some messages the tools print, cite sections of internal design documents
(`impl §N`, `design §N`) or numbered findings from early hardware testing (`finding N`). Those
documents are not included here. The text around each citation says what matters.
