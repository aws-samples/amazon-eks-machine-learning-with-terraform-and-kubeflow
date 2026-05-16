# B9: memledger v1.0.0 cluster integration runbook

This runbook validates **memledger v1.0.0** on the existing `kagents-mem-eks` EKS cluster
that already has kagent + memledger 0.4.0 deployed. We **upgrade** memledger in place,
**rebuild** the three agent images against `memledger==1.0.0` from PyPI, and run a
smoke test that exercises `add` / `search` / `chain_store.build_chain` end-to-end.

> **Scope:** v1 is pgvector-only. AWS-native composition (DynamoDB primary +
> OpenSearch search) is reserved for v2 — see
> `examples/agentic/eks-ops-agent/v2-preview/`. Do not enable
> `ENABLE_COMPOSITION=true` against v1.

> **Do not run `terraform apply`.** The cluster, Aurora pgvector instance, and IAM are
> already provisioned. This runbook is `helm upgrade` + image rebuild only.

---

## 1. Prereqs

Tools on your workstation:

- `kubectl` >= 1.30
- `helm` >= 3.14
- `awscli` v2 with credentials for the account that owns `kagents-mem-eks`
- `docker` with buildx (for the agent image rebuild step)
- `jq`

Environment:

```bash
export AWS_REGION=us-west-2          # cluster region (per terraform var "region")
export AWS_PROFILE=<your-profile>    # named profile that can assume EKS admin
export CLUSTER_NAME=kagents-mem-eks
```

---

## 2. Connect to the existing cluster

```bash
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION"

# Sanity: cluster up, kagent + memledger + phoenix + pgvector all present
kubectl get nodes
kubectl get pods -A | grep -E "kagent|memledger|phoenix|pgvector"
```

**Expected**: Nodes `Ready`. The grep returns running pods for `kagent-controller`,
`memledger-*` (chart from 0.4.0), `phoenix-*`, and either `memledger-postgresql-*`
(in-cluster pgvector) or none (Aurora-only deployment). If any pod is `CrashLoopBackOff`,
stop and investigate before upgrading.

---

## 3. Helm upgrade memledger to v1.0.0

The chart lives at `/Users/aratnch/Documents/2026/code/memledger-ai/memledger-core/charts/memledger`.
Pick **(a) in-cluster pgvector** for a quick smoke, or **(b) Aurora** for the
production-shape path. Don't run both — the chart owns the same release name.

```bash
export MEMLEDGER_CHART=/Users/aratnch/Documents/2026/code/memledger-ai/memledger-core/charts/memledger
```

### (a) In-cluster pgvector smoke

Deploys a `pgvector/pgvector:pg17` StatefulSet inside the cluster. Useful to confirm
the chart upgrades cleanly without touching Aurora.

```bash
helm upgrade --install memledger "$MEMLEDGER_CHART" \
  --namespace kagent \
  --set database.deploy=true \
  --set database.migration.enabled=true \
  --set database.migration.tableName=agent_memory \
  --set database.migration.vectorDimensions=384 \
  --set embeddings.provider=local \
  --set embeddings.model=BAAI/bge-small-en-v1.5 \
  --set embeddings.dimensions=384 \
  --set memledger.defaultBackend=pgvector \
  --wait --timeout 5m
```

### (b) Aurora production-shape

Uses the existing Aurora cluster `kagents-mem-eks-kagent-db` and `kagent-postgresql`
secret already in the `kagent` namespace. Bedrock Titan embeddings (1024 dim) — bump
`vectorDimensions` to match. **If the existing 0.4.0 schema used 1024 dim already, this
is a no-op migration. If it used 384, you must drop & recreate `agent_memory` (see
chart values.yaml note on fixed vector column size).**

```bash
helm upgrade --install memledger "$MEMLEDGER_CHART" \
  --namespace kagent \
  --set database.deploy=false \
  --set database.host=kagents-mem-eks-kagent-db.cluster-cy3og42oirp8.us-west-2.rds.amazonaws.com \
  --set database.port=5432 \
  --set database.database=engram \
  --set database.user=kagent \
  --set database.existingSecret=kagent-postgresql \
  --set database.secretKey=postgres-password \
  --set database.migration.enabled=true \
  --set database.migration.tableName=agent_memory \
  --set database.migration.vectorDimensions=1024 \
  --set embeddings.provider=bedrock \
  --set embeddings.model=amazon.titan-embed-text-v2:0 \
  --set embeddings.dimensions=1024 \
  --set memledger.defaultBackend=pgvector \
  --wait --timeout 5m
```

**Expected**: `helm` reports `STATUS: deployed`, revision incremented. Migration job
completes (`kubectl logs -n kagent job/memledger-migration` → `Migration complete`).

---

## 4. Build & push the three agent images

Each agent has its own `build-and-deploy.sh`. They build a `linux/amd64` image, push
to ECR (auto-creates repo if missing), and run `helm upgrade --install` for the agent
chart with IRSA-annotated SA. The Dockerfiles now install `memledger==1.0.0` from PyPI
(no wheel copy).

```bash
# Required by the build scripts:
export TF_DIR=/Users/aratnch/Documents/2026/code/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup
export REPO_DIR=/Users/aratnch/Documents/2026/code/amazon-eks-machine-learning-with-terraform-and-kubeflow

# eks-ops-agent — pgvector mode (matches Helm step 3 above)
cd "$REPO_DIR/examples/agentic/eks-ops-agent"
ENABLE_MCP_TOOLS=true ENABLE_MEMORY=true ./build-and-deploy.sh

# triage-agent
cd "$REPO_DIR/examples/agentic/triage-agent"
./build-and-deploy.sh

# compliance-agent
cd "$REPO_DIR/examples/agentic/compliance-agent"
./build-and-deploy.sh
```

**Expected**: `Push complete: <acct>.dkr.ecr.us-west-2.amazonaws.com/<agent>:<version>`,
followed by `Deploy complete!` and pods healthy in `kagent` namespace.

ECR repos created/updated: `eks-ops-agent`, `triage-agent`, `compliance-agent`.

---

## 5. Apply agent CRDs

The build scripts already run `helm upgrade --install`, which renders the kagent
`Agent` CRD. If you want to re-apply the CRD-only YAML manually (e.g. after editing
the agent prompt):

```bash
kubectl apply -f "$REPO_DIR/examples/agentic/triage-agent/triage-agent.yaml"
kubectl apply -f "$REPO_DIR/examples/agentic/eks-ops-agent/eks-ops-agent.yaml"
kubectl apply -f "$REPO_DIR/examples/agentic/compliance-agent/compliance-agent.yaml"
```

**Expected**: `agent.kagent.dev/<name> configured` for each. Verify with
`kubectl get agents -n kagent`.

---

## 6. Smoke tests

### 6.1 In-pod add / search / chain

Pick the eks-ops-agent pod (it has `MEMLEDGER_PG_DSN` wired in):

```bash
POD=$(kubectl get pod -n kagent -l app=eks-ops-agent -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n kagent "$POD" -- python - <<'PY'
import asyncio, os
from memledger import Memledger

async def main():
    async with await Memledger.create(connection_string=os.environ["MEMLEDGER_PG_DSN"]) as ml:
        a = (await ml.add("v1 smoke: root knowledge", namespace="/b9/smoke"))[0]
        b = (await ml.add("v1 smoke: derived insight",
                          namespace="/b9/smoke", derived_from=[a]))[0]
        c = (await ml.add("v1 smoke: superseding insight",
                          namespace="/b9/smoke", supersedes=b))[0]

        results = await ml.search("smoke", namespace="/b9/smoke", top_k=5)
        assert len(results.records) >= 3, f"expected >=3 results, got {len(results.records)}"

        chain = await ml.chain_store.build_chain(c, direction="upstream")
        assert len(chain.hops) >= 2, f"expected >=2 hops upstream, got {len(chain.hops)}"
        print("OK", a, b, c, "chain_hops=", len(chain.hops))

asyncio.run(main())
PY
```

**Expected**: `OK <id-a> <id-b> <id-c> chain_hops= 2` (or more). No tracebacks.

### 6.2 Phoenix spans

```bash
kubectl port-forward -n kagent svc/memledger-phoenix 6006:6006
# open http://localhost:6006 in your browser
```

**Expected**: Phoenix UI shows recent spans named `memledger.add`, `memledger.search`,
`memledger.chain_store.build_chain` from the smoke run. If empty, check that the
`OTEL_EXPORTER_OTLP_*` envs are present on the agent pod (`kubectl describe pod $POD`).

### 6.3 CLI inside the pod

```bash
# Use one of the IDs printed by the smoke run, e.g. the superseding record:
kubectl exec -it -n kagent "$POD" -- memledger get <id-c> --chain
```

**Expected**: A rendered chain showing `<id-c>` → `<id-b>` (supersedes) → `<id-a>`
(derived_from), with confidence per hop.

---

## 7. Expected outputs cheat-sheet

| Step | Pass | Fail |
|------|------|------|
| 2 | Nodes Ready, kagent/memledger pods Running | Any `CrashLoopBackOff` or `ImagePullBackOff` — fix before upgrading |
| 3 | `helm` STATUS deployed; migration job `Complete` | Migration job failure → check `kubectl logs job/memledger-migration` |
| 4 | Three `Push complete` lines, three `Deploy complete!` | ECR auth fail → re-run `aws ecr get-login-password`; build fail → see Dockerfile findings below |
| 5 | `agent.kagent.dev/* configured` x3 | CRD missing → ensure kagent controller is up |
| 6.1 | `OK ... chain_hops= 2` | Import error → 1.0.0 wheel didn't make it into image; re-run step 4 |
| 6.2 | Spans visible in Phoenix | No spans → OTEL env not propagated |
| 6.3 | Chain rendered | `memledger` CLI not on PATH in image → known: `memledger` console-script ships in v1.0.0 |

---

## 8. Rollback

If anything in step 3 or step 4 misbehaves:

```bash
helm rollback memledger 0   # drops back to the previous (0.4.0) release
```

Agent pods will continue to talk to the same `agent_memory` table — schema is
backward-compatible between 0.4.0 and 1.0.0 at the column level (no destructive
migrations in v1.0.0).
