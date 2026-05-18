# Agentic AI Examples on EKS + kagent + memledger

This directory contains a reference multi-agent setup running on
[kagent](https://github.com/kagent-dev/kagent) (a Kubernetes-native AI
agent framework) with a governed memory layer provided by
[memledger](https://pypi.org/project/memledger/) v1.0.0.

Three agents share the same memory store under explicit RBAC and full
attribution:

- **`triage-agent`** — ingests incoming alerts, correlates with past
  incidents, escalates.
- **`eks-ops-agent`** — troubleshoots EKS clusters, recalls
  remediations, learns from incidents.
- **`compliance-agent`** — audits provenance, enforces retention,
  generates trust attestations.

This README covers, in order:

1. Setting up kagent on EKS (Terraform + Helm).
2. Choosing an LLM (self-hosted, OpenAI, or Bedrock).
3. How the agents use memledger — agent roles, namespaces, attribution.
4. Memory Attribution Integrity (MAI) scoring rubric and the 3-tier
   evaluator suite.
5. Phoenix observability — span inventory + sample attributes.
6. Cluster integration procedure for memledger v1.0.0 (Helm chart +
   agent image rebuilds + smoke tests).

> **Scope note:** v1 is pgvector-only. AWS-native composition (DynamoDB
> primary + OpenSearch search index) is reserved for v2 — see
> `eks-ops-agent/v2-preview/`. Do not enable `ENABLE_COMPOSITION=true`
> against v1.

---

## 1. kagent infrastructure setup

### 1.1 Create kagent configuration

```bash
cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/eks-cluster/terraform/aws-eks-cluster-and-nodegroup

# Create kagent.tfvars
cat <<EOF > kagent.tfvars
kagent_enabled               = true
kagent_database_type         = "sqlite"
kagent_enable_ui             = true
kagent_enable_bedrock_access = true
EOF
```

### 1.2 Apply Terraform

**For Quick Start (Basic) users:**

```bash
terraform apply -var-file="basic.tfvars" -var-file="kagent.tfvars"
```

**For Advanced Setup users:**

```bash
terraform apply -var="profile=default" -var="region=us-west-2" \
  -var="cluster_name=my-eks-cluster" \
  -var='azs=["us-west-2a","us-west-2b","us-west-2c"]' \
  -var="import_path=s3://<YOUR_S3_BUCKET>/eks-ml-platform/" \
  -var-file="kagent.tfvars"
```

### 1.3 Configuration options

- `kagent_version`: Helm chart version (default: `"0.7.11"`, pinned for
  stability).
- `kagent_database_type`: `"sqlite"` (default, single replica) or
  `"postgresql"` (HA, multi-replica).
- `kagent_enable_ui`: enable web UI (default: `true`).
- `kagent_enable_istio_ingress`: expose UI via Istio ingress (default:
  `false`).
- `kagent_enable_bedrock_access`: enable IRSA for Amazon Bedrock access
  (default: `false`).

### 1.4 Access the kagent UI

```bash
kubectl port-forward -n kagent svc/kagent-ui 8080:8080
# or via Terraform output
$(terraform output -raw kagent_ui_access_command)
```

### 1.5 High availability

For production deployments with multiple controller replicas:

```bash
terraform apply \
  -var="kagent_enabled=true" \
  -var="kagent_database_type=postgresql" \
  -var="kagent_controller_replicas=3"
```

### 1.6 kmcp — Kubernetes MCP Server Controller

[kmcp](https://github.com/kagent-dev/kmcp) is a Kubernetes-native
controller for deploying [Model Context Protocol
(MCP)](https://modelcontextprotocol.io/) servers as Kubernetes
resources.

**Enable kmcp:**

```bash
terraform apply -var="kmcp_enabled=true"
```

| Variable | Description | Default |
|----------|-------------|---------|
| `kmcp_version` | Helm chart version | `"0.1.4"` |
| `kmcp_namespace` | Kubernetes namespace | `"kmcp-system"` |
| `kmcp_controller_replicas` | Number of controller replicas | `1` |
| `kmcp_enable_istio_injection` | Enable Istio sidecar injection | `false` |

kmcp is installed from two OCI Helm charts:

- CRDs: `oci://ghcr.io/kagent-dev/kmcp/helm/kmcp-crds`
- Controller: `oci://ghcr.io/kagent-dev/kmcp/helm/kmcp`

```bash
kubectl port-forward -n kmcp-system svc/kmcp-controller-metrics 8443:8443
```

Set `enable_service_monitor=true` to create a ServiceMonitor for
Prometheus scraping. When `metrics_secure_serving` is enabled, the
ServiceMonitor uses `scheme: https` with `insecureSkipVerify: true`. For
production, replace `insecureSkipVerify` with proper CA configuration
via `additional_helm_values`.

When `kmcp_enable_istio_injection=true`, the namespace is labeled with
`istio-injection=enabled`. If the controller readiness probe fails after
enabling Istio, ensure `holdApplicationUntilProxyStarts` is set in your
Istio mesh config, or add a `postStart` hook to wait for the sidecar.

**Example MCP server CRD:**

```yaml
apiVersion: kagent.dev/v1alpha1
kind: MCPServer
metadata:
  name: my-mcp-server
  namespace: kmcp-system
spec:
  transport: sse
  image: my-mcp-server-image:latest
  port: 8080
```

For full kmcp documentation, see the [kmcp
repository](https://github.com/kagent-dev/kmcp).

---

## 2. LLM integration options

kagent supports multiple LLM providers. Choose one before deploying the
agents.

### 2.1 Self-hosted models in EKS (recommended)

```yaml
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: llama-3-8b
  namespace: kagent
spec:
  provider: OpenAI  # vLLM provides OpenAI-compatible API
  model: meta-llama3-8b-instruct
  apiKeySecret: kagent-openai
  apiKeySecretKey: OPENAI_API_KEY
  openAI:
    baseUrl: http://vllm-service.inference.svc.cluster.local:8000/v1
```

See the [Inference Examples](../inference/README.md) for deploying vLLM,
Ray Serve, or Triton in EKS.

### 2.2 OpenAI or compatible APIs

A placeholder `kagent-openai` secret is automatically created. Update it
with your API key:

```bash
kubectl create secret generic kagent-openai \
  --from-literal=OPENAI_API_KEY=<your-openai-api-key> \
  -n kagent \
  --dry-run=client -o yaml | kubectl apply -f -
```

```yaml
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: gpt-4
  namespace: kagent
spec:
  provider: OpenAI
  model: gpt-4
  apiKeySecret: kagent-openai
  apiKeySecretKey: OPENAI_API_KEY
  openAI:
    baseUrl: https://api.openai.com/v1
```

### 2.3 Amazon Bedrock

Enable IRSA for Bedrock:

```bash
terraform apply \
  -var="kagent_enabled=true" \
  -var="kagent_enable_bedrock_access=true"
```

```yaml
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: claude-sonnet
  namespace: kagent
spec:
  provider: Bedrock
  model: anthropic.claude-3-5-sonnet-20241022-v2:0
  region: us-west-2
```

When `kagent_enable_bedrock_access=true`, the module configures
`controller.serviceAccount.name=kagent-sa` and
`controller.serviceAccount.create=false` and attaches the Bedrock IAM
role via IRSA.

---

## 3. How the agents use memledger

memledger is the **memory governance and trust layer** sitting between
agents and a vector store (pgvector). Every memory carries an
attribution surface so that when one agent acts on another agent's
recalled memory you can answer:

- **Where did this knowledge come from?** — `created_by` + `session_id`
- **How sure was the originator?** — `confidence` (and `hedged` if
  speculative)
- **What chain led here?** — `derived_from` / `supersedes` resolved via
  `chain_store.build_chain()`
- **Is this still trustworthy at retrieval time?** — weakest-link rule:
  `effective_confidence = min(declared, chain.min_confidence)`

### 3.1 `triage-agent` — `/triage/*`

**Role**: ingest incoming alerts, correlate with past incidents,
escalate.

**Tools that touch memledger:**

- `ingest_alert(source, severity, description)` — searches
  `/ops/incidents` and `/triage/alerts` for correlations, then writes a
  `RecordType.EPISODIC` record under `/triage/alerts` with
  severity-driven `confidence` (critical=0.9, high=0.85, medium=0.6,
  low=0.4) and `triggered_by=<alert-id>`.
- `correlate_incidents(query, namespace)` — `search_hybrid()` against
  ops history; surfaces by-agent + severity metadata.
- `escalate_to_ops(summary, alert_id, severity)` — writes a record
  under `/shared/escalations` with `confidence=0.85`, `importance=0.7`,
  `triggered_by=<alert-id>`.
- `remember_knowledge(content, namespace, confidence, hedged,
  derived_from, supersedes, workflow_id, triggered_by, ...)` — full
  v1 trust attribution surface.
- `recall_knowledge` / `recall_context` — semantic + hybrid search.

**Namespaces it owns**: `/triage/alerts`, `/triage/incidents/...`,
`/triage/findings`, `/shared/escalations`.

### 3.2 `eks-ops-agent` — `/eks-ops/*`

**Role**: troubleshoot EKS clusters, recall remediations, learn from
incidents.

**Tools that touch memledger:**

- `recall_knowledge(query, namespace)` — searches across `/incidents/*`,
  `/runbooks`, `/learnings`, and (cross-agent) `/triage/incidents`. The
  reply prints **full record UUIDs** so the agent can pass them as
  `derived_from` in a follow-up `remember_knowledge`.
- `remember_knowledge(content, namespace, confidence, derived_from,
  supersedes, workflow_id, triggered_by, ...)` — typically writes to
  `/eks-ops/remediations/<service>` with
  `derived_from=[<triage_record_id>]` for cross-agent provenance.
- `mark_memory_outcome(description, success, record_id)` — wraps
  `record_outcome()`; flags whether a recalled memory led to a successful
  resolution. Increments `success_count`/`failure_count` columns.
- `set_default_cluster` / `set_default_namespace` /
  `clear_my_defaults` — store user-scoped defaults at
  `/users/<user_id>/defaults`.
- `memory_audit` / `memory_lineage` — read-only views over the audit
  log and provenance chain.

**Namespaces it owns**: `/eks-ops/remediations/...`, `/incidents/...`,
`/runbooks`, `/learnings`, `/users/<user_id>/defaults`.

### 3.3 `compliance-agent` — `/compliance/*` + cross-namespace audit

**Role**: enforce retention, scan for staleness, generate trust
audits.

**Tools that touch memledger:**

- `memory_audit(record_id, last_n)` — read the audit log of operations.
- `memory_lineage(description, record_id)` — full provenance: created
  by, derived_from, supersedes_chain, accessed_by.
- `scan_staleness(days)` — finds records not accessed for N days; writes
  a compliance report to `/compliance/reports`.
- `enforce_lifecycle(action, scope, namespace, days)` — bulk
  expire-stale / archive-expired / deprecate-conflicting via
  `bulk_update_status()` + `bulk_archive()`.
- `check_namespace_compliance(namespace)` — RBAC-gated read of a
  namespace; surfaces low-confidence/hedged records.
- `update_rbac_policy(namespace, rule)` — declarative namespace RBAC
  setup.

**Namespaces it owns**: `/compliance/reports`. Read access into all
other namespaces it has been granted under the RBAC policy.

### 3.4 A canonical multi-agent flow

```
                                          (memledger pgvector + Bedrock Titan)
                                                       ▲
   alert ── triage-agent ── ingest_alert ──────────────┼── /triage/alerts/<id>
                              │                        │      conf=0.9 (severity=critical)
                              │                        │      triggered_by=alert-...
                              ▼                        │
                          (correlation)                │
                              │                        │
                              ▼                        │
   recall      ◄── eks-ops-agent ◄─── recall_knowledge ┤
                              │      (cross-agent       │
                              ▼       /triage/incidents)│
                          remember_knowledge ──────────┼── /eks-ops/remediations/payment
                              │  derived_from=[triage]  │      conf=0.85
                              ▼                        │      derived_from=[<triage_id>]
                                                        │
                          mark_memory_outcome ─────────┼── (success_count++)
                                                        │
   audit ── compliance-agent ── memory_lineage ───────┤
                                                        │
                          scan_staleness ───────────────┴── /compliance/reports
                                                              conf=0.95
```

When `compliance-agent` calls `memory_lineage` on the eks-ops
remediation, memledger walks the chain and returns:

```text
chain_depth     = 2
min_confidence  = 0.85          # weakest-link across agent boundaries
agents_involved = ['eks-ops-agent', 'triage-agent']
hops:
  hop 0 origin   eks-ops-agent (conf 0.85) — remediation
  hop 1 derived  triage-agent  (conf 0.90) — root incident
```

### 3.5 SDK integration pattern (every agent's code)

Every agent in this directory follows the same shape. If you write a
new agent, mirror this:

1. **`app.py:_init_otel()`** — set up a real `TracerProvider` +
   `BatchSpanProcessor` + `OTLPSpanExporter` pointed at the cluster
   OTEL collector on `:4317`. Without this the global tracer is a
   no-op `ProxyTracerProvider` and memledger spans are lost.

   ```python
   def _init_otel():
       import os
       endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
       if not endpoint:
           return
       from opentelemetry import trace
       from opentelemetry.sdk.trace import TracerProvider
       from opentelemetry.sdk.trace.export import BatchSpanProcessor
       from opentelemetry.sdk.resources import Resource
       from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

       grpc_endpoint = endpoint.replace(":4318", ":4317").replace("http://", "")
       resource = Resource.create({"service.name": "my-agent"})
       provider = TracerProvider(resource=resource)
       provider.add_span_processor(BatchSpanProcessor(
           OTLPSpanExporter(endpoint=grpc_endpoint, insecure=True)
       ))
       trace.set_tracer_provider(provider)
   ```

2. **`memory.py:MemoryService._get_*()`** — lazy-create a `Memledger`
   instance with `EmbeddingConfig(provider='bedrock',
   model='amazon.titan-embed-text-v2:0', dimensions=1024)`, then call
   `memledger.telemetry.instrument_engram(self._ml)`. v1's tracing is
   **opt-in at the instance level** by design.

   ```python
   from memledger import Memledger
   from memledger.embeddings import EmbeddingConfig
   from memledger.telemetry import instrument_engram

   self._ml = await Memledger.create(
       backend_name="pgvector",
       connection_string=self._pg_dsn,
       embedding_config=EmbeddingConfig(
           provider="bedrock",
           model="amazon.titan-embed-text-v2:0",
           dimensions=1024,
       ),
   )
   instrument_engram(self._ml)  # opt-in OTEL wrapping
   ```

3. **`memory.py:remember_knowledge`** (the `@tool`) — accept and
   forward the full v1 trust kwargs (`confidence`, `hedged`,
   `derived_from`, `supersedes`, `agent_id`, `created_by`,
   `workflow_id`, `triggered_by`).

4. **`memory.py:recall_knowledge`** — print **full record UUIDs** in
   the response so downstream agents can pass them as `derived_from`.
   Truncated 8-char prefixes break cross-agent linking.

---

## 4. Memory Attribution Integrity (MAI) scoring

memledger v1 ships a 3-tier MAI evaluator suite. All three tiers score
against the same canonical rubric:

> **Score 1 (well-attributed) when:**
> - Retrieved memories have attribution (source agent, confidence,
>   session)
> - Memory confidence ≥ 0.7 OR decision explicitly hedges on
>   low-confidence data
> - No memories in chain with confidence < 0.4 used as basis for the
>   decision
> - Derivation chains are present and consistent
>
> **Score 0 (unattributed) when:**
> - Decision uses unattributed or low-confidence memory as ground truth
> - Contradictory memories ignored
> - Memory without session/turn context treated as authoritative

### 4.1 Tier comparison

| Tier | Implementation | When to use | Cost | Latency |
|------|----------------|-------------|------|---------|
| A — deterministic | `evaluators.attribution_integrity.evaluate_attribution_integrity` (pure-Python rules over the record list and chain) | CI guardrail on every commit | $0 | < 10 ms |
| B — structural | `evaluators.attribution_integrity_structural.evaluate_from_memory_records` (5-criterion span-shape check) | CI guardrail with finer-grained criterion explanations | $0 | < 10 ms |
| C — RAGAS LLM-as-judge | `evaluators.evaluate_mai_ragas` (RAGAS `AspectCritic`, judge LLM via LiteLLM) | Pre-merge gate on attribution-sensitive PRs; production attribution scoring | judge token cost | 2–5 s/record set |

For tier C, the judge model is selected via `MEMLEDGER_JUDGE_MODEL`
(LiteLLM-routed, provider-agnostic). Validated values:

```
bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0
openai/gpt-4o-mini
anthropic/claude-3-5-sonnet-20241022
ollama/llama3.1
```

### 4.2 Evaluator results against the validation fixture set

| Scenario | Description | Ground truth | Tier A | Tier B | Tier C |
|----------|-------------|--------------|--------|--------|--------|
| w01 | Single high-confidence record (conf=0.92), full attribution | well_attributed | 1.000 PASS ✓ | 1.000 PASS ✓ | 1.000 PASS ✓ |
| w02 | Hedged speculative record (conf=0.30), no chain | poorly_attributed | 0.875 PASS ✗ | 0.800 PASS ✗ | **0.000 FAIL ✓** |
| w03 | Multi-agent chain (triage→eks-ops), conf=0.80, 0.85 | well_attributed | 1.000 PASS ✓ | 1.000 PASS ✓ | 1.000 PASS ✓ |

Tier C correctly identifies the w02 poorly-attributed case that tiers A
and B let through at the default 0.7 threshold. This is the intended
3-tier design — cheap rules first, LLM-judge as the tiebreaker for
cases where the rules are too lenient.

---

## 5. Phoenix observability

Each memledger SDK call (`add`, `search`, `get`, `record_outcome`,
`build_chain`, `conflict_detected`) emits an OpenTelemetry span with
trust attributes attached. Phoenix categorizes these by
`openinference.span.kind` (`RETRIEVER` for `add`/`search`/`get`).

### 5.1 Span inventory

| Span name | When emitted | Key attributes |
|-----------|--------------|----------------|
| `memledger.memory.add` | Every `Memledger.add()` | `memledger.memory.confidence`, `.hedged`, `.source_agent_id`, `.namespace`, `.parent_ids`, `.session_id`, `.id`, `openinference.span.kind=RETRIEVER` |
| `memledger.memory.search` | Every `Memledger.search()` | `memledger.search.query`, `.namespace`, `.requester_id`, `.records_returned`, `.confidence_gating.passed/flagged/filtered`, `openinference.span.kind=RETRIEVER` |
| `memledger.get` | `Memledger.get(record_id)` and chain hops | `memledger.record_id`, `.namespace`, `.confidence` |
| `memledger.conflict_detected` | When an `add()` finds a near-duplicate | `memledger.conflict.existing_id`, `.similarity` |
| `evaluators.mai_deterministic` | Tier A evaluator run | `evaluator.score`, `.passed`, `.threshold` |
| `evaluators.mai_ragas` | Tier C RAGAS evaluator run | `evaluator.score`, `.judge_model`, `.passed` |

### 5.2 Sample span attribute dump

Captured from a hedged speculative `add()` during validation:

```text
name=memledger.memory.add  span_kind=RETRIEVER
  openinference.span.kind            = RETRIEVER
  memledger.memory.operation         = add
  memledger.memory.type              = episodic
  memledger.memory.namespace         = /triage/incidents/payment-cache
  memledger.memory.source_agent_id   = triage-agent
  memledger.memory.confidence        = 0.3
  memledger.memory.hedged            = True
  memledger.memory.parent_ids        = []
  memledger.memory.id                = d4275482-e1ec-4f2a-b52f-d981b5b2bc20
```

### 5.3 Viewing Phoenix in your cluster

```bash
kubectl port-forward -n kagent svc/phoenix 6006:6006
# open http://localhost:6006 in a browser
```

Filter the Sessions view by `service.name in (triage-agent,
eks-ops-agent, compliance-agent)` and span kind `RETRIEVER` to focus on
memledger operations.

---

## 6. Cluster integration procedure

This section describes deploying memledger v1.0.0 to the cluster, then
rebuilding the three agent images against the v1 SDK and running the
end-to-end smoke tests.

### 6.1 Prerequisites

Tools on your workstation:

- `kubectl` >= 1.30
- `helm` >= 3.14
- `awscli` v2 with credentials for the EKS account
- `docker` with buildx (for the agent image rebuild)
- `jq`

Environment:

```bash
export AWS_REGION=us-west-2
export AWS_PROFILE=<your-profile>
export CLUSTER_NAME=<your-cluster-name>
```

### 6.2 Connect to the cluster

```bash
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION"

# Sanity check
kubectl get nodes
kubectl get pods -n kagent | grep -E "kagent|memledger|phoenix|otel"
```

Nodes should be `Ready`, kagent and phoenix pods Running. If any pod is
`CrashLoopBackOff`, fix that before proceeding.

### 6.3 Helm-install memledger v1.0.0

The chart lives in the [memledger-core
repo](https://github.com/memledger-ai/memledger-core) under
`charts/memledger`. Clone it locally and point `$MEMLEDGER_CHART` at the
chart directory.

```bash
export MEMLEDGER_CHART=/path/to/memledger-core/charts/memledger

helm upgrade --install memledger "$MEMLEDGER_CHART" \
  --namespace kagent \
  --set database.deploy=true \
  --set database.migration.enabled=true \
  --set database.migration.tableName=agent_memory \
  --set database.migration.vectorDimensions=1024 \
  --set embeddings.provider=bedrock \
  --set embeddings.model=amazon.titan-embed-text-v2:0 \
  --set embeddings.dimensions=1024 \
  --set memledger.defaultBackend=pgvector \
  --wait --timeout 5m
```

Expected: `STATUS: deployed`. Verify the schema:

```bash
kubectl exec -n kagent memledger-pgvector-0 -- \
  psql -U memledger -d memledger -c "\d agent_memory"
```

Should show 23 columns including `embedding vector(1024)`,
`confidence`, `hedged`, `derived_from text[]`, `supersedes`,
`created_by`, `workflow_id`, `triggered_by`, etc., plus an HNSW index
on `embedding`.

> **External Postgres alternative**: set `database.deploy=false` and
> point at an existing instance via `database.host`, `database.port`,
> `database.existingSecret`, `database.secretKey`. v1.0.0 has been
> validated against the in-cluster pgvector StatefulSet only. Aurora
> validation is reserved for v2.

### 6.4 Build & deploy the three agent images

Each agent has its own `build-and-deploy.sh` that builds a `linux/amd64`
image, pushes to ECR (auto-creates repo if missing), and runs `helm
upgrade --install` for the agent chart with IRSA-annotated
ServiceAccount.

```bash
export TF_DIR=/path/to/eks-cluster/terraform/aws-eks-cluster-and-nodegroup
export REPO_DIR=/path/to/this/repo
export MEMLEDGER_PG_DSN="postgresql://memledger:memledger-secret@memledger-pgvector:5432/memledger"

# eks-ops-agent — pgvector mode with EKS MCP tools
cd "$REPO_DIR/examples/agentic/eks-ops-agent"
ENABLE_MCP_TOOLS=true ENABLE_MEMORY=true ./build-and-deploy.sh

# triage-agent
cd "$REPO_DIR/examples/agentic/triage-agent"
./build-and-deploy.sh

# compliance-agent
cd "$REPO_DIR/examples/agentic/compliance-agent"
./build-and-deploy.sh
```

Expected: three `Push complete` lines and three `Deploy complete!`
banners. Pods should reach `1/1 Running` in `kagent` namespace.

If you re-run the build with the same `VERSION` tag, the kubelet may
keep the cached image (`imagePullPolicy: IfNotPresent`). Either bump
`VERSION=...` inline or pin the deployment to the new digest:

```bash
DIGEST=$(grep 'digest:' /tmp/build.log | head -1 | awk '{print $3}')
kubectl set image -n kagent deployment/eks-ops-agent \
  kagent=489829964455.dkr.ecr.us-west-2.amazonaws.com/eks-ops-agent@${DIGEST}
```

### 6.5 Apply agent CRDs (optional, for prompt-only edits)

The build scripts already render the kagent `Agent` CRD via Helm. To
re-apply the CRD-only YAML manually after editing an agent prompt:

```bash
kubectl apply -f "$REPO_DIR/examples/agentic/triage-agent/triage-agent.yaml"
kubectl apply -f "$REPO_DIR/examples/agentic/eks-ops-agent/eks-ops-agent.yaml"
kubectl apply -f "$REPO_DIR/examples/agentic/compliance-agent/compliance-agent.yaml"
```

Verify with `kubectl get agents -n kagent`.

### 6.6 Smoke tests

#### 6.6.1 SDK smoke (in-pod)

```bash
POD=$(kubectl get pod -n kagent -l kagent=eks-ops-agent -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n kagent "$POD" -- python - <<'PY'
import asyncio, os
from memledger import Memledger
from memledger.embeddings import EmbeddingConfig

async def main():
    cfg = EmbeddingConfig(
        provider="bedrock",
        model="amazon.titan-embed-text-v2:0",
        dimensions=1024,
    )
    async with await Memledger.create(
        connection_string=os.environ["MEMLEDGER_PG_DSN"],
        embedding_config=cfg,
    ) as ml:
        a = await ml.add("smoke: root knowledge",
                         namespace="/smoke", confidence=0.9)
        b = await ml.add("smoke: derived insight",
                         namespace="/smoke", derived_from=[a], confidence=0.7)
        c = await ml.add("smoke: superseding insight",
                         namespace="/smoke", supersedes=b, derived_from=[b], confidence=0.8)

        results = await ml.search("smoke", namespace="/smoke", top_k=5)
        chain = await ml.chain_store.build_chain(c, direction="upstream")
        print(f"records={len(results.records)} chain_depth={chain.chain_depth} "
              f"min_confidence={chain.min_confidence:.2f} "
              f"agents={chain.agents_involved}")

asyncio.run(main())
PY
```

Expected: `records=2 chain_depth=3 min_confidence=0.70 agents=['unknown']`
(or similar — the weakest link is `b`'s 0.7).

#### 6.6.2 CLI smoke (in-pod)

```bash
kubectl exec -n kagent "$POD" -- memledger status
kubectl exec -n kagent "$POD" -- memledger get <record_id> --chain
```

`status` reports connection + memory count. `get --chain` renders the
provenance chain with hop-by-hop attribution.

> **CLI caveat**: `memledger add` and `memledger search` in v1.0.0 are
> hardcoded to local fastembed. For Bedrock-flavored ad-hoc queries,
> use the Python SDK directly with `EmbeddingConfig(provider='bedrock',
> ...)`.

#### 6.6.3 Cross-agent flow via A2A JSON-RPC

Each agent exposes an A2A JSON-RPC endpoint on `:8080`. To drive a
multi-agent flow without the kagent UI, port-forward and `curl`:

```bash
kubectl port-forward -n kagent svc/triage-agent 18001:8080 &
kubectl port-forward -n kagent svc/eks-ops-agent 18002:8080 &
kubectl port-forward -n kagent svc/compliance-agent 18003:8080 &

# Send a message to the triage agent
curl -s http://localhost:18001/ -H 'Content-Type: application/json' -d @- <<'JSON'
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "message/send",
  "params": {
    "message": {
      "kind": "message",
      "messageId": "msg-1",
      "role": "user",
      "parts": [{"kind": "text", "text": "Ingest an OOM alert for payment-service. Use remember_knowledge to store it under /triage/incidents/payment-oom with confidence 0.9. Return the memory ID."}]
    }
  }
}
JSON
```

Then drive eks-ops to recall and add a derived remediation, and
compliance to audit the chain.

#### 6.6.4 Phoenix span verification

```bash
kubectl port-forward -n kagent svc/phoenix 16006:6006
# open http://localhost:16006 in a browser
```

After the cross-agent flow, Phoenix should show `memledger.memory.add`,
`memledger.memory.search`, `memledger.get`, and (if you ran the
evaluators) `evaluators.mai_*` spans. Filter by span kind `RETRIEVER`.

### 6.7 Rollback

```bash
helm rollback memledger 0
```

The agent_memory schema is backward-compatible at the column level —
existing 0.4.0 records remain readable from v1.0.0 SDK with empty
attribution columns.

### 6.8 Expected outputs cheat sheet

| Step | Pass | Fail |
|------|------|------|
| 6.2 | Nodes Ready, kagent / phoenix / otel pods Running | Any `CrashLoopBackOff` or `ImagePullBackOff` — fix before proceeding |
| 6.3 | `helm` STATUS deployed; migration job `Complete`; `\d agent_memory` shows the v1 column set | Migration job failure → check `kubectl logs -n kagent job/memledger-db-migration` |
| 6.4 | Three `Push complete` + three `Deploy complete!` | ECR auth fail → re-run `aws ecr get-login-password`; build context size 290 B → run `docker buildx prune -af` |
| 6.5 | `agent.kagent.dev/* configured` x3 | CRD missing → ensure kagent controller is up |
| 6.6.1 | `records>=2 chain_depth>=2` | Import error → v1.0.0 wheel didn't make it into the image; re-build |
| 6.6.2 | `status` shows connected + memory count; `get --chain` renders | CLI not on PATH → `pip install memledger==1.0.0` baked into image |
| 6.6.4 | Phoenix shows `memledger.*` spans | No spans → `_init_otel()` not called; or `instrument_engram(self._ml)` missing |

---

## 7. Where to find things

| Path | What it is |
|------|------------|
| `triage-agent/`, `eks-ops-agent/`, `compliance-agent/` | Reference agent implementations |
| `eks-ops-agent/v2-preview/` | DynamoDB + OpenSearch composition reservation for v2 (not used in v1) |
| memledger SDK | <https://pypi.org/project/memledger/> |
| memledger Helm chart | `charts/memledger/` in [memledger-core](https://github.com/memledger-ai/memledger-core) |
