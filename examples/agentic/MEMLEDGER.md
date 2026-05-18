# Multi-agent memory governance with memledger v1.0.0

This guide explains how the three reference agents in this directory —
**triage-agent**, **eks-ops-agent**, and **compliance-agent** — use
[memledger](https://pypi.org/project/memledger/) to coordinate memory
across agent boundaries with attribution, weakest-link provenance,
confidence-gated retrieval, namespace RBAC, and OpenTelemetry
observability into Arize Phoenix.

For infrastructure setup (kagent install, Terraform) see
[`README.md`](README.md). For the cluster-side upgrade procedure to
memledger v1.0.0 see [`B9-RUNBOOK.md`](B9-RUNBOOK.md). For the raw
end-to-end validation results see
[`B9-RESULTS-2026-05-17.md`](B9-RESULTS-2026-05-17.md).

---

## What memledger gives the agent fleet

memledger is the **memory governance and trust layer** that sits between
agents and a vector store. Every memory carries an attribution surface
(`created_by`, `confidence`, `session_id`, `derived_from`, `supersedes`,
`workflow_id`, `triggered_by`, `hedged`, `namespace`), so when one agent
acts on another agent's recalled memory you can answer:

- **Where did this knowledge come from?** — `created_by` + `session_id`
- **How sure was the originator?** — `confidence` (and `hedged` if
  speculative)
- **What chain led here?** — `derived_from` / `supersedes` resolved
  through `chain_store.build_chain()`
- **Is this still trustworthy at retrieval time?** — weakest-link rule:
  `effective_confidence = min(declared, chain.min_confidence)`

The three agents below each use memledger differently, but they all
write into the same `agent_memory` table and read from each other's
namespaces under explicit RBAC.

---

## The three agents

### `triage-agent` — `/triage/*`

**Role**: ingest incoming alerts, correlate against past incidents, escalate.

**Tools that touch memledger**:
- `ingest_alert(source, severity, description)` — searches
  `/ops/incidents` and `/triage/alerts` for correlations, then writes a
  `RecordType.EPISODIC` record under `/triage/alerts` with severity-driven
  `confidence` (critical=0.9, high=0.85, medium=0.6, low=0.4) and
  `triggered_by=<alert-id>`.
- `correlate_incidents(query, namespace)` — `search_hybrid()` against
  ops history; surfaces by:agent + severity metadata in the result.
- `escalate_to_ops(summary, alert_id, severity)` — writes a record under
  `/shared/escalations` with `confidence=0.85`, `importance=0.7`,
  `triggered_by=<alert-id>`.
- `remember_knowledge(content, namespace, confidence, hedged,
  derived_from, supersedes, workflow_id, triggered_by, ...)` — full v1
  trust attribution surface; defaults to `/triage/alerts`.
- `recall_knowledge` / `recall_context` — semantic + hybrid search
  with full per-record metadata in the response.

**Namespaces it owns**: `/triage/alerts`, `/triage/incidents/...`,
`/triage/findings`, `/shared/escalations`.

### `eks-ops-agent` — `/eks-ops/*`

**Role**: troubleshoot EKS clusters, recall remediations, learn from
incidents.

**Tools that touch memledger**:
- `recall_knowledge(query, namespace)` — searches across `/incidents/*`,
  `/runbooks`, `/learnings`, and (cross-agent) `/triage/incidents`.
  Reply prints **full record UUIDs** so the agent can pass them as
  `derived_from` in a follow-up `remember_knowledge` call.
- `remember_knowledge(content, namespace, confidence, derived_from,
  supersedes, workflow_id, triggered_by, ...)` — typically writes to
  `/eks-ops/remediations/<service>` with
  `derived_from=[<triage_record_id>]` to capture cross-agent provenance.
- `mark_memory_outcome(description, success, record_id)` — wraps
  `record_outcome()`; lets the agent flag whether a recalled memory
  led to a successful resolution. Increments
  `success_count`/`failure_count` columns on the record.
- `set_default_cluster` / `set_default_namespace` / `clear_my_defaults`
  — store user-scoped defaults at `/users/<user_id>/defaults`.
- `memory_audit` / `memory_lineage` — read-only views over the audit
  log and provenance chain for a record.

**Namespaces it owns**: `/eks-ops/remediations/...`, `/incidents/...`,
`/runbooks`, `/learnings`, `/users/<user_id>/defaults`.

### `compliance-agent` — `/compliance/*` + cross-namespace audit

**Role**: enforce retention, scan for staleness, generate trust audits.

**Tools that touch memledger**:
- `memory_audit(record_id, last_n)` — read the audit log of operations.
- `memory_lineage(description, record_id)` — full provenance: created
  by, derived_from, supersedes_chain, accessed_by.
- `scan_staleness(days)` — finds records not accessed for N days; writes
  a compliance report to `/compliance/reports`.
- `enforce_lifecycle(action, scope, namespace, days)` — bulk
  expire-stale / archive-expired / deprecate-conflicting operations
  via `bulk_update_status()` and `bulk_archive()`. Writes a report.
- `check_namespace_compliance(namespace)` — RBAC-gated read of a
  namespace's records; surfaces low-confidence/hedged records.
- `update_rbac_policy(namespace, rule)` — declarative namespace RBAC
  setup.

**Namespaces it owns**: `/compliance/reports`. Read access (via
`requester_id=compliance-agent`) into all other namespaces it has been
granted under the RBAC policy.

---

## A canonical multi-agent flow

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
min_confidence  = 0.85          # weakest-link rule across agent boundaries
agents_involved = ['eks-ops-agent', 'triage-agent']
hops:
  hop 0 origin   eks-ops-agent (conf 0.85) — remediation
  hop 1 derived  triage-agent  (conf 0.90) — root incident
```

---

## Memory Attribution Integrity (MAI) — scoring rubric

memledger v1 ships a 3-tier evaluator suite. All three tiers score the
same canonical rubric:

> Score 1 (well-attributed) when:
> - Retrieved memories have attribution (source agent, confidence, session)
> - Memory confidence ≥ 0.7 OR decision explicitly hedges on low-confidence data
> - No memories in chain with confidence < 0.4 used as basis for decision
> - Derivation chains are present and consistent
>
> Score 0 (unattributed) when:
> - Decision uses unattributed or low-confidence memory as ground truth
> - Contradictory memories ignored
> - Memory without session/turn context treated as authoritative

| Tier | Implementation | When to use | Cost | Latency |
|------|---------------|-------------|------|---------|
| A — deterministic | `evaluators.attribution_integrity.evaluate_attribution_integrity` — pure-Python rules over the record list and chain | CI guardrail, every commit | $0 | <10 ms |
| B — structural | `evaluators.attribution_integrity_structural.evaluate_from_memory_records` — span-shape style 5-criterion check | CI guardrail with finer-grained criterion explanations | $0 | <10 ms |
| C — RAGAS LLM-as-judge | `evaluators.evaluate_mai_ragas` — RAGAS `AspectCritic` with the canonical rubric, judge LLM via LiteLLM | Pre-merge gate on attribution-sensitive PRs; production attribution scoring | judge token cost | 2–5 s/record set |

For tier C, the judge model is selected via `MEMLEDGER_JUDGE_MODEL`
(LiteLLM-routed, provider-agnostic). Validated values:

```
bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0
openai/gpt-4o-mini
anthropic/claude-3-5-sonnet-20241022
ollama/llama3.1
```

### Validation results against the w01 fixture set

| Scenario | Description | Ground truth | Tier A | Tier B | Tier C |
|----------|-------------|--------------|--------|--------|--------|
| w01 | Single high-confidence record (conf=0.92), full attribution | well_attributed | 1.000 PASS ✓ | 1.000 PASS ✓ | 1.000 PASS ✓ |
| w02 | Hedged speculative record (conf=0.30), no chain | poorly_attributed | 0.875 PASS ✗ | 0.800 PASS ✗ | **0.000 FAIL ✓** |
| w03 | Multi-agent chain (triage→eks-ops), conf=0.80, 0.85 | well_attributed | 1.000 PASS ✓ | 1.000 PASS ✓ | 1.000 PASS ✓ |

Tier C correctly identified the w02 poorly-attributed case that tiers A
and B let through at the default 0.7 threshold. This is the intended
3-tier design: cheap rules first, LLM-judge as the tiebreaker for cases
where the rules are too lenient.

---

## Phoenix observability

Each memledger SDK call (`add`, `search`, `get`, `record_outcome`,
`build_chain`, `conflict_detected`) emits an OpenTelemetry span with
trust attributes attached. Phoenix categorizes these by
`openinference.span.kind` (`RETRIEVER` for `add`/`search`/`get`).

### Span inventory

| Span name | When emitted | Key attributes |
|-----------|--------------|----------------|
| `memledger.memory.add` | Every `Memledger.add()` | `memledger.memory.confidence`, `.hedged`, `.source_agent_id`, `.namespace`, `.parent_ids`, `.session_id`, `.id`, `openinference.span.kind=RETRIEVER` |
| `memledger.memory.search` | Every `Memledger.search()` | `memledger.search.query`, `.namespace`, `.requester_id`, `.records_returned`, `.confidence_gating.passed/flagged/filtered`, `openinference.span.kind=RETRIEVER` |
| `memledger.get` | `Memledger.get(record_id)` and chain hops | `memledger.record_id`, `.namespace`, `.confidence` |
| `memledger.conflict_detected` | When an `add()` finds a near-duplicate | `memledger.conflict.existing_id`, `.similarity` |
| `evaluators.mai_deterministic` | Tier A evaluator run | `evaluator.score`, `.passed`, `.threshold` |
| `evaluators.mai_ragas` | Tier C RAGAS evaluator run | `evaluator.score`, `.judge_model`, `.passed` |

### Sample span attribute dump

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

### Viewing Phoenix in your cluster

```bash
kubectl port-forward -n kagent svc/phoenix 6006:6006
# open http://localhost:6006 in a browser
```

Filter the Sessions view by `service.name in (triage-agent, eks-ops-agent,
compliance-agent)` and span kind `RETRIEVER` to focus on memledger
operations.

> **Screenshots**: TODO. Capture during the next validation pass:
> 1. Phoenix Sessions view filtered by `memledger.*` span names showing
>    a single end-to-end trace from `a2a.server.request_handlers...on_message_send`
>    down through `execute_tool remember_knowledge` to
>    `memledger.memory.add` with the trust attributes panel expanded.
> 2. Phoenix span detail showing `memledger.memory.confidence`,
>    `memledger.memory.hedged`, `memledger.memory.source_agent_id`
>    visible in the right-hand attributes panel.
> 3. `memledger get <id> --chain` CLI output rendering a 2-hop
>    cross-agent chain with `min_confidence` highlighted.

---

## Wiring summary (what each agent's code does)

All three agents follow the same shape:

1. `app.py:_init_otel()` — set up a real `TracerProvider` +
   `BatchSpanProcessor` + `OTLPSpanExporter` pointed at the cluster
   OTEL collector on `:4317`. Without this the global tracer is a
   no-op `ProxyTracerProvider` and memledger spans are lost.
2. `memory.py:MemoryService._get_*()` — lazy-create a `Memledger`
   instance with `EmbeddingConfig(provider='bedrock',
   model='amazon.titan-embed-text-v2:0', dimensions=1024)`, then call
   `memledger.telemetry.instrument_engram(self._ml)`. v1's tracing is
   **opt-in at the instance level** by design.
3. `memory.py:remember_knowledge` (the `@tool`) — accepts and forwards
   the full v1 trust kwargs (`confidence`, `hedged`, `derived_from`,
   `supersedes`, `agent_id`, `created_by`, `workflow_id`,
   `triggered_by`).
4. `memory.py:recall_knowledge` — prints **full record UUIDs** in the
   response so downstream agents can pass them as `derived_from`
   (truncated 8-char prefixes used to break cross-agent linking).

---

## CLI quick reference

```bash
# Generate a starter config (writes ./memledger.yaml)
memledger init

# Connect status check (Aurora/pgvector + Bedrock + AgentCore + Phoenix)
memledger status

# Inspect a record
memledger get <record_id>

# Walk the provenance chain for a record (cross-agent, weakest-link)
memledger get <record_id> --chain

# Run the deterministic + structural MAI evaluators against a session
memledger eval <session_id>

# Run a single evaluator
memledger eval <session_id> --evaluator mai-a   # deterministic
memledger eval <session_id> --evaluator mai-b   # structural
```

> **CLI caveat**: `memledger add` and `memledger search` in v1.0.0 are
> hardcoded to local fastembed (the OSS-quickstart-first default) and
> don't honor the embeddings block in `memledger.yaml`. For Bedrock-flavored
> ad-hoc queries against the cluster's `agent_memory` table, use the
> Python SDK directly with `EmbeddingConfig(provider='bedrock', ...)`.

---

## Known v1.0.0 limitations and v1.1 follow-ups

| Area | What v1.0.0 ships | Recommendation for v1.1 |
|------|-------------------|--------------------------|
| Default MAI thresholds | Tier A + B default `threshold=0.7`; w02-style cases pass | Tighten to 0.85 OR weight `confidence` more heavily |
| Per-record FLAG signal | Aggregate counts in `SearchResult.metadata.confidence_gating`; no per-record `signal` field | Add `MemoryRecord.confidence_signal` (PASS/FLAG) |
| RBAC observability | Silent-filter on deny; `metadata.rbac` empty | Populate `metadata.rbac.{denied, reason}`; optional `raise PermissionError` mode |
| RBAC runtime updates | `Memledger.rbac` is read-only after `__init__` | Expose `set_rbac()` method or property setter |
| `record_outcome()` confidence update | Raw counts updated; `confidence` not auto-adjusted | Either auto-update with smoothing or document the calibration knob |
| CLI embedding config | Hardcoded to local fastembed | Read embeddings block from `memledger.yaml` or `MEMLEDGER_EMBEDDING_*` env |
| Helm chart Aurora path | In-cluster pgvector validated; Aurora untested | v2 will validate Aurora end-to-end |
| `memledger[eval]` extras in agent images | Not installed by default | Add to Dockerfile if continuous in-cluster eval is desired |

---

## Where to find things

- Agent code: `triage-agent/`, `eks-ops-agent/`, `compliance-agent/`
- Helm chart for memledger: `charts/memledger/` in the
  [memledger-core](https://github.com/memledger-ai/memledger-core)
  repo (path `/Users/aratnch/Documents/2026/code/memledger-ai/memledger-core/charts/memledger`
  on this workstation)
- Cluster upgrade procedure: [`B9-RUNBOOK.md`](B9-RUNBOOK.md)
- Validation results (raw): [`B9-RESULTS-2026-05-17.md`](B9-RESULTS-2026-05-17.md)
- v2 reservation (DynamoDB + OpenSearch composition): `eks-ops-agent/v2-preview/`
- memledger SDK docs: <https://pypi.org/project/memledger/>
