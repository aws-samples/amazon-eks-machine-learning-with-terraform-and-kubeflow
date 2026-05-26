# v2-preview — DynamoDB + multi-backend composition (memledger v2.1)

This directory holds artifacts for memledger features that are **not in
v2.0**. They land together in **memledger v2.1 on 2026-06-15**:

- DynamoDB backend
- True multi-backend composition (e.g. DynamoDB primary + OpenSearch
  search index in a single Memledger instance)

Until then, do not wire these into the production build. The agents in
this repo run on memledger 2.0 with one of the three single-backend
choices documented in the agent README:

| Backend | Auth | Notes |
|---|---|---|
| Postgres + pgvector | password | OSS, in-cluster |
| Aurora + pgvector | IAM (`rds-db:connect`) | AWS-native, no password |
| OpenSearch | SigV4 (IRSA) | hybrid search (vector + BM25) |

## Why is composition deferred?

memledger 2.0's `Memledger.from_config()` only honors `default_backend`
in the YAML — there is no router that fans add/search across multiple
backends. The composition file format works *syntactically* but
collapses to a single backend at runtime.

DynamoDB on its own can't back the agents either: the v2.0 DynamoDB
backend has no `search()` method (it is positioned as a key-value
companion to a search backend). Agent tools like `recall_knowledge` /
`recall_context` would fail at runtime.

Lane 1's call (2026-05-26): ship 3 backends cleanly in v2.0; bundle
DynamoDB + composition in v2.1 when the multi-backend router lands.

## Files

| File | Purpose |
|------|---------|
| `memledger-composition-aws.yaml` | Composition config: DynamoDB primary + OpenSearch faiss + Aurora pgvector fallback. Will be honored end-to-end in v2.1. |
| `setup-memledger-composition.sh` | Provisions DynamoDB table, OpenSearch domain, and IAM policy bindings. Idempotent. |
| `memledger-composition-policy.json` | IAM policy template granting DynamoDB + OpenSearch + Aurora IAM-auth access. Replace `<REGION>`, `<ACCOUNT-ID>`, `<DB-RESOURCE-ID>` placeholders before applying. |

## Promotion plan (v2.1, 2026-06-15)

When v2.1 ships:

1. Move `memledger-composition-aws.yaml` back to the agent root and add
   a second `COPY` line in `Dockerfile`.
2. Bump `MEMLEDGER_VERSION=2.1.0` and add `dynamodb` to `MEMLEDGER_EXTRAS`
   in `build-and-deploy.sh`.
3. Substitute placeholders in `memledger-composition-policy.json`,
   then run `setup-memledger-composition.sh` to provision AWS resources
   and attach the policy to the agent's IRSA role.
4. In `eks-ops-agent.yaml`, uncomment `MEMLEDGER_DDB_TABLE` and
   `OPENSEARCH_ENDPOINT`, and switch `MEMLEDGER_CONFIG_PATH` to
   `/app/memledger-composition-aws.yaml`.
5. Re-run the backend smoke tests to validate the composed path
   (DynamoDB add/get + OpenSearch search returning records that were
   written through DynamoDB).
