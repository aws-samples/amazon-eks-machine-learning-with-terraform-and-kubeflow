# v2-preview

Files reserved for **memledger v2** (AWS SDK integration). **Not used by v1.**

memledger v1.0.0 ships with pgvector + bedrock support only. v2 will add
native DynamoDB and OpenSearch backends, at which point the files in this
directory get promoted back into the agent's build path.

| File | Purpose |
|------|---------|
| `memledger-composition.yaml` | Composition config: DynamoDB primary + OpenSearch search index |
| `setup-engram-composition.sh` | Provisions DynamoDB table, OpenSearch domain, and IAM policy |

To activate in v2 (when the SDK lands):

1. Move `memledger-composition.yaml` back to the agent root.
2. Restore `COPY memledger-composition.yaml /app/memledger-composition.yaml` in `Dockerfile`.
3. Add the `dynamodb,opensearch` extras to the `memledger[...]==<v2>` install.
4. Run `setup-engram-composition.sh` to provision AWS resources.
5. Set `MEMLEDGER_CONFIG_PATH=/app/memledger-composition.yaml` on the agent pod.
