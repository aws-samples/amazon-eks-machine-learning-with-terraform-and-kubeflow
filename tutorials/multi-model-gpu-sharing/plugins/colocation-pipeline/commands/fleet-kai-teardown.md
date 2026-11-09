---
description: Take a KAI deployment down and wait until every one of its pods has departed
argument-hint: <deployment name>
allowed-tools: [Bash]
---

From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
PYTHONPATH=src python3 -m colocation.kai teardown $ARGUMENTS
```

Uninstalls every Helm release labelled `colocation.kai/deployment=<name>` — one per tenant, then the
plan's — deletes anything still carrying that label, its load Jobs included, and returns only once no pod of it is left, so "torn down" means the fractions are
free. A split is one deployment per node (`<name>-n<i>`): tear each down by its own name. If pods are
still there after 900 s it exits 1 saying how many; report that rather than
deleting them by hand. The KAI node stays until Karpenter consolidates it, about an hour after it empties; say that
rather than deleting the node claim.
