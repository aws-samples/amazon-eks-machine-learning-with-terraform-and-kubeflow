---
description: Print one stored placement record, to check a single measurement
argument-hint: <candidate-record-id> --key <validity-key-hash> | <path/to/record.json> [--json] [--ordinals]
allowed-tools: [Bash, Read]
---

From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
PYTHONPATH=src python3 -m colocation.inspect record $ARGUMENTS
```

The id is **per candidate** (`search-rqolmi-001-chat-solo-4gpu`), not the run id. A run id gets you
`no record ... locally or on EFS`. A path to a record's `.json` needs no `--key`. `--json` dumps the
raw record; `--ordinals` labels GPUs by host ordinal instead of by UUID prefix, which for an older
record without `provenance.gpu_ordinal_by_uuid` needs the cluster and the node to still exist.

This prints one placement's record. It is for checking a measurement, not for reading an outcome: the
`BASELINE / STRICT / RELAXED` block belongs to `search`'s own output and its `--summary` JSON, and
`inspect` does not reproduce it.

`inspect` has no view of the hardware store, so filed shapes and device totals are only readable as raw
JSON under `~/.colocation/store/hwfacts/` (and `/efs/colocation/hwfacts/`, which is readable only from
inside a pod). You should not need to: `plan`'s header prints the total it planned against (`N GPUs,
X.XX GiB each`), and on an uncaptured node the load log names where that figure came from. Quote those
rather than a number dug out of the store.
