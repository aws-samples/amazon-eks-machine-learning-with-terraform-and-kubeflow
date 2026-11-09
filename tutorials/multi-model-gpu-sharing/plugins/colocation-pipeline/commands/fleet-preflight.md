---
description: Check what has to be true before a measurement round — one DCGM exporter, the GPU node held, the harness image, the Job template's cluster fields
argument-hint: (no arguments)
allowed-tools: [Bash, Read]
---

From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
bash plugins/colocation-pipeline/scripts/preflight.sh
```

Read-only: it installs, deletes and modifies nothing. Exit 1 means it found something that would
spoil a run.

Report each problem it names and stop there. **Do not work around one.** Each of these has already
cost a run, and each fails quietly rather than loudly:

- **A second DCGM exporter** — a GPU allows one host engine to own its profiling fields, so the
  second to ask crash-loops and every record of the run it was instrumenting comes back with no
  `per_gpu` block. The run reports success having lost every cause signal.
- **An unheld node** — reclaimed in about 600 s, and the replacement's GPU UUIDs make every stored
  placement un-reusable.
- **A wrong field in the harness Job template** — the Job sits Pending with no event explaining why,
  while an engine holds GPUs.

The image line is not a problem either way: absent simply means the first `profile` or `search`
builds and pushes it.
