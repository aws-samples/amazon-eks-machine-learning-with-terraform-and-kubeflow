---
description: Record how far a KAI deployment's real traffic and latency have drifted from the fleet file, per tenant and per hour
argument-hint: <deployment name>
allowed-tools: [Bash, Read]
---

From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
PYTHONPATH=src python3 -m colocation.kai drift $ARGUMENTS
```

Read-only against the cluster: it asks the cluster Prometheus for every whole hour since the deploy
that is not recorded yet, appends one JSON line per tenant per hour to `drift/<name>.jsonl` beside
the export's directory, and pushes the file to EFS. `no whole hour to record yet` is not a fault.

Report, per tenant, what the printed lines say: the request rate and the prompt and output lengths
against the fleet file (`vs fleet`), and ITL mean and TTFT p95 against the target (`vs target`).
`no traffic` is an idle hour, not zero drift. `too few requests` has no p95. TTFT p95 comes from
vLLM's coarse histogram: report it as a trend, not as the harness's figure.

A node of a split is its own deployment, `<name>-n<i>`, and is recorded against its group's fleet
file.

It decides nothing, and neither do you: do not edit the fleet file or start a run because of it.
Say which tenants have moved and how far, and leave whether to re-run the pipeline to the user.
`recorded here but not pushed to EFS` exits `1` with the local file intact; report it.
