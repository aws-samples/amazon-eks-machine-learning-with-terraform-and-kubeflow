---
description: Search for the fewest-GPU placement that holds every tenant's stated ceilings, measuring co-resident arrangements on a leased node
argument-hint: <fleets/name.yaml> [--budget N]
allowed-tools: [Bash, Read, Grep]
---

This installs a daemon that requests every GPU on the node, and it may order a machine. From the
repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
IMG=$(bash plugins/colocation-pipeline/scripts/harness-image.sh --ensure) || exit 1
COLOCATION_HARNESS_IMAGE="$IMG" PYTHONPATH=src python3 -m colocation.agent search --fleet $ARGUMENTS \
  --summary /tmp/search-summary.json
```

Give the Bash call a `timeout` well above the default. Never kill it part-way: a killed process
leaves behind a Pending pod requesting every GPU on the node, and under Karpenter a Pending pod is a
purchase order. On an instance type nobody has leased, `--gpu-count N` makes this run observe the
machine and stop, as `/fleet-profile` describes; that is a type that has not been profiled yet
either, so profile it first.

Confirm all three before it runs — these are yours to check:

- The node is held by `gpu-node-holder` at priority `-1000`, so this run's daemon **preempts** it. If
  a default-priority pod holds the node the daemon cannot preempt it and a second machine is ordered
  instead: stop and say so. `/fleet-preflight` answers this.
- `plan` has been re-run since the last `profile`, and reports that a co-resident placement exists,
  or prints `BUT NOT ON THE WEIGHTS`, in which case `search` warns and runs anyway.
- The image comes from the script above, so it is built from the current source and its selftest
  passed.

**A fleet with `hardware.max_nodes` above 1 is searched group by group** (impl §17). Nothing measures
the parent as one fleet: each group of a split, one per node, is this same search on its own fleet
file under `<store>/groups/<fleet>/`, with its own lease, `--budget`, summary and export. The reasoner
chooses which group to search (`--split-chooser drive` uses the deterministic driver instead, which
is also the reasoner's recorded fallback), and `--group-budget` (default 4) is how many groups may
be searched. Leases are taken one node at a time, so pass `--teardown --release-node` and expect one
node in use, never two. A group an earlier run already searched against the same fleet text is
recalled at no budget cost (`chosen by the store`). The split's verdict comes from the groups' own
verdicts, whatever the reasoner submits. Report the `=== splits ===` block: each group's verdict, run
token and who chose it, plus `N group(s) were searched by the fallback` when that line is printed.
A met split writes `<store>/groups/<fleet>/export.<stamp>.yaml` (schema_version 2), and that file,
not a group's export, is what the split is handed on as. The split summary is `split.<stamp>.json`
beside it.

**Read every refusal instead of routing around it.** The `resolve_tp` refusals all mean one thing: no
usable solo measurement settles a degree, and the remedy is `/fleet-profile`. A fleet file cannot state
a `tp:` at all — that escape route was removed because a hand-written `tp: 4` inflated a baseline once.
A refusal naming `gpu_memory_utilization is X, but ... implies a device total of Y GiB instead`, or
`GPU <uuid> is promised N GiB of envelope`, is the render boundary re-checking the placement against
the device it is about to deploy to; the fraction it rejected would have been accepted by the engine
and echoed back unchanged, leaving only an undersized KV pool as evidence.

`capture` can also refuse the node it just got, with `is marked Drifted on its claim`. Not a transient:
the holder carried `do-not-disrupt`, and removing it is what let the cluster act on a drift it had been
waiting to act on. Let the replacement register, confirm the holder is on it, capture again. If what the
node reports about its GPUs changed, the run stops with `EnvironmentMoved` — the new machine is already
filed and the same command run again starts a fresh store. That is not a fault and needs no flag.

The node is **not** released unless `--release-node` is passed, and keeping the lease is the default
worth keeping: an idle node costs money, a released one may not come back. Either way the run prints the
exact sequence that hands it back: run it whole and do not shorten it. It is usually three commands
rather than two — a search deliberately ends with its tenants still serving, they are default priority,
and the holder is `-1000`, so uninstalling only the daemon leaves the holder Pending and the node
effectively unheld. Confirm the holder reaches Running, not merely created.

Then read the result, which is the block `search` printed at the end of its own run: `BASELINE`,
`STRICT`, `RELAXED`, and the `vs solo` block. `/fleet-inspect` does not reproduce it. The run exits
1 when it measured nothing at all.

The reasoner may cap a co-resident tenant's share of the SMs (`sm_caps`, an MPS active-thread
percentage from 1 to 99) on a measurement. A capped placement is a different placement from the same
one uncapped, so a summary row and the export carry `active_thread_percentage` for every capped
tenant, and the printed report shows `SM cap <tenant> N%` beside every capped placement it names.
Report the cap beside any result that has one.

The run's last act is to write that result in canonical form — slots instead of GPU UUIDs and
ordinals, both of which expire with the node — and to push the same file to EFS. Read the `written to`
lines and report the path; it is the form anything downstream should be given. If it printed
`the result is stored but was not exported`, run `/fleet-export`: it is free and offline, and nothing
was lost.
