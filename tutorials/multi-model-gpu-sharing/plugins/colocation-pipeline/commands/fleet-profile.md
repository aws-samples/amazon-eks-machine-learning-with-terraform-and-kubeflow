---
description: Measure each tenant alone at its full envelope to settle minimal TP, which is what turns the planned lower bound into a real baseline
argument-hint: <fleets/name.yaml> [--budget N] [--gpu-count N on a first run]
allowed-tools: [Bash, Read, Grep]
---

This captures a node and deploys engines on it. From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
IMG=$(bash plugins/colocation-pipeline/scripts/harness-image.sh --ensure) || exit 1
COLOCATION_HARNESS_IMAGE="$IMG" PYTHONPATH=src python3 -m colocation.agent profile --fleet $ARGUMENTS
```

Give the Bash call a `timeout` well above the default — rounds take minutes each.

**On an instance type nobody has leased**, it refuses without `--gpu-count N`, and with it the run
only observes the machine: it holds one node, files its shape and device facts, gives it back to the
holder and exits 0 having measured nothing. Run the same command again for the real round. That is
the only way to file a new type: `plan` cannot do it.

One deploy-and-drive round runs per (tenant, degree) that the store does not already hold and
arithmetic has not settled. `--budget` caps the number of rounds. Each round measures a tenant
**alone at its full envelope**, walking degrees upward from the memory floor and stopping at the
first pass.

Three outcomes take no round at all and are normal: `forced by memory alone` (the floor is already
the widest degree the node offers), `already profiled` (a usable row exists — this is why profiles
amortize across fleets), and a tenant with no `itl_mean_ms` (there is no criterion to judge a degree
against, so it is skipped).

The walk widens on evidence about the degree, and only on that: a measured miss, a degree the
arithmetic would not place, an engine that failed to boot twice at the same degree (the first failure
is retried automatically), or a row that shows the server driven flat out and too narrow.

**Anything else that comes back unusable stops the walk rather than widening it. Do not fight
this.** An unusable row — no ITL, `delivery_passed: false`, a verdict other than `measured`, sampling
too thin — did not say the degree is too slow. It said the degree *was not measured*. Fix the cause
and run the same command again: on the way up, a degree with no usable row is measured again
without `--reprofile`. (Walking *down* from a degree that passed is the exception: there a stored
unusable row stops the descent until `--reprofile`, and the log says so.) Widening buys the baseline
a wider shard that nothing ruled the narrower one out at.

It exits 1 when any tenant's degree is still unresolved, which is what the stop above leaves.

Afterwards:

- If `sum(minimal tp)` now exceeds the node's GPU count, **there is nothing to search**: the disjoint
  arrangement being scored against does not fit. Report it and go back to `/fleet-sweep`.
- Otherwise run **`/fleet-plan` again** before `/fleet-search`. The degrees moved, so the baseline and
  every placement moved with them.
- The per-shard MPS reserve a round measured needs no paste anywhere. It is filed in the profile store
  and merged into every later load of that model at that degree, so re-running `plan` is what picks it
  up. Report the number and which envelopes it moved.
