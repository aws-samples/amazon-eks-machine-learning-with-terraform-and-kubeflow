---
description: Ask offline whether the demand written in a fleet file is serveable at all, before a measurement round finds out
argument-hint: <fleets/name.yaml> [--tenant NAME] [--rates 1.0,0.7,...] [--set TENANT.FIELD=VALUE]
allowed-tools: [Bash, Read, Grep]
---

From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
python3 tools/demand_sweep.py --fleet $ARGUMENTS
```

By default it sweeps the rate of every tenant whose prediction misses at its memory floor, at the
multipliers 1.0, 0.7, 0.6, 0.5 and 0.4. `--tenant` (repeatable) picks the tenants to sweep, `--rates`
the multipliers, and `--set chat.rate_hz=1.5` fixes an override on every row. On an instance type
nobody has leased it refuses exactly as `plan` does, with the same remedy.

Run it whenever a rate is new or a target looks tight — that is, before every first `/fleet-profile`
of a fleet. It answers the question `plan` does not: is this demand serveable at all? Skipping it
means learning arithmetic from a measurement round.

Its memory half is exactly as trustworthy as `plan`'s — the same `solver/` functions. **Its latency
half is a prediction**, marked `~` in every row and fitted per model in the `FIT` table. Treat it as
a ranking of candidate demands, never as a result.

- **`THE BASELINE DOES NOT FIT`** — the disjoint arrangement needs more GPUs than the node has.
  Co-location cannot rescue this. **Stop and report**: demand comes down, or a tenant leaves.
- **`no placement solved`, or a tenant listed as unpredicted** — an MoE off its measured degree, or a
  model absent from `FIT`. Report it as *unpredictable, not infeasible*: the sweep is declining to
  guess.
- **`cap N vs M decoding at target`** — the batch cap against the decoders the target implies. The
  cap is derived from the rate, the SLO and the engine's own default, sized on a whole slot with the
  prefill queue included, so it cannot come out below this number. There is nothing to set: if a row
  shows the cap losing, the derivation is wrong. Report it rather than editing the fleet.
- A row that lands only *without* the co-residency factor (1.27× anchor, 2.04× stacked rider) is a
  predicted co-resident miss. Say so.

**When demand has to change, change the rate — never the target.** Relaxing an ITL ceiling moves the
goalposts *and* raises the KV floor, so it changes the memory geometry at the same time. Lowering the
rate lowers both honestly. A target that genuinely has to move is the user's call.

If you do change a fleet file, read the comments around the change before moving on: a rate, a shape
or a target that moves usually makes the sentence next to it wrong. A fleet file's comments are the
author's reasons, not a cache of arithmetic — report the geometry `plan` prints, do not write it back
into the file. Then run `/fleet-plan` again.
