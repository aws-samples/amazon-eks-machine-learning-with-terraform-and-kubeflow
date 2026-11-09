---
description: Write the stored result in canonical form — slots instead of GPU UUIDs, absolute GiB, conditions by reference — so it outlives the node it was measured on
argument-hint: <fleets/name.yaml>
allowed-tools: [Bash, Read]
---

From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
PYTHONPATH=src python3 -m colocation.agent export --fleet $ARGUMENTS
```

Free and offline: it reads the store and the fleet file, spends no GPU time and touches no cluster
beyond the mirror write to EFS.

**A `search` already does this as it finishes**, through the same code, so this command is for the
cases where nothing did it for you: an older result, a different `--run-token` under the same key, a
key whose search predates the export, a fleet that has been profiled but never searched, or a run that
printed `the result is stored but was not exported`. It exports the baseline whether or not a search
found anything, because the baseline is arithmetic on measured degrees and therefore always exists.

It refuses a fleet with `hardware.max_nodes` above 1. A split is exported by `search` itself once
every group of it is met, as `<store>/groups/<fleet>/export.<stamp>.yaml` (schema_version 2: each
node's group export, copied whole). Report that path instead. A single group can still be exported
with this command by naming its fleet file under `<store>/groups/<fleet>/`.

It writes `observations/<validity-key-hash>/exports/<run-token>.yaml` in the local store and on EFS,
or `exports/baseline.yaml` for a key with no search at all. `--export-to <path>` puts a third copy
wherever you want one, which is what to use when the user asked to see the file. `--run-token <token>`
picks an older search under the same key; without it the newest is exported. A token the key does not
hold exits 1 and lists the tokens it does.

What the file says, and why each part is shaped the way it is:

- **slots, not devices.** A slot is one device with no identity — no UUID, no ordinal — carrying the
  `(tenant, shard)` pairs that sit on it. That is the part of a placement that survives the node:
  a UUID names hardware Karpenter reclaims in ten minutes, and an ordinal names a position in one
  driver's enumeration order.
- **`required_gib`, in absolute GiB per shard.** This is the figure a deployment sizes against. The
  `granted_gib` beside it is what the run happened to give that tenant, which for a tenant alone on a
  card is the whole card — it is measured data about the run, not a recommendation.
- **the conditions by reference**: the fleet file's path and the sha256 of its bytes, plus the
  validity key they hash to. The key's expansion is already filed as `key.json`; a second copy here
  could disagree with it.
- **three results**: the `baseline` always, the certified `strict` placement, and `relaxed`, which is
  `null` more often than not with `relaxed_why_not` saying why. Plus the placements that were
  measured and **rejected**, with the overshoot that ruled each one out — those cost a measurement
  round each, and they stop a consumer proposing an arrangement the hardware already refused.
- **`active_thread_percentage`** on a result's tenant, only when the search measured it under an SM
  cap. It is part of the placement: the same slots uncapped were not what was certified, and a KAI
  deploy serves the cap in that tenant's pod env.

It refuses to run at all when a tenant's TP degree is not settled by a solo measurement, and the
refusal names `profile`. That is the same loader refusal `search` gets, for the same reason: a degree
nothing measured would make an exported baseline a guess.

Report the path it wrote, the three device counts, and whether `relaxed` is null. Do not restate the
YAML in prose — the file is the artifact, and the numbers in it are `search`'s, not new ones.
