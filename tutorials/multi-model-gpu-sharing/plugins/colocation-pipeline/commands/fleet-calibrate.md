---
description: Write the between-run ITL half-width from repeats already in the store, so a delta between two placements can be called real
argument-hint: <fleets/name.yaml>
allowed-tools: [Bash, Read]
---

From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
PYTHONPATH=src python3 -m colocation.agent calibrate --fleet $ARGUMENTS
```

It reads the store and no cluster. It writes the between-run ITL half-width per co-residency bucket.

It exits 1 and writes nothing when no placement has been measured twice, which is the normal outcome
after a first search: a repeat is a placement the reasoner *chose* to re-measure, and nothing forces
one. When there is no calibration, the report says "uncalibrated" in words. **Never fill that gap with
a number of your own.** Getting one needs another `search` round spent on a placement the store
already holds.

Run it after a search that produced a repeat, and before quoting any delta between two placements as
real.
