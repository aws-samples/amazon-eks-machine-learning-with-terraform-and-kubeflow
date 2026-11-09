---
description: Build, self-test and push the harness image if the current source is not already in ECR, and print which image a run will use
argument-hint: (no arguments)
allowed-tools: [Bash, Read]
---

From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
bash plugins/colocation-pipeline/scripts/harness-image.sh --ensure
```

The image reference is the only thing on stdout; docker and aws output goes to stderr.

**The tag is a digest of the build context, never a number chosen by hand.** That is what makes
"pullable but stale" impossible: `colocation-harness:v5` sat in ECR pulling perfectly while
predating the per-tenant tokenizer, so it could not parse what `render.py` writes. A changed
harness hashes to a different tag automatically, and an unchanged one costs nothing — the script
checks ECR first and exits.

If the build's in-image selftest fails, the script stops before the push. **Report that and stop.**
A failing harness image measures wrongly rather than not at all, and a measuring defect in this
repo has voided complete search results before. Do not fall back to an older tag that happens to
pull.

Then say in your report which account, region and tag the run will use. `harness_image` is not in
the observation's provenance, so your report is the only place the instrument is written down.

`/fleet-profile` and `/fleet-search` invoke this script themselves, so running it separately is
only for seeing what they will use.
