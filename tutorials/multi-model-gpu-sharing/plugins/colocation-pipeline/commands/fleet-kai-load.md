---
description: Start a load test of a KAI deployment, a scenario's traffic through each tenant's Service for whole clock hours
argument-hint: <deployment name> <scenario file>
allowed-tools: [Bash, Read]
---

Only when the user asked for a load test: it puts a harness Job on the CPU pool and keeps the
deployment's GPU node busy for the scenario's hours. From the repository's `tutorials/multi-model-gpu-sharing` directory, in
one command so the image is the one just ensured:

```bash
IMG=$(bash plugins/colocation-pipeline/scripts/harness-image.sh --ensure) && \
  COLOCATION_HARNESS_IMAGE="$IMG" PYTHONPATH=src python3 -m colocation.kai load $ARGUMENTS
```

The scenario is a file under `load-tests/examples/<instance type>/<fleet>/`: `as-fleet.yaml` runs every
tenant at its fleet demand, `demand-drift.yaml` moves a few tenants away from it. It must be for
the fleet the deployment was exported from. A node of a split (`<name>-n<i>`) was exported from its
group's fleet file, so a parent fleet's scenario is refused on it for now; say so rather than
writing a scenario to get past it.

Exit `2` is a refusal before anything was applied: another fleet's scenario, a tenant the fleet
does not have, a request past a tenant's `max_model_len`, no harness image, or an engine not Ready
(`verify <name> --wait-ready 900` first). Report it; do not edit the scenario to get past it.

On `0` it has applied the Job and returned. Report the per-tenant traffic it printed and the Job
name. The Job calibrates, then drives from the next whole clock hour. After each hour its log
(`kubectl logs -n kubeflow-user-example-com job/<job>`) has one `SOAK HOUR` line per tenant, the
client's figures. Once the hours have passed, plus five minutes, `/fleet-kai-drift <name>` records
what the deployment saw. Report the two side by side, and leave what to do about them to the user.
`teardown` of the deployment also deletes its load Jobs.
