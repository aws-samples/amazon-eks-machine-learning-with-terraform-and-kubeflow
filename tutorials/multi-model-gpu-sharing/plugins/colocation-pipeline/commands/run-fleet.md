---
description: Take a fleet file through the whole co-location pipeline — image, plan, sweep, profile, plan again, search, read, export
argument-hint: <fleets/name.yaml>
allowed-tools: [Bash, Read, Edit, Glob, Grep, Skill]
---

Run the co-location pipeline on: $ARGUMENTS

Use the `run-fleet` skill in this plugin. It owns the sequence, the back-edges, and how to read
every refusal. Nothing in the code sequences these steps, so follow it in order rather than from
memory:

```
harness image -> plan -> demand sweep -> profile -> plan again -> search -> read -> export
```

The individual steps are also available on their own, if part of this is already done:
`/fleet-preflight`, `/fleet-image`, `/fleet-plan`, `/fleet-sweep`, `/fleet-profile`,
`/fleet-search`, `/fleet-calibrate`, `/fleet-inspect`, `/fleet-export`.

A fleet with `hardware.max_nodes` above 1 takes the same sequence, but `search` measures it one group
per node, and its export is the split's `export.<stamp>.yaml` that `search` writes. The `export`
mode refuses such a fleet.

The pipeline ends at the export. Serving it on KAI (`/fleet-kai-deploy`) is a separate step the user
starts; do not run it from here.

If no fleet file was named, list the fleet files under `fleets/` (the shipped ones are under
`fleets/examples/<instance type>/`) with one line each from the file's own header comment, and
ask which one.
