---
description: Deploy an exported result on KAI GPU fractions, check every tenant landed where its slot says, and retry when it did not
argument-hint: <export path or run token> [--result strict|relaxed|baseline]
allowed-tools: [Bash, Read]
---

Follow the procedure in the `kai-deploy` skill (`skills/kai-deploy/SKILL.md` in this plugin) for
`$ARGUMENTS`: `render` and report the plan and its warnings, `deploy` in the background, read the
exit code, retry a mismatch (exit 3) by `teardown` then `deploy` at most 3 attempts in all, then
`verify --wait-ready 1800`.

A split's export (schema_version 2) deploys each node as a deployment of its own, `<name>-n<i>`,
one after the other. Each later node's first tenant is kept off the earlier nodes' hosts and in the
first node's zone, so it needs as many `kai-gpu` nodes of the instance type in one zone as the split
has nodes. Retry, verify and teardown take a node's own name.

This command is the user starting the deploy, so the guard hook does not ask again for its
`deploy` or for the `teardown` a retry needs. It may launch a GPU node on the `kai-gpu` pool.
