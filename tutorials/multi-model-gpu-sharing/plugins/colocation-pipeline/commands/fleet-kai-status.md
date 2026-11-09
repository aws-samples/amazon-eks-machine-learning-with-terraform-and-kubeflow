---
description: Whether a KAI deployment is still placed as its slots say, and which engines are Ready
argument-hint: <deployment name>
allowed-tools: [Bash]
---

From the repository's `tutorials/multi-model-gpu-sharing` directory:

```bash
PYTHONPATH=src python3 -m colocation.kai verify $ARGUMENTS
```

Read-only. It reads the plan from the `kai-<name>-plan` ConfigMap and compares it with the cards the
binder recorded on each pod. Exit `0` is placed as planned (it then lists any engine not Ready yet).
`3` is placed differently: tenants spread over nodes, a tenant on the wrong number of cards or asking
for the wrong fraction, a tenant with two pods after a restart, or a card that also holds a pod this
plan did not put there. `1` is a tenant with no pod or not bound yet, or no `kai-<name>-plan`
ConfigMap — no deployment of that name. `2` is no name given.

A Deployment re-creates a deleted or evicted pod wherever binpack likes, so a deployment that verified
once can stop verifying later. Report the `NOT AS PLANNED` lines as printed; fixing it is a teardown
and a fresh `/fleet-kai-deploy`, and that is the user's call.

A split deploys one deployment per node, `<name>-n1`, `<name>-n2`, ...: verify each by its own name.

With no name, list what is deployed:

```bash
kubectl get configmap -n kubeflow-user-example-com -l colocation.kai/deployment
```

and `helm list -n kubeflow-user-example-com -l colocation.kai/deployment=<name>` lists one
deployment's releases.
