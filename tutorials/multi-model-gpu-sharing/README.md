# Multi-model GPU sharing

A platform team needs to deploy M models on N GPUs. How can it use the fewest GPUs while every model
still meets its service level objectives (SLOs)?

This folder answers that question on the EKS platform this repository builds. You describe the
models, their request rates and their latency SLOs in a fleet file. The tool finds a placement that
shares GPUs between models, measures that placement under load, and reports it only if every model
meets its SLOs. You can then deploy the placement on the KAI scheduler.

## How it works

1. **Plan.** From each model's size and SLOs, a solver lists the placements that fit in GPU memory.
   This step needs no GPU.
2. **Profile.** Each model runs alone on a GPU node, to find the fewest GPUs it needs to meet its
   SLOs. The sum over all models is the baseline: the GPU count without sharing.
3. **Search.** An agent, driven by a Claude model on Amazon Bedrock, picks placements that share
   GPUs. Each one is deployed with NVIDIA MPS, driven at the stated request rates, and measured. A
   placement counts only if every model meets its SLOs in that measurement.
4. **Export.** The best measured placement is written in a form you can deploy, and compared with
   the baseline.

[USER-GUIDE.md](USER-GUIDE.md) explains each step in detail.

## Where to start

- **[TUTORIAL.md](TUTORIAL.md)** takes one fleet of five models through every step. It has a
  CPU-only path, which runs on a laptop with no cluster, and a GPU path, which runs on this
  repository's cluster with Claude Code.
- **[USER-GUIDE.md](USER-GUIDE.md)** is the reference: the fleet file format, every mode and flag,
  what each step costs, what each refusal means, serving on KAI, and the Claude Code plugin.

The tools and the guide call a step *free* when it uses no GPU. It still runs on billed
infrastructure.

## What is in this folder

| path | contents |
|---|---|
| `src/colocation/` | the agent, the solver, the stores, and the self-test |
| `containers/colocation-harness/` | the load harness that measures each placement |
| `charts/machine-learning/model-serving/` | Helm charts for the MPS daemon, the serving engines, and KAI |
| `plugins/` | the Claude Code plugin: skills, commands, and safety hooks |
| `fleets/examples/` | example fleet files, one folder per instance type |
| `load-tests/examples/` | load-test scenarios for the example fleets |
| `tools/` | the demand sweep, and two scripts for the CPU-only path |
| `observability/`, `manifests/` | the DCGM exporter settings, and the GPU node holder |
