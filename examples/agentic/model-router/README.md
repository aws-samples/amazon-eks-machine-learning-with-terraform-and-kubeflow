# Model Router with a System One Model

This example is a small agent harness that uses [CLM-8B](../../inference/system-one/clm-8b/), a *System One* model, to make three decisions around an ordinary tool-calling agent loop:

| hook | when | question CLM-8B answers | effect |
|---|---|---|---|
| **route** | before the run | How hard is this request? How risky is it? | picks the first model tier |
| **gate** | before each call to a risky tool | Should the assistant carry out this action? | runs the tool, or returns "blocked" to the model |
| **escalate** | after a tier's final answer | Does this reply answer the request? | keeps the answer, or reruns the task on the next tier |

Each decision is one `/v1/systemone` call that returns probabilities from a single forward pass, with no text generated. The harness is plain Python with no agent framework, so every decision is visible in a few lines of [src/hooks.py](./src/hooks.py).

```
          ┌───────────────────── clm-serve (CLM-8B) ─────────────────────┐
          │  route                    gate                    escalate   │
          └────┬───────────────────────┬──────────────────────────┬──────┘
               │                       │                          │
 task ──► pick a tier ──► agent loop: model ⇄ tools ──► final answer ──► done
                              ▲      (risky tools                 │
                              │       go through the gate)        │
                              └─────── next tier ◄─── not answered ┘
```

## Two tier configurations

The same harness runs with either tiers file. Pick the one that matches where you want the models to run.

| tiers file | small tier | larger tiers | needs |
|---|---|---|---|
| [tiers/hybrid.yaml](./tiers/hybrid.yaml) | Qwen3-8B on your EKS GPUs, served by [Dynamo](../../inference/dynamo/vllm/qwen3-8b/README.md#tool-calling) | Claude on Amazon Bedrock | CLM-8B, Dynamo with tool calling, Bedrock access |
| [tiers/bedrock.yaml](./tiers/bedrock.yaml) | Claude Haiku on Amazon Bedrock | Claude Sonnet, then Opus, on Amazon Bedrock | CLM-8B, Bedrock access |

A tier is any OpenAI-compatible endpoint (`backend: openai`) or any Bedrock model that supports tool use through the Converse API (`backend: bedrock`). To add a tier or change a model, edit the tiers file.

Before proceeding, complete the [Prerequisites](../../../README.md#prerequisites) and [Getting started](../../../README.md#getting-started).

## Jupyter notebook

The [model-router.ipynb](./model-router.ipynb) notebook runs every step below. The sections that follow give the same steps as commands.

## Deploy the dependencies

1. **CLM-8B.** Follow [Serve CLM-8B](../../inference/system-one/clm-8b/README.md) to launch `clm-encoder` and `clm-serve`.
2. **Qwen3-8B with tool calling**, for `tiers/hybrid.yaml` only. Follow [Tool Calling](../../inference/dynamo/vllm/qwen3-8b/README.md#tool-calling) to launch `dyn-qwen3-8b` from `dgd-agg-tools.yaml`, and run its `tool_call_check.py`. Without the tool-call parser, Qwen3's tool calls come back as text and the agent loop never sees them.
3. **Amazon Bedrock.** The harness calls Bedrock with your AWS credentials. Enable access to the models in the tiers files, or change the model IDs to models you have enabled.

## Run the harness

The harness runs on your machine and reaches the services through port-forwards:

    kubectl port-forward -n kubeflow-user-example-com svc/clm-serve 8700:8700 &
    kubectl port-forward -n kubeflow-user-example-com svc/dyn-qwen3-8b-frontend 8000:8000 &   # hybrid only

Install the dependencies and run from this folder. `CLM_BASE_URL` and `DYNAMO_BASE_URL` default to the ports above:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow/examples/agentic/model-router
    pip install -r requirements.txt
    export AWS_REGION=<aws-region>

See where each task in [tasks.jsonl](./tasks.jsonl) would go, without calling any model:

    python run.py --tiers tiers/hybrid.yaml --tasks tasks.jsonl --route-only

Run the tasks:

    python run.py --tiers tiers/hybrid.yaml --tasks tasks.jsonl
    python run.py --tiers tiers/bedrock.yaml --tasks tasks.jsonl

Or one task of your own:

    python run.py --tiers tiers/bedrock.yaml --task "When will order A1001 ship?"

For each task, `run.py` prints the route and CLM-8B's reasons, the tools each tier called with the gate's verdict on the risky ones, any escalation, and the final answer. The full traces, including token usage and timings, are appended to `results/<tiers>.jsonl`.

### See the gate in front of the small model

Larger models often refuse a risky request by themselves, so the gate has the most to do in front of a small model. `--tier` skips the route and starts every task on one tier:

    python run.py --tiers tiers/hybrid.yaml --tasks tasks.jsonl --tier small \
        --only refund-requested,refund-unrequested,email

When the gate blocks a call, the model receives `blocked: this action needs the user's confirmation` as the tool result and has to tell the user.

## The tools

[src/tools.py](./src/tools.py) defines five demo tools over small in-memory tables: `get_weather`, `calculator`, `lookup_order`, `issue_refund` and `send_email`. They change nothing outside the process. `issue_refund` and `send_email` are marked `risky=True`, so every call to them goes through the gate. To use your own tools, add `Tool` entries and mark the ones that act on the user's behalf as risky.

## Things to know

- **The thresholds are not calibrated.** `hard_limits`, `high_stakes_floor`, `gate_below` and `escalate_below` in the tiers files are hand-set starting points. Log the traces on your own traffic, judge the outcomes, and set the thresholds from that.
- **How you ask matters.** CLM-8B's answers depend strongly on the wording and the type of a question. Asked for a difficulty score or for the probability of a yes/no statement, it gave nearly every task the same answer. A `choice` between two described options separated them, so every hook asks one. Test any question you change against tasks whose right answer you know.
- **The gate is a check, not a security boundary.** It catches actions the user clearly did not ask for. It can miss a plausible-looking action with a wrong argument, such as an email to the wrong address. Keep authorization and argument validation in the tools themselves.
- **CLM-8B does not verify facts or arithmetic.** The escalate hook judges whether a reply addresses the request. It cannot tell a wrong number from a right one.
- **Escalation starts over.** An escalated task reruns from the user's message on the next tier, so tool calls the lower tier made can run again. Make risky tools idempotent, or carry the lower tier's tool results forward if your tools are not.
- **2048-token states.** CLM-8B reads at most 2048 tokens. The hooks keep the start of a long state (the task) and its end (the latest step).
- **Reasoning is not sent back.** Qwen3's `reasoning_content` is dropped from the conversation history; only the reply and its tool calls are kept.

## Clean up

Stop the port-forwards. To remove the dependencies, follow the *Stop* sections of the [CLM-8B](../../inference/system-one/clm-8b/README.md#stop-the-services) and [Qwen3-8B](../../inference/dynamo/vllm/qwen3-8b/README.md#stop-service) examples.
