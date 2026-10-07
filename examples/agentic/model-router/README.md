# Model Router with a System One Model

This example is a small agent harness that uses [CLM-8B](../../inference/system-one/clm-8b/), a *System One* model, to make three decisions around an ordinary tool-calling agent loop:

| hook | when | question CLM-8B answers | effect |
|---|---|---|---|
| **route** | before the run | the questions in the tiers file, for example: How hard is this request? Will it send an email or move money? | the policy picks the first model tier from the answers |
| **gate** | before each call to a risky tool | Did the user ask for this action? Do its arguments match the tool results? | runs the tool, or returns "blocked" to the model |
| **escalate** | after a tier's final answer | Does this reply answer the request? | keeps the answer, or reruns the task on the next tier |

Each decision is one `/v1/systemone` call, with one or two questions, that returns probabilities from a single forward pass, with no text generated. The harness is plain Python with no agent framework, so every decision is visible in a few lines of [src/hooks.py](./src/hooks.py).

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

A tier is any OpenAI-compatible endpoint (`backend: openai`) or any Bedrock model that supports tool use through the Converse API (`backend: bedrock`). To add a tier or change a model, edit the tiers file. To change how tasks are routed, see [Customize the router](#customize-the-router).

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

The Bedrock model IDs default to global cross-Region inference profiles. If your account or Region needs a different model or profile, set `BEDROCK_SMALL_MODEL`, `BEDROCK_MEDIUM_MODEL` or `BEDROCK_LARGE_MODEL`, which override the tier of that name in the tiers file. Keep account-specific profile ARNs in the environment, not in the tiers file.

See where each task in [tasks/demo.jsonl](./tasks/demo.jsonl) would go, without calling any model:

    python run.py --tiers tiers/hybrid.yaml --tasks tasks/demo.jsonl --route-only

Run the tasks:

    python run.py --tiers tiers/hybrid.yaml --tasks tasks/demo.jsonl
    python run.py --tiers tiers/bedrock.yaml --tasks tasks/demo.jsonl

Or one task of your own:

    python run.py --tiers tiers/bedrock.yaml --task "When will order A1001 ship?"

For each task, `run.py` prints the route and CLM-8B's reasons, the tools each tier called with the gate's verdict on the risky ones, any escalation, and the final answer. The full traces, including token usage and timings, are appended to `results/<tiers>.jsonl`.

### See the gate in front of the small model

Larger models often refuse a risky request by themselves, so the gate has the most to do in front of a small model. `--tier` skips the route and starts every task on one tier:

    python run.py --tiers tiers/hybrid.yaml --tasks tasks/demo.jsonl --tier small \
        --only refund-requested,refund-unrequested,email

When the gate blocks a call, the model receives `blocked: this action needs the user's confirmation` as the tool result and has to tell the user.

## Customize the router

The route has three parts, all in the tiers file, so you change the routing without changing code:

- **`signals`** are the questions CLM-8B answers about each task. Each is a `choice` between described options, and the router sees the probability of every option, for example `p(hard)` and `p(money_or_email)`. Option names must be unique across questions.
- **`profile`**, on each tier, says what the tier offers: `max` sets the highest probability of an option the tier takes (a small tier with `max: {hard: 0.5}` does not take a task with `p(hard)` above 0.5), `in_cluster` says whether it runs in your cluster, and `price_per_mtok`, `tokens` and `latency_ms` are optional and empty in the repo.
- **`policy`** says which of the tiers that may take a task to start on: the first in the file (`prefer: order`), the cheapest (`cost`) or the fastest (`latency`). `in_cluster_only` and `latency_budget_ms` rule tiers out. Escalation moves to the next tier in the file that is not ruled out.

[src/policy.py](./src/policy.py) applies them and records the reason for every tier it skips. The same choices are available per run:

    python run.py --tiers tiers/hybrid.yaml --tasks tasks/demo.jsonl --in-cluster-only
    python run.py --tiers tiers/bedrock.yaml --tasks tasks/demo.jsonl --prefer cost
    python run.py --tiers tiers/bedrock.yaml --tasks tasks/demo.jsonl --latency-budget <ms> --profile results/measured-bedrock.yaml

`--prefer cost` needs `price_per_mtok` on every tier, and `--prefer latency` and `--latency-budget` need `latency_ms`. Fill in your own prices, or measure the latency and tokens with `eval.py`, which writes them to `results/measured-<tiers>.yaml` for `--profile`. Until every tier has a value, the policy keeps the file order and says so in the route's reasons.

## Evaluate the router

The demo tasks show what each hook does; they are too few to set thresholds on. [tasks/eval.jsonl](./tasks/eval.jsonl) has 42 tasks of four types (lookups, writing, actions and analysis), and each has a check in code, with no LLM judge, that decides whether an answer passed: the expected number, a word limit, valid JSON, a regex that matches the right strings, the tool calls that must or must not run. [src/checks.py](./src/checks.py) lists the checks.

    python eval.py --tiers tiers/hybrid.yaml
    python eval.py --tiers tiers/bedrock.yaml

`eval.py` asks CLM-8B the tiers file's questions about each task, runs each task on every tier, checks every answer and asks CLM-8B the escalate question about it. From those runs it compares, without calling any model again:

- **strategies**: always one tier, the router, the router with escalation, and an *oracle* that knows the first tier that passes. For each, the pass rate, where tasks started, and tokens, time and, with prices set, cost per task relative to always using the last tier.
- **pass rate by type** for each tier.
- **signals**: for each option, the AUC at predicting that the first tier fails. 0.5 is chance and 1.0 is perfect, so an option near 0.5 is not worth a limit.

The runs are saved to `results/eval-<tiers>.jsonl`. Re-analyse them after you edit the tiers file, with no model calls, and let `--fit` suggest the limits on one option:

    python eval.py --tiers tiers/bedrock.yaml --reuse results/eval-bedrock.jsonl --fit hard

`--fit` searches for the per-tier limits that keep the pass rate of always using the last tier at the least spend (by price, or by tier position without prices), and `--tolerance 0.05` lets it give up five points of pass rate. It also prints a 5-fold cross-validated pass rate: the limits are fitted on four fifths of the tasks and scored on the rest. Trust that figure over the one fitted on all the tasks. 42 tasks are enough to compare strategies and to see which signals carry information. For thresholds you rely on, build a suite from your own traffic.

## The tools

[src/tools.py](./src/tools.py) defines five demo tools over small in-memory tables: `get_weather`, `calculator`, `lookup_order`, `issue_refund` and `send_email`. They change nothing outside the process. `issue_refund` and `send_email` are marked `risky=True`, so every call to them goes through the gate. To use your own tools, add `Tool` entries and mark the ones that act on the user's behalf as risky.

## Things to know

- **The thresholds are starting points.** The `max` limits, the `gate` limits and `escalate_below` in the tiers files are hand-set. Measure them with `eval.py` on tasks like yours before you rely on them.
- **Ask CLM-8B concrete questions.** It answers questions about something observable in the request, such as whether it sends an email or moves money, more reliably than abstract judgments, such as how risky the request is. Prefer a signal that names a concrete property, check each signal's AUC with `eval.py`, and drop the ones near 0.5.
- **How you ask matters.** CLM-8B's answers depend strongly on the wording and the type of a question. Asked for a difficulty score or for the probability of a yes/no statement, it gave nearly every task the same answer. A `choice` between two described options separated them, so every hook asks one. Test any question you change against tasks whose right answer you know.
- **The gate asks two questions.** Asked only whether the user wanted the action, CLM-8B allowed an email to an address the order lookup never returned: the intent was right, and the argument was wrong. A second question, whether the arguments match the tool results, separates those cases.
- **The gate can miss a negated instruction.** For "draft, but do not send, an email", CLM-8B can still judge a `send_email` call as requested. Express such limits in the tools, for example with a separate tool for drafts.
- **The gate is a check, not a security boundary.** CLM-8B judges meaning; it does not compare strings. Keep authorization and exact checks, such as "the recipient is the customer on this order", in the tools themselves.
- **CLM-8B does not verify facts or arithmetic.** The escalate hook judges whether a reply addresses the request. It cannot tell a wrong number from a right one.
- **Escalation starts over.** An escalated task reruns from the user's message on the next tier, so tool calls the lower tier made can run again. Make risky tools idempotent, or carry the lower tier's tool results forward if your tools are not.
- **2048-token states.** CLM-8B reads at most 2048 tokens. The hooks keep the start of a long state (the task) and its end (the latest step).
- **Reasoning is not sent back.** Qwen3's `reasoning_content` is dropped from the conversation history; only the reply and its tool calls are kept.

## Clean up

Stop the port-forwards. To remove the dependencies, follow the *Stop* sections of the [CLM-8B](../../inference/system-one/clm-8b/README.md#stop-the-services) and [Qwen3-8B](../../inference/dynamo/vllm/qwen3-8b/README.md#stop-service) examples.
