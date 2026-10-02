# Request triage demo

`triage_demo.py` sends each prompt in [prompts.jsonl](./prompts.jsonl) to `clm-serve` once and asks three typed questions in that one call:

| question | type | answer |
|---|---|---|
| `task_type` | choice | code, math_reasoning, extraction, summarization, tool_use, open_chat, other |
| `tier` | choice | the capability tiers in [tiers.yaml](./tiers.yaml): small, medium, frontier, other |
| `high_stakes` | noul | probability that a wrong answer could cause financial, legal, medical, safety or production harm |

These are the decisions a routing layer in front of a model API has to make on every request. CLM answers them with probability distributions from a forward pass, not by generating text. There is nothing to parse and no prompt to keep in sync with a JSON schema, and the cost per decision is the same whatever the answer.

**This is an illustration, not a benchmark.** The 60 labels are one reviewer's judgement, and 10 prompts (`"ambiguous": true`) are deliberately underspecified. Read the accuracy table as "does the signal point the right way", not as a score to compare against other systems.

## Run

```bash
pip install -r requirements.txt
kubectl port-forward svc/clm-serve 8700:8700 -n kubeflow-user-example-com &
python triage_demo.py --url http://localhost:8700
```

If `clm-serve` was installed with `CLM_API_KEY`, export the same `CLM_API_KEY` before running. The results go to `results/`, which is git-ignored:

| file | contents |
|---|---|
| `summary.md` | the tables below, ready to paste |
| `metrics.json` | every number in `summary.md`, the `/health` response, and each prediction with its probabilities |
| `confidence_gating.png` | tier accuracy on the requests CLM keeps, by confidence threshold |

## What it reports

**Latency, three ways.** `clm-serve` caches state embeddings and projected vectors, so a single latency figure would be misleading:

| pass | what happens |
|---|---|
| first call | the first request after startup; the encoder embeds the option texts as well as the state |
| steady state | new prompts after a separate warm-up set ([warmup.jsonl](./warmup.jsonl)); only the state is embedded |
| repeat (cached) | the same 60 prompts again; answered from the cache with no encoder call (`input_tokens` is 0) |

Steady state is the number that describes production traffic. Each question adds its instructions to the state text, so three questions mean three state embeddings per request. The script records both wall-clock time at the client and `X-CLM-Latency-Ms` from the server, so you can tell port-forward overhead from model time.

**Accuracy** for each question, split into clear and ambiguous prompts.

**Confidence gating.** A router should not trust a low-confidence tier answer. Below a threshold, it routes up one tier. The table and chart show, for thresholds 0.0 to 0.9, what fraction of requests CLM keeps and how accurate it is on them. The escalation table lists the prompts that would escalate at 0.5. `noul` answers have no confidence field; the script uses `|p − 0.5| × 2`.

## Optional: a generative baseline

To compare against an LLM making the same three decisions, point the script at any OpenAI-compatible chat endpoint:

```bash
export BASELINE_API_KEY=...   # if the endpoint needs one
python triage_demo.py --url http://localhost:8700 \
  --baseline-url http://<openai-compatible-host> --baseline-model <model-name>
```

The baseline gets the same labels and tier descriptions in a JSON-only prompt. The summary adds its latency, its accuracy and the number of replies that would not parse. A follow-up model-routing example makes this comparison against Amazon Bedrock, with cost.

## Mock encoder

The upstream repository ships `tools/playground_mock.py`, which runs the real `clm-serve` app on character n-grams instead of Qwen3-8B. Use it to work on the script without a GPU. `/health` reports `"mock": true`; the script then prints a warning and stamps the summary and chart. Its numbers mean nothing, so never quote them.

## Changing the questions

The tiers are capability descriptions, not model names: the description is the text CLM scores the request against. To change what a tier means, edit `tiers.yaml`. Mapping tiers to models belongs in the routing layer. `example_model_mapping` in `tiers.yaml` only shows the shape and the script ignores it. To add a question, add it in `questions()` in `triage_demo.py`. Each question costs one more state embedding per request.
