# CLM-8B System One Model

This example shows how to use the [generic-server](../../../../charts/machine-learning/serving/generic-server/) Helm chart to serve [CLM-8B](https://github.com/Contrastive-LM/CLM), a *System One* model: you give it a state (a request, a document, a conversation) and typed questions about that state, and it answers with probabilities from a single forward pass. It generates no text.

| question type | answer | example |
|---|---|---|
| `choice` | a distribution over named options, plus a confidence | Which team should handle this ticket? |
| `score` | an expected level on an ordered scale, plus a confidence | How frustrated is the customer, from calm to very angry? |
| `noul` | the probability that a statement is true | This request is urgent. |

Because the answer is a distribution and not generated text, it suits decisions an application makes on every request: routing, triage, tagging, guardrail checks and ranking candidates. Using a generative LLM for these is often overkill. The [demo](./demo/) uses CLM to triage requests for a model-routing layer.

Before proceeding, complete the [Prerequisites](../../../../README.md#prerequisites) and [Getting started](../../../../README.md#getting-started). See [What is in the YAML file](../../../../README.md#yaml-recipes) to understand the common fields in the Helm values files.

## How it is served

CLM-8B is a frozen [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) encoder plus small projection heads (a 75 MB checkpoint). The example runs them as two `generic-server` releases:

| release | what it runs | where | values file |
|---|---|---|---|
| `clm-encoder` | Qwen3-8B as a vLLM pooling model (`/v1/embeddings`, last-token pooling) | 1 × g5.2xlarge (A10G 24 GB) | [clm-encoder.yaml](./clm-encoder.yaml) |
| `clm-serve` | the CLM heads and the System One HTTP API (`/v1/systemone`, `/v1/rank`) | 1 × m6i.xlarge, CPU only | [clm-serve.yaml](./clm-serve.yaml) |

`clm-serve` finds the encoder at `http://clm-encoder:8000`, so **the encoder release must be named `clm-encoder`**, in the same namespace. Both services are `ClusterIP`.

Splitting them keeps the GPU doing only what needs a GPU. The heads, the vector cache and the API run on a small CPU node. Several `clm-serve` replicas, or other embedding clients, can share one encoder.

## Jupyter notebook

The [serve.ipynb](./serve.ipynb) notebook runs every step below, including a parity check against the upstream reference values and the triage demo. The sections that follow give the same steps as commands.

## Build and push the Docker containers

The encoder uses the repository's vLLM container. `clm-serve` has its own small CPU container, [containers/clm-serve](../../../../containers/clm-serve/). Build and push both, replacing `aws-region` with your AWS Region name:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    ./containers/ray-pytorch-vllm/build_tools/build_and_push.sh aws-region
    ./containers/clm-serve/build_tools/build_and_push.sh aws-region

Each script prints the ECR URI it pushed. You pass these at install time with `--set image.name=...`; the values files do not hold them.

## Download the model weights

Qwen3-8B and the CLM-8B heads are both public, so a Hugging Face token is optional. To use one, add `{"name":"HF_TOKEN","value":"YourHuggingFaceToken"}` to `env`.

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug clm-qwen3-8b charts/machine-learning/model-prep/hf-snapshot \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"Qwen/Qwen3-8B"}]' \
        -n kubeflow-user-example-com
    helm install --debug clm-heads charts/machine-learning/model-prep/hf-snapshot \
        --set-json='env=[{"name":"HF_MODEL_ID","value":"Contrastive-LM/CLM-v0.1-8B"}]' \
        -n kubeflow-user-example-com

The weights land in `/fsx/pretrained-models/Qwen/Qwen3-8B` and `/fsx/pretrained-models/Contrastive-LM/CLM-v0.1-8B`. When both jobs complete, uninstall them:

    helm uninstall clm-qwen3-8b -n kubeflow-user-example-com
    helm uninstall clm-heads -n kubeflow-user-example-com

## Launch the services

Replace the two image URIs with the ones the build scripts printed:

    cd ~/amazon-eks-machine-learning-with-terraform-and-kubeflow
    helm install --debug clm-encoder charts/machine-learning/serving/generic-server \
        -f examples/inference/system-one/clm-8b/clm-encoder.yaml \
        --set image.name=<ray-pytorch-vllm ECR URI> \
        -n kubeflow-user-example-com
    helm install --debug clm-serve charts/machine-learning/serving/generic-server \
        -f examples/inference/system-one/clm-8b/clm-serve.yaml \
        --set image.name=<clm-serve ECR URI> \
        -n kubeflow-user-example-com

The encoder takes a few minutes to load. `clm-serve` starts without waiting for it, so check the `embedder` field and not only the status code. `/health` returns 200 even while the encoder is unreachable:

    kubectl port-forward svc/clm-serve 8700:8700 -n kubeflow-user-example-com &
    curl -s localhost:8700/health      # ready when it shows "embedder": true

## Ask typed questions

```python
import requests

r = requests.post("http://localhost:8700/v1/systemone", json={
    "state": "Customer: my invoice was charged twice and nobody answers the phone!",
    "model": "clm-latest",
    "questions": {
        "urgency": {"type": "noul", "instructions": "Is this urgent?"},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Charges, invoices, refunds",
                                    "technical": "Bugs and outages"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
})
print(r.json()["answers"], r.headers["X-CLM-Latency-Ms"])
```

The upstream README publishes reference values for this request: urgency ≈ 0.41, billing ≈ 0.94 and frustration ≈ 1.98. The notebook checks your deployment against them. A large difference usually means the encoder is not the one the heads were trained with, or its pooling or truncation differs.

To rank free-form candidates, for example tool names, best-of-N answers or next actions, use `/v1/rank`. It returns the answers best first:

```python
requests.post("http://localhost:8700/v1/rank", json={
    "context": "User: book me a table for two tomorrow at 7pm",
    "question": "Which tool should the agent call first?",
    "answers": ["search_restaurants", "create_calendar_event", "send_email"],
}).json()["ranked"]
```

You can also use the [upstream Python client](https://github.com/Contrastive-LM/CLM#api-reference) (`pip install contrastive-lm`, `CLM_BASE_URL=http://localhost:8700`).

## Run the triage demo

See [demo/README.md](./demo/README.md). In short:

    pip install -r examples/inference/system-one/clm-8b/demo/requirements.txt
    python examples/inference/system-one/clm-8b/demo/triage_demo.py --url http://localhost:8700

## Things to know

- **The heads are tied to the encoder.** The CLM-8B heads were trained on Qwen3-8B last-token embeddings. Swapping the encoder for another model, a quantized variant or a different pooling setting invalidates them. vLLM's default pooling for Qwen3-8B is last-token, and the parity check confirms it.
- **GPU memory.** Qwen3-8B in bf16 needs a GPU with at least 24 GB. `--max-model-len 2048` and `--gpu-memory-utilization 0.85` in `clm-encoder.yaml` are set for an A10G.
- **2048-token states.** Longer states are truncated, with `truncate_prompt_tokens` sent by `clm-serve`. To raise the limit, raise `--max-model-len` in `clm-encoder.yaml` and `CLM_EMB_MAX_TOKENS` in `clm-serve.yaml` together. This needs more GPU memory.
- **Probabilities are relative to the options.** A `choice` distribution says how the options compare with each other for this state. It is not a calibrated probability that any option is correct. Add an `other` option when none might fit, and calibrate thresholds on your own data, as the demo does for confidence gating.
- **Each question embeds the state once.** The question's instructions are appended to the state, so N questions cost N state embeddings. Option texts are embedded once and cached. `clm-serve` also caches state embeddings, so a repeated state costs no encoder call.
- **Security.** Both services are `ClusterIP` with no authentication by default. To require a bearer token on `clm-serve`, add a `CLM_API_KEY` entry to `server.env` in a copy of `clm-serve.yaml`; clients then send `Authorization: Bearer <key>`. `generic-server` takes literal env values only, so the key is visible to anyone who can read the Deployment in the namespace. Do not expose either service outside the cluster.
- **Alpha software.** `contrastive-lm` is at 0.1.0. The container pins it, and the API may change between releases.
- **Fine-tuning.** The heads are small, and upstream documents [fine-tuning them on your own data](https://github.com/Contrastive-LM/CLM#fine-tuning-clm-on-your-own-data) while the encoder stays frozen. A fine-tuned checkpoint is served the same way: point `CLM_CKPT` at it.

## Stop the services

    helm uninstall clm-serve -n kubeflow-user-example-com
    helm uninstall clm-encoder -n kubeflow-user-example-com

## License and attribution

CLM and the CLM-8B weights are by Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré and Azalia Mirhoseini, released under Apache 2.0 at [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) and [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B). Qwen3-8B is by the Qwen team, released under Apache 2.0 at [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B). This example downloads both from Hugging Face; it does not redistribute them.
