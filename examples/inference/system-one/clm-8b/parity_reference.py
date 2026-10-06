"""Compute the parity reference answers without vLLM, and compare with a deployed encoder.

serve.ipynb checks the deployment against reference answers for one request. This script
produces them independently: it computes the Qwen3-8B last-token embeddings with Hugging
Face Transformers on CPU and runs them through the same contrastive-lm engine and CLM-8B
heads that clm-serve uses. With --encoder-url it also embeds every text through a vLLM
encoder and prints the cosine similarity, so drift can be traced to the encoder.

Needs about 20 GB of free RAM and the two downloads (about 16 GB):

  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install transformers accelerate safetensors requests && pip install --no-deps contrastive-lm==0.1.0
  hf download Qwen/Qwen3-8B --local-dir qwen3-8b
  hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt --local-dir clm-heads
  python parity_reference.py --encoder qwen3-8b --heads clm-heads/CLM_v0.1-8B.pt \\
      [--encoder-url http://localhost:18000]   # kubectl port-forward svc/clm-encoder 18000:8000
"""
import argparse

import numpy as np
import requests
import torch
from clm.embedder import Embedder, l2
from clm.engine import Engine
from transformers import AutoModel, AutoTokenizer

STATE = "Customer: my invoice was charged twice and nobody answers the phone!"
QUESTIONS = {
    "urgency": {"type": "noul", "instructions": "Is this urgent?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
}


class TransformersEmbedder(Embedder):
    """Last-token pooling of the final hidden state, L2-normalised: what vLLM's pooling runner does."""

    def __init__(self, path, max_tokens=2048):
        super().__init__(max_tokens=max_tokens)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16).eval()
        self.vectors = {}

    def _fetch(self, texts):
        out, tokens = [], 0
        for t in texts:
            ids = self.tok(t, return_tensors="pt", truncation=True, max_length=self.max_tokens)
            tokens += ids["input_ids"].shape[1]
            with torch.no_grad():
                v = l2(self.model(**ids).last_hidden_state[0, -1].float().numpy())
            self.vectors[t] = v
            out.append(v)
        return out, tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, help="local Qwen3-8B directory")
    ap.add_argument("--heads", required=True, help="CLM_v0.1-8B.pt")
    ap.add_argument("--encoder-url", help="a vLLM encoder to compare embeddings with")
    args = ap.parse_args()

    emb = TransformersEmbedder(args.encoder)
    a = Engine(embedder=emb, checkpoint=args.heads, device="cpu").answer(STATE, QUESTIONS)["answers"]
    print(f'reference = {{"urgency": {a["urgency"]["noul"]:.5f}, '
          f'"billing": {a["department"]["probabilities"]["billing"]:.5f}, '
          f'"frustration": {a["frustration"]["score"]:.5f}}}')

    if args.encoder_url:
        for t, v in emb.vectors.items():
            j = requests.post(f"{args.encoder_url.rstrip('/')}/v1/embeddings", timeout=60,
                              json={"model": "qwen3-8b", "input": [t]}).json()
            w = l2(np.asarray(j["data"][0]["embedding"], dtype=np.float32))
            print(f"cosine {float(v @ w):.5f}  {t[:60]!r}")


if __name__ == "__main__":
    main()
