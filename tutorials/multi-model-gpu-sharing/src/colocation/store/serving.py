"""The engine flags a fleet file fixes, as one keyed value (impl §5).

**Why this exists.** Five per-tenant settings in a fleet file are rendered straight onto the serving
engine's command line — `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`, `--dtype`, and
whatever `extra_args` adds — and until 2026-09-09 not one of them appeared in the validity key or in
a solo row's identity. So two runs of one fleet that differed only in, say, `dtype: bfloat16` versus
an FP8 checkpoint's native precision, or in a batch cap that decides how much prefill a step admits,
hashed into the same store directory and were handed to the reasoner as evidence about one
condition. The failure has no symptom: every record is individually valid.

It was found the expensive way. `search-r2hk3z` refused its densest placement because
`--max-num-seqs=64` — a chart default no fleet file had ever stated — capped the anchor below the
concurrency its own latency target requires, and the obvious next step -- raise the cap and
re-measure -- would have read back solo rows taken under the old cap with nothing saying so
(2026-09-09).

**What is in and what is out.** Everything under the renderer's engine block (`colocation.engines`
decides its key and its spellings) except the memory budget and the shard count. The budget is
excluded because the search varies it deliberately — it is the envelope the solver assigns, so keying it would give every
candidate placement its own store directory and there would be no history to retrieve. TP is
excluded here only because `ValidityKey.tensor_parallel` already carries it, per tenant, for exactly
the reasons this module exists.

The rule to apply to a sixth flag: **if the fleet file fixes it and the engine reads it, it belongs
here; if the search decides it, it does not.**
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..engines import DEFAULT_ENGINE

# Length of the digest that goes in a solo row's filename. Short because it sits in a path a human
# reads, and a row also carries the flags in full inside its own JSON, so the digest never has to
# be reversed -- it only has to separate one configuration from another.
DIGEST_CHARS = 6


@dataclass(frozen=True)
class ServingConfig:
    """The engine flags one tenant is served with, other than TP and the envelope.

    Frozen for `ValidityKey`'s reason: a value that names a store directory must not be mutable
    after records have been filed under it.
    """

    max_num_seqs: int
    max_num_batched_tokens: int
    max_model_len: int
    dtype: str
    extra_args: tuple = ()
    # Which serving stack served this tenant. It belongs here by this module's own rule -- the fleet
    # file fixes it and the engine reads it -- and it belongs here more strongly than any flag does:
    # two rows taken on vLLM's OpenAI server and on Triton with a vLLM backend are latency
    # measurements of different software serving the same weights, which is precisely the comparison
    # this project exists to make. Filed under one hash they would be handed to the reasoner as
    # evidence about one condition, which is the 2026-09-09 failure with a bigger cause.
    engine: str = DEFAULT_ENGINE

    def canonical(self) -> dict:
        """The exact dict that gets hashed and written into a key or a row.

        `extra_args` is a list rather than a set: `--foo --bar` and `--bar --foo` are the same
        engine configuration, but ordering an arbitrary argv is a judgement about a syntax this
        module does not parse, and the safe direction for a *key* is to treat a reordering as a
        different configuration rather than to assert it is the same one.
        """
        out = {
            "dtype": str(self.dtype),
            "extra_args": [str(a) for a in self.extra_args],
            "max_model_len": int(self.max_model_len),
            "max_num_batched_tokens": int(self.max_num_batched_tokens),
            "max_num_seqs": int(self.max_num_seqs),
        }
        # Emitted only when it is not the default, and this is the one deliberate omission in a
        # module whose whole argument is that omissions cause collisions. It is the exception
        # `ValidityKey.canonical` already makes for images, for the same reason and not a weaker one:
        # a digest written before engines were nameable describes a `vllm-openai` run, because that
        # was the only engine this system could start. So the default adds no information, while
        # emitting it would re-key every measurement in the store -- discarding the history that
        # makes a second engine worth measuring against.
        if str(self.engine) != DEFAULT_ENGINE:
            out["engine"] = str(self.engine)
        return out

    @property
    def digest(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:DIGEST_CHARS]

    @classmethod
    def from_dict(cls, d: dict) -> ServingConfig:
        # `engine` absent means the default, which is the same reading `canonical` writes: a row from
        # before the field existed was served by the only engine there was.
        return cls(engine=str(d.get("engine") or DEFAULT_ENGINE),
                   max_num_seqs=int(d["max_num_seqs"]),
                   max_num_batched_tokens=int(d["max_num_batched_tokens"]),
                   max_model_len=int(d["max_model_len"]),
                   dtype=str(d["dtype"]),
                   extra_args=tuple(str(a) for a in (d.get("extra_args") or ())))
