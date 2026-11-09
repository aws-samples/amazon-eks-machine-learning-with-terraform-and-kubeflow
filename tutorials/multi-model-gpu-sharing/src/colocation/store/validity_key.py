"""The validity key (design §4.1, impl §5).

An observation is valid only within its key. The key names everything that, if changed,
makes previously collected numbers describe a different machine: the environment
(`environment.py` -- what the device reports about itself, plus the engine image, the sharing
mode and the metric definitions in force), the workload the fleet is conditioned on, and the
shard count each tenant is served at.

**It names no hand-written hardware fact.** The instance type and the driver version were both in
here and are now recorded as provenance on the rows instead: the instance type because it would
re-measure the same card once per SKU, the driver because it is a proxy for behaviour it never
observes -- see `environment.py` for that argument in full. What replaced them is measured.

The hash is the store's directory name, so it is load-bearing in a way that is easy to
underestimate: **if the same environment ever hashes two different ways, the agent silently
loses its entire history and starts over, and nothing about that failure looks like an
error.** Every rule below exists to make the hash a function of the environment and nothing
else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .environment import Environment
from .serving import ServingConfig

# Decimal places `alpha` is quantized to before hashing. Not cosmetic: alpha is derived by
# a sweep (design §8.3), so it arrives as the result of arithmetic. 0.35 * 2 is
# 0.7000000000000001, which hashes differently from the 0.7 a human writes in a YAML file
# while denoting the same load scale. Quantizing removes a whole class of silent
# history-loss, and four places is far finer than any alpha the sweep can resolve.
ALPHA_DECIMALS = 4

# Decimal places `offered_rates_hz` is quantized to, for the same reason as alpha: a rate
# computed as a fraction of a measured saturation rate arrives with float noise on it.
RATE_DECIMALS = 3

# Hash prefix length. Long enough that collision is not a practical concern for the number
# of distinct environments a tutorial ever visits, short enough to be a readable directory
# name and easy to quote.
HASH_CHARS = 12


@dataclass(frozen=True)
class ValidityKey:
    """Everything an observation is conditioned on.

    Frozen because a key that can be mutated after observations have been filed under its
    hash is a key that can orphan them.
    """

    # What the machine is, measured rather than asserted (`environment.py`).
    env: Environment
    # tenant name -> (isl_tokens, osl_tokens, image_count, image_pixels). **Per tenant, and
    # required.** A fleet-wide pair used to live here, which quietly asserted that every tenant
    # offered the same request shape -- and design §8.3 says a fleet like that is a fleet of one
    # workload wearing several models' names. Two runs of one fleet where a summarizer's ISL
    # changed from 512 to 8192 would have hashed identically, and prefill cost is most of what a
    # long-input tenant does to its neighbours.
    #
    # The image pair is in the same tuple rather than in a dict of its own, so a tenant cannot be
    # added to one and forgotten in the other. It matters here for a sharper version of the ISL
    # argument: at one ISL, a tenant sending four megapixel images and a tenant sending none are
    # doing different work -- most of the first one's prompt is visual tokens produced by a vision
    # tower that the second one never runs -- and without this field those two runs hashed
    # identically and landed in one directory, where retrieval hands both to the reasoner as
    # evidence about one condition.
    sequence_shapes: dict[str, tuple]
    # tenant name -> tensor-parallel degree. **Per tenant, and required**, and it belongs here for
    # exactly the reason `sequence_shapes` does. TP is not something the search varies: the search
    # decides *which* GPUs a tenant gets, never how many shards it runs in, so it is a fixed input
    # of the experiment in the same category as ISL. Two runs of one fleet where a 30B moved from
    # TP=2 to TP=4 are two different serving configurations of the same models -- different weights
    # per shard, different collectives, different achievable ITL -- and without this field they
    # hashed identically and landed in one directory, where retrieval hands both to the reasoner as
    # evidence about one condition.
    tensor_parallel: dict[str, int]
    # tenant name -> the engine flags it is served with, other than TP and the envelope. **Per
    # tenant, and required**, and the argument is `tensor_parallel`'s argument again with a longer
    # list of flags: `--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`, `--dtype` and
    # `extra_args` are all fixed by the fleet file, all read by the engine, and all decide what a
    # latency number means. Without this field two runs differing only in a batch cap or a dtype
    # hashed identically and landed in one directory. See `serving.py` for the rule and for how it
    # was found.
    serving: dict[str, ServingConfig]
    # None before the alpha sweep has run. Calibration itself executes at rates that are
    # not yet a fraction of anything, so a key with alpha=None is legitimate and distinct
    # from every keyed-by-alpha search that follows.
    alpha: float | None = None
    # tenant name -> offered request rate. **Required whenever `alpha` is None**, and the
    # reason is a hole this key had until it was used for two runs of one fleet at different
    # load. `alpha` is the load scale as a fraction of solo saturation, so when it is set the
    # absolute rates are derivable from it and this field is redundant. When it is None -- which
    # is legitimate, and is the case for every run before the saturation probe exists -- there
    # was nothing in the key naming how hard the fleet was driven, so a 2 Hz run and a 16 Hz run
    # of the same fleet on the same image hashed IDENTICALLY and landed in one directory. Every
    # such record is individually valid, so nothing anywhere reports the collision: retrieval
    # hands both to the reasoner as evidence about one condition.
    offered_rates_hz: dict[str, float] = field(default_factory=dict)
    # served name -> Hugging Face revision. Impl §13 flags that revisions should be pinned
    # in fleet definitions rather than resolved at pull time; until they are, this may be
    # empty, and an empty mapping is recorded as such rather than treated as "no models".
    model_revisions: dict[str, str] = field(default_factory=dict)

    def canonical(self) -> dict:
        """The exact dict that gets hashed.

        Every field appears, including the ones that are None. Dropping a null would make
        `alpha=None` hash identically to a key that never had an alpha field at all, and
        those are different environments.
        """
        alpha = None if self.alpha is None else round(float(self.alpha), ALPHA_DECIMALS)
        rates = {str(k): round(float(v), RATE_DECIMALS)
                 for k, v in sorted(self.offered_rates_hz.items())}
        shapes = {}
        for k, v in sorted(self.sequence_shapes.items()):
            shape = {"isl_tokens": int(v[0]), "osl_tokens": int(v[1])}
            # Emitted only for a tenant that actually sends images, which is the one deliberate
            # exception to the every-field-appears rule below, and it is not the same case. A key
            # with `alpha=None` and a key written before alpha existed describe *different*
            # environments, so dropping the null would conflate two things. A key written before
            # images existed describes a text-only run, because the harness of that era could not
            # send an image at all -- so absence and `image_count: 0` are the same environment, and
            # emitting the zeros would re-key every correct measurement in the store to say
            # nothing new.
            if int(v[2]):
                shape["image_count"] = int(v[2])
                shape["image_pixels"] = int(v[3])
            shapes[str(k)] = shape
        return {
            "environment": self.env.canonical(),
            "workload": {
                "sequence_shapes": shapes,
                "alpha": alpha,
                "offered_rates_hz": rates,
            },
            # Sorted so two keys built by iterating a fleet in different orders agree.
            "tensor_parallel": {str(k): int(v)
                                for k, v in sorted(self.tensor_parallel.items())},
            "serving": {str(k): v.canonical() for k, v in sorted(self.serving.items())},
            "model_revisions": dict(sorted(self.model_revisions.items())),
        }

    @property
    def hash(self) -> str:
        # sort_keys and a fixed separator: the hash must not depend on dict insertion
        # order or on json's default spacing.
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:HASH_CHARS]

    def key_json(self) -> str:
        """What gets written to `key.json` (impl §8).

        The hash is one-way, so the full key is stored beside the observations it governs.
        Without it a directory name is unattributable six months later, which defeats the
        point of keying at all.
        """
        return json.dumps({"hash": self.hash, **self.canonical()},
                          indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, d: dict) -> ValidityKey:
        """Rebuild from a `key.json` payload or a fleet chart's `validity_key` block."""
        w = d.get("workload", d)
        if "environment" not in d:
            # Refused rather than lifted, for the reason the three refusals below give. A key written
            # when the environment was a driver string and an instance type does not record what the
            # card addressed, what it was allowed to draw, how its devices were linked or how they
            # were shared -- and those are the terms every number under it depends on. The one-off
            # migration that re-keyed this store read those fields off the live node instead, which is
            # the only honest way to supply them, and it is not something a reader can do later.
            raise ValueError(
                "this key has no environment block. It was written when the environment was named by "
                "a driver version and an instance type, neither of which says what the device "
                "reported about itself, so it cannot be read as an address for these rows. "
                "Re-measure under a fresh key.")
        if "sequence_shapes" not in w:
            # A pre-per-tenant key.json. Refused rather than lifted to the new shape, because
            # the lift would have to guess that the fleet-wide pair applied to every tenant --
            # and a guessed key hashes to a directory whose records are not what it claims.
            raise ValueError(
                "this key has no workload.sequence_shapes. It was written when the sequence "
                "shape was fleet-wide, and there is no honest way to read it as a per-tenant "
                "key: whether every tenant really offered that one shape is not recorded. "
                "Re-measure under a fresh key.")
        if "serving" not in d:
            # Same refusal as the two above, same reason. A key written before the engine flags were
            # in it does not say what `--max-num-seqs`, `--max-num-batched-tokens`,
            # `--max-model-len`, `--dtype` or `extra_args` were, and the chart defaults that would
            # be guessed are exactly the values that turned out to be load-bearing. Guessing them
            # would file rows measured at one batch cap under a key claiming another.
            raise ValueError(
                "this key has no serving block. It was written when the engine flags a fleet file "
                "fixes -- max_num_seqs, max_num_batched_tokens, max_model_len, dtype, "
                "extra_args -- "
                "were outside the key, so two runs differing only in one of them hashed into this "
                "one directory. Re-measure under a fresh key.")
        if "tensor_parallel" not in d:
            # Same refusal, same reason. A key written before TP was in it does not say how many
            # shards each tenant ran in, and the fleet file that would say is not this file --
            # nor necessarily the one it was measured against. Guessing TP=1 would file a TP=4
            # run's records under a key that claims they are single-GPU measurements.
            raise ValueError(
                "this key has no tensor_parallel. It was written when TP was outside the key, so "
                "which tenant ran in how many shards is not recorded, and a TP=2 run and a TP=4 "
                "run of the same fleet hashed into this one directory. Re-measure under a fresh "
                "key.")
        return cls(
            env=Environment.from_dict(d["environment"]),
            # An absent image pair reads as no images, for `canonical`'s reason: a key without one
            # was written by a harness that had no route for an image, so the run it governs was
            # text. This is not the guess the three refusals above decline to make -- there, what
            # the run was conditioned on is genuinely unrecorded.
            sequence_shapes={k: (int(v["isl_tokens"]), int(v["osl_tokens"]),
                                 int(v.get("image_count", 0)), int(v.get("image_pixels", 0)))
                             for k, v in (w["sequence_shapes"] or {}).items()},
            tensor_parallel={k: int(v) for k, v in (d["tensor_parallel"] or {}).items()},
            serving={k: ServingConfig.from_dict(v) for k, v in (d["serving"] or {}).items()},
            alpha=w.get("alpha"),
            offered_rates_hz=dict(w.get("offered_rates_hz") or {}),
            model_revisions=dict(d.get("model_revisions") or {}),
        )

    def __post_init__(self) -> None:
        if not self.sequence_shapes:
            raise ValueError(
                "a key must name every tenant's sequence shape. Without it nothing in the key "
                "says what was being asked of the fleet, and a 512-in run and an 8192-in run "
                "land in one directory.")
        for name, shape in self.sequence_shapes.items():
            shape = tuple(shape)
            if len(shape) != 4:
                # Four, not "two or four". A shape that may arrive in either form is a shape with
                # two representations, and then a text tenant written the short way and the same
                # tenant written the long way are one condition with two hashes.
                raise ValueError(
                    f"sequence_shapes[{name!r}] must be "
                    "(isl_tokens, osl_tokens, image_count, image_pixels); a text tenant states "
                    f"0 and 0 for the last two. Got {shape!r}")
            if not all(int(v) >= 1 for v in shape[:2]):
                raise ValueError(
                    f"sequence_shapes[{name!r}]: isl_tokens and osl_tokens must both be >= 1; "
                    f"got {shape!r}")
            if not all(int(v) >= 0 for v in shape[2:]):
                raise ValueError(
                    f"sequence_shapes[{name!r}]: image_count and image_pixels must both be >= 0; "
                    f"got {shape!r}")
            if bool(int(shape[2])) != bool(int(shape[3])):
                raise ValueError(
                    f"sequence_shapes[{name!r}]: image_count={shape[2]} and "
                    f"image_pixels={shape[3]} -- state both or neither. Images of no size and a "
                    "size nothing is sent at are each half a workload, and a key that names half "
                    "a workload does not identify what was measured.")
        if set(self.tensor_parallel) != set(self.sequence_shapes):
            raise ValueError(
                f"the key names TP degrees for {sorted(self.tensor_parallel)} but shapes for "
                f"{sorted(self.sequence_shapes)}. Both are per tenant and both come from the same "
                "fleet, so a disagreement means one of them was assembled by hand.")
        for name, tp in self.tensor_parallel.items():
            if int(tp) < 1:
                raise ValueError(f"tensor_parallel[{name!r}] must be >= 1; got {tp!r}")
        if set(self.serving) != set(self.sequence_shapes):
            raise ValueError(
                f"the key names serving flags for {sorted(self.serving)} but shapes for "
                f"{sorted(self.sequence_shapes)}. Both are per tenant and both come from the same "
                "fleet, so a disagreement means one of them was assembled by hand.")
        if self.alpha is None and not self.offered_rates_hz:
            raise ValueError(
                "a key with alpha=None must name its offered rates. Without either, nothing "
                "in the key says how hard the fleet was driven, so two runs of the same fleet "
                "at different load hash to the same directory -- and every record in it is "
                "individually valid, so nothing warns anybody.")
        if self.offered_rates_hz and set(self.offered_rates_hz) != set(self.sequence_shapes):
            raise ValueError(
                f"the key names rates for {sorted(self.offered_rates_hz)} but shapes for "
                f"{sorted(self.sequence_shapes)}. Both are per tenant and both come from the "
                "same fleet, so a disagreement means one of them was assembled by hand.")
