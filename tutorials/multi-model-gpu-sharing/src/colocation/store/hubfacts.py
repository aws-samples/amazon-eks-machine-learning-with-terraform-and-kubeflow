"""Checkpoint facts from the model hub: what the file holds, read without downloading it.

This is the third source of facts on this stack and the only one that needs no GPU at all.
`store/profile.py` files what a driven round observed; `store/hwfacts.py` files what a card is; this
module files what a *checkpoint* is, and the difference that matters is that a checkpoint is a
published artifact. Its geometry and its byte count are not measurements of anything -- they are
properties of files sitting behind a URL, checkable by anyone, identical on every machine.

**Why this exists: the fleet file used to state them, and a person had to put them there.** Two
fields (impl §14.8, Fix 2) -- `weight_gib` and a `kv:` block of `{num_layers, num_kv_heads,
head_dim}` -- were transcribed by hand, one per model, from `config.json` and from a `du` or a
header sum on whatever filesystem the weights happened to be on. Every shipped fleet file carried a
paragraph arguing that its numbers were right, and the arguments were good. That is the problem: a
format whose correctness rests on a paragraph is a format that a second engine, a second cluster or
a second model cannot be onboarded into without writing another paragraph. The numbers are derivable,
so they are derived, and what the fleet file states about a model is its id.

**Three things are read, and none of them is a weight.**

    /api/models/<id>?revision=<rev>            the commit sha, the file list, and the Hub's own
                                               parameter count by dtype (a free cross-check on the
                                               header sum below; they agree to the byte on all
                                               eight checkpoints in `fleets/`)
    /<id>/resolve/<sha>/config.json            the architecture: layer count, layer *types*, KV
                                               heads, head dim, tied embeddings
    /<id>/resolve/<sha>/<file>.safetensors     the HEADER ONLY, by HTTP range request: eight bytes
                                               for its length, then the JSON. Every tensor's name,
                                               dtype and shape, which is every tensor's byte count
                                               -- from a few hundred KB of a checkpoint that may be
                                               105 GiB. The 47-shard GLM costs 94 range requests
                                               and no weights.

**What `weight_gib` becomes: a lower bound on resident weights, and deliberately not the file size.**
Three quantities differ and the fleet files have been bitten by all three:

  - *file bytes* -- what `ls -l` sums. Includes the safetensors headers, which on Qwen3.5-9B alone
    is 0.45 GiB of nothing. Never used here.
  - *checkpoint tensor bytes* -- the header sum. What the file holds.
  - *resident tensor bytes* -- what an engine puts on the device, which is the checkpoint minus the
    tensors it does not load. Two such exclusions exist on the eight checkpoints here and both are
    arithmetic on tensor names rather than a report from a boot:

      a duplicated tied head. `tie_word_embeddings: true` and a `lm_head.weight` tensor present
      means the checkpoint stores the embedding twice and the engine resides it once. Qwen3-0.6B:
      1.4000 GiB of tensors, 0.2898 of it a second copy, 1.1102 resident. The Qwen3.5 models are
      also tied and store no `lm_head` at all, so there is nothing to subtract -- which is why the
      rule tests for the tensor instead of trusting the flag.

      draft blocks for speculative decoding. A trailing layer past `num_hidden_layers`, or an
      `mtp.*` block, is a multi-token-prediction head that is resident only if the serving config
      turns speculative decoding on. GLM-4.5-Air-FP8's layer 46 is 3.982 GiB of a 104.83 GiB
      checkpoint -- about 1 GiB per shard at tp4 that the fleet file was reserving and no boot was
      using. The four Qwen3.5 checkpoints each carry an `mtp.*` block of 0.04 to 0.45 GiB.

    `weight_gib` is the third quantity. It is the one §14.2 needs: the weight floor is the only term
    allowed to *refuse* a degree, so it must be something no boot can come back needing less than.

    The risk of that choice is real and is stated rather than padded away. A lower bound under-states
    the fixed part of a shard, which over-states the KV pool priced on top of it, and the failure mode
    is a boot that asks for more cache than fits. §14.2 accepts exactly that: a boot that fails for
    want of memory is a witness that moves the walk, measured on the driver. What is NOT done here is
    add a fudge factor to cover it -- the shipped fleets measured an engine residing 0.13 to 0.16 GiB
    per shard more than their header sums, which is allocator overhead, and a constant typed in here
    to cover it would be a hardware-specific and engine-specific constant in the one place this
    design has none.

**What the `kv:` block becomes: attention layers, not all layers.** The product a person used to
write counts every layer as holding a KV page. A hybrid family breaks it, and `config.json` says so
in a field nobody was reading: `layer_types` marks Qwen3.5-9B's 32 layers as 8 `full_attention` and
24 `linear_attention`, so the page is 8 x 4 x 256 x 2 x 2 = 32,768 B/token and not the 131,072 the
all-layers product gives. That is the 3.76x over-statement the multimodal fleet documents, fixed by
reading one more config field rather than by measuring anything.

**And the term that is NOT fixed, reported here and priced nowhere yet.** A linear-attention layer
keeps a recurrent state that is charged *per sequence*, not per token, so it does not belong in a
page size at all -- which is why the measured pages on that family move with `max_model_len` and why
no single byte count can be right at two contexts. `state_bytes_per_seq` is derived here from the
`linear_*` config fields and is carried for whoever prices it (Fix 3, where `max_num_seqs` is already
the input that decides how many of them exist). Checked against the four measured pages in
`qwen35-multimodal.yaml`: the state those pages imply runs 7-11% ABOVE this estimate on the
4B, 2B and 0.8B and 37% above on the 9B, so it is an estimate of the right shape and not a fact, and
it errs in the direction that under-prices memory. Nothing refuses on it.

**Caching, and why this store is the one that is not mirrored to EFS.** A manifest is keyed by
`(model_id, commit sha, schema version)` and is immutable under that key, so it is written once and
read forever from `<root>/hubfacts/<model>/<sha>-v<n>.json`. It is deliberately *not* pushed to
`/efs/colocation` the way profiles, observations and hardware sheets are: those are irreplaceable --
re-deriving one costs a GPU round on a leased node -- while this is a public URL and a few hundred KB
of range requests. Mirroring it would put a free, refetchable convenience behind a `kubectl exec`
into a pod, which is the cost this store's mirrors exist to justify for evidence that cannot be
refetched at all.

An unpinned `revision` ("main") does need the network once per run, to learn which sha it points at.
When that fails and a manifest for this model is already cached, the newest cached sha is used and
the fall-back is logged -- `plan` must keep working on a machine that cannot reach the Hub, and the
alternative is refusing to answer an offline question that has an answer sitting on disk. A fleet
that wants this pinned states `model_revision` and then nothing is resolved at all.

Pure standard library, like every other module under `store/`.
"""

from __future__ import annotations

import json
import logging
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Bumped when the set of fields read out of `config.json` or the shape of the tensor rollup changes.
# It is in the cache *filename*, so a bump refetches rather than reinterpreting -- which costs a few
# HTTP requests and no GPU time, the whole reason this store can be casual about invalidation.
HUBFACTS_SCHEMA_VERSION = 1

DEFAULT_HUB_ENDPOINT = "https://huggingface.co"

# Bytes per element, by safetensors dtype name. FP8 is here because a quantized checkpoint is the
# case where a parameter count times two is wrong by half: GLM-4.5-Air-FP8 is 108.5G F8_E4M3 params
# and 1.9G BF16, and pricing it as bf16 would over-read its weights by 100 GiB.
DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
    "F4": 1, "F6_E2M3": 1, "F6_E3M2": 1,   # sub-byte, rounded up: over-reads, which is the safe way
}

# A layer type whose name contains one of these keeps a recurrent/convolutional state instead of a
# KV page. Matched as substrings because the spellings are not standardised -- `linear_attention`,
# `mamba`, `mamba2`, `recurrent` have all appeared -- while everything that does cache K and V spells
# it `*attention` (`full_attention`, `sliding_attention`, `chunked_attention`), and those count.
STATE_LAYER_MARKERS = ("linear", "mamba", "recurrent", "ssm", "conv")

# The header of one safetensors file, fetched blind before its declared length is known. Big enough
# for every checkpoint in `fleets/` (the largest is ~1.2 MB over 47 shards) and re-fetched exactly
# when it is not.
_HEADER_PROBE_BYTES = 4 << 20


class HubUnavailable(Exception):
    """The Hub could not be reached and no cached manifest could stand in.

    Its own exception type because the remedy is different from every other failure in this package:
    nothing is wrong with the fleet file, the store or the cluster, and the fix is a network or a
    `model_revision` that pins what is already on disk.
    """


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(name).strip().lower())


def _tensor_group(name: str) -> str:
    """The bucket a tensor's bytes are counted under.

    Not the tensor itself: Qwen3-30B-A3B-Base has 18,867 of them and GLM-4.5-Air 36,323, and a cache
    entry listing every one would be megabytes of JSON to answer questions about a dozen blocks. The
    bucket is the name truncated at its first numeric component, so `model.layers.46.mlp.experts.7
    .gate_proj.weight` counts under `model.layers.46` and a 128-expert layer is one entry. A name with
    no index keeps everything but its trailing `.weight`/`.bias`, so `lm_head` and
    `model.embed_tokens` stay distinguishable -- which is what the tied-head rule needs.

    Truncating at the index rather than at a fixed depth is what makes this work on a stack whose
    layers are nested differently: `model.layers.N` on Qwen3 and GLM, `model.language_model.layers.N`
    on Qwen3.5, `model.visual.blocks.N` for the vision tower beside it.
    """
    parts = str(name).split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[: i + 1])
    return ".".join(parts[:-1]) or str(name)


def _stacks(groups: dict) -> dict:
    """Indexed group prefixes -> the set of indices under each.

    A checkpoint can hold more than one stack of numbered blocks -- a text decoder and a vision
    tower -- and the rule that drops a draft layer is about the *text* stack's count. Keyed by the
    prefix so the two are never confused; a vision tower deeper than the decoder would otherwise
    have its last blocks read as draft weights and silently un-reserved.
    """
    out: dict = {}
    for name in groups:
        head, _, tail = name.rpartition(".")
        if head and tail.isdigit():
            out.setdefault(head, set()).add(int(tail))
    return out


@dataclass(frozen=True)
class KvGeometry:
    """What a checkpoint's own config says about its KV cache, and what it says about state.

    `kv_layers` is the count that holds K and V, which is every layer on a dense model and a
    quarter of them on Qwen3.5. `state_layers` is the rest -- reported so that a page derived from
    the first is never mistaken for the whole cost of a hybrid stack.
    """

    kv_layers: int
    state_layers: int
    num_kv_heads: int
    head_dim: int
    state_bytes_per_seq: int = 0

    @property
    def is_hybrid(self) -> bool:
        return self.state_layers > 0


@dataclass(frozen=True)
class HubManifest:
    """One checkpoint at one commit: its architecture and where its bytes are.

    Frozen and cache-shaped: everything in here came off the Hub, so two readers of the same
    `(model_id, revision)` hold the same object and no consumer can adjust one.
    """

    model_id: str
    revision: str
    config: dict
    groups: dict
    hub_parameters: dict
    schema_version: int = HUBFACTS_SCHEMA_VERSION

    # --- the architecture ----------------------------------------------------------------------

    @property
    def text_config(self) -> dict:
        """The config the decoder is described by.

        A multimodal checkpoint nests it: Qwen3.5's top-level `config.json` has `num_hidden_layers:
        null` and the real numbers under `text_config`, beside a `vision_config` that has layers and
        heads of its own. Reading the top level and finding None is how a KV page becomes a crash or,
        worse, a zero.
        """
        inner = self.config.get("text_config")
        return dict(inner) if isinstance(inner, dict) else dict(self.config)

    def kv_geometry(self) -> KvGeometry:
        cfg = self.text_config
        types = cfg.get("layer_types")
        layers = cfg.get("num_hidden_layers")
        if isinstance(types, list) and types:
            kv_layers = sum(1 for t in types
                            if not any(m in str(t).lower() for m in STATE_LAYER_MARKERS))
            state_layers = len(types) - kv_layers
        elif layers is None:
            raise HubUnavailable(
                f"{self.model_id}: config.json states neither `num_hidden_layers` nor `layer_types`, "
                "so nothing here knows how many layers cache K and V. A fleet may state "
                "`kv_bytes_per_token` for this model to price it anyway.")
        else:
            kv_layers, state_layers = int(layers), 0
        heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
        head_dim = cfg.get("head_dim")
        if not head_dim:
            hidden, n_heads = cfg.get("hidden_size"), cfg.get("num_attention_heads")
            if not hidden or not n_heads:
                raise HubUnavailable(
                    f"{self.model_id}: config.json states no `head_dim` and no `hidden_size` / "
                    "`num_attention_heads` to derive one from.")
            head_dim = int(hidden) // int(n_heads)
        if not heads:
            raise HubUnavailable(
                f"{self.model_id}: config.json states no `num_key_value_heads` or "
                "`num_attention_heads`.")
        return KvGeometry(kv_layers=int(kv_layers), state_layers=int(state_layers),
                          num_kv_heads=int(heads), head_dim=int(head_dim),
                          state_bytes_per_seq=self._state_bytes_per_seq(state_layers, cfg))

    @staticmethod
    def _state_bytes_per_seq(state_layers: int, cfg: dict) -> int:
        """The recurrent state a hybrid stack holds for one sequence, as an estimate.

        Per linear layer: one matrix state of `value_heads x key_head_dim x value_head_dim` elements,
        in `mamba_ssm_dtype` (float32 on Qwen3.5, which doubles it against the model's bf16). The
        short convolution state beside it is two orders of magnitude smaller and is left out rather
        than guessed at, since its layout is not stated in the config.

        An estimate, and labelled one everywhere it travels: against the four pages in
        `qwen35-multimodal.yaml` -- the only measurements of this quantity that exist here --
        the state they imply is 7-11% above this estimate on the 4B, 2B and 0.8B and 37% above on
        the 9B -- low, i.e. in the direction that under-prices memory. Nothing refuses on it.
        """
        if not state_layers:
            return 0
        v_heads = cfg.get("linear_num_value_heads")
        k_dim = cfg.get("linear_key_head_dim")
        v_dim = cfg.get("linear_value_head_dim")
        if not (v_heads and k_dim and v_dim):
            return 0
        elem = DTYPE_BYTES.get(str(cfg.get("mamba_ssm_dtype", "float32")).upper()
                               .replace("FLOAT32", "F32").replace("BFLOAT16", "BF16"), 4)
        return int(state_layers) * int(v_heads) * int(k_dim) * int(v_dim) * int(elem)

    # --- the bytes -----------------------------------------------------------------------------

    @property
    def checkpoint_bytes(self) -> int:
        """Every tensor in the checkpoint. What the file holds, not what a device holds."""
        return sum(int(v) for v in self.groups.values())

    def tied_head_bytes(self) -> int:
        """The second copy of an embedding a tied checkpoint stores, or zero.

        Both conditions, not either: the flag says the engine will use one tensor for two jobs, and
        the presence of `lm_head` says this file wrote it down twice anyway. Qwen3-0.6B is both;
        Qwen3.5-4B is tied and stores no head, so the flag alone would subtract a gigabyte of
        embedding that was never there.
        """
        tied = bool(self.config.get("tie_word_embeddings")
                    or self.text_config.get("tie_word_embeddings"))
        if not tied:
            return 0
        return int(self.groups.get("lm_head", 0))

    def draft_bytes(self) -> dict:
        """Group -> bytes for blocks an engine resides only under speculative decoding.

        Two spellings, one idea. `mtp.*` is a named multi-token-prediction block. A trailing index in
        the text stack past `num_hidden_layers` is the same thing written as an extra layer -- GLM
        declares 46 layers and ships 47, and `num_nextn_predict_layers: 1` is the config saying so.

        Scoped to the text stack on purpose: the vision tower is a numbered stack too, and it is
        resident on every boot of a multimodal tenant.
        """
        out: dict = {}
        for name, byts in self.groups.items():
            first = name.split(".")[0]
            if first == "mtp" or ".mtp." in f".{name}.":
                out[name] = int(byts)
        declared = self.text_config.get("num_hidden_layers")
        if declared:
            stacks = _stacks(self.groups)
            text_prefix, best = None, -1
            for prefix, idx in stacks.items():
                if prefix.startswith("mtp") or "visual" in prefix or "vision" in prefix:
                    continue
                if len(idx) > best:
                    text_prefix, best = prefix, len(idx)
            if text_prefix:
                for i in sorted(stacks[text_prefix]):
                    if i >= int(declared):
                        out[f"{text_prefix}.{i}"] = int(self.groups[f"{text_prefix}.{i}"])
        return out

    def resident_bytes(self) -> int:
        """A lower bound on what the weights cost on the device. See this module's docstring."""
        return (self.checkpoint_bytes - self.tied_head_bytes()
                - sum(self.draft_bytes().values()))

    def hub_parameter_bytes(self) -> int | None:
        """The same total, from the Hub's own dtype census. A cross-check, not a source.

        Independent of the header sum only in the weak sense that somebody else computed it, which
        is still worth having: it is one field of a response already being fetched, and it agreed to
        the byte on all eight checkpoints in `fleets/`. A disagreement is logged at load, not
        refused -- the header sum is the more specific evidence and there is no third opinion to
        break a tie with.
        """
        params = (self.hub_parameters or {}).get("parameters")
        if not isinstance(params, dict) or not params:
            return None
        total = 0
        for dtype, count in params.items():
            width = DTYPE_BYTES.get(str(dtype).upper())
            if width is None:
                return None
            total += int(count) * width
        return total

    # --- serialisation -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"model_id": self.model_id, "revision": self.revision,
                "schema_version": int(self.schema_version), "config": self.config,
                "groups": self.groups, "hub_parameters": self.hub_parameters}

    @classmethod
    def from_dict(cls, d: dict) -> HubManifest:
        return cls(model_id=str(d["model_id"]), revision=str(d["revision"]),
                   config=dict(d.get("config") or {}), groups=dict(d.get("groups") or {}),
                   hub_parameters=dict(d.get("hub_parameters") or {}),
                   schema_version=int(d.get("schema_version", HUBFACTS_SCHEMA_VERSION)))


def http_get(url: str, headers: dict | None = None, timeout_s: float = 60.0) -> bytes:
    """One GET, standard library only.

    The transport is a plain function so that a test can pass another one: every fetch in this
    module goes through it, and a suite that reached the real Hub would be a suite whose result
    depends on somebody else's uptime.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "colocation-agent", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise HubUnavailable(f"GET {url}: {exc}") from None


def _safetensors_header(get, url: str) -> dict:
    """The tensor table of one safetensors file, without its tensors.

    The format puts an 8-byte little-endian header length first, so the exact request is two round
    trips. One speculative range covers both for every checkpoint here, and the second request
    happens only when a header is bigger than the probe -- which is the case worth handling
    correctly rather than the case worth optimising.
    """
    blob = get(url, {"Range": f"bytes=0-{_HEADER_PROBE_BYTES - 1}"})
    if len(blob) < 8:
        raise HubUnavailable(f"{url}: {len(blob)} bytes returned for a safetensors header")
    (length,) = struct.unpack("<Q", blob[:8])
    if 8 + length > len(blob):
        blob = b"\x00" * 8 + get(url, {"Range": f"bytes=8-{8 + length - 1}"})
    try:
        return json.loads(blob[8:8 + length].decode())
    except (ValueError, UnicodeDecodeError) as exc:
        raise HubUnavailable(f"{url}: safetensors header is not JSON ({exc})") from None


class HubFactsStore:
    """Fetches checkpoint manifests, caches them under `(model_id, sha, schema)`, never mirrors.

    `endpoint` and `get` are both injectable, and for the same reason the mirrors take a pod name:
    the thing on the other side of the network is not this package's, so the seam is where the tests
    are. `token` is accepted and never logged; a gated checkpoint needs one and none of the eight in
    `fleets/` does.
    """

    def __init__(self, root: str | Path, *, endpoint: str = DEFAULT_HUB_ENDPOINT,
                 get=http_get, token: str | None = None):
        self.root = Path(root).expanduser() / "hubfacts"
        self.endpoint = str(endpoint).rstrip("/")
        self._get = get
        self._token = token
        self._memo: dict = {}

    # --- cache -------------------------------------------------------------------------------

    def _dir(self, model_id: str) -> Path:
        return self.root / _safe(model_id)

    def _path(self, model_id: str, sha: str) -> Path:
        return self._dir(model_id) / f"{_safe(sha)}-v{HUBFACTS_SCHEMA_VERSION}.json"

    def cached(self, model_id: str, sha: str) -> HubManifest | None:
        path = self._path(model_id, sha)
        if not path.exists():
            return None
        try:
            return HubManifest.from_dict(json.loads(path.read_text()))
        except (ValueError, KeyError) as exc:
            # Not a hard error, unlike an unreadable device law: this one is refetchable from a URL,
            # so a corrupt entry costs a few HTTP requests rather than an answer.
            log.warning("hub manifest %s is unreadable (%s); refetching", path, exc)
            return None

    def newest_cached(self, model_id: str) -> HubManifest | None:
        """The most recently written manifest for this model, whatever sha it is of.

        Only reached when the Hub cannot be asked which sha `main` points at. Newest by mtime
        because there is no ordering on shas, and the alternative to a stale answer here is no
        answer to an offline question that has one on disk.
        """
        paths = sorted(self._dir(model_id).glob(f"*-v{HUBFACTS_SCHEMA_VERSION}.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths:
            try:
                return HubManifest.from_dict(json.loads(path.read_text()))
            except (ValueError, KeyError):
                continue
        return None

    def write(self, manifest: HubManifest) -> Path:
        path = self._path(manifest.model_id, manifest.revision)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    # --- fetch -------------------------------------------------------------------------------

    def _api(self, path: str) -> dict:
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        blob = self._get(f"{self.endpoint}{path}", headers)
        try:
            return json.loads(blob.decode())
        except (ValueError, UnicodeDecodeError) as exc:
            raise HubUnavailable(f"GET {path}: response is not JSON ({exc})") from None

    def _file(self, model_id: str, sha: str, name: str) -> str:
        return f"{self.endpoint}/{model_id}/resolve/{sha}/{name}"

    def fetch(self, model_id: str, revision: str = "main") -> HubManifest:
        """Resolve a revision, read the config and every safetensors header, cache the result."""
        info = self._api(f"/api/models/{model_id}?revision={revision}")
        sha = str(info.get("sha") or revision)
        cached = self.cached(model_id, sha)
        if cached is not None:
            return cached
        files = sorted(s.get("rfilename", "") for s in (info.get("siblings") or [])
                       if str(s.get("rfilename", "")).endswith(".safetensors"))
        if not files:
            raise HubUnavailable(
                f"{model_id}@{sha} publishes no .safetensors files, so there is no manifest to read "
                "its weights from. A fleet may state `kv_bytes_per_token` and nothing else here can "
                "stand in for the weight bytes.")
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        config = json.loads(self._get(self._file(model_id, sha, "config.json"), headers).decode())
        groups: dict = {}
        for name in files:
            table = _safetensors_header(
                lambda url, h: self._get(url, {**headers, **h}), self._file(model_id, sha, name))
            for tensor, meta in table.items():
                if tensor == "__metadata__" or not isinstance(meta, dict):
                    continue
                width = DTYPE_BYTES.get(str(meta.get("dtype", "")).upper())
                if width is None:
                    raise HubUnavailable(
                        f"{model_id}@{sha}: tensor {tensor!r} has dtype {meta.get('dtype')!r}, "
                        "which this module has no element width for. Add it to `DTYPE_BYTES`; "
                        "guessing one would misprice every weight in the file.")
                count = 1
                for dim in meta.get("shape") or []:
                    count *= int(dim)
                key = _tensor_group(tensor)
                groups[key] = groups.get(key, 0) + count * width
        manifest = HubManifest(model_id=str(model_id), revision=sha, config=config, groups=groups,
                               hub_parameters={"parameters": (info.get("safetensors") or {})
                                               .get("parameters") or {}})
        self.write(manifest)
        log.info("hub manifest for %s@%s: %d tensor group(s) over %d file(s), %.4f GiB of tensors, "
                 "%.4f GiB resident lower bound", model_id, sha[:12], len(groups), len(files),
                 manifest.checkpoint_bytes / (1 << 30), manifest.resident_bytes() / (1 << 30))
        return manifest

    def manifest(self, model_id: str, revision: str | None = None) -> HubManifest:
        """The manifest for this model, from memory, then the cache, then the Hub.

        A pinned `revision` is a cache key, so a fleet that states one never touches the network on a
        warm machine. An unpinned one asks the Hub which sha it means -- and falls back to the newest
        cached manifest, loudly, when the Hub cannot be reached at all.
        """
        memo_key = (str(model_id), str(revision or "main"))
        if memo_key in self._memo:
            return self._memo[memo_key]
        if revision:
            hit = self.cached(model_id, str(revision))
            if hit is not None:
                self._memo[memo_key] = hit
                return hit
        try:
            manifest = self.fetch(model_id, str(revision or "main"))
        except HubUnavailable as exc:
            stale = self.newest_cached(model_id)
            if stale is None:
                raise HubUnavailable(
                    f"{model_id}: the model hub could not be read and nothing is cached for it, so "
                    f"this fleet's weight bytes and KV geometry are unknown ({exc}). They are not "
                    "defaultable: a guessed weight figure decides how many GPUs this fleet needs. "
                    "Restore network access to the hub, or pin `model_revision` to a commit already "
                    "cached under ~/.colocation/store/hubfacts.") from None
            log.warning("hub unreachable (%s); using the cached manifest for %s@%s, which may not be "
                        "what %r points at now", exc, model_id, stale.revision[:12],
                        revision or "main")
            manifest = stale
        self._memo[memo_key] = manifest
        return manifest
