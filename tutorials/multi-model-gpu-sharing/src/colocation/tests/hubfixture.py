"""The checkpoint manifests this suite prices fleets from, on disk and never over the network.

`store/hubfacts.py` made a `models:` entry an id: the weight bytes and the KV page size behind it are
read off the checkpoint's own `config.json` and safetensors headers rather than stated in the fleet
file (impl §14.8, Fix 2). That leaves the suite needing the answer for eight real checkpoints -- the
shipped fleets name them, and so do the loader fixtures -- while the suite's first line still says no
network.

**So the manifests are fixtures, and a manifest is the one fact here that legitimately can be.** A
profile row or a device law is a measurement: copying one into `fixtures/` would be freezing a claim
about hardware that only a GPU can make. A manifest is a *published artifact keyed by its own commit
sha* and immutable under that key -- `Qwen/Qwen3-0.6B` at `c1899de2` holds the same tensors on every
machine forever, and anybody can check it with `curl`. Storing it here costs 100 KB and buys a suite
that runs on a plane and gives the same answer in every environment.

They are the real cache files, byte for byte as `HubFactsStore.write` produced them, so the suite
reads them through the same `from_dict` a run does. Refresh one with

    PYTHONPATH=src python3 -c "from colocation.store import HubFactsStore; \
        print(HubFactsStore('/tmp/hub').fetch('Qwen/Qwen3-0.6B').revision)"

and copy the file it writes under `/tmp/hub/hubfacts/` into the matching directory here. A new sha is
a different file, not an edit: the old one keeps working, which is exactly the property that lets a
fleet pin `model_revision` and stop resolving anything.
"""

from __future__ import annotations

from pathlib import Path

from ..store.hubfacts import HubFactsStore, HubManifest, HubUnavailable

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


class FixtureHub(HubFactsStore):
    """A hub store that can only answer from the fixture directory, plus explicit overrides.

    `manifest` is overridden rather than the transport, and that is deliberate: the transport's own
    seam (`get`) is tested separately against a synthetic hub, while *this* class exists to make it
    impossible for a load anywhere in the suite to reach the network by accident -- including through
    the unpinned-revision path, which asks the Hub which sha `main` means before it looks in the
    cache at all. Here an unpinned revision means the newest fixture, silently: there is one sha per
    model in `fixtures/hubfacts/`, so "newest" is "the one there is".

    `overrides` is `{model_id: HubManifest}` and is how a test asks a question no published
    checkpoint can answer -- a 400 GiB model, say. It replaces what the checkpoint says about a
    model *without* editing the fleet file, which is itself the property Fix 2 bought: the weight
    figure is no longer a line in the file that a test could rewrite.
    """

    def __init__(self, root: str | Path = FIXTURE_ROOT, *, overrides: dict | None = None, **kw):
        super().__init__(root, **kw)
        self._overrides = dict(overrides or {})

    def manifest(self, model_id: str, revision: str | None = None) -> HubManifest:
        if str(model_id) in self._overrides:
            return self._overrides[str(model_id)]
        hit = (self.cached(model_id, str(revision)) if revision else None)
        if hit is None:
            hit = self.newest_cached(model_id)
        if hit is None:
            raise HubUnavailable(
                f"{model_id}: no manifest for it in {FIXTURE_ROOT / 'hubfacts'}, and this suite may "
                "not fetch one. Add the checkpoint's cache file (see tests/hubfixture.py) or point "
                "the fixture at a model that is already there.")
        return hit


def fixture_hub(overrides: dict | None = None) -> FixtureHub:
    """The one hub every offline load in this package goes through."""
    return FixtureHub(FIXTURE_ROOT, overrides=overrides)


def synthetic_manifest(model_id: str, *, resident_gib: float, num_layers: int = 28,
                       num_kv_heads: int = 8, head_dim: int = 128,
                       revision: str = "fixture") -> HubManifest:
    """A manifest with chosen numbers, for the checks no real checkpoint can exhibit.

    One tensor group and no tied head and no draft block, so `resident_bytes()` is exactly what was
    asked for: the cases this serves are about what the *loader* does with a weight figure over the
    card, and 400 GiB of weights is not a checkpoint anybody publishes. The geometry defaults are
    Qwen3-0.6B's, so a fleet fixture keeps its page size when only the weight is being moved.
    """
    return HubManifest(
        model_id=str(model_id), revision=revision,
        config={"num_hidden_layers": int(num_layers), "num_key_value_heads": int(num_kv_heads),
                "head_dim": int(head_dim)},
        groups={"model.weights": int(round(float(resident_gib) * (1 << 30)))},
        hub_parameters={})
