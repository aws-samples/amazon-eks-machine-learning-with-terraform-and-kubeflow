"""How a number was obtained. Three tiers, and the distinction is not pedantry.

    derived   arithmetic on a config file. Free, offline, no engine involved. As strong as the
              assumption underneath it and no stronger -- `memory.py`'s per-token page size is
              derived, lands to 0.12% on a dense model, and on a hybrid one is 3.27x too large on
              Qwen3.5-9B and 1.06x too SMALL on Qwen3.5-0.8B, because the assumption "every layer
              holds a KV page" is not in the arithmetic anywhere to be checked. Note the two
              directions: `derived` does not mean conservative, it means unchecked.
    executed  ran the reference implementation. Deterministic, exact, repeatable, and still no
              GPU: summing a `model.safetensors.index.json` for a weight total, or calling the
              engine's own `smart_resize` for a visual token count, are executions rather than
              estimates. There is no measurement a GPU could add, because the function is pure.
    measured  read off a physical device. Authoritative about that device, noisy, and scoped to
              the validity key it was taken under. The only tier that can contradict the other
              two, and the only one that costs GPU time.

The tiers exist because the useful question about a number is not "what is it" but "what is it
worth", and the answer determines what a consumer may do with it: an over-safe `derived` envelope
is fine for exploring a plan and wrong to commit a fleet on, while a `measured` one is worth
nothing outside its key. A system that stores numbers without their tier cannot tell a reader which
situation they are in, and the failure is silent in both directions.

The rule that goes with them: **nothing upgrades a tier.** A derived number that happens to agree
with a measurement is still derived, because what made it derived was the assumption it rests on,
not its value. Code that "promotes" a tier on agreement is code that will promote a coincidence.

Top-level and dependency-free on purpose. `agent/fleetfile.py` labels what it loads and
`solver/memory.py` carries the label, and `store/` has never imported `solver/` -- the solver needs
OR-Tools and the store runs on the critical path without it. A shared vocabulary between layers that
must not depend on each other belongs above all of them. Two former homes for it are gone, both with
`onboard` (impl §14.4): `solver/identify.py`, which re-exported it, and `store/facts.py`, which filed
the tiered fact sheets a person pasted into a fleet file. What is left of the vocabulary is one
label on one field -- `ModelMemory.kv_bytes_per_token_tier` -- and after §14.4 every loaded value of
it is `derived`, because the tiers above `derived` all named an engine's own report of itself. The
vocabulary stays because the *rule* below is what keeps it that way.
"""

from __future__ import annotations

TIER_DERIVED = "derived"
TIER_EXECUTED = "executed"
TIER_MEASURED = "measured"

# Weakest first. `TIERS.index(...)` is the comparison, so a consumer can ask for "at least
# executed" without hard-coding an order that would drift from this list.
TIERS = (TIER_DERIVED, TIER_EXECUTED, TIER_MEASURED)


def at_least(tier: str, floor: str) -> bool:
    """Whether `tier` is as strong as `floor`. Raises on an unknown tier rather than returning False.

    Raising because an unrecognised tier is a typo or a version skew, and the safe-looking answer
    -- False, "not strong enough" -- would make a fact sheet silently unusable instead of loudly
    malformed. A caller that wanted leniency can catch it; a caller that got False would never know.
    """
    for name, value in (("tier", tier), ("floor", floor)):
        if value not in TIERS:
            raise ValueError(f"{name}={value!r} is not one of {TIERS}")
    return TIERS.index(tier) >= TIERS.index(floor)
