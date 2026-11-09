"""Which candidate to try next (impl §11, "What the agent decides").

The loop is fixed; the policy is the seam. This module holds the seam itself -- `Policy`, what a
policy may look at (`Context`), and what it hands back (`Proposal`) -- and exactly one policy.

**There is deliberately no search rule here.** Earlier versions shipped two: `FixedSequence`,
which walked a designed co-residency sweep, and `ExploreExploit`, which ranked by a predicted
latency and departed from that ranking on a calibrated noise floor. Both are gone, along with the
predictor the second one ranked by. impl §11 says of
credit assignment that fixing a discipline in advance "would pre-empt the reasoning the agent
exists to perform", and a schedule deciding when to explore is exactly such a discipline. The
reasoner declares exploration by saying it does not know what will happen (`expectation: unknown`,
impl §11.2), and nothing sits above it converting that into a quota.

`TopRanked` is what remains, and it is not a search rule: it takes whatever the solver already
ranked first and never forms a belief of its own. It exists because impl §11.2 requires a
malformed or failed reasoner reply to fall back to the solver's top-ranked candidate rather than
abandon the run -- and requires the fallback to be recorded, since a silent one would make the
transcript a lie.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..retrieval import placement_of, sm_caps_of

log = logging.getLogger(__name__)


def canonical(assignment: dict, sm_caps: dict | None = None) -> tuple:
    """A hashable identity for a placement's *assignment* and SM caps, ignoring envelopes.

    Envelopes are excluded on purpose: two candidates differing by one envelope step are the
    same experiment as far as co-residency goes, and letting them count as distinct would let
    the search spend its budget re-measuring one structure.

    SM caps (impl §16) are included, because a tenant held to a quarter of the SMs is not the same
    experiment as that tenant uncapped. Uncapped, the identity is exactly what it was before caps
    existed, so the store's earlier records keep theirs.
    """
    base = tuple(sorted((str(n), tuple(sorted(int(g) for g in gpus)))
                        for n, gpus in assignment.items()))
    if not sm_caps:
        return base
    return (base, tuple(sorted((str(n), int(c)) for n, c in sm_caps.items())))


@dataclass(frozen=True)
class Attempt:
    """One completed iteration: what was proposed, and what came back."""

    run_id: str
    label: str
    candidate: object             # solver.Candidate as proposed
    record: dict | None           # the observation, or None if it never got measured
    outcome: str                  # "measured", "infeasible_deploy", "readiness:<state>", ...
    sm_caps: dict = field(default_factory=dict)   # impl §16; {tenant: percent}, empty uncapped

    @property
    def key(self) -> tuple:
        return canonical(self.candidate.placement.assignment, self.sm_caps)


@dataclass(frozen=True)
class Proposal:
    """What the policy wants tried next."""

    candidate: object             # solver.Candidate
    label: str                    # short, goes in the run id and the record's provenance
    reason: str                   # why this one; recorded so a search is auditable
    # Merged into the observation's provenance by the loop. Exists for policies whose decision is
    # not reconstructible from the store: a placement chosen by a language model, without the
    # prompt and the reply that chose it, is a measurement nobody can audit. Empty for `TopRanked`,
    # whose `reason` string is the whole of its justification -- it took what it was handed.
    provenance: dict = field(default_factory=dict)
    # The reasoner's SM ceilings for this measurement (impl §16), `{tenant: percent}`. Not part of
    # the candidate, because the solver never sees them: they change what is measured, not what fits.
    sm_caps: dict = field(default_factory=dict)


@dataclass
class Context:
    """Everything a policy is allowed to look at."""

    fleet: object                                  # FleetSpec
    history: list = field(default_factory=list)    # list of Attempt
    calibration: object = None                     # Calibration or None
    iteration: int = 0
    solve: object = None                           # (cap) -> FeasibleSet
    solve_pinned: object = None                    # (assignment) -> Candidate
    # Every usable record under this validity key, not just this run's attempts. `TopRanked` does
    # not look at it, but a reasoner asked to explain a *miss* needs the measurements themselves,
    # not a summary of them -- and under the design the records are the whole of what it has to
    # reason from. The reasoner-driven loop reads them through `retrieval` instead, per query.
    observations: list = field(default_factory=list)

    @property
    def tried(self) -> set:
        """Placements attempted *in this run*. Not the same as what the store holds.

        A resumed search therefore re-measures placements already in the store. That is a known
        gap rather than an intention; it is recorded here because the loop's
        docstring claims a resumed search behaves like an uninterrupted one, and this is the one
        respect in which it does not.
        """
        return {a.key for a in self.history}

    @property
    def gpu_of(self) -> dict:
        """`{uuid: ordinal}` for this node. The only thing relating a record to a candidate.

        Records name GPUs by UUID; a candidate names them by host ordinal. Nothing in a record
        carries the correspondence, so it comes from the fleet's node map -- which is why this is a
        property of the context rather than of an observation, and why `retrieval` takes it as an
        argument instead of deriving it.
        """
        return {u: g for g, u in zip(self.fleet.node.gpu_ids, self.fleet.node.uuids)}

    @property
    def measured(self) -> set:
        """Placements in the store, by `canonical()` identity, as far as they can be resolved.

        `placement_of` returns None -- not a partial map -- for a record from another node or one
        with no UUIDs, and such a record is skipped: claiming a placement was already measured when
        it was not would hide a candidate that still needs measuring.
        """
        of = self.gpu_of
        return {canonical(p, sm_caps_of(obs))
                for obs, p in ((o, placement_of(o, of)) for o in self.observations) if p}

    def threshold_ms(self, bucket: str) -> float | None:
        """The calibrated between-run half-width for a co-residency bucket, or None.

        None is not a zero. Without calibration there is no measured resolution, so nothing
        can be declared indistinguishable and the search must run to its budget instead.
        """
        if self.calibration is None:
            return None
        return self.calibration.threshold_ms(bucket)


class Policy:
    def next(self, ctx: Context) -> Proposal | None:
        """The next thing to try, or None to stop. None means done, not failed."""
        raise NotImplementedError


class TopRanked(Policy):
    """Take the solver's first candidate. The fallback of impl §11.2, and nothing more.

    Used when the reasoner cannot be reached or cannot produce a usable reply. It has no opinion
    about what to measure: the solver ranks by packing density under the SLO-derived KV floors,
    and this hands back whatever came out first. Candidates already attempted in this run are
    skipped, because falling back twice in a row would otherwise re-measure one placement until
    the budget ran out -- and that is bookkeeping, not a search rule.

    The iteration budget is enforced here as well, so a run whose reasoner has gone away still
    terminates on its own budget rather than on the solver running out of candidates.
    """

    def __init__(self, top_k: int = 4, max_iterations: int | None = 12):
        self.top_k = int(top_k)
        self.max_iterations = max_iterations

    def next(self, ctx: Context) -> Proposal | None:
        if self.max_iterations is not None and ctx.iteration >= self.max_iterations:
            log.info("stopping: iteration budget of %d spent", self.max_iterations)
            return None

        feasible = ctx.solve(self.top_k)
        tried = ctx.tried
        untried = [c for c in feasible
                   if canonical(c.placement.assignment) not in tried]
        if not untried:
            log.info("stopping: all %d candidate(s) the solver offers have been attempted in "
                     "this run (%s)", len(feasible), feasible.stopped_because)
            return None

        best = untried[0]      # solve() returns them ranked
        return Proposal(best, f"top-{ctx.iteration:02d}",
                        "the solver's top-ranked remaining candidate, taken as ranked; no "
                        "judgement was applied to it")
