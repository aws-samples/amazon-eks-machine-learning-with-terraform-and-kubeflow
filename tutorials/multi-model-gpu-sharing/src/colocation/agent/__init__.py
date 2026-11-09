"""The search agent (impl §11): propose, deploy, measure, record.

The other packages are the pieces this one closes the loop over -- `solver` proposes what is
feasible, `store` records what happened. There is no third package that refines a model between
those two, because impl §9 predicts nothing to refine. What was missing until this package existed
was actuation: a function from a `Placement` to running pods, and back from a finished run to an
observation.

    spec        what the search is over (`FleetSpec`) and what it holds fixed
    hardware    instance type -> GPU count, GPU product, measured device memory per GPU
    capture     `NodeCapture`: lease a node by installing the daemon that holds it  (+ a fake)
    fleetfile   a fleet as YAML, plus the GPU UUID map read off the cluster
    render      Placement -> chart values, and Placement -> harness run config
    deploy      `FleetDeployer`: apply, the readiness gate, teardown  (+ a fake)
    measure     `HarnessRunner`: run the §6 harness as a Job, return the record  (+ a fake)
    mirror      move records between EFS and the agent's local store
    policy      `Proposal`, and `TopRanked` for when the reasoner cannot answer
    loop        `SearchAgent`: render, deploy, gate, measure, record. Decides nothing.
    prompt      what the reasoner is told, and what it may say back (impl §11.2)
    tools       the six tools, and every invariant impl §11.2 puts at a tool boundary
    packer      `ModelPackerAgent`: the reasoner *is* the loop, over the `converse` seam
    sdkpacker   `SdkPackerAgent`: the same search on the Claude Agent SDK, and what ships

**One agent lives here now, and it asks which *placement* to measure.** A second one used to live
beside it: `onboard`/`onboarder`/`sdkonboarder` asked which *experiment* settled a model's memory
law, and `boot`/`booter`/`modelinfo` were its instruments. All six are gone (impl §14.4). What they
measured, they measured by reading an engine's own account of its memory -- which settles nothing
that generalises past the engine whose log was parsed -- and what they produced, a human pasted into
a fleet file. The refusal that motivated them is gone too: arithmetic over a fleet file's KV geometry
may now only choose experiments, never rule a degree out (§14.2), so nothing has to be measured
before a fleet can be planned.

`brief.py` and `llmpolicy.py` are gone, superseded rather than left inert. They were the one-shot
shape of this: the whole store rendered into a single prompt, and a reply that had to be a
decision. Two things were wrong with it beyond style -- the store grows until such a prompt cannot
be sent at all, and *which* prior runs bear on a candidate was decided by the renderer rather than
by the reasoner. `prompt.py` states the problem, `tools.py` answers questions about it, and what
gets looked at is the reasoner's choice.

Deploy and measure are interfaces with fakes, so the loop is testable with no cluster. That is
not tidiness: every other layer here is already checkable offline, and a loop that could only be
exercised against a live g6e node would be the one component nobody could check before spending
GPU time on it.

Two drivers for the same search, and the split follows from that. `SdkPackerAgent` is the runtime
impl §11 names and the one a live run uses; `ModelPackerAgent` drives the `Reasoner.converse` seam,
which is what a `ReplayReasoner` can be substituted into, so the self-tests exercise the loop,
the fallback and every refusal for free. Neither decides anything the other does not: the tools,
the invariants, the fallback and the derived verdict are shared code, and only the transport
differs. Importing this package still pulls in no SDK -- `sdkpacker` imports it inside the method
that needs it, so the package remains checkable with nothing installed. `SdkPackerAgent` enforces its
tool set with a deny-by-default hook rather than asserting it in a prompt, and `measure` is the tool
that gate exists for: the one that costs anything.
"""

from .capture import (CaptureError, FakeCapture, HelmCapture, NodeCapture, NodeLease,
                      new_run_token, reconcile_hardware)
from .deploy import (FakeDeployer, FleetDeployer, HelmDeployer, ReadyResult, DeployError,
                     READY, NOT_READY_ENGINE, NOT_READY_SCHEDULE, NOT_READY_TIMEOUT,
                     NOT_READY_VERIFY, guard_release)
from .hardware import (SHAPE_OBSERVED, SHAPE_STATED, Hardware, UnknownHardware,
                       hardware_for, normalize_gpu)
from .loop import SearchAgent, SearchResult, closest_to_targets
from .measure import FakeHarness, HarnessRunner, JobHarness, MeasureError
from .mirror import (EfsMirror, HardwareMirror, MirrorError, PodMirror,
                     ProfileMirror)
from .packer import PACKER_FALLBACK, ModelPackerAgent, PackerResult
from .policy import Attempt, Context, Policy, Proposal, TopRanked, canonical
from .profiler import (ALREADY_PROFILED, DID_MEASURE, FORCED_BY_MEMORY, NOT_PLACEABLE, NOT_READY,
                       Round, SoloProfiler, measured_reserve_gib, profile_from_record, solo_fleet,
                       solo_key)
from .prompt import InvalidReply, ToolError
from .render import (RenderError, check_agreement, exporter_service_url, render_releases,
                     render_run_config, render_tenant_values, render_values, service_url)
from .sdkpacker import SdkPackerAgent
from .spec import (AGENT_ENVELOPE_STEP_MIB, DEFAULT_IMAGE, FleetSpec, NodeInventory, TenantSlo,
                   TenantTemplate, WorkloadSpec)
from .tools import Tools

__all__ = [
    "ALREADY_PROFILED", "DID_MEASURE", "FORCED_BY_MEMORY", "NOT_PLACEABLE", "NOT_READY",
    "Round", "SoloProfiler", "measured_reserve_gib", "profile_from_record", "solo_fleet",
    "solo_key",
    "AGENT_ENVELOPE_STEP_MIB", "Attempt", "CaptureError", "Context",
    "SHAPE_OBSERVED", "SHAPE_STATED", "reconcile_hardware",
    "DEFAULT_IMAGE",
    "HardwareMirror",
    "render_values",
    "DeployError", "EfsMirror", "FakeCapture", "FakeDeployer", "FakeHarness",
    "FleetDeployer", "FleetSpec", "Hardware", "HarnessRunner", "HelmCapture", "HelmDeployer",
    "InvalidReply", "JobHarness",
    "MeasureError", "MirrorError", "ModelPackerAgent", "NOT_READY_ENGINE", "NOT_READY_SCHEDULE",
    "NOT_READY_TIMEOUT", "NOT_READY_VERIFY", "NodeCapture", "NodeInventory", "NodeLease",
    "PACKER_FALLBACK", "PackerResult", "PodMirror", "ProfileMirror",
    "Policy", "Proposal", "READY", "ReadyResult",
    "RenderError", "SdkPackerAgent", "SearchAgent", "SearchResult", "TenantSlo",
    "closest_to_targets",
    "TenantTemplate", "ToolError", "Tools", "TopRanked", "UnknownHardware", "WorkloadSpec",
    "canonical", "check_agreement", "exporter_service_url", "guard_release", "hardware_for",
    "new_run_token", "normalize_gpu",
    "render_releases", "render_run_config", "render_tenant_values", "service_url",
]
