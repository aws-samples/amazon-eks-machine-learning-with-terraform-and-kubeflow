"""Offline self-test for the store, the solver and the agent.

No GPU, no network, no third-party packages beyond OR-Tools, which the solver needs, and PyYAML,
which the fleet-file reader does. No cluster is needed either; if kubectl does reach one, two
`plan` checks below make a read-only `kubectl exec` into `attach-pvc` against a path that does
not exist, and pass whether or not it answers.
Run from `tutorials/multi-model-gpu-sharing/src`:

    python -m colocation.tests.selftest

The fixtures are **real records** from live runs 5 and 6 on the g6e node, not synthetic ones.
That used to matter because there was a predictor to check against what the hardware did. There is
no predictor now (impl §9), and the fixtures matter for a different reason: a hand-written record
contains only the fields whoever wrote it remembered, while these contain the fields the harness
actually writes -- including the awkward ones, like a suppressed latency and a chunk-count
divergence nobody has explained.

What this file no longer contains is the interference model's tests, because the package is gone.
The checks that survived it are the ones that were never about prediction: the envelope arithmetic
(impl §4.2), which is arithmetic on a model's own config, and `coresidency`, which describes a
placement rather than scoring one.
"""

from __future__ import annotations

import ast
import atexit
import contextlib
import copy
import dataclasses
import hashlib
import inspect
import io
import json
import logging
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import yaml

from colocation.agent import (AGENT_ENVELOPE_STEP_MIB, Attempt, CaptureError, Context,
                              DEFAULT_IMAGE, DeployError, FakeCapture,
                              FakeDeployer, FakeHarness, FleetSpec, Hardware,
                              HelmCapture,
                              ModelPackerAgent,
                              NodeInventory, NOT_READY_ENGINE, NOT_READY_SCHEDULE,
                              NOT_READY_TIMEOUT, NOT_READY_VERIFY, PACKER_FALLBACK, PackerResult,
                              Policy, READY,
                              RenderError, SdkPackerAgent,
                              SearchAgent, SHAPE_OBSERVED,
                              SHAPE_STATED, TenantSlo, TenantTemplate,
                              Tools, TopRanked, UnknownHardware, WorkloadSpec,
                              check_agreement, closest_to_targets,
                              exporter_service_url,
                              guard_release, hardware_for,
                              normalize_gpu,
                              render_releases, render_run_config,
                              render_tenant_values)
from colocation.agent import __main__ as agent_main
from colocation.agent import capture as agent_capture
from colocation.agent import smi as agent_smi
from colocation import engines as _eng
from colocation.agent import export as exportmod
from colocation.agent import fleetfile
from colocation.agent.hardware import DRIVER_TOTAL_QUANTUM_GIB
from colocation.agent import profiler as agent_profiler
from colocation.agent import prompt
from colocation.agent import render
from colocation.agent.calibrate import between_run_halfwidth, bucket, placement_key
from colocation.agent.deploy import _classify
from colocation.agent.measure import JobHarness, MeasureError, _parse_interleaved
from colocation.agent.packer import (STOP_SUBMITTED, STOP_TURNS, derive_verdict,
                                     expectation_report, fallback_note, recorded_fallback)
from colocation.agent.tools import SUBMITTED, expectation_lines
from colocation.agent.mirror import EfsMirror, HardwareMirror, MirrorError
from colocation.agent.policy import Proposal
from colocation.agent import splitpacker
from colocation.agent.reason import (BedrockReasoner, EMPTY_REPLY, NullReasoner, ReplayReasoner,
                                     Transcript, Turn, ask)
from types import SimpleNamespace

from colocation.agent import sdkpacker
from colocation import retrieval
from colocation.inspect import _ordinal_names, record_lines
from colocation.retrieval import (EXCLUDED_KEY, coresident_pairs, placement_of, read_run,
                                  recall, required_relaxation, target_lines, targets_met)
from colocation.solver import (ACTIVATION_GIB_DEFAULT,
                               ENVELOPE_STEP_MIB_DEFAULT, FeasibleSet,
                               HEADROOM_DEFAULT,
                               Infeasible, ModelMemory, MPS_RESERVE_TP1_GIB, MPS_RESERVE_TPN_GIB,
                               NodeSpec, Placement,
                               PlacementSolver, SolverBug,
                               Unproven,
                               TenantRequest, at_tp,
                               candidate_tp_degrees, capacity_steps, coresidency,
                               envelope_gib_for_kv_tokens,
                               fleet_baseline, gpu_memory_utilization,
                               kv_bytes_per_token,
                               kv_tokens_for_envelope, memory_floor_tp, min_envelope_gib,
                               weight_floor_tp)
from colocation.store.profile import CHECKED as _CHECKED, UNCHECKED as _UNCHECKED
from colocation.store import (Calibration, DOES_NOT_FIT,
                              DeviceEnvironment, Disagreement, Environment,
                              HWFACTS_SCHEMA_VERSION, HardwareFacts,
                              HardwareIndex, HardwareKey,
                              HardwareStore, InstanceShape,
                              MISSED, NONE_MEETS, NOT_PROFILED, TOO_NARROW,
                              NO_TARGET, Observation, ObservationStore, ProfileKey, ProfileStore,
                              RESOLVED, SoloProfile, SuppressedMetric, TpResolution, UNPROFILED,
                              UNUSABLE_DELIVERY, UNUSABLE_NO_LATENCY, UNUSABLE_SAMPLING,
                              UNUSABLE_SCHEDULE_LAG, MAX_SCHEDULE_LAG_MS,
                              ServingConfig, UNUSABLE_VERDICT, ValidityKey, VERDICT_DELIVERY,
                              topology_digest,
                              VERDICT_DEPLOY,
                              VERDICT_MEASURED, VERDICT_NO_STEADY_STATE, minimal_tp,
                              profile_id)
from colocation.store.hubfacts import (DTYPE_BYTES, HUBFACTS_SCHEMA_VERSION, HubFactsStore,
                                       HubManifest, HubUnavailable, _HEADER_PROBE_BYTES,
                                       _safetensors_header, _tensor_group)
from colocation.tests.hubfixture import fixture_hub, synthetic_manifest
from colocation.tiers import TIER_DERIVED, TIER_EXECUTED, TIER_MEASURED, at_least

FIXTURES = Path(__file__).parent / "fixtures"

# Every fleet load in this file goes through it, and it can only answer from `fixtures/hubfacts/`.
# A `models:` entry is an id now (impl §14.8, Fix 2), so loading any fleet -- a fixture or a shipped
# one -- means reading a checkpoint's config.json and safetensors headers, and this suite's first
# line says no network. See `tests/hubfixture.py` for why a manifest is a fixture and a profile row
# is not.
HUB = fixture_hub()
fails: list = []
ran: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    ran.append(name)
    if not cond:
        fails.append(name)


def load_fixture(fname: str) -> Observation:
    d = json.loads((FIXTURES / fname).read_text())
    return Observation.parse(d, str(FIXTURES / fname))


# The engine flags a tenant is served with, as the key and the row identity carry them. A
# fixture rather than a default anywhere in the code: `ServingConfig` has no defaults, so a test
# that forgets it fails loudly instead of comparing two keys built on invented flags.
SERVING = ServingConfig(max_num_seqs=64, max_num_batched_tokens=2048, max_model_len=4096,
                        dtype="bfloat16")


# The machine every key on this page is addressed to, as a node reports it (`store/environment.py`).
# Fixtures, and every one of them is an OBSERVATION in a real run -- there is no fleet file field for
# any of them -- so a test that wants a different environment says so here rather than editing a
# document. The topology is digested rather than written as a literal, so the fixture cannot drift
# away from what the digest function produces.
ADDRESSABLE_MIB = 46070
POWER_LIMIT_W = 350


def topo(gpus: int, link: str = "NODE") -> str:
    """The digest of a `gpus`-device node whose every pair speaks `link`. One spelling per count."""
    return topology_digest({(a, b): link for a in range(gpus) for b in range(gpus) if a != b})


TOPOLOGY = topo(8)


def env(**kw) -> Environment:
    base = dict(gpu_product="l40s", gpu_count=8, addressable_mib=ADDRESSABLE_MIB,
                power_limit_w=POWER_LIMIT_W, p2p_topology=TOPOLOGY, engine_image=DEFAULT_IMAGE)
    base.update(kw)
    return Environment(**base)


def key(**kw) -> ValidityKey:
    base = dict(env=env(),
                sequence_shapes={"m1": (512, 128, 0, 0)}, tensor_parallel={"m1": 1},
                offered_rates_hz={"m1": 2.0})
    base.update(kw)
    # Filled in from the shapes unless a test states it, because that is what the loader does --
    # every per-tenant field in a key comes off the same set of tenants. A test varying the fleet's
    # membership should not have to restate flags it is not varying.
    base.setdefault("serving", {n: SERVING for n in base["sequence_shapes"]})
    return ValidityKey(**base)


# The card every synthetic node on this page is of: what the driver reports for ONE device on the
# g6e node this stack was built against. There is exactly one such number now -- it is both the
# denominator of `gpu_memory_utilization` and the capacity a GPU's envelopes must fit inside
# (`NodeSpec.device_total_gib`) -- so there is one constant here, and the tests below that used to
# name a second, smaller "usable" figure name this instead. What that change buys is that a
# self-test can no longer accidentally pass by pricing a placement against a card the engine is not
# on. `S42_DENOMINATOR_GIB` further down is the exception and says why.
CARD_GIB = 44.99

# The store this suite resolves instance types against, written once here at import.
#
# There is no table in the source any more (`agent/hardware.py`), so a test that asks what a
# `g6e.48xlarge` is has to supply the same thing a real run does: a store somebody has already
# leased against. Filing it here rather than reaching for a constant is the point -- every
# `hardware_for` and `fleetfile.load` below now goes through the reader a real mode uses, and a
# regression in that reader fails the suite instead of being masked by a lookup that cannot fail.
#
# Two shapes, because two of the checks below turn on a lease landing on the wrong instance type,
# and one device law, because `CARD_GIB` is one card. The figures are a real card's, so that anything
# asserting on provenance sees a realistic one.
_HW_ROOT = tempfile.mkdtemp(prefix="colocation-selftest-hwfacts-")
atexit.register(shutil.rmtree, _HW_ROOT, ignore_errors=True)
HW_INDEX = HardwareIndex(_HW_ROOT)
for _it, _n in (("g6e.48xlarge", 8), ("g6e.12xlarge", 4)):
    # The device facts are on the shape because an offline load derives this store's own address from
    # them (`agent/hardware.Hardware.environment`): a shape without them is a machine no fleet can be
    # planned against. Both types present the same card, which is the case the address is designed for
    # -- one set of measurements, however the SKU is spelled.
    HW_INDEX.write_shape(InstanceShape(instance_type=_it, gpu_count=_n, gpu_product="l40s",
                                       addressable_mib=ADDRESSABLE_MIB,
                                       power_limit_w=POWER_LIMIT_W, p2p_topology=topo(_n),
                                       observed_at="2026-09-04T00:00:00Z", node=f"node-{_n}gpu"))
# A DEVICE environment, which is the hardware key: no GPU count and no metrics version, because a
# card's memory law does not expire when the box holds a different number of them.
_HW_KEY = HardwareKey(env=env().device())
_HW_SHEET = HardwareStore(_HW_ROOT, _HW_KEY)
_HW_SHEET.write_key()
_HW_SHEET.write(HardwareFacts(gpu_product="l40s", total_gib=CARD_GIB, total_source="slope",
                              total_uncertainty_gib=0.01, total_falsifiable=True,
                              driver="580.65.06", image=DEFAULT_IMAGE, gpu_node="node-8gpu"))

# --- validity key ------------------------------------------------------------
check("same key -> same hash", key().hash == key().hash)
check("hash is stable across field order",
      ValidityKey(env=env(),
                  sequence_shapes={"x": (1, 2, 0, 0), "y": (3, 4, 0, 0)},
                  tensor_parallel={"x": 1, "y": 2}, serving={"x": SERVING, "y": SERVING},
                  alpha=0.5, model_revisions={"x": "1", "y": "2"}).hash
      == ValidityKey(env=env(),
                     sequence_shapes={"y": (3, 4, 0, 0), "x": (1, 2, 0, 0)},
                     tensor_parallel={"y": 2, "x": 1}, serving={"y": SERVING, "x": SERVING},
                     alpha=0.5, model_revisions={"y": "2", "x": "1"}).hash)
# What replaced the driver string, and why: a driver version is a proxy that is wrong in both
# directions (`store/environment.py`). Each of these is a machine that measures differently and would
# have hashed identically before; the driver bump that changes none of them no longer discards a
# store, because it is not a field of the key at all.
check("a capped card is a different key", key().hash != key(env=env(power_limit_w=300)).hash)
check("  as is a card presenting less addressable memory",
      key().hash != key(env=env(addressable_mib=ADDRESSABLE_MIB - 1)).hash)
check("  as are the same cards over different links",
      key().hash != key(env=env(p2p_topology=topology_digest(
          {(a, b): "NV4" for a in range(8) for b in range(8) if a != b}))).hash)
check("  as is a different engine image", key().hash != key(env=env(engine_image="vllm:x")).hash)
check("  as is a different number of them", key().hash != key(env=env(gpu_count=4)).hash)
check("  as is a redefinition of what ITL means",
      key().hash != key(env=env(metrics_version="v2")).hash)
check("  and a different way of sharing them, which `compute_mode` cannot report",
      key().hash != key(env=env(sharing_mode="timeslice")).hash)
check("while the driver version is not in the key at all, so a bump keeps the whole store",
      "driver" not in json.dumps(key().canonical()), json.dumps(key().canonical()["environment"]))
check("a different ISL is a different key",
      key().hash != key(sequence_shapes={"m1": (1024, 128, 0, 0)}).hash)
# The image payload is in the key for the ISL argument's sharper form: at one ISL, a tenant sending
# images spends most of its prompt on visual tokens produced by a vision tower the text tenant never
# runs. Without this the two hashed identically -- which is what let a multimodal fleet be driven
# with pure text and filed as a multimodal measurement.
check("sending images is a different key from sending none",
      key().hash != key(sequence_shapes={"m1": (512, 128, 1, 262144)}).hash)
check("a different image count is a different key",
      key(sequence_shapes={"m1": (512, 128, 1, 262144)}).hash
      != key(sequence_shapes={"m1": (512, 128, 4, 262144)}).hash)
check("a different image resolution is a different key",
      key(sequence_shapes={"m1": (512, 128, 1, 262144)}).hash
      != key(sequence_shapes={"m1": (512, 128, 1, 1048576)}).hash)
# The one place absence is read as a value rather than refused, and the argument is that a key
# written before images existed governs runs a harness with no route for an image produced. Text
# then and text now are the same condition, so every correct measurement in the store keeps its
# directory instead of being orphaned to say nothing new.
check("a key.json with no image fields reads as a text run, and keeps its hash",
      "image_count" not in json.loads(key().key_json())["workload"]["sequence_shapes"]["m1"]
      and ValidityKey.from_dict(json.loads(key().key_json())).sequence_shapes["m1"]
      == (512, 128, 0, 0))
check("  and an image key round-trips to the same hash",
      ValidityKey.from_dict(json.loads(key(
          sequence_shapes={"m1": (512, 128, 4, 262144)}).key_json())).hash
      == key(sequence_shapes={"m1": (512, 128, 4, 262144)}).hash)
# Half a payload is not a payload: images of no size, or a size nothing is sent at.
for _bad in ((512, 128, 2, 0), (512, 128, 0, 262144), (512, 128, -1, 262144)):
    try:
        key(sequence_shapes={"m1": _bad})
        check(f"a key with a half-stated image payload is refused {_bad}", False)
    except ValueError:
        check(f"a key with a half-stated image payload is refused {_bad}", True)
# Two spellings of one condition would be two hashes for it, so the short form is refused outright
# rather than padded -- padding is what makes a second representation feel harmless.
try:
    key(sequence_shapes={"m1": (512, 128)})
    check("a two-element shape is refused rather than padded", False)
except ValueError:
    check("a two-element shape is refused rather than padded", True)
# The point of moving the shape into the key per tenant: two fleets can offer the same rates
# and the same *set* of shapes and still be different experiments, because which tenant got
# which shape decides what shares a GPU with what.
check("the same shapes assigned to different tenants are different keys",
      key(sequence_shapes={"m1": (512, 128, 0, 0), "m2": (128, 512, 0, 0)},
          tensor_parallel={"m1": 1, "m2": 1}, offered_rates_hz={"m1": 2.0, "m2": 2.0}).hash
      != key(sequence_shapes={"m1": (128, 512, 0, 0), "m2": (512, 128, 0, 0)},
             tensor_parallel={"m1": 1, "m2": 1}, offered_rates_hz={"m1": 2.0, "m2": 2.0}).hash)
# A tenant named in one map and not the other means the key does not describe one fleet.
try:
    key(offered_rates_hz={"m1": 2.0, "m2": 2.0})
    check("rates and shapes must name the same tenants", False)
except ValueError:
    check("rates and shapes must name the same tenants", True)
# TP is in the key for the same reason the shape is, and it is the same silent failure: the search
# never varies TP, so a fleet re-served at TP=4 is a different experiment whose records would
# otherwise land in the TP=2 directory and be handed to the reasoner as one condition.
check("a different TP is a different key",
      key().hash != key(tensor_parallel={"m1": 2}).hash)
check("the same TP degrees assigned to different tenants are different keys",
      key(sequence_shapes={"m1": (512, 128, 0, 0), "m2": (512, 128, 0, 0)},
          tensor_parallel={"m1": 2, "m2": 4},
          offered_rates_hz={"m1": 2.0, "m2": 2.0}).hash
      != key(sequence_shapes={"m1": (512, 128, 0, 0), "m2": (512, 128, 0, 0)},
             tensor_parallel={"m1": 4, "m2": 2},
             offered_rates_hz={"m1": 2.0, "m2": 2.0}).hash)
try:
    key(tensor_parallel={"m1": 1, "m2": 2})
    check("TP degrees and shapes must name the same tenants", False)
except ValueError:
    check("TP degrees and shapes must name the same tenants", True)
# Refused rather than lifted: nothing in an old key.json says what TP each tenant ran at, and
# defaulting to 1 would file a TP=4 run under a key claiming single-GPU measurements.
try:
    d = json.loads(key().key_json())
    del d["tensor_parallel"]
    ValidityKey.from_dict(d)
    check("a key.json without tensor_parallel is refused", False)
except ValueError:
    check("a key.json without tensor_parallel is refused", True)
# The engine flags are in the key for the same reason TP is, and this is the defect search-r2hk3z
# was refused by: five per-tenant settings reach vLLM's command line and none of them was keyed, so
# two runs differing only in a batch cap or a dtype hashed into one directory.
check("a different batch cap is a different key",
      key().hash != key(serving={"m1": dataclasses.replace(SERVING, max_num_seqs=96)}).hash)
check("a different dtype is a different key",
      key().hash != key(serving={"m1": dataclasses.replace(SERVING, dtype="float16")}).hash)
check("the same flags assigned to different tenants are different keys",
      key(sequence_shapes={"m1": (512, 128, 0, 0), "m2": (512, 128, 0, 0)},
          tensor_parallel={"m1": 1, "m2": 1}, offered_rates_hz={"m1": 2.0, "m2": 2.0},
          serving={"m1": SERVING,
                   "m2": dataclasses.replace(SERVING, max_num_seqs=96)}).hash
      != key(sequence_shapes={"m1": (512, 128, 0, 0), "m2": (512, 128, 0, 0)},
             tensor_parallel={"m1": 1, "m2": 1}, offered_rates_hz={"m1": 2.0, "m2": 2.0},
             serving={"m1": dataclasses.replace(SERVING, max_num_seqs=96),
                      "m2": SERVING}).hash)
try:
    key(serving={"m1": SERVING, "m2": SERVING})
    check("serving flags and shapes must name the same tenants", False)
except ValueError:
    check("serving flags and shapes must name the same tenants", True)
# Refused rather than lifted, and this is the one refusal that is load-bearing rather than tidy: the
# chart defaults that would be guessed are exactly the values that turned out to decide whether a
# target was reachable at all.
try:
    d = json.loads(key().key_json())
    del d["serving"]
    ValidityKey.from_dict(d)
    check("a key.json without the serving flags is refused", False)
except ValueError:
    check("a key.json without the serving flags is refused", True)
check("  and a key.json that has them round-trips to the same hash",
      ValidityKey.from_dict(json.loads(key().key_json())).hash == key().hash)
# The failure this guards is silent: alpha arrives from a sweep, so 0.35*2 produces
# 0.7000000000000001, which would hash differently from the 0.7 a human writes while meaning
# the same load scale -- and the agent would lose its entire history with no error.
check("alpha is quantized so float noise cannot orphan a history",
      key(alpha=0.35 * 2).hash == key(alpha=0.7).hash,
      f"{0.35 * 2!r} vs 0.7")
check("alpha=None is distinct from an alpha that was set",
      key(alpha=None).hash != key(alpha=0.7).hash)
check("a real alpha difference still separates keys",
      key(alpha=0.7).hash != key(alpha=0.8).hash)
check("key.json round-trips to the same hash",
      ValidityKey.from_dict(json.loads(key(alpha=0.7).key_json())).hash
      == key(alpha=0.7).hash)
# Found by using the key, not by reading it: without alpha there was nothing in the key naming
# how hard the fleet was driven, so the 2 Hz co-residency sweep and a 16 Hz one hashed into the
# same directory. Both records are individually valid, so nothing would have complained -- and
# retrieval would hand a 16 Hz measurement to an agent reasoning about a 2 Hz fleet as if it were
# evidence about it.
check("two load levels of the same fleet are different keys",
      key(offered_rates_hz={"m1": 2.0}).hash != key(offered_rates_hz={"m1": 16.0}).hash)
check("offered rates are quantized like alpha",
      key(offered_rates_hz={"m1": 0.1 * 3}).hash == key(offered_rates_hz={"m1": 0.3}).hash,
      f"{0.1 * 3!r} vs 0.3")
try:
    ValidityKey(env=env(),
                sequence_shapes={"m1": (1, 2, 0, 0)}, tensor_parallel={"m1": 1},
                serving={"m1": SERVING})
    check("a key with neither alpha nor rates is refused", False)
except ValueError:
    check("a key with neither alpha nor rates is refused", True)
# A record written when the shape was fleet-wide cannot be read as a per-tenant one: the pair
# in the old key says nothing about which tenant offered it.
try:
    d = json.loads(key().key_json())
    d["workload"] = {"isl_tokens": 512, "osl_tokens": 128, "offered_rates_hz": {"m1": 2.0}}
    ValidityKey.from_dict(d)
    check("a fleet-wide key.json is refused rather than reinterpreted", False)
except ValueError:
    check("a fleet-wide key.json is refused rather than reinterpreted", True)

# --- schema: the real records read ------------------------------------------
run6 = load_fixture("run6-no-sidecar.json")
run5 = load_fixture("run5-sidecar.json")
check("a real record parses", run6.verdict == VERDICT_MEASURED and len(run6.per_tenant) == 4,
      f"verdict={run6.verdict} tenants={sorted(run6.per_tenant)}")
# 60.008s, not 60.0: the window closes on a real clock and the record keeps the actual
# elapsed time rather than the configured minimum. Asserting equality here would be
# asserting that the harness rounds, which it deliberately does not.
check("window and steady state read back",
      60.0 <= run6.window_s < 60.5 and run6.steady_state_reached_s is not None,
      f"window={run6.window_s:.3f} ss={run6.steady_state_reached_s:.1f}")
check("per-GPU records are keyed by UUID and none throttled",
      len(run6.per_gpu) == 6 and not any(g.throttled for g in run6.per_gpu.values()),
      f"gpus={len(run6.per_gpu)}")
check("a real record is usable as evidence", run6.usable_as_evidence)
check("sub-sampled record is not flagged as phase-locked",
      not run6.sampling_suspect, f"subsamples={run6.subsamples_per_sample}")

# The record class that must never be cited as evidence: run 2 read occupancy 0.0 on every
# sample while DCGM saw the GPU 47% busy, because the sampler was phase-locked to the
# arrival schedule. It looked healthy. This is the check that quarantines it. Nothing fits a
# model to records any more, but impl §10 hands them to a reasoner, and a plausible wrong number
# is worse in a brief than a missing one.
phase_locked = json.loads((FIXTURES / "run6-no-sidecar.json").read_text())
phase_locked["sampling"]["subsamples_per_sample_mean"] = 1
po = Observation.parse(phase_locked)
check("a point-read record is flagged suspect", po.sampling_suspect)
check("  and is refused as evidence", not po.usable_as_evidence)
no_sampling = {k: v for k, v in phase_locked.items() if k != "sampling"}
check("a record with no sampling block at all is suspect, not assumed innocent",
      Observation.parse(no_sampling).sampling_suspect)

try:
    Observation.parse({**phase_locked, "verdict": "probably_fine"})
    check("unknown verdict is rejected, not coerced", False)
except ValueError:
    check("unknown verdict is rejected, not coerced", True)

# --- schema: suppressed latency must raise, never default -------------------
suppressed = json.loads((FIXTURES / "run6-no-sidecar.json").read_text())
suppressed["verdict"] = VERDICT_DELIVERY
m1 = suppressed["per_tenant"]["m1"]
for k in ("itl_mean_ms", "ttft_mean_ms", "ttft_p95_ms", "itl_ci_halfwidth_ms"):
    m1.pop(k, None)
m1["latency_suppressed"] = "delivered fraction below threshold"
m1["delivered_fraction"] = 0.5
so = Observation.parse(suppressed)
t = so.per_tenant["m1"]
check("suppressed latency is reported as unavailable", not t.latency_available)
try:
    t.itl_mean_ms
    check("asking for suppressed latency raises rather than returning None", False)
except SuppressedMetric:
    check("asking for suppressed latency raises rather than returning None", True)
check("a failed delivery gate is visible as such", not t.delivery_passed,
      f"deliv={t.delivered_fraction} thr={t.delivered_fraction_threshold}")
check("delivery gate passes on the real record", run6.per_tenant["m1"].delivery_passed)

# The unexplained divergence stays visible as data rather than being smoothed away.
check("chunk/token divergence is surfaced from the real records",
      run5.per_tenant["m1"].chunk_divergence == 7
      and run6.per_tenant["m1"].chunk_divergence == 3,
      f"run5={run5.per_tenant['m1'].chunk_divergence} "
      f"run6={run6.per_tenant['m1'].chunk_divergence}")

# --- store round-trip --------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    k = key(alpha=0.7)
    st = ObservationStore(tmp, k)
    st.write_key()
    check("key.json written and re-readable", st.read_key().hash == k.hash)
    check("re-writing an identical key is a no-op, not an error",
          st.write_key().exists())

    check("no calibration -> None, which is the agent's cue to run it",
          st.read_calibration() is None)
    # Keyed by co-residency bucket, not by a predicted regime. The regime was a label the deleted
    # model attached to a placement before measuring it, so the resolution of a measurement was
    # filed under a forecast; the bucket is a fact about what was deployed.
    st.write_calibration(Calibration(
        between_run_itl_halfwidth_ms={"coresident=0": 0.21, "coresident=3": 0.86},
        solo_saturation_rate_hz={"Qwen/Qwen3-0.6B-tp1": 34.0}, repeats=5))
    cal = st.read_calibration()
    check("calibration round-trips with a half-width per co-residency bucket",
          cal.threshold_ms("coresident=3") == 0.86 and cal.repeats == 5)
    check("a bucket that was never calibrated returns None, not another bucket's number",
          cal.threshold_ms("coresident=1") is None)
    check("  and a caller with no bucket to look up gets the TIGHTEST half-width, since that is "
          "the one that demands the longest measurement window",
          cal.tightest_ms == 0.21, str(cal.tightest_ms))

    rec = json.loads((FIXTURES / "run6-no-sidecar.json").read_text())
    rec["validity_key_hash"] = k.hash
    p1 = st.append_observation(rec)
    p2 = st.append_observation(rec)
    check("a repeat is a new observation, never an overwrite", p1 != p2 and p1.exists(),
          f"{p1.name} then {p2.name}")
    check("both records load back", len(st.observations()) == 2)

    # Misfiling would make the agent compare across hardware without noticing.
    try:
        st.append_observation({**rec, "validity_key_hash": "someotherkey"})
        check("a record from another validity key is refused", False)
    except ValueError:
        check("a record from another validity key is refused", True)

    (st.observations_dir / "truncated.json").write_text("{not json")
    report = st.load()
    check("an unreadable record is counted and named, never silently dropped",
          len(report.observations) == 2 and len(report.unreadable) == 1 and not report.ok,
          str(report.unreadable[0][1])[:60])

    st.write_feasibility("Qwen/Qwen3-0.6B", 1, {"device_total_gib": CARD_GIB})
    f = st.read_feasibility("Qwen/Qwen3-0.6B", 1)
    check("feasibility facts round-trip and are marked provisional",
          f["device_total_gib"] == CARD_GIB and f["provisional"] is True)
    check("a model id with a slash does not become a subdirectory",
          st.feasibility_path("Qwen/Qwen3-0.6B", 1).parent == st.feasibility_dir)

# --- co-residency: what a placement is, never what it will do ----------------
# `coresidency` is all that was rescued from the deleted interference package, and it lives in
# `solver/topology.py` now. It is descriptive: it counts how many other tenants share each of a
# tenant's GPUs. Nothing ranks candidates by it -- if anything ever does, that is the predictor
# coming back in through a side door.
#
# Runs 5 and 6 placed four tenants on six GPUs with six shards and no GPU shared, so the
# co-residency the hardware saw was ZERO. A node-wide count would have filed these under
# "3 neighbours" and every latency in the store would carry contention that never happened.
placement6 = {t["name"]: tuple(t["gpu_uuids"])
              for t in run6.configuration["tenants"]}
check("the real records are recognized as a disjoint placement, not a contended one",
      coresidency(placement6) == {"m1": 0, "r1": 0, "t2": 0, "r2": 0},
      str(coresidency(placement6)))
check("sharing a GPU is counted, and a TP group takes the max over its shards",
      coresidency({"a": ("g0",), "b": ("g0",), "c": ("g1", "g0")}) == {"a": 2, "b": 2, "c": 2})
check("a TP group is not credited for its uncontended shard",
      coresidency({"a": ("g0", "g1"), "b": ("g1",)}) == {"a": 1, "b": 1})
# The bucket the noise floor is filed under, read off a record's own UUIDs. It has to work on a
# record written before the agent existed, which is why it derives the degree rather than reading
# a provenance key: runs 5 and 6 have no agent provenance at all.
check("a record buckets itself by the co-residency it actually had, needing no provenance",
      bucket(run6) == "coresident=0" and bucket(run5) == "coresident=0",
      f"{bucket(run6)} / {bucket(run5)}")

# --- solver: the envelope arithmetic against every measured row -------------
# Qwen3-0.6B: 28 layers, 8 KV heads, head_dim 128, bf16. Not a fit -- arithmetic on the
# model's own config, which is why it lands on the measured block counts exactly.
QWEN_KV_BYTES = kv_bytes_per_token(num_layers=28, num_kv_heads=8, head_dim=128)
check("KV bytes per token is arithmetic on the model config, not a measurement",
      QWEN_KV_BYTES == 112 * 1024, f"{QWEN_KV_BYTES} B = {QWEN_KV_BYTES // 1024} KiB")

QWEN = ModelMemory(model_id="Qwen/Qwen3-0.6B",
                   weight_gib=1.110,                       # 596M params x 2 bytes (bf16)
                   kv_bytes_per_token_tp1=QWEN_KV_BYTES,
                   activation_gib={1: 0.416, 2: 0.2845})   # impl §4.2 residuals

# Every row in impl §4.2's table. The envelope was varied 2.5x with everything else fixed,
# so this checks the *shape* of the map, not one operating point.
MEASURED_ROWS = [(1, 8, 48032), (1, 20, 154432), (2, 8, 99776), (2, 20, 313408)]
worst_kv = 0.0
for tp, envelope_gib, measured_tokens in MEASURED_ROWS:
    got = kv_tokens_for_envelope(envelope_gib, tp, QWEN)
    err = abs(got / measured_tokens - 1)
    worst_kv = max(worst_kv, err)
    check(f"  envelope {envelope_gib} GiB at tp{tp} -> KV tokens within 0.5%", err < 0.005,
          f"predicted {got} vs measured {measured_tokens} ({err * 100:.2f}%)")
check("the envelope -> KV-tokens map holds across a 2.5x envelope range and both TP degrees",
      worst_kv < 0.005, f"worst {worst_kv * 100:.2f}%")

# The util the chart consumes. Cross-checked against what impl §4.2's measured extra-memory column
# implies was actually used: resident 7435 MiB - 0.602 GiB extra = 6.659 GiB = util x D. What is
# being tested is the FORMULA, so D is stated here as that run's own datum -- the denominator the
# §4.2 rows were computed against -- and not read from a table. There is no longer a table to read it
# from, and the subject of this whole phase is that a denominator has to travel with the measurement
# that used it rather than sit in a constant anyone can reach.
S42_DENOMINATOR_GIB = 44.39
util_tp1_8 = gpu_memory_utilization(8, 1.0, S42_DENOMINATOR_GIB)
check("gpu_memory_utilization reproduces the util the measured run implies",
      abs(util_tp1_8 * S42_DENOMINATOR_GIB - 6.659) < 0.02, f"util={util_tp1_8:.4f}")
try:
    gpu_memory_utilization(0.8, 1.0, S42_DENOMINATOR_GIB)
    check("an envelope inside the MPS reserve is refused, not scaled down", False)
except ValueError:
    check("an envelope inside the MPS reserve is refused, not scaled down", True)
# The denominator is the thing this phase made un-defaultable. A default here would be a claim about
# a card living in the one function that turns an envelope into a utilisation, which is the number
# every KV pool on the node is sized by.
try:
    gpu_memory_utilization(8, 1.0)                                 # type: ignore[call-arg]
    check("a utilisation cannot be computed without saying which card it is of", False)
except TypeError as _exc:
    check("a utilisation cannot be computed without saying which card it is of",
          "device_total_gib" in str(_exc), str(_exc))

check("mps_reserve falls back to the safe default per TP degree, and says so",
      QWEN.reserve_gib(1) == (1.0, "default(tp1)")
      and QWEN.reserve_gib(4) == (1.5, "default(tpN)"), str(QWEN.reserve_gib(4)))
check("a measured reserve wins over the default when one exists",
      ModelMemory("m", 1.0, QWEN_KV_BYTES, measured_reserve_gib={1: 0.611}).reserve_gib(1)
      == (0.611, "measured@tp1"))

# --- the tier vocabulary, and what a tier is allowed to do (colocation/tiers.py) ---
check("tiers order weakest-first, so 'at least executed' does not hard-code an order",
      at_least(TIER_MEASURED, TIER_EXECUTED) and at_least(TIER_EXECUTED, TIER_EXECUTED)
      and not at_least(TIER_DERIVED, TIER_EXECUTED))
try:
    at_least("guessed", TIER_EXECUTED)
    check("an unknown tier raises rather than reading as 'not strong enough'", False)
except ValueError:
    check("an unknown tier raises rather than reading as 'not strong enough'", True)

# --- solver: the run-6 fleet, and what "best" means without a prediction ------
GPUS = tuple(f"GPU-{i}" for i in range(8))
NODE = NodeSpec(gpu_ids=GPUS, device_total_gib=CARD_GIB)
MIN_KV = 640 * 16          # 16 concurrent requests at ISL 512 + OSL 128


def req(name: str, tp: int, rate: float = 2.0, mem: ModelMemory = QWEN,
        isl: int = 512, osl: int = 128, min_kv: int = MIN_KV, floor: int = 0,
        floor_src: str = "no ITL ceiling stated") -> TenantRequest:
    """A tenant to place. `floor` is impl §9's SLO-derived KV floor, in tokens.

    Two floors, deliberately separate: `min_kv` is the pool below which the tenant cannot serve
    its own shape at all, and `floor` is what its stated ITL ceiling implies at its stated rate.
    The larger binds. `floor=0` is not "no floor" by accident -- it is how a tenant that stated no
    ceiling is encoded, and the solver must enforce nothing on its behalf rather than invent one.
    """
    return TenantRequest(name=name, model_id="Qwen/Qwen3-0.6B", tp=tp, rate_hz=rate,
                         isl_tokens=isl, osl_tokens=osl, mem=mem, min_kv_tokens=min_kv,
                         kv_floor_tokens=floor, floor_source=floor_src)


fleet6 = [req("m1", 1), req("r1", 1), req("t2", 2), req("r2", 2)]
solver = PlacementSolver(NODE, fleet6)
feasible = solver.solve(cap=3)
best = feasible[0]
check("the run-6 fleet solves and returns a feasible SET, not a ranking of forecasts",
      isinstance(feasible, FeasibleSet) and len(feasible) == 3,
      f"{len(feasible)} candidates, best on {best.gpus_used} GPU(s), tightest margin "
      f"{best.min_margin:.2f}x at {best.tightest_tenant}")
check("every TP group has its shards on distinct GPUs",
      all(len(set(best.placement.assignment[r.name])) == r.tp for r in fleet6),
      str({n: len(v) for n, v in best.placement.assignment.items()}))
loads = best.placement.gpu_load_mib()
check("no GPU is promised more than its measured usable capacity",
      all(mib <= NODE.capacity_mib for mib in loads.values()),
      f"max {max(loads.values()) / 1024:.2f} GiB of {NODE.device_total_gib} GiB")
check("candidates are distinct placements, not relabellings of one",
      len({tuple(sorted(c.placement.assignment.items())) for c in feasible}) == len(feasible))
check("the reserve is computed from the placement, per TP degree",
      best.placement.mps_reserve_gib["m1"] == 1.0
      and best.placement.mps_reserve_gib["t2"] == 1.5,
      str(best.placement.mps_reserve_gib))

# **This is the change of behaviour Phase 3 is.** Six shards on eight GPUs used to come back
# disjoint, because the ranking was driven by predicted occupancy and spreading always predicted
# best. Nothing predicts now, and impl §9's first objective is packing density: the best candidate
# puts all four tenants on TWO GPUs, and the disjoint arrangement is not in the top of the set at
# all. Density is the thing being searched for; whether it costs latency is what a run measures.
check("the densest candidate wins, and it is a shared placement rather than a disjoint one",
      best.gpus_used == 2 and min(best.coresidency.values()) >= 1,
      f"{best.gpus_used} GPUs, coresidency {best.coresidency}")
check("  and a candidate on more GPUs ranks below it even with a bigger KV margin",
      feasible[2].gpus_used == 3 and feasible[2].min_margin > best.min_margin,
      f"#3: {feasible[2].gpus_used} GPUs at {feasible[2].min_margin:.2f}x vs "
      f"#1: {best.gpus_used} GPUs at {best.min_margin:.2f}x")
check("  which is exactly what rank_key says, GPUs before margin",
      [c.rank_key for c in feasible] == sorted(c.rank_key for c in feasible)
      and best.rank_key == (2, -best.min_margin))
# The margin is arithmetic on the candidate's own numbers, not a score handed down by a model.
check("each margin is the KV the tenant gets over the floor it was solved against",
      all(abs(best.margin[n] - best.placement.kv_tokens[n] / best.floor_tokens[n]) < 1e-9
          for n in best.margin),
      str({n: round(v, 2) for n, v in best.margin.items()}))
check("  and min_margin/tightest_tenant name the tenant with the least room, not the slowest one",
      best.min_margin == min(best.margin.values())
      and best.margin[best.tightest_tenant] == best.min_margin)
check("the floor detail says which floor bound and where it came from, for the record",
      all("minimum viable pool" in best.floor_detail[n] for n in best.floor_detail),
      best.floor_detail["m1"])
check("the solver is deterministic across runs",
      PlacementSolver(NODE, fleet6).solve(cap=3)[0].placement == best.placement)

# --- solver: complete set vs truncated one (impl §9) -------------------------
# The distinction the flag exists for: an agent told the set is complete may conclude the fleet is
# infeasible, one told it was truncated may not. So `exhausted` is true only when CP-SAT proved
# there is no further distinct assignment -- never when the cap or the clock ran out.
solo = PlacementSolver(NodeSpec(gpu_ids=("GPU-0",), device_total_gib=CARD_GIB), [req("solo", 1)])
one_only = solo.solve(cap=3)
check("one tenant on one GPU has exactly one assignment, and the set says so is all of them",
      len(one_only) == 1 and one_only.exhausted and not one_only.truncated,
      one_only.stopped_because)
capped = PlacementSolver(NODE, fleet6).solve(cap=1)
check("a set cut off by the cap is TRUNCATED, and never reports itself as complete",
      len(capped) == 1 and capped.truncated and not capped.exhausted,
      capped.stopped_because)
check("  and it says the cap is why, so a caller is not left to infer it",
      "cap of 1" in capped.stopped_because, capped.stopped_because)

# --- solver: the KV floor is a constraint, not a preference ------------------
# impl §9's substitute for predicting a latency. A placement below a floor is not a worse
# candidate; it is an uncertified one, so the floor is a hard constraint and an unmeetable floor
# is an answer about the fleet rather than a low score.
tight = PlacementSolver(NODE, [req("m1", 1), req("hungry", 1, min_kv=0, floor=120_000,
                                                 floor_src="ITL 12 ms at 6 Hz, ISL 512")])
tight_best = tight.solve(cap=1)[0]
check("a tenant's SLO floor is met, not traded away for a neighbour's comfort",
      tight_best.placement.kv_tokens["hungry"] >= 120_000,
      f"{tight_best.placement.kv_tokens['hungry']} tokens vs floor 120000")
check("  and the stated ceiling is what the floor is attributed to in the record",
      "ITL 12 ms at 6 Hz" in tight_best.floor_detail["hungry"],
      tight_best.floor_detail["hungry"])
try:
    PlacementSolver(NODE, [req("greedy", 1, min_kv=0, floor=5_000_000,
                               floor_src="ITL 5 ms at 64 Hz")]).solve(cap=1)
    check("a floor no single GPU can hold is refused with a local reason", False)
except Infeasible as exc:
    # Named because it is actionable: raise TP, relax the target, or lower the rate. "CP-SAT found
    # nothing" would be true and useless.
    check("a floor no single GPU can hold is refused with a local reason",
          "greedy" in str(exc) and "564" in str(exc), str(exc)[:90])
    # And since impl §14.2, WHICH refusal it is. This tenant's weights are 1.11 GiB on a 44.99 GiB
    # card; what does not fit is a 564 GiB KV pool derived from a page size and a latency target.
    # That is not a fact about the fleet, so it may not be reported as one.
    check("  and a floor that overflows on the KV pool rather than the weights is UNPROVEN, not "
          "infeasible: the terms that overflowed are the computed ones",
          isinstance(exc, Unproven) and "weights are only 1.11 GiB" in str(exc)
          and "profile" in str(exc), str(exc)[-120:])

# --- solver: the envelope really is a decision variable ---------------------
# One GPU, two tenants, so the capacity constraint binds and the split is the only freedom left.
# What decides the split is the second objective -- maximize the smallest margin over the floors --
# so the tenant with the larger floor is given the larger envelope. Note what is NOT being
# equalized: not the envelopes, and not a predicted occupancy either, since none is computed.
one_gpu = NodeSpec(gpu_ids=("GPU-0",), device_total_gib=CARD_GIB)
pair = [req("modest", 1), req("hungry", 1, min_kv=0, floor=120_000,
                              floor_src="ITL 12 ms at 6 Hz, ISL 512")]
shared = PlacementSolver(one_gpu, pair).solve(cap=1)[0]
check("a forced-share placement is found and both tenants land on the one GPU",
      shared.coresidency == {"modest": 1, "hungry": 1}, str(shared.coresidency))
check("the tenant with the larger floor is given the larger envelope",
      shared.placement.envelope_gib["hungry"] > shared.placement.envelope_gib["modest"],
      f"hungry {shared.placement.envelope_gib['hungry']} GiB vs "
      f"modest {shared.placement.envelope_gib['modest']} GiB")
check("  and the split leaves the two margins close, since the smallest one is what is maximized",
      abs(shared.margin["hungry"] / shared.margin["modest"] - 1) < 0.15,
      f"{shared.margin['hungry']:.2f}x vs {shared.margin['modest']:.2f}x")
# The third objective. Unused envelope buys nothing, so what the capacity constraint leaves is
# handed out: every whole 512 MiB step the card holds, with the remainder under one step. Stated as
# `capacity_steps` rather than as a number of GiB so that this asserts the *behaviour* -- the last
# objective spends what is left -- on whatever card the constant above names, instead of pinning one
# card's step count.
# At the solver's OWN step, not the agent's: `PlacementSolver` was constructed with the default.
_FULL_STEPS = capacity_steps(one_gpu.device_total_gib, ENVELOPE_STEP_MIB_DEFAULT)
check("the last objective spends the capacity down to the final envelope step",
      sum(shared.placement.envelope_gib.values()) == _FULL_STEPS * ENVELOPE_STEP_MIB_DEFAULT / 1024,
      f"{sum(shared.placement.envelope_gib.values())} GiB = {_FULL_STEPS} steps of "
      f"{one_gpu.device_total_gib} GiB")

# --- baseline: the disjoint number, and the two refutations that are free (impl §0) ---
# A 30B checkpoint, so the memory floor is a real constraint rather than a formality: 56.873 GiB
# does not fit one GPU of this card at any envelope, so this model's TP degree is set by its weights
# and not by anybody's latency target. That is §0's "fleet worth searching" shape, and it is the
# only kind of tenant that has SM headroom and stranded memory at the same time.
BIG = ModelMemory(model_id="Qwen/Qwen3-30B-A3B-Base", weight_gib=56.873,
                  kv_bytes_per_token_tp1=kv_bytes_per_token(num_layers=48, num_kv_heads=4,
                                                            head_dim=128))
big_req = TenantRequest(name="reason", model_id=BIG.model_id, tp=4, rate_hz=1.0,
                        isl_tokens=1024, osl_tokens=1024, mem=BIG, min_kv_tokens=16384,
                        kv_floor_tokens=61440, floor_source="ITL 30 ms at 1 Hz")
big_base = memory_floor_tp(big_req, device_total_gib=CARD_GIB, envelope_step_mib=AGENT_ENVELOPE_STEP_MIB,
                           gpu_count=8)
check("the memory floor is the smallest TP degree whose shard fits one GPU",
      big_base.floor_tp == 2, f"floor tp{big_base.floor_tp}: {big_base.why}")
check("  and the degree below it is recorded with what it would have needed, so the floor is "
      "checkable rather than asserted",
      big_base.ruled_out and big_base.ruled_out[-1][0] == 1
      and big_base.ruled_out[-1][1] > CARD_GIB, str(big_base.ruled_out))
# The §0 mistake, in the form it took: TP=4 stated on a model whose weights need 2. The extra
# shards are GPUs the BASELINE pays for, so a search scored against them credits co-location with
# work that re-sharding did.
check("a stated degree above the floor is flagged as inflated, with the GPUs it costs",
      big_base.above_floor and big_base.excess_gpus == 2,
      f"stated tp{big_base.resolved_tp} vs floor tp{big_base.floor_tp}")
check("  and a tenant AT its floor is not flagged, since the floor is where TP stops being a "
      "choice",
      not memory_floor_tp(at_tp(big_req, 2), device_total_gib=CARD_GIB,
                          envelope_step_mib=AGENT_ENVELOPE_STEP_MIB, gpu_count=8).above_floor)
# The floor is a LOWER BOUND on minimal TP and the tests must not read as though it were the value:
# a degree that fits may still miss the target, which only a solo measurement settles.
check("the floor is memory-only, so it moves with the KV floor and not with any latency target",
      memory_floor_tp(dataclasses.replace(big_req, kv_floor_tokens=1_000_000), device_total_gib=CARD_GIB,
                      envelope_step_mib=AGENT_ENVELOPE_STEP_MIB, gpu_count=8).floor_tp == 4,
      "the same weights need tp4 once the pool the tenant must hold is 1M tokens")
check("a tenant that fits at no degree reports no floor rather than raising",
      memory_floor_tp(dataclasses.replace(big_req, kv_floor_tokens=4_000_000),
                      device_total_gib=CARD_GIB, envelope_step_mib=AGENT_ENVELOPE_STEP_MIB,
                      gpu_count=8).floor_tp is None,
      "58.22 GiB/shard even at tp8, the widest this node offers")

# --- impl §14.2: the one floor a refusal may rest on ------------------------------------------
# Everything above is the floor a walk should START at -- full envelope, every term, including the
# KV page size that was 3.27x too large on the 9B. What follows is the floor a walk may REFUSE at.
# The two are different numbers and conflating them is what let `plan` rule out tp1 for a model
# nobody had ever booted at tp1.
big_wf = weight_floor_tp(BIG, device_total_gib=CARD_GIB, gpu_count=8)
check("the weight floor is the narrowest degree the weights alone permit",
      big_wf.floor_tp == 2, f"weight floor tp{big_wf.floor_tp}: {big_wf.why}")
check("  and it names what it ruled out in weights alone, so the refusal is checkable",
      big_wf.ruled_out == ((1, 56.873),), str(big_wf.ruled_out))
check("  so on THIS model tp1 may be eliminated without a round: 56.873 GiB of weights do not fit "
      "on a 44.99 GiB card however generous every other term is",
      big_wf.floor_tp == big_base.floor_tp == 2,
      "weight floor and memory floor coincide here, which is why the live case costs no GPU time")
check("the weight bound is never above the full envelope, so the weight floor is never narrower "
      "than the memory floor is wide",
      all(BIG.weight_lower_bound_gib(k) <= min_envelope_gib(big_req, k)
          for k in candidate_tp_degrees(8)),
      "the bound is a subset of the envelope's terms, by construction")

# The case §14.2 exists for: modest weights, a KV floor big enough to push the memory floor up.
# The band between the two floors is degrees arithmetic doubts and only a boot may settle.
band_req = dataclasses.replace(big_req, mem=QWEN, kv_floor_tokens=1_000_000)
band_mem = memory_floor_tp(band_req, device_total_gib=CARD_GIB,
                           envelope_step_mib=AGENT_ENVELOPE_STEP_MIB, gpu_count=8)
band_wf = weight_floor_tp(QWEN, device_total_gib=CARD_GIB, gpu_count=8)
check("a tenant whose memory floor is driven by its KV pool has a weight floor below it, and the "
      "gap is the degrees no arithmetic may eliminate",
      band_mem.floor_tp == 4 and band_wf.floor_tp == 1 and band_wf.ruled_out == (),
      f"memory floor tp{band_mem.floor_tp}, weight floor tp{band_wf.floor_tp}: tp1 and tp2 are "
      f"doubted by arithmetic and settled only by a boot")
check("  and a weight floor at the narrowest candidate says so in words rather than implying it",
      "eliminate" in band_wf.why and "nothing" in band_wf.why, band_wf.why)

# The tied `lm_head`, which is the only known way `weight_gib` can over-read the device -- and the
# reason the bound needs a subtraction at all. Checked against a booted engine: the fleet file's
# 0.6B block records 1,503,300,328 checkpoint bytes = 1.400 GiB gross, an allowance of
# 151936 x 1024 x 2 bytes, and 1.12 GiB measured by `onboard`.
TIED_ALLOWANCE_GIB = 151936 * 1024 * 2 / 2 ** 30
tied = ModelMemory(model_id="gross/Qwen3-0.6B", weight_gib=1503300328 / 2 ** 30,
                   kv_bytes_per_token_tp1=QWEN_KV_BYTES,
                   tied_embedding_gib=TIED_ALLOWANCE_GIB)
check("a gross checkpoint figure minus the tied-head allowance lands UNDER the measured resident "
      "figure, which is the direction a lower bound has to err in",
      1.10 < tied.weight_lower_bound_gib(1) < 1.12,
      f"{tied.weight_lower_bound_gib(1):.4f} GiB bound vs 1.12 GiB measured on the 0.6B, "
      f"0.9% under")
check("  and the allowance defaults to zero, so a `weight_gib` that is already resident is never "
      "subtracted twice",
      QWEN.tied_embedding_gib == 0.0
      and QWEN.weight_lower_bound_gib(1) == QWEN.weight_gib, str(QWEN.tied_embedding_gib))

# Optimistic in every direction but the weights: a shard that clears the bound may still fail to
# come up, and that is the point -- clearing it earns a measurement, not a verdict.
check("the bound excludes the reserve, the activation and the KV pool, so it is strictly below "
      "anything an engine would actually need",
      QWEN.weight_lower_bound_gib(1) < QWEN.fixed_gib(1) < min_envelope_gib(band_req, 1),
      f"{QWEN.weight_lower_bound_gib(1):.3f} < {QWEN.fixed_gib(1):.3f} < "
      f"{min_envelope_gib(band_req, 1):.3f} GiB")
check("a model whose weights do not fit at the widest candidate reports no weight floor, and that "
      "is the one refusal that is about the node",
      weight_floor_tp(ModelMemory(model_id="huge", weight_gib=400.0,
                                  kv_bytes_per_token_tp1=QWEN_KV_BYTES),
                      device_total_gib=CARD_GIB, gpu_count=8).floor_tp is None,
      "50.0 GiB/shard of weights at tp8 on a 44.99 GiB card")
check("the bound is compared as GiB and not in solver steps, so a shard that exactly fills the "
      "card is not refused by rounding",
      weight_floor_tp(ModelMemory(model_id="exact", weight_gib=CARD_GIB,
                                  kv_bytes_per_token_tp1=QWEN_KV_BYTES),
                      device_total_gib=CARD_GIB, gpu_count=8).floor_tp == 1,
      "rounding a bound up would make a refusal stricter than the hardware")
try:
    weight_floor_tp(QWEN, device_total_gib=CARD_GIB)
    check("the weight floor needs a candidate list or a GPU count and refuses to invent one", False)
except ValueError as exc:
    check("the weight floor needs a candidate list or a GPU count and refuses to invent one",
          True, str(exc))

# The fleet-level view. `sum(tp)` is the baseline; `sum(floor tp)` is what it would be with no
# co-location at all, and the gap is the part of any reported saving that re-sharding would explain.
base8 = fleet_baseline(NODE, [req("m1", 1), req("r1", 1), big_req],
                       envelope_step_mib=AGENT_ENVELOPE_STEP_MIB)
check("the baseline is sum(tp) GPUs, counted from the fleet and not from a placement",
      base8.resolved_gpus == 6 and base8.floor_gpus == 4,
      f"stated {base8.resolved_gpus}, at the memory floor {base8.floor_gpus}")
check("  and the inflated tenants are named, because that gap is not co-location's to claim",
      base8.above_floor == ("reason",), str(base8.above_floor))
check("a fleet whose shards outnumber the node's GPUs cannot be served disjointly at all",
      not fleet_baseline(NodeSpec(gpu_ids=GPUS[:4], device_total_gib=CARD_GIB),
                         [req("m1", 1), req("r1", 1), big_req],
                         envelope_step_mib=AGENT_ENVELOPE_STEP_MIB).fits_disjoint,
      "6 shards on 4 GPUs")
# Refutation, not a preference: overlap needs one GPU to hold shards of two different tenants, and
# the solver may raise an envelope above a tenant's floor but never below it. So a fleet with no
# shareable pair has no co-resident placement at ANY envelope -- which `search` refuses to spend a
# node-hour confirming.
check("a fleet with a shareable pair is reported as searchable",
      base8.can_coreside and len(base8.pairs) == 3, f"{len(base8.shareable)} of 3 pairs fit")
# Two shards of the 30B at its own memory floor: 35.45 GiB each, so either fits alone on one GPU of
# this card and no two fit together. The fleet is perfectly feasible -- it just cannot co-locate.
hogs = [dataclasses.replace(big_req, name=n, tp=2) for n in ("a", "b")]
no_share = fleet_baseline(NODE, hogs, envelope_step_mib=AGENT_ENVELOPE_STEP_MIB)
check("a fleet where no two shards fit on one GPU is refuted before any node is leased",
      not no_share.can_coreside and no_share.tightest_pair.deficit_gib > 0,
      f"closest pair needs {no_share.tightest_pair.total_gib:.2f} GiB of {CARD_GIB}")
# The refutation has to agree with the solver, or it is worse than no refutation at all.
check("  and the solver, asked exhaustively, finds only disjoint placements for it",
      all(max(c.coresidency.values()) == 0
          for c in PlacementSolver(NODE, hogs).solve(cap=8)),
      "every candidate has coresidency 0")
check("  while the shareable fleet's densest candidate does share a GPU",
      max(PlacementSolver(NODE, [req("m1", 1), req("r1", 1), big_req]
                          ).solve(cap=1)[0].coresidency.values()) >= 1)
check("the whole baseline serializes for the run summary",
      json.loads(json.dumps(base8.as_dict()))["baseline_gpus_at_memory_floor"] == 4)

# --- solver: agent-supplied cuts, scoped to the run -------------------------
# impl §9 lets the agent say "not these two on one GPU" after reading what a run did. Nothing here
# writes the cut anywhere: it is a belief formed from retrieved runs, and a mistaken one must cost
# iterations rather than correctness.
free = PlacementSolver(NODE, [req("a", 1), req("b", 1)]).solve(cap=1)[0]
check("two tenants pack onto one GPU when nothing forbids it",
      free.gpus_used == 1, str(free.placement.assignment))
cut = PlacementSolver(NODE, [req("a", 1), req("b", 1)]).solve(cap=1, avoid=[("a", "b")])[0]
check("  and an avoid cut separates them, at the cost of a second GPU",
      cut.gpus_used == 2 and set(cut.coresidency.values()) == {0},
      str(cut.placement.assignment))
for bad, why in ((("a", "ghost"), "a cut naming a tenant the fleet does not have is refused"),
                 (("a", "a"), "a cut naming one tenant twice is refused as vacuous")):
    try:
        PlacementSolver(NODE, [req("a", 1), req("b", 1)]).solve(cap=1, avoid=[bad])
        check(why, False)
    except ValueError:
        # Silently dropping it is the failure worth preventing: the agent would believe it had
        # constrained a search that then returned the placement it ruled out.
        check(why, True)

# --- solver: the invariant, from both sides ---------------------------------
# impl §9 calls a capacity violation a solver bug, not a runtime hazard, because §4.2.2
# measured the non-crashing case: two 28 GiB envelopes on one GPU of this card both reached
# Ready and served correctly, with 24.15 GiB and 14.23 GiB of KV cache. This is that.
oversub = Placement(
    assignment={"a": ("GPU-0",), "b": ("GPU-0",)},
    envelope_gib={"a": 28.0, "b": 28.0}, kv_tokens={"a": MIN_KV, "b": MIN_KV},
    util={"a": 0.5779, "b": 0.5779}, mps_reserve_gib={"a": 1.0, "b": 1.0},
    reserve_source={"a": "default(tp1)", "b": "default(tp1)"})
try:
    oversub.verify(one_gpu, [req("a", 1), req("b", 1)])
    check("an over-subscribed placement is caught by verify(), not left to the runtime", False)
except SolverBug:
    check("an over-subscribed placement is caught by verify(), not left to the runtime", True)

# The other side of the invariant, and the one prediction-free solving adds: a placement that
# fits the GPU but starves a tenant below its floor. It would deploy, reach Ready and serve --
# and the run would be filed as evidence about a configuration nobody asked for.
starved = Placement(
    assignment={"a": ("GPU-0",)}, envelope_gib={"a": 20.0}, kv_tokens={"a": MIN_KV // 2},
    util={"a": 0.4}, mps_reserve_gib={"a": 1.0}, reserve_source={"a": "default(tp1)"})
try:
    starved.verify(one_gpu, [req("a", 1)])
    check("a placement below a tenant's KV floor is a solver bug too, not a low-ranked option",
          False)
except SolverBug as exc:
    check("a placement below a tenant's KV floor is a solver bug too, not a low-ranked option",
          "floor" in str(exc), str(exc)[:80])

try:
    PlacementSolver(NODE, [req("wide", 16)])
    check("a TP degree wider than the node is refused up front", False)
except Infeasible:
    check("a TP degree wider than the node is refused up front", True)
# The complement, and the one that matters: more *tenants* than GPUs is the whole design, so
# it must not be mistaken for the case above.
nine = PlacementSolver(NODE, [req(f"t{i}", 1) for i in range(9)]).solve(cap=1)[0]
check("more tenants than GPUs is normal, not infeasible",
      nine.gpus_used <= 8 and max(nine.coresidency.values()) >= 1,
      f"{nine.gpus_used} GPUs, max coresidency {max(nine.coresidency.values())}")

# TP=4 has no measured activation or reserve on this image (impl §4.2: NCCL's buffer size
# changed, so the earlier TP=4 result does not transfer). The old solver refused to place it,
# because it had to predict an ITL for it and would not guess. There is nothing to guess now:
# TP=4 is priced by the same arithmetic as every other degree, on the documented fallbacks.
tp4 = PlacementSolver(NODE, [req("a", 4), req("b", 1)]).solve(cap=1)[0]
check("an unmeasured TP degree is placed on the stated fallbacks, not refused",
      len(set(tp4.placement.assignment["a"])) == 4
      and tp4.placement.reserve_source["a"] == "default(tpN)"
      and tp4.placement.kv_tokens["a"] >= tp4.floor_tokens["a"],
      f"{tp4.placement.assignment['a']} reserve={tp4.placement.mps_reserve_gib['a']} "
      f"({tp4.placement.reserve_source['a']})")

try:
    PlacementSolver(NODE, [req("huge", 1, mem=ModelMemory("h", 200.0, QWEN_KV_BYTES))]).solve()
    check("a tenant that cannot fit on one GPU is refused with a local reason", False)
except Infeasible as exc:
    check("a tenant that cannot fit on one GPU is refused with a local reason", True)
    # The other side of the §14.2 split, and the reason `Unproven` is a subclass: this refusal is
    # witnessed by 200 GiB of weights on a 44.99 GiB card, so it stays a statement about the fleet.
    # A test that only caught `Infeasible` would pass on either, which is why the class is asserted.
    check("  and a tenant whose WEIGHTS exceed a GPU is refused outright, not deferred to a boot",
          not isinstance(exc, Unproven) and "weights alone are 200.00 GiB" in str(exc)
          and "neither can a measurement" in str(exc), str(exc)[-110:])

# The fleet-level witness, which `solve` and `solve_pinned` consult instead of pricing envelopes.
# Two bounds, and the second exists because a fleet can be impossible while no single tenant is.
check("no witness when every tenant's weights fit one GPU, whatever the envelopes say",
      PlacementSolver(NODE, [req("greedy", 1, min_kv=0, floor=5_000_000)]
                      ).witnessed_infeasibility() is None,
      "a 564 GiB envelope is not a witness; 1.11 GiB of weights is not over")
_w1 = PlacementSolver(NODE, [req("huge", 1, mem=ModelMemory("h", 200.0, QWEN_KV_BYTES))]
                      ).witnessed_infeasibility()
check("a shard heavier than one GPU is a witness, and says so in weights",
      _w1 is not None and "200.00 GiB of weights on one GPU" in _w1, _w1)
# 8 x 44.99 = 359.92 GiB on the node. Nine tenants at 40 GiB of weights each is 360.0 -- over by
# 0.08 GiB, with every individual shard fitting comfortably.
_w2 = PlacementSolver(NODE, [req(f"h{i}", 1, mem=ModelMemory("h", 40.0, QWEN_KV_BYTES))
                             for i in range(9)]).witnessed_infeasibility()
check("a fleet whose weights exceed the whole node is a witness even though every shard fits",
      _w2 is not None and "across 9 shard(s)" in _w2 and "359.92 GiB" in _w2, _w2)
check("  and eight of the same tenants are not a witness, so the bound is the node and not a "
      "round number",
      PlacementSolver(NODE, [req(f"h{i}", 1, mem=ModelMemory("h", 40.0, QWEN_KV_BYTES))
                             for i in range(8)]).witnessed_infeasibility() is None,
      "320.0 GiB of weights on 359.92 GiB of node")

# Twelve tp=1 tenants: with density first they will share, and the invariant has to hold on
# every shared GPU. This is the case where an expression bug in the capacity constraint would
# show up, which is why `verify()` -- which shares no code with the model -- runs on every
# candidate the solver returns, including this one.
twelve = PlacementSolver(NODE, [req(f"t{i}", 1) for i in range(12)]).solve(cap=1)[0]
check("twelve tp=1 tenants share GPUs while every GPU stays within capacity",
      max(twelve.coresidency.values()) >= 1
      and all(mib <= NODE.capacity_mib for mib in twelve.placement.gpu_load_mib().values()),
      f"{twelve.gpus_used} GPUs, max coresidency {max(twelve.coresidency.values())}, "
      f"max load {max(twelve.placement.gpu_load_mib().values()) / 1024:.2f} GiB")
# Pinned: the agent's "can this exact arrangement be hosted at all". Same objectives, same
# verification -- the point is that a co-residency comparison varies the arrangement and nothing
# else, so the envelopes must still be solved rather than copied from the free solve.
pinned6 = PlacementSolver(NODE, fleet6).solve_pinned(
    {"m1": ("GPU-0",), "r1": ("GPU-1",), "t2": ("GPU-0", "GPU-1"), "r2": ("GPU-2", "GPU-3")})
check("a pinned assignment comes back exactly as asked for, with envelopes still optimized",
      {n: tuple(sorted(g)) for n, g in pinned6.placement.assignment.items()}
      == {"m1": ("GPU-0",), "r1": ("GPU-1",), "t2": ("GPU-0", "GPU-1"),
          "r2": ("GPU-2", "GPU-3")}
      and pinned6.min_margin > 1.0,
      f"{pinned6.gpus_used} GPUs, tightest {pinned6.min_margin:.2f}x")

# --- solver: a fleet split over nodes, each tenant on one (solver/partition.py) ---
from colocation.solver.partition import disjoint_nodes, split as _split    # noqa: E402

check("the disjoint node count packs whole tenants, not shards, onto nodes",
      disjoint_nodes([4, 4, 2, 1, 1], 8) == 2 and disjoint_nodes([4, 4, 4], 8) == 2
      and disjoint_nodes([8, 1], 8) == 2 and disjoint_nodes([], 8) == 0)
# One node: the split is the fleet, and its one group is exactly what `PlacementSolver` returns.
_one = _split(NODE, fleet6, 1, cap=1)
check("over one node there is one group, and it is the single-node solver's own best placement",
      len(_one.splits) == 1 and _one.splits[0].nodes == 1
      and _one.splits[0].groups[0].candidate.placement == best.placement,
      f"{_one.splits[0].gpus if _one.splits else 0} GPU(s)")
# Nine tenants of 20 GiB: one fits a GPU with its KV floor and two do not, so one node of eight
# GPUs cannot hold them and two nodes can. Their weights alone would pair up two to a GPU, so "no
# split over one node" is a computed answer, and impl §14.2 says that is not a proof.
_nine = [req(f"h{i}", 1, mem=ModelMemory("h", 20.0, QWEN_KV_BYTES)) for i in range(9)]
_n1 = _split(NODE, _nine, 1, cap=1)
check("nine tenants a node cannot hold have no split over one node, and it is not called proven",
      not _n1.splits and _n1.exhausted and not _n1.proven_none and not _n1.unplaced)
_n2 = _split(NODE, _nine, 2, cap=2)
check("  and over two nodes they split, every tenant in exactly one group",
      _n2.splits and all(s.nodes == 2 and s.gpus == 9 for s in _n2.splits)
      and all(sorted(n for g in s.groups for n in g.tenants) == sorted(r.name for r in _nine)
              for s in _n2.splits),
      "; ".join(" | ".join(",".join(g.tenants) for g in s.groups) for s in _n2.splits))
check("  with every group's placement within each GPU's capacity",
      all(mib <= NODE.capacity_mib for s in _n2.splits for g in s.groups
          for mib in g.candidate.placement.gpu_load_mib().values()))
check("  and distinct splits, not one split's node labels swapped",
      len({frozenset(g.tenants for g in s.groups) for s in _n2.splits}) == len(_n2.splits))
check("  ranked fewest nodes, then fewest GPUs, then the largest smallest-margin",
      [s.rank_key for s in _n2.splits] == sorted(s.rank_key for s in _n2.splits))
_n2c = _split(NODE, _nine, 2, cap=2, apart=[("h0", "h1")], together=[("h0", "h2")])
check("  and an `apart` pair lands on different nodes, a `together` pair on the same one",
      _n2c.splits and all(
          any("h0" in g.tenants and "h2" in g.tenants for g in s.groups)
          and not any("h0" in g.tenants and "h1" in g.tenants for g in s.groups)
          for s in _n2c.splits),
      "; ".join(" | ".join(",".join(g.tenants) for g in s.groups) for s in _n2c.splits))
try:
    _split(NODE, _nine, 2, apart=[("h0", "zz")])
    check("  and a pair naming a tenant the fleet does not have is refused", False)
except ValueError:
    check("  and a pair naming a tenant the fleet does not have is refused", True)
_huge = _split(NODE, fleet6 + [req("huge", 1, mem=ModelMemory("h", 200.0, QWEN_KV_BYTES))], 4,
               cap=1)
# At 30 GiB two tenants cannot share a GPU even on their weights, so the same refusal is a fact.
check("  and at 30 GiB, where two cannot share a GPU on weights alone, the refusal is proven",
      _split(NODE, [req(f"h{i}", 1, mem=ModelMemory("h", 30.0, QWEN_KV_BYTES)) for i in range(9)],
             1, cap=1).proven_none)
check("a tenant too big for any node is named, and no split is offered, however many nodes",
      not _huge.splits and list(_huge.unplaced) == ["huge"] and _huge.proven_none,
      str(_huge.unplaced)[:160])
try:
    _split(NODE, fleet6, 0)
    check("zero nodes is refused, not read as one", False)
except ValueError:
    check("zero nodes is refused, not read as one", True)

# --- agent: `search` over several nodes, one ordinary search per group (agent/splits.py) ---
from colocation.agent import splits as agent_splits                    # noqa: E402

_pdoc = {"hardware": {"instance_type": "g6e.48xlarge", "max_nodes": 2}, "workload": {"x": 1},
         "models": {"a": {"model_id": "A"}, "b": {"model_id": "B"}},
         "tenants": [{"name": "t1", "model": "a"}, {"name": "t2", "model": "b"},
                     {"name": "t3", "model": "a"}]}
_gd = agent_splits.group_doc(_pdoc, ("t3", "t1"))
check("a group is the parent fleet cut to its tenants, on one node, with only the models it uses",
      [t["name"] for t in _gd["tenants"]] == ["t1", "t3"] and list(_gd["models"]) == ["a"]
      and _gd["hardware"] == {"instance_type": "g6e.48xlarge"} and _gd["workload"] == {"x": 1}
      and _pdoc["hardware"]["max_nodes"] == 2, str(_gd))
check("  and the same tenant set of the same fleet is the same file, in any order",
      agent_splits.group_path("/r", "f/fleet.yaml", ("t1", "t3"))
      == agent_splits.group_path("/r", "f/fleet.yaml", ("t3", "t1"))
      != agent_splits.group_path("/r", "f/fleet.yaml", ("t1", "t2")))
try:
    agent_splits.group_doc(_pdoc, ("t1", "nope"))
    check("  and a tenant the fleet does not have is refused", False)
except ValueError:
    check("  and a tenant the fleet does not have is refused", True)


def _g(names, margin):
    return SimpleNamespace(tenants=tuple(names), min_margin=margin)


# Split #1 is {t1,t2} | {t3} and #2 is {t1} | {t2,t3}. {t3} is the tighter group of #1, so it is
# searched first; when it misses, #1 is done without a node for {t1,t2}, and #2 reuses nothing.
_sp = [SimpleNamespace(groups=(_g(["t1", "t2"], 2.0), _g(["t3"], 1.1))),
       SimpleNamespace(groups=(_g(["t1"], 3.0), _g(["t2", "t3"], 1.5)))]


def _runner(met):
    calls = []

    def run(tenants):
        calls.append(tenants)
        return agent_splits.GroupRun(tenants, "-", "met" if frozenset(tenants) in met else "budget")
    return run, calls


_run, _calls = _runner({frozenset({"t2", "t3"}), frozenset({"t1"})})
_dr = agent_splits.drive(_sp, _run, budget=4)
check("the driver searches the tightest group first, and drops a split at its first miss",
      _calls == [("t3",), ("t2", "t3"), ("t1",)] and _dr.split is _sp[1]
      and _dr.tried == [(1, _dr.runs[0]), (2, None)], f"{_calls}; {_dr.stopped_because}")
_run, _calls = _runner({frozenset({"t3"})})
_sp3 = _sp + [SimpleNamespace(groups=(_g(["t1", "t2"], 2.0), _g(["t3"], 1.1)))]
_dr = agent_splits.drive(_sp3, _run, budget=4)
check("  and never searches a group twice: a group's result is the same in every split",
      len(_calls) == len(set(_calls)) == 3 and _dr.split is None
      and "none of the 3" in _dr.stopped_because, f"{_calls}; {_dr.stopped_because}")
_run, _calls = _runner(set())
_dr = agent_splits.drive(_sp, _run, budget=1)
check("  and stops when the group budget is spent, saying so",
      len(_calls) == 1 and _dr.split is None and "budget of 1" in _dr.stopped_because,
      _dr.stopped_because)
check("a group's search is the same command, with its own fleet file and summary and no run token",
      agent_main._group_argv(["search", "--fleet", "p.yaml", "--budget", "3", "--run-token=abc",
                              "--export-to", "x.yaml", "--summary", "s.json", "--release-node"],
                             "g.yaml", "g.json")
      == ["search", "--budget", "3", "--release-node", "--fleet", "g.yaml", "--summary", "g.json",
          "--allow-disjoint-only"])
check("  and a group no two of whose tenants can share a GPU is measured, not refused",
      agent_main._group_argv(["search", "--allow-disjoint-only"], "g.yaml", "g.json").count(
          "--allow-disjoint-only") == 1)

# An earlier run's group is recalled by the digest of its fleet file, not by its name: the same
# tenants under an edited target are the same file name and a different group.
with tempfile.TemporaryDirectory() as _rd:
    _gp = agent_splits.write_group(_rd, "f/fleet.yaml", _pdoc, ("t1",))
    _sha = agent_splits.text_sha256(_gp.read_text())
    for _stamp, _body in (("20261008T000000Z", {"verdict": "met", "group_fleet_sha256": _sha,
                                                "run_token": "old"}),
                          ("20261009T000000Z", {"verdict": "budget", "group_fleet_sha256": "x",
                                                "run_token": "edited"}),
                          ("20261010T000000Z", {"group_fleet_sha256": _sha, "run_token": "none"})):
        _gp.with_name(f"{_gp.stem}.{_stamp}.summary.json").write_text(json.dumps(_body))
    _rc = agent_splits.recall_group(_rd, "f/fleet.yaml", _pdoc, ("t1",))
    check("a group an earlier run searched is recalled, newest first, only for the same file text",
          _rc is not None and _rc.recalled and _rc.met and _rc.summary["run_token"] == "old",
          str(_rc))
    _pdoc2 = copy.deepcopy(_pdoc)
    _pdoc2["tenants"][0]["slo"] = {"itl_mean_ms": 55.0}
    check("  and not once the parent's text for that group has changed",
          agent_splits.recall_group(_rd, "f/fleet.yaml", _pdoc2, ("t1",)) is None
          and agent_splits.recall_group(_rd, "f/fleet.yaml", _pdoc, ("t3",)) is None)


def _recall_of(met):
    return lambda tenants: (agent_splits.GroupRun(tenants, "-", "met", {}, recalled=True)
                            if frozenset(tenants) in met else None)


_run, _calls = _runner({frozenset({"t1"}), frozenset({"t2", "t3"})})
_dr = agent_splits.drive(_sp, _run, budget=1, recall=_recall_of({frozenset({"t3"})}))
check("  and the driver searches no recalled group, and spends no budget on one",
      _calls == [("t1", "t2")] and _dr.split is None and "budget of 1" in _dr.stopped_because
      and [r.recalled for r in _dr.runs] == [True, False], f"{_calls}; {_dr.stopped_because}")

check("the driver's next step is the tightest group not yet searched of the first live split",
      agent_splits.next_group(_sp, {}) == ("t3",)
      and agent_splits.next_group(_sp, {frozenset({"t3"}): agent_splits.GroupRun(
          ("t3",), "-", "budget")}) == ("t2", "t3")
      and agent_splits.next_group(_sp[1:], {frozenset(g.tenants): agent_splits.GroupRun(
          g.tenants, "-", "met") for g in _sp[1].groups}) is None)

# --- agent: the reasoner choosing groups (splitpacker) --------------------------------------
# The same two splits, given what `partition.Split` has, and a `solve_splits` that honours `apart`.


def _gx(names, margin, gpus):
    return SimpleNamespace(tenants=tuple(names), min_margin=margin, gpus=gpus)


def _sx(*groups):
    return SimpleNamespace(groups=groups, nodes=len(groups), gpus=sum(g.gpus for g in groups),
                           min_margin=min(g.min_margin for g in groups),
                           rank_key=(len(groups), sum(g.gpus for g in groups)))


_spx = [_sx(_gx(["t1", "t2"], 2.0, 3), _gx(["t3"], 1.1, 1)),
        _sx(_gx(["t1"], 3.0, 1), _gx(["t2", "t3"], 1.5, 4))]


def _solve_x(apart=(), together=()):
    out = [s for s in _spx if not any(
        any(a in g.tenants and b in g.tenants for g in s.groups) for a, b in apart)]
    return SimpleNamespace(splits=tuple(out), unplaced={}, exhausted=True, proven_none=not out,
                           max_nodes=2)


def _sxtools(met, budget=4):
    run, calls = _runner(met)
    fleet = SimpleNamespace(tenants=[SimpleNamespace(name=n) for n in ("t1", "t2", "t3")])
    return splitpacker.SplitTools(fleet, _solve_x, run, budget), calls


def _refused(tools, tool, args):
    try:
        tools.dispatch(tool, args)
        return False
    except prompt.ToolError:
        return True


_st, _calls = _sxtools({frozenset({"t1"}), frozenset({"t2", "t3"})})
check("a group is refused until `partition` has listed a split that contains it",
      _refused(_st, "search_group", {"tenants": ["t3"], "rationale": "r"}) and not _calls)
_listing = _st.dispatch("partition", {"apart": [["t1", "t2"]]})
check("  and `partition` lists only the splits that keep an `apart` pair apart",
      "S1:" in _listing and "S2:" not in _listing and len(_st.splits) == 1
      and _st.splits[0] is _spx[1], _listing)
_st.dispatch("partition", {})
check("  and a split keeps the id it was first listed under",
      _st.splits[0] is _spx[1] and _st.splits[1] is _spx[0])
check("  and `apart` names that are not tenants are refused",
      _refused(_st, "partition", {"apart": [["t1", "zz"]]})
      and _refused(_st, "partition", {"apart": [["t1"]]}))
check("  and a search needs a rationale",
      _refused(_st, "search_group", {"tenants": ["t1"]}) and not _calls)
_st.dispatch("search_group", {"tenants": ["t1"], "rationale": "r"})
check("  and `met` is refused while a group of the split is unsearched",
      _refused(_st, "submit", {"split": "S1", "verdict": "met", "rationale": "r"}))
check("  and a group is never searched twice",
      _refused(_st, "search_group", {"tenants": ["t1"], "rationale": "r"}) and len(_calls) == 1)
_st.dispatch("search_group", {"tenants": ["t3", "t2"], "rationale": "r"})
check("  and a group named in any order is the listed group, searched in fleet order",
      _calls == [("t1",), ("t2", "t3")])
check("  and `met` is accepted for a split every group of which met",
      _st.dispatch("submit", {"split": "S1", "verdict": "met", "rationale": "r"}) == "submitted"
      and _st.met_split() is _spx[1])
_st, _calls = _sxtools(set(), budget=1)
_st.dispatch("partition", {})
_st.dispatch("search_group", {"tenants": ["t3"], "rationale": "r"})
check("  and the group budget is enforced inside the tool",
      _refused(_st, "search_group", {"tenants": ["t1", "t2"], "rationale": "r"})
      and len(_calls) == 1)
_st, _calls = _sxtools({frozenset({"t1"}), frozenset({"t2", "t3"})})
_spa = splitpacker.SplitPackerAgent(_st, ReplayReasoner([
    '{"tool": "partition", "args": {}}',
    '{"tool": "search_group", "args": {"tenants": ["t3"], "rationale": "tightest"}}',
    '{"tool": "recall", "args": {"tenants": ["t3"]}}',
    '{"tool": "partition", "args": {"apart": [["t1", "t3"]]}}',
    '{"tool": "search_group", "args": {"tenants": ["t1"], "rationale": "S2"}}',
    '{"tool": "search_group", "args": {"tenants": ["t2", "t3"], "rationale": "S2"}}',
    '{"tool": "submit", "args": {"split": "S2", "verdict": "met", "rationale": "both met"}}']),
    max_nodes=2, instance_type="x")
_spa.opening = lambda: "problem"
_spr = _spa.run()
check("the split reasoner reacts to a missed group, and the verdict is derived from the runs",
      _spr.verdict == "met" and _spr.split is _spx[1]
      and _spr.stopped_because == splitpacker.STOP_SUBMITTED
      and _calls == [("t3",), ("t1",), ("t2", "t3")] and _spr.chosen_by == ["reasoner"] * 3,
      f"{_spr.verdict} {_spr.stopped_because} {_calls}")
_st, _calls = _sxtools({frozenset({"t2", "t3"})}, budget=1)
_st.recall = _recall_of({frozenset({"t1"})})
_listing = _st.dispatch("partition", {})
check("  and a group an earlier run searched is listed as searched, and costs no budget",
      "searched by an earlier run: met: t1" in _listing and _st.remaining == 1
      and _refused(_st, "search_group", {"tenants": ["t1"], "rationale": "r"}) and not _calls,
      _listing)
_st.dispatch("search_group", {"tenants": ["t2", "t3"], "rationale": "r"})
check("  and a split met through it is met",
      _st.met_split() is _spx[1] and [by for _, by in _st.order] == ["store", "reasoner"])
_st, _calls = _sxtools({frozenset({"t1"}), frozenset({"t2", "t3"})})
_spa = splitpacker.SplitPackerAgent(_st, ReplayReasoner(["no"] * 9), max_nodes=2,
                                    instance_type="x", max_turns=9)
_spa.opening = lambda: "problem"
_spr = _spa.run()
check("  and unusable replies fall back to the driver's next group, marked as such",
      _calls == [("t3",), ("t2", "t3"), ("t1",)] and _spr.chosen_by == ["fallback"] * 3
      and _spr.verdict == "met" and _spr.split is _spx[1],
      f"{_calls} {_spr.chosen_by} {_spr.verdict}")
check("  and the reasoner is told what was searched on its behalf",
      any("recorded as a fallback" in t.text for t in _spr.transcript.turns))

# --- agent: the fleet spec ---------------------------------------------------
# The whole agent layer is exercised here with `FakeDeployer` and `FakeHarness`. That is the
# point of those two existing: the loop is the one component that would otherwise be first
# tested by spending GPU time on it.
INV = NodeInventory(node_name="node-a", gpu_ids=tuple(range(8)),
                    uuids=tuple(f"GPU-{i:04d}" for i in range(8)), device_total_gib=CARD_GIB)
try:
    NodeInventory("n", (0, 1), ("GPU-0",), CARD_GIB)
    check("a GPU ordinal list longer than its UUID list is refused", False)
except ValueError:
    check("a GPU ordinal list longer than its UUID list is refused", True)
try:
    NodeInventory("n", (0,), ("0",), CARD_GIB)
    check("an ordinal used where a UUID belongs is refused, not silently ignored", False)
except ValueError:
    check("an ordinal used where a UUID belongs is refused, not silently ignored", True)


def tmpl(name: str, served: str, tp: int, rate: float = 16.0,
         isl: int = 512, osl: int = 128, **kw) -> TenantTemplate:
    return TenantTemplate(name=name, model_id="Qwen/Qwen3-0.6B",
                          model_path="/fsx/pretrained-models/Qwen/Qwen3-0.6B",
                          served_name=served, tp=tp, rate_hz=rate,
                          isl_tokens=isl, osl_tokens=osl, mem=QWEN, **kw)


FLEET = FleetSpec(tenants=(tmpl("m1", "qwen3-0-6b-a", 1), tmpl("r1", "qwen3-0-6b-i3a", 1),
                           tmpl("r2", "qwen3-0-6b-r2", 2), tmpl("t2", "qwen3-0-6b-tp2", 2)),
                  node=INV, workload=WorkloadSpec())
# max_model_len is checked against the tenant's OWN span. A fleet-wide check could only ask
# whether the longest shape fits every tenant, which passes a mixed fleet that does not fit.
try:
    FleetSpec(tenants=(TenantTemplate("s", "Qwen/Qwen3-0.6B", "/p", "s", 1, 2.0, 512, 128,
                                      QWEN, max_model_len=256),), node=INV)
    check("a shape longer than the tenant's max_model_len is refused as a config error", False)
except ValueError:
    check("a shape longer than the tenant's max_model_len is refused as a config error", True)
check("  and the check is per tenant, so a short-shaped tenant is not judged by a long one",
      FleetSpec(tenants=(tmpl("short", "s", 1, isl=100, osl=28),
                         tmpl("long", "l", 1, isl=2048, osl=1024)),
                node=INV).sequence_shapes
      == {"short": (100, 28, 0, 0), "long": (2048, 1024, 0, 0)})
try:
    tmpl("bad", "b", 1, isl=0, osl=128)
    check("a shape below one token is refused", False)
except ValueError:
    check("a shape below one token is refused", True)

# --- agent: per-tenant SLOs as an input (design §2) --------------------------
# Design §2 states the problem as "given a node, a set of models, and per-model SLOs, produce a
# configuration that serves all of them". These check the input side: that a target can be
# stated, that an unstated one stays unstated, and that stating one does not disturb the
# validity key. Nothing consumes these yet -- the solver still minimises occupancy.
check("a tenant states its own SLO, and 'stated' names only what it carries",
      TenantSlo(itl_mean_ms=8.0).stated == ("itl_mean_ms",)
      and TenantSlo(ttft_p95_ms=400.0, itl_mean_ms=8.0).stated
      == ("ttft_p95_ms", "itl_mean_ms"))
try:
    TenantSlo()
    check("an empty slo block is refused: 'no target' and 'a target of nothing' differ", False)
except ValueError:
    check("an empty slo block is refused: 'no target' and 'a target of nothing' differ", True)
for _bad in ({"itl_mean_ms": 0}, {"ttft_p95_ms": -1.0}):
    try:
        TenantSlo(**_bad)
        check(f"a non-positive ceiling is refused ({_bad})", False)
    except ValueError:
        check(f"a non-positive ceiling is refused ({_bad})", True)

# THE BATCH CAP IS DERIVED FROM THE SAME DEMAND THE POOL IS (impl §14.9), and it used to be a
# fleet-file field. The defect search-r2hk3z was refused by is what it has to keep out: steady state
# needs N = rate x osl x ITL sequences resident and `--max-num-seqs` caps N, so under the cap the
# tenant does not miss its target -- it never reaches steady state, and the harness reports that as
# infeasibility of the placement rather than as a latency. A number that must not be under the
# demand is a number nobody should be typing. The tenant below is that fleet's anchor.
_cap_t = tmpl("ok", "o", 1, isl=1024, osl=1024, rate=1.0, slo=TenantSlo(itl_mean_ms=80.0))
check("a tenant says how much of its batch cap its own target needs",
      abs(_cap_t.concurrency_at_target() - 81.92) < 1e-9)
check("  and the derived cap admits that concurrency, which is the thing it may never fail to do",
      _cap_t.max_num_seqs >= math.ceil(_cap_t.concurrency_at_target()),
      f"cap {_cap_t.max_num_seqs} vs {_cap_t.concurrency_at_target():.2f} at target")
# 81.92 mean + 2.33 sqrt(81.92) = 102.97 slots at the q0.99 arrival burst. The cap is sized for the
# BURST and reported against the mean, because the mean is the load the file states and the burst is
# what a Poisson arrival process does to it -- see `burst_concurrency`.
check("  and it is the burst occupancy of a slot rather than the mean, rounded up",
      _cap_t.max_num_seqs == 103, str(_cap_t.max_num_seqs))
check("  and reports the mean as a fraction of it rather than thresholding it",
      abs(_cap_t.concurrency_load() - 81.92 / 103) < 1e-9)
# A slot is held from admission to the last token, so a stated TTFT ceiling is part of the
# occupancy -- and it is the half the decode-only arithmetic missed on the GLM anchor profiled alone,
# where a 56 s prefill queue at 0.5 Hz was 28 slots nobody had counted.
check("  and a stated TTFT ceiling raises the cap, because a queued request holds a slot too",
      tmpl("ttft", "t", 1, isl=1024, osl=1024, rate=1.0,
           slo=TenantSlo(ttft_p95_ms=4000.0, itl_mean_ms=80.0)).max_num_seqs > _cap_t.max_num_seqs)
# The floor under the derivation is the ENGINE's own default, not this arithmetic. Below the offered
# burst a queue diverges rather than degrades, so a cap the arithmetic would have made tighter than
# the engine's own is a risk taken for no memory worth having -- and taking the engine's number
# keeps every stored row that was measured at it (`store/serving.py` keys on the cap).
_small = tmpl("small", "s", 1, isl=512, osl=16, rate=1.0, slo=TenantSlo(itl_mean_ms=20.0))
check("  while a tenant whose demand is under the engine's own default keeps the default",
      _small.max_num_seqs == _eng.get(_eng.DEFAULT_ENGINE).default_max_num_seqs
      and _small.concurrency_at_target() < _small.max_num_seqs,
      f"{_small.concurrency_at_target():.2f} wanted, cap {_small.max_num_seqs}")
check("  and a tenant with no ITL target is not judged against a cap at all, and gets the default",
      tmpl("notarget", "n", 1, isl=1024, osl=1024, rate=1.0).concurrency_at_target() is None
      and tmpl("notarget", "n", 1, isl=1024, osl=1024, rate=1.0).max_num_seqs
      == _eng.get(_eng.DEFAULT_ENGINE).default_max_num_seqs)
# The invariant, over shapes rather than on one: no demand this arithmetic can be handed produces a
# cap under the mean concurrency that demand implies. A fleet file could state that combination and
# every shipped one had to be checked by hand; a derivation cannot.
check("  and no shape at all derives a cap below its own mean concurrency",
      all(tmpl("x", "x", 1, isl=512, osl=_o, rate=_r,
               slo=TenantSlo(itl_mean_ms=_i)).max_num_seqs
          >= math.ceil(tmpl("x", "x", 1, isl=512, osl=_o, rate=_r,
                            slo=TenantSlo(itl_mean_ms=_i)).concurrency_at_target())
          for _o in (8, 128, 1024) for _r in (0.5, 6.0, 16.0) for _i in (5.0, 60.0, 200.0)))
# A fleet file that still states either derived field is refused by name rather than ignored, so a
# file carried over from before §14.9 cannot go on looking like it is being obeyed. The loader owns
# that refusal; this is the field list it refuses.
for _gone in ("min_kv_tokens", "max_num_seqs"):
    check(f"  and `{_gone}` is not a field of a tenant template any more",
          _gone not in {f.name for f in dataclasses.fields(TenantTemplate)})

# Little's law at the *target* ITL, which is what makes the target actionable rather than
# decorative: 16 Hz x 128 output tokens x 8 ms = 16.384 requests resident in the MEAN, 25.80 at the
# q0.99 arrival burst the pool is sized for, x 640 tokens each. Sizing for the mean would leave the
# pool short half the time, and a short pool preempts -- which the engine serves as a worse ITL, so
# the tenant would miss its target for a reason nothing in its record names (impl §14.9).
_slo_t = TenantTemplate(name="s1", model_id="Qwen/Qwen3-0.6B", model_path="/p",
                        served_name="s1", tp=1, rate_hz=16.0, isl_tokens=512, osl_tokens=128,
                        mem=QWEN, slo=TenantSlo(itl_mean_ms=8.0))
_kv = _slo_t.kv_tokens_at_target()
check("the KV a tenant needs to hold its ITL target follows from Little's law, not a guess",
      _kv is not None and _kv[0] == 16513, str(_kv))
_tight = TenantTemplate(name="s1", model_id="Qwen/Qwen3-0.6B", model_path="/p",
                        served_name="s1", tp=1, rate_hz=16.0, isl_tokens=512, osl_tokens=128,
                        mem=QWEN, slo=TenantSlo(itl_mean_ms=4.0))
check("  and a TIGHTER latency ceiling needs LESS KV, because requests finish sooner",
      _tight.kv_tokens_at_target()[0] == 9505,
      str(_tight.kv_tokens_at_target()))
# The floor is the tenant's own shape all the way through: same rate, same ceiling, a longer
# input, and the requirement rises in proportion to the span it must hold resident.
_long_in = TenantTemplate(name="s1", model_id="Qwen/Qwen3-0.6B", model_path="/p",
                          served_name="s1", tp=1, rate_hz=16.0, isl_tokens=1152,
                          osl_tokens=128, mem=QWEN, slo=TenantSlo(itl_mean_ms=8.0))
check("  and a tenant with a longer input needs proportionally more, from its own shape",
      _long_in.kv_tokens_at_target()[0] == 33025
      and abs(_long_in.kv_tokens_at_target()[0] - 2 * _kv[0]) <= 1,
      f"{_long_in.kv_tokens_at_target()} vs {_kv}")
# The burst is where the only new INPUT of §14.9 lives, and both ways of stating it are the fleet's.
# A stated peak rate assumes no distribution at all -- the occupancy scales with the rate, so the
# burst occupancy is the mean one at the peak -- which is the honest form for a demand somebody has
# actually characterised. The quantile is for one nobody has.
_peaked = dataclasses.replace(_slo_t, peak_rate_hz=32.0)
check("a stated peak rate sizes the pool at the peak, with no distribution assumed",
      _peaked.kv_tokens_at_target()[0] == 2 * 10486
      and "32 Hz peak" in _peaked.kv_tokens_at_target()[1],
      str(_peaked.kv_tokens_at_target()))
check("  and a stated quantile moves the burst, and says the tenant stated it",
      dataclasses.replace(_slo_t, burst_quantile=0.5).kv_tokens_at_target()[0] == 10486
      and "stated by this tenant"
      in dataclasses.replace(_slo_t, burst_quantile=0.5).kv_tokens_at_target()[1])
for _bad, _why in (({"peak_rate_hz": 8.0}, "a peak below the mean rate"),
                   ({"burst_quantile": 1.0}, "a quantile of 1, which has no finite value"),
                   ({"burst_quantile": 0.4}, "a quantile below the median")):
    try:
        dataclasses.replace(_slo_t, **_bad)
        check(f"{_why} is refused rather than sized for", False)
    except ValueError:
        check(f"{_why} is refused rather than sized for", True)
check("  a tenant with no ITL target yields None, not zero -- 'no requirement' is not "
      "'a requirement of nothing'",
      tmpl("m1", "m1", 1).kv_tokens_at_target() is None)
check("  and a tenant with only a TTFT target also yields None, since the derivation needs ITL",
      TenantTemplate(name="s1", model_id="Qwen/Qwen3-0.6B", model_path="/p", served_name="s1",
                     tp=1, rate_hz=16.0, isl_tokens=512, osl_tokens=128, mem=QWEN,
                     slo=TenantSlo(ttft_p95_ms=400.0)).kv_tokens_at_target() is None)
check("a fleet reports which tenants stated no target rather than filling one in",
      FLEET.untargeted == ("m1", "r1", "r2", "t2"))

# The parser refuses what it cannot honour. A silently dropped target would read downstream as
# "this tenant never stated one", so the search would decline to certify a tenant whose target
# was in the file all along, misspelt.
for _bad_slo, _why in (({"itl_ms": 8.0}, "a misspelt key"),
                       ({"itl_mean_ms": 8.0, "p99_ms": 1.0}, "an extra key"),
                       ([8.0], "a list instead of a mapping")):
    try:
        fleetfile.load_slo("t", _bad_slo)
        check(f"the loader refuses {_why} in an slo block", False)
    except fleetfile.FleetFileError:
        check(f"the loader refuses {_why} in an slo block", True)
check("an absent slo block loads as None, which is how 'no target' is expressed",
      fleetfile.load_slo("t", None) is None)

# The load-bearing claim: a target is a criterion for judging a measurement, not a condition of
# taking one, so it must not enter the validity key. If it did, tightening a threshold would
# orphan a store full of good records and force a re-measure that could learn nothing.
_base_yaml = Path(__file__).resolve().parent / "fixtures" / "fleets" / "loader-16hz.yaml"
_src = _base_yaml.read_text()


def _floorload(path, **kw):
    """Load a loader fixture at its memory floors, which is the only way it can be loaded.

    A fleet file may not state `tp:` -- see `fleetfile`'s header -- and these fixtures state no
    latency target either, so no degree of theirs is resolvable from a measurement. Every check that
    is about something *other* than TP resolution therefore loads them the way `plan` does. Passing
    the flag rather than deleting the checks keeps the thing each one tests as the reason it fails:
    without it, a fleet that names a node would be refused for having no profile store instead.

    `index` defaults to the suite's store for the same reason `plan` requires one: an unleased load
    has to resolve the instance type somewhere, and there is no table left to resolve it from.
    """
    kw.setdefault("index", HW_INDEX)
    kw.setdefault("hub", HUB)
    return fleetfile.load(path, allow_unresolved_tp=True, **kw)
_with_slo = _src.replace("/fsx/pretrained-models/Qwen/Qwen3-0.6B}",
                         "/fsx/pretrained-models/Qwen/Qwen3-0.6B, slo: {ttft_p95_ms: 400, itl_mean_ms: 8.0}}")
check("  (the fixture edit applied to every tenant)", _with_slo.count("slo:") == 4)
with tempfile.TemporaryDirectory() as _td:
    _p = Path(_td) / "with-slo.yaml"
    _p.write_text(_with_slo)
    _slo_loaded = _floorload(_p)
    _tighter = Path(_td) / "tighter.yaml"
    _tighter.write_text(_with_slo.replace("itl_mean_ms: 8.0", "itl_mean_ms: 6.0"))
    _tight_loaded = _floorload(_tighter)
_plain = _floorload(_base_yaml)
check("an SLO survives the fleet file into the tenant it was written for",
      _slo_loaded["fleet"].tenant("t2").slo == TenantSlo(ttft_p95_ms=400.0, itl_mean_ms=8.0))
check("  and a fleet with every tenant targeted reports nothing untargeted",
      _slo_loaded["fleet"].untargeted == ())
check("stating an SLO does not change the validity key -- a target is not a condition of "
      "measurement", _slo_loaded["key"].hash == _plain["key"].hash, _plain["key"].hash)
check("  nor does tightening one, so ten records survive a change of mind about the target",
      _tight_loaded["key"].hash == _plain["key"].hash)
check("  while the offered rate still does key, because it changes what the hardware does",
      _plain["key"].offered_rates_hz == {"m1": 16.0, "r1": 16.0, "r2": 16.0, "t2": 16.0})
check("  and so does the shape, for the same reason",
      _plain["key"].sequence_shapes
      == {"m1": (512, 128, 0, 0), "r1": (512, 128, 0, 0), "r2": (512, 128, 0, 0),
          "t2": (512, 128, 0, 0)})

# --- the loader's "does not fit at any TP degree" refusal, split by what put it over (§14.2) ---
#
# This is the FIRST refusal a fleet can meet -- earlier than anything in `solver/cpsat` -- and the
# one most able to close a fleet on a number nobody checked, because it fires while the envelope is
# still entirely computed. The split is not cosmetic: one branch says "a wider machine or a smaller
# model", which is a fact, and the other says "nothing here has ruled this tenant out", which is a
# statement about the arithmetic. Both come out of the same code path, so both are checked here.
with tempfile.TemporaryDirectory() as _td:
    # A KV pool nothing can hold: 8M tokens at 14336 B/token/shard is ~107 GiB per shard even at
    # tp8. The weights are 1.110 GiB, so 0.14 GiB per shard -- comfortably under the card.
    _kvbig = Path(_td) / "kv-over.yaml"
    _kvbig.write_text(_src.replace("/fsx/pretrained-models/Qwen/Qwen3-0.6B}", "/fsx/pretrained-models/Qwen/Qwen3-0.6B, max_model_len: 4000000}"))
    # `plan` and `profile` LOAD it, at the widest degree the node offers. They are the two modes that
    # exist to resolve an open degree, so refusing them here made the refusal's own advice --
    # "nothing has ruled this tenant out, go and measure" -- impossible to take.
    _kvloaded = _floorload(_kvbig)
    check("a tenant whose KV floor fits nowhere still LOADS for `plan` and `profile`, at tp8",
          _kvloaded["fleet"].tp_degrees == {"m1": 8, "r1": 8, "r2": 8, "t2": 8}
          and fleetfile.unresolved_tp(_kvloaded) == ["m1", "r1", "r2", "t2"],
          str(_kvloaded["fleet"].tp_degrees))
    check("  and the degree it enters at is named a floor, never a minimum, since nothing measured it",
          all(r.tp is None and not r.certain for r in _kvloaded["tp_resolutions"].values()))
    try:
        fleetfile.load(_kvbig, index=HW_INDEX, hub=HUB)
        check("a tenant whose KV floor exceeds the card at every degree is refused", False)
    except fleetfile.FleetFileError as exc:
        _kvmsg = str(exc)
        check("a tenant whose KV floor exceeds the card at every degree is refused",
              "does not fit on a g6e.48xlarge at any TP degree" in _kvmsg, _kvmsg[:160])
        check("  by the modes that would certify a result, which is whose refusal it is",
              "this refusal is THIS mode's" in _kvmsg
              and "`plan` and `profile` load this fleet" in _kvmsg)
        check("  but the refusal says its WEIGHTS FIT, so the refusal is about computed terms",
              "BUT ITS WEIGHTS FIT" in _kvmsg and "0.14 GiB" in _kvmsg, _kvmsg[-400:])
        check("  and names the two ways out that do not require a bigger machine",
              "lower its rate" in _kvmsg and "run `profile`" in _kvmsg
              and "Neither needs a wider machine" in _kvmsg, _kvmsg[-300:])
        check("  and forbids the reading that closed fleets before it was split",
              "may NOT be read as is 'this tenant cannot be served'" in _kvmsg)
        check("  and does NOT claim no measurement changes it, because one might",
              "no measurement changes it" not in _kvmsg)

    # The other branch: weights alone over the card at the widest degree. 400 GiB / 8 = 50 GiB per
    # shard against ~45. Nothing optimistic is left to find -- no KV pool, no activation, no
    # reserve, no headroom divisor -- so this one may close the fleet.
    #
    # The FILE is not edited any more, and cannot be: since Fix 2 a weight figure is not a line in it
    # (`store/hubfacts.py`). What moves instead is what the checkpoint says, through an overriding
    # manifest -- no published checkpoint is 400 GiB, and the refusal is about the loader's arithmetic
    # rather than about anything real being that big.
    _HUGE_HUB = fixture_hub({"Qwen/Qwen3-0.6B": synthetic_manifest("Qwen/Qwen3-0.6B",
                                                                   resident_gib=400.0)})
    try:
        _floorload(_base_yaml, hub=_HUGE_HUB)
        check("a tenant whose WEIGHTS exceed the card at every degree is refused even to "
              "`plan` and `profile`, which is the asymmetry: there is nothing to go and measure",
              False)
    except fleetfile.FleetFileError as exc:
        _wmsg = str(exc)
        check("a tenant whose WEIGHTS exceed the card at every degree is refused even to "
              "`plan` and `profile`, which is the asymmetry: there is nothing to go and measure",
              "WEIGHTS alone are 50.00 GiB per shard at tp8" in _wmsg, _wmsg[-300:])
        check("  and THAT refusal does close it: no placement and no measurement changes it",
              "no measurement changes it" in _wmsg and "wider machine" in _wmsg)
        check("  and it does not offer the computed-terms way out, which does not exist here",
              "BUT ITS WEIGHTS FIT" not in _wmsg and "lower its rate" not in _wmsg)

# --- and the same split applied to "no co-resident placement is feasible" ---------------------
#
# `can_coreside` is computed on full minimum envelopes, so a False there is a claim about a derived
# page size, an activation fallback and a reserve constant as much as about the hardware. The
# witness question is whether the two shards' WEIGHTS fit one GPU; if they do, a node-hour can still
# find a pair the arithmetic could not.
from colocation.agent.__main__ import pairs_fitting_on_weights    # noqa: E402
from colocation.agent.__main__ import _baseline as _bl_w          # noqa: E402

_pw_base = _bl_w(_plain["fleet"])
check("four 1.110 GiB tenants: every pair fits one GPU on weights alone",
      len(pairs_fitting_on_weights(_pw_base, _plain["fleet"])) == len(_pw_base.pairs) == 6,
      f"{len(pairs_fitting_on_weights(_pw_base, _plain['fleet']))} of {len(_pw_base.pairs)}")
check("  and the fleet can co-reside on full envelopes too, so nothing is being papered over",
      _pw_base.can_coreside)
# 30 GiB of weights each: one shard fits a ~45 GiB card, two cannot. The witness is empty, and that
# is the one case where a refusal to co-reside rests on a figure that cannot shrink. Again through the
# checkpoint rather than the file, for the reason above.
_heavy_loaded = _floorload(_base_yaml, hub=fixture_hub(
    {"Qwen/Qwen3-0.6B": synthetic_manifest("Qwen/Qwen3-0.6B", resident_gib=30.0)}))
_hb = _bl_w(_heavy_loaded["fleet"])
check("30 GiB tenants: no pair fits one GPU even on weights alone",
      pairs_fitting_on_weights(_hb, _heavy_loaded["fleet"]) == () and not _hb.can_coreside,
      f"{len(pairs_fitting_on_weights(_hb, _heavy_loaded['fleet']))} pairs, "
      f"can_coreside={_hb.can_coreside}")
check("the witness is silent, not affirmative, when there is no fleet to ask about",
      pairs_fitting_on_weights(_pw_base, None) == ())

# The shape comes off each tenant, and a fleet file that states it fleet-wide is refused rather
# than broadcast. Broadcasting is the failure that reads correctly in the record afterwards: the
# key would name one pair, every tenant would have been driven at it, and nothing would say the
# mix the fleet asked for was never offered.
with tempfile.TemporaryDirectory() as _td:
    _mixed = Path(_td) / "mixed.yaml"
    _mixed.write_text(_src.replace(
        "     isl_tokens: 512, osl_tokens: 128,\n", "     isl_tokens: 2048, osl_tokens: 64,\n",
        1))
    _mixed_loaded = _floorload(_mixed)
    check("a fleet may state a different shape per tenant, and the key records the mix",
          _mixed_loaded["fleet"].sequence_shapes["m1"] == (2048, 64, 0, 0)
          and _mixed_loaded["key"].hash != _plain["key"].hash,
          str(_mixed_loaded["fleet"].sequence_shapes))

    _legacy = Path(_td) / "legacy.yaml"
    _legacy.write_text(_src.replace("workload:\n", "workload:\n  isl_tokens: 512\n"))
    try:
        _floorload(_legacy)
        check("a fleet-wide workload shape is refused, not broadcast to every tenant", False)
    except fleetfile.FleetFileError:
        check("a fleet-wide workload shape is refused, not broadcast to every tenant", True)

    _missing = Path(_td) / "missing.yaml"
    _missing.write_text(_src.replace("     isl_tokens: 512, osl_tokens: 128,\n", "", 1))
    try:
        _floorload(_missing)
        check("a tenant with no shape is refused rather than given a default", False)
    except fleetfile.FleetFileError:
        check("a tenant with no shape is refused rather than given a default", True)

    # `max_nodes` is a budget, not part of what a group is, so it moves no key: a group is recalled
    # as a fleet of exactly its tenants on one node, whatever budget it was split under.
    _two = Path(_td) / "two-nodes.yaml"
    _two.write_text(_src.replace("  instance_type: g6e.48xlarge\n",
                                 "  instance_type: g6e.48xlarge\n  max_nodes: 2\n", 1))
    _two_loaded = _floorload(_two)
    check("hardware.max_nodes loads, and leaves the fleet's validity key as it was",
          _two_loaded["max_nodes"] == 2 and _plain["max_nodes"] == 1
          and _two_loaded["key"].hash == _plain["key"].hash, _two_loaded["key"].hash)
    for _bad in ("0", "true", "1.5", "two"):
        _badp = Path(_td) / "bad-nodes.yaml"
        _badp.write_text(_src.replace("  instance_type: g6e.48xlarge\n",
                                      f"  instance_type: g6e.48xlarge\n  max_nodes: {_bad}\n", 1))
        try:
            _floorload(_badp)
            check(f"  and max_nodes: {_bad} is refused", False)
        except fleetfile.FleetFileError as _exc:
            check(f"  and max_nodes: {_bad} is refused", "max_nodes" in str(_exc), str(_exc)[:100])

    # --- the image payload, and the engine flags it has to agree with -----------------------
    # The defect this closes was silent and shipped: a fleet file configured for 24 images per
    # prompt, priced for the encoder budget that reserves, driven with pure text. Nothing in the
    # record said so, because nothing in the record knew what the driver sent.
    _IMG = "     isl_tokens: 512, osl_tokens: 128,\n"

    def _with_tenant0(body: str, path: str):
        p = Path(_td) / path
        p.write_text(_src.replace(_IMG, _IMG + body, 1))
        return p

    _ok = _with_tenant0(
        '     images: {count: 2, pixels: 65536},\n'
        '     extra_args: [\'--limit-mm-per-prompt={"image": 2, "video": 0}\',\n'
        '                  \'--mm-processor-kwargs={"min_pixels": 1024, "max_pixels": 262144}\'],\n',
        "images-ok.yaml")
    _img_loaded = _floorload(_ok)
    check("a tenant's image payload survives the fleet file into the key and the tenant",
          _img_loaded["fleet"].tenant("m1").image_count == 2
          and _img_loaded["fleet"].sequence_shapes["m1"] == (512, 128, 2, 65536)
          and _img_loaded["key"].hash != _plain["key"].hash,
          str(_img_loaded["fleet"].sequence_shapes["m1"]))
    for _body, _why, _name in (
        ('     images: {count: 4, pixels: 65536},\n'
         '     extra_args: [\'--limit-mm-per-prompt={"image": 2, "video": 0}\'],\n',
         "more images than --limit-mm-per-prompt admits", "over-limit.yaml"),
        ('     images: {count: 1, pixels: 1048576},\n'
         '     extra_args: [\'--limit-mm-per-prompt={"image": 1, "video": 0}\',\n'
         '                  \'--mm-processor-kwargs={"max_pixels": 262144}\'],\n',
         "images bigger than max_pixels, which the engine resizes silently", "over-pixels.yaml"),
        ('     images: {count: 1, pixels: 1024},\n'
         '     extra_args: [\'--limit-mm-per-prompt={"image": 1, "video": 0}\',\n'
         '                  \'--mm-processor-kwargs={"min_pixels": 262144}\'],\n',
         "images smaller than min_pixels, which the engine scales up", "under-pixels.yaml"),
        ('     extra_args: [\'--limit-mm-per-prompt={"image": 24, "video": 0}\'],\n',
         "a multimodal engine with no images stated -- the defect this closes", "text-only.yaml"),
        ('     images: {count: 1, pixels: 0},\n'
         '     extra_args: [\'--limit-mm-per-prompt={"image": 1, "video": 0}\'],\n',
         "an image count with no size", "half-spec.yaml"),
        ('     images: {count: 1, pixles: 65536},\n'
         '     extra_args: [\'--limit-mm-per-prompt={"image": 1, "video": 0}\'],\n',
         "a misspelled key, which would otherwise leave the payload empty", "typo.yaml"),
        ('     extra_args: [\'--limit-mm-per-prompt={"image": 0, "video": 2}\'],\n',
         "video, which the harness cannot send at all", "video.yaml"),
    ):
        try:
            _floorload(_with_tenant0(_body, _name))
            check(f"a fleet is refused for {_why}", False)
        except fleetfile.FleetFileError:
            check(f"a fleet is refused for {_why}", True)
    # Both spellings reach vLLM identically, so both have to be read here -- a file using the
    # space-separated form would otherwise look like a text tenant to the loader and pass.
    check("the loader reads --flag=json and --flag json alike",
          fleetfile._engine_flag_json(("--limit-mm-per-prompt", '{"image": 3}'), "--limit-mm-per-prompt")
          == fleetfile._engine_flag_json(('--limit-mm-per-prompt={"image": 3}',),
                                         "--limit-mm-per-prompt") == {"image": 3})

# --- profiles: minimal TP, measured solo, and what an unmeasured degree costs (impl §0) ---
# The floor above is arithmetic and is a LOWER bound. This is the other half: the smallest degree
# that actually meets the tenant's target, which only a solo measurement settles. Solo at the full
# envelope is the best case for a (model, tp, rate, shape), so a miss there rules the degree out for
# every co-resident placement too -- sharing only ever takes SM time away.
_pkey = ProfileKey.from_validity_key(key())
check("a profile key is the environment with the fleet dropped",
      _pkey.canonical() == {"environment": env().canonical()}, json.dumps(_pkey.canonical()))
check("  and the environment is the machine, not the SKU: nothing in it is a string from a file",
      set(_pkey.canonical()["environment"]) == {
          "gpu_product", "gpu_count", "addressable_mib", "power_limit_w", "p2p_topology",
          "engine_image", "sharing_mode", "metrics_version"})
# The whole reason this store exists. TP is per tenant and inside the validity key, so a sweep
# across degrees writes its rows under a different key each time and no single validity key can
# hold the comparison that establishes a minimum.
check("  so two fleets differing only in TP share one profile key, though not one validity key",
      ProfileKey.from_validity_key(key(tensor_parallel={"m1": 4})).hash == _pkey.hash
      and key(tensor_parallel={"m1": 4}).hash != key().hash, _pkey.hash)
check("  while a capped card is a different profile key, because it conditions the measurement",
      ProfileKey.from_validity_key(key(env=env(power_limit_w=300))).hash != _pkey.hash)
check("a rate is quantized into the row's identity exactly as the validity key quantizes it",
      profile_id("Qwen/Qwen3-0.6B", 2, 512, 128, 2.0000000000000004, SERVING)
      == profile_id("Qwen/Qwen3-0.6B", 2, 512, 128, 2.0, SERVING),
      profile_id("Qwen/Qwen3-0.6B", 2, 512, 128, 2.0, SERVING))
check("  and a model id with a slash does not become a subdirectory",
      "/" not in profile_id("Qwen/Qwen3-0.6B", 1, 512, 128, 2.0, SERVING))
# The defect `search-r2hk3z` exposed. Raising a batch cap and re-measuring used to overwrite the
# row taken at the old cap, so `for_model` would hand `minimal_tp` -- and therefore the §0
# baseline -- a number produced by a different engine, with nothing anywhere recording it.
check("  and two rows differing only in the batch cap are two rows, not one",
      profile_id("Qwen/Qwen3-0.6B", 2, 512, 128, 2.0, SERVING)
      != profile_id("Qwen/Qwen3-0.6B", 2, 512, 128, 2.0,
                    dataclasses.replace(SERVING, max_num_seqs=96)))
for _flag, _val in (("max_num_batched_tokens", 8192), ("max_model_len", 8192), ("dtype", "float16"),
                    ("extra_args", ("--enable-prefix-caching",))):
    check(f"  and so does a change of {_flag}, which reaches the same command line",
          profile_id("Qwen/Qwen3-0.6B", 2, 512, 128, 2.0, SERVING)
          != profile_id("Qwen/Qwen3-0.6B", 2, 512, 128, 2.0,
                        dataclasses.replace(SERVING, **{_flag: _val})))
check("  while the envelope is deliberately NOT in the row's identity: the search varies it",
      "gpu_memory_utilization" not in json.dumps(SERVING.canonical()))


_POOL = 600_000                  # the KV pool a fixture row and its co-resident run share
_RSRC = "measured@tpN"           # ...and the reserve both were given


def _solo(tp: int, itl: float | None, **kw):
    return SoloProfile(model_id="Qwen/Qwen3-0.6B", tp=tp, isl_tokens=512, osl_tokens=128,
                       rate_hz=16.0, envelope_gib=40.0, serving=kw.pop("serving", SERVING),
                       itl_mean_ms=itl,
                       # The conditions the row was DRIVEN under, which is what makes a ratio against
                       # it a comparison rather than a subtraction. Defaulted to match `_oatt`'s so
                       # the ordinary fixture is a controlled comparison and the interesting cases
                       # are the ones that override it.
                       kv_tokens=kw.pop("kv_tokens", _POOL),
                       reserve_source=kw.pop("reserve_source", _RSRC),
                       delivery_passed=kw.pop("delivery_passed", True), **kw)


check("a row says what engine it was measured under, and its identity carries the digest",
      _solo(2, 5.0).to_dict()["serving"] == SERVING.canonical()
      and _solo(2, 5.0).id.endswith(f"-cfg{SERVING.digest}"))
# The same refusal as the key's, and for a sharper reason: a row read back with guessed flags sets
# minimal TP, and minimal TP is the §0 baseline. Guessing would move the yardstick.
try:
    _d = _solo(2, 5.0).to_dict()
    del _d["serving"]
    SoloProfile.parse(_d)
    check("a row that does not say what engine produced it is refused, not defaulted", False)
except ValueError:
    check("a row that does not say what engine produced it is refused, not defaulted", True)

_target = TenantTemplate(name="m1", model_id="Qwen/Qwen3-0.6B", model_path="/p", served_name="m1",
                         tp=4, rate_hz=16.0, isl_tokens=512, osl_tokens=128, mem=QWEN,
                         slo=TenantSlo(itl_mean_ms=8.0))
with tempfile.TemporaryDirectory() as _td:
    _ps = ProfileStore(_td, _pkey)
    _ps.write_key()
    check("a profile key.json round-trips, so the directory name stays attributable",
          json.loads((_ps.key_dir / "key.json").read_text())["hash"] == _pkey.hash)
    _cands = candidate_tp_degrees(8)

    # Nothing measured yet. The one thing that must not happen here is an integer coming back:
    # falling back to the stated tp, or to the floor, returns a number that reads downstream as a
    # measured minimum, which is the §0 mistake with a different provenance.
    _r = minimal_tp(_ps, _target, floor_tp=2, candidates=_cands)
    check("an unprofiled model yields no TP degree at all, rather than the stated one",
          _r.status == UNPROFILED and _r.tp is None, f"{_r.status}, tp={_r.tp}")
    check("  and it names the degrees nobody has measured, which is the work to do",
          _r.missing == (2, 4, 8) and all(c == NOT_PROFILED for _, c, _ in _r.below),
          str(_r.as_dict()["below"]))

    _ps.write(_solo(2, 11.4))
    _ps.write(_solo(4, 6.2, ttft_p95_ms=310.0, activation_gib=1.9, reserve_gib=1.48,
                    run_id="search-001-profile-02"))
    _r = minimal_tp(_ps, _target, floor_tp=2, candidates=_cands)
    check("minimal TP is the smallest MEASURED degree that meets the tenant's own target",
          _r.status == RESOLVED and _r.tp == 4, f"{_r.status}, tp={_r.tp}")
    check("  and it is certain only because every narrower degree was measured and missed",
          _r.certain and _r.below == (
              (2, MISSED, "missed: 11.400 ms ITL solo over a 8.000 ms ceiling"),), str(_r.below))
    check("  with the solo number carried through as the reference a shared run is judged against",
          _r.solo_itl_ms == 6.2 and _r.as_dict()["solo_itl_ms"] == 6.2)
    # The candidate order comes from the LIST, never from sorting it. Both 4 and 2 have rows here and
    # only 4 meets the target, so the answer cannot distinguish the two orders -- what this pins is
    # that the walk consults the list's own sequence. It matters because a degree is on its way to
    # being a shape (a TP width and a PP depth, so a walk can widen onto a second node) and a shape
    # has no total order for `sorted` to find: only the caller knows which candidate is next-cheapest.
    check("  and the walk takes its order from the candidate list rather than sorting it, because a "
          "degree will not always be a number",
          minimal_tp(_ps, _target, floor_tp=2, candidates=(4, 2)).tp == 4
          and "sorted(" not in inspect.getsource(minimal_tp),
          str(minimal_tp(_ps, _target, floor_tp=2, candidates=(4, 2)).as_dict()["below"]))
    # A miss names the ceiling it actually breached. This is the anchor's real case, seen on a live
    # round: 73.5 ms ITL inside an 80 ms target but 56 s TTFT p95 against 4 s. Reporting that as
    # "missed: 73.536 ms ITL solo" prints a passing number and calls it the failure, and the person
    # reading the line is deciding which knob in the fleet file to turn. The ceilings here are the
    # shape of that case, not its numbers. The rows are written at the tenant's OWN serving digest
    # rather than at this file's `SERVING`, because a 4 s TTFT ceiling at 16 Hz is 64 slots of
    # prefill queue on its own and the derived cap follows it (impl §14.9) -- a row measured at some
    # other cap is a row about a different engine configuration, and the store is right to ignore it.
    with tempfile.TemporaryDirectory() as _ttd:
        _tps = ProfileStore(_ttd, _pkey)
        _both = dataclasses.replace(_target, slo=TenantSlo(itl_mean_ms=30.0, ttft_p95_ms=4000.0))
        _tps.write(_solo(2, 25.5, ttft_p95_ms=56365.0, serving=_both.serving))
        _rt = minimal_tp(_tps, _both, floor_tp=2, candidates=_cands)
        check("  and a miss on TTFT alone says TTFT, not the ITL number that passed",
              _rt.below[0][1] == MISSED
              and _rt.below[0][2] == "missed: 56365 ms TTFT p95 over a 4000 ms ceiling",
              str(_rt.below[0]))
        _tps.write(_solo(4, 35.0, ttft_p95_ms=56365.0, serving=_both.serving))
        _rt2 = minimal_tp(_tps, _both, floor_tp=4, candidates=(4,))
        check("    and a row that breaches both ceilings names both, in one line",
              _rt2.below[0][2] == ("missed: 35.000 ms ITL solo over a 30.000 ms ceiling and "
                                   "56365 ms TTFT p95 over a 4000 ms ceiling"),
              str(_rt2.below[0]))
    # The gap that matters, and the reason `certain` is a field rather than an assumption: with
    # tp2 unmeasured, tp4 is an upper bound on minimal TP, so a baseline built from it may still
    # carry GPUs the fleet does not need.
    (_ps.solo_dir / f"{_solo(2, 11.4).id}.json").unlink()
    _r = minimal_tp(_ps, _target, floor_tp=2, candidates=_cands)
    check("a gap below the answer makes it an upper bound, and says so",
          _r.status == RESOLVED and _r.tp == 4 and not _r.certain
          and _r.missing == (2,), f"certain={_r.certain}, missing={_r.missing}")

    # A suppressed metric is not a fast run and not a slow one. Treating it as a failure would
    # push the tenant to a wider shard count on the strength of a measurement that did not happen.
    _ps.write(_solo(2, None, delivery_passed=False, delivered_fraction=0.71))
    check("a row whose delivery gate failed cannot rule a degree in or out",
          _ps.read("Qwen/Qwen3-0.6B", 2, 512, 128, 16.0, SERVING).meets(_target.slo) is None)
    _r = minimal_tp(_ps, _target, floor_tp=2, candidates=_cands)
    check("  so it reads as unmeasured rather than as a miss",
          _r.tp == 4 and not _r.certain and _r.missing == (2,), str(_r.below))
    check("  and it is not reported as never profiled, because the run needs fixing not repeating",
          _r.below[0][1] == UNUSABLE_NO_LATENCY and "profiled but unusable" in _r.below[0][2],
          str(_r.below[0]))
    # The reason is quoted from the row, never inferred. The anchor's tp8 round is why: the gate
    # stopped it, but `delivered_fraction` came back null -- nothing completed inside the window,
    # so the fraction had no denominator -- and a reader sent to look for a failed gate finds no
    # field in the row that says so. The harness's own sentence is the only account there is.
    _ps.write(_solo(2, None, delivery_passed=None, delivered_fraction=None,
                    notes={"latency_suppressed": "delivered fraction below threshold"}))
    check("    and a suppressed row quotes the harness's own reason rather than inferring one",
          _ps.read("Qwen/Qwen3-0.6B", 2, 512, 128, 16.0, SERVING).unusable_reason
          == (UNUSABLE_NO_LATENCY,
              "no latency in the row: delivered fraction below threshold"),
          str(_ps.read("Qwen/Qwen3-0.6B", 2, 512, 128, 16.0, SERVING).unusable_reason))
    _ps.write(_solo(2, None, delivery_passed=None, delivered_fraction=None))
    check("      and one that never said why says THAT, instead of naming a cause it cannot see",
          "does not say why" in
          _ps.read("Qwen/Qwen3-0.6B", 2, 512, 128, 16.0, SERVING).unusable_reason[1])
    # A failed gate normally reaches the store as an absent latency, since the harness suppresses
    # the number itself. The delivery code is for the row that reports both -- believe the gate.
    _ps.write(_solo(2, 5.1, delivery_passed=False, delivered_fraction=0.71))
    _r = minimal_tp(_ps, _target, floor_tp=2, candidates=_cands)
    check("  and a row carrying a latency its own delivery gate failed is unusable on the gate",
          _r.tp == 4 and _r.below[0][1] == UNUSABLE_DELIVERY, str(_r.below[0]))

    # The harness's verdict is not a judgement this store may second-guess. A row whose latency
    # looks perfectly reasonable is still unusable if the run it came from was not vouched for:
    # `ObservationRecord.usable_as_evidence` has always required a `measured` verdict, and this
    # asks the same question of the same run. GLM-4.5-Air at tp4 is the case that found it -- a
    # sound-looking 55 ms that set the §0 baseline off an `infeasible_no_steady_state` run.
    _ps.write(_solo(2, 5.1, notes={"verdict": VERDICT_NO_STEADY_STATE}))
    check("a row the harness did not vouch for is not evidence, however good its latency looks",
          not _ps.read("Qwen/Qwen3-0.6B", 2, 512, 128, 16.0, SERVING).usable,
          "5.1 ms would have met the 8 ms target and set the minimum at tp2")
    _r = minimal_tp(_ps, _target, floor_tp=2, candidates=_cands)
    check("  so it neither resolves the minimum at that degree nor counts as a miss above it",
          _r.status == RESOLVED and _r.tp == 4 and not _r.certain and _r.missing == (2,),
          f"{_r.status}, tp={_r.tp}, certain={_r.certain}, missing={_r.missing}")
    check("  and the reason given is the verdict itself, named in the report",
          _r.below[0][1] == UNUSABLE_VERDICT and VERDICT_NO_STEADY_STATE in _r.below[0][2],
          str(_r.as_dict()["below"]))
    # With every narrower degree unvouched there is no measured degree at all, so the honest answer
    # is that nothing is settled -- not the narrowest row that happens to sit in the store.
    _ps.write(_solo(4, 6.2, ttft_p95_ms=310.0, notes={"verdict": VERDICT_NO_STEADY_STATE}))
    _r = minimal_tp(_ps, _target, floor_tp=2, candidates=_cands)
    check("  and a sweep where nothing was vouched for yields no degree, not the narrowest row",
          _r.status == UNPROFILED and _r.tp is None and _r.missing == (2, 4, 8),
          f"{_r.status}, tp={_r.tp}, missing={_r.missing}")
    _ps.write(_solo(4, 6.2, ttft_p95_ms=310.0, activation_gib=1.9, reserve_gib=1.48,
                    run_id="search-001-profile-02", notes={"verdict": VERDICT_MEASURED}))
    check("  while an explicit measured verdict is what a usable row looks like",
          _ps.read("Qwen/Qwen3-0.6B", 4, 512, 128, 16.0, SERVING).usable)

    # Thin sampling is the other ground `usable_as_evidence` names. It is written as a positive
    # flag at row-write time, so absence means the flag did not exist yet, not that the run was
    # thinly sampled -- the opposite of the observation record, where absence is itself suspect.
    _ps.write(_solo(2, 5.1, notes={"verdict": VERDICT_MEASURED, "sampling_suspect": True}))
    _r = minimal_tp(_ps, _target, floor_tp=2, candidates=_cands)
    check("a row sampled too thinly to trust is unchecked for the same reason",
          _r.tp == 4 and not _r.certain and _r.below[0][1] == UNUSABLE_SAMPLING, str(_r.below[0]))
    _ps.write(_solo(2, 11.4, notes={"verdict": VERDICT_MEASURED}))

    # Below the memory floor there is nothing to measure, so a floor of 4 must not report tp2 as
    # a gap -- that would send a run to profile a shard that cannot be loaded.
    check("degrees below the memory floor are not counted as unmeasured",
          minimal_tp(_ps, _target, floor_tp=4, candidates=_cands).certain,
          "tp4 meets the target and is itself the floor")

    _no_target = tmpl("m1", "m1", 4)
    check("a tenant that stated no target has no minimum to find, and none is invented",
          minimal_tp(_ps, _no_target, floor_tp=1, candidates=_cands).status == NO_TARGET)
    _impossible = dataclasses.replace(_target, slo=TenantSlo(itl_mean_ms=1.0))
    _ps.write(_solo(2, 11.4))
    _ps.write(_solo(8, 5.9))
    _r = minimal_tp(_ps, _impossible, floor_tp=2, candidates=_cands)
    check("a target no measured degree meets is reported as that, not as the widest degree",
          _r.status == NONE_MEETS and _r.tp is None and len(_r.below) == 3, str(_r.status))

    # Both ceilings, when both are stated. The tp4 row is the only one carrying a TTFT, so a
    # tenant with a TTFT target cannot be certified by the others even though their ITL passes.
    _ttft = dataclasses.replace(_target, slo=TenantSlo(itl_mean_ms=8.0, ttft_p95_ms=200.0))
    check("a stated TTFT ceiling is applied too, so a row that misses it does not resolve",
          minimal_tp(_ps, _ttft, floor_tp=2, candidates=_cands).status == NONE_MEETS,
          "310 ms p95 against a 200 ms ceiling")

    # A re-profile supersedes, unlike an observation. The measurement behind each row is still
    # append-only under its own validity key, and the row names the run, so nothing is lost.
    _ps.write(_solo(4, 6.9, run_id="search-002-profile-01"))
    check("re-profiling a degree replaces the row rather than accumulating two answers",
          len(_ps.for_model("Qwen/Qwen3-0.6B", 512, 128, 16.0, SERVING)) == 3
          and _ps.read("Qwen/Qwen3-0.6B", 4, 512, 128, 16.0, SERVING).run_id == "search-002-profile-01")
    check("a row for another shape is a different row, not a replacement",
          _ps.read("Qwen/Qwen3-0.6B", 4, 1024, 128, 16.0, SERVING) is None)
    check("the whole row round-trips through JSON, provenance included",
          SoloProfile.parse(json.loads(json.dumps(
              _solo(4, 6.2, gpu_ids=("GPU-0002",), run_id="r").to_dict()))).gpu_ids
          == ("GPU-0002",))

    # Merged, not offered for pasting -- and that reverses what this block used to assert (impl
    # §14.4). The old rule was that the fleet file is the single source of truth for what the solver
    # prices with, so a measured reserve had to be transcribed into it by a person before it could
    # count. What overturned it: a figure only a GPU can produce cannot be in a file a person writes
    # for a second engine, and the *shard count* already arrives this way -- resolved from this store
    # and reported rather than stated. So the reserve is read here and reported in the load log, and
    # the fleet file no longer has a field to hold it.
    _ps.write(_solo(4, 6.2, reserve_gib=1.48))
    check("the measured reserve is offered per TP degree for the loader to price with",
          _ps.reserve_by_tp("Qwen/Qwen3-0.6B") == {4: 1.48},
          str(_ps.reserve_by_tp("Qwen/Qwen3-0.6B")))
    # One degree can hold several rows -- a re-profile replaces its own key, but a different
    # sequence shape at the same degree is a different key -- and the largest reading wins. The
    # direction is the safe one: a larger reserve buys a smaller KV pool, which over-states
    # occupancy rather than risking an OOM at the bottom of a search.
    _ps.write(dataclasses.replace(_solo(4, 6.9, reserve_gib=0.91), isl_tokens=1024,
                                  run_id="search-002-profile-02"))
    check("  and the largest reading wins when a degree has been measured at more than one shape",
          _ps.reserve_by_tp("Qwen/Qwen3-0.6B") == {4: 1.48},
          str(_ps.reserve_by_tp("Qwen/Qwen3-0.6B")))
    # The activation half is not merged because nothing writes it and nothing can: the reserve is
    # resident memory minus the utilisation budget, a difference of two numbers visible from
    # outside, while activation is a slice of the inside of that budget and the engine fills what
    # activation leaves with KV cache. Separating them needs the engine's own start-up report.
    check("  and no reader of this store offers an activation figure, because none is measurable",
          not any("activation" in n for n in dir(ProfileStore)))

    (_ps.solo_dir / "truncated.json").write_text("{not json")
    check("an unreadable row makes its degree look unprofiled rather than crashing the read",
          len(_ps.all()) == 4, f"{len(_ps.all())} readable rows beside the broken one")

# --- three placements at three contention levels -----------------------------
# Written out here rather than constructed, because the generator that used to build them
# (`agent/sweeps.py`) has been deleted along with the rest of the hand-built search: choosing what
# to measure is the reasoner's job now, and a module that decided it for a designed sweep was a
# search rule wearing an experiment's clothes.
#
# The tests below still need placements that differ *structurally* -- the renderer, the loop, the
# store and the UUID join all behave differently at different contention -- and they need to know
# which is which, which driving the solver's own ranking would not tell them. Degrees 0, 1 and 3
# are asserted rather than assumed: if these were mislabelled the loop tests would still pass
# while exercising the wrong thing.
PLACEMENTS = [
    ("disjoint", {"m1": (0,), "r1": (1,), "r2": (2, 3), "t2": (4, 5)}),
    ("paired", {"m1": (0,), "r1": (0,), "r2": (1, 2), "t2": (1, 2)}),
    ("stacked", {"m1": (0,), "r1": (0,), "r2": (0, 1), "t2": (0, 1)}),
]
check("the three placements are degree 0, 1 and n-1 for every tenant",
      [sorted(set(coresidency(a).values())) for _, a in PLACEMENTS] == [[0], [1], [3]],
      " | ".join(f"{lab}: {dict(coresidency(a))}" for lab, a in PLACEMENTS))
check("  and the middle one never puts two shards of one TP group on a GPU, which would give "
      "one group two claims on the same envelope",
      all(len(set(g)) == len(g) for g in dict(PLACEMENTS)["paired"].values()))


class Steps(Policy):
    """Walk a list of `(label, assignment)`, letting the solver pick envelopes for each.

    Test scaffolding, and deliberately not shipped: a policy that walks a predetermined list has
    already decided what to measure. It lives here so the loop can be driven to a known sequence
    of placements without a reasoner and without a cluster. A repeated step is a repeat
    measurement, not a no-op -- list one twice to ask for two observations of it.
    """

    def __init__(self, steps):
        self.steps = [(str(label), dict(a)) for label, a in steps]

    def next(self, ctx):
        if ctx.iteration >= len(self.steps):
            return None
        label, assignment = self.steps[ctx.iteration]
        return Proposal(ctx.solve_pinned(assignment), label,
                        "a step of a fixed list, in a test")

# --- agent: render ------------------------------------------------------------
solver8 = PlacementSolver(INV.node_spec, FLEET.requests,
                          envelope_step_mib=AGENT_ENVELOPE_STEP_MIB)
pinned = {lab: solver8.solve_pinned(a) for lab, a in PLACEMENTS}
check("solve_pinned returns exactly the placement it was asked for",
      all(pinned[lab].placement.assignment == {n: tuple(g) for n, g in a.items()}
          for lab, a in PLACEMENTS))
check("  and still optimizes envelopes, so the three are not handed different budgets",
      len({tuple(sorted(p.placement.envelope_gib.items())) for p in pinned.values()}) > 1,
      " | ".join(f"{lab}: {sorted(p.placement.envelope_gib.values())}"
                 for lab, p in pinned.items()))
try:
    solver8.solve_pinned({"m1": (0,)})
    check("a pin that does not name every tenant is refused", False)
except Infeasible:
    check("a pin that does not name every tenant is refused", True)
# The symmetry breaks are valid only up to GPU relabelling, so a pin that contradicts them
# must not come back INFEASIBLE with nothing to say why.
check("a pin the prefix symmetry break would have forbidden still solves",
      solver8.solve_pinned({"m1": (7,), "r1": (6,), "r2": (4, 5), "t2": (2, 3)}
                           ).placement.assignment["m1"] == (7,))

rel = render_releases(pinned["stacked"].placement, FLEET)
check("rendered values pair every GPU ordinal with the UUID of that ordinal, in shard order",
      all(v["placement"]["gpu_uuids"] == [INV.uuid_for(g) for g in v["placement"]["gpu_indices"]]
          for _, v in rel))
check("  and the release set is exactly the fleet's tenants",
      [n for n, _ in rel] == [t.name for t in FLEET.tenants])
cfg = render_run_config(pinned["stacked"].placement, FLEET, "run-x", "hash-x")


def agree(releases, run_config, prefix="", *, placement=pinned["stacked"].placement,
          total=CARD_GIB):
    """`check_agreement` with the placement's own reserves and this node's own device total.

    A helper rather than four repetitions because the two new arguments are always the same two
    things -- the reserves the placement was solved with and the total it was priced against --
    and a test that passed something else would be testing a placement nobody rendered.
    """
    return check_agreement(releases, run_config, prefix,
                           device_total_gib=total,
                           mps_reserve_gib=placement.mps_reserve_gib)


check("the run config drives the same tenants at the same envelopes as the chart values",
      agree(rel, cfg) is None)
# The shape reaches the harness per tenant and nowhere else. A run-level pair would be refused
# by the harness's own loader, so a regression here fails at deploy time rather than quietly
# driving four tenants at one shape.
check("the run config carries each tenant's shape and no fleet-wide pair",
      {t["name"]: (t["isl_tokens"], t["osl_tokens"], t["image_count"], t["image_pixels"])
       for t in cfg["tenants"]}
      == FLEET.sequence_shapes
      and "isl_tokens" not in cfg and "osl_tokens" not in cfg,
      str({k: v for k, v in cfg.items() if k != "tenants"})[:120])
# Same treatment for the tokenizer, and for a sharper reason than symmetry: it decides what
# `isl_tokens` above actually delivers. One tokenizer for a mixed fleet would make the run config
# name 512 tokens for every tenant while sending 512 to one of them and something else to the
# rest, with nothing in the record distinguishing the two cases.
check("  and each tenant's tokenizer travels with it, with no fleet-wide one left",
      {t["name"]: t["tokenizer_path"] for t in cfg["tenants"]}
      == {t.name: t.model_path for t in FLEET.tenants}
      and "tokenizer_path" not in cfg,
      str(sorted({t["tokenizer_path"] for t in cfg["tenants"]})))
check("  and a tenant that states no tokenizer resolves to its own weights, not to None",
      tmpl("z1", "z-1", 1).tokenizer_path == "/fsx/pretrained-models/Qwen/Qwen3-0.6B")
# Allowed, and the reason it stays a field rather than becoming a derivation: two tenants driven
# with a byte-identical prompt is a legitimate experiment. The record says `isl_exact: false` for
# whichever of them the tokenizer does not belong to, so it is readable rather than hidden.
_borrowed = dataclasses.replace(tmpl("z2", "z-2", 1), tokenizer_path="/fsx/other/tok")
check("  and an explicit tokenizer is honoured, so borrowing one stays sayable",
      _borrowed.tokenizer_path == "/fsx/other/tok")
try:
    FleetSpec(tenants=FLEET.tenants, node=INV, workload=WorkloadSpec(tokenizer_path="/p"))
    check("  and WorkloadSpec no longer accepts one at all", False)
except TypeError:
    check("  and WorkloadSpec no longer accepts one at all", True)
check("  and a record that predates the per-tenant shape reports None, not a substitute",
      run6.sequence_shape("m1") is None)
# The failure this prevents leaves no trace in the record: the harness joins DCGM series to
# tenants by UUID, so a disagreement files every per-GPU signal against the wrong tenant.
bad = [(n, json.loads(json.dumps(v))) for n, v in rel]
bad[0][1]["placement"]["gpu_uuids"] = ["GPU-9999"]
try:
    agree(bad, cfg)
    check("a chart/harness disagreement about GPUs is refused before deploying", False)
except RenderError:
    check("a chart/harness disagreement about GPUs is refused before deploying", True)

# --- the denominator, checked where the numbers become a deployment -----------
# `gpu_memory_utilization` is a fraction of ONE number, and the engine multiplies it by whatever
# the driver reports on the machine rather than by whatever it was computed against. So a fraction
# computed against a different card is undetectable afterwards: vLLM echoes the fraction it was
# handed and its KV pool comes out the size the wrong denominator bought. The whole point of the
# checks below is that this is refused at the render boundary or not at all.
_nameplate = [(n, json.loads(json.dumps(v))) for n, v in rel]
for _n, _v in _nameplate:
    _res = pinned["stacked"].placement.mps_reserve_gib[_n]
    # 48.0: what an L40S says on its spec sheet, against the 44.99 the driver reports. This is
    # the exact mistake -- a remembered nameplate figure -- and it runs in the fatal direction,
    # because the fraction it produces buys more memory than the envelope contains.
    _v["vllm"]["gpu_memory_utilization"] = round(
        (_v["mps"]["memory_limit_gib"] - _res) * HEADROOM_DEFAULT / 48.0, 6)
_said = ""
try:
    agree(_nameplate, cfg)
    check("a utilisation computed against a nameplate device total is refused", False)
except RenderError as _exc:
    _said = str(_exc)
    check("a utilisation computed against a nameplate device total is refused",
          f"{CARD_GIB:.2f}" in _said and "48.00 GiB instead" in _said, _said[:100])
check("  and the refusal names the denominator the chart's fraction implies, not just 'disagrees'",
      "implies a device total" in _said)
# The other direction is a real error too and it is the quiet one: the fraction buys less than the
# envelope holds, every gate passes, and the packing spent memory the engine never took.
_shrunk = [(n, json.loads(json.dumps(v))) for n, v in rel]
for _n, _v in _shrunk:
    _v["vllm"]["gpu_memory_utilization"] = round(
        _v["vllm"]["gpu_memory_utilization"] * 0.9, 6)
try:
    agree(_shrunk, cfg)
    check("  and so is one computed against a total that is too LARGE, which only strands memory",
          False)
except RenderError:
    check("  and so is one computed against a total that is too LARGE, which only strands memory",
          True)
# A rounding difference is not a denominator difference. 6 decimals is what `render_values` writes,
# and on a card this size that is worth ~2e-5 GiB -- refusing it would refuse every real render.
_rounded = [(n, json.loads(json.dumps(v))) for n, v in rel]
for _n, _v in _rounded:
    _v["vllm"]["gpu_memory_utilization"] += 4e-7
check("  while the 6-decimal rounding the chart schema imposes is still accepted",
      agree(_rounded, cfg) is None)
try:
    agree(rel, cfg, total=CARD_GIB / 2)
    check("  and a caller that supplies the wrong device total is refused, not trusted", False)
except RenderError:
    check("  and a caller that supplies the wrong device total is refused, not trusted", True)
try:
    check_agreement(rel, cfg, device_total_gib=CARD_GIB, mps_reserve_gib={})
    check("  and a placement's reserves cannot be omitted, since the fraction needs them", False)
except RenderError:
    check("  and a placement's reserves cannot be omitted, since the fraction needs them", True)
# impl §9 at the render boundary. The solver proved this about the total it was BUILT with; this
# re-proves it against the total the node being deployed to measures, which is the last chance to
# notice that the two are not the same card. Constructed as the honest version of that mistake:
# every fraction below is recomputed against the smaller total, so the utilisation check passes
# and the only thing left wrong is that the envelopes no longer fit on the busiest GPU.
_busiest_mib = max(pinned["stacked"].placement.gpu_load_mib().values())
_small = _busiest_mib / 1024.0 - 0.5
_over = [(n, json.loads(json.dumps(v))) for n, v in rel]
for _n, _v in _over:
    _v["vllm"]["gpu_memory_utilization"] = round(gpu_memory_utilization(
        _v["mps"]["memory_limit_gib"], pinned["stacked"].placement.mps_reserve_gib[_n],
        _small, HEADROOM_DEFAULT), 6)
try:
    agree(_over, cfg, total=_small)
    check("envelopes that oversubscribe one GPU are refused at the render boundary too", False)
except RenderError as _exc:
    check("envelopes that oversubscribe one GPU are refused at the render boundary too",
          "is promised" in str(_exc) and "GPU-" in str(_exc), str(_exc)[:90])
try:
    render_tenant_values(FLEET.tenants[0], INV, (0,), 12.5, 0.27)
    check("a fractional envelope is refused, since the chart enforces integer GiB", False)
except RenderError:
    check("a fractional envelope is refused, since the chart enforces integer GiB", True)

# --- no capacity constant, structurally ---------------------------------------
# The check above refuses a wrong denominator at the boundary. This one refuses the way the wrong
# denominator got there in the first place: a number the size of a card, written into the source
# because somebody knew what card it was. There used to be several -- an L40S "usable" figure and a
# nameplate one -- and every one of them was correct when it was typed and wrong the first time this
# ran on a different machine, silently, in the direction that hands the engine memory the packing
# never priced. So the property asserted here is not "the ones we deleted are gone" but "one cannot
# be added": the shipped source is parsed, and any numeric constant that could BE a device memory
# size is a failure whatever it is named.
#
# The test suite is exempt and has to be: `CARD_GIB` above is a synthetic card, and a suite with no
# card cannot check arithmetic about one. That is also why the scan works on text -- it can be
# pointed at the shipped tree without importing it, and a planted example below proves it is not
# vacuous.
_CAP_WORDS = re.compile(r"USABLE|NAMEPLATE|PHYSICAL_TOTAL|CAPACITY")
# A name that says what it measures. Without a unit suffix the number is read as GiB, which is the
# unit a person who remembers a card remembers it in.
_SIZE_WORDS = re.compile(r"TOTAL|CARD|DEVICE|CAPACITY|USABLE|NAMEPLATE|MEM")
# The window a device memory size lives in. 8 GiB is below anything this stack can serve a model on
# and 10,000 GiB is above any single device, so a constant inside it is a card and a constant
# outside it is a step, a tolerance or a unit conversion. `BYTES_PER_GIB` is outside by design.
_CARD_GIB_LO, _CARD_GIB_HI = 8.0, 10_000.0


def _numeric(node):
    """The value of a constant expression, or None. Folds `1.0 / 1024.0` and `1 << 30`.

    Folding matters: a reintroduced capacity is at least as likely to arrive as `45 * 1024` MiB or
    `48 - 3.01` as it is to arrive as a bare literal, and a scan that only reads literals would
    report the tree clean while the number sat there.
    """
    try:
        value = ast.literal_eval(node)
    except Exception:                                                        # noqa: BLE001
        ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
               ast.Pow: lambda a, b: a ** b, ast.LShift: lambda a, b: a << b}
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = _numeric(node.left), _numeric(node.right)
            if left is None or right is None:
                return None
            try:
                return ops[type(node.op)](left, right)
            except Exception:                                                # noqa: BLE001
                return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = _numeric(node.operand)
            return None if inner is None else -inner
        return None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_gib(name: str, value: float):
    """`value` in GiB if the name says what unit it is in, else None."""
    upper = name.upper()
    for suffix, scale in (("_GIB", 1.0), ("_GB", 1.0), ("_MIB", 1 / 1024), ("_MB", 1 / 1024),
                          ("_BYTES", 1 / 2 ** 30), ("_B", 1 / 2 ** 30)):
        if upper.endswith(suffix):
            return value * scale
    return value if _SIZE_WORDS.search(upper) else None


def capacity_constants(source: str, where: str = "") -> list:
    """Every place `source` writes down something that could be a device memory size.

    Three shapes, because the constant has arrived as all three: a module-level constant, a
    function parameter's default (which is how `gpu_memory_utilization` used to carry one -- a
    default there is a claim about a card made by every caller that forgets to override it), and a
    name whose words admit what it is at any magnitude.
    """
    found = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        value = _numeric(node.value) if node.value is not None else None
        if value is None:
            continue
        for name in names:
            if _CAP_WORDS.search(name.upper()):
                found.append(f"{where}:{node.lineno}: {name} = {value}")
                continue
            gib = _as_gib(name, value)
            if gib is not None and _CARD_GIB_LO <= abs(gib) < _CARD_GIB_HI:
                found.append(f"{where}:{node.lineno}: {name} = {value}")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        positional = args.posonlyargs + args.args
        pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
        for arg, default in pairs:
            if not re.search(r"USABLE|CAPACITY|TOTAL_GIB|TOTAL_MIB|NAMEPLATE", arg.arg.upper()):
                continue
            if _numeric(default) is not None:
                found.append(f"{where}:{node.lineno}: {node.name}({arg.arg}={_numeric(default)})")
    return found


_PKG_ROOT = Path(__file__).resolve().parents[2]
_shipped = sorted(p for p in (_PKG_ROOT / "colocation").rglob("*.py") if "tests" not in p.parts)
_planted = [capacity_constants(t, "planted") for t in (
    "L40S_USABLE_GIB = 44.39\n",
    "PHYSICAL_TOTAL_GIB = 48.0\n",
    "CARD_MIB = 46068\n",
    "DEVICE_TOTAL = 45 * 1.0\n",
    "def util(e, r, device_total_gib=44.39):\n    return e / device_total_gib\n",
)]
check("the scan for a device capacity written into the source catches one however it is written -- "
      "as GiB, as a nameplate, as MiB, as arithmetic, or as a parameter default",
      all(len(hit) == 1 for hit in _planted), f"{len(_planted)} plants, hits {_planted}")
check("  and it reads a whole package rather than one file",
      len(_shipped) > 25 and all(p.suffix == ".py" for p in _shipped), f"{len(_shipped)} modules")
_offenders = [hit for p in _shipped
              for hit in capacity_constants(p.read_text(), str(p.relative_to(_PKG_ROOT)))]
check("  and there is no such constant anywhere in the shipped package: the device total is "
      "measured, filed and read, and there is nowhere to type it", not _offenders,
      "; ".join(_offenders))

# The other half of the same guarantee, and the half a reviewer would have to spot by hand: the only
# thing a utilisation may be multiplied by is a device total, because that is the multiplication
# vLLM itself performs on the machine. Every such product in the package is listed and checked,
# which is what stops a third conversion appearing beside the two that exist.
#
# Four spellings of "the device total" are accepted, and they are the only four. `capacity_gib` is
# the oldest: it is a KEY in model fact sheets already filed on EFS, and the reader of those sheets
# is itself deleted (impl §14.4) while the files are not, so it is a name that cannot be dropped from
# this pattern without making filed evidence unreadable if anything ever reads it again. `total_unc`
# was that total's uncertainty, propagated by `solver/devicelaw.py` -- gone with `onboard` too, and
# left here for the same reason. A product against anything else is the bug
# this whole change was about, and it fails here rather than on a card nobody tested on.
_TOTAL_NAMES = re.compile(r"device_total|total_gib|total_unc|capacity_gib")
_mults = []
for _p in _shipped:
    for _node in ast.walk(ast.parse(_p.read_text())):
        if not (isinstance(_node, ast.BinOp) and isinstance(_node.op, ast.Mult)):
            continue
        _sides = [ast.unparse(_node.left), ast.unparse(_node.right)]
        if any(re.search(r"\butil\b|utilization|utilisation", s) for s in _sides):
            _mults.append((str(_p.relative_to(_PKG_ROOT)), _node.lineno, " * ".join(_sides)))
# The floor was 5 until impl §14.4 deleted `agent/booter.py` and `solver/devicelaw.py`, which between
# them held three of the products -- `envelope_gib_for_util` and the uncertainty propagation. What is
# checked is the same thing: every product that survives is against a device total. The count is a
# floor rather than an equality so that adding one is not a test failure; that it is *two* is stated
# here so a reader can tell a deletion from a regression.
check("every multiplication of a utilisation in the package is by a device total and nothing else",
      len(_mults) >= 2 and all(_TOTAL_NAMES.search(expr) for _, _, expr in _mults),
      f"{len(_mults)} products: " + "; ".join(f"{f}:{n} {e}" for f, n, e in _mults))
# And the function that converts a utilisation to GiB requires the total rather than defaulting it. A
# default there is a claim about a card made on behalf of every caller that forgets to override it,
# which is exactly how the deleted constant used to be reached. There were two, one each way;
# `envelope_gib_for_util` went with `booter.py`, and nothing converts in that direction any more.
_sigs = {}
for _p in _shipped:
    for _node in ast.walk(ast.parse(_p.read_text())):
        if isinstance(_node, ast.FunctionDef) and _node.name in ("gpu_memory_utilization",
                                                                 "envelope_gib_for_util"):
            _args = _node.args.posonlyargs + _node.args.args
            _need = [a.arg for a in _args[len(_args) - len(_node.args.defaults):]]
            _sigs[_node.name] = ("device_total_gib" in [a.arg for a in _args],
                                 "device_total_gib" not in _need)
check("  and the utilisation conversion takes the device total as a REQUIRED "
      "argument, so no caller can inherit a card by forgetting one",
      set(_sigs) == {"gpu_memory_utilization"}
      and all(_sigs.values()) and all(all(v) for v in _sigs.values()), str(_sigs))

# --- agent: the readiness gate classifies, it does not just fail -------------
# Three failures that helm --wait reports identically and that mean different things.
def pod(init=(), containers=(), phase="Running", conditions=()) -> dict:
    return {"metadata": {"name": "p"},
            "status": {"phase": phase, "initContainerStatuses": list(init),
                       "containerStatuses": list(containers), "conditions": list(conditions)}}


check("a failed verification init container reads as verification_failed, not a timeout",
      _classify(pod(init=[{"name": "verify-mps",
                           "state": {"terminated": {"exitCode": 1, "reason": "Error"}}}]))[0]
      == NOT_READY_VERIFY)
check("a crash-looping engine reads as engine_failed -- a feasibility fact, not a retry",
      _classify(pod(containers=[{"name": "vllm",
                                 "state": {"waiting": {"reason": "CrashLoopBackOff"}}}]))[0]
      == NOT_READY_ENGINE)
check("a pod still pulling weights reads as a timeout, which says nothing about the placement",
      _classify(pod(phase="Pending"))[0] == NOT_READY_TIMEOUT)
check("an unschedulable pod is its own state, since no amount of waiting fixes it",
      _classify(pod(phase="Pending",
                    conditions=[{"type": "PodScheduled", "status": "False",
                                 "reason": "Unschedulable"}]))[0] == NOT_READY_SCHEDULE)
check("a Ready pod is Ready", _classify(pod(conditions=[{"type": "Ready", "status": "True"}]))[0]
      == READY)
check("no pod at all is a timeout, not a success",
      _classify(None)[0] == NOT_READY_TIMEOUT)

# --- agent: three bugs the first live sweep found, each a lost measurement ----
# Every check here corresponds to a way the run of 2026-09-03 threw away a completed
# `verdict: measured` record. None of them were reachable offline before that run, which is why
# they are pinned here now.

# 1. The Job template reads HARNESS_RUN_ID from `fieldRef: metadata.name`, which inside a pod is
#    the *pod* name. The agent cannot read back a record filed under a suffix it did not choose.
_c = {"name": "harness", "env": [{"name": "OTHER", "value": "keep"},
                                 {"name": "HARNESS_RUN_ID",
                                  "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}}]}
JobHarness._pin_run_id(_c, "orderings-000-disjoint")
_env = {e["name"]: e for e in _c["env"]}
check("HARNESS_RUN_ID is pinned to the agent's run id, not the pod's name",
      _env["HARNESS_RUN_ID"] == {"name": "HARNESS_RUN_ID", "value": "orderings-000-disjoint"})
check("pinning the run id leaves the template's other env vars alone",
      _env["OTHER"]["value"] == "keep")
_c2 = {"name": "harness"}
JobHarness._pin_run_id(_c2, "r")
check("a container with no env block gets HARNESS_RUN_ID added rather than raising",
      _c2["env"] == [{"name": "HARNESS_RUN_ID", "value": "r"}])

# 2. `kubectl logs` merges stdout and stderr, so the traceback raised by the failing write landed
#    two lines inside the record being dumped for recovery.
_real = json.dumps({"run_id": "x", "verdict": "measured",
                    "per_tenant": {"m1": {"itl_mean_ms": 2.634}}}, indent=2, sort_keys=True)
_lines = _real.split("\n")
_spliced = "\n".join(_lines[:3] + ["Traceback (most recent call last):",
                                   '  File "<frozen runpy>", line 198, in _run_module_as_main']
                     + _lines[3:])
check("a record with a traceback interleaved into it is still recovered",
      _parse_interleaved(_spliced, "x")["per_tenant"]["m1"]["itl_mean_ms"] == 2.634)
check("an uninterleaved record parses unchanged", _parse_interleaved(_real, "x")["verdict"]
      == "measured")
try:
    _parse_interleaved("this was never a record", "x")
    check("unrecoverable log output raises rather than returning a partial record", False)
except MeasureError as exc:
    # The message has to say the measurement may be intact: re-running is the expensive wrong
    # move when the record is sitting in a log.
    check("unrecoverable log output raises rather than returning a partial record",
          "recover it by hand" in str(exc))

# 3. `attach-pvc` runs as root and the harness Job as uid 1000, so a key directory the mirror
#    created with a default umask was one `write_record` could not mkdir into -- discovered only
#    after the measurement window closed.
_mk = EfsMirror("abc123", "/unused", efs_root="/efs/colocation")._mkdirs()
check("the mirror creates the observations dir, not just the key dir",
      "/efs/colocation/observations/abc123/observations" in _mk)
check("the mirror makes the key dir writable by the harness's uid too, with the sticky bit",
      "chmod 1777" in _mk)

# The conclusion goes beside the evidence, but NOT among it. `ObservationStore` globs
# `observations/*.json` and parses whatever it finds as an Observation, so a summary written in
# there is either a parse failure or -- worse -- read as a measurement.
_summary_mirror = EfsMirror("abc123", "/unused", efs_root="/efs/colocation")
_sent: list = []
_summary_mirror._exec_in = lambda cmd, data: _sent.append((cmd, data))    # type: ignore[method-assign]
_remote = _summary_mirror.push_summary("tok42", '{"verdict": "met"}\n')
check("a search's summary is pushed under the key, one file per run token",
      _remote == "/efs/colocation/observations/abc123/searches/tok42.json", _remote)
check("  and not into observations/, which the store parses as evidence",
      "/observations/tok42.json" not in _remote
      and _remote.rsplit("/", 2)[-2] == "searches")
check("  and its directory is created and made writable first, like the key dir",
      "mkdir -p /efs/colocation/observations/abc123/searches" in _sent[0][0]
      and "chmod 1777 /efs/colocation/observations/abc123/searches" in _sent[0][0]
      and _sent[0][1] == '{"verdict": "met"}\n')

# --- agent: the harness follows the node under test into its zone -----------------
# 4. A fourth way to lose a measurement, found on 2026-09-14: the Job template pinned a literal
#    `us-west-2b`, the cluster had moved to ap-northeast-2, and the driver could not be scheduled
#    at all. There is no signal in that -- the Job sits Pending until the measure timeout (90
#    minutes by default) with the engine loaded and the GPUs burning behind it. No AZ is written
#    down anywhere now; the zone is read off the node under test, so it is right by construction on
#    any cluster and identical between runs, which is what keeps `calibrate`'s spread about the
#    fleet rather than about where the driver happened to land.
_jh = JobHarness.__new__(JobHarness)
_jh.template = yaml.safe_load(Path("containers/colocation-harness/deploy/job.yaml").read_text())
_jh.namespace, _jh.image, _jh.kubectl = "ns", "img", "kubectl"
_jh.node_name, _jh._zone = "ip-10-0-0-1.some-region.compute.internal", "some-region-1c"
_sel = _jh._job_manifest("r-000-solo-tp2", "cm")["spec"]["template"]["spec"]["nodeSelector"]
check("the harness Job is pinned to the zone of the node under test",
      _sel["topology.kubernetes.io/zone"] == "some-region-1c", str(_sel))
check("  and the template's own nodepool pin survives that patch",
      _sel["karpenter.sh/nodepool"] == "cpu-compute", str(_sel))
check("the shipped Job template hardcodes no availability zone",
      "topology.kubernetes.io/zone" not in
      (_jh.template["spec"]["template"]["spec"].get("nodeSelector") or {}))
_jh2 = JobHarness.__new__(JobHarness)
_jh2.template = copy.deepcopy(_jh.template)
_jh2.namespace, _jh2.image, _jh2.kubectl = "ns", "img", "kubectl"
_jh2.node_name, _jh2._zone = None, None
check("with no node to follow the driver is left unpinned rather than guessed at",
      "topology.kubernetes.io/zone" not in
      _jh2._job_manifest("r", "cm")["spec"]["template"]["spec"]["nodeSelector"])

# 5. And a fifth, the same day: a solo round's EFS key directory was never created, so the harness
#    -- uid 1000, writing into a root-owned 755 `observations/` -- could not write its record at
#    all. All five rounds ended as Failed Jobs with a PermissionError and survived only because
#    `write_record` dumps the record into the log. `push_key` is what creates that directory as
#    root and chmods it, and it has to happen BEFORE the Job, which is a property of the *order* of
#    two calls and so cannot be checked by calling either one. Read as source, deliberately: a
#    mirror handed to a solo round without its key pushed first is the bug, however it is spelled.
_profile_src = inspect.getsource(agent_main._profile)
check("a solo round's mirror has its key pushed before the round is built",
      "push_key" in _profile_src
      and _profile_src.index("push_key") < _profile_src.index("SearchAgent("))
check("  and the mirror the round is given is that same one, not a second bare mirror",
      "mirror=mirror" in _profile_src and _profile_src.count("EfsMirror(") == 1)

# --- agent: the hardware store on EFS, which is why deleting the table cost nothing ---
# The GPU capacity table lived in the source, so every checkout had it. Filing the same facts in a
# store instead is only an improvement if the store is shared: otherwise a second operator's `plan`
# refuses an instance type a colleague leased last week, and the two boots that measured the card get
# paid for twice. This mirror is that sharing, and it is keyed by NOTHING -- every environment's
# sheets come down, because `HardwareIndex` reads across them.
#
# The pod is stubbed at the three reading methods rather than at the shell, so what is checked is the
# assembly -- which remote paths are read and where each file lands -- and not a shell parser nobody
# would run.
_REMOTE = {
    "hwfacts/k1/key.json": json.dumps(
        HardwareKey(env=env(engine_image="vllm:0.24.0").device()).canonical()),
    "hwfacts/k1/devices/l40s.json": json.dumps(
        HardwareFacts(gpu_product="l40s", total_gib=CARD_GIB, total_source="slope",
                      total_uncertainty_gib=0.01, total_falsifiable=True,
                      driver="580.65.06").to_dict()),
    "hwfacts/k2/devices/l40s.json": json.dumps(
        HardwareFacts(gpu_product="l40s", total_gib=CARD_GIB, total_source="driver",
                      total_uncertainty_gib=0.5).to_dict()),
    "hwfacts/instances/g6e.48xlarge.json": json.dumps(
        InstanceShape(instance_type="g6e.48xlarge", gpu_count=8, gpu_product="l40s",
                      addressable_mib=ADDRESSABLE_MIB, power_limit_w=POWER_LIMIT_W,
                      p2p_topology=TOPOLOGY, node="ip-10-0-0-1").to_dict()),
}


def _stub_remote(mirror):
    """Serve `_REMOTE` as though it were `/efs/colocation`, and record every write."""
    base = "/efs/colocation/"
    mirror.sent = []
    mirror._cat = lambda path: _REMOTE.get(path[len(base):]) if path.startswith(base) else None
    mirror._ls = lambda d: sorted(k.rsplit("/", 1)[1] for k in _REMOTE
                                 if k.startswith(d[len(base):] + "/"))
    mirror._ls_dirs = lambda d: sorted({k[len(d) - len(base) + 1:].split("/")[0]
                                        for k in _REMOTE if k.startswith(d[len(base):] + "/")})
    mirror._exec = lambda script, timeout_s=120.0: mirror.sent.append(script) or ""
    mirror._exec_in = lambda script, data, timeout_s=120.0: (
        mirror.sent.append((script, data)) or "")
    return mirror


with tempfile.TemporaryDirectory() as _td:
    _hm = _stub_remote(HardwareMirror(_td, efs_root="/efs/colocation"))
    _got = _hm.pull()
    check("the hardware mirror pulls every environment's device laws, not just this run's",
          (_got["devices"], _got["shapes"], _got["keys"]) == (2, 1, ["k1", "k2"]), str(_got))
    check("  and `instances/` is not mistaken for an environment, since it is keyed by nothing",
          "instances" not in _got["keys"])
    # The check that says why the mirror exists at all: a desktop that has never held a GPU can now
    # size this instance type, with no table in the source and no node in hand.
    _pulled_hw = hardware_for("g6e.48xlarge", HardwareIndex(_td))
    check("  so a machine that has never captured resolves the instance type offline after a pull",
          (_pulled_hw.gpu_count, _pulled_hw.device_total_gib) == (8, CARD_GIB),
          _pulled_hw.provenance())
    check("  and the two environments' sheets are both there, which is what makes it falsifiable",
          len(HardwareIndex(_td).sheets_for("l40s")) == 2 and _pulled_hw.total_falsifiable)

    # Push. A device law supersedes -- a later slope replaces a driver reading and carries its own
    # boots -- and the directories are created 1777 for `_chmod_dirs`'s reason: `attach-pvc` is root
    # and the harness Job is uid 1000.
    _hm.sent = []
    _remote_path = _hm.push_sheet("k1", "l40s", {"gpu_product": "l40s"})
    check("pushing a device law writes it under the environment that measured it",
          _remote_path == "/efs/colocation/hwfacts/k1/devices/l40s.json", _remote_path)
    check("  and creates the directory writable by both uids first, like every other mirror",
          "chmod 1777" in _hm.sent[0][0] and "/efs/colocation/hwfacts/k1/devices" in _hm.sent[0][0])
    # A shape is the one file here that does NOT supersede. Two nodes disagreeing about how many
    # GPUs an instance type has means one of them was not what it claimed, and the local store
    # already refuses to overwrite it; the shared copy has to refuse for the same reason.
    _same = dict(InstanceShape(instance_type="g6e.48xlarge", gpu_count=8, gpu_product="l40s",
                               addressable_mib=ADDRESSABLE_MIB, power_limit_w=POWER_LIMIT_W,
                               p2p_topology=TOPOLOGY, node="ip-10-0-0-9").to_dict())
    _hm.sent = []
    _hm.push_shape("g6e.48xlarge", _same)
    check("re-pushing a shape that agrees writes nothing, since it is the same fact",
          all(not isinstance(x, tuple) for x in _hm.sent), str(_hm.sent))
    try:
        _hm.push_shape("g6e.48xlarge", dict(_same, gpu_count=4))
        check("  while a shape contradicting EFS is refused, not overwritten", False)
    except MirrorError as exc:
        check("  while a shape contradicting EFS is refused, not overwritten",
              "one of the two nodes was not what it claimed" in str(exc), str(exc)[:70])
    try:
        _hm.push_key("k1", '{"driver": "OTHER", "image": "vllm:0.24.0", "schema_version": 1}')
        check("  and so is a key.json whose hash matches but whose contents do not", False)
    except MirrorError as exc:
        check("  and so is a key.json whose hash matches but whose contents do not",
              "two environments have been filed together" in str(exc), str(exc)[:70])


# --- agent: the loop, end to end, with no cluster ----------------------------
# A synthetic harness whose ITL depends on co-residency degree. The numbers are invented, and
# nothing consumes them as a fit any more: what is being tested is that a measured record reaches
# the store and comes back out as a record the next iteration can be shown (impl §10). The loop
# does not update anything between iterations, because there is nothing to update.
def fake_record(run_config: dict, run_id: str) -> dict:
    placement = {t["name"]: tuple(t["gpu_uuids"]) for t in run_config["tenants"]}
    degrees = coresidency(placement)
    per_tenant = {}
    for t in run_config["tenants"]:
        itl = 2.50 * (1.0 + 0.40 * degrees[t["name"]])
        per_tenant[t["name"]] = {
            "requests_dispatched": 100, "requests_completed": 100, "requests_failed": 0,
            "delivered_fraction": 1.0, "delivered_fraction_threshold": 0.99,
            "itl_mean_ms": itl, "itl_ci_halfwidth_ms": 0.05,
            "ttft_mean_ms": 40.0, "ttft_p95_ms": 60.0,
            "kv_usage_mean": 0.05, "num_requests_running_mean": 2.0,
        }
    # `node` is the harness Job's node -- the load driver's, deliberately NOT the GPU node here.
    # A real record's two node names differ (the driver runs on CPU, Karpenter reclaims it), and any
    # reader that wants the machine the measurement was taken on must read `provenance.gpu_node`.
    # Making them equal in the fixture would let that confusion pass the suite.
    return {"run_id": run_id, "validity_key_hash": run_config["validity_key_hash"],
            "verdict": VERDICT_MEASURED, "node": "cpu-driver-node", "window_s": 60.0,
            "steady_state_reached_s": 30.0, "per_tenant": per_tenant, "per_gpu": {},
            # Shaped like what the harness writes: per tenant, tokenizer included, nothing
            # fleet-wide. `isl_exact` is absent here for the same reason it is absent from a
            # synthesized record -- no prompt was built -- and its absence must not read as false.
            "workload": {"tenants": {t["name"]: {"isl_tokens": t["isl_tokens"],
                                                 "osl_tokens": t["osl_tokens"],
                                                 "tokenizer_path": t.get("tokenizer_path")}
                                     for t in run_config["tenants"]}},
            "configuration": {"tenants": run_config["tenants"]},
            "sampling": {"subsamples_per_sample_mean": 36.0},
            "provenance": dict(run_config.get("provenance") or {})}


# A record written from a rendered run config reads its shape back per tenant, so a retrieved
# run says what each tenant was driven at rather than what the fleet file says today.
_shaped = Observation.parse(fake_record(cfg, "run-x"))
check("a record reports each tenant's own shape back",
      {n: _shaped.sequence_shape(n) for n in _shaped.per_tenant} == FLEET.sequence_shapes,
      str({n: _shaped.sequence_shape(n) for n in _shaped.per_tenant}))


class FakeMirror:
    def __init__(self) -> None:
        self.pushed: list = []

    def push_record(self, run_id: str, record: dict) -> bool:
        self.pushed.append(run_id)
        return True


AGENT_KEY = key(alpha=None, sequence_shapes=FLEET.sequence_shapes,
                tensor_parallel=FLEET.tp_degrees,
                offered_rates_hz={t.name: t.rate_hz for t in FLEET.tenants})
with tempfile.TemporaryDirectory() as td:
    # An EMPTY store under a brand-new key, and no seed store anywhere. This is the situation the
    # agent could not run in at all before Phase 3: the solver needed a measured ITL to size a KV
    # pool, refused to guess one, and so a new load level had to borrow another key's records
    # before it could propose anything -- which is in tension with what a validity key is for.
    # Nothing is borrowed now. The floors come from the fleet's own stated targets, so day one of a
    # new key searches like any other day.
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    deployer, harness, mirror = FakeDeployer(), FakeHarness(fake_record), FakeMirror()
    agent = SearchAgent(FLEET, store, deployer, harness, run_prefix="sweep", mirror=mirror,
                        ready_timeout_s=1.0, measure_timeout_s=1.0)
    check("a brand-new key has no evidence, and that is not an obstacle to solving",
          agent._observations() == []
          and len(PlacementSolver(INV.node_spec, FLEET.requests,
                                  envelope_step_mib=AGENT_ENVELOPE_STEP_MIB).solve(cap=1)) == 1)

    result = agent.run(Steps(PLACEMENTS), teardown_at_end=True)
    check("the loop ran every step it was given",
          len(result.attempts) == len(PLACEMENTS)
          and [a.label for a in result.attempts] == [lab for lab, _ in PLACEMENTS],
          f"{[a.outcome for a in result.attempts]}")
    check("  and every run has its own id, since the id is the Job name and the filename",
          len({a.run_id for a in result.attempts}) == len(result.attempts),
          str([a.run_id for a in result.attempts]))
    check("  and each iteration deployed exactly the fleet's four releases",
          all(len(app) == 4 for app in deployer.applied) and len(deployer.applied) == 3)
    check("  and the fleet was torn down at the end, because it was asked for",
          deployer.live_tenant_releases() == [] and deployer.destroyed)

    report = store.load()
    check("every run left a readable record in the store", len(report.observations) == 3
          and report.ok, f"{len(report.observations)} records, unreadable={report.unreadable}")
    # What the record carries instead of a prediction: the requirement the placement was solved
    # against. A prediction beside a measurement says whether the model was wrong; a floor beside a
    # measurement says whether the fleet was certified, which is the question the design asks.
    check("the record carries the floors and margins the placement was solved to, not a forecast",
          all(o.provenance.get("kv_floor_tokens") and o.provenance.get("kv_margin")
              and o.provenance.get("tightest_margin") and o.provenance.get("gpus_used")
              for o in report.observations),
          str(sorted(report.observations[0].provenance))[:150])
    check("  and carries no prediction keys at all, since nothing predicted anything",
          not any(k in o.provenance for o in report.observations
                  for k in ("predicted_itl_ms", "predicted_regime", "predicted_worst_occupancy",
                            "itl_sources", "all_measured", "surrogate_occupancy")))
    check("  and says where each floor came from, so a certified fleet is auditable",
          all("minimum viable pool" in v
              for v in report.observations[0].provenance["kv_floor_source"].values()),
          str(report.observations[0].provenance["kv_floor_source"])[:120])

    # The records come back as records. Nothing summarizes them, indexes them or fits anything to
    # them (impl §10) -- three runs in, `_observations` hands back the three runs.
    after = agent._observations()
    check("what the loop accumulates is records, handed over as written",
          len(after) == 3 and all(isinstance(o, Observation) for o in after)
          and {o.provenance["policy_label"] for o in after} == {lab for lab, _ in PLACEMENTS},
          f"{len(after)} records")

    # A repeat step is a repeat measurement, not a no-op: between-run spread is a quantity
    # the design measures, so the store must append rather than overwrite.
    again = SearchAgent(FLEET, store, FakeDeployer(), FakeHarness(fake_record),
                        run_prefix="sweep", ready_timeout_s=1.0)
    again.run(Steps([PLACEMENTS[0], PLACEMENTS[0]]), teardown_at_end=False)
    check("a repeated step files a second observation rather than overwriting the first",
          len(store.load().observations) == 5)

# --- agent: a placement that never comes up is still an observation ----------
with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    names = frozenset(t.name for t in FLEET.tenants)
    dead = FakeDeployer(outcomes={names: NOT_READY_ENGINE})
    mirror = FakeMirror()
    agent = SearchAgent(FLEET, store, dead, FakeHarness(fake_record),
                        mirror=mirror, ready_timeout_s=1.0)
    res = agent.run(Steps([PLACEMENTS[2]]), teardown_at_end=False)
    check("a readiness-gate failure is recorded, not skipped -- it is what stops the agent "
          "re-proposing it",
          len(store.load().observations) == 1
          and res.attempts[0].outcome == f"not_ready:{NOT_READY_ENGINE}",
          res.attempts[0].outcome)
    synth = store.load().observations[0]
    check("  and it parses as an observation with the deploy verdict",
          synth.verdict == VERDICT_DEPLOY and synth.notes["readiness_gate"] == NOT_READY_ENGINE)
    check("  and is never citable as evidence about latency, for two independent reasons",
          not synth.usable_as_evidence and synth.sampling_suspect and not synth.measured)
    check("  and is marked as synthesized, so it cannot be read as a measurement",
          synth.provenance.get("synthesized_by") == "agent")
    check("  and is pushed to the shared store, since no harness Job wrote it there",
          mirror.pushed == [res.attempts[0].run_id], str(mirror.pushed))
    check("no harness run was attempted for a fleet that never came up",
          agent.harness.calls == [])

# --- agent: TopRanked, the fallback of impl §11.2 ----------------------------
# What runs when the reasoner cannot answer. It takes the solver's first candidate and forms no
# belief of its own, which is the whole of its contract -- so what it must be checked for is mostly
# what it does NOT do.
#
# The convergence rule that used to be tested here is gone with the policy that had it. "Stop when
# candidate spread falls below the calibrated noise floor" was in impl §11, and taken literally it
# stopped the search on iteration 1: with only co-residency 0 in the store the predictor gave every
# candidate the same ITL, so the spread was zero, and zero from genuine similarity is the opposite
# of zero from ignorance. The guard that told them apart was a rule about when to keep measuring,
# which is the decision the reasoner exists to make. Nothing now stops a search on predicted
# spread; only the reasoner and the budget stop it.
class _Cand:
    """A `Candidate` as far as the policies and the brief are concerned.

    There is no latency to stub, because a `Candidate` carries none. What is read off one is its
    assignment (which is its identity), the KV it gives each tenant against that tenant's floor,
    and how many GPUs it uses -- so that is all this has.

    With no assignment given, each stub gets its own GPU, counted, so two of them are two
    placements. This used to read `id(self) % 8`, which collides often enough to make "a candidate
    already attempted is skipped" pass or fail depending on where the allocator put the objects.
    """

    _next_gpu = 0

    def __init__(self, assignment: dict | None = None, margin: float = 3.0,
                 floor: int = 10_240) -> None:
        if assignment is None:
            gpu, _Cand._next_gpu = _Cand._next_gpu, _Cand._next_gpu + 1
            assignment = {"a": (gpu,)}
        assignment = {n: tuple(sorted(g)) for n, g in assignment.items()}
        names = sorted(assignment)
        self.placement = type("P", (), {
            "assignment": assignment,
            "envelope_gib": {n: 22.0 for n in names},
            "kv_tokens": {n: int(floor * margin) for n in names},
            "util": {n: 0.5 for n in names},
            "mps_reserve_gib": {n: 1.0 for n in names},
            "reserve_source": {n: "default(tp1)" for n in names},
            "gpus_used": len({g for gs in assignment.values() for g in gs})})()
        self.margin = {n: margin for n in names}
        self.floor_tokens = {n: floor for n in names}
        self.floor_detail = {n: f"minimum viable pool {floor} tokens" for n in names}
        self.coresidency = coresidency(assignment)
        self.gpus_used = self.placement.gpus_used
        self.min_margin = margin
        self.tightest_tenant = names[0]


def offered(cands, exhausted: bool = False, why: str = "the cap of 3 was reached") -> FeasibleSet:
    """Wrap stubs in a real `FeasibleSet`, because the flag is part of the interface.

    A bare list would work in Python and hide a contract: both policies read `exhausted` and
    `stopped_because` off what `solve` returns, and a test that passed a list would not notice if
    one of them stopped doing so.
    """
    return FeasibleSet(candidates=tuple(cands), exhausted=exhausted, cap=3,
                       stopped_because=("every distinct assignment has been enumerated"
                                        if exhausted else why))


CAL = Calibration.parse({"between_run_itl_halfwidth_ms": {"coresident=0": 0.20},
                         "repeats": 3, "notes": {}})
tr = TopRanked(top_k=3, max_iterations=12)
ranked = [_Cand(margin=15.8), _Cand(margin=2.1)]
prop = tr.next(Context(fleet=FLEET, calibration=CAL, solve=lambda k: offered(ranked)))
check("TopRanked takes the solver's first candidate, whatever it is",
      prop is not None and prop.candidate is ranked[0])
check("  and claims nothing about it, since it applied no judgement",
      "no judgement" in prop.reason and prop.provenance == {})

# The rule the deleted policy had, checked to be absent: two candidates a noise floor apart used to
# end the search. A candidate set that looks converged is no longer a reason to stop -- and it never
# was one, since what a set of candidates spans is a fact about envelopes and not about latency.
tight = [_Cand(margin=3.00), _Cand(margin=3.01)]
check("a candidate set whose members barely differ does not stop the search",
      tr.next(Context(fleet=FLEET, calibration=CAL,
                      solve=lambda k: offered(tight))) is not None)
check("  and nothing is preferred or avoided for how it was priced; rank decides",
      tr.next(Context(fleet=FLEET, calibration=CAL,
                      solve=lambda k: offered([tight[1], tight[0]]))
              ).candidate is tight[1])

# Bookkeeping, not a search rule: falling back twice in a row must not re-measure one placement
# until the budget is gone.
tried = [Attempt("r0", "x", tight[0], None, "measured")]
prop = tr.next(Context(fleet=FLEET, history=tried, solve=lambda k: offered(tight)))
check("a candidate already attempted in this run is skipped for the next one",
      prop is not None and prop.candidate is tight[1])
tried += [Attempt("r1", "x", tight[1], None, "measured")]
check("  and an exhausted candidate set stops the fallback rather than looping",
      tr.next(Context(fleet=FLEET, history=tried,
                      solve=lambda k: offered(tight, exhausted=True))) is None)
check("the budget stops it too, so a run whose reasoner went away still terminates",
      TopRanked(top_k=3, max_iterations=2).next(
          Context(fleet=FLEET, iteration=2, solve=lambda k: offered(tight))) is None)

# --- agent: fleet files parse, and key distinctly ----------------------------
# The fixtures, not `fleets/`. The top of `fleets/` is where a USER writes their own fleet and ships
# empty on purpose (a fleet is the user's to state); the examples we maintain sit under
# `fleets/examples/<instance type>/`. So the loader is tested against inputs that live with the tests,
# and everything under `fleets/` is checked separately below -- ours and whatever a user drops there
# must both still load.
FLEETS = Path(__file__).resolve().parent / "fixtures" / "fleets"
loaded = {p.name: _floorload(p) for p in sorted(FLEETS.glob("*.yaml"))}
check("every fleet fixture loads with no cluster", len(loaded) >= 2, str(sorted(loaded)))

_user_fleets = Path(__file__).resolve().parents[3] / "fleets"
# Recursive, and named by the path under `fleets/` rather than by file name: the examples are nested a
# directory deep per instance type, and a non-recursive glob would quietly check none of them.
_user_yaml = sorted(_user_fleets.rglob("*.yaml"))
_user_rel = {p: p.relative_to(_user_fleets).as_posix() for p in _user_yaml}
# Not `>= 1`: a fleets/ holding nothing but the examples is the shipped state, and a user who has
# deleted those must not fail the suite either. This asserts only that nothing unloadable is sitting
# there, which is the check that stays useful once it is in use.
#
# `allow_unresolved_tp=True` because a user fleet is entitled to omit `tp:` -- that is the §0
# recommendation -- and this suite runs with no cluster and so no profile store to resolve it from.
# It tolerates exactly one unknown, the unmeasured degree, and still enforces every other rule in
# the loader: a missing SLO, an unknown instance type, a restated validity-key field, a model
# without weights. Those are the errors a file sitting here would otherwise hide until a run.
#
# What this suite holds is two filed shapes and the hub fixtures, so a fleet for an instance type
# nobody has leased, or a model nobody has fixtured, cannot be loaded here -- and inventing a shape
# for it would put a device figure in source. Such a file is held to `check_doc`, every rule a file
# can break by itself, and named on a NOT CHECKED line with what was missing. It is not a pass: the
# rest of it is checked by `plan`, against a store that has leased one.
from .snapshot import facts_missing as _facts_missing          # noqa: E402

_user_docs = {p: fleetfile.read_doc(p) for p in _user_yaml}
_user_bad = {}
for _p, _doc in _user_docs.items():
    try:
        fleetfile.check_doc(_doc)
    except fleetfile.FleetFileError as _exc:
        _user_bad[_user_rel[_p]] = str(_exc)[:200]
check("every file in fleets/ passes the rules a fleet file can break by itself",
      not _user_bad, str(_user_bad) if _user_bad else f"{len(_user_yaml)} file(s)")
_user_partial = {_user_rel[p]: m for p, d in _user_docs.items()
                 if _user_rel[p] not in _user_bad and (m := _facts_missing(d, HW_INDEX, HUB))}
# The partial check has to bite, or it is a skip with a nicer name: a p5 fleet nobody here can load
# still has its stated `tp:` refused, with no shape and no hub facts behind it.
_p5_doc = copy.deepcopy(next(d for p, d in _user_docs.items() if "p5.48xlarge" in _user_rel[p])) \
    if any("p5.48xlarge" in r for r in _user_rel.values()) else None
if _p5_doc is not None:
    _p5_doc["tenants"][0]["tp"] = 1
    try:
        fleetfile.check_doc(_p5_doc)
        check("  and a fleet for hardware this suite has no shape for is still refused a stated tp:",
              False)
    except fleetfile.FleetFileError as _exc:
        check("  and a fleet for hardware this suite has no shape for is still refused a stated tp:",
              "states tp:" in str(_exc), str(_exc)[:120])
for _n, _m in sorted(_user_partial.items()):
    print(f"NOT CHECKED here: {_n} -- {'; '.join(_m)}. Plan it against a store that has leased one.")
_user_loaded = {_user_rel[p]: fleetfile.load(p, index=HW_INDEX, hub=HUB, allow_unresolved_tp=True)
                for p in _user_yaml if _user_rel[p] not in _user_partial | _user_bad.keys()}
check("  and every one this suite has the shape and the hub facts for loads in full",
      len(_user_loaded) == len(_user_yaml) - len(_user_partial) - len(_user_bad),
      f"{len(_user_loaded)} loaded, {len(_user_partial)} not checked here")
_split_nodes = {}
for _n, _d in _user_loaded.items():
    _unres = fleetfile.unresolved_tp(_d)
    if _unres:
        # Said, not hidden: this file's degrees are memory floors here, so the placement checked
        # below is the one that exists IF the floor is enough. `plan` prints the same caveat.
        check(f"  {_n}: states no tp for {', '.join(_unres)}, so its degrees are floors here",
              all(t.tp >= 1 for t in _d["fleet"].tenants))
    if _d["max_nodes"] > 1:
        # A fleet allowed more than one node is hostable when it splits; each group is then placed
        # by the same solver as the check below.
        _sp = _split(_d["fleet"].node.node_spec, _d["fleet"].requests, _d["max_nodes"], cap=1,
                     max_seconds=20.0, envelope_step_mib=AGENT_ENVELOPE_STEP_MIB).splits
        _split_nodes[_n] = _sp[0].nodes if _sp else 0
        check(f"  {_n}: splits over its {_d['max_nodes']} nodes at the degrees it loads with",
              bool(_sp), f"best on {_split_nodes[_n]} node(s)")
        continue
    check(f"  {_n}: has at least one hostable placement at the degrees it loads with",
          len(PlacementSolver(_d["fleet"].node.node_spec, _d["fleet"].requests,
                              envelope_step_mib=AGENT_ENVELOPE_STEP_MIB
                              ).solve(cap=1, max_seconds=20.0)) == 1)
check("  and at least one shipped fleet's best split needs two nodes, so a split is exercised",
      any(n > 1 for n in _split_nodes.values()), str(_split_nodes))

# And `search` on that fleet, with each group's search replaced by one that writes the summary a
# real one would: the wiring from split to group fleet file to verdict, with no cluster.
_mn_name = next(n for n, k in _split_nodes.items() if k > 1)
_mn_calls = []


def _fake_group_main(argv):
    fleet_path = Path(argv[argv.index("--fleet") + 1])
    _mn_calls.append(argv)
    names = [t["name"] for t in fleetfile.read_doc(fleet_path)["tenants"]]
    n, summary = len(_mn_calls), Path(argv[argv.index("--summary") + 1])
    summary.write_text(json.dumps({"verdict": "met", "run_token": f"r{n}", "tenants": names,
                                   "validity_key_hash": f"k{n}"}))
    # Where a real group's search exports itself (`ObservationStore.exports_dir`): the least of a
    # version 1 document the split export reads, one tenant per slot.
    exports = summary.parents[2] / "observations" / f"k{n}" / "exports"
    exports.mkdir(parents=True)
    (exports / f"r{n}.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "generated_at": "2026-10-08T00:00:00+00:00",
        "fleet": {"path": str(fleet_path), "instance_type": "g6e.48xlarge",
                  "devices_per_node": 8, "validity_key_hash": f"k{n}"},
        "tenants": {m: {"shards": 1} for m in names},
        "baseline": {"devices": len(names)},
        "strict": {"devices": n, "slots": [{"residents": [{"tenant": m, "shard": 0}]}
                                           for m in names]}}))
    return 0


_real_main = agent_main.main
with tempfile.TemporaryDirectory() as _mn_root:
    agent_main.main = _fake_group_main
    try:
        _mn_args = SimpleNamespace(fleet=str(_user_fleets / _mn_name), store=_mn_root, top_k=1,
                                   solver_seconds=20.0, group_budget=4, summary=None,
                                   mode="search", split_chooser="drive", export_to=None)
        with contextlib.redirect_stdout(io.StringIO()):
            _mn_rc = agent_main._search_splits(
                _mn_args, ["search", "--fleet", _mn_args.fleet, "--budget", "2"],
                _user_loaded[_mn_name])
    finally:
        agent_main.main = _real_main
    _mn_out = json.loads(next(Path(_mn_root).rglob("split.*.json")).read_text())
    _mn_groups = [fleetfile.read_doc(Path(c[c.index("--fleet") + 1])) for c in _mn_calls]
    _mn_text = Path(_mn_out["export"]).read_text() if _mn_out.get("export") else ""
check(f"`search` on {_mn_name} searches each group of its best split as a fleet of its own",
      _mn_rc == 0 and len(_mn_calls) == 2 and all("--budget" in c for c in _mn_calls)
      and all("max_nodes" not in d["hardware"] for d in _mn_groups)
      and sorted(t["name"] for d in _mn_groups for t in d["tenants"])
      == sorted(t.name for t in _user_loaded[_mn_name]["fleet"].tenants),
      _mn_out["stopped_because"])
check("  and records which split it was and each group's run",
      len(_mn_out["split"]) == 2 and [g["run_token"] for g in _mn_out["groups"]] == ["r1", "r2"]
      and all(g["verdict"] == "met" for g in _mn_out["groups"]))
_mn_v2 = yaml.safe_load(_mn_text) if _mn_text else {}
_mn_nodes = exportmod.nodes_of(_mn_v2) if _mn_v2 else []
check("  and exports the met split as schema_version 2: one node per group, each its own export",
      _mn_v2.get("schema_version") == 2 and len(_mn_nodes) == 2
      and sorted(n for d in _mn_nodes for n in d["tenants"])
      == sorted(t.name for t in _user_loaded[_mn_name]["fleet"].tenants)
      and sorted(Path(d["export"]).name for d in _mn_nodes) == ["r1.yaml", "r2.yaml"]
      and _mn_v2["split"]["devices"] == 3 and _mn_v2["fleet"]["max_nodes"] == 2
      and "schema_version" not in _mn_nodes[0],
      json.dumps(_mn_v2.get("split")))
check("  and its rows are one line again after the round trip through each group's file",
      "residents: [{" in _mn_text, _mn_text[-300:])
_v1 = {"schema_version": 1, "tenants": {}}
check("a version 1 export reads as a single node, itself",
      exportmod.nodes_of(_v1) == [_v1])
for _what, _bad, _expect in (
        ("an unknown schema_version", {"schema_version": 3}, "schema_version 3"),
        ("a tenant on two nodes", None, "on two nodes"),
        ("two instance types", None, "one instance type")):
    try:
        if _bad is not None:
            exportmod.nodes_of(_bad)
        else:
            _n1 = {"schema_version": 1, "fleet": {"instance_type": "g6e.48xlarge",
                                                   "devices_per_node": 8},
                   "tenants": {"a": {}}, "baseline": {"devices": 1}, "strict": None}
            _n2 = copy.deepcopy(_n1)
            if "two nodes" not in _expect:
                _n2["tenants"], _n2["fleet"]["instance_type"] = {"b": {}}, "g6e.12xlarge"
            exportmod.build_split(fleet_path=_user_fleets / _mn_name, max_nodes=2,
                                  verdict="met", chooser="drive", stopped_because="",
                                  nodes=[("x", _n1), ("y", _n2)])
        check(f"  and {_what} is refused", False)
    except exportmod.ExportError as _exc:
        check(f"  and {_what} is refused", _expect in str(_exc), str(_exc))

# The reasoner's path through the same wiring: every reply unusable, so both groups are the
# fallback's, and its submission after them is checked against the runs it did not choose.
_mn_calls.clear()
_real_reasoner = agent_main._split_reasoner
with tempfile.TemporaryDirectory() as _mn_root:
    agent_main.main = _fake_group_main
    _mn_replay = ReplayReasoner(["?"] * 6 + [
        '{"tool": "submit", "args": {"split": "S1", "verdict": "met", "rationale": "met"}}'])
    agent_main._split_reasoner = lambda args: _mn_replay
    try:
        _mn_args.store, _mn_args.split_chooser = _mn_root, "reasoner"
        with contextlib.redirect_stdout(io.StringIO()):
            _mn_rc = agent_main._search_splits(
                _mn_args, ["search", "--fleet", _mn_args.fleet, "--budget", "2"],
                _user_loaded[_mn_name])
    finally:
        agent_main.main = _real_main
        agent_main._split_reasoner = _real_reasoner
    _mn_out = json.loads(next(Path(_mn_root).rglob("split.*.json")).read_text())
check(f"`search` on {_mn_name} with the reasoner choosing: the fallback's groups are marked",
      _mn_rc == 0 and len(_mn_calls) == 2 and _mn_out["verdict"] == "met"
      and _mn_out["chooser"] == "reasoner" and _mn_out["submitted"]["split"] == "S1"
      and [g["chosen_by"] for g in _mn_out["groups"]] == ["fallback", "fallback"],
      json.dumps(_mn_out)[:300])
check("  and the reasoner was shown the fleet, its node count and its group budget",
      "up to 2 node(s)" in _mn_replay.asked[0][0].text
      and "4 group search(es)" in _mn_replay.asked[0][0].text)
hashes = {n: d["key"].hash for n, d in loaded.items()}
check("the 2 Hz and 16 Hz fixtures hash to different validity keys",
      len(set(hashes.values())) == len(hashes), str(hashes))
check("  and the key names the offered rates, which is the field that separates them",
      all(d["key"].offered_rates_hz for d in loaded.values()))
for name, d in loaded.items():
    f = d["fleet"]
    # Not "the file parses" -- that is the check above. This one asks whether the fleet it
    # describes can be hosted on this node at all, which is the thing a typo in a TP degree or a
    # KV floor breaks silently: the file loads, and then nothing can be deployed from it.
    check(f"  {name}: the fleet it describes has at least one hostable placement",
          len(PlacementSolver(f.node.node_spec, f.requests,
                              envelope_step_mib=AGENT_ENVELOPE_STEP_MIB
                              ).solve(cap=1, max_seconds=20.0)) == 1)

# --- store: the hub transport, against an injected `get` (impl §14.8, Fix 2) -------------------
# Every other check in this file reads its manifests out of `fixtures/hubfacts/`, which is how the
# suite stays offline -- and which means the code that TURNS a checkpoint into a manifest is the one
# part of Fix 2 nothing here exercises. It cannot be covered by a fixture (a fixture is its output)
# and must not be covered by the real Hub (a suite whose result depends on somebody else's uptime is
# not a check). `http_get` is a plain function and `HubFactsStore` takes it as an argument for exactly
# this reason: what follows hands it a dict of URLs and keeps the request log, and asserts on what was
# ASKED FOR as well as on what came back. Asking for the wrong thing is the failure with no symptom --
# a manifest built off a file whose tensors were downloaded too is correct and costs 100 GiB of
# transfer, and nothing in a plan's output would say so.
#
# The synthetic checkpoint below is built to carry every case at once, because they interact: a
# multimodal config whose real numbers are nested, a hybrid `layer_types`, an FP8 expert layer, a
# tied head stored twice, both spellings of a draft block, and a vision tower whose block indices run
# PAST the decoder's layer count -- which is the one that turns a resident tower into a draft head if
# the draft rule is not scoped to the text stack.
_FAKE_EP = "https://hub.invalid"          # `.invalid` never resolves, so a leak is a failure not a GET
_FAKE_SHA = "d4d4" + "c" * 36             # deliberately not "main": the sha is what keys the cache
_FAKE_ID = "fixture-org/hybrid-moe-with-a-draft-head"
_FAKE_CONFIG = {
    # The real numbers are under `text_config` and the top level says None, which is Qwen3.5's shape.
    # A reader of the top level gets a crash or a zero; a reader of `vision_config` gets 6 layers and
    # 8 heads, which would price a page 12x too large and look plausible.
    "num_hidden_layers": None,
    "tie_word_embeddings": True,
    "num_nextn_predict_layers": 1,
    "text_config": {"num_hidden_layers": 4,
                    "layer_types": ["full_attention", "linear_attention", "linear_attention",
                                    "linear_attention"],
                    "num_key_value_heads": 2, "head_dim": 128,
                    "linear_num_value_heads": 16, "linear_key_head_dim": 64,
                    "linear_value_head_dim": 64, "mamba_ssm_dtype": "float32"},
    "vision_config": {"num_hidden_layers": 6, "num_attention_heads": 8, "hidden_size": 1024},
}
# name -> (shape, dtype). Byte counts in the comments, and the checks below state the sums by hand:
# a check that recomputes the same product it is checking proves the arithmetic ran, not that it is
# right.
_FAKE_SHARDS = {
    "model-00001-of-00002.safetensors": {
        "model.embed_tokens.weight": ([1000, 128], "BF16"),                       # 256,000
        "model.language_model.layers.0.self_attn.q_proj.weight": ([128, 128], "BF16"),  # 32,768
        "model.language_model.layers.0.self_attn.k_proj.weight": ([64, 128], "BF16"),   # 16,384
        "model.language_model.layers.1.mlp.experts.0.gate_proj.weight": ([64, 64], "F8_E4M3"),
        "model.language_model.layers.1.mlp.experts.1.gate_proj.weight": ([64, 64], "F8_E4M3"),
        "model.visual.blocks.5.attn.qkv.weight": ([32, 32], "BF16"),              # 2,048, RESIDENT
    },
    "model-00002-of-00002.safetensors": {
        "model.language_model.layers.2.mlp.gate_proj.weight": ([16, 16], "F32"),  # 1,024
        "model.language_model.layers.3.mlp.gate_proj.weight": ([16, 16], "F32"),  # 1,024
        "model.language_model.layers.4.mlp.gate_proj.weight": ([16, 16], "F32"),  # 1,024, DRAFT
        "mtp.0.weight": ([8, 8], "BF16"),                                         # 128, DRAFT
        "lm_head.weight": ([1000, 128], "BF16"),                                  # 256,000, TIED
    },
}
# The Hub's own dtype census, which is a field of a response already being fetched and the only
# second opinion on the header sum that exists. Params, not bytes: 281,664 BF16 + 8,192 FP8 + 768 F32.
_FAKE_CENSUS = {"BF16": 281664, "F8_E4M3": 8192, "F32": 768}
_FAKE_CKPT_BYTES = 574592           # 256,000 + 49,152 + 8,192 + 2,048 + 3,072 + 128 + 256,000
_FAKE_RESIDENT_BYTES = 317440       # less the 256,000 tied head and the 1,152 of draft blocks


def _st_bytes(table: dict, pad: int = 0) -> bytes:
    """One safetensors file as this store is allowed to see it: the header, and nothing after it.

    64 bytes of zeros stand in for the tensors -- deliberately far too few to hold what the header
    describes, so a store that reaches for a tensor fails a check rather than passing a slow one.
    """
    body = {name: {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
            for name, (shape, dtype) in table.items()}
    if pad:
        # `__metadata__` is skipped by the reader, so this is header size with nothing else changed.
        body["__metadata__"] = {"pad": "x" * pad}
    blob = json.dumps(body).encode()
    return struct.pack("<Q", len(blob)) + blob + bytes(64)


def _fake_blobs(shards=None, config=None, census=None, *, sha=_FAKE_SHA, pad=0,
                safetensors=True) -> dict:
    shards = _FAKE_SHARDS if shards is None else shards
    files = ["config.json"] + sorted(shards)
    blobs = {f"{_FAKE_EP}/api/models/{_FAKE_ID}?revision=main": json.dumps({
        "sha": sha, "siblings": [{"rfilename": f} for f in (files if safetensors else ["config.json"])],
        "safetensors": {"parameters": _FAKE_CENSUS if census is None else census},
    }).encode()}
    blobs[f"{_FAKE_EP}/{_FAKE_ID}/resolve/{sha}/config.json"] = json.dumps(
        _FAKE_CONFIG if config is None else config).encode()
    if safetensors:
        for name, table in shards.items():
            blobs[f"{_FAKE_EP}/{_FAKE_ID}/resolve/{sha}/{name}"] = _st_bytes(table, pad)
    return blobs


class _FakeHub:
    """Every byte the store is allowed to see, plus the log of what it asked for.

    Honours `Range` rather than ignoring it, because the two-round-trip path in
    `_safetensors_header` only exists for headers larger than the probe and a fake that returned
    everything would never take it.
    """

    def __init__(self, blobs: dict, *, down: bool = False):
        self.blobs = dict(blobs)
        self.down = bool(down)
        self.log: list = []
        self.served = 0

    def __call__(self, url, headers=None, timeout_s=60.0):
        self.log.append((url, dict(headers or {})))
        if self.down:
            raise HubUnavailable(f"GET {url}: fixture hub is unreachable")
        if url not in self.blobs:
            raise HubUnavailable(f"GET {url}: 404 from the fixture hub")
        blob = self.blobs[url]
        rng = (headers or {}).get("Range")
        if rng:
            lo, _, hi = rng.split("=", 1)[1].partition("-")
            blob = blob[int(lo):int(hi) + 1]
        self.served += len(blob)
        return blob

    @property
    def urls(self) -> list:
        return [u for u, _h in self.log]


class _LogLines(logging.Handler):
    """Every log message emitted while it is installed, as text."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def __enter__(self):
        self._was = logging.getLogger("colocation").level
        logging.getLogger("colocation").addHandler(self)
        logging.getLogger("colocation").setLevel(logging.DEBUG)
        return self

    def __exit__(self, *exc):
        logging.getLogger("colocation").removeHandler(self)
        logging.getLogger("colocation").setLevel(self._was)
        return False


with tempfile.TemporaryDirectory() as _td:
    _fake = _FakeHub(_fake_blobs())
    _store = HubFactsStore(_td, endpoint=_FAKE_EP, get=_fake)
    _man = _store.manifest(_FAKE_ID)
    # Four requests and no fifth: the API for the sha and the file list, `config.json`, and one ranged
    # read per shard. The tensors are 574 KB of the fixture and a real checkpoint is 100 GB of them.
    check("a checkpoint becomes a manifest from one API call, one config and one range per shard",
          _fake.urls == [f"{_FAKE_EP}/api/models/{_FAKE_ID}?revision=main",
                         f"{_FAKE_EP}/{_FAKE_ID}/resolve/{_FAKE_SHA}/config.json",
                         f"{_FAKE_EP}/{_FAKE_ID}/resolve/{_FAKE_SHA}/model-00001-of-00002"
                         ".safetensors",
                         f"{_FAKE_EP}/{_FAKE_ID}/resolve/{_FAKE_SHA}/model-00002-of-00002"
                         ".safetensors"],
          f"{len(_fake.log)} requests, {_fake.served:,} bytes served")
    check("  and both shard reads are ranged, which keeps a 100 GB checkpoint off the wire",
          all(h.get("Range") == f"bytes=0-{_HEADER_PROBE_BYTES - 1}"
              for u, h in _fake.log if u.endswith(".safetensors")))
    check("  and an unpinned revision resolves to the sha, which is what the manifest is keyed by",
          _man.revision == _FAKE_SHA and _FAKE_SHA not in ("main", ""),
          f"main -> {_man.revision[:12]}")
    # The rollup, exactly. One entry per numbered block and per unindexed tensor, so a 128-expert
    # layer is one number rather than 128 -- and the two stacks stay distinguishable, which the draft
    # rule below depends on.
    check("  and the tensors roll up to one entry per block, at the byte",
          _man.groups == {"model.embed_tokens": 256000,
                          "model.language_model.layers.0": 49152,
                          "model.language_model.layers.1": 8192,
                          "model.language_model.layers.2": 1024,
                          "model.language_model.layers.3": 1024,
                          "model.language_model.layers.4": 1024,
                          "model.visual.blocks.5": 2048,
                          "mtp.0": 128,
                          "lm_head": 256000},
          json.dumps(_man.groups, sort_keys=True))
    check("  and the experts are priced at one byte an element, not two",
          _man.groups["model.language_model.layers.1"] == 8192
          and DTYPE_BYTES["F8_E4M3"] == 1,
          "8,192 B of FP8 against the 16,384 a bf16 reading would charge")
    check("  and the checkpoint total is the sum of them",
          _man.checkpoint_bytes == _FAKE_CKPT_BYTES, f"{_man.checkpoint_bytes:,} B")
    # The two subtractions, each for a reason a config states rather than a heuristic.
    check("a tied head stored twice is subtracted once, and only when BOTH conditions hold",
          _man.tied_head_bytes() == 256000
          and dataclasses.replace(
              _man, config={k: v for k, v in _FAKE_CONFIG.items()
                            if k != "tie_word_embeddings"}).tied_head_bytes() == 0
          and dataclasses.replace(
              _man, groups={k: v for k, v in _man.groups.items()
                            if k != "lm_head"}).tied_head_bytes() == 0,
          f"{_man.tied_head_bytes():,} B")
    check("both spellings of a draft block are excluded, and the deeper vision tower is not",
          _man.draft_bytes() == {"model.language_model.layers.4": 1024, "mtp.0": 128},
          json.dumps(_man.draft_bytes(), sort_keys=True))
    check("  so the resident lower bound is the checkpoint less the head and the drafts",
          _man.resident_bytes() == _FAKE_RESIDENT_BYTES
          == _FAKE_CKPT_BYTES - 256000 - 1152, f"{_man.resident_bytes():,} B")
    # The geometry, off the nested config. Getting this from the wrong level is not a crash: the
    # vision tower has layers and heads of its own and they are plausible numbers.
    _fg = _man.kv_geometry()
    check("the KV geometry comes off the NESTED text config, not the top level or the vision tower",
          (_fg.kv_layers, _fg.state_layers, _fg.num_kv_heads, _fg.head_dim) == (1, 3, 2, 128)
          and kv_bytes_per_token(num_layers=_fg.kv_layers, num_kv_heads=_fg.num_kv_heads,
                                 head_dim=_fg.head_dim) == 1024,
          f"{_fg.kv_layers} of 4 layers cache K and V, {_fg.num_kv_heads} heads x {_fg.head_dim}")
    check("  and the per-sequence state of the linear layers is derived from their own dims",
          _fg.is_hybrid and _fg.state_bytes_per_seq == 3 * 16 * 64 * 64 * 4,
          f"{_fg.state_bytes_per_seq / (1 << 20):.2f} MiB per sequence")
    # The census: one field of a response already fetched, and the only independent opinion. A
    # cross-check and never a source -- so it agrees here, it refuses to answer when a dtype in it is
    # unknown, and when it is simply wrong the manifest still prices the headers.
    check("the hub's own parameter census agrees with the header sum to the byte",
          _man.hub_parameter_bytes() == _FAKE_CKPT_BYTES)
    check("  and reports nothing rather than a total when a dtype in it has no known width",
          dataclasses.replace(
              _man, hub_parameters={"parameters": {"F5_MADE_UP": 12}}).hub_parameter_bytes() is None)
    check("  and a census that disagrees changes no priced byte, because it is not a source",
          dataclasses.replace(_man, hub_parameters={"parameters": {"BF16": 1}}
                              ).resident_bytes() == _FAKE_RESIDENT_BYTES)
    # The cache, which is the reason this store is not mirrored to EFS: it is keyed by the sha and a
    # pinned revision is answered off disk without a socket.
    check("the manifest is cached under the sha and the schema version",
          (Path(_td) / "hubfacts" / "fixture-org-hybrid-moe-with-a-draft-head"
           / f"{_FAKE_SHA}-v{HUBFACTS_SCHEMA_VERSION}.json").exists())
    _cold = _FakeHub({}, down=True)
    _warm = HubFactsStore(_td, endpoint=_FAKE_EP, get=_cold)
    _from_disk = _warm.manifest(_FAKE_ID, _FAKE_SHA)
    check("  and a fleet that pins that sha loads with no network at all",
          _from_disk == _man and _cold.log == [], f"{len(_cold.log)} requests")
    # Unpinned and unreachable: the newest cached manifest, and a warning saying it may not be what
    # `main` means now. The alternative is refusing an offline question that has an answer on disk.
    with _LogLines() as _lines:
        _stale = HubFactsStore(_td, endpoint=_FAKE_EP, get=_FakeHub({}, down=True)
                               ).manifest(_FAKE_ID)
    check("an unpinned revision falls back to the newest cached manifest, loudly",
          _stale.revision == _FAKE_SHA
          and any("hub unreachable" in ln and "may not be what" in ln for ln in _lines.lines),
          next((ln[:88] for ln in _lines.lines if "hub unreachable" in ln), "NO WARNING LOGGED"))

# The refusals, each of which has to name its own remedy: this is the failure mode Fix 2 introduced
# (a fleet used to load with no network at all) and a traceback out of a store would leave a person
# guessing which of a fleet's models it was about.
with tempfile.TemporaryDirectory() as _td:
    try:
        HubFactsStore(_td, endpoint=_FAKE_EP, get=_FakeHub({}, down=True)).manifest(_FAKE_ID)
        check("a hub that cannot be reached with nothing cached refuses and says what to do", False)
    except HubUnavailable as exc:
        check("a hub that cannot be reached with nothing cached refuses and says what to do",
              "model_revision" in str(exc) and "hubfacts" in str(exc)
              and "not defaultable" in str(exc), str(exc)[:96])
    # The next two are raised by `fetch` and reach a person through the same wrapper, so they are
    # asserted where they are raised: what a fleet sees is this text inside the message above, and a
    # check that only read the wrapper would pass on a store that had lost the reason entirely.
    # An unknown dtype is the one place this module refuses rather than rounding -- a guessed width
    # misprices every tensor in the file, and the fix is one line in `DTYPE_BYTES` by whoever knows
    # what the dtype is.
    _weird = {"model-00001-of-00001.safetensors":
              {"model.language_model.layers.0.w": ([8, 8], "F3_MADE_UP")}}
    for _label, _blobs, _wants in (
            ("  and a repo publishing no safetensors names what a fleet may state instead",
             _fake_blobs(safetensors=False), ("no .safetensors", "kv_bytes_per_token")),
            ("  and an element width nobody has stated is refused, not rounded",
             _fake_blobs(shards=_weird),
             ("F3_MADE_UP", "DTYPE_BYTES", "model.language_model.layers.0.w"))):
        try:
            HubFactsStore(_td, endpoint=_FAKE_EP, get=_FakeHub(_blobs)).fetch(_FAKE_ID)
            check(_label, False)
        except HubUnavailable as exc:
            check(_label, all(w in str(exc) for w in _wants), str(exc)[:104])
        _wlabel = "    ...and that reason survives the wrapper `manifest` puts round it"
        try:
            HubFactsStore(_td, endpoint=_FAKE_EP, get=_FakeHub(_blobs)).manifest(_FAKE_ID)
            check(_wlabel, False)
        except HubUnavailable as exc:
            check(_wlabel, all(w in str(exc) for w in _wants) and "model_revision" in str(exc))

# A header larger than the speculative probe, which is the case worth being correct about rather than
# fast about: the reader asks again for exactly the declared length and gets the same answer.
with tempfile.TemporaryDirectory() as _td:
    _big = _FakeHub(_fake_blobs(pad=_HEADER_PROBE_BYTES))
    with _LogLines() as _tlines:
        _big_man = HubFactsStore(_td, endpoint=_FAKE_EP, get=_big,
                                 token="hf_NOT_A_REAL_TOKEN").manifest(_FAKE_ID)
    _ranges = [h.get("Range") for u, h in _big.log if u.endswith("model-00001-of-00002.safetensors")]
    check("a safetensors header bigger than the probe is re-read at its declared length",
          len(_ranges) == 2 and _ranges[0] == f"bytes=0-{_HEADER_PROBE_BYTES - 1}"
          and _ranges[1].startswith("bytes=8-"), str(_ranges)[:80])
    check("  and the manifest is byte-identical to the one the single round trip produced",
          _big_man.groups == _man.groups
          and _big_man.checkpoint_bytes == _FAKE_CKPT_BYTES)
    # A gated checkpoint needs the token on the resolve URLs too, not only on the API call -- and
    # none of the eight in `fleets/` is gated, so this seam has never run against a real one.
    check("  and a token, when there is one, is sent on every request including the ranged ones",
          all(h.get("Authorization") == "Bearer hf_NOT_A_REAL_TOKEN" for _u, h in _big.log),
          f"{len(_big.log)} requests, all authorized")
    check("  and it appears in no log line, which is the only place a fetch could leak it",
          _tlines.lines and not any("hf_NOT_A_REAL_TOKEN" in ln for ln in _tlines.lines),
          f"{len(_tlines.lines)} lines logged, none of them carrying it")

# Directly on the frame reader, because its two failure modes are silent on the happy path: a file
# that answers a range request with nothing, and a header that is not JSON. Both arrive as bytes that
# `json.loads` would turn into something or nothing depending on the day.
for _label, _blob, _want in (
        ("a truncated safetensors file is refused, not read as an empty tensor table",
         b"\x00\x00", "bytes returned"),
        ("  and a header that is not JSON says so, with the URL it came from",
         struct.pack("<Q", 4) + b"nope", "not JSON")):
    try:
        _safetensors_header(lambda u, h, _b=_blob: _b, f"{_FAKE_EP}/x.safetensors")
        check(_label, False)
    except HubUnavailable as exc:
        check(_label, _want in str(exc) and f"{_FAKE_EP}/x.safetensors" in str(exc), str(exc)[:88])
# The grouping rule itself, on the four stack shapes that exist in `fleets/` plus the unindexed case
# the tied-head rule needs. Truncating at the first numeric component rather than at a fixed depth is
# what makes one rule work on all of them.
check("a tensor's bucket is its name truncated at the first index, whatever the stack is nested in",
      [_tensor_group(n) for n in ("model.layers.7.mlp.down_proj.weight",
                                  "model.language_model.layers.7.mlp.down_proj.weight",
                                  "model.visual.blocks.7.attn.qkv.bias",
                                  "model.layers.46.mlp.experts.127.up_proj.weight",
                                  "lm_head.weight", "model.embed_tokens.weight")]
      == ["model.layers.7", "model.language_model.layers.7", "model.visual.blocks.7",
          "model.layers.46", "lm_head", "model.embed_tokens"])

# --- agent: a `models:` entry is an id, and where the two numbers it used to state come from ----
# Until impl §14.8's Fix 2 an entry stated `weight_gib` and a `kv:` block of dimensions, and both were
# transcribed by hand from the checkpoint. Both now come off the checkpoint itself, through a manifest
# `store/hubfacts.py` fetched once and cached under the commit sha (`tests/hubfixture.py` serves this
# suite's from disk). The checks below are about the two derivations and the one thing the file may
# still say: an override, for the case the derivation cannot cover.
_QWEN_MANIFEST = HUB.manifest("Qwen/Qwen3-0.6B")
_mm_derived = fleetfile.load_model_memory("dense", {"model_id": "Qwen/Qwen3-0.6B"},
                                          manifest=_QWEN_MANIFEST)
check("a model entry of nothing but an id prices a KV page, derived from the checkpoint's config",
      _mm_derived.kv_bytes_per_token_tp1 == QWEN_KV_BYTES
      and _mm_derived.kv_bytes_per_token_tier == TIER_DERIVED,
      f"{_mm_derived.kv_bytes_per_token_tp1:,} B ({_mm_derived.kv_bytes_per_token_tier})")
check("  and the same page size a person used to write into the file, to the byte",
      _mm_derived.kv_bytes_per_token_tp1 == 112 * 1024)
# The weight figure is the RESIDENT lower bound, not the checkpoint: Qwen3-0.6B ties its embedding and
# stores it twice, so 0.2898 GiB of the file is a second copy of one tensor that the engine resides
# once. A hand-written figure had to choose between three quantities (file bytes, checkpoint tensor
# bytes, resident bytes) and the fleet files chose differently from each other; §14.2 needs the third,
# because the weight floor is the only term allowed to REFUSE a degree.
check("  and a weight figure that is the checkpoint's tensors less its duplicated tied head",
      _QWEN_MANIFEST.tied_head_bytes() == 311164928
      and abs(_mm_derived.weight_gib
              - (_QWEN_MANIFEST.checkpoint_bytes - 311164928) / 2 ** 30) < 1e-9,
      f"{_QWEN_MANIFEST.checkpoint_bytes / 2 ** 30:.4f} GiB of tensors -> "
      f"{_mm_derived.weight_gib:.4f} GiB resident")
check("  which is within 0.001 GiB of the 1.110 the file used to state, so this is the same model",
      abs(_mm_derived.weight_gib - 1.110) < 0.001, f"{_mm_derived.weight_gib:.4f} GiB")
check("  and `tied_embedding_gib` stays zero, because nothing may subtract that head twice",
      _mm_derived.tied_embedding_gib == 0.0
      and _mm_derived.weight_lower_bound_gib(1) == _mm_derived.weight_gib)

# The hybrid, which is the reason the `kv:` block had to go rather than be checked. Its config marks 8
# of 32 layers as caching K and V and the other 24 as linear; the all-layers product a person wrote is
# 4x the real page, and no dimension in the block was wrong -- the assumption was, and a file cannot
# contradict an assumption while a config can.
_Q35 = HUB.manifest("Qwen/Qwen3.5-9B")
_Q35_GEOM = _Q35.kv_geometry()
_mm_hybrid = fleetfile.load_model_memory("hybrid", {"model_id": "Qwen/Qwen3.5-9B"}, manifest=_Q35)
check("a hybrid's page counts the layers that cache K and V, not every layer",
      (_Q35_GEOM.kv_layers, _Q35_GEOM.state_layers) == (8, 24)
      and _mm_hybrid.kv_bytes_per_token_tp1 == 32768,
      f"{_Q35_GEOM.kv_layers} of {_Q35_GEOM.kv_layers + _Q35_GEOM.state_layers} layers, "
      f"{_mm_hybrid.kv_bytes_per_token_tp1:,} B/token")
check("  and the all-layers product the old `kv:` block implied is 4x that",
      kv_bytes_per_token(num_layers=_Q35_GEOM.kv_layers + _Q35_GEOM.state_layers,
                         num_kv_heads=_Q35_GEOM.num_kv_heads, head_dim=_Q35_GEOM.head_dim)
      == 4 * _mm_hybrid.kv_bytes_per_token_tp1)
# Reported, and priced nowhere yet: the linear layers keep a per-SEQUENCE recurrent state, which is
# why the measured pages on this family move with the context length and why no single byte count is
# right at two of them. It belongs to whatever owns `max_num_seqs` (Fix 3). An estimate, so the check
# is on its order of magnitude, not on a figure.
check("  and the per-sequence state those layers keep is derived and reported, not folded into "
      "the page", _Q35_GEOM.is_hybrid and 8 << 20 < _Q35_GEOM.state_bytes_per_seq < 128 << 20,
      f"{_Q35_GEOM.state_bytes_per_seq / (1 << 20):.1f} MiB per sequence")
check("  while a dense checkpoint reports no state layers and no state at all",
      not _QWEN_MANIFEST.kv_geometry().is_hybrid
      and _QWEN_MANIFEST.kv_geometry().state_bytes_per_seq == 0)

# The other exclusion, and the one that pays for itself: a trailing layer past `num_hidden_layers` is a
# multi-token-prediction head, resident only if the serving config turns speculative decoding on.
# GLM-4.5-Air-FP8 declares 46 layers and ships 47.
_GLM = HUB.manifest("zai-org/GLM-4.5-Air-FP8")
check("a draft block past the declared layer count is named and excluded from resident weights",
      _GLM.draft_bytes() == {"model.layers.46": 4275691008}
      and _GLM.resident_bytes() == _GLM.checkpoint_bytes - 4275691008,
      f"{_GLM.checkpoint_bytes / 2 ** 30:.4f} GiB of tensors -> "
      f"{_GLM.resident_bytes() / 2 ** 30:.4f} GiB resident")
check("  and the hub's own dtype census agrees with the header sum to the byte, which is the only "
      "independent opinion there is",
      all(m.hub_parameter_bytes() == m.checkpoint_bytes
          for m in (_QWEN_MANIFEST, _Q35, _GLM)))

# The one number the file may still state, and it is an override rather than an input: a hybrid's page
# is not constant across context lengths, so a fleet whose context needs a different figure says so and
# the loader logs the derivation beside it. Refusing it would leave that family unpriceable; accepting
# it silently would make the file unreadable.
_mm_override = fleetfile.load_model_memory(
    "hybrid", {"model_id": "Qwen/Qwen3.5-9B", "kv_bytes_per_token": 40128}, manifest=_Q35)
check("a stated page size overrides the derivation, to the byte",
      _mm_override.kv_bytes_per_token_tp1 == 40128
      and _mm_override.kv_bytes_per_token_tp1 != _mm_hybrid.kv_bytes_per_token_tp1,
      f"{_mm_override.kv_bytes_per_token_tp1:,} B over a derived "
      f"{_mm_hybrid.kv_bytes_per_token_tp1:,}")
check("  and is still the derived tier, because an edited file may not claim a GPU produced it",
      _mm_override.kv_bytes_per_token_tier == TIER_DERIVED, _mm_override.kv_bytes_per_token_tier)
check("  and carries no activation figure, since the fallback is the only source left for one",
      _mm_override.activation_gib is None
      and _mm_override.activation_for(1) == (ACTIVATION_GIB_DEFAULT, "default"),
      str(_mm_override.activation_for(1)))

# The fields the format no longer accepts, each refused BY NAME rather than ignored. A quietly ignored
# field is the worse failure: the file would keep reading as though the number in it were pricing the
# fleet while the loader used another one -- the checkpoint's, the store's, or a fallback. Refusing says
# which and why. `weight_gib` and `kv:` are Fix 2's two; the other three went with `onboard` (§14.4).
for _gone, _value, _want in (
        ("weight_gib", 1.0, "safetensors headers"),
        ("kv", {"num_layers": 28, "num_kv_heads": 8, "head_dim": 128}, "Delete the block"),
        ("activation_gib", {1: 0.5}, "cannot separate it from the KV pool"),
        ("provenance", {"facts_id": "abc123"}, "Delete the block"),
        ("measured_reserve_gib", {1: 0.5}, "merged from the profile store")):
    try:
        fleetfile.load_model_memory("stale", {"model_id": "Qwen/Qwen3-0.6B", _gone: _value},
                                    manifest=_QWEN_MANIFEST)
        check(f"a model still stating `{_gone}` is refused rather than ignored", False)
    except fleetfile.FleetFileError as exc:
        check(f"a model still stating `{_gone}` is refused rather than ignored",
              _want in str(exc), str(exc)[:96])

# And the refusals that are about what is missing rather than what is stale. No manifest means no
# weight figure and no geometry, and there is no field left to state either in -- so the entry is
# refused, which is the shape of every refusal here: a guessed number would decide how many GPUs the
# fleet needs and nothing downstream would say it was guessed.
for _label, _entry, _manifest, _want in (
        ("a model with no manifest and nothing to stand in for one",
         {"model_id": "Qwen/Qwen3-0.6B"}, None, "no KV page size"),
        ("a model with no manifest even when it states a page size",
         {"model_id": "Qwen/Qwen3-0.6B", "kv_bytes_per_token": 114688}, None,
         "weight bytes are unknown"),
        ("a model with no id at all", {}, _QWEN_MANIFEST, "missing 'model_id'"),
        ("a page size of zero", {"model_id": "Qwen/Qwen3-0.6B", "kv_bytes_per_token": 0},
         _QWEN_MANIFEST, "positive"),
        ("a non-integer page size", {"model_id": "Qwen/Qwen3-0.6B", "kv_bytes_per_token": "big"},
         _QWEN_MANIFEST, "integer")):
    try:
        fleetfile.load_model_memory("bad", _entry, manifest=_manifest)
        check(f"  and so is {_label}", False)
    except fleetfile.FleetFileError as exc:
        check(f"  and so is {_label}", _want in str(exc), str(exc)[:88])

# End to end through the real loader, not just `load_model_memory`: the override stated in the file
# itself, at the page size the derivation already gives. Same number, so every downstream figure is
# unchanged -- which is what makes the override one seam rather than a second code path with its own
# arithmetic in it.
with tempfile.TemporaryDirectory() as _td:
    _stated_yaml = Path(_td) / "stated-page.yaml"
    _stated_yaml.write_text(_src.replace(
        "    model_id: Qwen/Qwen3-0.6B\n",
        f"    model_id: Qwen/Qwen3-0.6B\n    kv_bytes_per_token: {QWEN_KV_BYTES}\n", 1))
    _measured = _floorload(_stated_yaml)
    _plain = _floorload(_base_yaml)
    check("a whole fleet file loads with a stated page size over the derived one",
          all(t.mem.kv_bytes_per_token_tp1 == QWEN_KV_BYTES
              for t in _measured["fleet"].tenants),
          f"{len(_measured['fleet'].tenants)} tenants at {QWEN_KV_BYTES:,} B")
    check("  and both forms report the same tier, because both are arithmetic a person could do",
          all(t.mem.kv_bytes_per_token_tier == TIER_DERIVED
              for t in _measured["fleet"].tenants)
          and all(t.mem.kv_bytes_per_token_tier == TIER_DERIVED
                  for t in _plain["fleet"].tenants))
    # The tier is provenance about a number, not part of what makes measurements comparable: two
    # files agreeing on the page size describe the same hardware behaviour. If the tier reached the
    # key, relabelling a number would orphan every record taken before it for no physical reason.
    check("  and the validity key is unchanged, because how a number was learned is not a "
          "condition of the measurement",
          _measured["key"].hash == _plain["key"].hash, _measured["key"].hash[:12])
    check("  and the fleet it describes still has a hostable placement at the same degrees",
          len(PlacementSolver(_measured["fleet"].node.node_spec, _measured["fleet"].requests,
                              envelope_step_mib=AGENT_ENVELOPE_STEP_MIB
                              ).solve(cap=1, max_seconds=20.0)) == 1)
    # A fleet whose checkpoint nothing can read is refused by NAME, with the model id in the message.
    # This is the failure mode Fix 2 introduced -- a fleet used to load with no network at all -- so it
    # has to arrive as a refusal that says which model and what to do, not as a traceback from inside
    # a store. `plan` on a laptop that has loaded this fleet once before still works: the manifest is
    # cached, and only an unpinned revision needs the network to learn which sha it means.
    _unknown_yaml = Path(_td) / "unknown-model.yaml"
    _unknown_yaml.write_text(_src.replace("    model_id: Qwen/Qwen3-0.6B\n",
                                          "    model_id: nobody/never-published\n", 1))
    try:
        _floorload(_unknown_yaml)
        check("a fleet naming a checkpoint no manifest can be had for is refused by name", False)
    except fleetfile.FleetFileError as exc:
        check("a fleet naming a checkpoint no manifest can be had for is refused by name",
              "nobody/never-published" in str(exc) and "qwen3-0.6b" in str(exc), str(exc)[:120])
    _doc = fleetfile.read_doc(_unknown_yaml)
    check("  and the same file still yields a profile key, so what is missing is the checkpoint and "
          "nothing else",
          fleetfile.profile_key(_unknown_yaml, HW_INDEX).hash
          == ProfileKey(env=fleetfile.load_hardware(_doc, HW_INDEX).environment(
              DEFAULT_IMAGE)).hash)

# --- agent: the fleet file may not state tp at all, and where the degree comes from instead ---
# The §0 mistake was a hand-written TP degree wider than the model needs, which puts GPUs into the
# BASELINE and lets a search claim them. `tp:` was optional for a while and honoured when stated;
# since 2026-09-08 it is REFUSED. A shard count asserts that a narrower degree would miss a latency
# target, which is the one input this file cannot check -- and checking a stated degree would have
# cost the same solo runs as computing it, so accepting one bought nothing but a check that could
# later be skipped. The degree is resolved from the memory floor plus a solo measurement instead.
check("the fixtures state no tp on any tenant, because nothing may",
      all(t.get("tp") is None for d in loaded.values() for t in (d["raw"]["tenants"] or [])))
with tempfile.TemporaryDirectory() as _td:
    _stated = Path(_td) / "stated.yaml"
    _stated.write_text(_src.replace("     isl_tokens: 512", "     tp: 2, isl_tokens: 512", 1))
    # Loaded twice, the second time with `allow_unresolved_tp=True`: the refusal has to survive the
    # one flag that tolerates an unresolved degree. An unmeasured degree and an unmeasurable claim
    # are different problems, and the flag is only for the first.
    _msgs = []
    for _kw in ({}, {"allow_unresolved_tp": True}):
        try:
            fleetfile.load(_stated, index=HW_INDEX, hub=HUB, **_kw)
            _msgs.append("")
        except fleetfile.FleetFileError as exc:
            _msgs.append(str(exc))
    check("a stated tp is refused, not honoured and not quietly re-sharded",
          "tp: 2" in _msgs[0] and "latency claim" in _msgs[0], _msgs[0][:90])
    check("  and the refusal survives allow_unresolved_tp, which is not an escape hatch for it",
          _msgs[1] == _msgs[0] and bool(_msgs[0]))
    check("  and it says what to do instead, since deleting the line is not the whole fix",
          "profile" in _msgs[0] and "slo.itl_mean_ms" in _msgs[0])
with tempfile.TemporaryDirectory() as _td:
    _p = Path(_td) / "no-tp.yaml"
    _p.write_text(_src)
    try:
        fleetfile.load(_p, index=HW_INDEX, hub=HUB)
        check("a tenant with neither tp nor an ITL target is refused, not given a degree", False)
    except fleetfile.FleetFileError as exc:
        check("a tenant with neither tp nor an ITL target is refused, not given a degree",
              "itl_mean_ms" in str(exc), str(exc)[:80])

    # With a target but no measurements, the refusal has to say what to measure. A loader that
    # picked the floor here would hand back an integer that every report downstream reads as a
    # measured minimum.
    _p.write_text(_src.replace("/fsx/pretrained-models/Qwen/Qwen3-0.6B}",
                                 "/fsx/pretrained-models/Qwen/Qwen3-0.6B, slo: {itl_mean_ms: 8.0}}"))
    try:
        fleetfile.load(_p, index=HW_INDEX, hub=HUB)
        check("no tp and no profile store is a refusal naming `profile`, not a default", False)
    except fleetfile.FleetFileError as exc:
        check("no tp and no profile store is a refusal naming `profile`, not a default",
              "profile" in str(exc), str(exc)[:80])

    _ps = ProfileStore(Path(_td) / "store", _pkey)
    try:
        fleetfile.load(_p, index=HW_INDEX, hub=HUB, profiles=_ps)
        check("an empty profile store is refused too, with the degrees still to measure", False)
    except fleetfile.FleetFileError as exc:
        check("an empty profile store is refused too, with the degrees still to measure",
              "tp1, tp2, tp4, tp8" in str(exc), str(exc)[:120])

    # --- and the two modes that tolerate it, because they exist to resolve it ------------------
    # `profile` is about to take the measurement and `plan` is what says which one to take, so
    # neither can refuse a fleet for want of it. What they must do instead is label it: the degree
    # they load with is a memory floor, and the sum of floors is a LOWER BOUND on the baseline, not
    # the baseline. Reporting it as the baseline is impl §0's mistake with better provenance.
    from colocation.agent.__main__ import _baseline as _bl               # noqa: E402
    from colocation.agent.__main__ import _print_baseline                # noqa: E402
    _floors = fleetfile.load(_p, index=HW_INDEX, hub=HUB, profiles=_ps, allow_unresolved_tp=True)
    check("`plan` and `profile` load an unmeasured fleet instead of refusing it",
          _floors["fleet"].tp_degrees == {"m1": 1, "r1": 1, "r2": 1, "t2": 1},
          str(_floors["fleet"].tp_degrees))
    check("  and every degree so loaded is named as a floor, not left to look like a minimum",
          fleetfile.unresolved_tp(_floors) == ["m1", "r1", "r2", "t2"])
    check("  and each carries the status saying why, with the degrees still to measure",
          all(r.status == "unprofiled" and r.tp is None and r.missing == (1, 2, 4, 8)
              for r in _floors["tp_resolutions"].values()),
          str(_floors["tp_resolutions"]["m1"].as_dict()))
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        _print_baseline(_bl(_floors["fleet"]), fleetfile.unresolved_tp(_floors))
    _floortext = _buf.getvalue()
    check("  so `plan` prints BASELINE NOT ESTABLISHED rather than a number to beat",
          "BASELINE NOT ESTABLISHED" in _floortext and "LOWER BOUND" in _floortext)
    check("  and never claims the degrees were stated, since the file states none of them",
          "stated tp" not in _floortext and "TP degrees this file states" not in _floortext)
    check("  and marks each degree as a floor on the tenant's own line",
          _floortext.count("FLOOR, not a measured minimum") == 4)

    # The one unmeasured degree that IS resolved, and the reason the rule above is not "no tp means
    # unresolved": 200 GiB of WEIGHTS is 25 GiB per shard at tp8 and 50 at tp4, against a 44.99 GiB
    # card. Every narrower degree is ruled out by the one term that cannot come back smaller than
    # stated, and there is no wider one, so no measurement could change the answer.
    _p.write_text(_src.replace(
        "/fsx/pretrained-models/Qwen/Qwen3-0.6B}", "/fsx/pretrained-models/Qwen/Qwen3-0.6B, slo: {itl_mean_ms: 8.0}}"))
    _forced = fleetfile.load(_p, index=HW_INDEX, hub=fixture_hub(
        {"Qwen/Qwen3-0.6B": synthetic_manifest("Qwen/Qwen3-0.6B", resident_gib=200.0)}))
    check("a tenant whose WEIGHT floor is the widest degree resolves with no measurement at all",
          _forced["fleet"].tp_degrees == {"m1": 8, "r1": 8, "r2": 8, "t2": 8},
          str(_forced["fleet"].tp_degrees))
    check("  and is reported as resolved, not as a floor: no profile could contradict it",
          fleetfile.unresolved_tp(_forced) == []
          and all(r.status == "resolved" and r.certain
                  for r in _forced["tp_resolutions"].values()))

    # And the case that used to take that same branch and may not (impl §14.2): at 2M KV tokens the
    # 0.6B's *memory* floor is tp8 too, but its weights are 1.110 GiB -- 0.14 GiB per shard at tp8,
    # and under the card at tp1. So what rules out tp1, tp2 and tp4 here is a KV pool sized from a
    # page size that was measured 3.27x too large the one time it met a booted engine, plus an
    # activation fallback measured 4.4x too large. Resolving tp8 `certain` from that spent no round
    # and closed the question; it now goes the way every unmeasured degree goes.
    _p.write_text(_src.replace(
        "/fsx/pretrained-models/Qwen/Qwen3-0.6B}", "/fsx/pretrained-models/Qwen/Qwen3-0.6B, max_model_len: 1000000, slo: {itl_mean_ms: 8.0}}"))
    try:
        fleetfile.load(_p, index=HW_INDEX, hub=HUB)
        check("a memory floor at the widest degree is no longer resolved without a measurement",
              False)
    except fleetfile.FleetFileError as exc:
        check("a memory floor at the widest degree is no longer resolved without a measurement",
              "no profile store to resolve its TP degree from" in str(exc), str(exc)[:140])
    _kvfloored = fleetfile.load(_p, index=HW_INDEX, hub=HUB, profiles=_ps, allow_unresolved_tp=True)
    check("  and `plan`/`profile` load it at tp8 as a FLOOR, with the narrower degrees still open",
          _kvfloored["fleet"].tp_degrees == {"m1": 8, "r1": 8, "r2": 8, "t2": 8}
          and fleetfile.unresolved_tp(_kvfloored) == ["m1", "r1", "r2", "t2"],
          str(_kvfloored["tp_resolutions"]["m1"].as_dict()["status"]))
    _p.write_text(_src.replace("/fsx/pretrained-models/Qwen/Qwen3-0.6B}",
                                 "/fsx/pretrained-models/Qwen/Qwen3-0.6B, slo: {itl_mean_ms: 8.0}}"))

    # Now the measurements exist. tp1 misses the 8 ms ceiling solo and tp2 meets it, so tp2 is this
    # tenant's minimal TP -- and it is minimal because the narrower degree was MEASURED to miss,
    # not because nobody looked.
    _ps.write(_solo(1, 11.4))
    _ps.write(_solo(2, 6.2))
    _resolved = fleetfile.load(_p, index=HW_INDEX, hub=HUB, profiles=_ps)
    check("with the sweep in the store, every tenant resolves to its measured minimal TP",
          _resolved["fleet"].tp_degrees == {"m1": 2, "r1": 2, "r2": 2, "t2": 2},
          str(_resolved["fleet"].tp_degrees))
    check("  and the resolution is returned with its evidence, so a report can show the working",
          all(r.certain and r.solo_itl_ms == 6.2
              for r in _resolved["tp_resolutions"].values()),
          str(_resolved["tp_resolutions"]["m1"].as_dict()["below"]))
    # TP is in the validity key and a target is not, so resolving one re-keys and stating the other
    # does not. Both are correct: a shard count changes what the hardware does, a target does not.
    check("a resolved TP degree keys the store, exactly as a stated one would",
          _resolved["key"].hash != _plain["key"].hash
          and _resolved["key"].tensor_parallel == {"m1": 2, "r1": 2, "r2": 2, "t2": 2})

    # The refusal that saves the most: solo at the full envelope is the best case, so a target no
    # degree meets alone is a target no placement can meet.
    _p.write_text(_src.replace("/fsx/pretrained-models/Qwen/Qwen3-0.6B}",
                                 "/fsx/pretrained-models/Qwen/Qwen3-0.6B, slo: {itl_mean_ms: 1.0}}"))
    for _k in (1, 2, 4, 8):
        _ps.write(_solo(_k, 5.9))
    try:
        fleetfile.load(_p, index=HW_INDEX, hub=HUB, profiles=_ps)
        check("a target met at no measured degree is refused as unachievable, not widened", False)
    except fleetfile.FleetFileError as exc:
        check("a target met at no measured degree is refused as unachievable, not widened",
              "ALONE" in str(exc), str(exc)[:100])

    # --- and every one of those refusals reaches the person as a refusal ------------------------
    # Each message above says which tenant and what to change. A traceback buries that under a
    # stack and reads as a broken tool, which matters most for the audience most likely to see one:
    # `fleets/` ships empty, so the first fleet file anybody loads is one they just wrote.
    from colocation.agent.__main__ import main as _cli                    # noqa: E402
    _bad = Path(_td) / "bad.yaml"
    # A `validity_key:` block, which is the refusal a fleet written before this store keyed itself to
    # what a node reports will hit -- and the one that has to arrive as a message, since the remedy is
    # to delete four lines rather than to fix anything.
    _bad.write_text(_src + '\nvalidity_key:\n  driver: "580.178.04"\n  metrics_version: v1\n')
    # `--store` is where the hardware facts live too, so the store this `plan` is pointed at has to
    # be one somebody has captured against -- otherwise it refuses for the unfiled instance type
    # before it reaches the error under test. Copied rather than shared so the mode may write to it.
    shutil.copytree(Path(_HW_ROOT) / "hwfacts", Path(_td) / "hwfacts")
    # `--efs-store` is pointed somewhere that does not exist, and every `_cli` call below does the
    # same. Without it these three read the REAL canonical store: `plan` mirrors `hwfacts/` before it
    # resolves anything, so the fixture card above and whatever the cluster's EFS holds for the same
    # product both land in this temp store -- and on 2026-09-14, when a corrected device law reached
    # EFS, that made this check fail with a `Disagreement` about an L40S instead of the fleet-file
    # error it is about. A self-test whose outcome depends on what a cluster measured last is not a
    # self-test. The unreachable path exercises the documented degrade (warn, use local facts).
    _no_efs = str(Path(_td) / "no-such-efs")
    try:
        _cli(["plan", "--fleet", str(_bad), "--store", _td, "--efs-store", _no_efs])
        check("a fleet-file error exits with the message, not a traceback", False)
    except fleetfile.FleetFileError:
        check("a fleet-file error exits with the message, not a traceback", False,
              "raised through main() instead of being reported")
    except SystemExit as exc:
        check("a fleet-file error exits with the message, not a traceback",
              "bad.yaml" in str(exc) and "validity_key: is no longer part of a fleet file" in str(exc),
              str(exc)[:90])
    try:
        _cli(["plan", "--fleet", str(Path(_td) / "nope.yaml"), "--store", _td,
              "--efs-store", _no_efs])
        check("  and so does a fleet file that is not there", False)
    except SystemExit as exc:
        check("  and so does a fleet file that is not there", "no fleet file at" in str(exc),
              str(exc)[:60])
    # And so does the store having nothing filed for the machine, which is NOT a fleet-file error --
    # the file is correct and nobody has leased that instance type. It is the first thing anybody
    # pointing this at new hardware will hit, so it is the last thing that should arrive as a stack.
    _unfiled = Path(_td) / "unfiled.yaml"
    _unfiled.write_text(_src.replace("instance_type: g6e.48xlarge", "instance_type: p5.48xlarge"))
    try:
        _cli(["plan", "--fleet", str(_unfiled), "--store", _td, "--efs-store", _no_efs])
        check("  and so does an instance type the store has never seen", False)
    except UnknownHardware:
        check("  and so does an instance type the store has never seen", False,
              "raised through main() instead of being reported")
    except SystemExit as exc:
        check("  and so does an instance type the store has never seen",
              "p5.48xlarge" in str(exc) and "--gpu-count" in str(exc), str(exc)[:60])
    # And so does the SDK not being installed -- before the fleet file is even opened, which is what
    # these two checks pin. Both SDK drivers import the SDK inside `run()`, which is reached with a
    # node already leased, so an ImportError there would strand a lease over a `pip install`. The
    # fleet path below does not exist, so a refusal naming the driver rather than the file is proof
    # the question is asked first.
    # `search` is the only reasoner-driven mode; `onboard` was the second until impl §14.4, and the
    # gate was checked over both. One mode is not a weaker check -- what is pinned is the ordering.
    _real_sdk = agent_main.sdk_available
    agent_main.sdk_available = lambda: False
    try:
        try:
            _cli(["search", "--fleet", str(Path(_td) / "nope.yaml"), "--store", _td,
                  "--harness-image", "repo/harness:t", "--efs-store", _no_efs])
            check("search --driver sdk is refused when the SDK is not installed", False)
        except SystemExit as exc:
            check("search --driver sdk is refused when the SDK is not installed, before a node "
                  "can be leased over a missing pip package",
                  "pip install claude-agent-sdk" in str(exc) and "no fleet file" not in str(exc),
                  str(exc)[:72])
        try:
            _cli(["search", "--fleet", str(Path(_td) / "nope.yaml"), "--store", _td,
                  "--harness-image", "repo/harness:t", "--driver", "converse",
                  "--efs-store", _no_efs])
            check("  while --driver converse needs no such thing and gets on with the run", False)
        except SystemExit as exc:
            check("  while --driver converse needs no such thing and gets on with the run",
                  "no fleet file at" in str(exc), str(exc)[:60])
    finally:
        agent_main.sdk_available = _real_sdk

# --- agent: hardware, from an instance type and a store -----------------------
# Everything here used to be answered by two tables in `agent/hardware.py`, and the reason they
# are gone is generality: a table has to be edited before this stack can run on hardware nobody
# here has leased, and every row of it is a claim somebody typed about a machine they may not have
# seen. So there are two facts now and each is READ -- a shape somebody observed on a node, a total
# somebody measured on one -- and the two ways of not having them are deliberately different
# failures, because one of them anybody can supply and the other nobody can without a GPU.
HW = hardware_for("g6e.48xlarge", HW_INDEX)
check("an instance type resolves against the store to a count, a product and a MEASURED capacity",
      (HW.gpu_count, HW.gpu_product, HW.device_total_gib) == (8, "l40s", CARD_GIB),
      f"{HW.gpu_count}x{HW.gpu_product}, {HW.device_total_gib} GiB")
check("  and to the GPU ordinals the node will have, which is what the solver is given",
      HW.gpu_ids == tuple(range(8)))
check("  and says where each of the two facts came from, since a plan is only worth its inputs",
      "observed" in HW.provenance() and "slope" in HW.provenance()
      and f"{CARD_GIB:.3f}" in HW.provenance(), HW.provenance())
try:
    hardware_for("p5.48xlarge", HW_INDEX)
    check("an instance type nobody has leased is refused, not defaulted", False)
except UnknownHardware as exc:
    # Not "add a row": there is nowhere to add one. The message has to name both ways out -- lease a
    # node of this type, or state the count once and have it verified -- and list what IS filed, so
    # that a typo in an instance type reads as a typo rather than as missing hardware.
    check("an instance type nobody has leased is refused, not defaulted",
          "p5.48xlarge" in str(exc) and "--gpu-count" in str(exc)
          and "g6e.48xlarge" in str(exc), str(exc)[:80])

# The bootstrap. An unfiled type may be stated once, and what a stated shape buys is a `plan`; what
# it does not buy is silence about being stated, because nothing has checked it against a machine.
_hw_stated = hardware_for("p5.48xlarge", HW_INDEX, gpu_count=8, gpu_product="H100")
check("--gpu-count bootstraps an unfiled instance type instead of blocking on a lease",
      (_hw_stated.gpu_count, _hw_stated.gpu_product) == (8, "h100"), _hw_stated.provenance())
check("  and the plan made from it says the count was stated and never verified",
      _hw_stated.shape_source == SHAPE_STATED and "never verified" in _hw_stated.provenance())
check("  while the total stays unknown, because stating a card does not measure one",
      not _hw_stated.total_known and _hw_stated.device_total_gib is None)
try:
    _hw_stated.require_total_gib("solving a placement")
    check("  and the first thing that needs the total refuses, naming what wanted it", False)
except UnknownHardware as exc:
    # The refusal is here rather than at construction so that it can name the caller: `plan` may
    # legitimately resolve a machine it cannot yet price, and `capture` must, since it leases by
    # count. What may never happen is a utilisation multiplied by a guess. The way out it names is a
    # lease -- it used to name `onboard`, which is deleted (impl §14.4), and the replacement is any
    # mode that leases, because the figure comes off the driver on the node and not out of a mode.
    check("  and the first thing that needs the total refuses, naming what wanted it",
          "solving a placement" in str(exc) and "profile" in str(exc)
          and "no default" in str(exc), str(exc)[:80])
try:
    hardware_for("g6e.48xlarge", HW_INDEX, gpu_count=4)
    check("  but a stated count may not contradict a shape read off a real node", False)
except UnknownHardware as exc:
    check("  but a stated count may not contradict a shape read off a real node",
          "contradicts the hardware store" in str(exc) and "8x" in str(exc), str(exc)[:80])
try:
    hardware_for("g6e.48xlarge", HW_INDEX, gpu_product="a100")
    check("  nor a stated product, which would price the fleet against another card", False)
except UnknownHardware:
    check("  nor a stated product, which would price the fleet against another card", True)
# A shape filed before the device facts were part of it. Every store that existed when the address
# became an observation has one, so this is the first thing such a store does -- and what it did was
# let `InstanceShape.parse`'s `ValueError` out through `main` as a traceback, which reads as a crash
# rather than as a store needing one node held once. It is an unknown shape, so it is reported as
# one: that is the exception `profile` and `search` already route to `_bootstrap_environment`, which
# reads those three facts off a leased node and re-files the shape.
with tempfile.TemporaryDirectory() as _td:
    _legacy = Path(_td) / "hwfacts" / "instances"
    _legacy.mkdir(parents=True)
    (_legacy / "g6e.48xlarge.json").write_text(json.dumps({
        "schema_version": HWFACTS_SCHEMA_VERSION, "instance_type": "g6e.48xlarge", "gpu_count": 8,
        "gpu_product": "l40s", "observed_at": "2026-01-01T00:00:00Z", "node": "old-node"}))
    try:
        hardware_for("g6e.48xlarge", HardwareIndex(Path(_td)))
        check("a shape filed before the device facts existed is an unknown shape, not a traceback",
              False)
    except UnknownHardware as exc:
        check("a shape filed before the device facts existed is an unknown shape, not a traceback",
              "cannot be read by this build" in str(exc)
              and "addressable_mib" in str(exc), str(exc)[:80])
        check("  and it names the lease as the remedy, because nothing offline can observe a machine",
              "profile" in str(exc) and "search" in str(exc)
              and "nothing offline can observe" in str(exc), str(exc)[:80])
check("normalize_gpu reduces a vendor-prefixed name to the store's key form",
      normalize_gpu("NVIDIA L40S") == normalize_gpu("l40s") == "l40s")
# Stated as a check because it is the reason `capture._observe` keys off the Karpenter label and
# not off nvidia-smi: this normalization is not an inverse of AWS's naming, it is a common form
# applied to both sides of a comparison.
check("  and does not claim to be AWS's label -- 'Tesla T4' is not the token 't4'",
      normalize_gpu("Tesla T4") == "tesla-t4")
try:
    # The failure that matters: a GPU product whose device total nobody has measured. Carrying one
    # card's capacity onto a card several times its size solves the fleet onto a fraction of the
    # GPUs it needs, and the symptom arrives hours later as an engine OOM that reads as a bad
    # placement. Which is why the RESOLVER refuses, and why this constructor does not: a `Hardware`
    # may be built for any product by a caller that HAS the measurement.
    Hardware(instance_type="x.large", gpu_count=1, gpu_product="h200",
             device_total_gib=CARD_GIB, total_source="slope")
    check("a Hardware may be built for any product -- the lookup is what refuses one", True)
except Exception as exc:                                                # noqa: BLE001
    check("a Hardware may be built for any product -- the lookup is what refuses one", False,
          str(exc))
try:
    HW.verify(gpu_count=4, gpu_product="l40s")
    check("a node with fewer GPUs than the store says is refused, not adopted", False)
except UnknownHardware:
    # Raises rather than adopting, because `plan` has already printed placements sized from the
    # filed shape and the store directory is already named from the instance type.
    check("a node with fewer GPUs than the store says is refused, not adopted", True)
try:
    HW.verify(gpu_count=8, gpu_product="a100")
    check("  and so is a node whose GPU is not the product the store names", False)
except UnknownHardware:
    check("  and so is a node whose GPU is not the product the store names", True)
check("  while the node it does describe verifies silently",
      HW.verify(gpu_count=8, gpu_product="l40s") is None)
# The driver's own reading is the independent check on a slope, so it is compared and not adopted.
# The tolerance is the sheet's own error bar rather than a constant, because a slope measured across
# a narrow utilisation spread is honestly fuzzier than one across a wide one.
check("  and the driver's memory.total agreeing with the measured slope verifies silently",
      HW.verify(driver_total_gib=CARD_GIB + 0.01) is None)
try:
    HW.verify(driver_total_gib=CARD_GIB * 0.9)
    check("  while a driver that reports a different card is a finding, not an update", False)
except UnknownHardware as exc:
    check("  while a driver that reports a different card is a finding, not an update",
          "further apart" in str(exc) and "not something to proceed past" in str(exc),
          str(exc)[:80])
try:
    HW.with_driver_total(CARD_GIB)
    check("  and a machine that already has a measured total refuses to overwrite it", False)
except UnknownHardware as exc:
    check("  and a machine that already has a measured total refuses to overwrite it",
          "does not replace it silently" in str(exc), str(exc)[:80])

# `with_driver_total` on the one machine it is legal for: the bootstrap above, now holding a node.
# What it must NOT claim is that anything cross-checked the figure -- the driver said it, and no
# engine has yet demonstrated that its own arithmetic uses the same number.
_boot = _hw_stated.with_shape(gpu_count=8, gpu_product="NVIDIA H100").with_driver_total(79.65,
                                                                                    node="node-h")
check("an unmeasured card adopts the driver's memory.total, which is enough to plan with",
      (_boot.device_total_gib, _boot.total_source) == (79.65, "driver"), _boot.provenance())
check("  and records it as not cross-checked, because nothing has contradicted it yet",
      _boot.total_falsifiable is False
      and "nothing has contradicted it yet" in _boot.provenance(), _boot.provenance())
check("  and its uncertainty is the driver's own print quantum, whole MiB and no better",
      _boot.total_uncertainty_gib == DRIVER_TOTAL_QUANTUM_GIB)
check("  while the shape it now carries was read off a node rather than stated",
      _boot.shape_source == SHAPE_OBSERVED, _boot.provenance())
try:
    _boot.shape(node="node-h")
    check("  and is still not filable, because a count and a product are no longer a shape", False)
except UnknownHardware as exc:
    # The count and the product used to BE the shape. They are not any more: the addressable total,
    # the power limit and the link matrix are what an offline `plan` computes the store's address
    # from, so a shape filed without them would leave the next run unable to name its own directory
    # -- and it would do it while reporting a successful observation.
    check("  and is still not filable, because a count and a product are no longer a shape",
          "carries no device facts" in str(exc) and "with_device_facts" in str(exc), str(exc)[:80])
_boot = _boot.with_device_facts(addressable_mib=81090, power_limit_w=700,
                                p2p_topology=topo(8, "NV18"), node="node-h")
check("  and what makes it filable is the capture's own reading of the three device facts",
      (_boot.shape().addressable_mib, _boot.shape().power_limit_w, _boot.shape().gpu_count)
      == (81090, 700, 8), _boot.provenance())
try:
    _hw_stated.shape()
    check("  and a shape that was only stated is not, which would file a claim as a reading", False)
except UnknownHardware as exc:
    check("  and a shape that was only stated is not, which would file a claim as a reading",
          "record a claim as an observation" in str(exc), str(exc)[:80])

# --- the device total across environments, and the refusal that makes it safe ---
# `HardwareIndex.device_total` reads every hardware key, which `HardwareStore` deliberately won't.
# The justification is that a card's capacity is not conditioned by the runtime opening a context on
# it -- the three environments below differ by engine image and agree about the card -- and
# what makes it safe is that the claim is TESTED here rather than assumed: two environments that
# measured the same card differently are a refusal, because that would mean the premise is false for
# that hardware, and averaging two figures would bury it.
with tempfile.TemporaryDirectory() as _td:
    _ix = HardwareIndex(_td)
    check("a card nobody has measured reads as unknown, not as a default", _ix.device_total("l40s")
          is None)
    for _img, _drv, _tot, _unc, _fals in (("vllm:0.24.0", "580.65.06", CARD_GIB, 0.02, False),
                                          ("vllm:0.25.0", "585.00.00", CARD_GIB + 0.005, 0.01,
                                           True)):
        _s = HardwareStore(_td, HardwareKey(env=env(engine_image=_img).device()))
        _s.write_key()
        _s.write(HardwareFacts(gpu_product="l40s", total_gib=_tot, total_source="slope",
                              total_uncertainty_gib=_unc, total_falsifiable=_fals, driver=_drv))
    _dt = _ix.device_total("l40s")
    check("two environments agreeing on a card's size report the BEST-resolved figure, not a mean",
          _dt.total_gib == CARD_GIB + 0.005 and _dt.uncertainty_gib == 0.01, _dt.provenance())
    check("  and the agreement itself is what makes the figure falsifiable, and is reported",
          _dt.falsifiable and len(_dt.keys) == 2, _dt.provenance())
    # A sheet that recorded no uncertainty is not thereby exact: it is the loosest of the group, so
    # it can never be the figure chosen and can never lend slack to a comparison.
    _s = HardwareStore(_td, HardwareKey(env=env(engine_image="vllm:0.26.0").device()))
    _s.write_key()
    _s.write(HardwareFacts(gpu_product="l40s", total_gib=CARD_GIB + 0.004, total_source="driver"))
    check("  and a sheet with no error bar never becomes the answer, however close it is",
          _ix.device_total("l40s").uncertainty_gib == 0.01)
    _s.write(HardwareFacts(gpu_product="l40s", total_gib=CARD_GIB * 0.5, total_source="driver",
                           total_uncertainty_gib=0.01))
    try:
        _ix.device_total("l40s")
        check("two environments disagreeing about a card's size is refused, never resolved", False)
    except Disagreement as exc:
        check("two environments disagreeing about a card's size is refused, never resolved",
              "cannot all be right" in str(exc) and "nothing here will pick one" in str(exc),
              str(exc)[:80])
    # And it propagates: a resolver that swallowed it would hand back one of the two sizes as
    # though it were the card, which is the exact mistake the refusal exists to prevent.
    _ix.write_shape(InstanceShape(instance_type="g6e.48xlarge", gpu_count=8, gpu_product="l40s",
                                  addressable_mib=ADDRESSABLE_MIB, power_limit_w=POWER_LIMIT_W,
                                  p2p_topology=topo(8)))
    try:
        hardware_for("g6e.48xlarge", _ix)
        check("  and `hardware_for` lets it through rather than falling back to a size", False)
    except Disagreement:
        check("  and `hardware_for` lets it through rather than falling back to a size", True)

# --- agent: capture, and what a fleet loaded against a lease gains -------------
# `plan` runs with no node at all. The unleased inventory is what makes that possible, and
# `uuid_for` refusing is what stops it being mistaken for something deployable.
_unleased = loaded["loader-16hz.yaml"]["fleet"].node
check("with no node captured the fleet is planned against the instance type's shape",
      not _unleased.leased and _unleased.node_name == "any g6e.48xlarge"
      and len(_unleased.gpu_ids) == 8, _unleased.node_name)
try:
    _unleased.uuid_for(0)
    check("  and asking such a fleet for a GPU UUID is refused, never a placeholder", False)
except KeyError as exc:
    # Not a generated "GPU-00000000" either: a UUID naming no device is ACCEPTED by MPS and
    # ignored, so the tenant would run with no envelope and the record would look healthy
    # (finding 9).
    check("  and asking such a fleet for a GPU UUID is refused, never a placeholder",
          "captured" in str(exc))

_cap = FakeCapture(INV, HW)
LEASE = _cap.capture("rtest", HW)
check("a lease names the release that IS it, and the node it learned from the pod",
      (LEASE.release, LEASE.node_name) == ("mps-rtest", "node-a"), LEASE.summary())
check("  its exporter URL is its own release's, not a frozen one",
      LEASE.exporter_url.startswith("http://mps-rtest-dcgm-exporter."))
check("  and its tenant prefix is the run token, so two runs cannot collide on a release name",
      LEASE.tenant_prefix == "rtest")
_leased = _floorload(FLEETS / "loader-16hz.yaml", lease=LEASE)
check("loading the same fleet against a lease binds it to that node, with real GPU UUIDs",
      _leased["fleet"].node.leased and _leased["fleet"].node.uuid_for(0) == "GPU-0000")
check("  and does not change the validity key, because nothing in the key is read off the node",
      _leased["key"].hash == loaded["loader-16hz.yaml"]["key"].hash)

with tempfile.TemporaryDirectory() as _td:
    _src16 = (FLEETS / "loader-16hz.yaml").read_text()
    _oldnode = Path(_td) / "old-node.yaml"
    _oldnode.write_text(_src16.replace("hardware:\n", "node:\n  name: ip-1-2-3-4\nhardware:\n"))
    try:
        _floorload(_oldnode)
        check("a fleet file that still names a node is refused, with the replacement named",
              False)
    except fleetfile.FleetFileError as exc:
        check("a fleet file that still names a node is refused, with the replacement named",
              "instance_type" in str(exc) and "CAPTURES" in str(exc))

    _vkinst = Path(_td) / "vk-instance.yaml"
    _vkinst.write_text(_src16 + '\nvalidity_key:\n  driver: "580.178.04"\n'
                                "  instance_type: g6e.48xlarge\n  metrics_version: v1\n")
    try:
        _floorload(_vkinst)
        check("  and so is a `validity_key:` block, which is refused rather than ignored", False)
    except fleetfile.FleetFileError as exc:
        # Ignoring it would be worse than refusing. Every field it used to carry either moved to
        # something the node reports about itself, which no author can type, or was never read at
        # all -- and a file that still states a driver reads as though the driver still named the
        # directory the rows go in. So the refusal has to say where each field went.
        check("  and so is a `validity_key:` block, which is refused rather than ignored",
              "driver" in str(exc) and "metrics_version" in str(exc)
              and "instance_type" in str(exc), str(exc)[:90])

    _nohw = Path(_td) / "no-hardware.yaml"
    _nohw.write_text(_src16.replace("  instance_type: g6e.48xlarge\n", ""))
    check("  (the fixture really removed the instance type)",
          "g6e.48xlarge" not in _nohw.read_text())
    try:
        _floorload(_nohw)
        check("  and a fleet that names no instance type at all", False)
    except fleetfile.FleetFileError:
        check("  and a fleet that names no instance type at all", True)

    _extrahw = Path(_td) / "extra-hardware.yaml"
    _extrahw.write_text(_src16.replace("hardware:\n", "hardware:\n  device_total_gib: 48.0\n"))
    try:
        # A second key here would be a second source of truth for a fact the cluster can be asked
        # about -- and the nameplate 48.0 is exactly the wrong answer somebody would put there.
        _floorload(_extrahw)
        check("  and a hardware block with a second key, which would re-state a derived fact",
              False)
    except fleetfile.FleetFileError:
        check("  and a hardware block with a second key, which would re-state a derived fact",
              True)

# The driver is read off the node and recorded, and it addresses NOTHING. This used to be the
# opposite: the file asserted a driver, the lease was refused if it disagreed, and a bump minted a new
# directory -- a whole history discarded for a string that may have changed no measurable thing, with
# an empty directory afterwards indistinguishable from a fresh start. What a bump does change, if it
# changes anything at all, the environment's own observed fields catch (store/environment.py).
_other = FakeCapture(INV, HW, driver="999.99.99").capture("rdrv", HW)
_otherload = _floorload(FLEETS / "loader-16hz.yaml", lease=_other)
check("a node running a different driver keeps the same key, so a bump does not discard the store",
      _otherload["key"].hash == loaded["loader-16hz.yaml"]["key"].hash,
      f"{_otherload['key'].hash} on driver {_other.driver}")
check("  and the driver appears nowhere in the key, which is what makes that true and not a tolerance",
      "999.99.99" not in json.dumps(_otherload["key"].canonical())
      and "driver" not in json.dumps(_otherload["key"].canonical()))
check("  while the lease still carries it, because a number nobody can attribute is not a measurement",
      _other.driver == "999.99.99" and _other.observed["driver"] == "999.99.99")

# A perfectly self-consistent lease on the wrong machine: 4 GPUs, a g6e.12xlarge, everything
# verified. It is refused because the validity key names the instance type, so measuring here would
# file the records under a machine they were not taken on.
_HW4 = hardware_for("g6e.12xlarge", HW_INDEX)
_INV4 = NodeInventory(node_name="node-b", gpu_ids=tuple(range(4)),
                      uuids=tuple(f"GPU-{i:04d}" for i in range(4)), device_total_gib=CARD_GIB)
_small = FakeCapture(_INV4, _HW4).capture("rwrg", _HW4)
try:
    _floorload(FLEETS / "loader-16hz.yaml", lease=_small)
    check("  and so is a lease on a different instance type than the fleet asks for", False)
except fleetfile.FleetFileError as exc:
    check("  and so is a lease on a different instance type than the fleet asks for",
          "g6e.12xlarge" in str(exc))

_cap.release(LEASE)
check("releasing a lease is recorded as releasing the daemon release, not the node",
      _cap.released == ["mps-rtest"])
try:
    FakeCapture(INV, HW, fail_with="no capacity").capture("rfail", HW)
    check("a capture that fails raises rather than returning a lease to nothing", False)
except CaptureError:
    check("a capture that fails raises rather than returning a lease to nothing", True)

# The serving chart's own directory, which the capture checks below reach through to find the
# `mps-daemon` chart beside it. It used to be defined with `HelmBooter`'s checks and outlived them
# (impl §14.4); it is a path on disk and nothing installs from it here.
_VLLM_CHART = (Path(__file__).resolve().parents[3]
               / "charts/machine-learning/model-serving/vllm-openai")

# --- one DCGM host engine per GPU, checked before the node is bought ------------
# A GPU lets exactly one DCGM host engine own its profiling fields; the second one to ask exits with
# `The third-party Profiling module returned an unrecoverable error` and crash-loops. That costs no
# latency measurement and every cause signal: the records come back with no `per_gpu` block and the
# run looks successful. Live on 2026-09-14 -- the `amazon-cloudwatch-observability` addon's own
# dcgm-exporter DaemonSet was on the leased g6e and had claimed the profiling groups first; ours
# restarted 11 times, and disabling only the addon's copy brought all seven PROF fields back.
# So the check is cluster-wide (a DaemonSet follows the lease onto whatever node Karpenter buys) and
# it runs in the pre-flight, before the install that orders the node.
def _k8s(kind, who, image, tolerations=None):
    ns, _, name = who.partition("/")
    spec = {"containers": [{"image": image}], "tolerations": tolerations or []}
    return {"metadata": {"namespace": ns, "name": name},
            "spec": {"template": {"spec": spec}} if kind == "ds" else spec}


def _items(*items):
    return json.dumps({"items": list(items)})


# The addon's exporter tolerates every taint, as a node-monitoring DaemonSet does; the GPU Operator's
# tolerates only the KAI pool's, which is what it looked like live on 2026-10-08.
_ALL = [{"operator": "Exists"}]
_KAI = [{"key": "kai.scheduler/gpu", "operator": "Exists", "effect": "NoSchedule"}]
_CUDA_TAINTS = json.dumps([{"key": "nvidia.com/gpu", "value": "true", "effect": "NoSchedule"}])
_CNI = _k8s("ds", "kube-system/aws-node", "<account>.dkr.ecr.<region>.amazonaws.com/amazon-k8s-cni:v1.19")
_DCGM_DS = _items(_k8s("ds", "amazon-cloudwatch/dcgm-exporter",
                       "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.2-4.7.1", _ALL), _CNI)
_OPERATOR_DS = _items(_k8s("ds", "gpu-operator/nvidia-dcgm-exporter",
                           "nvcr.io/nvidia/k8s/dcgm-exporter:4.6.1-4.8.4-distroless", _KAI), _CNI)
_DCGM_PODS = _items(
    _k8s("pod", "amazon-cloudwatch/dcgm-exporter-dlmmn", "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.2",
         _ALL),
    _k8s("pod", "kubeflow-user-example-com/mps-r2j5kq-dcgm-exporter-78db75fb87-mf9zn",
         "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.2-4.7.1-ubuntu22.04"))
_NO_DCGM = _items(_CNI)


def _dcgm_refusal(daemonsets: str, pods: str, taints=_CUDA_TAINTS) -> str:
    """`_check_one_dcgm_exporter` against a canned cluster. "" means it let the run proceed.
    `taints` is the lease pool's, or None for a pool whose taints cannot be read."""
    cap = HelmCapture(str(_VLLM_CHART.parent / "mps-daemon"))
    real = agent_capture._run

    def fake(argv, **k):
        if "nodepool" in argv:
            if taints is None:
                raise DeployError("nodepools.karpenter.sh \"cuda\" not found")
            return taints
        return daemonsets if "daemonsets" in argv else pods
    agent_capture._run = fake
    try:
        cap._check_one_dcgm_exporter()
        return ""
    except CaptureError as exc:
        return str(exc)
    finally:
        agent_capture._run = real


def _on_node_refusal(rows: str) -> str:
    cap = HelmCapture(str(_VLLM_CHART.parent / "mps-daemon"))
    real = agent_capture._run
    agent_capture._run = lambda argv, **k: rows
    try:
        cap._refuse_foreign_dcgm_on("node-a")
        return ""
    except CaptureError as exc:
        return str(exc)
    finally:
        agent_capture._run = real


check("a foreign dcgm-exporter DaemonSet is refused before the install that buys the node",
      "amazon-cloudwatch/dcgm-exporter" in _dcgm_refusal(_DCGM_DS, _NO_DCGM),
      "one host engine per GPU; the second to ask loses every PROF field it was installed for")
check("  and the refusal carries the command that fixes it, not just the diagnosis",
      "dcgmExporter" in _dcgm_refusal(_DCGM_DS, _NO_DCGM)
      and "update-addon" in _dcgm_refusal(_DCGM_DS, _NO_DCGM),
      "deleting the DaemonSet or the DcgmExporter CR is undone by the addon's operator in seconds")
check("  a foreign exporter POD counts too, wherever its DaemonSet went",
      "amazon-cloudwatch/dcgm-exporter-dlmmn" in _dcgm_refusal("", _DCGM_PODS))
check("  but this project's OWN lease exporter does not, or no run could ever hold a second lease",
      "mps-r2j5kq" not in _dcgm_refusal("", _DCGM_PODS))
check("a cluster with no other exporter is not refused, and an empty listing is not an error",
      _dcgm_refusal(_NO_DCGM, _NO_DCGM) == "" and _dcgm_refusal("", "") == "",
      "a check that raises on 'nothing here' would refuse every run on a healthy cluster")
check("an exporter that does not tolerate the lease pool's taints is not counted: it cannot get there",
      _dcgm_refusal(_OPERATOR_DS, _NO_DCGM) == "",
      "the GPU Operator's exporter on a cluster with KAI tolerates only kai.scheduler/gpu")
check("  but the same exporter counts when the pool's taints cannot be read",
      "gpu-operator/nvidia-dcgm-exporter" in _dcgm_refusal(_OPERATOR_DS, _NO_DCGM, taints=None))
check("  and an untainted pool excuses nothing",
      "gpu-operator/nvidia-dcgm-exporter" in _dcgm_refusal(_OPERATOR_DS, _NO_DCGM, taints="[]"))
check("  and an exporter that does tolerate the pool's taint is still refused",
      "amazon-cloudwatch/dcgm-exporter" in _dcgm_refusal(_items(_k8s(
          "ds", "amazon-cloudwatch/dcgm-exporter", "dcgm-exporter:4",
          [{"key": "nvidia.com/gpu", "operator": "Equal", "value": "true"}])), _NO_DCGM))
check("  with Kubernetes' own toleration rule: key, value, effect, and the empty-key Exists",
      agent_capture._tolerates([{"operator": "Exists"}], {"key": "a", "effect": "NoSchedule"})
      and agent_capture._tolerates([{"key": "a", "operator": "Exists", "effect": "NoSchedule"}],
                                   {"key": "a", "value": "x", "effect": "NoSchedule"})
      and not agent_capture._tolerates([{"key": "a", "operator": "Exists",
                                         "effect": "NoExecute"}],
                                       {"key": "a", "effect": "NoSchedule"})
      and not agent_capture._tolerates([{"key": "a", "value": "y"}],
                                       {"key": "a", "value": "x", "effect": "NoSchedule"})
      and not agent_capture._tolerates([], {"key": "a", "effect": "NoSchedule"}))
check("a foreign exporter found on the leased node itself is refused before anything is measured",
      "amazon-cloudwatch/dcgm-exporter-x" in _on_node_refusal(
          "amazon-cloudwatch/dcgm-exporter-x\tnvcr.io/nvidia/k8s/dcgm-exporter:4.4.2 \n")
      and _on_node_refusal("kubeflow-user-example-com/mps-r2j5kq-dcgm-exporter-7\t"
                           "nvcr.io/nvidia/k8s/dcgm-exporter:4.4.2 \n") == ""
      and _on_node_refusal("") == "")

# --- a node the cluster has already decided to replace is refused at capture ----
# The holder carries `karpenter.sh/do-not-disrupt`, so while it sits on the node nothing can replace
# it -- including for a drift the autoscaler has been wanting to act on for days. A capture PREEMPTS
# the holder, which removes the only blocker: on 2026-09-10 the claim took a deletionTimestamp three
# seconds after the daemon went in, the run finished only because its own pods held the drain open
# for two hours, and the g6e.48xlarge could not be reacquired in any AZ of the region. The check is
# free and the answer is knowable the moment the node name is known, which is also the earliest it
# CAN be asked -- capture is discovery, so there is no node to ask about before the pod lands.
_CLAIM_OK = {"metadata": {"name": "cuda-g9jcr"},
             "status": {"nodeName": "node-a",
                        "conditions": [{"type": "Launched", "status": "True"},
                                       {"type": "Registered", "status": "True"},
                                       {"type": "Ready", "status": "True"}]}}
_CLAIM_DRIFTED = {"metadata": {"name": "cuda-g9jcr"},
                  "status": {"nodeName": "node-a",
                             "conditions": [{"type": "Ready", "status": "True"},
                                            {"type": "Drifted", "status": "True",
                                             "reason": "NodeClassDrift",
                                             "message": "AMIID drifted"}]}}
_CLAIM_GOING = {"metadata": {"name": "cuda-g9jcr",
                             "deletionTimestamp": "2026-09-10T19:48:26Z"},
                "status": {"nodeName": "node-a", "conditions": []}}


def _drift_refusal(claims, fail: bool = False, protected: bool = False) -> str:
    """`_refuse_if_disrupted('node-a')` against a canned cluster. "" means the run proceeded.

    `protected` is what the node's `karpenter.sh/do-not-disrupt` annotation reads as, which is the
    difference between "the cluster is about to take this node" and "the cluster would like to and
    cannot". Routed on the argv rather than by call order, so the test does not encode how many
    times the code looks at either thing.
    """
    cap = HelmCapture(str(_VLLM_CHART.parent / "mps-daemon"))
    real = agent_capture._run

    def fake(argv, **kw):
        if "node" in argv:                       # the annotation read, not the claim listing
            return "true" if protected else ""
        if fail:
            raise DeployError("error: the server doesn't have a resource type \"nodeclaims\"")
        return json.dumps({"items": list(claims)})

    agent_capture._run = fake
    try:
        cap._refuse_if_disrupted("node-a")
        return ""
    except CaptureError as exc:
        return str(exc)
    finally:
        agent_capture._run = real


check("a node whose claim is marked Drifted is refused, with the node and the claim named",
      all(s in _drift_refusal([_CLAIM_DRIFTED]) for s in ("node-a", "cuda-g9jcr", "Drifted")),
      "the holder was the only thing keeping it, and the capture is what removed the holder")
check("  and the refusal carries the cluster's own reason, not just the condition type",
      "NodeClassDrift" in _drift_refusal([_CLAIM_DRIFTED])
      and "AMIID drifted" in _drift_refusal([_CLAIM_DRIFTED]))
check("  and it names the remedy -- let the replacement come, then capture again",
      "capture again" in _drift_refusal([_CLAIM_DRIFTED]),
      "there is deliberately no flag to proceed: a drifted node will be taken away mid-run")
check("  and it warns that the replacement may not carry the driver the validity key names",
      "validity key" in _drift_refusal([_CLAIM_DRIFTED]))
check("a claim already being deleted is refused too, and the timestamp is quoted",
      "2026-09-10T19:48:26Z" in _drift_refusal([_CLAIM_GOING]),
      "this is the 2026-09-10 sequence exactly: deleted three seconds after the daemon went in")
check("a healthy claim is not refused, and Launched/Registered/Ready are not disruption",
      _drift_refusal([_CLAIM_OK]) == "",
      "a check that refuses a normal node refuses every run")
check("  nor is a Drifted condition that is present and False",
      _drift_refusal([{"metadata": {"name": "c"},
                       "status": {"nodeName": "node-a",
                                  "conditions": [{"type": "Drifted", "status": "False"}]}}]) == "",
      "matched on status as well as type: these are added when they apply, not carried False")
check("  and ANOTHER node's drift is not this run's problem, because the join is on nodeName",
      _drift_refusal([dict(_CLAIM_DRIFTED,
                           status=dict(_CLAIM_DRIFTED["status"], nodeName="node-z"))]) == "",
      "refusing on a cluster-wide drift would refuse every run whenever any node drifted")
check("a node no claim owns is not refused: nothing there can take it away mid-run",
      _drift_refusal([]) == "")
check("and a cluster with no claim resource at all is not refused either",
      _drift_refusal([], fail=True) == "",
      "no autoscaler of this kind means nothing to check, not a failed check -- and refusing a "
      "lease over a transient kubectl error would turn an API blip into a lost node")
check("the check runs before anything is measured on the node, which is the whole point of it",
      (lambda src: 0 < src.index("_refuse_if_disrupted(") < src.index("self._observe("))(
          inspect.getsource(HelmCapture.capture)),
      "a lease is not cheaper for having been spent before the question was asked")

# --- ...unless the run is holding the node itself -------------------------------
# Refusing every drifted node makes GPU work impossible on a cluster whose nodepool tracks a moving
# AMI alias: the node is drifted from the hour it boots and stays that way, so the refusal above
# would never lift. What actually kept this node for six days was the holder's own
# `karpenter.sh/do-not-disrupt`, and the honest fix is to put the same annotation on the node for the
# length of the lease rather than to add a flag that proceeds regardless. That narrows the refusal to
# what an annotation cannot call off -- a disruption already begun, and a claim already deleted.
check("a Drifted node the run has protected is NOT refused: the autoscaler cannot act on drift",
      _drift_refusal([_CLAIM_DRIFTED], protected=True) == "",
      "the nodepool tracks al2023@latest, so this node is permanently Drifted; refusing it "
      "unconditionally means no measurement can ever be taken on this cluster again")
check("  and it says so out loud rather than proceeding silently past a real drift",
      "log.warning" in inspect.getsource(HelmCapture._refuse_if_disrupted),
      "the node IS drifted and the run is standing on that; a silent pass hides it from the log")
check("  and the closing line does not then say the node is unmarked, which would contradict it",
      (lambda src: src.index("tolerated = True")
       < src.index("held against the mark above")
       < src.index("is not marked %s on its claim"))(
          inspect.getsource(HelmCapture._refuse_if_disrupted)),
      "the first pass logged the drift and then logged 'nothing scheduled to remove it' about the "
      "same node; of two contradictory lines the reassuring one is what a reader keeps")
check("but a disruption already BEGUN is refused even when the node is protected",
      "Disrupted" in _drift_refusal(
          [{"metadata": {"name": "cuda-g9jcr"},
            "status": {"nodeName": "node-a",
                       "conditions": [{"type": "Disrupted", "status": "True",
                                       "reason": "Drifted", "message": "draining"}]}}],
          protected=True),
      "the annotation blocks voluntary disruption from starting; nothing calls one back")
check("  and so is a claim with a deletionTimestamp, annotation or no annotation",
      "2026-09-10T19:48:26Z" in _drift_refusal([_CLAIM_GOING], protected=True),
      "the decision is already taken and recorded; an annotation cannot un-delete it")
check("  and an unprotected drifted node is still refused, and the refusal says the annotation is "
      "what is missing",
      "karpenter.sh/do-not-disrupt" in _drift_refusal([_CLAIM_DRIFTED]),
      "the difference between the two outcomes is exactly that annotation, so name it")


def _protect_calls(annotation: str, nodes=("node-a",), held=None):
    """(`what _protect/_protect_held_nodes returned`, `the argv this issued`). No cluster."""
    cap = HelmCapture(str(_VLLM_CHART.parent / "mps-daemon"))
    real = agent_capture._run
    argvs: list = []

    def fake(argv, **kw):
        argvs.append(argv)
        if "annotate" in argv:
            return ""
        if "pods" in argv:
            return "\n".join(held or ())
        return annotation                            # the `get node -o jsonpath` read

    agent_capture._run = fake
    try:
        if held is None:
            return cap._protect(nodes[0]), argvs
        return cap._protect_held_nodes(), argvs
    finally:
        agent_capture._run = real


check("a node that already carries the annotation is left alone, and _protect says it did not add it",
      _protect_calls("true")[0] is False
      and not any("annotate" in c for c in _protect_calls("true")[1]),
      "it may be somebody else's -- taking it off at release would remove protection this run "
      "never granted itself")
check("  a node without it is annotated, and _protect says so, which is what makes release symmetric",
      _protect_calls("")[0] is True
      and any("annotate" in c and "karpenter.sh/do-not-disrupt=true" in c
              for c in _protect_calls("")[1]),
      "only what this run added is undone, so the record of what to undo has to be the return value")
check("_protect_held_nodes reports only the nodes it actually annotated",
      _protect_calls("", held=("node-a", "node-b"))[0] == ["node-a", "node-b"]
      and _protect_calls("true", held=("node-a", "node-b"))[0] == [],
      "an already-protected node is somebody else's to unprotect")
check("  and no holder anywhere is not an error: there is simply no protection to carry over",
      _protect_calls("", held=())[0] == []
      and not any("annotate" in c for c in _protect_calls("", held=())[1]),
      "a cluster with no holder pod has no lease to inherit, which is normal on a first capture")
check("the protection goes on BEFORE the install, because the install is what removes the holder",
      (lambda src: 0 < src.index("_protect_held_nodes(") < src.index("self._install("))(
          inspect.getsource(HelmCapture.capture)),
      "the daemon preempts the holder; annotating afterwards leaves a window where the node "
      "carries neither, and ~600s is not the only thing that can take a node in that window")
check("and it comes off AFTER the holder is reapplied, for the same reason in reverse",
      (lambda src: 0 < src.index('"apply", "-f"') < src.index("_hand_back_protection("))(
          inspect.getsource(HelmCapture.release)),
      "un-annotating first would unprotect a node whose holder is still Pending")
check("  _abandon undoes the protection too, so a failed capture does not leave the node pinned",
      "_hand_back_protection(" in inspect.getsource(HelmCapture._abandon),
      "an annotation that outlives its lease keeps a machine nobody is holding")


def _handback(held, timeout_s: float = 0.0) -> list:
    """The nodes `_hand_back_protection(['node-a'])` actually un-annotated."""
    cap = HelmCapture(str(_VLLM_CHART.parent / "mps-daemon"))
    real = agent_capture._run
    off: list = []

    def fake(argv, **kw):
        if "pods" in argv:
            return "\n".join(held)
        if "annotate" in argv:
            off.append(argv[argv.index("node") + 1])
        return ""

    agent_capture._run = fake
    try:
        cap._hand_back_protection(["node-a"], timeout_s=timeout_s)
        return off
    finally:
        agent_capture._run = real


check("the annotation is handed back only where a holder is actually on the node again",
      _handback(["node-a"]) == ["node-a"] and _handback([]) == [],
      "the annotation and the holder guard the same thing; the node must never carry neither")
check("  and a node left protected is warned about loudly rather than quietly un-annotated",
      "no holder pod is on it" in inspect.getsource(HelmCapture._hand_back_protection)
      and "thing keeping the machine" in inspect.getsource(HelmCapture._hand_back_protection),
      "over-protected is a machine that stays; unprotected with no holder is one that goes")
check("  and the wait is a parameter, so nothing in this suite sleeps for a minute to test it",
      "timeout_s" in str(inspect.signature(HelmCapture._hand_back_protection)),
      "a test that waited out a real deadline would be a minute of nothing per run")

# --- the closing handoff names every release the run installed, not just the lease ----
# A search ends with its tenants still serving, deliberately (`profiler.run`): somebody may want to
# look at the placement, keep it up or measure it again, which is why teardown is opt-in. The defect
# was never that default -- it was a closing message which, followed exactly, left the node unheld.
# Tenant pods are DEFAULT priority and `gpu-node-holder` is -1000, so uninstalling only the daemon
# leaves the holder Pending; a Pending holder is not on the node, so it carries neither the GPUs nor
# its own `do-not-disrupt`. Both runs on 2026-09-10 ended exactly that way.
_RARGS = SimpleNamespace(release_node=False, chart="charts/x", namespace="ns",
                         holder_manifest="manifests/gpu-node-holder.yaml")


def _handoff(tenants, fail: bool = False, protected=()) -> str:
    """What `_release` prints when the lease is left standing. No cluster, no helm."""
    class _Dep:
        def __init__(self, *a, **k):
            pass

        def live_tenant_releases(self):
            if fail:
                raise RuntimeError("helm list failed")
            return list(tenants)

    real = agent_main.HelmDeployer
    agent_main.HelmDeployer = _Dep
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            agent_main._release(_RARGS, None,
                                dataclasses.replace(LEASE, protected_nodes=tuple(protected)),
                                failed=False)
    finally:
        agent_main.HelmDeployer = real
    return buf.getvalue()


_HAND = _handoff(["rtest-chat", "rtest-guard"])
check("the closing handoff names every tenant release still serving, not only the lease",
      "rtest-chat" in _HAND and "rtest-guard" in _HAND and "mps-rtest" in _HAND,
      "what it prints, followed exactly, has to leave the node held")
check("  and it orders the tenants before the lease, because that order is the whole fix",
      _HAND.index("rtest-chat") < _HAND.rindex("mps-rtest"))
check("  and it says why: the holder is -1000 and cannot preempt a default-priority tenant",
      "-1000" in _HAND and "preempt" in _HAND)
check("  and it says doing nothing is a legitimate choice, since that is the default",
      "do nothing" in _HAND,
      "the tenants are left up on purpose; the message must not read as a cleanup instruction")
_BARE = _handoff([])
check("with no tenant left, the handoff is the short form and invents no empty uninstall",
      "mps-rtest" in _BARE and "tenant release" in _BARE and "uninstall  " not in _BARE)
_WARN = _handoff([], fail=True)
check("and when the releases cannot be listed, it says so and still names the lease",
      "WARNING" in _WARN and "rtest-" in _WARN and "mps-rtest" in _WARN,
      "a failed helm list must not silently produce the message that leaves the node unheld")
# Ordering the uninstalls right is necessary and not sufficient. `helm uninstall` returns when the API
# server accepts the deletion of the objects it created, while the pod those objects made goes on
# holding its GPUs until the container exits -- 0.9s against ~30s, measured in `deploy._await_departed`.
# So `uninstall && apply` lands the holder on a node whose devices are still held, which is the same
# Pending-holder-buys-a-second-node failure as the one above, reached from the other side. The code
# path already waits (`_await_departed` for the tenants, `_daemon_gone` for the daemon); a handoff
# followed by hand runs neither, so the printed form has to carry the wait itself.
for _label, _text in (("with tenants", _HAND), ("bare", _BARE)):
    # The commands are the indented lines; the prose around them names `&&` to say not to use it,
    # so it is the typed lines that have to be checked and not the whole message.
    _cmds = [ln for ln in _text.splitlines() if ln.startswith("    ")]
    check(f"the handoff ({_label}) does not &&-join an uninstall to the holder apply",
          _cmds and not any("&&" in ln for ln in _cmds),
          "helm returning is not a GPU released, so a one-liner can apply the holder into a node "
          "that is still held")
    check(f"  and it ({_label}) gives each step its own line, so one cannot outrun another",
          sum(1 for ln in _cmds if "helm uninstall" in ln) >= 1
          and any("apply -f" in ln for ln in _cmds)
          and not any("helm uninstall" in ln and "apply -f" in ln for ln in _cmds))
    check(f"  and it ({_label}) waits on the POD, keyed by the release label, before applying",
          "app.kubernetes.io/instance in (" in _text
          and _text.index("while kubectl get pod") < _text.index("apply -f"),
          "polling the release would clear the moment helm returns, which is the bug")
check("  and the wait covers every release being removed, tenants and lease alike",
      "in (rtest-chat,rtest-guard,mps-rtest)" in _HAND,
      "any one of them left resident keeps the devices held")
check("  and the message says why the wait is there rather than just printing it",
      "load-bearing" in _HAND and "0.9s" in _HAND,
      "an unexplained sleep is the first thing a reader deletes")
check("  and it says the missing timeout is deliberate: what it waits on is what holds the node",
      "no timeout on purpose" in _HAND,
      "a bounded wait that expires would go on to apply the holder anyway")
# ...and the same message has to name the annotation the run put on the node, for exactly the reason
# above: a handoff is followed BY HAND, so anything `capture.release` would have undone is a step the
# reader now owns. Left out, the annotation outlives the lease and keeps a machine nobody chose to
# keep -- the safe side of the mistake, but state nobody asked for and nobody knows is there.
_PROT = _handoff([], protected=("node-a",))
check("the handoff names the do-not-disrupt this run added, since nothing else will remove it",
      f"annotate node node-a {agent_capture.DO_NOT_DISRUPT}-" in _PROT,
      "capture.release undoes it; a handoff followed by hand does not run capture.release")
check("  and it comes AFTER the holder apply in the printed command, not before",
      _PROT.index("apply -f") < _PROT.index("annotate node node-a"),
      "un-annotating first unprotects a node whose holder is still Pending -- the same ordering "
      "bug as the tenant/lease one, in a different place")
check("  and the message says what that last step is for, not just what to type",
      "gives that job back to the holder" in _PROT)
check("  while a run that annotated nothing prints no annotate step at all",
      "annotate" not in _BARE,
      "a cluster with no such autoscaler has nothing that reads it; a step that does nothing "
      "teaches the reader to skip steps")

# --- the device total is what a CUDA context can address, not what the board holds ----
# `memory.total` is the board; the driver keeps a region of it for page-retirement and ECC structures
# that no context ever sees, reported separately as `memory.reserved`. The engine's
# gpu_memory_utilization multiplies total MINUS reserved, so a planner multiplying total hands every
# engine more memory than the packing priced -- and nothing fails, which is why it went unnoticed.
# Found 2026-09-14 the hard way: `onboard` fitted the card from three boots as d(kv_gib)/d(util),
# got a figure many times its own uncertainty below the board total, and refused to file the device
# law over the disagreement. The slope was right. `memory.total - memory.reserved` on the same node
# matched it to under a thousandth of a GiB.
_SMI_ROWS = "NVIDIA L40S, 580.178.04, {total}\n" * 2
# Two devices a hop apart, tab-separated, with the host columns on the right that the real command
# prints. Those columns are the reason `parse_topology` reads only `GPU<n>` headers: a NUMA id in the
# digest would re-key the store because a scheduler placed the pod differently.
_SMI_TOPO = ("\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\n"
             "GPU0\t X \tSYS\t0-23\t0\tN/A\n"
             "GPU1\tSYS\t X \t24-47\t1\tN/A\n")


def _observed_total(total_mib: int, reserved: str, power: str = "350.00") -> dict:
    """One whole observation from a canned nvidia-smi, with no cluster.

    The full dict rather than the three memory figures, because the same call is what reads the power
    limit and the link matrix -- the other two thirds of the address these rows get filed under -- and
    a fixture that answered only the memory query would leave those two unexercised offline.
    """
    cap = HelmCapture(str(_VLLM_CHART.parent / "mps-daemon"))
    node = {"metadata": {"labels": {"node.kubernetes.io/instance-type": "g6e.48xlarge",
                                    "karpenter.k8s.aws/instance-gpu-name": "l40s",
                                    "karpenter.k8s.aws/instance-gpu-count": "2"}},
            "status": {"allocatable": {"nvidia.com/gpu": "2"}, "nodeInfo": {}}}

    def fake(argv, **kw):
        joined = " ".join(argv)
        if "get" in argv and "node" in argv:
            return json.dumps(node)
        if "memory.reserved" in joined:
            if reserved == "unsupported":
                raise DeployError("Field \"memory.reserved\" is not a valid field to query.")
            return f"{reserved}\n{reserved}\n"
        if "enforced.power.limit" in joined:
            return f"{power}\n{power}\n"
        if "topo" in argv:
            return _SMI_TOPO
        return _SMI_ROWS.format(total=total_mib)

    # Both are patched, and that is the point of the pair: `capture` asks for used/total through its
    # own `_run`, and the reserve comes back through `agent/smi.py`, which `booter` also calls. When
    # the reserve reader moved out of `capture` into that shared module, patching only the first left
    # this check passing a board total and calling it addressable -- so the seam is pinned here.
    real, real_smi = agent_capture._run, agent_smi._run
    agent_capture._run = agent_smi._run = fake
    try:
        obs = cap._observe("mps-t0k3n", "node-a", {"metadata": {"name": "mps-t0k3n-mps-daemon-x"}})
    finally:
        agent_capture._run, agent_smi._run = real, real_smi
    return obs


def _totals(obs: dict) -> tuple:
    """(addressable_gib, board_gib, reserved_gib) out of an observation."""
    return obs["device_total_gib"], obs["device_total_board_gib"], obs["device_reserved_gib"]


_obs = _observed_total(46068, "611")
_addr, _board, _res = _totals(_obs)
check("the device total a lease files is what a context can ADDRESS: memory.total minus reserved",
      abs(_addr - (_board - _res)) < 1e-9 and _res > 0.0 and _addr < _board,
      f"board {_board:.3f} GiB, reserved {_res:.3f} GiB, filed {_addr:.3f} GiB")
check("  and both halves of the subtraction are recorded, because the result alone hides the reserve",
      _board > 0.0 and _res > 0.0)
check("  a driver too old to report memory.reserved costs a warning, not the lease it already holds",
      _totals(_observed_total(46068, "unsupported")) == (_board, _board, 0.0),
      "reverting to the board total is generous, and a stranded lease is worse than generous")
check("  as does a driver that answers with a placeholder rather than a number",
      _totals(_observed_total(46068, "[N/A]")) == (_board, _board, 0.0))
# And the rest of the address, read on the same node in the same call, because that is the only place
# any of it is a fact rather than a claim. In the driver's own units and as integers: the key needs no
# tolerance because equality at the instrument's quantum is the comparison.
check("the same observation carries the whole address: addressable MiB, the enforced watts, the links",
      (_obs["addressable_mib"], _obs["power_limit_w"]) == (46068 - 611, 350)
      and _obs["p2p_topology"] == topology_digest({(0, 1): "SYS"}),
      f"{_obs['addressable_mib']} MiB, {_obs['power_limit_w']} W, {_obs['p2p_topology']}")
check("  and the matrix behind the digest, since a digest alone is a directory name nobody can read",
      _obs["p2p_links"] == {"0-1": "SYS", "1-0": "SYS"}, str(_obs["p2p_links"]))
check("  while the MiB figure and the GiB one are one subtraction, not two spellings of it",
      _obs["addressable_mib"] / 1024.0 == _obs["device_total_gib"],
      "the MiB names the store directory and the GiB sizes the envelopes; disagreeing would mean "
      "the rows describe a different machine than the directory says")
# The power limit is the one field with no degrade, unlike the reserve two checks up. A missing reserve
# costs accuracy in a known direction and the run can still measure; a fabricated power limit would
# file rows taken on a capped card under a key claiming an uncapped one, which is silent forever.
try:
    _observed_total(46068, "611", power="[N/A]")
    check("  and a card that will not state its power limit stops the capture rather than guessing",
          False)
except CaptureError as exc:
    check("  and a card that will not state its power limit stops the capture rather than guessing",
          "capped card" in str(exc) and "nothing to fall back on" in str(exc), str(exc)[:80])
# Stated the other way round, because the conversion that ran this direction went with `booter.py`
# (impl §14.4): the SAME envelope demands a HIGHER utilisation of the addressable total than of the
# board, and the utilisation is what the chart hands the engine. So the correction reaches the
# packing, which is the claim -- a planner that multiplied the board total would hand every engine
# more memory than the placement was priced against, and nothing would fail.
check("a utilisation computed against the addressable total is higher, so the correction reaches "
      "the packing",
      gpu_memory_utilization(20.0, 1.0, _addr) > gpu_memory_utilization(20.0, 1.0, _board),
      f"{gpu_memory_utilization(20.0, 1.0, _addr):.4f} addressable vs "
      f"{gpu_memory_utilization(20.0, 1.0, _board):.4f} board")

# --- capture on hardware the store has never seen: the whole bootstrap, once ----
# This is the run that makes a new card usable, and it is the reason there is no table any more. The
# operator states a count, the lease checks it against the node and reads `memory.total`, and both
# facts are filed -- so the second run on this instance type resolves it offline with no flag. What
# it must not do is let the stated count through unchecked, or file the driver's figure as though
# something had cross-checked it.
_NEWCARD = 79.647            # what the fake driver reports for a card nothing has measured
with tempfile.TemporaryDirectory() as _td:
    _ix = HardwareIndex(_td)
    _new = hardware_for("p5.48xlarge", _ix, gpu_count=8, gpu_product="H100")
    _invn = NodeInventory(node_name="node-h100", gpu_ids=tuple(range(8)),
                          uuids=tuple(f"GPU-h{i:04d}" for i in range(8)),
                          device_total_gib=_NEWCARD)
    _lease = FakeCapture(_invn, _new, driver="580.65.06").capture(
        "rboot", _new, index=_ix, image=DEFAULT_IMAGE)
    check("a capture on unmeasured hardware comes back priced by the driver's own memory.total",
          (_lease.hardware.device_total_gib, _lease.hardware.total_source)
          == (_NEWCARD, "driver"), _lease.hardware.provenance())
    check("  and the stated shape is now an observation, naming the node it was read on",
          (_lease.hardware.shape_source, _lease.hardware.shape_node)
          == (SHAPE_OBSERVED, "node-h100"))
    check("  and the lease's summary carries that provenance, so the run says what it priced with",
          "driver" in _lease.summary() and "node-h100" in _lease.summary(), _lease.summary())
    _filed = _ix.read_shape("p5.48xlarge")
    check("  and the shape is FILED, which is what makes --gpu-count a bootstrap and not a habit",
          (_filed.gpu_count, _filed.gpu_product, _filed.node) == (8, "h100", "node-h100"))
    _resolved = hardware_for("p5.48xlarge", _ix)
    check("  so the next run resolves the same machine offline, with no flag and no table",
          (_resolved.gpu_count, _resolved.gpu_product, _resolved.device_total_gib)
          == (8, "h100", _NEWCARD)
          and _resolved.shape_source == SHAPE_OBSERVED, _resolved.provenance())
    check("  while the filed total says it rests on the driver alone and nothing has tested it",
          _resolved.total_source == "driver" and _resolved.total_falsifiable is False
          and _resolved.total_uncertainty_gib == DRIVER_TOTAL_QUANTUM_GIB,
          _resolved.provenance())
    _sheets = _ix.sheets_for("h100")
    # Compared against the key derived from the FILED shape rather than against a hand-written one,
    # because that equality is the property the bootstrap exists for: the directory this capture wrote
    # into is the one the next, offline run computes from the shape left behind. The driver appears in
    # neither -- it is on the sheet as provenance, two checks down.
    check("  and the sheet is filed under the address the node's own facts compute, not a typed string",
          len(_sheets) == 1
          and _ix.read_key(_sheets[0][0]) == HardwareKey(
              env=_filed.device_environment(DEFAULT_IMAGE)),
          str(_ix.read_key(_sheets[0][0])))
    check("  and says who read it, since a driver reading and a measured slope are not the same "
          "claim",
          "nvidia-smi" in str(_sheets[0][1].notes.get("read_by", "")),
          str(_sheets[0][1].notes))
    # The check that makes the bootstrap safe: a stated count that the node contradicts is refused
    # before the run measures anything: every placement already printed was sized from it.
    _wrong = hardware_for("p6.48xlarge", _ix, gpu_count=4, gpu_product="H100")
    try:
        FakeCapture(_invn, _wrong, driver="580.65.06").capture("rbad", _wrong, index=_ix)
        check("a stated count the leased node contradicts is refused, never quietly corrected",
              False)
    except UnknownHardware as exc:
        check("a stated count the leased node contradicts is refused, never quietly corrected",
              "--gpu-count" in str(exc) and "8 GPU(s)" in str(exc), str(exc)[:80])
    check("  and nothing was filed for the type it refused, so a bad claim leaves no trace",
          _ix.read_shape("p6.48xlarge") is None)
    # A second capture of the same machine files the same facts, which must not read as two cards.
    _again = FakeCapture(_invn, _resolved, driver="580.65.06").capture(
        "ragain", _resolved, index=_ix, image=DEFAULT_IMAGE)
    check("capturing the same machine twice files the same facts, not a disagreement",
          _again.hardware.device_total_gib == _NEWCARD
          and _ix.device_total("h100").total_gib == _NEWCARD)

# The UUID ConfigMap name is derived from the release, not frozen. It was a constant while there
# was one hand-installed daemon, and a constant now reads ANOTHER run's UUIDs.
check("the GPU-UUID ConfigMap name follows the daemon's release name",
      fleetfile.node_name_to_configmap("ip-1-2.compute.internal", "mps-rtest")
      == "mps-rtest-gpu-uuids-ip-1-2-compute-internal")
check("  and defaults to the hand-installed daemon's, which is what a human has installed",
      fleetfile.uuid_configmap_prefix() == "mps-daemon-gpu-uuids-")

# --- agent: run-scoped release names, and the guard that keys off them ---------
# The prefix is not cosmetic. `helm upgrade --install` does not treat a name collision as an
# error: it adopts the existing release and re-pins its pod to this run's node, so a second run
# silently moves the first run's tenant out from under it.
_pref_rel = render_releases(pinned["disjoint"].placement, FLEET, prefix="rtest")
check("a run prefix reaches every tenant release name",
      [n for n, _ in _pref_rel] == ["rtest-m1", "rtest-r1", "rtest-r2", "rtest-t2"],
      str([n for n, _ in _pref_rel]))
check("  and the Service DNS name the harness is given, since the chart derives it from the "
      "release", render.service_url(FLEET.tenant("m1"), prefix="rtest")
      .startswith("http://rtest-m1-"))
_pref_cfg = render_run_config(pinned["disjoint"].placement, FLEET, "run-x", "hash-x",
                              prefix="rtest")
check("  so the values and the run config still agree about every GPU UUID",
      agree(_pref_rel, _pref_cfg, "rtest",
            placement=pinned["disjoint"].placement) is None)
try:
    # A prefix mismatch cannot weaken the check: every lookup misses and every release is
    # reported, rather than the two halves quietly being compared against nothing.
    agree(_pref_rel, _pref_cfg, "rother", placement=pinned["disjoint"].placement)
    check("  and a prefix that disagrees fails loudly instead of matching nothing", False)
except RenderError:
    check("  and a prefix that disagrees fails loudly instead of matching nothing", True)
check("the tenant name in the run config stays unprefixed, because it is not a release",
      {t["name"] for t in _pref_cfg["tenants"]} == {"m1", "r1", "r2", "t2"})

for _bad, _why in ((("mps-daemon", "", ""), "a hand-installed daemon stays protected by name"),
                   (("mps-rtest", "mps-rtest", "rtest"), "this run's OWN lease is refused too"),
                   (("rother-m1", "mps-rtest", "rtest"), "another run's tenant is refused")):
    try:
        guard_release(*_bad)
        check(f"  {_why}", False)
    except DeployError:
        check(f"  {_why}", True)
check("  while this run's own tenant is allowed",
      guard_release("rtest-m1", "mps-rtest", "rtest") is None)

# The loop takes the prefix from the lease and nowhere else, so the deployer's guard and the
# release names cannot disagree.
with tempfile.TemporaryDirectory() as _ptd:
    _pstore = ObservationStore(_ptd, AGENT_KEY)
    _pstore.write_key()
    try:
        SearchAgent(FLEET, _pstore, FakeDeployer(), FakeHarness(fake_record),
                    render_kwargs={"prefix": "rtest"})
        check("the prefix cannot be passed to the loop by hand -- it comes from the lease", False)
    except ValueError:
        check("the prefix cannot be passed to the loop by hand -- it comes from the lease", True)
    _leased_agent = SearchAgent(
        FLEET, _pstore,
        FakeDeployer(lease_release=LEASE.release, tenant_prefix=LEASE.tenant_prefix),
        FakeHarness(fake_record), lease=LEASE)
    _leased_attempt = _leased_agent.execute(
        Proposal(pinned["disjoint"], "disjoint", "a fixed placement, in a test"), 0)
    check("a search holding a lease installs its tenants under the lease's prefix",
          sorted(n for n, _ in _leased_agent.deployer.applied[0])
          == ["rtest-m1", "rtest-r1", "rtest-r2", "rtest-t2"])
    check("  and its run_id carries the token, so two runs sharing a store cannot overwrite each "
          "other", "rtest" in _leased_attempt.run_id, _leased_attempt.run_id)
    check("  and the record says which lease produced it, not just which hostname",
          (_leased_attempt.record["provenance"]["run_token"],
           _leased_attempt.record["provenance"]["daemon_release"]) == ("rtest", "mps-rtest"))

# --- agent: `profile`, which is where a minimal TP degree actually comes from ---
# The block above resolved TP from rows somebody had put in the store by hand. This is the mode that
# puts them there: one tenant, alone, at ascending degrees, stopping at the first one that meets its
# target. Driven through `FakeDeployer`/`FakeHarness`, so the ordering, the stopping rule, the
# budget and what gets written are all checkable with no cluster -- which matters more here than
# elsewhere, since every round of the real thing is a deploy-and-drive cycle on a held node.
_prof_tenant = dataclasses.replace(_leased["fleet"].tenants[0], slo=TenantSlo(itl_mean_ms=8.0))
_prof_fleet = dataclasses.replace(_leased["fleet"], tenants=(_prof_tenant,))
_deployed: list = []


def _tp_record(run_config: dict, run_id: str) -> dict:
    """A harness whose ITL depends on the TP degree: tp1 misses 8 ms, anything wider meets it."""
    rec = fake_record(run_config, run_id)
    for t in run_config["tenants"]:
        rec["per_tenant"][t["name"]]["itl_mean_ms"] = 11.5 if t["tp"] == 1 else 5.4
        rec["per_gpu"] = {u: {"fb_used_mean": (float(t["envelope_gib"]) + 1.5) * 1024.0}
                          for u in t["gpu_uuids"]}
    return rec


def _make_agent(solo, store, skey, builder=_tp_record):
    dep = FakeDeployer(lease_release=LEASE.release, tenant_prefix=LEASE.tenant_prefix)
    _deployed.append(dep)
    return SearchAgent(solo, store, dep, FakeHarness(builder), lease=LEASE,
                       run_prefix="profile", ready_timeout_s=1.0, measure_timeout_s=1.0)


# The narrowing itself, before anything is deployed. A round is keyed like any other run -- same
# environment, one tenant -- so its record lands in the store a later reader would look for it in.
_skey = agent_profiler.solo_key(_leased["key"], _prof_tenant, 4)
check("a solo round keys to one tenant at one degree, inheriting the whole environment",
      (_skey.tensor_parallel, _skey.sequence_shapes, _skey.offered_rates_hz)
      == ({"m1": 4}, {"m1": (512, 128, 0, 0)}, {"m1": 16.0})
      and _skey.env == _leased["key"].env)
check("  and a different degree is a different validity key, which is why the profile store exists",
      agent_profiler.solo_key(_leased["key"], _prof_tenant, 2).hash != _skey.hash
      and ProfileKey.from_validity_key(_skey).hash
      == ProfileKey.from_validity_key(agent_profiler.solo_key(_leased["key"], _prof_tenant, 2)).hash)
check("  and the solo fleet keeps the delivery gate, which decides if its latency may be read",
      agent_profiler.solo_fleet(_prof_fleet, _prof_tenant, 2).workload
      .delivered_fraction_threshold == _prof_fleet.workload.delivered_fraction_threshold)

# The reserve derivation, on a record built by hand so every input is known. It is `fb_used -
# util x usable` averaged over the shard's GPUs -- the same derivation the mps_reserve_gib
# calibration used, so the two are comparable -- and it is measurable ONLY because the tenant is
# alone: with a co-tenant, fb_used is the sum of both and MPS attributes nothing.
_fbrec = {"configuration": {"tenants": [{"name": "m1", "gpu_uuids": ["GPU-a", "GPU-b"],
                                         "envelope_gib": 40}]},
          "per_gpu": {"GPU-a": {"fb_used_mean": (40.0 + 1.6) * 1024.0},
                      "GPU-b": {"fb_used_mean": (40.0 + 1.4) * 1024.0}}}
check("the measured reserve is what the engine used beyond its own envelope, per shard",
      agent_profiler.measured_reserve_gib(_fbrec, "m1", CARD_GIB, util=40.0 / CARD_GIB) == 1.5)
check("  and a record with no gauges for those GPUs reports None, not a reassuring zero",
      agent_profiler.measured_reserve_gib({**_fbrec, "per_gpu": {}}, "m1", CARD_GIB, util=0.9) is None
      and agent_profiler.measured_reserve_gib(_fbrec, "absent", CARD_GIB, util=0.9) is None)
# A round whose resident memory came in at or under the budget measured no overhead at all. That is
# not a reserve of zero and certainly not a negative one: it is a round that put no lower bound on
# the overhead, and the row has to say nothing rather than offer `for_fleet_file` a number a person
# would paste and the solver would read as extra room.
_under = {"configuration": {"tenants": [{"name": "m1", "gpu_uuids": ["GPU-a", "GPU-b"],
                                         "envelope_gib": 40}]},
          "per_gpu": {"GPU-a": {"fb_used_mean": 39.0 * 1024.0},
                      "GPU-b": {"fb_used_mean": 39.4 * 1024.0}}}
check("  and a round that used LESS than its budget states no reserve, never a negative one",
      agent_profiler.measured_reserve_gib(_under, "m1", CARD_GIB, util=40.0 / CARD_GIB) is None)
check("    which is a statement about that round, not a floor of zero, so the default still stands",
      agent_profiler.profile_from_record(
          {**_under, "verdict": VERDICT_MEASURED}, _prof_tenant, 2, envelope_gib=40.0,
          device_total_gib=CARD_GIB, util=40.0 / CARD_GIB, run_id="under").reserve_gib is None)

with tempfile.TemporaryDirectory() as _ptd:
    _pstore2 = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _prof = agent_profiler.SoloProfiler(_prof_fleet, _leased["key"], HW, _pstore2,
                                        make_agent=_make_agent, store_root=Path(_ptd) / "obs",
                                        solver_seconds=5.0)
    check("profiling starts at the memory floor, because nothing below it fits on one GPU at all",
          _prof.degrees_for(_prof_tenant) == (1, 2, 4, 8)
          and _prof.floor_for(_prof_tenant).floor_tp == 1)
    _out = _prof.run()
    check("a round is spent per degree, narrowest first, and STOPS at the first one that passes",
          [(r.tenant, r.tp, r.status) for r in _out["rounds"]]
          == [("m1", 1, agent_profiler.DID_MEASURE), ("m1", 2, agent_profiler.DID_MEASURE)]
          and _out["spent"] == 2, str([(r.tp, r.status) for r in _out["rounds"]]))
    _res = _out["resolutions"]["m1"]
    check("  so minimal TP is tp2, and it is minimal because tp1 was MEASURED to miss",
          (_res.status, _res.tp, _res.certain) == (RESOLVED, 2, True)
          and _res.below == (
              (1, MISSED, "missed: 11.500 ms ITL solo over a 8.000 ms ceiling"),),
          str(_res.as_dict()))
    check("  and each round deployed exactly one release, at the lease's prefix and its own degree",
          [len(d.applied[0]) for d in _deployed] == [1, 1]
          and [d.applied[0][0][0] for d in _deployed] == ["rtest-m1", "rtest-m1"])
    _rows = {p.tp: p for p in _pstore2.all()}
    check("the rows land in the profile store, one per degree, each naming its own run",
          sorted(_rows) == [1, 2] and _rows[1].run_id != _rows[2].run_id)
    # A row's `gpu_node` is the machine the measurement was taken on, so it has to come from
    # `provenance.gpu_node` and not from the record's top-level `node`, which is the load driver's.
    # The fixture keeps the two names different precisely so this can fail.
    check("  and each row names the GPU node it was measured on, not the harness Job's node",
          all(p.gpu_node == LEASE.node_name for p in _rows.values())
          and _rows[2].gpu_node != "cpu-driver-node",
          str({t: p.gpu_node for t, p in _rows.items()}))
    # A synthesized record -- a deploy that never came ready, so no harness ran -- has no
    # provenance, and its top-level `node` IS the GPU node, because the agent wrote it from the
    # lease. That one falls back rather than filing a row with no node at all.
    check("  while a record with no provenance falls back to its one node, which is the GPU's",
          agent_profiler.profile_from_record(
              {"node": "held-gpu-node", "per_gpu": {}, "verdict": VERDICT_MEASURED,
               "configuration": {"tenants": [{"name": "m1", "gpu_uuids": ["GPU-a"]}]}},
              _prof_tenant, 2, envelope_gib=40.0, device_total_gib=HW.device_total_gib,
              run_id="synth").gpu_node == "held-gpu-node")
    check("  and each names the validity key of the measurement behind it, which differs by degree",
          _rows[1].observation_key_hash != _rows[2].observation_key_hash
          and _rows[2].observation_key_hash
          == agent_profiler.solo_key(_leased["key"], _prof_tenant, 2).hash)
    check("  and the observation itself is filed under that key, append-only, not in the row",
          ObservationStore(Path(_ptd) / "obs",
                           agent_profiler.solo_key(_leased["key"], _prof_tenant, 2)
                           ).load().observations[0].run_id == _rows[2].run_id)
    check("  and the row says it was taken solo and how its reserve was derived, since neither is "
          "recoverable from the number", _rows[2].notes["solo"] is True
          and "fb_used_mean" in _rows[2].notes["reserve_derivation"]
          and _rows[2].reserve_gib is not None, str(_rows[2].reserve_gib))
    # Filed per degree, and from here the loader merges it into every fleet that names this model on
    # this hardware (impl §14.4). The tenant in hand still carries None because it was built from a
    # fleet file loaded *before* these rounds ran: the merge happens at load, so the figure reaches a
    # plan on the next load rather than mutating the fleet the measurement came from.
    check("  and the measured reserve is filed against the model, per degree, for the loader",
          set(_pstore2.reserve_by_tp(_prof_tenant.model_id)) == {1, 2}
          and _prof_tenant.mem.measured_reserve_gib is None,
          str(_pstore2.reserve_by_tp(_prof_tenant.model_id)))
    # The mode ends with the node clear. Every round but the last is replaced by the next one's
    # install, so the leak is exactly one release -- the tenant that happened to pass -- and it
    # holds its whole envelope on gpu0..gpu(tp-1) until somebody notices. Unlike a search, there is
    # nothing left to look at: the row is written, so keeping the pod up buys a diagnostic nobody
    # wants and costs the memory the next round is measured against.
    check("  and the run leaves no tenant up, so the next round is not measured around this one",
          _deployed[-1].live_tenant_releases() == []
          and _deployed[-1].destroyed == [["rtest-m1"]],
          str([(d.live_tenant_releases(), d.destroyed) for d in _deployed]))

    # The point of a store keyed by the environment rather than by the fleet: the second run of the
    # mode costs nothing, so adding a tenant to a fleet is cheap and re-running is not a mistake.
    _deployed.clear()
    _again = agent_profiler.SoloProfiler(_prof_fleet, _leased["key"], HW, _pstore2,
                                        make_agent=_make_agent, store_root=Path(_ptd) / "obs")
    _out2 = _again.run()
    check("re-running the mode re-measures nothing it already has, and still stops at tp2",
          _out2["spent"] == 0 and _deployed == []
          and [(r.tp, r.status) for r in _out2["rounds"]]
          == [(1, agent_profiler.ALREADY_PROFILED), (2, agent_profiler.ALREADY_PROFILED)]
          and _out2["resolutions"]["m1"].tp == 2)
    _reprof = agent_profiler.SoloProfiler(_prof_fleet, _leased["key"], HW, _pstore2,
                                         make_agent=_make_agent, store_root=Path(_ptd) / "obs",
                                         reprofile=True)
    _out3 = _reprof.run()
    check("  while --reprofile measures again and REPLACES the rows rather than accumulating them",
          _out3["spent"] == 2 and len(_pstore2.all()) == 2
          and {p.tp: p.run_id for p in _pstore2.all()} != {t: p.run_id
                                                           for t, p in _rows.items()})

with tempfile.TemporaryDirectory() as _ptd:
    # The caller can still say leave it up, which is what `search` does. The flag exists so that
    # "leave it up" is a decision somebody made, not what happens when nobody thought about it.
    _keep = agent_profiler.SoloProfiler(_prof_fleet, _leased["key"], HW,
                                        ProfileStore(Path(_ptd) / "profiles", _pkey),
                                        make_agent=_make_agent, store_root=Path(_ptd) / "obs")
    _deployed.clear()
    _keep.run(teardown_at_end=False)
    check("teardown_at_end=False leaves the last round's tenant serving, deliberately",
          _deployed[-1].live_tenant_releases() == ["rtest-m1"] and _deployed[-1].destroyed == [],
          str([(d.live_tenant_releases(), d.destroyed) for d in _deployed]))

with tempfile.TemporaryDirectory() as _ptd:
    # A budget stops the run without pretending it answered. tp1 is measured and misses; tp2 is
    # never reached, so the resolution says so instead of returning the degree that missed.
    _pstore3 = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _capped = agent_profiler.SoloProfiler(_prof_fleet, _leased["key"], HW, _pstore3,
                                          make_agent=_make_agent, store_root=Path(_ptd) / "obs",
                                          budget=1)
    _out4 = _capped.run()
    check("a spent budget leaves the answer open rather than settling for the degree that missed",
          _out4["spent"] == 1 and _out4["resolutions"]["m1"].status == UNPROFILED
          and _out4["resolutions"]["m1"].tp is None
          and _out4["resolutions"]["m1"].missing == (2, 4, 8),
          str(_out4["resolutions"]["m1"].as_dict()))

with tempfile.TemporaryDirectory() as _ptd:
    # A round whose delivery gate failed measured nothing about latency (impl §6.3). It must not
    # rule its degree out: that would widen a tenant's shard count on the strength of a measurement
    # that did not happen.
    def _ungated(run_config: dict, run_id: str) -> dict:
        rec = _tp_record(run_config, run_id)
        for name in rec["per_tenant"]:
            rec["per_tenant"][name].update(delivered_fraction=0.42, itl_mean_ms=None,
                                           latency_suppressed="delivered 0.42 of 1.0")
        return rec

    _pstore4 = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _ungated_prof = agent_profiler.SoloProfiler(
        _prof_fleet, _leased["key"], HW, _pstore4, store_root=Path(_ptd) / "obs", budget=1,
        make_agent=lambda solo, store, skey: _make_agent(solo, store, skey, builder=_ungated))
    _out5 = _ungated_prof.run()
    check("a round the delivery gate failed is unmeasured, not a miss -- so it rules nothing out",
          _out5["rounds"][0].status == agent_profiler.DID_MEASURE
          and _out5["rounds"][0].profile.usable is False
          and _out5["rounds"][0].meets(_prof_tenant.slo) is None
          and _out5["resolutions"]["m1"].missing == (1, 2, 4, 8),
          str(_out5["resolutions"]["m1"].as_dict()))

with tempfile.TemporaryDirectory() as _ptd:
    # A round the harness declined to vouch for, whose latency happens to miss the target. Ascending
    # is earned by a MEASURED miss and nothing else: this round did not say the degree is too slow,
    # it said the degree was not measured, and going wider on it spends a GPU round to hand the
    # baseline a degree nothing ruled the narrower one out at. No budget here, so stopping is the
    # profiler's own decision rather than the cap's.
    def _unvouched(run_config: dict, run_id: str) -> dict:
        return {**_tp_record(run_config, run_id), "verdict": VERDICT_NO_STEADY_STATE}

    _pstore6 = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _unv_prof = agent_profiler.SoloProfiler(
        _prof_fleet, _leased["key"], HW, _pstore6, store_root=Path(_ptd) / "obs",
        make_agent=lambda solo, store, skey: _make_agent(solo, store, skey, builder=_unvouched))
    _out7 = _unv_prof.run()
    check("an unvouched round STOPS the sweep instead of spending the next degree on it",
          _out7["spent"] == 1 and [(r.tp, r.status) for r in _out7["rounds"]]
          == [(1, agent_profiler.DID_MEASURE)],
          str([(r.tp, r.status) for r in _out7["rounds"]]))
    check("  and the answer stays open, with the verdict named as the thing to fix",
          _out7["resolutions"]["m1"].status == UNPROFILED
          and _out7["resolutions"]["m1"].tp is None
          and _out7["resolutions"]["m1"].missing == (1, 2, 4, 8)
          and _out7["resolutions"]["m1"].below[0][1] == UNUSABLE_VERDICT,
          str(_out7["resolutions"]["m1"].as_dict()))

with tempfile.TemporaryDirectory() as _ptd:
    # The walk as the authority. tp1 is driven into overload -- flat out, nothing delivered, nothing
    # failed -- which under the old single rule read as "not measured" and stopped the sweep on the
    # most conclusive round it had. It now widens, and tp2 answers. The narrow degree is still never
    # cited for latency: it is ruled out, not measured.
    def _overloaded(run_config: dict, run_id: str) -> dict:
        rec = _tp_record(run_config, run_id)
        for t in run_config["tenants"]:
            if t["tp"] != 1:
                continue
            rec["verdict"] = VERDICT_DELIVERY
            rec["per_tenant"][t["name"]].update(
                itl_mean_ms=None, delivered_fraction=0.0, delivered_fraction_threshold=0.95,
                latency_suppressed="delivered 0.0 of 0.95: survivorship bias (impl §6.3)",
                requests_launched=300, requests_dispatched=0, requests_completed=0,
                requests_failed=0, schedule_lag_max_ms=4.0,
                # The backlog, from the driver's own in-flight level rather than from the server's
                # queue depth and KV high-water mark, which is what this used to carry (impl §14.6).
                in_flight_mean=61.4, in_flight_max=123.0, in_flight_samples=30,
                in_flight_first_quarter_mean=17.2, in_flight_last_quarter_mean=112.5)
            rec["workload"] = {"tenants": {t["name"]: {"isl_exact": True, "isl_calibrated": True}}}
            for u in t["gpu_uuids"]:
                rec["per_gpu"].setdefault(u, {}).update(gr_engine_active_mean=0.9949,
                                                        sm_active_mean=0.8879)
        return rec

    _pstore7 = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _ovl = agent_profiler.SoloProfiler(
        _prof_fleet, _leased["key"], HW, _pstore7, store_root=Path(_ptd) / "obs",
        make_agent=lambda solo, store, skey: _make_agent(solo, store, skey, builder=_overloaded))
    _out8 = _ovl.run()
    check("a round driven into overload WIDENS the sweep: delivering nothing is too narrow",
          [(r.tp, r.status) for r in _out8["rounds"]]
          == [(1, agent_profiler.DID_MEASURE), (2, agent_profiler.DID_MEASURE)],
          str([(r.tp, r.status) for r in _out8["rounds"]]))
    check("  and the degree above it is a MEASURED minimum, not an upper bound, because tp1 was "
          "ruled out on evidence rather than left unmeasured",
          _out8["resolutions"]["m1"].status == RESOLVED
          and _out8["resolutions"]["m1"].tp == 2
          and _out8["resolutions"]["m1"].certain is True
          and _out8["resolutions"]["m1"].missing == (),
          str(_out8["resolutions"]["m1"].as_dict()))
    check("  and tp1 is recorded as too narrow, in its own words, not as a gap in the sweep",
          _out8["resolutions"]["m1"].below[0][:2] == (1, TOO_NARROW)
          and "did not keep up" in _out8["resolutions"]["m1"].below[0][2],
          str(_out8["resolutions"]["m1"].below))
    check("  and the row it widened on is STILL unusable for latency, so nothing quotes it",
          _out8["rounds"][0].profile.usable is False
          and _out8["rounds"][0].meets(_prof_tenant.slo) is None)
    check("  and the evidence it widened on is in the row, so a later run can re-judge it without "
          "the record", _pstore7.read(_prof_tenant.model_id, 1, _prof_tenant.isl_tokens,
                                      _prof_tenant.osl_tokens, _prof_tenant.rate_hz,
                                      _prof_tenant.serving).notes["saturation"]
          ["in_flight_last_quarter_mean"] == 112.5)
    check("    and not one number in that evidence came off the server: the row carries none of the "
          "four fields the walk used to widen on",
          not (set(_pstore7.read(_prof_tenant.model_id, 1, _prof_tenant.isl_tokens,
                                 _prof_tenant.osl_tokens, _prof_tenant.rate_hz,
                                 _prof_tenant.serving).notes["saturation"])
               & {"num_requests_waiting_mean", "num_requests_running_mean", "kv_usage_max",
                  "preemptions"}),
          str(sorted(_pstore7.read(_prof_tenant.model_id, 1, _prof_tenant.isl_tokens,
                                   _prof_tenant.osl_tokens, _prof_tenant.rate_hz,
                                   _prof_tenant.serving).notes["saturation"])))

# --- a boot that failed for lack of room widens the walk, but only once it has REPEATED ---------
# The arithmetic's soft terms -- the activation fallback, the reserve, the KV page size -- are the
# reason a degree may not be eliminated on paper. The other side of that bargain is that when the
# round IS spent, its failure has to count: a tenant deployed alone at the full envelope, whose MPS
# limit was applied (the verification init container passed) and whose engine still could not come
# up, has produced `executed` evidence that this degree is too narrow. That widens.
#
# But one failed boot is not that evidence, because a failed boot is the one outcome in this walk
# whose cause is ambiguous between the degree and the run, and the two want opposite answers: too
# narrow means widen, a transient fault means measure this same degree again. The engine's own
# account of why it died is not read and would not settle it anyway -- an out-of-memory abort and a
# failed weight load both arrive as a nonzero exit. So the walk buys the distinguishing evidence: it
# repeats the degree once. A shard genuinely too narrow fails identically the second time and the
# widening happens one round later with the evidence in hand; a fault does not repeat, and the retry
# measures the degree the fleet asked for.
#
# Everything else the readiness gate can say is about the RUN and not the degree, and stops.
def _gate_agent(solo, store, skey, *, states, seen=None, fail_first_only=False):
    """An agent whose readiness gate returns `states[tp]` for this round's degree.

    With `fail_first_only`, the scripted state applies to the FIRST visit to that degree only, so a
    test can distinguish a shard too narrow (fails every time) from a transient fault (does not).
    """
    tp = solo.tenants[0].tp
    dep = FakeDeployer(lease_release=LEASE.release, tenant_prefix=LEASE.tenant_prefix)
    state = states.get(tp)
    if seen is not None:
        seen[tp] = seen.get(tp, 0) + 1
    if state is not None and not (fail_first_only and seen and seen.get(tp, 0) > 1):
        dep.outcomes[frozenset({f"{LEASE.tenant_prefix}-m1"})] = state
    _deployed.append(dep)
    return SearchAgent(solo, store, dep, FakeHarness(_tp_record), lease=LEASE,
                       run_prefix="profile", ready_timeout_s=1.0, measure_timeout_s=1.0)


with tempfile.TemporaryDirectory() as _ptd:
    _pstore8 = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _eng = agent_profiler.SoloProfiler(
        _prof_fleet, _leased["key"], HW, _pstore8, store_root=Path(_ptd) / "obs",
        make_agent=lambda s, st, k: _gate_agent(s, st, k, states={1: NOT_READY_ENGINE}))
    _out9 = _eng.run()
    check("an engine that could not come up in its own envelope is MEASURED AGAIN at the same "
          "degree before the walk will widen on it",
          [(r.tp, r.status, r.gate) for r in _out9["rounds"]][:2]
          == [(1, agent_profiler.NOT_READY, NOT_READY_ENGINE),
              (1, agent_profiler.NOT_READY, NOT_READY_ENGINE)],
          str([(r.tp, r.status, r.gate) for r in _out9["rounds"]]))
    check("  and once it HAS failed the same way twice, that widens -- a boot is the only thing "
          "that can settle the terms the arithmetic may not eliminate on",
          [(r.tp, r.status, r.gate) for r in _out9["rounds"]]
          == [(1, agent_profiler.NOT_READY, NOT_READY_ENGINE),
              (1, agent_profiler.NOT_READY, NOT_READY_ENGINE),
              (2, agent_profiler.DID_MEASURE, "")],
          str([(r.tp, r.status, r.gate) for r in _out9["rounds"]]))
    check("  and the repeat is bought exactly once, not per degree visited and not twice",
          sum(1 for r in _out9["rounds"] if r.tp == 1) == 2)
    check("  and tp2 is where it lands -- but as an UPPER BOUND, since tp1 has no row and so no "
          "stored evidence a later reader could re-judge",
          (_out9["resolutions"]["m1"].status, _out9["resolutions"]["m1"].tp,
           _out9["resolutions"]["m1"].certain) == (RESOLVED, 2, False)
          and _out9["resolutions"]["m1"].missing == (1,),
          str(_out9["resolutions"]["m1"].as_dict()))
    check("    and no row was written for the failed boot, deliberately: a permanent 'too narrow' "
          "row would never be retried, and the fix may be a chart or an image",
          sorted(p.tp for p in _pstore8.all()) == [2])
    check("  and every failed round cost a measurement, because each held the node to find out",
          _out9["spent"] == 3)

# The other half of the same rule, and the reason it is worth a round: a boot failure that does NOT
# repeat was never about the shard count. Measured on 2026-09-21 -- `qwen35-multimodal-2`'s `docqa`
# failed to boot at tp1 once, the walk read it as evidence and widened to tp2, tp4 and tp8 (three
# rounds, ~45 min of g6e) which all came up and then stalled. Retried at tp1 it measured 56.253 ms
# against a 60 ms target. Here the retry finds tp1 slow rather than absent, which is the point: tp2
# is still where it lands, but now with tp1 RULED OUT on a row instead of skipped.
with tempfile.TemporaryDirectory() as _ptd:
    _pstore8b = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _seen: dict = {}
    _flaky = agent_profiler.SoloProfiler(
        _prof_fleet, _leased["key"], HW, _pstore8b, store_root=Path(_ptd) / "obs",
        make_agent=lambda s, st, k: _gate_agent(s, st, k, states={1: NOT_READY_ENGINE},
                                                seen=_seen, fail_first_only=True))
    _out9b = _flaky.run()
    check("a boot failure that does not repeat was never evidence about the shard: the retry "
          "measures the degree instead of the walk widening past it",
          [(r.tp, r.status, r.gate) for r in _out9b["rounds"]]
          == [(1, agent_profiler.NOT_READY, NOT_READY_ENGINE),
              (1, agent_profiler.DID_MEASURE, ""),
              (2, agent_profiler.DID_MEASURE, "")],
          str([(r.tp, r.status, r.gate) for r in _out9b["rounds"]]))
    check("  and tp1 now has a row, so the degree was ruled out on latency rather than skipped",
          sorted(p.tp for p in _pstore8b.all()) == [1, 2]
          and _pstore8b.read(_prof_tenant.model_id, 1, _prof_tenant.isl_tokens,
                             _prof_tenant.osl_tokens, _prof_tenant.rate_hz,
                             _prof_tenant.serving).itl_mean_ms == 11.5,
          str(sorted(p.tp for p in _pstore8b.all())))
    check("    which is what the round bought: tp2 is CERTAIN here and an upper bound when the "
          "boot failure repeats, on the same landing degree",
          (_out9b["resolutions"]["m1"].tp, _out9b["resolutions"]["m1"].certain,
           _out9b["resolutions"]["m1"].missing) == (2, True, ()),
          str(_out9b["resolutions"]["m1"].as_dict()))

# And the retry is spent from the same budget as everything else, so a budget of one cannot be
# turned into two rounds by a failed boot. It stops short of widening instead, which is the safe
# direction: an unmeasured degree leaves the tenant unresolved, while a widened one would hand the
# baseline a shard nothing ruled the narrower one out at.
with tempfile.TemporaryDirectory() as _ptd:
    _b1 = agent_profiler.SoloProfiler(
        _prof_fleet, _leased["key"], HW, ProfileStore(Path(_ptd) / "profiles", _pkey),
        store_root=Path(_ptd) / "obs",
        make_agent=lambda s, st, k: _gate_agent(s, st, k, states={1: NOT_READY_ENGINE}))
    _out9c = _b1.run(budget=1)
    check("the retry comes out of the stated budget, so --budget 1 buys one round and not two",
          _out9c["spent"] == 1
          and [(r.tp, r.gate) for r in _out9c["rounds"]] == [(1, NOT_READY_ENGINE)]
          and _out9c["resolutions"]["m1"].status == UNPROFILED,
          str((_out9c["spent"], [(r.tp, r.gate) for r in _out9c["rounds"]])))

for _gate in (NOT_READY_VERIFY, NOT_READY_TIMEOUT, NOT_READY_SCHEDULE):
    with tempfile.TemporaryDirectory() as _ptd:
        # None of these three deployed an engine into the envelope, so none of them tested the
        # degree. Widening on one would hand the baseline a wider shard on the strength of a broken
        # run -- the §0 error `tp:` was deleted for, arrived at from the deploy side.
        _stop = agent_profiler.SoloProfiler(
            _prof_fleet, _leased["key"], HW, ProfileStore(Path(_ptd) / "profiles", _pkey),
            store_root=Path(_ptd) / "obs",
            make_agent=lambda s, st, k, _g=_gate: _gate_agent(s, st, k, states={1: _g}))
        _outs = _stop.run()
        check(f"a round that did not come up as {_gate!r} STOPS the walk: nothing about the degree "
              "was tested, so a wider one is unearned",
              [(r.tp, r.status, r.gate) for r in _outs["rounds"]]
              == [(1, agent_profiler.NOT_READY, _gate)]
              and _outs["resolutions"]["m1"].status == UNPROFILED
              and _outs["resolutions"]["m1"].tp is None
              and _outs["resolutions"]["m1"].missing == (1, 2, 4, 8),
              str([(r.tp, r.status, r.gate) for r in _outs["rounds"]])
              + str(_outs["resolutions"]["m1"].as_dict()))

with tempfile.TemporaryDirectory() as _ptd:
    # A degree the arithmetic will not place deployed nothing and held no node, so it must not be
    # charged to the budget. This is the tenant the "doubted band" is widest for: its WEIGHTS fit
    # from tp4 up, so tp4 and tp8 are both legal degrees, and its computed envelope fits at neither
    # -- 2M KV tokens against a 176 GiB model. The pool comes from the CONTEXT, which is where a
    # minimum viable pool is derived from (`MIN_POOL_REQUESTS` x `max_model_len`, impl §14.9): a 1M
    # context is a 2M-token floor. The walk therefore starts at the widest degree the node offers
    # (`start_tp_for` with no memory floor) and the round there is refused by arithmetic.
    _big = dataclasses.replace(
        _prof_tenant, model_id="wide/model", max_model_len=1_000_000,
        mem=dataclasses.replace(_prof_tenant.mem, model_id="wide/model",
                                weight_gib=4.0 * 44.0, activation_gib=None))
    _bigp = agent_profiler.SoloProfiler(
        dataclasses.replace(_prof_fleet, tenants=(_big,)), _leased["key"], HW,
        ProfileStore(Path(_ptd) / "profiles", _pkey), make_agent=_make_agent,
        store_root=Path(_ptd) / "obs", budget=1)
    check("  a tenant whose weights fit but whose computed envelope fits nowhere still has degrees "
          "to walk, starting at the widest the node offers",
          (_bigp.degrees_for(_big), _bigp.floor_for(_big).floor_tp, _bigp.ascent_for(_big))
          == ((4, 8), None, (8,)),
          str((_bigp.degrees_for(_big), _bigp.floor_for(_big).floor_tp)))
    _out10 = _bigp.run()
    check("  and a degree the arithmetic will not place is NOT charged to the budget, because it "
          "deployed nothing and held no node",
          [(r.tp, r.status) for r in _out10["rounds"]] == [(8, agent_profiler.NOT_PLACEABLE)]
          and _out10["spent"] == 0,
          str([(r.tp, r.status) for r in _out10["rounds"]]) + f" spent={_out10['spent']}")
    check("    and the tenant is left unresolved rather than settled on arithmetic nobody measured",
          _out10["resolutions"]["m1"].status == UNPROFILED
          and _out10["resolutions"]["m1"].tp is None,
          str(_out10["resolutions"]["m1"].as_dict()))

# --- the second question: does a row with no latency still rule its degree out? --------------
# `unusable_reason` asks whether a row may be cited about latency, and the answer for an overloaded
# round is no. That is not the same as the row saying nothing: a server driven flat out that
# completed nothing is the strongest possible evidence that one shard cannot hold the work. Both
# questions are asked, and only the first one is allowed to set a minimum.
#
# Built from a real record's numbers (the 9B docqa tenant, one L40S, 24 megapixel images) rather
# than invented ones, because the shape that matters here is the one a real overload produces --
# including `requests_dispatched: 0`, which is what a driver reports when every request it sent was
# still streaming when the window closed.
#
# The four engine gauges that used to sit in here -- 93.8 requests waiting, 29.6 running, KV at
# 0.9892, zero preemptions -- are gone (impl §14.6). The backlog they described is the same one the
# driver saw as its in-flight count climbing from 17 to 112 over the window, and the second number is
# available from any server that speaks HTTP.
_SAT_FULL = {"requests_launched": 300, "requests_dispatched": 0, "requests_completed": 0,
             "requests_failed": 0, "schedule_lag_max_ms": 4.0, "isl_exact": True,
             "isl_calibrated": True,
             "in_flight_mean": 61.4, "in_flight_max": 123.0, "in_flight_samples": 30,
             "in_flight_first_quarter_mean": 17.2, "in_flight_last_quarter_mean": 112.5,
             "gr_engine_active_mean": 0.9949, "sm_active_mean": 0.8879}


def _narrow(**over) -> str | None:
    """`too_narrow_reason` for an overloaded row with `over` applied to its evidence."""
    ev = {**_SAT_FULL, **over}
    return SoloProfile(model_id="m", tp=1, isl_tokens=1, osl_tokens=1, rate_hz=1.0,
                       envelope_gib=1.0, serving=_prof_tenant.serving,
                       itl_mean_ms=None,
                       notes={"verdict": VERDICT_DELIVERY,
                              "latency_suppressed": "delivered 0.0 of 1.0",
                              "saturation": ev}).too_narrow_reason


check("a saturated round that delivered nothing rules its degree out as TOO NARROW",
      _narrow() is not None and "did not keep up" in _narrow(), str(_narrow()))
check("  while still being unusable for latency, so it sets no minimum and is never quoted as one",
      SoloProfile(model_id="m", tp=1, isl_tokens=1, osl_tokens=1, rate_hz=1.0, envelope_gib=1.0,
                  serving=_prof_tenant.serving, itl_mean_ms=None,
                  notes={"verdict": VERDICT_DELIVERY, "saturation": _SAT_FULL}).usable is False)
# Each of the six grounds, removed one at a time. Every one of them is the difference between a
# server that was working and behind and a run that simply broke, and a broken run must not buy the
# baseline a wider shard -- the same error §0 deleted `tp:` for, reached from the other side.
check("  and it needs the server to have priced the payload itself: no exact ISL, no conclusion",
      _narrow(isl_exact=False) is None and _narrow(isl_exact=None) is None)
check("  and it needs load to have been offered, by the driver's own books and nothing else",
      _narrow(requests_launched=0, requests_dispatched=0) is None)
check("  and a transport failure is the server saying no, not saying slow",
      _narrow(requests_failed=3) is None and _narrow(requests_failed=None) is None)
check("  and a driver that fell behind its own schedule never offered the stated rate",
      _narrow(schedule_lag_max_ms=9_000.0) is None)
check("  and an idle device is not an overloaded one, however fast the backlog grew",
      _narrow(gr_engine_active_mean=0.02, sm_active_mean=0.01) is None)
check("  and a busy device that KEPT UP is just a healthy server at its target",
      _narrow(in_flight_first_quarter_mean=4.0, in_flight_last_quarter_mean=4.0,
              requests_completed=300) is None)
# The busy half is carried by either DCGM field, so a card reporting only one still loses this
# argument.
check("  and DCGM's sm_active alone carries the busy half when gr_engine_active is absent",
      _narrow(gr_engine_active_mean=None) is not None
      and "sm_active_mean" in _narrow(gr_engine_active_mean=None))
# --- the backlog half, after impl §14.6 -------------------------------------------------------
# It used to consult `num_requests_waiting_mean`, then `kv_usage_max`, then a preemption count, and
# only then the client's books -- so the authority to widen a degree rested on gauges one engine
# publishes and another does not. All three are gone, and the two checks below are the replacement's
# two directions: a row where the driver's own in-flight level climbed rules the degree out, and a row
# where it stayed flat does not, however far short of its offered load the row fell.
check("  and the in-flight level climbing is what rules the degree out, with no server reading at all",
      _narrow(requests_completed=280) is not None
      and "climbed from 17.2 to 112" in _narrow(requests_completed=280),
      str(_narrow(requests_completed=280)))
check("    so a degree that is merely falling behind widens the walk, not only one that collapsed",
      _narrow(requests_completed=280) == _narrow(requests_completed=0))
check("  while a FLAT in-flight level settles nothing, even beside a shortfall the old rule missed",
      _narrow(in_flight_first_quarter_mean=8.0, in_flight_last_quarter_mean=8.4,
              requests_completed=200) is None)
check("    and flat with a total collapse still widens, on the client-side shortfall alone",
      _narrow(in_flight_first_quarter_mean=8.0, in_flight_last_quarter_mean=8.4,
              requests_completed=11) is not None
      and "11 of 300 requests completed" in _narrow(
          in_flight_first_quarter_mean=8.0, in_flight_last_quarter_mean=8.4,
          requests_completed=11))
check("    and too thin a series is not a flat one: two samples cannot carry a direction",
      _narrow(in_flight_samples=2, requests_completed=280) is None
      and _narrow(in_flight_samples=None, requests_completed=280) is None)
check("  and a row from before this evidence existed says nothing either way, rather than guessing",
      SoloProfile(model_id="m", tp=1, isl_tokens=1, osl_tokens=1, rate_hz=1.0, envelope_gib=1.0,
                  serving=_prof_tenant.serving, itl_mean_ms=None,
                  notes={"verdict": VERDICT_DELIVERY}).too_narrow_reason is None)
# `requests_launched` is the counter this needed and the harness did not have. `requests_dispatched`
# can only count requests a result came back for, so it reads 0 under total overload -- the record
# that motivated all of this says `requests_dispatched: 0`. A row with neither used to be salvaged
# through the server's own report that it was holding requests nobody but this driver could have sent
# it; that was a server reading, so it is gone, and a row that cannot say what it offered now settles
# nothing (impl §14.6).
check("  and a record that cannot say what it offered settles nothing, with no salvage off the server",
      _narrow(requests_launched=None, requests_dispatched=0) is None)

# A tenant whose memory floor is already the widest degree the node offers is resolved by
# arithmetic, and measuring it would cost a round that could not change the answer.
_huge = dataclasses.replace(_prof_tenant, model_id="big/model", mem=dataclasses.replace(
    _prof_tenant.mem, model_id="big/model", weight_gib=8.0 * 35.0, activation_gib=None))
with tempfile.TemporaryDirectory() as _ptd:
    _pstore5 = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _forced = agent_profiler.SoloProfiler(
        dataclasses.replace(_prof_fleet, tenants=(_huge,)), _leased["key"], HW, _pstore5,
        make_agent=_make_agent, store_root=Path(_ptd) / "obs")
    _deployed.clear()
    _out6 = _forced.run()
    check("a degree forced by memory alone is resolved without spending a round on it",
          _out6["spent"] == 0 and _deployed == []
          and [(r.tp, r.status) for r in _out6["rounds"]]
          == [(8, agent_profiler.FORCED_BY_MEMORY)]
          and (_out6["resolutions"]["m1"].tp, _out6["resolutions"]["m1"].certain) == (8, True),
          str([(r.tp, r.status) for r in _out6["rounds"]]))
    check("  and its row count is zero, because a store of rows nobody needed is a store of "
          "GPU time nobody needed", _pstore5.all() == [])
    # The narrower degrees are ruled out by arithmetic, which is a CHECKED ground: a degree that
    # cannot hold the weights is not a degree that might have been narrower, so it must not turn
    # the answer into an upper bound the way an unmeasured degree does.
    check("  and the degrees it skipped are ruled out by fit, not left as gaps in the sweep",
          _out6["resolutions"]["m1"].missing == ()
          and all(c == DOES_NOT_FIT for _, c, _ in _out6["resolutions"]["m1"].below),
          str(_out6["resolutions"]["m1"].as_dict()["below"]))

    # The mode's report is rendered here rather than trusted, because it crashed once on exactly
    # this shape -- an UNRESOLVED tenant with a non-empty `below` -- AFTER every row had been
    # written. The store was correct and the round's conclusion was reachable only by reading it
    # by hand. `below` entries are triples; the renderer unpacked two.
    from colocation.agent.__main__ import _profile_report                 # noqa: E402
    _unres = TpResolution(tenant="m1", model_id="Qwen/Qwen3-0.6B", status=NONE_MEETS, tp=None,
                          below=((4, MISSED, "missed: 9.000 ms ITL solo over a 8.000 ms ceiling"),
                                 (8, UNUSABLE_DELIVERY, "profiled but unusable -- gate failed")))
    _rep = _profile_report({"rounds": [], "resolutions": {"m1": _unres}, "spent": 0},
                           dataclasses.replace(_prof_fleet, tenants=(_prof_tenant,)), HW)
    check("the profile report renders an unresolved tenant with reasons per degree",
          _rep["unresolved"] == ["m1"], str(_rep))

# --- the doubted band: degrees the computed envelope excludes and the weights allow (§14.2) ----
#
# 1.2M KV tokens puts this 0.6B tenant's MEMORY floor at tp4 -- tp2 would need more than a card once
# the pool, the activation fallback and the reserve are added. Its WEIGHTS are 1.110 GiB, so tp1 is
# 1.11 GiB of weights against a 44.99 GiB card. tp1 and tp2 are therefore doubted, not eliminated:
# the pool's page size was measured 3.27x too large once and the activation fallback 4.4x, and either
# correction moves the floor. The walk must keep them as candidates while still STARTING where a boot
# is most likely to come up, which is the memory floor.
_band_t = dataclasses.replace(_prof_tenant, max_model_len=600_000)
with tempfile.TemporaryDirectory() as _ptd:
    _pstore7 = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _bandp = agent_profiler.SoloProfiler(
        dataclasses.replace(_prof_fleet, tenants=(_band_t,)), _leased["key"], HW, _pstore7,
        make_agent=_make_agent, store_root=Path(_ptd) / "obs")
    check("the memory floor and the weight floor disagree, which is what makes a band",
          (_bandp.floor_for(_band_t).floor_tp, _bandp.weight_floor_for(_band_t).floor_tp) == (4, 1),
          f"{_bandp.floor_for(_band_t).floor_tp} vs {_bandp.weight_floor_for(_band_t).floor_tp}")
    check("  and the candidate degrees run from the WEIGHT floor, so the band is not eliminated",
          _bandp.degrees_for(_band_t) == (1, 2, 4, 8), str(_bandp.degrees_for(_band_t)))
    check("  while the walk still STARTS at the memory floor, which is a guess and not a refusal",
          _bandp.start_tp_for(_band_t) == 4 and _bandp.ascent_for(_band_t) == (4, 8))
    check("  and the band belongs to the descent: exactly the degrees the ascent will never reach",
          _bandp.band_for(_band_t) == (1, 2), str(_bandp.band_for(_band_t)))
    check("  so `plan` prices the two walks apart -- the ascent it will spend, the band it may",
          (_bandp.plan()[0][2], _bandp.plan()[0][4]) == ((4, 8), (1, 2)),
          str((_bandp.plan()[0][2], _bandp.plan()[0][4])))
    # Rendered, not trusted: `--budget` is chosen off this preview, so a walk that can spend more
    # rounds than the preview names would have the reader budget for the ascent alone. (The report
    # renderer beside it crashed once on a shape the store was perfectly happy with.)
    from colocation.agent.__main__ import _print_profile_plan                 # noqa: E402
    _io = io.StringIO()
    with contextlib.redirect_stdout(_io):
        _worst = _print_profile_plan(_bandp, HW)
    check("  and the cost preview prices the descent too, so --budget is set against the whole walk",
          _worst == 4 and "DOWNWARD into tp1, tp2" in _io.getvalue()
          and "upper bound" in _io.getvalue(), f"worst={_worst}\n{_io.getvalue()}")
    _deployed.clear()
    _out7 = _bandp.run()
    # Up to the first pass, then DOWN through the band one rung at a time. The descent is what makes
    # a minimum out of what would otherwise be an upper bound at the memory floor: tp2 and tp1 are
    # refused by the computed envelope, and the computed envelope's terms have each measured smaller
    # than stated, so only a boot can say which of them is really over the card.
    check("a banded tenant starts at the memory floor and then DESCENDS into the doubted band",
          [(r.tp, r.status, r.descent) for r in _out7["rounds"]]
          == [(4, agent_profiler.DID_MEASURE, False), (2, agent_profiler.DID_MEASURE, True),
              (1, agent_profiler.DID_MEASURE, True)]
          and _out7["spent"] == 3,
          str([(r.tp, r.status, r.descent) for r in _out7["rounds"]]))
    _bres = _out7["resolutions"]["m1"]
    check("  and the answer is tp2 as a MEASURED minimum, not tp4 as an upper bound",
          (_bres.status, _bres.tp, _bres.certain) == (RESOLVED, 2, True)
          and _bres.missing == (), str(_bres.as_dict()))
    check("    with tp1 ruled out by a measured miss rather than by the arithmetic that doubted it",
          _bres.below[0][:2] == (1, MISSED), str([(tp, c) for tp, c, _ in _bres.below]))
    # A descent round is driven with its KV floors relaxed to nothing, because the floors are the only
    # thing that can refuse the degree and refusing is precisely what a doubt may not do. The row has
    # to SAY so: it was measured against a pool the fleet file would not have asked for, and a reader
    # comparing it with an ascent row needs that on the record rather than in a log.
    _drow = _pstore7.read(_band_t.model_id, 2, _band_t.isl_tokens, _band_t.osl_tokens,
                          _band_t.rate_hz, _band_t.serving)
    check("  and every descent row records that its pool was relaxed, and which degree vouched for it",
          (_drow.notes["descent"]["relaxed_floor"], _drow.notes["descent"]["booted_at_tp"],
           _drow.notes["descent"]["stated_floor_tokens"]) == (True, 4, 1_200_000),
          str(_drow.notes["descent"]))
    check("    while an ascent row carries no such note, because nothing was relaxed for it",
          _pstore7.read(_band_t.model_id, 4, _band_t.isl_tokens, _band_t.osl_tokens,
                        _band_t.rate_hz, _band_t.serving).notes.get("descent") is None)
    _rbuf = io.StringIO()
    with contextlib.redirect_stdout(_rbuf):
        _profile_report(_out7, dataclasses.replace(_prof_fleet, tenants=(_band_t,)), HW)
    check("      and the mode's own table marks those rounds and explains the mark, because a "
          "relaxed pool is not the experiment the rows above it are",
          " v2 " in _rbuf.getvalue() and " v1 " in _rbuf.getvalue()
          and "marks a DESCENT round" in _rbuf.getvalue(), _rbuf.getvalue())
    # A band degree whose refusal came FROM the token floor can only be driven on a smaller pool than
    # the floor names -- that is what made it a band degree. The round still met its target and
    # delivered at the offered rate, which is the only evidence about the floor that exists, so the
    # degree is reported as the minimum (keeping the wider one would inflate the baseline, §0). But
    # the file still states the larger floor, so the solver will keep refusing this degree, and the
    # report has to hand that contradiction over rather than resolve it: the demand is the file's.
    check("        and a minimum that rests on a pool the fleet file does not ask for says so, and "
          "names the lever, because changing a stated demand is not the mode's call",
          # 726,885 rather than the 740,285 this check asserted before impl §14.4: the fixture used to
          # state a fitted `activation_gib` of 0.2845 GiB/shard at tp2 and now gets the 1.0 GiB
          # fallback, and the 0.7155 GiB difference comes straight out of the pool at 57,344 B/token.
          # That is the cost of the removal, priced, in the one place the suite can see it. The last
          # two of those tokens went with Fix 2: the fleet file used to round Qwen3-0.6B's weight to
          # 1.110 GiB and the safetensors headers sum to 1.1102294921875, which is 4.3 tokens of pool
          # over the whole model and 2 on each of two shards. A rounded weight is a rounded pool.
          _bres.evidence.pool_below_stated_floor == (726885, 1_200_000)
          and "against the 1,200,000 this fleet file states" in _rbuf.getvalue()
          and "yours to decide" in _rbuf.getvalue(), _rbuf.getvalue())
    check("          and the closing line does not send the reader to `search`, which the solver "
          "would refuse at that degree",
          "will NOT load this fleet as written" in _rbuf.getvalue()
          and "without needing this mode again" not in _rbuf.getvalue(), _rbuf.getvalue())
    check("          while a row measured at its stated floor reports no such conflict",
          _pstore7.read(_band_t.model_id, 4, _band_t.isl_tokens, _band_t.osl_tokens,
                        _band_t.rate_hz, _band_t.serving).pool_below_stated_floor is None)

# A descent round whose engine cannot start is the one boot failure that earns a row. The control is
# what makes it evidence about the DEGREE and not about the run: tp4 came up moments earlier in this
# same run, with the same image, chart, model path and flags. Without that row the degree would read
# "not profiled" forever and the minimum above it would stay an upper bound -- which is the whole
# thing the descent exists to remove.
with tempfile.TemporaryDirectory() as _ptd:
    _pstore9 = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _dfail = agent_profiler.SoloProfiler(
        dataclasses.replace(_prof_fleet, tenants=(_band_t,)), _leased["key"], HW, _pstore9,
        store_root=Path(_ptd) / "obs",
        make_agent=lambda s, st, k: _gate_agent(s, st, k, states={2: NOT_READY_ENGINE}))
    _out11 = _dfail.run()
    check("a descent round whose engine cannot start ENDS the descent, because narrower is bigger",
          [(r.tp, r.status, r.descent) for r in _out11["rounds"]]
          == [(4, agent_profiler.DID_MEASURE, False), (2, agent_profiler.NOT_READY, True)]
          and _out11["spent"] == 2,
          str([(r.tp, r.status, r.descent) for r in _out11["rounds"]]))
    check("  and unlike the same failure on the way UP, it is written down: a wider degree came up "
          "in this run, so the failure is attributable to the degree",
          sorted(p.tp for p in _pstore9.all()) == [2, 4]
          and _pstore9.read(_band_t.model_id, 2, _band_t.isl_tokens, _band_t.osl_tokens,
                            _band_t.rate_hz, _band_t.serving).boot_failed_at == 4)
    _dres = _out11["resolutions"]["m1"]
    check("  so tp4 is reported as MEASURED and not as an upper bound, which is the point of the row",
          (_dres.status, _dres.tp, _dres.certain) == (RESOLVED, 4, True)
          and _dres.missing == (), str(_dres.as_dict()))
    check("    with tp2 ruled out by the boot itself, in words that name the control",
          _dres.below[1][:2] == (2, TOO_NARROW)
          and "at tp4 came up in the same run" in _dres.below[1][2], str(_dres.below[1]))
    check("      and tp1 ruled out by arithmetic off that boot, not by a round nobody needed to "
          "spend: a tp1 shard is larger than the tp2 one the engine already could not start in",
          _dres.below[0][:2] == (1, DOES_NOT_FIT) and "larger still" in _dres.below[0][2],
          str(_dres.below[0]))
    check("        and the witness row is never cited for latency, having none",
          _pstore9.read(_band_t.model_id, 2, _band_t.isl_tokens, _band_t.osl_tokens,
                        _band_t.rate_hz, _band_t.serving).usable is False)

# The control has to be a boot THIS run watched. A pass read out of the store came up under some
# earlier run's chart -- which is exactly what `--reprofile` exists for -- so a boot failure beneath it
# is not attributable to the degree, and writing a permanent "too narrow" row off it would libel a
# degree a fixed chart might serve. The descent still happens; only the row is withheld.
with tempfile.TemporaryDirectory() as _ptd:
    _pstoreD = ProfileStore(Path(_ptd) / "profiles", _pkey)
    # One round, so the store ends up holding tp4 and nothing else: the budget stops the descent.
    agent_profiler.SoloProfiler(
        dataclasses.replace(_prof_fleet, tenants=(_band_t,)), _leased["key"], HW, _pstoreD,
        make_agent=_make_agent, store_root=Path(_ptd) / "obs", budget=1).run()
    _deployed.clear()
    _stale = agent_profiler.SoloProfiler(
        dataclasses.replace(_prof_fleet, tenants=(_band_t,)), _leased["key"], HW, _pstoreD,
        store_root=Path(_ptd) / "obs",
        make_agent=lambda s, st, k: _gate_agent(s, st, k, states={2: NOT_READY_ENGINE}))
    _out14 = _stale.run()
    check("a descent under a pass READ FROM THE STORE still runs, since the band is still unmeasured",
          [(r.tp, r.status, r.descent) for r in _out14["rounds"]]
          == [(4, agent_profiler.ALREADY_PROFILED, False),
              (2, agent_profiler.NOT_READY, True)] and _out14["spent"] == 1,
          str([(r.tp, r.status, r.descent) for r in _out14["rounds"]]))
    check("  but its failed boot gets NO row, because no boot this run vouched for the image and "
          "chart -- a stored row came up under whatever chart was current then",
          sorted(p.tp for p in _pstoreD.all()) == [4])
    check("    so tp4 stays an UPPER BOUND rather than being confirmed by a row nothing controlled",
          (_out14["resolutions"]["m1"].tp, _out14["resolutions"]["m1"].certain) == (4, False)
          and _out14["resolutions"]["m1"].missing == (1, 2),
          str(_out14["resolutions"]["m1"].as_dict()))

# Everything the readiness gate can say other than `engine_failed` is about the RUN, on the way down
# exactly as on the way up: nothing was deployed into the envelope, so nothing tested the degree. It
# stops, and it writes NO row -- a permanent "too narrow" here would libel a degree a fixed chart
# might serve, and `--reprofile` could never retry it.
for _gate in (NOT_READY_VERIFY, NOT_READY_TIMEOUT, NOT_READY_SCHEDULE):
    with tempfile.TemporaryDirectory() as _ptd:
        _pstoreA = ProfileStore(Path(_ptd) / "profiles", _pkey)
        _dstop = agent_profiler.SoloProfiler(
            dataclasses.replace(_prof_fleet, tenants=(_band_t,)), _leased["key"], HW, _pstoreA,
            store_root=Path(_ptd) / "obs",
            make_agent=lambda s, st, k, _g=_gate: _gate_agent(s, st, k, states={2: _g}))
        _outd = _dstop.run()
        check(f"a descent round that did not come up as {_gate!r} stops and writes nothing: the "
              "degree was never tested, so it may not be marked too narrow either",
              [(r.tp, r.status, r.gate) for r in _outd["rounds"]]
              == [(4, agent_profiler.DID_MEASURE, ""), (2, agent_profiler.NOT_READY, _gate)]
              and sorted(p.tp for p in _pstoreA.all()) == [4]
              and _outd["resolutions"]["m1"].missing == (1, 2),
              str([(r.tp, r.status, r.gate) for r in _outd["rounds"]])
              + str(_outd["resolutions"]["m1"].as_dict()["below"]))

# The budget stops the descent the same way it stops the ascent, and the answer then says what it is:
# a degree that met the target with a narrower one left unmeasured is an UPPER BOUND on the minimum.
with tempfile.TemporaryDirectory() as _ptd:
    _pstoreB = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _bud = agent_profiler.SoloProfiler(
        dataclasses.replace(_prof_fleet, tenants=(_band_t,)), _leased["key"], HW, _pstoreB,
        make_agent=_make_agent, store_root=Path(_ptd) / "obs", budget=2)
    _out12 = _bud.run()
    check("a budget that runs out mid-descent leaves the band unmeasured rather than unreported",
          [(r.tp, r.status, r.descent) for r in _out12["rounds"]]
          == [(4, agent_profiler.DID_MEASURE, False), (2, agent_profiler.DID_MEASURE, True)]
          and _out12["spent"] == 2,
          str([(r.tp, r.status, r.descent) for r in _out12["rounds"]]))
    check("  and the minimum it reports is tp2, still an UPPER BOUND, because tp1 is unmeasured",
          (_out12["resolutions"]["m1"].tp, _out12["resolutions"]["m1"].certain) == (2, False)
          and _out12["resolutions"]["m1"].missing == (1,),
          str(_out12["resolutions"]["m1"].as_dict()))

# Rows already in the store settle a band degree without a round, which is what makes the descent
# affordable to repeat: the second `profile` of the same fleet spends nothing and reaches the same
# answer. A row that RULES A DEGREE OUT counts here as much as one that passes -- on the way up an
# unusable row stops the walk, and on the way down a too-narrow one ends it, having done its job.
with tempfile.TemporaryDirectory() as _ptd:
    _pstoreC = ProfileStore(Path(_ptd) / "profiles", _pkey)
    _again = agent_profiler.SoloProfiler(
        dataclasses.replace(_prof_fleet, tenants=(_band_t,)), _leased["key"], HW, _pstoreC,
        make_agent=_make_agent, store_root=Path(_ptd) / "obs")
    _again.run()
    _deployed.clear()
    _out13 = _again.run()
    check("re-running a banded tenant spends nothing: the ascent and the descent both read the store",
          [(r.tp, r.status) for r in _out13["rounds"]]
          == [(4, agent_profiler.ALREADY_PROFILED), (2, agent_profiler.ALREADY_PROFILED),
              (1, agent_profiler.ALREADY_PROFILED)]
          and _out13["spent"] == 0 and _deployed == [],
          str([(r.tp, r.status) for r in _out13["rounds"]]))
    check("  and reaches the same measured minimum it reached by spending GPU time the first time",
          (_out13["resolutions"]["m1"].tp, _out13["resolutions"]["m1"].certain) == (2, True))

# --- inspect: the views that make a record legible ---------------------------
# Not assertions about numbers -- assertions that the qualifiers a number is meaningless
# without are actually shown. Run against the two real fixtures.
_lines = "\n".join(record_lines(run5))
check("the record view names the verdict, the window and the sampling together",
      all(w in _lines for w in ("verdict", "window", "sub-samples/sample")))
check("  and separates the GPU node under test from the harness Job's node",
      "gpu node" in _lines and "driver node" in _lines)
check("  and shows a co-residency degree per tenant, from the recorded UUIDs",
      "degree" in _lines)

# A record whose latency was suppressed must not print a zero anywhere.
supp = json.loads((FIXTURES / "run5-sidecar.json").read_text())
for t in supp["per_tenant"].values():
    t.pop("itl_mean_ms", None)
    t["latency_suppressed"] = "delivery gate failed"
supp["verdict"] = VERDICT_DELIVERY
_slines = "\n".join(record_lines(Observation.parse(supp)))
check("a suppressed latency prints as suppressed, never as a number",
      "suppressed" in _slines and "0.000ms" not in _slines)

# `model_lines` is gone with the model it printed. There is no table of ITL cells to render as
# "measured here, not measured there", because nothing holds one: the records are the whole of what
# a reader or a reasoner gets, and `record_lines` above is how one is read.
check("nothing in inspect renders a predictor, because there is none to render",
      not hasattr(sys.modules["colocation.inspect"], "model_lines"))

# --- calibrate: the noise floor, from repeats that already exist --------------
# Inputs derived from the real run 6 record by copying it and moving one tenant's ITL: the
# estimator is arithmetic over passes, and what has to be tested is which passes it counts.
# Nothing is added to the fixture's provenance, and that is the change: the bucket used to be
# `provenance.predicted_regime`, so these inputs had to be stamped with a forecast before they could
# be calibrated at all. The bucket is now read off the record's own GPU UUIDs.
_base = json.loads((FIXTURES / "run6-no-sidecar.json").read_text())
BUCKET6 = "coresident=0"        # run 6 was disjoint: six shards, six GPUs, nothing shared


def pass_of(run_id: str, shifts: dict, **over) -> Observation:
    d = json.loads(json.dumps(_base))
    d["run_id"] = run_id
    d.update(over)
    for name, delta in shifts.items():
        d["per_tenant"][name]["itl_mean_ms"] += delta
    return Observation.parse(d, f"synthetic:{run_id}")


check("a store with nothing repeated is not calibrated, rather than calibrated to zero",
      between_run_halfwidth([pass_of("solo", {})]) is None)

_p1, _p2 = pass_of("p1", {}), pass_of("p2", {"m1": 0.400, "r1": 0.100})
_cal = between_run_halfwidth([_p1, _p2])
check("two passes of one placement give half the full range of the widest tenant",
      abs(_cal.between_run_itl_halfwidth_ms[BUCKET6] - 0.200) < 1e-6,
      str(_cal.between_run_itl_halfwidth_ms))
check("  and the widest pair is named, so the floor is attributable to a tenant and a run",
      "m1" in _cal.notes["widest_pair_per_bucket"][BUCKET6])
check("  and the pass count travels with it, since a range from two draws is biased low",
      _cal.repeats == 2 and "lower bound" in _cal.notes["biased_low"],
      _cal.notes["biased_low"])
check("  and solo saturation is left empty rather than defaulted, since none was measured",
      _cal.between_run_itl_halfwidth_ms and not _cal.solo_saturation_rate_hz)

# A different placement, measured once, has no spread -- and must not become one.
_od = json.loads(json.dumps(_base))
_od["run_id"] = "elsewhere"
_od["per_tenant"]["m1"]["itl_mean_ms"] += 40.0
_od["configuration"]["tenants"][0]["gpu_uuids"] = ["GPU-somewhere-else"]
_other = Observation.parse(_od, "synthetic:elsewhere")
check("a placement measured once cannot widen the floor, however far it sits from the others",
      abs(between_run_halfwidth([_p1, _p2, _other])
          .between_run_itl_halfwidth_ms[BUCKET6] - 0.200) < 1e-6,
      "otherwise a placement effect is reported as noise and the search stops immediately")

# Runs 5 and 6 predate the agent, so they carry no agent provenance at all. Under the old bucketing
# they could not be calibrated from -- the only repeats this store will ever hold of the original
# sweep would have been unusable for the sidecar they are the evidence for.
_bare = json.loads(json.dumps(_base))
_bare["provenance"] = {}
_bare["run_id"] = "p3"
_bare["per_tenant"]["m1"]["itl_mean_ms"] += 0.400
_bcal = between_run_halfwidth([_p1, Observation.parse(_bare, "synthetic:p3")])
check("a record with no provenance whatsoever still buckets itself, from the UUIDs it recorded",
      _bcal is not None and abs(_bcal.between_run_itl_halfwidth_ms[BUCKET6] - 0.200) < 1e-6,
      str(None if _bcal is None else _bcal.between_run_itl_halfwidth_ms))
# The other side: a record that cannot say which GPUs it used cannot be bucketed, and is skipped
# rather than filed under a guess -- pairing it with a placement it may not have shared would put a
# placement effect into the noise floor.
_nogpu = json.loads(json.dumps(_base))
_nogpu["run_id"] = "p3b"
for _t in _nogpu["configuration"]["tenants"]:
    _t.pop("gpu_uuids")
check("  while a record that does not say which GPUs it used is skipped, not guessed at",
      bucket(Observation.parse(_nogpu, "synthetic:p3b")) is None
      and between_run_halfwidth([_p1, Observation.parse(_nogpu, "synthetic:p3b")]) is None)

_supp = json.loads(json.dumps(_base))
_supp["run_id"] = "p4"
_supp["per_tenant"]["m1"].pop("itl_mean_ms")
_supp["per_tenant"]["m1"]["latency_suppressed"] = "delivery gate failed"
_supp["verdict"] = VERDICT_DELIVERY
_scal = between_run_halfwidth([_p2, Observation.parse(_supp, "synthetic:p4")])
check("a pass whose latency was suppressed contributes nothing, rather than contributing a zero",
      _scal is None, "the whole record is unusable for fitting, so there is no second pass")

_ed = json.loads(json.dumps(_base))
for _t in _ed["configuration"]["tenants"]:
    _t["envelope_gib"] = _t["envelope_gib"] + 3
check("placement identity is GPU UUIDs and ignores the envelope, like the policy's canonical",
      placement_key(_p1) == placement_key(Observation.parse(_ed, "synthetic:env")) is not None)
_ud = json.loads(json.dumps(_base))
for _t in _ud["configuration"]["tenants"]:
    _t.pop("gpu_uuids")
check("  and a record with no UUIDs has no identity, so it is not paired with anything",
      placement_key(Observation.parse(_ud, "synthetic:nouuid")) is None)

# --- agent: the reasoner drives the loop, with six tools (impl §11) ----------
# impl §11 makes the reasoner the loop, and the solver, the store and the cluster its tools.
# `ReplayReasoner` answers from a list, so the whole contract -- what it is told, what it may say
# back, what each tool refuses, and what happens when it says something unusable -- is tested
# offline, deterministically, and for nothing. No case here calls Bedrock and none touches a GPU.
#
# The two placements below are the real finding: both are co-residency degree 1 for r2 and t2, and
# they measured 5.76 ms and 10.88 ms. They stay as the fixture for a reason that outlived the model
# that could not tell them apart -- *nothing* can tell them apart from a summary of their
# structure, which is why the reasoner is shown the assignment itself and `recall` returns runs.
SYM = {"m1": [0], "r1": [1], "r2": [2, 3], "t2": [2, 3]}
ASYM = {"m1": [0], "r1": [1], "r2": [2, 3], "t2": [3, 4]}


def _obs_on(run_id: str, placement: dict, itl_ms: dict) -> Observation:
    """A record of `placement` (tenant -> GPU ordinals) on INV's node, with those latencies."""
    d = json.loads(json.dumps(_base))
    d["run_id"] = run_id
    d["node"] = INV.node_name
    for t in d["configuration"]["tenants"]:
        t["gpu_uuids"] = [INV.uuids[g] for g in placement[t["name"]]]
    for name, ms in itl_ms.items():
        d["per_tenant"][name]["itl_mean_ms"] = ms
    return Observation.parse(d, f"synthetic:{run_id}")


LLM_OBS = [_obs_on("search-001-sym", SYM, {"r2": 5.760, "t2": 5.763}),
           _obs_on("search-002-asym", ASYM, {"r2": 10.194, "t2": 10.877})]
LLM_CANDS = offered([_Cand(SYM, margin=15.83), _Cand(ASYM, margin=2.62)])

# A fleet with a target on three tenants and none on the fourth. FLEET states none at all, and a
# fleet where nobody stated a target cannot exercise the two invariants that matter most here: that
# a tenant with no target may only be `unknown`, and that `met` is derived from a record.
SFLEET = dataclasses.replace(FLEET, tenants=tuple(
    dataclasses.replace(t, slo=None if t.name == "t2" else TenantSlo(itl_mean_ms=8.0))
    for t in FLEET.tenants))

# --- what the reasoner is told -----------------------------------------------
_prob_kw = dict(fleet=SFLEET, budget=6, store_runs=len(LLM_OBS), key_hash="abc123",
                calibration=None)
_prob = prompt.problem(**_prob_kw)
check("the problem states each tenant's own rate, shape and target, since not one of them is "
      "fleet-wide",
      "16.0Hz" in _prob and "itl_mean_ms=8" in _prob
      and "can be measured but not certified" in _prob)
check("  and the KV floor each target implies, with the direction that surprises people said "
      "outright",
      "tokens" in _prob and "implies a *smaller* floor" in _prob)
check("  and renders none of the store, because which runs bear on a placement is the reasoner's "
      "judgement and `recall` is how it asks",
      "search-001-sym" not in _prob and "2 record(s)" in _prob
      and "`recall` is how you see them" in _prob)
check("  and the budget in measurements, which is the only thing that costs GPU time",
      "6 measurement(s)" in _prob and "Nothing else you can call costs GPU time" in _prob)
check("  and says UNCALIBRATED rather than quoting a resolution nobody measured",
      "UNCALIBRATED" in _prob and "do not stop because they look small" in _prob)
_calprob = prompt.problem(**dict(_prob_kw, calibration=CAL))
check("  while with calibration it quotes the measured half-width per co-residency bucket",
      "0.2000 ms" in _calprob and "coresident=0" in _calprob and "UNCALIBRATED" not in _calprob)
check("the problem statement is byte-stable and free of trailing whitespace, so a committed one "
      "is a fixture",
      _prob == prompt.problem(**_prob_kw)
      and all(ln == ln.rstrip() for ln in _prob.splitlines()))
check("the system prompt names all six tools and there is no seventh",
      all(t in prompt.SYSTEM for t in prompt.TOOLS)
      and prompt.TOOLS == ("solve", "recall", "read_run", "measure", "cluster_status", "submit"))
check("  and says the solver's order is a packing order rather than a forecast",
      "packing order, not a forecast" in prompt.SYSTEM
      and "Nothing in this system predicts a latency" in prompt.SYSTEM)
check("  and requires an expectation on every measurement, saying what it is for",
      "required for every tenant" in prompt.SYSTEM
      and "separates a wrong belief from a wrong choice" in prompt.SYSTEM)
check("  and names neither the answer nor today's hypothesis, since a prompt that does is a test "
      "of reading rather than of reasoning",
      not any(w in prompt.SYSTEM.lower() for w in ("symmetric", "neighbour", "per-shard")))
check("  and says that stopping and \"no evidence either way\" are real answers",
      "Stopping is a real answer" in prompt.SYSTEM and "no evidence either way" in prompt.SYSTEM)

# --- how a feasible set is rendered ------------------------------------------
_cl = "\n".join(prompt.candidate_lines(LLM_CANDS, measured={prompt.canonical(SYM)}))
check("a candidate prints the GPU set each tenant holds, not a co-residency degree -- a summary "
      "of structure can only reproduce the summary's own blind spot",
      "gpu2,gpu3" in _cl and "gpu3,gpu4" in _cl)
check("  and its KV against each floor, with no latency attached to any of it",
      "tightest margin 15.83x" in _cl and "floor   10240" in _cl
      and "says nothing about which will measure fastest" in _cl)
check("  and flags one the store has already measured, with what choosing it buys",
      _cl.count("ALREADY MEASURED") == 1)
check("  and says whether the list is all of them, which decides what may be concluded from it",
      "TRUNCATED" in _cl and "Do not conclude the fleet is" in _cl)
check("  while an exhausted set says the opposite outright",
      "EVERY feasible placement" in "\n".join(prompt.candidate_lines(
          offered(list(LLM_CANDS), exhausted=True))))
check("  and a bare list claims neither, because a list carries no such fact",
      "not recorded for this list" in "\n".join(prompt.candidate_lines([LLM_CANDS[0]])))

# --- a turn: one JSON tool call, or a refusal written to be read by the model -


def _bad_call(name: str, text: str, needle: str) -> None:
    try:
        prompt.parse_call(text)
        check(name, False, "the reply was accepted")
    except prompt.InvalidReply as exc:
        check(name, needle in str(exc), str(exc)[:120])


def _bad_args(name: str, fn, args, needle: str) -> None:
    try:
        fn(args)
        check(name, False, "the call was accepted")
    except prompt.ToolError as exc:
        check(name, needle in str(exc), str(exc)[:160])


_bad_call("prose instead of a tool call is rejected, not scavenged for an intention",
          "I would try candidate 1 -- it looks least contended.", "no JSON object")
_bad_call("  and a malformed object is rejected as invalid JSON, with what to avoid",
          '{"tool": "solve", "args": {},}', "not valid JSON")
_bad_call("  and a tool that does not exist is rejected by naming the ones that do",
          '{"tool": "predict_itl", "args": {}}', "solve, recall, read_run")
_bad_call("  and args that are not an object, since {} is how a tool with none is called",
          '{"tool": "solve", "args": [4]}', "must be a JSON object")
check("a fenced call is accepted, since a fence is a formatting habit and not a refusal",
      prompt.parse_call("```json\n{\"tool\": \"solve\", \"args\": {\"cap\": 3}}\n```")
      == {"tool": "solve", "args": {"cap": 3}, "note": ""})
check("  and `args` may be omitted entirely for a tool that takes none",
      prompt.parse_call('{"tool": "cluster_status"}')["args"] == {})
# Seen live (2026-10-09): `args` closed, the call not. Closing what was opened, at the end, has
# one reading, and a brace inside a string is not a bracket.
check("a call that leaves off only its closing braces is closed, and nothing else is changed",
      prompt.parse_call('{"tool": "recall", "args": {"run_ids": ["a}"], "x": "y{"}')
      == {"tool": "recall", "args": {"run_ids": ["a}"], "x": "y{"}, "note": ""})
_bad_call("  and one that also has another defect is refused, saying what to close",
          '{"tool": "recall", "args": {"a": "b" "c"}', "needs '}' at the end")
_bad_call("  and one that closes more than it opens is refused as invalid JSON",
          '{"tool": "recall", "args": {}}}', "not valid JSON")
check("  and both JSON loops are told where each closing brace goes, by the same words",
      prompt.JSON_RULES in prompt._JSON_ENVELOPE and prompt.JSON_RULES in splitpacker.SYSTEM
      and "{rules}" not in prompt._JSON_ENVELOPE)

# Argument checking lives beside the tool rather than in the prompt, so it holds whichever runtime
# drives the loop. Every message is addressed to the model: what was wrong *and* what would be
# right, since a refusal it cannot act on is a refusal it will repeat.
_ca = (lambda a: prompt.check_assignment(a, SFLEET))
_bad_args("an assignment missing a tenant is refused, naming which one", _ca,
          {"m1": [0], "r1": [1], "r2": [2, 3]}, "missing ['t2']")
_bad_args("  and one naming a tenant the fleet does not have", _ca,
          dict(SYM, t3=[5]), "unknown ['t3']")
_bad_args("  and one giving a tenant the wrong number of GPUs for its TP degree", _ca,
          dict(SYM, t2=[2]), "needs exactly 2")
_bad_args("  and one putting two shards of a TP group on the same device", _ca,
          dict(SYM, t2=[2, 2]), "same GPU appears twice")
_bad_args("  and one naming a GPU this node does not have", _ca,
          dict(SYM, t2=[3, 9]), "no GPU [9]")
_bad_args("  and one whose GPU list is not a list of integers", _ca,
          dict(SYM, m1="gpu0"), "list of integer ordinals")
check("a valid assignment comes back as sorted tuples, ready to pin",
      _ca(ASYM) == {n: tuple(sorted(g)) for n, g in ASYM.items()})

_ce = (lambda e: prompt.check_expectation(e, SFLEET))
_bad_args("an expectation is required for every tenant, since a measurement is of the whole "
          "placement and a partial expectation cannot be compared with it", _ce,
          {"m1": "hold", "r1": "hold", "r2": "miss"}, "missing ['t2']")
_bad_args("  and one outside the contract is refused rather than guessed at", _ce,
          {"m1": "probably", "r1": "hold", "r2": "hold", "t2": "unknown"}, "probably")
_bad_args("  and a tenant that stated no target cannot be expected to hold one, which is the "
          "invariant impl §11.2 states outright", _ce,
          {"m1": "hold", "r1": "hold", "r2": "hold", "t2": "hold"}, "states no latency target")
check("  while `unknown` for that tenant is accepted, because ignorance about it is the truth",
      _ce({"m1": "hold", "r1": "hold", "r2": "miss", "t2": "unknown"})
      == {"m1": "hold", "r1": "hold", "r2": "miss", "t2": "unknown"})
_bad_args("a label that is not a DNS-1123 label is refused, since it becomes a Job name",
          prompt.check_label, "Probe_01", "'label' must be")
_bad_args("  and one over 20 characters is too, before it truncates into a collision",
          prompt.check_label, "a-very-long-label-that-keeps-going", "'label' must be")
_bad_args("a cut naming one tenant twice forbids nothing, so it is refused",
          (lambda a: prompt.check_avoid(a, SFLEET)), [["r2", "r2"]], "names one tenant twice")
_bad_args("  and one naming a tenant the fleet does not have",
          (lambda a: prompt.check_avoid(a, SFLEET)), [["r2", "ghost"]], "not in this fleet")
_bad_args("an empty rationale is refused, since it is what the record will carry",
          prompt.check_rationale, "   ", "must be a non-empty string")
check("a rationale citing no run id is accepted, because \"there is no evidence and I am "
      "exploring\" has to be sayable",
      prompt.check_rationale("nothing in the store bears on this; exploring")
      == "nothing in the store bears on this; exploring")

# --- the tools, against the real solver and a fake cluster -------------------


def _tools(store, *, budget=2, fleet=SFLEET, deployer=None, harness=None) -> Tools:
    agent = SearchAgent(fleet, store, deployer or FakeDeployer(),
                        harness or FakeHarness(fake_record), run_prefix="pack",
                        ready_timeout_s=1.0, measure_timeout_s=1.0)
    return Tools(fleet, store, agent, budget=budget, top_k=2, reasoner_name="replay")


with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    T = _tools(store)
    _sol = T.tool_solve({"cap": 2})
    check("`solve` answers with the feasible set the real solver found, GPU sets and all",
          "candidate 1" in _sol and "gpus=gpu" in _sol and "GPU(s) used" in _sol,
          _sol.splitlines()[0])
    check("  and it is a scratchpad: asking twice gives the same answer and costs nothing",
          T.tool_solve({"cap": 2}) == _sol and T.spent == 0)
    _cut = T.tool_solve({"cap": 2, "avoid": [["r2", "t2"]]})
    check("  and a cut the reasoner declares narrows what it is offered, and is quoted back as "
          "JSON it could send again",
          "your cuts [['r2', 't2']]" in _cut, _cut.splitlines()[0])
    check("  while a cut passed to `solve` does not become a standing one, since a belief held "
          "for one question is not a belief held for the run",
          T.avoid == ())
    _first = {n: list(g) for n, g in T.last_feasible[0].placement.assignment.items()}
    _pinned = T.tool_solve({"assignment": _first})
    check("a placement named outright is pinned, solved for envelopes and checked by "
          "`Placement.verify` in arithmetic the solver does not share",
          "FEASIBLE" in _pinned)
    check("  and a pinned solve leaves the numbered list alone, so \"candidate 2\" keeps meaning "
          "what the reasoner was last shown",
          len(T.last_feasible) == 2)

    _rec = T.tool_recall({"candidate": 1})
    check("`recall` on an empty store says NO EVIDENCE in those words, which is the answer a "
          "predictor cannot give",
          "NO EVIDENCE" in _rec)
    _bad_args("`read_run` on a run that does not exist says what the store does hold",
              T.tool_read_run, {"run_id": "search-999"}, "no run called 'search-999'")
    _bad_args("submitting \"met\" is refused when no record supports it: the reasoner cannot "
              "certify, and the loop derives certification from a measurement",
              T.tool_submit, {"verdict": "met", "rationale": "it looks good to me"},
              "not yours to declare")

    _bad_args("`measure` refuses a call with no expectation, before anything is deployed",
              T.tool_measure,
              {"choice": {"candidate": 1}, "rationale": "r", "label": "x"},
              "'expectation' is required")
    check("  and the refusal deployed nothing, which is what makes it a gate",
          T.spent == 0 and not T.attempts and not T.agent.deployer.applied)

    # The measurement itself. `FakeHarness` writes a record whose ITL depends only on co-residency
    # degree, so what is checked here is the path -- deploy, gate, record, store, and what the
    # reasoner is handed back -- and never a latency number.
    _m1 = T.tool_measure({"choice": {"candidate": 1}, "label": "first",
                          "expectation": {"m1": "hold", "r1": "hold", "r2": "hold",
                                          "t2": "unknown"},
                          "rationale": "no evidence yet; the densest placement is the cheapest "
                                       "thing to learn from"})
    check("a measurement spends one of the budget and returns the record in full, not a verdict",
          T.spent == 1 and len(T.attempts) == 1 and "ITL mean" in _m1
          and "targets in force now" in _m1)
    check("  and says what is left, since the reasoner is the thing deciding how to spend it",
          "1 of 2 measurement(s) left" in _m1)
    check("  and puts what was expected beside what happened, which is what makes an expectation "
          "falsifiable",
          "as you expected (hold)" in _m1 and "you said you did not know" in _m1)
    _stored = store.load().observations[0]
    check("  and the record carries the expectation, recorded before the measurement",
          _stored.provenance["reasoner_expectation"]
          == {"m1": "hold", "r1": "hold", "r2": "hold", "t2": "unknown"})
    check("  and who chose it and why, which is the whole of what makes it auditable",
          _stored.provenance["policy_kind"] == "model-packer-agent"
          and _stored.provenance["chosen_by_reasoner"] == "replay"
          and "densest placement" in _stored.provenance["reasoner_rationale"])
    check("  while what the agent required of the placement cannot be shadowed by any of it",
          _stored.provenance["chosen_by"] == "agent"
          and _stored.provenance["kv_floor_tokens"] and _stored.provenance["kv_margin"])
    # A UUID names one device on one machine, so everything ordinal-shaped about a placement is
    # lost when the node goes unless the run writes it down. The g6e.48xlarge interconnect gap
    # (13.957 vs 21.138 ms for the same tenant on different quads) is invisible in a record that
    # carries only UUIDs, and there is no way to recover it afterwards.
    check("  and the shard ordinals, because a UUID does not survive the node and a placement is "
          "a set of positions",
          _stored.provenance["gpu_indices"]
          and set(_stored.provenance["gpu_indices"]) == set(_stored.per_tenant)
          and all(all(isinstance(g, int) for g in v)
                  for v in _stored.provenance["gpu_indices"].values()))
    check("  with the map that joins them to the UUIDs the harness recorded, so the per-GPU "
          "gauges stay readable too",
          all(u in _stored.provenance["gpu_ordinal_by_uuid"]
              for n in _stored.per_tenant
              for u in _stored.tenant_config(n).get("gpu_uuids") or []))
    _off = _ordinal_names(_stored, SimpleNamespace(namespace="nonexistent-namespace"))
    check("  and that is the whole point: --ordinals resolves off the record alone, with no "
          "cluster and no surviving node",
          _off and set(_off.values()) <= {f"gpu{g}" for g in range(8)}
          and all(v.startswith("gpu") for v in _off.values()))

    _cs = T.tool_cluster_status({})
    check("`cluster_status` reports what is installed right now, which a transcript cannot after "
          "an interruption",
          "tenant releases installed right now" in _cs
          and "measurements this run: 1 of 2" in _cs)

    _m2 = T.tool_measure({"choice": {"candidate": 2}, "label": "second",
                          "expectation": {n: "unknown" for n in ("m1", "r1", "r2", "t2")},
                          "rationale": "the first run settles nothing about the other structure",
                          "avoid": [["r2", "t2"]]})
    check("a cut declared on a measurement stands for the rest of the run",
          T.avoid == (("r2", "t2"),)
          and "cuts in force: [['r2', 't2']]" in T.tool_cluster_status({}))
    check("  and reaches no record, because a belief must not be inherited by a later run "
          "wearing the authority of a measurement",
          all("cuts_in_force" not in json.dumps(o.provenance)
              and "avoid" not in json.dumps(o.provenance)
              for o in store.load().observations))
    _bad_args("the budget is a gate at the tool boundary, not a line in the prompt",
              T.tool_measure,
              {"choice": {"candidate": 1}, "label": "third",
               "expectation": {n: "unknown" for n in ("m1", "r1", "r2", "t2")},
               "rationale": "one more"},
              "the budget is spent")
    check("  and it refuses before validating the rest, so a malformed call is not what discovers "
          "the budget is gone",
          T.spent == 2 and len(T.attempts) == 2)

    # Now the derivation the reasoner is not allowed to make. `fake_record` measures 2.5-5.5 ms
    # against an 8 ms ceiling, so these runs do hold every stated target.
    check("`met` is derived from a record: every tenant that stated a target held it",
          T.met_run() is not None and T.met_run().run_id == T.attempts[0].run_id,
          T._met_summary()[:120])
    _sub = T.tool_submit({"verdict": "met", "rationale": "run 1 holds every stated target",
                          "choice": {"candidate": 1}})
    check("  so submitting it is accepted once a measurement supports it, and names the run it "
          "was derived from",
          _sub == SUBMITTED and T.submitted["derived_met_run"] == T.attempts[0].run_id)
    check("  and t2, which stated no target, is certified by none of it",
          targets_met(Observation.parse(T.attempts[0].record), T.targets)["t2"] is None)

# A placement this node cannot host is refused with the memory constraint that refused it, rather
# than with a ranking. The floors here are impossible on purpose: the fleet's models are small
# enough that the densest real placement fits, which is why this needs its own fleet.
with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    _huge = dataclasses.replace(SFLEET, tenants=tuple(
        dataclasses.replace(t, max_model_len=100_000_000) for t in SFLEET.tenants))
    _bad_args("a pinned placement the node cannot host is refused, in terms of the memory it "
              "would need rather than as a bad choice",
              _tools(store, fleet=_huge).tool_solve, {"assignment": SYM},
              "cannot be hosted on this node")

# A target tightened below what the hardware did un-certifies the same records, with nothing
# rewritten. This is why the verdict is derived at read time rather than stamped into a run.
with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    T = _tools(store, budget=1, fleet=dataclasses.replace(SFLEET, tenants=tuple(
        dataclasses.replace(t, slo=None if t.name == "t2" else TenantSlo(itl_mean_ms=1.0))
        for t in SFLEET.tenants)))
    T.tool_solve({"cap": 1})
    T.tool_measure({"choice": {"candidate": 1}, "label": "tight",
                    "expectation": {"m1": "miss", "r1": "miss", "r2": "miss", "t2": "unknown"},
                    "rationale": "1 ms is below anything this node has measured"})
    check("a run that misses cannot be submitted as met, however it was expected to go",
          T.met_run() is None)
    check("  and the miss having been expected is reported as a different finding from a surprise",
          "as you expected (miss)" in "\n".join(expectation_lines(
              T.expectations[T.attempts[0].run_id],
              targets_met(Observation.parse(T.attempts[0].record), T.targets))))
    check("  while the record itself is untouched: the verdict is judged now, not stored",
          "MISSED" in "\n".join(read_run(Observation.parse(T.attempts[0].record),
                                         T.targets, T.gpu_of))
          and "verdict" not in T.attempts[0].record["per_tenant"]["m1"])

# --- SM caps as the reasoner's option (impl §16) ------------------------------
_SYM_T = {n: tuple(g) for n, g in SYM.items()}
_bad_args("an SM cap of 100 is refused, since it is no cap and would give one placement two "
          "identities", lambda c: prompt.check_sm_caps(c, SFLEET, _SYM_T), {"t2": 100}, "1 to 99")
_bad_args("  and so is a bool, which JSON would otherwise pass as 1", lambda c: prompt.check_sm_caps(c, SFLEET, _SYM_T),
          {"t2": True}, "1 to 99")
_bad_args("  and a tenant the fleet does not have",
          lambda c: prompt.check_sm_caps(c, SFLEET, _SYM_T), {"zz": 25}, "not in this fleet")
_bad_args("  and a cap on a tenant alone on its card, which leaves its SMs to nobody",
          lambda c: prompt.check_sm_caps(c, SFLEET, _SYM_T), {"m1": 25}, "shares no GPU")
check("a cap on a tenant sharing a card is accepted, and none is the empty map",
      prompt.check_sm_caps({"t2": 25}, SFLEET, _SYM_T) == {"t2": 25}
      and prompt.check_sm_caps(None, SFLEET, _SYM_T) == {})
from colocation.agent.policy import canonical as _pcanon  # noqa: E402
check("uncapped, a placement's identity is the one it had before caps existed; capped, it is "
      "another placement", _pcanon(SYM) == prompt.canonical(SYM) == _pcanon(SYM, {})
      and _pcanon(SYM, {"t2": 25}) != _pcanon(SYM) and _pcanon(SYM, {"t2": 25}) != _pcanon(SYM, {"t2": 50}))

with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    T = _tools(store)
    _ex = {n: "unknown" for n in ("m1", "r1", "r2", "t2")}
    T.tool_measure({"choice": {"assignment": SYM}, "label": "open", "expectation": _ex,
                    "rationale": "exploring"})
    T.tool_measure({"choice": {"assignment": SYM}, "label": "capped", "expectation": _ex,
                    "rationale": "t2 had slack in the open run", "sm_caps": {"t2": 25}})
    _o_open, _o_cap = (Observation.parse(a.record) for a in T.attempts)
    check("a capped measurement of a measured placement is not declared a repeat of it",
          _o_cap.provenance["reasoner_declared_repeat"] is False
          and T.attempts[0].key != T.attempts[1].key)
    _cap_vals = dict(T.agent.deployer.applied[-1])
    check("  and the chart is handed the cap for that tenant and no other",
          [v["mps"]["active_thread_percentage"] for _, v in sorted(_cap_vals.items())].count(25) == 1
          and all(v["mps"]["active_thread_percentage"] in (None, 25) for v in _cap_vals.values()))
    check("  and the record carries it as part of the placement, uncapped tenants without the key",
          retrieval.sm_caps_of(_o_cap) == {"t2": 25} and retrieval.sm_caps_of(_o_open) == {}
          and "active_thread_percentage" not in _o_cap.tenant_config("r2"))
    _rc = T.tool_recall({"assignment": SYM, "sm_caps": {"t2": 25}})
    _cap_line = next(ln for ln in _rc.splitlines() if _o_cap.run_id in ln)
    _open_line = next(ln for ln in _rc.splitlines() if _o_open.run_id in ln)
    check("`recall` with the caps calls only the capped run THE SAME PLACEMENT",
          "THE SAME PLACEMENT" in _cap_line and "THE SAME PLACEMENT" not in _open_line
          and "t2 SM cap none here vs 25% in the candidate" in _open_line, _open_line)
    check("  and prints each run's cap beside its tenant", "sm-cap 25%" in _rc)
    check("  and does not group the two as repeats of one placement",
          "measured 2 times" not in _rc)
    _bad_args("  and a cap with no placement beside it is refused",
              T.tool_recall, {"tenants": ["t2"], "sm_caps": {"t2": 25}}, "needs a placement")
    check("calibration's placement key separates the capped run from the open one",
          placement_key(_o_cap) != placement_key(_o_open) is not None)

_capped_rel = render_releases(pinned["stacked"].placement, FLEET,
                              sm_caps={FLEET.tenants[0].name: 30})
_capped_cfg = render_run_config(pinned["stacked"].placement, FLEET, "run-x", "hash-x",
                                sm_caps={FLEET.tenants[0].name: 30})
check("chart values and run config that agree on a cap pass the agreement check",
      agree(_capped_rel, _capped_cfg) is None)
try:
    agree(rel, _capped_cfg)
    check("  and a cap in the run config that the chart does not carry is refused", False)
except RenderError as _exc:
    check("  and a cap in the run config that the chart does not carry is refused",
          "SM cap" in str(_exc), str(_exc))
check("an uncapped run config carries no cap key, so it is the one written before caps existed",
      all("active_thread_percentage" not in t for t in cfg["tenants"]))

# --- the loop: one conversation, six tools, a budget -------------------------


def _call(tool: str, **args) -> str:
    return json.dumps({"tool": tool, "args": args, "note": "because"})


_EXPECT_ALL = {"m1": "hold", "r1": "hold", "r2": "hold", "t2": "unknown"}


def _packer(store, replies, **kw) -> tuple:
    agent = SearchAgent(SFLEET, store, FakeDeployer(), FakeHarness(fake_record),
                        run_prefix=kw.pop("run_prefix", "pack"),
                        ready_timeout_s=1.0, measure_timeout_s=1.0)
    rr = ReplayReasoner(replies) if isinstance(replies, list) else replies
    return ModelPackerAgent(SFLEET, store, agent, rr, top_k=2, **kw), rr


with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    packer, rr = _packer(store, [
        _call("solve", cap=2),
        _call("recall", candidate=1),
        _call("measure", choice={"candidate": 1}, label="reasoned", expectation=_EXPECT_ALL,
              rationale="NO EVIDENCE for either structure; start with the densest"),
        _call("read_run", run_id="pack-000-reasoned"),
        _call("submit", verdict="met", choice={"candidate": 1},
              rationale="run pack-000-reasoned holds every stated target"),
    ], budget=2)
    res = packer.run(teardown_at_end=True)
    check("the reasoner drives the whole run: it solves, recalls, measures, reads and submits",
          res.turns == 5 and len(res.measured) == 1
          and res.stopped_because == STOP_SUBMITTED,
          f"{res.turns} turns, {res.stopped_because}")
    check("  and every tool answer came back as the next turn of one conversation, so what it "
          "was shown is recoverable from the transcript",
          # The opening, five replies, and a tool answer for each but `submit`, which ends the run.
          len(res.transcript.turns) == 10
          and "NO EVIDENCE" in res.transcript.turns[4].text,
          f"{len(res.transcript.turns)} turns")
    check("  and the verdict is derived from the record rather than accepted from the reply",
          res.verdict == "met" and res.met_run_id() == res.attempts[0].run_id)
    check("  and the store holds the measurement with the reasoner's own words attached",
          len(store.load().observations) == 1
          and "NO EVIDENCE" in store.load().observations[0].provenance["reasoner_rationale"])
    check("  and it spent one of the two measurements, because it decided one was enough",
          len(res.measured) == 1 and res.attempts[0].run_id == "pack-000-reasoned")
    check("  and the fleet was torn down at the end, because it was asked for",
          packer.agent.deployer.live_tenant_releases() == [])
    # Read back out of the store's provenance rather than from the live tool object, so a run
    # somebody else did can be reported the same way.
    check("belief against outcome is reported for the whole run, and stored nowhere",
          any("as you expected (hold)" in ln
              for ln in expectation_report(res, {t.name: t.slo for t in SFLEET.tenants}))
          and "expectation_met" not in json.dumps(store.load().observations[0].provenance))

# --- the best result, whether or not it passed -------------------------------
# A search that met nothing still learned which direction was closest, and until this existed that
# was thrown away: the summary recorded the failure and not the ranking. The numbers here are the
# first 14B/30B run's, because they are what showed the old `best()` to be wrong.
_TARGETS = {"chat": TenantSlo(itl_mean_ms=40.0, ttft_p95_ms=1000.0),
            "summ": TenantSlo(itl_mean_ms=55.0, ttft_p95_ms=2500.0),
            "reason": TenantSlo(itl_mean_ms=30.0, ttft_p95_ms=1000.0)}


def _att(run_id, gpus_used, verdict, itls, outcome="measured") -> Attempt:
    return Attempt(run_id=run_id, label=run_id, outcome=outcome,
                   candidate=SimpleNamespace(
                       gpus_used=gpus_used,
                       placement=SimpleNamespace(assignment={n: ("GPU-0",) for n in itls})),
                   record={"verdict": verdict,
                           "per_tenant": {n: {"itl_mean_ms": v} for n, v in itls.items()}})


_a001 = _att("001", 4, VERDICT_MEASURED,
             {"chat": 53.410, "summ": 40.0, "reason": 28.0})     # worst 53.410/40 = 1.335x
_a005 = _att("005", 7, VERDICT_MEASURED,
             {"chat": 34.0, "summ": 41.954, "reason": 41.940})   # worst 41.940/30 = 1.398x
# Raw ms ranks 005 first (41.954 < 53.410). By margin ratio 001 is the closer of the two, and it
# used four GPUs rather than seven -- so the old ordering preferred the worse placement on both
# counts at once.
_found = closest_to_targets([_a001, _a005], _TARGETS)
check("the closest placement is ranked by margin ratio, not by raw milliseconds",
      _found is not None and _found[0].run_id == "001",
      f"{_found[0].run_id} at {_found[1]:.3f}x" if _found else "None")
check("  and a tie on ratio breaks toward fewer GPUs",
      closest_to_targets(
          [_att("wide", 8, VERDICT_MEASURED, {"chat": 40.0, "summ": 55.0, "reason": 30.0}),
           _att("tight", 4, VERDICT_MEASURED, {"chat": 40.0, "summ": 55.0, "reason": 30.0})],
          _TARGETS)[0].run_id == "tight")
# The substantive fix. `outcome` says the harness ran; the record's verdict says the numbers mean
# something. A no-steady-state run's latency is not a fast placement.
check("  and a record whose verdict is not 'measured' is not eligible, however fast it reads",
      closest_to_targets(
          [_att("nss", 4, "infeasible_no_steady_state",
                {"chat": 1.0, "summ": 1.0, "reason": 1.0}), _a005], _TARGETS)[0].run_id == "005")
# Suppression is the delivery gate refusing to report latency it does not stand behind. Treating
# the absence as fast would rank an overloaded placement first -- the survivorship bias the
# suppression exists to prevent.
check("  and a targeted tenant with no ITL disqualifies the run rather than counting as fast",
      closest_to_targets(
          [_att("supp", 2, VERDICT_MEASURED,
                {"chat": 10.0, "summ": 10.0, "reason": None}), _a005], _TARGETS)[0].run_id
      == "005")
check("  and a fleet with nothing to be near has no closest, rather than an arbitrary winner",
      closest_to_targets([_a001, _a005], {"chat": None, "summ": None, "reason": None}) is None)
check("  and a run that measured nothing returns None instead of raising",
      closest_to_targets([], _TARGETS) is None)

# The summary has to carry the thresholds its `targets_met` booleans were judged against. The
# records deliberately do not -- a verdict is computed when a record is read, so tightening a target
# un-certifies an old pass, which is right for evidence. It is exactly why the *document* must
# write the targets down: without them it records an opinion whose premise is a file that has since
# moved on.
from colocation.agent.__main__ import _tenant_definitions          # noqa: E402
_defs = _tenant_definitions(SFLEET)
check("the summary records the tenants it was judged against, targets included",
      set(_defs) == {t.name for t in SFLEET.tenants}
      and _defs["r2"]["slo"]["itl_mean_ms"] == 8.0
      and _defs["r2"]["tp"] == 2 and _defs["r2"]["rate_hz"] == 16.0
      and _defs["r2"]["isl_tokens"] == 512)
check("  and a tenant that stated no target records None, not an absent key or a default",
      "slo" in _defs["t2"] and _defs["t2"]["slo"] is None)

# --- what the run is finally scored on: GPUs against the disjoint baseline (impl §0) ---
# Everything above this point is a measurement. This is the conclusion, and it is one number in one
# unit: how many GPUs the fleet occupies sharing, against how many it occupies not sharing. A run
# that measured beautifully and freed nothing did not pay for itself, and the report has to say so.
from colocation.agent.__main__ import (_baseline, _certified,       # noqa: E402
                                       _print_outcome, _report, _vs_solo)

# Every tenant targeted, because a fleet with an untargeted tenant can never be certified -- checked
# separately below, on SFLEET, which has one.
OFLEET = dataclasses.replace(FLEET, tenants=tuple(
    dataclasses.replace(t, slo=TenantSlo(itl_mean_ms=8.0)) for t in FLEET.tenants))
_obase = _baseline(OFLEET)


def _oatt(run_id, gpus_used, itls, margin=2.0, outcome="measured", verdict=VERDICT_MEASURED,
          pool=_POOL, reserve_source=_RSRC):
    return Attempt(run_id=run_id, label=run_id, outcome=outcome,
                   candidate=SimpleNamespace(
                       gpus_used=gpus_used, min_margin=margin,
                       # Everything on one GPU, so the co-residency bucket is the tenant count --
                       # which is what the calibrated half-width is looked up by.
                       coresidency={"GPU-0": len(itls)},
                       placement=SimpleNamespace(assignment={n: ("GPU-0",) for n in itls})),
                   record={"verdict": verdict,
                           "per_tenant": {n: {"itl_mean_ms": v} for n, v in itls.items()},
                           # What this run was DRIVEN with, as `loop._provenance` writes it. The
                           # `vs solo` block reads it against the solo row's own to tell whether the
                           # two are the same experiment.
                           "provenance": {"kv_tokens": {n: pool for n in itls},
                                          "mps_reserve_source": {n: reserve_source for n in itls}}})


_ALLMET = {"m1": 5.0, "r1": 5.0, "r2": 5.0, "t2": 5.0}
_res = SimpleNamespace(attempts=[_oatt("wide", 6, _ALLMET), _oatt("tight", 3, _ALLMET)],
                       stopped_because="budget", verdict="met", submitted=None, turns=1)
check("the placement a run may claim is the fewest-GPU one that met every target",
      _certified(_res, OFLEET)[0].run_id == "tight")
check("  and a tie on GPUs breaks toward the wider KV margin, as the solver's own tie-break does",
      _certified(SimpleNamespace(attempts=[_oatt("thin", 3, _ALLMET, margin=1.1),
                                           _oatt("roomy", 3, _ALLMET, margin=4.0)]),
                 OFLEET)[0].run_id == "roomy")
# The two rejections that matter, and they are not the same rejection. A miss is a measured answer;
# no verdict is the absence of one. Counting either as a pass certifies a fleet on evidence that
# does not exist -- which is the one failure mode this whole design is arranged against.
_mixed = SimpleNamespace(attempts=[_oatt("miss", 2, dict(_ALLMET, r2=90.0)),
                                   _oatt("nov", 2, dict(_ALLMET, t2=None)),
                                   _oatt("ok", 5, _ALLMET)])
_best, _rej = _certified(_mixed, OFLEET)
check("  and a placement that missed is not certified, with the tenant that missed named",
      _best.run_id == "ok" and ("miss", "missed r2") in [(a.run_id, w) for a, w in _rej])
check("  and neither is one where a tenant has no verdict, which is not a pass at 0 ms",
      ("nov", "no verdict for t2") in [(a.run_id, w) for a, w in _rej])
check("  and an attempt the harness never measured is not considered at all",
      _certified(SimpleNamespace(attempts=[_oatt("nr", 1, _ALLMET, outcome="deploy_failed")]),
                 OFLEET) == (None, []))
check("a fleet with an untargeted tenant can be measured but never certified, however fast it ran",
      _certified(SimpleNamespace(attempts=[_oatt("fast", 2, {n: 0.5 for n in _ALLMET})]),
                 SFLEET)[0] is None)

with tempfile.TemporaryDirectory() as _otd:
    _ops = ProfileStore(Path(_otd) / "profiles", _pkey)
    _ops.write(_solo(1, 4.0, run_id="solo-a"))
    _ops.write(_solo(2, 3.0, run_id="solo-b"))
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        _out = _print_outcome(_res, OFLEET, _obase, _ops)
    _txt = _buf.getvalue()
    # m1+r1 at tp1 and r2+t2 at tp2 is 6 GPUs disjoint; the claimed placement used 3.
    check("the report ends on BASELINE, STRICT and RELAXED, all in GPUs, each carrying its own "
          "saving rather than a separate FREED line",
          _out["baseline_gpus"] == 6 and _out["strict"]["gpus"] == 3
          and _out["strict"]["freed"] == 3 and _out["strict"]["run_id"] == "tight"
          and "BASELINE  6 GPU(s) of 8" in _txt
          and "STRICT    3 GPU(s), 3 freed (50% of the baseline)" in _txt
          and "RELAXED" in _txt,
          f"{_out['baseline_gpus']} -> {_out['strict']['gpus']}")
    # The word did the work `ACHIEVED` could not: a certification says so on its own line, and the
    # relaxed reading is never allowed to look like one.
    check("  and only STRICT certifies -- the word is on the line, and RELAXED never carries it",
          _out["strict"]["certified"] is True and "as written. CERTIFIED." in _txt
          and _out["relaxed"]["certified"] is False
          and (not _out["relaxed"]["offered"] or "NOT CERTIFIED" in _txt))
    # It is printed even with nothing in it. An empty slot is what tells a reader that the question
    # was asked and answered, rather than never asked.
    check("  and the RELAXED slot prints even when there is nothing on offer, with the reason",
          "RELAXED   nothing on offer:" in _txt and _out["relaxed"]["offered"] is False
          and _out["relaxed"]["why_not"])
    # r2 and t2 carry tp2 on a 0.6B model that fits at tp1, so 2 of those 6 GPUs are above the
    # memory floor. Whether they are co-location's to sell depends on something arithmetic cannot
    # see: if tp1 was MEASURED to miss their target then 6 is the baseline, and if tp1 was never
    # measured then 6 is an upper bound. While the fleet file could state `tp:` the second case was
    # the one to assume; now that it cannot, the first is, and the flag comes from the resolution
    # (`fleetfile.upper_bound_tp`) rather than from `resolved_tp > floor_tp`.
    check("  a baseline above the memory floor is not called inflated on arithmetic alone, because "
          "the ordinary reason for it is that the narrower degree was measured and missed",
          not _out["baseline_is_possibly_inflated"]
          and _out["baseline_tenants_above_memory_floor"] == ["r2", "t2"]
          and "and possibly" not in _txt)
    _bufu = io.StringIO()
    with contextlib.redirect_stdout(_bufu):
        _outu = _print_outcome(_res, OFLEET, _obase, _ops, upper_bound=("r2", "t2"))
    check("  and it IS called inflated when a narrower degree was never profiled, so the saving is "
          "not credited to sharing what re-sharding alone would have freed",
          _outu["baseline_is_possibly_inflated"]
          and _outu["baseline_at_memory_floor_gpus"] == 4
          and _outu["baseline_upper_bound_tenants"] == ["r2", "t2"]
          and "possibly as low as 4" in _bufu.getvalue()
          and "never profiled" in _bufu.getvalue())
    # The price §0 says co-location pays. Read at the tenant's deployed degree, so it holds whether
    # that degree was hand-stated or resolved from this same store.
    check("  and each tenant's co-resident latency is printed against its own solo latency at the "
          "degree it was deployed at",
          _out["vs_solo"]["m1"] == {"tp": 1, "coresident_itl_ms": 5.0, "solo_itl_ms": 4.0,
                                    "ratio": 1.25, "target_itl_ms": 8.0, "solo_run_id": "solo-a",
                                    "solo_kv_tokens": _POOL, "coresident_kv_tokens": _POOL,
                                    "solo_reserve_source": _RSRC,
                                    "coresident_reserve_source": _RSRC,
                                    "comparable": True, "incomparable_reason": None,
                                    "bound_violated": False}
          and _out["vs_solo"]["r2"]["solo_itl_ms"] == 3.0
          and _out["vs_solo"]["r2"]["ratio"] == round(5.0 / 3.0, 4)
          and "1.25x" in _txt and "of its 8.000 ms target" in _txt,
          json.dumps(_out["vs_solo"]))

    # --- the ratio is a comparison only if both sides were the same experiment ------------------
    # A reserve can only be measured on a solo round, so the FIRST row for a (model, degree,
    # environment) is measured on `default(tpN)` and writes the figure the fleet loader then merges
    # in; every run after it, co-resident runs included, is driven on `measured@tpN`. The reference
    # and the thing compared against it therefore differ BY CONSTRUCTION the first time a fleet goes
    # through the pipeline. That is not an anomaly to spot in one run, which is why it is checked in
    # code: on 2026-09-21 `qwen35-multimodal-2`'s docqa measured 56.253 ms solo on a 668,777-token
    # pool and 52.254 ms co-resident on 678,627 -- 4 ms apart, against a 0.0085 ms calibrated
    # between-run half-width, with the co-resident run FASTER than the bound it was quoted against.
    _bufp = io.StringIO()
    with contextlib.redirect_stdout(_bufp):
        _outp = _print_outcome(
            SimpleNamespace(attempts=[_oatt("pool", 3, _ALLMET, pool=int(_POOL * 1.02),
                                            reserve_source="default(tpN)")],
                            stopped_because="budget", verdict="met", submitted=None, turns=1),
            OFLEET, _obase, _ops)
    _txtp = _bufp.getvalue()
    check("a ratio taken across two different KV pools is WITHHELD rather than quoted",
          _outp["vs_solo"]["m1"]["comparable"] is False
          and _outp["vs_solo"]["m1"]["ratio"] == 1.25
          and "ratio withheld" in _txtp and "1.25x" not in _txtp,
          json.dumps(_outp["vs_solo"]["m1"]))
    check("  and both latencies still print, because both are measurements and both are true",
          "5.000 ms" in _txtp and "4.000 ms solo" in _txtp)
    check("  and the reason names the two pools and the reserve that moved between them",
          "600,000-token KV pool" in _txtp and "612,000" in _txtp
          and "default(tpN)" in _txtp and "2.00% apart" in _txtp,
          _outp["vs_solo"]["m1"]["incomparable_reason"] or "")
    check("  and it says what one round would buy, since the remedy is not a thing to decide",
          "--reprofile" in _txtp and "becomes quotable" in _txtp)
    check("  while the conditions themselves reach the summary, so it can be re-judged without "
          "the store and the record both to hand",
          (_outp["vs_solo"]["m1"]["solo_kv_tokens"],
           _outp["vs_solo"]["m1"]["coresident_kv_tokens"],
           _outp["vs_solo"]["m1"]["coresident_reserve_source"])
          == (_POOL, int(_POOL * 1.02), "default(tpN)"))
    # Pools within the tolerance but a reserve that still came from somewhere else. Checked
    # separately because it is a different fact: the experiments happen to have landed together, and
    # the INPUT still differed -- which says the next fleet will differ too.
    _vs_src = _vs_solo(_oatt("src", 3, _ALLMET, reserve_source="default(tpN)"), OFLEET, _ops)
    check("  and a reserve from a different source is enough on its own, pools notwithstanding",
          _vs_src["m1"]["comparable"] is False
          and "different inputs" in _vs_src["m1"]["incomparable_reason"],
          _vs_src["m1"]["incomparable_reason"] or "")
    check("  and a pool inside the tolerance with the same reserve is still a comparison",
          _vs_solo(_oatt("near", 3, _ALLMET, pool=int(_POOL * 1.004)),
                   OFLEET, _ops)["m1"]["comparable"] is True)
    # A row written before these fields existed cannot support the claim that the conditions
    # matched. Blank reads as "not known", never as "the same" -- the safe direction, and the same
    # answer a genuinely different pool gets.
    with tempfile.TemporaryDirectory() as _otd2:
        _ops_old = ProfileStore(Path(_otd2) / "profiles", _pkey)
        _ops_old.write(_solo(1, 4.0, run_id="old-a", kv_tokens=None, reserve_source=""))
        _vs_old = _vs_solo(_oatt("x", 3, _ALLMET), OFLEET, _ops_old)
        check("  and a row that never recorded its conditions is not assumed to have matched them",
              _vs_old["m1"]["comparable"] is False
              and "does not record the KV pool" in _vs_old["m1"]["incomparable_reason"],
              _vs_old["m1"]["incomparable_reason"] or "")
    # The other way the ratio stops meaning what it says, and it needs no second condition to
    # detect: solo is the same tenant alone on whole GPUs, so it is the best case by construction
    # and 1.00 is the ratio's floor. Below it, the bound is not bounding -- whatever the cause.
    _bufb = io.StringIO()
    with contextlib.redirect_stdout(_bufb):
        _outb = _print_outcome(
            SimpleNamespace(attempts=[_oatt("fastr", 3, dict(_ALLMET, m1=3.0))],
                            stopped_because="budget", verdict="met", submitted=None, turns=1),
            OFLEET, _obase, _ops)
    check("a co-resident run faster than its own solo reference is called out as a bound that did "
          "not bound, not reported as a saving",
          _outb["vs_solo"]["m1"]["bound_violated"] is True
          and "FASTER than solo" in _bufb.getvalue()
          and "25.0%" in _bufb.getvalue(),
          json.dumps(_outb["vs_solo"]["m1"]))
    check("  and the ordinary case is not flagged, so the flag stays worth reading",
          _out["vs_solo"]["m1"]["bound_violated"] is False
          and "FASTER than solo" not in _txt)

    _buf2 = io.StringIO()
    with contextlib.redirect_stdout(_buf2):
        _out2 = _print_outcome(_mixed, OFLEET, _obase, None)
    # Why the saving stopped where it did. Both denser attempts were measured and refused, and a
    # reader who sees only the claimed 5 GPUs cannot tell that from nobody having tried 2.
    check("  and a denser placement that was measured and refused is named beside the claim, so "
          "the saving is not read as the limit of what was attempted",
          _out2["strict"]["gpus"] == 5
          and "denser but not certified: miss on 2 GPU(s), missed r2" in _buf2.getvalue()
          and "nov on 2 GPU(s), no verdict for t2" in _buf2.getvalue())
    check("  and with no profile store there is no solo column invented, only where to get one",
          _out2["vs_solo"]["m1"]["solo_itl_ms"] is None
          and "run `profile`" in _buf2.getvalue())
    # A capped placement holds only with its caps (impl §16), so the report names them wherever it
    # names the placement -- the summary carrying them is not enough for a reader of the screen.
    def _full(att, **caps):
        # The results table reads the solved envelope too, which the outcome fixture leaves out.
        att.candidate.floor_tokens = {}
        att.candidate.placement.envelope_gib = att.candidate.placement.kv_tokens = {}
        return dataclasses.replace(att, sm_caps=caps)
    _capped = SimpleNamespace(attempts=[
        _full(_oatt("miss", 2, dict(_ALLMET, r2=90.0)), r2=40),
        _full(_oatt("ok", 5, _ALLMET), t2=25, m1=60),
        _full(_oatt("open", 6, _ALLMET))], stopped_because="budget", verdict="met",
        submitted=None, turns=1)
    _buf3 = io.StringIO()
    with contextlib.redirect_stdout(_buf3):
        _print_outcome(_capped, OFLEET, _obase, None)
        _rep3 = _report(_capped, OFLEET)
    _t3 = _buf3.getvalue()
    check("the report prints a capped placement's caps beside it, on STRICT, on the denser miss and "
          "in the results table, and an uncapped one with none",
          "t2:GPU-0, SM cap m1 60% t2 25%)" in _t3
          and "miss on 2 GPU(s), SM cap r2 40%, missed r2" in _t3
          and sum("SM cap" in ln for ln in _t3.splitlines() if ln.startswith("open ")) == 0
          and any(ln.startswith("ok ") and ln.endswith(", SM cap m1 60% t2 25%")
                  for ln in _t3.splitlines()), _t3)
    check("  and the closest-to-targets row carries them, an uncapped one without the key",
          _rep3["closest_to_targets"]["run_id"] == "ok"
          and _rep3["closest_to_targets"]["active_thread_percentage"] == {"t2": 25, "m1": 60}
          and "on 5 GPUs, SM cap m1 60% t2 25%, " in _t3
          and "active_thread_percentage" not in _report(
              SimpleNamespace(attempts=[_capped.attempts[2]], stopped_because="budget",
                              verdict="met", submitted=None, turns=1), OFLEET)["closest_to_targets"])
    # A degree that was never profiled must not borrow another degree's number: tp4 is a different
    # measurement, and reporting it as this tenant's solo baseline would overstate or flatter the
    # cost of sharing depending on which way it fell.
    _wide = dataclasses.replace(OFLEET, tenants=tuple(
        dataclasses.replace(t, tp=4) for t in OFLEET.tenants))
    check("  and an unprofiled degree gets no comparison rather than the nearest degree's row",
          _vs_solo(_oatt("x", 4, _ALLMET), _wide, _ops)["m1"]["solo_itl_ms"] is None)
    # `usable` is the profile store's own rule: a row whose delivery gate failed is not a latency.
    _ops.write(_solo(4, 2.0, delivery_passed=False, run_id="solo-bad"))
    check("  and a solo row whose delivery gate failed is not used as a baseline either",
          _vs_solo(_oatt("x", 4, _ALLMET), _wide, _ops)["m1"]["solo_itl_ms"] is None)

_buf3 = io.StringIO()
with contextlib.redirect_stdout(_buf3):
    _out3 = _print_outcome(
        SimpleNamespace(attempts=[_oatt("miss", 2, dict(_ALLMET, r2=90.0))]), OFLEET, _obase, None)
check("a run where nothing met every target certifies no saving at all, not the densest miss",
      _out3["strict"]["gpus"] is None and _out3["strict"]["freed"] is None
      and _out3["strict"]["certified"] is False
      and [{k: v for k, v in e.items() if k != "required_relaxation"}
           for e in _out3["not_claimable"]] == [{"run_id": "miss", "gpus_used": 2,
                                                 "why": "missed r2"}]
      and "STRICT    no placement. All 6 GPU(s) stand, 0 freed" in _buf3.getvalue())
# r2 at 90 ms against an 8 ms ceiling is 1025% over -- far past any tolerance -- so the placement is
# named in the beyond-tolerance list and never as an offer. The two must not format alike: that a
# 0.3% miss and a tenfold one appeared under one heading is what made the old report unreadable.
check("  and a miss far beyond every tolerance is listed as beyond it, never offered as RELAXED",
      _out3["relaxed"]["offered"] is False
      and "RELAXED   nothing on offer" in _buf3.getvalue()
      and "tolerance" in _buf3.getvalue()
      and "1025" in _buf3.getvalue())
_buf4 = io.StringIO()
with contextlib.redirect_stdout(_buf4):
    _out4 = _print_outcome(SimpleNamespace(attempts=[_oatt("dis", 6, _ALLMET)]), OFLEET,
                           _obase, None)
check("  and a placement that met everything while freeing nothing says so in those words, since "
      "co-location that sells nothing is a result about the fleet",
      _out4["strict"]["freed"] == 0 and "The saving is nothing" in _buf4.getvalue()
      and "strands no memory" in _buf4.getvalue())

# --- what it would take, in requirements rather than in GPUs ------------------
# Printed for every rejected placement denser than the claim, always -- not only when the search
# freed nothing. "missed r2" is not the finding; "missed r2 by 0.2 ms, which is 2.5% of one target
# out of four and 9x the calibrated noise" is, and only the second can be handed to the person who
# wrote the target. Nothing here is a claim: the placement stays rejected and the record unstamped.
from colocation.agent.__main__ import (DEFAULT_RELAX_TARGETS_PCT,     # noqa: E402
                                       _pct, _relaxation_lines, _relaxed_baseline, _relaxed_slos)

check("relaxing a target multiplies every stated ceiling and invents none for a tenant without one",
      _relaxed_slos(SFLEET, 1.05)["m1"].itl_mean_ms == 8.4
      and _relaxed_slos(SFLEET, 1.05)["t2"] is None)

_CAL = Calibration(between_run_itl_halfwidth_ms={"coresident=4": 0.022, "coresident=0": 0.5},
                   repeats=2)
_near = SimpleNamespace(attempts=[_oatt("near", 2, dict(_ALLMET, r2=8.2)), _oatt("ok", 5, _ALLMET)])
_buf5 = io.StringIO()
with contextlib.redirect_stdout(_buf5):
    _out5 = _print_outcome(_near, OFLEET, _obase, None, DEFAULT_RELAX_TARGETS_PCT, _CAL)
_t5 = _buf5.getvalue()
check("a denser placement that missed is reported with the smallest relaxation that would pass it",
      "denser but not certified: near on 2 GPU(s), missed r2" in _t5
      and "it would certify at r2 ITL mean 8.200 ms against its 8.000 ms ceiling" in _t5
      and "a 2.50% relaxation of one target and nothing else moves, within its default 5% "
          "tolerance" in _t5,
      _t5)
# "within the 5% cap" said nothing about who agreed to 5%. Whether the bound is the tenant's own
# statement or a fallback nobody wrote decides how much weight the reading carries.
check("  and the tolerance is named as the tenant's own or as the default, not just as a number",
      "its default 5% tolerance" in _t5 and "declared" not in _t5.split("RELAXED")[0], _t5)
# A fraction of a percent on one tenant is exactly the case with a real decision in it, so it is the
# case that must not round to nothing. `0.1f` printed 0.34% as "0.3%" and 0.04% as "0.0%".
check("  and a sub-percent relaxation keeps enough digits to stay a number rather than a rounding",
      _pct(0.0034) == "0.340%" and _pct(0.0004) == "0.040%" and _pct(0.025) == "2.50%"
      and _pct(0.361837) == "36.2%", f"{_pct(0.0034)} {_pct(0.0004)} {_pct(0.025)}")
# The number that decides whether relaxing is a decision about the requirement or about noise. A
# miss inside the calibrated half-width is not a miss anybody should move a target for.
check("  and the miss is stated against the calibrated half-width for that co-residency, so it can "
      "be told apart from run-to-run spread",
      "the miss is 0.200 ms, which is 9x the calibrated 0.0220 ms half-width, coresident=4's own"
      in _t5
      and "relaxing the target is a decision about the requirement rather than about noise" in _t5,
      _t5)
# The live run's own case: the dense placement sat at a co-residency degree nothing had been repeated
# at, so the number came from `tightest_ms`. That fallback is right for sizing a window (loop.py) and
# flattering here -- the smallest spread in the store makes any miss look like a larger multiple of
# noise than this arrangement's own might -- so the borrowing has to be on the line.
_buf5b = io.StringIO()
with contextlib.redirect_stdout(_buf5b):
    _print_outcome(_near, OFLEET, _obase, None, DEFAULT_RELAX_TARGETS_PCT,
                   Calibration(between_run_itl_halfwidth_ms={"coresident=2": 0.0221}, repeats=2))
check("    and a half-width borrowed from another co-residency bucket says it is borrowed, since the "
      "tightest in the store flatters every miss",
      "nothing repeated at coresident=4, so this is borrowed from another arrangement and is the "
      "smallest spread in the store" in _buf5b.getvalue(), _buf5b.getvalue())
check("  and it is recorded in the summary per run, not only printed",
      _out5["required_relaxation"]["near"]["factor"] == 1.025
      and _out5["required_relaxation"]["near"]["within_cap"] is True
      and _out5["not_claimable"][0]["required_relaxation"]["per_tenant"]["r2"]["over_by"]
      == round(8.2 - 8.0, 6),
      json.dumps(_out5["required_relaxation"]))
_buf6 = io.StringIO()
with contextlib.redirect_stdout(_buf6):
    _out6 = _print_outcome(_near, OFLEET, _obase, None, 1.0, _CAL)
check("  and the tolerance changes what is offered but never whether the number is printed, because "
      "the number is the finding",
      "a 2.50% relaxation of one target and nothing else moves, beyond its default 1% tolerance"
      in _buf6.getvalue() and _out6["relaxed"]["offered"] is False)
# The flag is a way to ask a narrower question than the fleet allows, never a way to widen it. If it
# could widen, a concession could be invented at the command line once the result was known -- which
# is the whole thing declaring the tolerance in the fleet file exists to prevent.
check("  and --relax-targets-pct can only lower a tenant's tolerance, never raise it",
      TenantSlo(itl_mean_ms=8.0).tolerance(1.0) == 1.0
      and TenantSlo(itl_mean_ms=8.0).tolerance(50.0) == DEFAULT_RELAX_TARGETS_PCT
      and TenantSlo(itl_mean_ms=8.0, tolerance_pct=1.0).tolerance(50.0) == 1.0
      and TenantSlo(itl_mean_ms=8.0, tolerance_pct=1.0).tolerance(None) == 1.0)
# 0 is a statement, not an absence: this ceiling does not move, so no reading may offer to move it.
_ZFLEET = dataclasses.replace(OFLEET, tenants=tuple(
    dataclasses.replace(t, slo=TenantSlo(itl_mean_ms=8.0,
                                         tolerance_pct=0.0 if t.name == "r2" else None))
    for t in OFLEET.tenants))
_buf6b = io.StringIO()
with contextlib.redirect_stdout(_buf6b):
    _out6b = _print_outcome(_near, _ZFLEET, _obase, None, None, _CAL)
check("  and a tenant that declared tolerance_pct: 0 is never offered a relaxed reading at all, "
      "because refusing to negotiate is a thing a fleet file can say",
      _out6b["relaxed"]["offered"] is False
      and "beyond its declared 0% tolerance" in _buf6b.getvalue()
      and "r2 needs 2.50% against a 0% tolerance" in _buf6b.getvalue(), _buf6b.getvalue())
# The case the whole slot exists for: nothing certifies, and a concession the fleet already permits
# frees GPUs. Under the old vocabulary this was an indented sub-bullet beneath a rejected candidate.
_TFLEET = dataclasses.replace(OFLEET, tenants=tuple(
    dataclasses.replace(t, slo=TenantSlo(itl_mean_ms=8.0,
                                         tolerance_pct=5.0 if t.name == "r2" else None))
    for t in OFLEET.tenants))
_buf6c = io.StringIO()
with contextlib.redirect_stdout(_buf6c):
    _out6c = _print_outcome(SimpleNamespace(attempts=[_oatt("near", 2, dict(_ALLMET, r2=8.2))]),
                            _TFLEET, _obase, None, None, _CAL)
_t6c = _buf6c.getvalue()
check("a run that certifies nothing still reports what a permitted concession would free, in the "
      "same unit and its own slot rather than buried under a rejected candidate",
      _out6c["strict"]["gpus"] is None and _out6c["relaxed"]["offered"] is True
      and _out6c["relaxed"]["gpus"] == 2 and _out6c["relaxed"]["freed"] == 4
      and _out6c["relaxed"]["certified"] is False
      and "RELAXED   2 GPU(s), 4 freed (67% of the baseline)" in _t6c, _t6c)
# In milliseconds of ceiling, because that is what the operator edits. "x1.025" is the solver's unit.
check("  and the concession is stated in the units the operator sets, not as a bare factor",
      "r2 ITL mean ceiling 8.000 -> 8.200 ms (+2.50%), within its declared 5% tolerance" in _t6c
      and _out6c["relaxed"]["concession"]["r2"]["declared_tolerance"] is True, _t6c)
check("  and it says how many tenants concede and that the others met their ceilings as written",
      "1 of 4 tenants concedes; the rest met their ceilings as written" in _t6c)
check("  and it is labelled NOT CERTIFIED on the headline and again beneath it",
      _t6c.count("NOT CERTIFIED") >= 2 and "No record is stamped with it" in _t6c)
# Without a placement to report there is no cost of sharing to state either, and inventing the
# comparison from the strict best when there is no strict best is how a number gets misattributed.
check("  and the cost of sharing is reported against whichever placement is being reported, naming "
      "it, so a relaxed offer is not compared against a certification that does not exist",
      "vs solo" in _t6c and "(near)" in _t6c)
_buf7 = io.StringIO()
with contextlib.redirect_stdout(_buf7):
    _print_outcome(_near, OFLEET, _obase, None, DEFAULT_RELAX_TARGETS_PCT, None)
check("  and with nothing repeated on this key it says so instead of substituting a noise floor",
      "no calibrated half-width" in _buf7.getvalue() and "one replicate settles it"
      in _buf7.getvalue() and "0.0220" not in _buf7.getvalue())
# Inside the noise the honest next step is another measurement, not a looser requirement -- and
# saying which is which is the whole reason the half-width is printed beside the miss.
_buf8 = io.StringIO()
with contextlib.redirect_stdout(_buf8):
    _print_outcome(SimpleNamespace(attempts=[_oatt("hair", 2, dict(_ALLMET, r2=8.01)),
                                             _oatt("ok", 5, _ALLMET)]),
                   OFLEET, _obase, None, DEFAULT_RELAX_TARGETS_PCT, _CAL)
check("  and a miss smaller than the half-width is called noise, with a replicate as the next step "
      "rather than a looser target",
      "INSIDE the noise" in _buf8.getvalue() and "a replicate is the honest next step"
      in _buf8.getvalue())
# No relaxation reaches a tenant that was not measured, and the report must not offer one. This is
# the delivery gate's edge (impl §6.3): a suppressed latency means the run served less than it was
# asked to, and moving a ceiling cannot supply the measurement that was skipped.
_buf9 = io.StringIO()
with contextlib.redirect_stdout(_buf9):
    _print_outcome(SimpleNamespace(attempts=[_oatt("nov", 2, dict(_ALLMET, t2=None)),
                                             _oatt("ok", 5, _ALLMET)]),
                   OFLEET, _obase, None, DEFAULT_RELAX_TARGETS_PCT, _CAL)
check("a placement with an unmeasured tenant is offered no relaxation at all, since a ceiling "
      "cannot supply a measurement",
      "no relaxation reaches it: t2 was not measured on this run" in _buf9.getvalue()
      and "would certify" not in _buf9.getvalue(), _buf9.getvalue())
# And when it both missed and left a tenant unmeasured, the miss is still worth stating -- but not as
# "it would certify", because relaxing every ceiling named would still leave the run unjudgeable.
_buf9b = io.StringIO()
with contextlib.redirect_stdout(_buf9b):
    _out9b = _print_outcome(SimpleNamespace(attempts=[_oatt("both", 2, dict(_ALLMET, r2=8.2,
                                                                           t2=None)),
                                                     _oatt("ok", 5, _ALLMET)]),
                            OFLEET, _obase, None, DEFAULT_RELAX_TARGETS_PCT, _CAL)
check("  and one that missed AND left a tenant unmeasured is never said to certify, however small "
      "the miss it would take",
      "it would still not certify, but it is at r2 ITL mean 8.200 ms" in _buf9b.getvalue()
      and "it would certify" not in _buf9b.getvalue()
      and "cap" not in _buf9b.getvalue()
      and _out9b["required_relaxation"]["both"]["within_cap"] is False,
      _buf9b.getvalue())
_buf10 = io.StringIO()
with contextlib.redirect_stdout(_buf10):
    _print_outcome(SimpleNamespace(attempts=[_oatt("untgt", 2, {n: 0.5 for n in _ALLMET})]),
                   SFLEET, _baseline(SFLEET), None, DEFAULT_RELAX_TARGETS_PCT, _CAL)
check("  and a tenant nobody set a target for is told to state one, not offered a looser version "
      "of a ceiling that does not exist",
      "t2 state no target, so there is no ceiling to move" in _buf10.getvalue())

# The trap the whole relaxation report is built around. Relaxing a target can narrow a tenant's
# minimal TP, and then `sum(tp)` -- the baseline -- falls too. Scoring a relaxed placement against
# the strict baseline would credit co-location with GPUs the relaxation alone freed, and §0 is a
# comparison of like with like or it is nothing.
with tempfile.TemporaryDirectory() as _rtd:
    _rps = ProfileStore(Path(_rtd) / "profiles", _pkey)
    _rps.write(_solo(1, 8.5, run_id="solo-tp1"))     # misses 8.0 solo, meets anything over 8.5
    _rps.write(_solo(2, 3.0, run_id="solo-tp2"))
    # r2 and t2 sit at tp2; at 7.5% every ceiling is 8.6, which tp1's 8.5 now meets.
    _rb = _relaxed_baseline(OFLEET, _obase, _rps, 1.075)
    check("a relaxation that narrows a tenant's minimal TP moves the baseline, and the report says "
          "so rather than scoring the saving against the strict one",
          _rb["moves_the_baseline"] and _rb["relaxed_baseline_gpus"] == 4
          and _rb["measured_minimal_baseline_gpus"] == 6
          and set(_rb["narrowed_by_the_relaxation"]) == {"r2", "t2"} and _rb["checked"],
          json.dumps(_rb))
    check("  and the same arithmetic at the stated targets moves nothing, so the two numbers differ "
          "by exactly what relaxing bought",
          _relaxed_baseline(OFLEET, _obase, _rps, 1.0)["moves_the_baseline"] is False
          and _relaxed_baseline(OFLEET, _obase, _rps, 1.0)["relaxed_baseline_gpus"] == 6)
    _buf11 = io.StringIO()
    with contextlib.redirect_stdout(_buf11):
        _print_outcome(SimpleNamespace(attempts=[_oatt("wide2", 2, dict(_ALLMET, r2=8.6)),
                                                 _oatt("ok", 5, _ALLMET)]),
                       OFLEET, _obase, _rps, None, _CAL)
    # Only r2 is conceding, so only r2's ceiling moves. Relaxing t2's too -- which a single global
    # factor did -- hands it slack nobody asked for, and slack is what lets a degree narrow, so the
    # baseline read as moving further than the reading would actually move it.
    check("  and it is printed beside the reading, with only the conceding tenant's ceiling moved, "
          "so the reading is never read as a saving relaxation had already made cheaper",
          "moves the BASELINE to 5 GPU(s) from 6 (r2 tp2->tp1 narrows once the target moves)"
          in _buf11.getvalue()
          and "credited with GPUs the relaxation alone freed" in _buf11.getvalue(),
          _buf11.getvalue())
    # And when it does move it, the offer's own saving is scored against the moved number on screen,
    # not just recorded in the summary -- that guard was in the JSON only and a reader had to dig.
    _TOL10 = dataclasses.replace(OFLEET, tenants=tuple(
        dataclasses.replace(t, slo=TenantSlo(itl_mean_ms=8.0,
                                             tolerance_pct=10.0 if t.name == "r2" else None))
        for t in OFLEET.tenants))
    _buf11b = io.StringIO()
    with contextlib.redirect_stdout(_buf11b):
        _o11b = _print_outcome(SimpleNamespace(attempts=[_oatt("wide2", 2, dict(_ALLMET, r2=8.6))]),
                               _TOL10, _obase, _rps, None, _CAL)
    check("    and the offer's saving is then scored against the narrowed baseline, on screen",
          _o11b["relaxed"]["offered"] is True
          and _o11b["relaxed"]["scored_against_baseline_gpus"] == 5
          and _o11b["relaxed"]["freed"] == 3
          and _o11b["relaxed"]["baseline_moves_under_the_concession"] is True
          and "narrows the BASELINE to 5 GPU(s), so the 3 above is scored against that and not "
              "against 6" in _buf11b.getvalue(), _buf11b.getvalue())
    # The mis-attribution on the other side. A tenant that would narrow at its *stated* target too is
    # an inflated baseline, reported on its own line above -- calling that the relaxation's doing
    # would name the wrong cause for GPUs that were never co-location's to sell either way.
    _fast = ProfileStore(Path(_rtd) / "fast", _pkey)
    _fast.write(_solo(1, 4.0, run_id="fast-tp1"))
    _rbs = _relaxed_baseline(OFLEET, _obase, _fast, 1.075)
    check("  and a tenant that would narrow at its stated target too is not credited to the "
          "relaxation, because that is an inflated baseline and it is reported as one",
          _rbs["relaxed_baseline_gpus"] == 4 and _rbs["measured_minimal_baseline_gpus"] == 4
          and _rbs["moves_the_baseline"] is False
          and set(_rbs["tenants_that_would_narrow"]) == {"r2", "t2"}
          and _rbs["narrowed_by_the_relaxation"] == {}, json.dumps(_rbs))
    # A tenant already at its memory floor cannot move however loose the target gets -- that is the
    # one case where "the baseline does not move" is certain rather than merely unrefuted.
    _flr = dataclasses.replace(OFLEET, tenants=tuple(
        dataclasses.replace(t, tp=1) for t in OFLEET.tenants))
    _rbf = _relaxed_baseline(_flr, _baseline(_flr), _rps, 2.0)
    check("  and a tenant at its memory floor cannot narrow at any relaxation, so that answer is "
          "certain rather than unchecked",
          _rbf["moves_the_baseline"] is False and _rbf["checked"]
          and _rbf["relaxed_baseline_gpus"] == 4 and _rbf["unchecked"] == {},
          json.dumps(_rbf))
# An unmeasured narrower degree is an open question, and answering it "no" is the same trap wearing
# the opposite sign: a saving scored against a baseline nobody checked.
_rbu = _relaxed_baseline(OFLEET, _obase, None, 1.5)
check("  and with no solo row below a stated degree the answer is unchecked, never a silent no",
      _rbu["checked"] is False and _rbu["unchecked"] == {"r2": [1], "t2": [1]}
      and _rbu["moves_the_baseline"] is False, json.dumps(_rbu))
_buf12 = io.StringIO()
with contextlib.redirect_stdout(_buf12):
    _print_outcome(_near, OFLEET, _obase, None, DEFAULT_RELAX_TARGETS_PCT, _CAL)
check("    and it names the profile run that would settle it, rather than another search",
      "whether it also moves the baseline is unchecked: no solo row for r2 at tp1, t2 at tp1"
      in _buf12.getvalue() and "`profile` is what settles it" in _buf12.getvalue())
# Print always. A run that freed two GPUs and stopped 1.45 ms short of a third has measured
# something the person who wrote that target needs to see, and the earlier rule -- print only when
# the search consolidated nothing -- would have hidden exactly that case.
check("the relaxation is printed whether or not the search consolidated anything, since the run "
      "that freed two GPUs and nearly freed a third is where the number matters most",
      _out5["strict"]["freed"] == 1 and "it would certify at r2" in _t5
      and "it would still not certify" not in _t5 and _out3["strict"]["freed"] is None)

# A reply that cannot be used is fed back, and after the retries the run does not die: it measures
# the solver's top-ranked candidate and records that nothing chose it (impl §11.2).
with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    packer, rr = _packer(store, ["no", "still no", "no again",
                                 _call("submit", verdict="unresolvable",
                                       rationale="I could not answer and the loop measured for me")],
                         budget=1, max_retries=2, run_prefix="fb")
    logging.getLogger("colocation.agent.packer").setLevel(logging.CRITICAL)
    res = packer.run()
    logging.getLogger("colocation.agent.packer").setLevel(logging.NOTSET)
    check("an unusable reply is fed back with the reason, and the model's own words are left in "
          "the conversation",
          "cannot be used" in rr.asked[1][2].text and rr.asked[1][1].text == "no")
    check("  and after the retries the run falls back to the solver's top-ranked candidate rather "
          "than dying",
          len(res.measured) == 1 and res.fallbacks == [res.attempts[0].run_id],
          f"{res.fallbacks}")
    check("  and the fallback is in the record, not only in the run summary: a later run reads the "
          "store, and a record claiming a reasoner chose it would be a lie",
          store.load().observations[0].provenance["policy_kind"] == PACKER_FALLBACK
          and store.load().observations[0].provenance["reasoner_failure"]
          and store.load().observations[0].provenance["fallback_policy"] == "TopRanked")
    check("  with every expectation `unknown`, because nothing formed a belief",
          set(store.load().observations[0].provenance["reasoner_expectation"].values())
          == {"unknown"})
    check("  and the reasoner is told what was measured on its behalf, so no record is left that "
          "no turn of the transcript accounts for",
          any("on your behalf" in t.text for t in res.transcript.turns))
    # Who chose the placement does not bear on whether the targets were held: the derivation asks
    # the record. So both verdicts are kept, and where they differ the difference is the finding.
    check("  and the derived verdict comes from the record even for a run nothing chose, while "
          "what the reasoner submitted is kept beside it",
          res.verdict == "met" and res.submitted["verdict"] == "unresolvable")

# The budget refusal reaches the reasoner as a refusal it can act on, and the run ends on its own
# conclusion rather than on the loop giving up.
with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    packer, rr = _packer(store, [
        _call("solve", cap=2),
        _call("measure", choice={"candidate": 1}, label="one", expectation=_EXPECT_ALL,
              rationale="start with the densest"),
        _call("measure", choice={"candidate": 2}, label="two", expectation=_EXPECT_ALL,
              rationale="and now the other structure"),
        _call("submit", verdict="budget", rationale="the budget is spent after one run"),
    ], budget=1, run_prefix="bud")
    res = packer.run()
    check("a measurement past the budget is refused as a tool refusal, which is not a malformed "
          "reply and does not end the run",
          len(res.measured) == 1 and res.submitted is not None
          and any("budget is spent" in t.text for t in res.transcript.turns)
          and res.transcript.retries == 0)
    check("  and a refused call is fed back naming the tool that refused it",
          any(t.text.startswith("measure refused:") for t in res.transcript.turns))

# A reasoner that never submits still terminates, on the turn cap, and the cap is named as what
# ended the run rather than dressed up as a conclusion.
with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    packer, rr = _packer(store, [_call("solve", cap=1)] * 3, budget=1, max_turns=3,
                         run_prefix="cap")
    res = packer.run()
    check("free tools cannot run forever: the turn cap ends the run and says so",
          res.turns == 3 and res.stopped_because == STOP_TURNS and not res.attempts)
    check("  and the verdict is not a conclusion the reasoner never reached",
          res.verdict == "unresolvable" and res.submitted is None)
# A NullReasoner raises if consulted, which is how a path claimed to reach no model is proved to.
with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    packer, _ = _packer(store, NullReasoner(), budget=1, max_turns=0, run_prefix="nil")
    check("a run with no turns left consults no model at all, and says the cap is why it stopped",
          packer.run().stopped_because == STOP_TURNS)

# `reason.ask` is the one-shot form of the same feedback rule, kept because Phase 7's hosting will
# ask a model a single question that is not a search. Its contract is that it raises rather than
# falling back: what to do when the reasoner cannot answer belongs to the caller.
_validate = (lambda t: prompt.parse_call(t))
_rr = ReplayReasoner(["that would be candidate 2", _call("solve", cap=2)])
_dec, _tr = ask(_rr, "sys", "the problem", validate=_validate)
check("a rejected reply is fed back and the corrected one is accepted",
      _dec["tool"] == "solve" and _tr.retries == 1)
_rr = ReplayReasoner(["no", "still no", "no again", "unreached"])
try:
    ask(_rr, "sys", "the problem", validate=_validate, max_retries=2)
    check("when the retries run out `ask` raises rather than choosing for the caller", False)
except prompt.InvalidReply:
    check("when the retries run out `ask` raises rather than choosing for the caller",
          len(_rr.asked) == 3, f"{len(_rr.asked)} asks")
check("the transcript can drop the prompt and keep the replies, which are the unrecoverable part",
      set(_tr.to_dict(include_prompt=False)) == {"reasoner", "usage", "latency_s", "retries",
                                                 "replies"}
      and "system" in _tr.to_dict())

# --- the same search, driven by the SDK (impl §11, §11.4) --------------------
#
# Everything above the SDK boundary and nothing that needs it installed: the schemas the six tools
# are registered with, the gate's decisions, and the translation of a tool's answer into text. What
# cannot be checked here is the boundary itself -- that the SDK spawns its CLI, authenticates and
# routes a `tool_use` block to the right function -- so it is not faked here either. It is stated in
# `sdkpacker.py` as the one thing the first live run is the first to exercise.

check("the shipped driver's module imports without the SDK present, which is what makes the rest "
      "of this section checkable at all",
      "claude_agent_sdk" not in sys.modules)
check("both drivers fall back and derive their verdict through one implementation, so a run that "
      "fell back looks the same in the store whichever drove it",
      sdkpacker.recorded_fallback is recorded_fallback
      and sdkpacker.derive_verdict is derive_verdict)
# The note is shared; the sentence telling the model *how* to reply is the driver's, because under
# the SDK there is no such envelope and the instruction would be an instruction to fail.
_fb = PackerResult(fallbacks=["pack-000-x"])
check("  while what a fallback tells the reasoner is shared and only the transport instruction "
      "differs between them",
      "on your behalf" in fallback_note(_fb) and "JSON" not in fallback_note(_fb))

check("every one of the six tools has both a schema and a one-line summary, so none can reach the "
      "registration surface nameless or undescribed",
      sorted(prompt.TOOL_SCHEMAS) == sorted(prompt.TOOLS) == sorted(prompt.TOOL_SUMMARIES))
# A schema that rejected a call first would replace a refusal written to be read by the model with a
# generic transport error, and put a second unreviewed copy of the invariants outside `tools.py`.
check("  and a schema requires only what its tool cannot proceed without, leaving every real rule "
      "in the tool body where the refusal explains itself",
      prompt.TOOL_SCHEMAS["measure"]["required"] == ["choice", "expectation", "rationale"]
      and not any("required" in prompt.TOOL_SCHEMAS[t]
                  for t in ("solve", "recall", "cluster_status")))
check("  and the model is told what a tool *means*, not just its argument shape: the catalogue is "
      "in both system prompts verbatim",
      prompt._CATALOGUE in prompt.SYSTEM and prompt._CATALOGUE in prompt.SYSTEM_SDK
      and prompt._RULES in prompt.SYSTEM and prompt._RULES in prompt.SYSTEM_SDK
      and prompt._PREAMBLE in prompt.SYSTEM and prompt._PREAMBLE in prompt.SYSTEM_SDK)
check("  and the only thing that differs is how a call is delivered, so what the agent is told "
      "cannot drift between the two runtimes",
      "single JSON object" in prompt.SYSTEM and "single JSON object" not in prompt.SYSTEM_SDK
      and "no shell" in prompt.SYSTEM_SDK)

with tempfile.TemporaryDirectory() as td:
    store = ObservationStore(td, AGENT_KEY)
    store.write_key()
    agent = SearchAgent(SFLEET, store, FakeDeployer(), FakeHarness(fake_record),
                        run_prefix="sdk", ready_timeout_s=1.0, measure_timeout_s=1.0)
    sdk = SdkPackerAgent(SFLEET, store, agent, budget=1, top_k=2, discovery_tools=())

    check("the registration surface is built from the prompt, so a seventh tool cannot appear at "
          "the SDK boundary without appearing in the catalogue the model reads",
          [n for n, _, _ in sdk.tool_specs()] == list(prompt.TOOLS)
          and all(desc and schema for _, desc, schema in sdk.tool_specs()))

    # This gate is what makes the measurement budget mean anything: `tools.tool_measure` enforces
    # it, and a reachable shell would make it decorative -- the model could install a fleet itself,
    # spend the node, and write no record for a later run to read.
    check("the six tools are reachable, and the gate is written in the qualified names the hook is "
          "actually handed",
          all(sdk.gate(sdkpacker.qualified(t), submitted=False) is None for t in prompt.TOOLS)
          and sdkpacker.OUR_TOOLS == frozenset(sdkpacker.qualified(t) for t in prompt.TOOLS))
    _bash = sdk.gate("Bash", submitted=False)
    check("  a shell is refused, and the refusal is a sentence naming what the model does have, "
          "not a mute failure it will retry",
          _bash and "not available" in _bash and all(t in _bash for t in prompt.TOOLS))
    check("  and so is every builtin the CLI ships that writes, reads or spawns another agent",
          all(sdk.gate(b, submitted=False) for b in sdkpacker.BUILTINS_DENIED))
    check("  and so is a bare name that collides with one of ours, because a builtin sharing one "
          "would otherwise be allowed through by the collision",
          all(sdk.gate(t, submitted=False) for t in prompt.TOOLS))
    # Measured against the fakes: with no exception configured the model tried `ToolSearch` first,
    # was refused, read the refusal and called all six by name. So the schemas load eagerly, the
    # escape hatch is unnecessary, and the deny-by-default set is allowed to have no exception.
    check("  and by default nothing is excepted from deny-by-default, not even a tool that only "
          "reads schemas, because the six arrive loaded and none needs discovering",
          SdkPackerAgent(SFLEET, store, agent).discovery_tools == frozenset()
          and SdkPackerAgent(SFLEET, store, agent).gate("ToolSearch", submitted=False))
    # And the prompt says so too -- though measured, saying it is not what works: in three probe
    # runs the model opened with a schema lookup anyway, was refused, and recovered immediately
    # ("Understood -- calling the six directly"). Two turns, no GPU. It is the refusal that teaches,
    # which is the whole argument for writing one to be read instead of failing mutely.
    check("  and the prompt says as much as well, so the refusal is a correction and not the first "
          "the model hears of it",
          "nothing to look up" in prompt.SYSTEM_SDK)
    check("  while the exception remains configurable, since a future runtime may defer a schema "
          "and a model that cannot read one cannot call anything at all",
          SdkPackerAgent(SFLEET, store, agent, discovery_tools=("ToolSearch",)
                         ).gate("ToolSearch", submitted=False) is None)
    check("  and after `submit` no further call is accepted, so a run cannot spend the node on a "
          "question whose answer is already recorded",
          sdk.gate(sdkpacker.qualified("measure"), submitted=True)
          and sdk.gate(sdkpacker.qualified("solve"), submitted=True))
    check("  except `cluster_status`, which changes nothing and is how the node is checked to have "
          "been left as intended",
          sdk.gate(sdkpacker.qualified("cluster_status"), submitted=True) is None)

    tools = Tools(SFLEET, store, agent, budget=1, top_k=2, reasoner_name=sdk.name)
    tr = Transcript(reasoner=sdk.name, system=prompt.SYSTEM_SDK,
                    turns=[Turn("user", sdk.opening(tools))])
    check("the opening turn is the same problem statement, closing on the one instruction this "
          "runtime can honour rather than on the other driver's envelope",
          "== THE PROBLEM ==" in tr.turns[0].text
          and prompt.SDK_CLOSING in tr.turns[0].text
          and prompt.JSON_CLOSING not in tr.turns[0].text)

    _out = sdk.call_tool(tools, "solve", {"cap": 2}, tr)
    check("a tool's answer comes back as text and is appended to the transcript, so what the model "
          "was shown is recoverable from the record and not only from the SDK's own log",
          _out and tr.turns[-1].text == _out and tr.turns[-1].role == "user")
    _refused = sdk.call_tool(tools, "read_run", {"run_id": "no-such-run"}, tr)
    check("  a refused call becomes a refusal the model can read, naming the tool that refused it, "
          "rather than an exception the SDK would turn into a transport error",
          _refused.startswith("read_run refused:") and "no run called" in _refused)
    _done = sdk.call_tool(tools, "submit", {"verdict": "unresolvable",
                                           "rationale": "NO EVIDENCE either way"}, tr)
    check("  and `submit` comes back as a sentence telling the model to stop: under this runtime it "
          "sees the result of submitting, and the bare sentinel does not say that",
          _done != SUBMITTED and "no further tool call" in _done
          and tools.submitted is not None)


class _Unreadable:
    """A store that cannot be fully parsed. Not a refusal, and not the model's problem to route."""

    def dispatch(self, name, args):
        raise RuntimeError("half the store cannot be parsed")


try:
    sdk.call_tool(_Unreadable(), "recall", {}, tr)
    check("a fault is never handed to the model as a tool result", False)
except RuntimeError:
    # A `ToolError` is a judgement about a call; anything else is a fault in the agent. Told about
    # one, the model would work around it -- and a search against a history it cannot fully see is
    # exactly the failure the exception exists to cause.
    check("a fault is never handed to the model as a tool result, because a model told about one "
          "reasons around it instead of stopping", True)


# --- what the reasoner has no tool for ---------------------------------------
_agent_src = Path(__file__).resolve().parents[1] / "agent"
_tools_src = (_agent_src / "tools.py").read_text()
check("there are exactly six tools, and no seventh reachable by name",
      sorted(m[len("tool_"):] for m in dir(Tools) if m.startswith("tool_"))
      == sorted(prompt.TOOLS))
check("no tool predicts a latency, under that name or another",
      not any(w in _tools_src.lower()
              for w in ("predict", "expected_itl", "estimate_itl", "surrogate")))
check("the one-shot brief and the policy that wrapped it are gone, superseded rather than left "
      "inert beside their replacement",
      not (_agent_src / "brief.py").exists() and not (_agent_src / "llmpolicy.py").exists())
check("the measurement path is the loop's own, so there is not a second implementation of "
      "deploy-gate-record for the reasoner-driven run",
      "self.agent.execute(proposal" in _tools_src
      and "self.execute(proposal" in (_agent_src / "loop.py").read_text())

# What a policy claims about its own choice must not be able to overwrite what the agent measured.
_shadow = SearchAgent(FLEET, ObservationStore(tempfile.gettempdir(), AGENT_KEY),
                      FakeDeployer(), FakeHarness(fake_record))._provenance(
    Proposal(LLM_CANDS[0], "lbl", "because", provenance={"chosen_by": "the reasoner",
                                                         "policy_kind": "model-packer-agent"}),
    LLM_CANDS[0])
check("a policy's provenance cannot shadow the agent's own facts about the run",
      _shadow["chosen_by"] == "agent" and _shadow["policy_kind"] == "model-packer-agent")

# --- the instrument, and the three places its definition has to agree --------
#
# The dcgm-exporter is configured by a Helm chart, scraped by a container that ships
# separately, and named by a URL this package renders before either exists. Nothing in the
# type system connects those, and every failure in the seam is SILENT: `scrape_dcgm` filters
# by field name, so a field the harness asks for and the exporter does not publish is just
# absent from every record's per_gpu block. That is what had happened, undetected, to
# DCGM_FI_DEV_GPU_UTIL and DCGM_FI_DEV_MEM_COPY_UTIL. The checks below are the reason it
# cannot happen twice, and they are text-level on purpose -- Helm templates have no other
# test in this repo, and the properties asserted here are exactly the ones whose violation
# leaves no trace in a record.
_chart = Path(__file__).resolve().parents[3] / "charts/machine-learning/model-serving/mps-daemon"
_chart_values = yaml.safe_load((_chart / "values.yaml").read_text())
_published = tuple(line.split(",")[0].strip()
                   for line in _chart_values["dcgmExporter"]["customMetrics"].splitlines()
                   if line.strip().startswith("DCGM_FI_"))
_scrape_src = ast.parse((Path(__file__).resolve().parents[3]
                         / "containers/colocation-harness/harness/scrape.py").read_text())
_read = next(ast.literal_eval(n.value) for n in _scrape_src.body
             if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "DCGM_FIELDS")
check("the exporter publishes exactly the DCGM fields the harness reads, both ways",
      set(_published) == set(_read),
      f"published-only {sorted(set(_published) - set(_read))}, "
      f"read-only {sorted(set(_read) - set(_published))}")
check("  and neither list has a duplicate, which would read as agreement while changing the count",
      len(_published) == len(set(_published)) == len(_read) == len(set(_read)) == 17)

def _directives(path: Path) -> str:
    """A template with its commentary removed.

    Necessary, not fastidious: these templates carry more prose than YAML, and every
    absence this section asserts -- no DaemonSet, no hostPath, no fixed resource name --
    is discussed at length in a comment somewhere. A negative check against the raw text
    would pass or fail on what the file SAYS rather than on what it renders. Which is the
    same class of mistake as the drift these checks exist to catch.
    """
    return "\n".join(ln for ln in path.read_text().splitlines()
                     if not ln.strip().startswith("#"))


_helpers = _directives(_chart / "templates/_helpers.tpl")
_exporter_tpl = _directives(_chart / "templates/dcgm-exporter.yaml")
check("the scrape URL this package derives is the name the chart actually gives the Service",
      'printf "%s-dcgm-exporter" .Release.Name' in _helpers
      and exporter_service_url("run-7", "ns", 9400)
      == "http://run-7-dcgm-exporter.ns.svc.cluster.local:9400/metrics")
check("  and it is derived from the release name, so a per-run release moves the scrape target "
      "with it rather than leaving it pointed at the one release installed by hand",
      render.DEFAULT_EXPORTER_URL == exporter_service_url()
      and exporter_service_url("run-7") != exporter_service_url("run-8"))
check("every exporter resource is release-prefixed, which is what upstream's chart does not do "
      "and what makes two runs installable in one namespace",
      all(t in _exporter_tpl for t in ('include "mps-daemon.exporterName"',
                                       'include "mps-daemon.exporterMetricsConfigMap"'))
      and 'printf "%s-dcgm-metrics" .Release.Name' in _helpers
      and "exporter-metrics-config-map" not in _exporter_tpl)
check("the exporter is one replica pinned to the daemon's node by REQUIRED pod affinity, so the "
      "Service has exactly one endpoint and it is the captured node's",
      "kind: Deployment" in _exporter_tpl and "replicas: 1" in _exporter_tpl
      and "kind: DaemonSet" not in _exporter_tpl
      and "requiredDuringSchedulingIgnoredDuringExecution" in _exporter_tpl
      and "topologyKey: kubernetes.io/hostname" in _exporter_tpl
      and 'include "mps-daemon.selectorLabels"' in _exporter_tpl)
check("  and the exporter pod does not carry the daemon's own name label, which would let the "
      "daemon Deployment adopt it and scale it away",
      "app.kubernetes.io/name: dcgm-exporter" in _helpers)
check("  and no hostPath is mounted at all: --kubernetes=false makes the kubelet pod-resources "
      "socket's answer wrong rather than merely unused, so the mount is dropped with it",
      "--kubernetes=false" in yaml.dump(_chart_values["dcgmExporter"]["arguments"])
      and "hostPath" not in _exporter_tpl)
check("  and the instrument cannot preempt the workload it measures",
      _chart_values["dcgmExporter"]["priorityClassName"] == ""
      and "{{- with .Values.dcgmExporter.priorityClassName }}" in _exporter_tpl
      and "system-node-critical" not in _exporter_tpl)

_daemon_tpl = _directives(_chart / "templates/deployment.yaml")
check("one daemon per node is stated as anti-affinity, not inferred from gpuRequest happening to "
      "equal the node's GPU count",
      _chart_values["antiAffinity"]["enabled"] is True
      and "podAntiAffinity" in _daemon_tpl
      and "app.kubernetes.io/instance" not in
          _daemon_tpl.split("podAntiAffinity")[1].split("tolerations")[0],
      "matched on the chart name alone, so it excludes OTHER releases' daemons")
check("the runtime UUID ConfigMap is owned by the pod that measured it, so helm uninstall cannot "
      "leave a stale GPU-UUID map for the next run to read",
      "ownerReferences" in _daemon_tpl and "fieldPath: metadata.uid" in _daemon_tpl
      and 'include "mps-daemon.uuidConfigMapBase"' in _daemon_tpl)

# --- retrieval: prior runs, never a prediction (impl §10) --------------------
# The whole package is a query over the store, so what has to be tested is what it returns and --
# more importantly -- what it refuses to return. Every case here runs against the same two records
# the reasoner section uses: SYM and ASYM, both co-residency degree 1 for r2 and t2, measured
# 5.760 ms and 10.877 ms apart.
GPU_OF = {u: g for g, u in zip(INV.gpu_ids, INV.uuids)}
TARGETS = {"m1": TenantSlo(itl_mean_ms=8.0), "r1": TenantSlo(itl_mean_ms=8.0),
           "r2": TenantSlo(itl_mean_ms=8.0), "t2": None}

check("a record's UUIDs resolve to the ordinals a candidate is written in",
      placement_of(LLM_OBS[0], GPU_OF) == {n: tuple(g) for n, g in SYM.items()},
      str(placement_of(LLM_OBS[0], GPU_OF)))
_foreign = json.loads(json.dumps(_base))
_foreign["run_id"] = "elsewhere"
for _t in _foreign["configuration"]["tenants"]:
    _t["gpu_uuids"] = ["GPU-somewhere-else"]
_foreign_obs = Observation.parse(_foreign, "synthetic:elsewhere")
# None rather than a partial map. A half-resolved placement would claim a structural relation to
# the candidate that was never checked, which is worse than admitting the record is unplaceable.
check("  and a record from another node resolves to nothing, not to the part that matched",
      placement_of(_foreign_obs, GPU_OF) is None)
check("  while with no map at all the UUIDs stand as identity, so two records still compare",
      placement_of(LLM_OBS[0])["r2"] == (INV.uuids[2], INV.uuids[3]))

# **The reason this package exists, and it is stronger than "a count is too coarse".** These two
# placements measured 1.89x apart. Their co-residency degrees are identical AND so are their
# co-resident pair sets: both are {(r2,t2)}. They differ only in that one shares both of r2's GPUs
# with t2 and the other shares one. No summary of the structure separates them -- so retrieval
# returns the assignment itself, and any lookup key built from a summary would average the two.
_SYMP = {n: tuple(g) for n, g in SYM.items()}
_ASYMP = {n: tuple(g) for n, g in ASYM.items()}
check("no summary of structure separates the two placements: same degree AND same pair set",
      coresidency(_SYMP) == coresidency(_ASYMP)
      and coresident_pairs(_SYMP) == coresident_pairs(_ASYMP)
      and _SYMP != _ASYMP,
      f"degree {coresidency(_SYMP)}, pairs {sorted(coresident_pairs(_SYMP))}, "
      f"and yet 5.760 ms vs 10.877 ms")

_ev = recall(LLM_OBS, targets=TARGETS, assignment={n: tuple(g) for n, g in SYM.items()},
             gpu_of=GPU_OF, key_hash=LLM_OBS[0].validity_key_hash)
check("recall returns the runs themselves, both of them, with no number computed over them",
      _ev.has_evidence and [r.run_id for r in _ev.runs]
      == ["search-001-sym", "search-002-asym"],
      str([r.run_id for r in _ev.runs]))
check("  and each run carries every tenant's own offered rate, shape and measured ITL",
      _ev.runs[1].tenants["t2"].itl_mean_ms == 10.877
      and _ev.runs[1].tenants["t2"].rate_hz == 2.0
      and _ev.runs[1].tenants["t2"].tp == 2,
      _ev.runs[1].tenants["t2"].line)
check("  and says which run IS the candidate, since identity is a fact and not a similarity",
      _ev.runs[0].relation.same_placement and not _ev.runs[1].relation.same_placement)
check("  and states how the other one differs at the level where it actually differs: per tenant, "
      "with both GPU sets, since the pair set is the same in both",
      _ev.runs[1].relation.alike == ("m1", "r1", "r2")
      and _ev.runs[1].relation.differs == (("t2", (3, 4), (2, 3)),)
      and ("r2", "t2") in _ev.runs[1].relation.shared_pairs,
      _ev.runs[1].relation.text)
_lines = "\n".join(_ev.lines())
check("  and both disagreeing figures appear, since a mean of 5.760 and 10.877 describes neither",
      "5.760" in _lines and "10.877" in _lines and "8.3" not in _lines)
# The judgement retrieval will not make. It prints the two numbers and the measured resolution and
# stops there: deciding whether the difference matters for this candidate is the delegated question.
check("  and nothing in the output labels a difference significant, real, or expected",
      not any(w in _lines.lower() for w in ("significant", "expected itl", "predicted",
                                            "likely to", "estimate")))

# Judged at read time against the targets in force NOW, which is what makes a target an input
# (design §2) rather than something frozen into a record when it was written.
_met = targets_met(LLM_OBS[1], TARGETS)
check("a run is judged against the targets in force now, per tenant",
      _met == {"m1": True, "r1": True, "r2": False, "t2": None}, str(_met))
check("  and a tenant that stated no target gets no verdict, not a passing one",
      _met["t2"] is None and "no target stated" in _ev.runs[1].tenants["t2"].met_detail)
check("  and tightening the target flips the verdict on the same record, with nothing rewritten",
      targets_met(LLM_OBS[0], dict(TARGETS, r2=TenantSlo(itl_mean_ms=4.0)))["r2"] is False
      and targets_met(LLM_OBS[0], TARGETS)["r2"] is True)
_sup = json.loads(json.dumps(_base))
_sup["run_id"] = "sup"
_sup["node"] = INV.node_name
for _t in _sup["configuration"]["tenants"]:
    _t["gpu_uuids"] = [INV.uuids[g] for g in SYM[_t["name"]]]
for _t in _sup["per_tenant"].values():
    _t.pop("itl_mean_ms", None)
    _t["latency_suppressed"] = "delivery gate failed"
_sup["verdict"] = VERDICT_DELIVERY
_sup_obs = Observation.parse(_sup, "synthetic:sup")
# The failure this prevents is specific: a zero flowing into the comparison, and an
# under-delivering run certifying itself as the fastest thing in the store.
check("a suppressed latency yields no verdict and never a zero that would pass every target",
      targets_met(_sup_obs, TARGETS) == {"m1": None, "r1": None, "r2": None, "t2": None}
      and "suppressed" in "\n".join(target_lines(_sup_obs, TARGETS)))

# The same record read the other way round: not "did it pass" but "by how much did it fail". A miss
# is the most informative thing a search finds, and "missed r2" throws away the part of it somebody
# can act on -- a target is an input (design §2), so what a denser placement would cost in
# requirements is a number the person who wrote the requirement should see.
_rx = required_relaxation(LLM_OBS[1], TARGETS)
check("a record that missed reports how far every target would have to move, not just that it missed",
      _rx["factor"] == round(10.194 / 8.0, 6) and set(_rx["per_tenant"]) == {"r2"}
      and _rx["per_tenant"]["r2"]["over_by"] == round(10.194 - 8.0, 6)
      and _rx["per_tenant"]["r2"]["metric"] == "ITL mean", json.dumps(_rx))
check("  and a record that met every target asks for nothing, which is 1.0 and not a small excess",
      required_relaxation(LLM_OBS[0], TARGETS)["factor"] == 1.0
      and required_relaxation(LLM_OBS[0], TARGETS)["per_tenant"] == {})
# Two ceilings, one tenant. Reporting the slacker of the two would understate what relaxing takes,
# and the whole point of the number is that it can be handed to somebody as a decision.
check("  and a tenant over on both ceilings is reported at the binding one, not the kinder one",
      required_relaxation(
          LLM_OBS[1], dict(TARGETS, r2=TenantSlo(itl_mean_ms=8.0, ttft_p95_ms=5.0))
      )["per_tenant"]["r2"]["metric"] == "TTFT p95")
# The trap this closes. `blocked` is not a pass and not a miss: no ceiling exists to move, or the
# measurement the ceiling would judge was never taken. A factor that quietly skipped those tenants
# would read as "2.5% away" when the truth is "unmeasured", and that is the direction that flatters
# co-location -- which is why the delivery gate itself is never relaxed (impl §6.3).
check("  and a tenant no relaxation can reach is named as blocked rather than folded into the factor",
      required_relaxation(_sup_obs, TARGETS)["blocked"] == ("m1", "r1", "r2", "t2")
      and required_relaxation(_sup_obs, TARGETS)["factor"] == 1.0
      and _rx["blocked"] == ("t2",))

# An infeasible record is evidence -- about deployment, not about latency. Keeping it and marking it
# is what stops a placement that already failed to deploy being proposed again (impl §8); mixing it
# in with the measurements would breach the rule that retrieval never decides feasibility.
_ev_inf = recall(LLM_OBS + [_sup_obs], targets=TARGETS,
                 assignment={n: tuple(g) for n, g in ASYM.items()}, gpu_of=GPU_OF,
                 key_hash=LLM_OBS[0].validity_key_hash)
check("an infeasible run comes back marked as evidence about feasibility, not about latency",
      len(_ev_inf.runs) == 3 and len(_ev_inf.latency_runs) == 2
      and "not about latency" in "\n".join(_ev_inf.lines()))

# Nothing is dropped in silence. A record the reasoner never sees is a record it believes does not
# exist, and then "no evidence" means two different things.
_other_key = json.loads(json.dumps(_base))
_other_key["run_id"] = "other-key"
_other_key["validity_key_hash"] = "a-different-key"
_phase = json.loads(json.dumps(_base))
_phase["run_id"] = "phase-locked"
_phase["node"] = INV.node_name
_phase["sampling"]["subsamples_per_sample_mean"] = 1
_ev_x = recall(LLM_OBS + [Observation.parse(_other_key, "x"), Observation.parse(_phase, "y")],
               targets=TARGETS, assignment={n: tuple(g) for n, g in SYM.items()},
               gpu_of=GPU_OF, key_hash=LLM_OBS[0].validity_key_hash)
check("every record set aside is reported, by reason and by run id",
      len(_ev_x.runs) == 2 and _ev_x.considered == 4
      and _ev_x.excluded[EXCLUDED_KEY] == ("other-key",)
      and any("phase-locked" in ids for ids in _ev_x.excluded.values()),
      str({k[:28]: v for k, v in _ev_x.excluded.items()}))
check("  and a query about a tenant no record names says NO EVIDENCE in those words",
      not recall(LLM_OBS, targets=TARGETS, tenants=("ghost",)).has_evidence
      and "NO EVIDENCE" in "\n".join(recall(LLM_OBS, targets=TARGETS,
                                            tenants=("ghost",)).lines()))
# An empty result must not read as agreement. This is the answer a lookup key cannot give: it
# returns a confident number for a neighbourhood the store has never been near.
check("  and an empty store does too, rather than returning an empty list that reads as assent",
      "should be labelled as one" in "\n".join(recall([], targets=TARGETS).lines()))

# impl §13: a cap on how many runs come back would be the similarity rule §10 refuses to write,
# only cruder -- a truncation by recency or count instead of by relevance. So there is none, and
# breadth is one line per run while depth is `read_run` on the ids worth opening.
_many = [_obs_on(f"bulk-{i:03d}", SYM, {"r2": 5.0 + i / 100}) for i in range(24)]
_ev_many = recall(_many, targets=TARGETS, assignment={n: tuple(g) for n, g in SYM.items()},
                  gpu_of=GPU_OF, key_hash=_many[0].validity_key_hash)
check("nothing truncates the result, because a cap is the similarity rule at its crudest",
      len(_ev_many.runs) == 24, f"{len(_ev_many.runs)} runs returned")
check("  and breadth is one line per tenant plus the run's own qualifiers, not the whole record",
      len(_ev_many.runs[0].summary_lines)
      == 1 + len(_ev_many.runs[0].tenants) + len(_ev_many.runs[0].qualifiers)
      and "delivered" not in "\n".join(_ev_many.runs[0].summary_lines),
      f"{len(_ev_many.runs[0].summary_lines)} lines per run")
check("  and the order is by run id, which is arbitrary on purpose: any other order is a ranking",
      [r.run_id for r in _ev_many.runs] == sorted(r.run_id for r in _ev_many.runs))
check("  and repeats of one placement are grouped and shown with their spread, not their mean",
      len(_ev_many.repeats) == 1 and len(_ev_many.repeats[0]) == 24
      and "spread 0.230 ms" in "\n".join(_ev_many.lines()))
_ev_cal = recall(_many[:2], targets=TARGETS, assignment={n: tuple(g) for n, g in SYM.items()},
                 gpu_of=GPU_OF, key_hash=_many[0].validity_key_hash, calibration=CAL)
check("  and the calibrated half-width is printed beside the spread as a fact, applied to nothing",
      "0.2000 ms" in "\n".join(_ev_cal.lines())
      and "yours to decide" in "\n".join(_ev_cal.lines()))

# `read_run` is the depth half. It is what makes the absence of a cap affordable: the reasoner
# spends its own context on the runs it chose to open.
_rr = "\n".join(read_run(LLM_OBS[1], TARGETS, GPU_OF))
check("read_run gives one record in full: raw numbers, then margins, then the signals",
      all(w in _rr for w in ("ITL mean 10.8770", "within-run", "in flight",
                             "delivered", "targets in force now", "impl §6.5"))
      and _rr.index("ITL mean") < _rr.index("targets in force now") < _rr.index("impl §6.5"))
check("  and a suppressed record prints as suppressed there too, with no 0.000 anywhere",
      "SUPPRESSED" in "\n".join(read_run(_sup_obs, TARGETS, GPU_OF))
      and "0.0000 ms" not in "\n".join(read_run(_sup_obs, TARGETS, GPU_OF)))
# A live search died here: a record with an ITL mean but no within-run half-width beside it (g-reason
# at 7 requests), which `read_run` formatted as if it were there, and the group's search with it.
_no_ci = copy.deepcopy(LLM_OBS[1])
_no_ci_t = sorted(_no_ci.per_tenant)[0]
_no_ci.per_tenant[_no_ci_t] = dataclasses.replace(_no_ci.per_tenant[_no_ci_t],
                                                  _itl_ci_halfwidth_ms=None)
check("  and a record missing a figure beside its ITL mean is read, with that figure not recorded",
      "not recorded within-run" in "\n".join(read_run(_no_ci, TARGETS, GPU_OF)))
# Every record names two nodes, and this view has to label both. It printed a bare `node=` -- the
# harness Job's, which is a cpu-compute node -- and a live search read it as the machine under test:
# it hedged a result it had reproduced twice on the grounds that "all four runs executed on
# <driver node>, not the <leased node> named in the problem". They are different machines by design,
# and the one the GPUs were on is in provenance, written by the agent from its lease.
_two = json.loads(json.dumps(_base))
_two["node"] = "ip-cpu-driver"
_two["provenance"] = {**(_two.get("provenance") or {}), "gpu_node": "ip-leased-g6e"}
_rr2 = "\n".join(read_run(Observation.parse(_two, "synthetic:two-nodes"), TARGETS, GPU_OF))
check("  and it names the GPU node under test and the driver's node, each said to be which",
      "gpu node under test=ip-leased-g6e" in _rr2 and "ip-cpu-driver" in _rr2
      and "not the GPUs'" in _rr2 and "\n  node=" not in _rr2,
      _rr2.splitlines()[1])
check("    while a record written before that key existed says not recorded, rather than "
      "promoting the driver's node into its place",
      "gpu node under test=not recorded" in _rr
      and f"load driver ran on={INV.node_name}" in _rr, _rr.splitlines()[1])

# The layering, asserted rather than trusted. `inspect` imports `agent.mirror`, and the agent
# imports retrieval, so a single import here would close a cycle -- and it would also cost the
# property that a record can be read and judged with no OR-Tools and no cluster.
_rsrc = "\n".join(p.read_text() for p in
                  sorted((Path(retrieval.__file__).parent).glob("*.py")))
check("retrieval imports nothing from the agent, the solver's CP-SAT model, or inspect",
      not any(w in _rsrc for w in ("from ..agent", "from ..inspect", "import ortools",
                                   "from ..solver.cpsat")),
      "store/ and solver/topology.py only")
check("  and there is no predictor hiding in it under another name",
      not any(hasattr(retrieval, n) for n in ("predict", "predict_one", "expected_itl",
                                              "estimate", "model", "OccupancyModel")))

# ---------------------------------------------------------------------------------------------
# The engine seam, agent side (impl §13). Written against a throwaway engine rather than the
# shipped one, because what is under test is the seam: every check here is one a second adapter
# has to keep passing.
# ---------------------------------------------------------------------------------------------
from colocation import engines as _engines                                     # noqa: E402
from colocation.agent import deploy as agent_deploy                            # noqa: E402


def _refuses(fn) -> bool:
    """True if calling `fn` raises. For the checks whose subject is a refusal, not a value."""
    try:
        fn()
    except Exception:                                                          # noqa: BLE001
        return True
    return False


check("the default engine is vLLM's OpenAI server, and the image and chart come off it",
      _engines.DEFAULT_ENGINE == "vllm-openai"
      and DEFAULT_IMAGE == _engines.get("vllm-openai").default_image
      and agent_deploy.DEFAULT_CHART == _engines.get("vllm-openai").chart,
      DEFAULT_IMAGE)
_eng_err = None
try:
    _engines.get("triton-trtllm")
except KeyError as _exc:
    _eng_err = _exc.args[0]
check("  and an unregistered engine is refused with the legal names, not silently defaulted",
      _eng_err is not None and "vllm-openai" in _eng_err and "no generic fallback" in _eng_err)

# A second engine, present only for this test. It differs in the three ways that matter: the values
# key its chart uses, the words it takes a shard count and a memory budget in, and the batch cap it
# would have served at if nobody derived one (impl §14.9 -- the floor under a derived cap belongs to
# the engine, so it cannot be a constant in `spec.py`).
_TEST_ENGINE = _engines.register(_engines.Engine(
    name="test-engine", chart="charts/test/engine", chart_release_prefix="test-engine-",
    default_image="example.invalid/test-engine:1", values_key="testengine",
    default_max_num_seqs=32,
    values=lambda *, tp, util, max_num_seqs, max_num_batched_tokens, max_model_len, dtype,
    extra_args=(): {"shards": int(tp), "mem_fraction": round(float(util), 6),
                    "concurrency": int(max_num_seqs)},
    read_tp=lambda b: int(b["shards"]),
    read_util=lambda b: float(b["mem_fraction"])))

_t_vllm = FLEET.tenants[0]
_t_test = dataclasses.replace(_t_vllm, engine="test-engine")
_util_e = gpu_memory_utilization(40.0, 0.0, CARD_GIB)
_v_vllm = render_tenant_values(_t_vllm, INV, [0], 40.0, _util_e)
_v_test = render_tenant_values(_t_test, INV, [0], 40.0, _util_e)
check("a tenant's engine decides which values key its flags go under, and moves nothing else in "
      "the chart's schema except the name of the stack itself",
      "vllm" in _v_vllm and "testengine" not in _v_vllm
      and "testengine" in _v_test and "vllm" not in _v_test
      and {k: v for k, v in _v_vllm.items() if k not in ("vllm", "engine")}
      == {k: v for k, v in _v_test.items() if k not in ("testengine", "engine")},
      str(sorted(_v_test)))
# The values file has to say which stack it is for, because it is what `deploy.py` holds when it
# picks the chart to install a tenant from -- and picking that from the presence of a block would
# break the moment two stacks spelled their block alike.
check("  and the values say which stack they are for, so the chart can be chosen per tenant",
      _v_vllm["engine"] == "vllm-openai" and _v_test["engine"] == "test-engine")
_pick = agent_deploy.HelmDeployer(namespace="ns")._chart_for
check("  which is what the deployer does with it: one fleet, two engines, two charts",
      _pick("r1", _v_vllm).endswith(_engines.get("vllm-openai").chart)
      and _pick("r2", _v_test).endswith("charts/test/engine"),
      _pick("r1", _v_vllm))
# --chart is for a copy of a chart under development. Against a tenant of another engine it is
# refused, because Helm would not refuse it: an unrecognised values block is ignored, so the tenant
# would come up on the wrong chart's defaults with every number this run solved for discarded.
_override = agent_deploy.HelmDeployer(
    chart_path=str(Path(__file__).resolve().parents[3]
                   / _engines.get("vllm-openai").chart), namespace="ns")
check("  and an explicit --chart is accepted for that engine's tenants and refused for another's",
      _override._chart_for("r1", _v_vllm).endswith("vllm-openai")
      and _refuses(lambda: _override._chart_for("r2", _v_test)))
check("  and the engine's own block is in the engine's vocabulary, not a renamed copy of vLLM's",
      _v_test["testengine"] == {"shards": _t_test.tp, "mem_fraction": round(_util_e, 6),
                                "concurrency": _t_test.max_num_seqs},
      str(_v_test["testengine"]))
check("  and the floor under its derived batch cap is ITS default, not the default engine's",
      _t_test.max_num_seqs == 32 and _t_vllm.max_num_seqs == 64
      and _t_test.slo is None,
      f"{_t_test.max_num_seqs} vs {_t_vllm.max_num_seqs}")
check("  while a tenant that names no engine renders what it rendered before the seam existed",
      _v_vllm["vllm"]["tensor_parallel_size"] == _t_vllm.tp
      and _v_vllm["vllm"]["gpu_memory_utilization"] == round(_util_e, 6)
      and _v_vllm["image"] == DEFAULT_IMAGE)

# Teardown has to mean *departed*, not "the release is gone". `helm uninstall --wait` waits for the
# objects it made, and a Deployment is deleted while the pod it made is still running: measured on a
# leased node, uninstall returned in 0.9 s and the outgoing pod held 43 GiB of weights for another
# half-minute. The next tenant then failed its own verification init container -- free memory short
# of the envelope -- over memory that was already leaving. So the wait is part of destroy.


class _FakeClock:
    """`monotonic` that only moves when something waits, so the poll loop costs no wall clock."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += float(s)


_dep_wait = agent_deploy.HelmDeployer(namespace="ns")
_polls, _still_there = [], ["rtest-m1", "rtest-m1", None]


def _any_pods_stub(names):
    _polls.append(tuple(sorted(names)))
    left = _still_there.pop(0) if _still_there else None
    return {left} & set(names) if left else set()


_dep_wait._any_pods = _any_pods_stub
_real_time = agent_deploy.time
agent_deploy.time = _FakeClock()
try:
    _dep_wait._await_departed(["rtest-m1"])
    check("destroy waits for the outgoing pod to be gone, not just its release",
          len(_polls) == 3 and _polls[0] == ("rtest-m1",), str(_polls))
    # It gives up rather than hanging, because a run that stalls forever on a stuck pod is worse
    # than one that says why the next envelope check read low.
    _still_there[:] = ["rtest-m1"] * 200
    _polls.clear()
    _dep_wait._await_departed(["rtest-m1"], timeout_s=30.0)
    check("  and gives up after its timeout instead of waiting on a pod that will not go",
          0 < len(_polls) <= 8, str(len(_polls)))
    _polls.clear()
    _dep_wait._await_departed([])
    check("  and asks nothing at all when nothing was uninstalled", _polls == [], str(_polls))
finally:
    agent_deploy.time = _real_time
# The Terminating pod is the one being waited for, so this reader must NOT share `_pods`' rule of
# skipping it -- that is the difference between the two and the reason there are two.
_dep_wait2 = agent_deploy.HelmDeployer(namespace="ns")
_pod_json = json.dumps({"items": [
    {"metadata": {"labels": {"app.kubernetes.io/instance": "rtest-m1"},
                  "deletionTimestamp": "2026-09-19T18:34:39Z"}},
    {"metadata": {"labels": {"app.kubernetes.io/instance": "other-run-m1"}}}]})
_real_run = agent_deploy._run
agent_deploy._run = lambda argv, **kw: _pod_json
try:
    check("  and it counts a Terminating pod, which is the whole point, and only this run's",
          _dep_wait2._any_pods(["rtest-m1", "rtest-m2"]) == {"rtest-m1"},
          str(_dep_wait2._any_pods(["rtest-m1", "rtest-m2"])))
finally:
    agent_deploy._run = _real_run

# `check_agreement` re-derives the envelope from what was written into the values, and it is the
# only thing standing between a wrong denominator and a healthy-looking run (§4.2.2). It has to
# survive the engine changing its vocabulary, and it has to refuse an engine disagreement outright.
def _cfg_for(engine_name):
    return {"tenants": [{"name": _t_test.name, "gpu_uuids": [INV.uuid_for(0)],
                         "envelope_gib": 40, "tp": _t_test.tp, "engine": engine_name}]}


def _agree(values, cfg):
    return check_agreement([(_t_test.name, values)], cfg, device_total_gib=CARD_GIB,
                           mps_reserve_gib={_t_test.name: 0.0})


check("check_agreement re-derives the envelope through the engine's own field names",
      _agree(_v_test, _cfg_for("test-engine")) is None
      and _agree(_v_vllm, _cfg_for("vllm-openai")) is None)
check("  and refuses when the two artifacts name different engines, rather than reading the "
      "wrong block or no block",
      _refuses(lambda: _agree(_v_test, _cfg_for("vllm-openai")))
      and _refuses(lambda: _agree(_v_vllm, _cfg_for("test-engine"))))
_v_bad = copy.deepcopy(_v_test)
_v_bad["testengine"]["mem_fraction"] = round(_util_e * 1.05, 6)
check("  and a wrong memory fraction stated in the new engine's vocabulary is still caught",
      _refuses(lambda: _agree(_v_bad, _cfg_for("test-engine"))))
_v_shards = copy.deepcopy(_v_test)
_v_shards["testengine"]["shards"] = _t_test.tp + 1
check("  and so is a shard count the harness was not told about",
      _refuses(lambda: _agree(_v_shards, _cfg_for("test-engine"))))

# The fleet file's side of the seam.
check("a fleet file's `engine:` is a fleet-level default with a per-tenant override",
      fleetfile._tenant_engine({"engine": "test-engine"}, {"name": "a"}) == "test-engine"
      and fleetfile._tenant_engine({"engine": "test-engine"},
                                   {"name": "a", "engine": "vllm-openai"}) == "vllm-openai"
      and fleetfile._tenant_engine({}, {"name": "a"}) == "vllm-openai")
check("  and a misspelled engine is refused while the file is read, naming the tenant",
      _refuses(lambda: fleetfile._tenant_engine({}, {"name": "a", "engine": "vlm"})))
check("  and a fleet that names an engine and no image gets THAT engine's image",
      fleetfile._tenant_image({"engine": "test-engine"}, {"name": "a"})
      == "example.invalid/test-engine:1"
      and fleetfile._tenant_image({}, {"name": "a"}) == DEFAULT_IMAGE
      and fleetfile._tenant_image({"engine": "test-engine"},
                                  {"name": "a", "image": "pinned:1"}) == "pinned:1")
check("  and a mixed-engine fleet is NOT refused, because the engine is keyed per tenant",
      fleetfile.fleet_engines({"tenants": [{"name": "a"}, {"name": "b",
                                                           "engine": "test-engine"}]})
      == frozenset({"vllm-openai", "test-engine"}))

# A mixed-engine fleet is necessarily a mixed-IMAGE fleet, because an image belongs to an engine.
# `fleet_image` used to refuse the second image and so refused the arrangement; it now names them
# all. The property that matters is the one on the second line: a single-image fleet still produces
# the identical string, so no row already in the store is re-keyed by this.
_MIXED = {"tenants": [{"name": "a"}, {"name": "b", "engine": "test-engine"}]}
check("a fleet may be served from more than one image, and the key names every one of them",
      fleetfile.fleet_image(_MIXED) == ",".join(sorted([DEFAULT_IMAGE,
                                                        "example.invalid/test-engine:1"])),
      fleetfile.fleet_image(_MIXED))
check("  while a fleet on one engine keys byte-identically to what it keyed before",
      fleetfile.fleet_image({"tenants": [{"name": "a"}, {"name": "b"}]}) == DEFAULT_IMAGE)
check("  and each tenant's own image is recoverable from the key, because one engine means one image",
      fleetfile.fleet_engine_images(_MIXED)
      == {"vllm-openai": DEFAULT_IMAGE, "test-engine": "example.invalid/test-engine:1"})
check("  so ONE engine served from TWO images is what is refused, naming the second tenant",
      _refuses(lambda: fleetfile.fleet_image(
          {"tenants": [{"name": "a"}, {"name": "b", "image": "pinned:1"}]}))
      and _refuses(lambda: fleetfile.fleet_engine_images({"tenants": []})))
check("  and a composite never collides with either single image it is made of",
      len({fleetfile.fleet_image(_MIXED), DEFAULT_IMAGE, "example.invalid/test-engine:1"}) == 3
      and "," not in DEFAULT_IMAGE)

# `serves_images` is a capability, and the fleet file is where it is enforced -- offline, before a
# node is leased. The harness-side adapter refuses too, as the backstop, but that refusal arrives
# after the weights are on the card.
_TEXT_ONLY = _engines.register(dataclasses.replace(
    _TEST_ENGINE, name="text-only-engine", chart_release_prefix="text-only-engine-",
    serves_images=False))
_IMG_T = {"name": "docqa", "images": {"count": 2, "pixels": 1024},
          "extra_args": ['--limit-mm-per-prompt={"image": 2, "video": 0}']}
_img_err = None
try:
    fleetfile._load_image_spec("docqa", _IMG_T, _IMG_T["extra_args"], "text-only-engine")
except fleetfile.FleetFileError as _exc:
    _img_err = str(_exc)
check("a tenant that sends images is refused on a stack that cannot serve them, while the file is read",
      _img_err is not None and "text-only-engine" in _img_err)
check("  and the refusal lists the stacks that CAN, asked of the registry rather than named in it",
      _img_err is not None and "vllm-openai" in _img_err and "test-engine" in _img_err
      and "text-only-engine" not in _img_err.split("naming one of")[1])
check("  while a text tenant on the same stack loads, so it is the images that are refused and not "
      "the stack",
      not _refuses(lambda: fleetfile._load_image_spec(
          "gate", {"name": "gate"}, [], "text-only-engine"))
      and not _refuses(lambda: fleetfile._load_image_spec(
          "docqa", _IMG_T, _IMG_T["extra_args"], "test-engine")))
check("  and the stack this was written for is the one that declares it, the default engine not",
      _engines.get("vllm-triton-inference-server").serves_images is False
      and _engines.get(_engines.DEFAULT_ENGINE).serves_images is True)

# The third stack, Dynamo + vLLM. What is under test is what it adds to the seam rather than the seam
# itself: a tenant that is two pods, a controller that must not be allowed to roll an update out, and
# an operator the cluster may not have. Each check is one a live run would otherwise find the hard way.
from colocation.engines import dynamo_vllm as _dyn                             # noqa: E402

_DYN = _engines.get("dynamo-vllm")
_t_dyn = dataclasses.replace(_t_vllm, engine="dynamo-vllm")
_v_dyn = render_tenant_values(_t_dyn, INV, [0], 40.0, _util_e)
check("a dynamo-vllm tenant renders the sibling's vLLM flags under its own block, and its own chart",
      _v_dyn["engine"] == "dynamo-vllm" and "vllm" not in _v_dyn
      and _v_dyn["dynamo_vllm"] == _v_vllm["vllm"]
      and _pick("r3", _v_dyn).endswith(_DYN.chart), str(sorted(_v_dyn)))
check("  and refuses the flags that would make a tenant half of a prefill/decode pair",
      all(_refuses(lambda f=f: _dyn._refuse([f])) for f in
          ("--disaggregation-mode=prefill", "--is-prefill-worker", "--is-decode-worker true"))
      and not _refuses(lambda: _dyn._refuse(["--no-enable-prefix-caching"])))
check("  and the two flags the chart sets from the shared schema, so there is one place for each",
      _refuses(lambda: _dyn._refuse(["--model=/fsx/other"]))
      and _refuses(lambda: _dyn._refuse(["--served-model-name", "x"])))
check("  and serves no image tenant, since that path was never measured on this stack",
      _DYN.serves_images is False)
check("  and is two pods, replaced rather than upgraded, and needs its operator; the others none of it",
      _DYN.pods_per_tenant == 2 and _DYN.replace_on_change is True
      and _DYN.requires_crds == ("dynamographdeployments.nvidia.com",) and _DYN.requires_controller
      and all(_engines.get(n).pods_per_tenant == 1 and not _engines.get(n).replace_on_change
              and not _engines.get(n).requires_crds and not _engines.get(n).requires_controller
              for n in ("vllm-openai", "vllm-triton-inference-server")))


# A chart property rather than an adapter one, and read off the template because the failure it
# guards is silent: MPS serves one Unix user at a time, this image runs as UID 1000 and every other
# client on the node as 0, and a worker left on the image's user hangs at attach behind a co-tenant.
_dyn_tpl = (Path(__file__).resolve().parents[3] / _DYN.chart / "templates"
            / "dynamo-graph-deployment.yaml").read_text()
_dyn_worker = _dyn_tpl.split("type: worker", 1)[-1]
check("  and its worker pod runs as root, the one MPS user every stack on the node shares",
      "runAsUser: 0" in _dyn_worker and "runAsUser: 0" not in _dyn_tpl.split("type: worker", 1)[0])


def _named(name, **kw):
    p = pod(**kw)
    p["metadata"]["name"] = name
    return p


_READY_C = [{"type": "Ready", "status": "True"}]
check("a two-pod tenant with one pod Ready and the other not yet created is waiting, not Ready",
      agent_deploy._classify_tenant([_named("fe", conditions=_READY_C)], 2)[0] == NOT_READY_TIMEOUT
      and "1 of 2" in agent_deploy._classify_tenant([_named("fe", conditions=_READY_C)], 2)[1])
check("  and Ready only once both are",
      agent_deploy._classify_tenant([_named("fe", conditions=_READY_C),
                                     _named("wk", conditions=_READY_C)], 2)[0] == READY)
_dyn_v = agent_deploy._classify_tenant(
    [_named("fe", conditions=_READY_C),
     _named("wk", init=[{"name": "verify-mps-limit",
                         "state": {"terminated": {"exitCode": 1, "reason": "Error"}}}])], 2)
check("  and a failed verification on the worker fails the tenant, naming the pod, while the "
      "frontend beside it is Ready",
      _dyn_v[0] == NOT_READY_VERIFY and _dyn_v[1].startswith("wk:"), str(_dyn_v))
check("  and a terminal verdict wins even while the other pod does not exist yet",
      agent_deploy._classify_tenant(
          [_named("wk", containers=[{"name": "main",
                                     "state": {"waiting": {"reason": "CrashLoopBackOff"}}}])],
          2)[0] == NOT_READY_ENGINE)
check("  while a one-pod tenant is judged exactly as before: on its newest pod alone",
      agent_deploy._classify_tenant([_named("new", conditions=_READY_C),
                                     _named("old", phase="Pending")], 1)[0] == READY
      and agent_deploy._classify_tenant([], 1) == _classify(None))


class _ApplyProbe(agent_deploy.HelmDeployer):
    """`apply` with helm stubbed: records what it would uninstall and install."""

    def __init__(self, installed):
        super().__init__(namespace="ns")
        self.installed, self.calls = installed, []

    def _installed_values(self, name):
        return self.installed

    def destroy(self, release_names):
        self.calls.append(("destroy", tuple(release_names)))


def _apply_calls(values, installed):
    probe = _ApplyProbe(installed)
    real = agent_deploy._run
    agent_deploy._run = lambda argv, **kw: probe.calls.append(("helm", argv[1])) or ""
    try:
        probe.apply([("rtest-m1", values)])
    finally:
        agent_deploy._run = real
    return probe.calls


check("a changed dynamo-vllm release is uninstalled, and seen gone, before it is installed again",
      _apply_calls(_v_dyn, {**_v_dyn, "name": "changed"})
      == [("destroy", ("rtest-m1",)), ("helm", "upgrade")])
check("  and an unchanged one is left alone, nulls Helm may not echo back notwithstanding",
      _apply_calls(_v_dyn, _v_dyn) == [("helm", "upgrade")]
      and _apply_calls({**_v_dyn, "unset_field": None}, _v_dyn) == [("helm", "upgrade")])
check("  and a first install replaces nothing",
      _apply_calls(_v_dyn, None) == [("helm", "upgrade")])
check("  while the other stacks are upgraded in place and never asked what is installed",
      _apply_calls(_v_vllm, {**_v_vllm, "name": "changed"}) == [("helm", "upgrade")])

_probe3 = _ApplyProbe({**_v_dyn, "name": "changed"})
_real_run = agent_deploy._run
agent_deploy._run = lambda argv, **kw: _probe3.calls.append(("helm", argv[3])) or ""
try:
    _probe3.apply([("rtest-a", _v_dyn), ("rtest-b", _v_dyn), ("rtest-c", _v_dyn)])
finally:
    agent_deploy._run = _real_run
check("  and several changed at once all go down before any comes up, so none returns onto a GPU a "
      "later one has not left",
      _probe3.calls == [("destroy", ("rtest-a", "rtest-b", "rtest-c")), ("helm", "rtest-a"),
                        ("helm", "rtest-b"), ("helm", "rtest-c")])


def _prereq(responses):
    """check_prerequisites against a cluster that answers `responses[argv-kind]`."""
    seen = []

    def fake(argv, **kw):
        seen.append(argv[2])
        r = responses[argv[2]]
        if isinstance(r, Exception):
            raise r
        return r

    real = agent_deploy._run
    agent_deploy._run = fake
    try:
        agent_deploy.check_prerequisites(["dynamo-vllm", "vllm-openai"])
        return "ok", seen
    except agent_deploy.DeployError as exc:
        return str(exc), seen
    finally:
        agent_deploy._run = real


_ctrl_ready = json.dumps({"items": [pod(conditions=_READY_C)]})
check("the Dynamo operator is checked for before a node is leased, and a Ready one passes",
      _prereq({"crd": "crd/x", "pods": _ctrl_ready}) == ("ok", ["crd", "pods"]))
check("  and missing CRDs are refused with how to install the operator",
      "dynamo_enabled=true" in _prereq({"crd": agent_deploy.DeployError("NotFound"),
                                        "pods": _ctrl_ready})[0])
check("  and CRDs with no Ready controller are refused too, which is the failure nothing else sees",
      "nothing would ever come up" in _prereq(
          {"crd": "crd/x", "pods": json.dumps({"items": [pod(phase="Pending")]})})[0])
_seen_none = []
_real_run = agent_deploy._run
agent_deploy._run = lambda argv, **kw: _seen_none.append(argv) or ""
try:
    agent_deploy.check_prerequisites(["vllm-openai", "vllm-triton-inference-server"])
finally:
    agent_deploy._run = _real_run
check("  while a fleet on the single-server stacks asks the cluster nothing", _seen_none == [])

_ld_real = agent_deploy._run
agent_deploy._run = lambda argv, **kw: json.dumps([
    {"name": "rtest-m1", "chart": "dynamo-vllm-0.1.0"},
    {"name": "rtest-m2", "chart": "vllm-openai-0.1.0"},
    {"name": "rtest-m3", "chart": "dynamo-platform-1.5.0"}])
try:
    _live = agent_deploy.HelmDeployer(namespace="ns", tenant_prefix="rtest").live_tenant_releases()
finally:
    agent_deploy._run = _ld_real
check("a dynamo-vllm release is one of this run's tenants for teardown, and the operator is not",
      _live == ["rtest-m1", "rtest-m2"], str(_live))

# The engine in the key. This decides whether two engines' rows can be told apart in the store,
# and it must NOT re-key the history that already exists.
_sc_test = dataclasses.replace(SERVING, engine="test-engine")
check("the engine is part of a measurement's identity, so two stacks' rows cannot collide",
      SERVING.digest != _sc_test.digest and _sc_test.canonical()["engine"] == "test-engine")
check("  while the default engine is OMITTED from the digest, so no existing row is re-keyed",
      "engine" not in SERVING.canonical()
      and SERVING.digest == ServingConfig(
          max_num_seqs=SERVING.max_num_seqs,
          max_num_batched_tokens=SERVING.max_num_batched_tokens,
          max_model_len=SERVING.max_model_len, dtype=SERVING.dtype,
          extra_args=tuple(SERVING.extra_args)).digest)
check("  and a stored row with no engine field reads back as the only engine there was",
      ServingConfig.from_dict(SERVING.canonical()).engine == "vllm-openai"
      and ServingConfig.from_dict(_sc_test.canonical()).engine == "test-engine")
check("  and a tenant's ServingConfig carries the engine it is served by",
      _t_test.serving.engine == "test-engine"
      and _t_vllm.serving.engine == "vllm-openai")

# The seam is only worth having if nothing outside it still spells one engine's vocabulary. This
# is the check that a third adapter is a new file rather than another search-and-replace.
_ENGINE_BLIND = ("agent/render.py", "agent/deploy.py", "agent/fleetfile.py", "store/serving.py",
                 "store/profile.py", "solver/memory.py")
_SEAM_ROOT = Path(render.__file__).resolve().parents[1]
_SPELLINGS = ('"vllm"', "'vllm'", "vllm-openai", "nvcr.io", "tensor_parallel_size",
              '"gpu_memory_utilization"')
_leaks = {}
for _rel in _ENGINE_BLIND:
    _src = (_SEAM_ROOT / _rel).read_text()
    # Docstrings and comments are excluded, and the distinction is the point of the check rather
    # than a loophole in it: prose saying "this used to be vLLM's `tensor_parallel_size`" is a
    # reference a reader wants, while a dict key spelling it is a coupling a second engine trips on.
    _doc_lines = set()
    for _n in ast.walk(ast.parse(_src)):
        if isinstance(_n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            _d = _n.body[0] if _n.body else None
            if (isinstance(_d, ast.Expr) and isinstance(_d.value, ast.Constant)
                    and isinstance(_d.value.value, str)):
                _doc_lines |= set(range(_d.lineno, (_d.end_lineno or _d.lineno) + 1))
    _bad = [ln.strip() for i, ln in enumerate(_src.splitlines(), start=1)
            if i not in _doc_lines and not ln.lstrip().startswith("#")
            and any(w in ln for w in _SPELLINGS)]
    if _bad:
        _leaks[_rel] = _bad[:2]
check("no module outside engines/ still names one engine's chart key, image or flag spellings",
      not _leaks, str(_leaks)[:400])

# The engine seam's proof (impl §13). `snapshot.py` renders every shipped fleet, every tenant, at
# every legal degree, against a labelled hardware fixture; this compares that to the bytes on disk.
# It is here rather than left as a command someone remembers to run because the failure it catches is
# a silent one: an adapter that still deploys and still measures while asking the engine for a
# different memory fraction or a different shard count than it did before.
from . import snapshot as _snap                                       # noqa: E402

_snap_stored = _snap.SNAPSHOT_PATH.read_text() if _snap.SNAPSHOT_PATH.exists() else None
# Every fleet is loaded with unresolved degrees on purpose (the snapshot pins each degree itself), and
# the loader warns once per fleet about exactly that. Left on, those 86 lines bury the PASS lines --
# and this file's whole failure mode is output nobody can tell was truncated.
logging.getLogger("colocation.agent.fleetfile").setLevel(logging.CRITICAL)
_snap_now = _snap.dumps(_snap.build())
logging.getLogger("colocation.agent.fleetfile").setLevel(logging.NOTSET)
check("the rendered chart values for every shipped fleet match the stored snapshot",
      _snap_stored == _snap_now,
      "identical" if _snap_stored == _snap_now else
      (f"no snapshot at {_snap.SNAPSHOT_PATH}; run "
       "`python3 -m colocation.tests.snapshot --write` and read the diff" if _snap_stored is None
       else f"{_snap.SNAPSHOT_PATH.name} differs; run "
            "`python3 -m colocation.tests.snapshot` to see the diff"))

# Both of the below guard against the check above passing *vacuously*. A snapshot of nothing compares
# equal to a snapshot of nothing, and the refactor this file exists to police is one that could
# plausibly rename or remove the very sub-dict being held still.
_snap_now_d = json.loads(_snap_now)
_snap_fleets = _snap_now_d["fleets"]
check("  covering every fleet file shipped in fleets/, each snapshotted or named with what is missing",
      set(_snap_fleets) | set(_snap_now_d["not_snapshotted"])
      == {p.relative_to(_snap.FLEETS_DIR).as_posix() for p in _snap.FLEETS_DIR.rglob("*.yaml")}
      and len(_snap_fleets) >= 3,
      f"{len(_snap_fleets)} fleets, {len(_snap_now_d['not_snapshotted'])} not snapshotted")
_snap_degrees, _snap_engine, _snap_keys = set(), set(), set()
for _f in _snap_fleets.values():
    for _t in _f["tenants"].values():
        _snap_degrees |= set(_t)
        for _deg, _vals in _t.items():
            # Read through the engine the values themselves name, rather than through one engine's
            # block spelling. A fleet served by a second stack has no `vllm:` key at all, so a
            # hard-coded one turns this from a check into a crash the moment the seam is used -- and
            # asking the registry is also what makes the check cover both stacks instead of one.
            _e = _engines.get(_vals["engine"])
            _block = _vals[_e.values_key]
            _snap_keys.add(_e.values_key)
            _snap_engine.add(_e.read_tp(_block) == int(_deg)
                             and 0.0 < _e.read_util(_block) < 1.0
                             and len(_vals["placement"]["gpu_uuids"]) == int(_deg)
                             and len(_vals["placement"]["gpu_indices"]) == int(_deg))
check("  and holding the engine-coupled values still at each of the node's legal degrees",
      _snap_degrees == {"1", "2", "4", "8"} and _snap_engine == {True},
      f"degrees {sorted(_snap_degrees, key=int)}")
check("    for more than one engine, each read through its own block",
      len(_snap_keys) > 1
      and _snap_keys <= {_engines.get(n).values_key for n in _engines.names()},
      f"blocks {sorted(_snap_keys)}")
check("  with the run config in the same artifact, so the two renderings cannot drift apart",
      all(f["run_config"]["tenants"] and f["run_config"]["validity_key_hash"]
          == f["validity_key_hash"] for f in _snap_fleets.values()))

print()
if fails:
    print(f"{len(fails)} FAILED of {len(ran)} run: {fails}")
    sys.exit(1)
# --- Phase C: the driver's own schedule gates citability, not just too-narrow -----------------
# `too_narrow_reason` has always refused to call a degree insufficient when the driver fell behind
# its arrival schedule. The same fact was missing from `unusable_reason`, in the direction that
# matters more: late requests are still launched, so `delivered_fraction` shrinks numerator and
# denominator together and stays near 1.0, and the row is cited as this model meeting its target at
# a rate it was never actually driven at -- which sets minimal TP and therefore the baseline.
def _cite(**over):
    """`unusable_reason` for an otherwise-perfect row with `over` applied to its evidence."""
    ev = {**_SAT_FULL, **over, "requests_completed": 300,
          "in_flight_first_quarter_mean": 4.0, "in_flight_last_quarter_mean": 4.0}
    return SoloProfile(model_id="m", tp=1, isl_tokens=1, osl_tokens=1, rate_hz=8.0,
                       envelope_gib=1.0, serving=_prof_tenant.serving,
                       itl_mean_ms=22.0, delivered_fraction=1.0, delivery_passed=True,
                       notes={"verdict": VERDICT_MEASURED, "saturation": ev}).unusable_reason


check("a row whose driver kept its schedule may be cited about latency", _cite() is None,
      str(_cite()))
check("  but one whose driver fell behind may not, however good the latency looks",
      _cite(schedule_lag_max_ms=9_000.0) is not None
      and _cite(schedule_lag_max_ms=9_000.0)[0] == UNUSABLE_SCHEDULE_LAG,
      str(_cite(schedule_lag_max_ms=9_000.0)))
check("  and the refusal names the rate the row would otherwise have been cited at",
      "8.0 Hz" in _cite(schedule_lag_max_ms=9_000.0)[1],
      _cite(schedule_lag_max_ms=9_000.0)[1])
check("  while a lag inside the limit is not held against it",
      _cite(schedule_lag_max_ms=MAX_SCHEDULE_LAG_MS - 1.0) is None)
check("  and an absent figure is no claim either way, because notes.saturation postdates the "
      "earliest rows",
      _cite(schedule_lag_max_ms=None) is None
      and SoloProfile(model_id="m", tp=1, isl_tokens=1, osl_tokens=1, rate_hz=8.0,
                      envelope_gib=1.0, serving=_prof_tenant.serving, itl_mean_ms=22.0,
                      delivery_passed=True,
                      notes={"verdict": VERDICT_MEASURED}).unusable_reason is None)
check("  and the same lag still blocks the other question too, so neither direction is loosened",
      _narrow(schedule_lag_max_ms=9_000.0) is None)
# The delivery gate cannot substitute for it, which is precisely why the guard is needed: a driver
# that ran late launched fewer requests, so the fraction it delivered stays high.
check("  and delivered_fraction does not catch it, since the shortfall is in what was offered",
      _cite(schedule_lag_max_ms=9_000.0)[0] != UNUSABLE_DELIVERY)
# Classification, which is a separate decision from disqualification and the one §0 cares about: an
# unusable row makes its degree UNCHECKED, so the answer above it is an upper bound rather than a
# measured minimum. A code in neither set would keep the degree out of `missing` and report
# NONE_MEETS about a degree nobody validly drove.
check("  and the degree it disqualifies is unchecked, never checked, so the answer stays an upper "
      "bound", UNUSABLE_SCHEDULE_LAG in _UNCHECKED and UNUSABLE_SCHEDULE_LAG not in _CHECKED)
check("  which holds for every ground that disqualifies a row, not just this one",
      {UNUSABLE_NO_LATENCY, UNUSABLE_DELIVERY, UNUSABLE_VERDICT, UNUSABLE_SAMPLING,
       UNUSABLE_SCHEDULE_LAG} <= _UNCHECKED,
      str(sorted({UNUSABLE_NO_LATENCY, UNUSABLE_DELIVERY, UNUSABLE_VERDICT, UNUSABLE_SAMPLING,
                  UNUSABLE_SCHEDULE_LAG} - _UNCHECKED)))

# --- the canonical export (agent/export.py) ----------------------------------
# What leaves this pipeline for somebody else to deploy from. Two properties carry the whole of it:
# the document says the same thing when the ordinals move, and it never states a memory requirement
# larger than the tenant's own -- a solo row's envelope is the whole card, and exporting that as a
# requirement would ask a downstream scheduler for every device the fleet has.

_ex_dir = Path(tempfile.mkdtemp(prefix="colocation-export-"))
atexit.register(shutil.rmtree, _ex_dir, ignore_errors=True)
_ex_fleet_file = _ex_dir / "fleet.yaml"
_ex_fleet_file.write_text("tenants: []  # only its bytes are read, for the hash\n")
_ex_hardware = SimpleNamespace(instance_type="g6e.48xlarge", gpu_product="l40s", gpu_count=8)
_ex_base = fleet_baseline(NODE, FLEET.requests, envelope_step_mib=AGENT_ENVELOPE_STEP_MIB)
_ex_resolutions = {t.name: SimpleNamespace(tp=t.tp, certain=True,
                                           evidence=SimpleNamespace(run_id=f"profile-{t.name}",
                                                                    itl_mean_ms=20.0))
                   for t in FLEET.tenants}


def _ex_build(summary=None, **kw):
    return exportmod.build(fleet=FLEET, base=_ex_base, fleet_path=_ex_fleet_file,
                           validity_key_hash="deadbeefcafe", profile_key_hash="1234abcd5678",
                           hardware=_ex_hardware, summary=summary,
                           summary_path=None if summary is None else "searches/tok.json",
                           resolutions=_ex_resolutions, **kw)


# One placement, in two different sets of ordinals. Same slots, same document: this is what "free of
# GUIDs" has to mean to be worth anything, because a node reclaimed overnight comes back with both
# its UUIDs and its enumeration order changed.
_ex_here = exportmod.slots_for({"m1": [2], "r1": [2], "r2": [0, 1], "t2": [3, 4]})
_ex_there = exportmod.slots_for({"m1": [7], "r1": [7], "r2": [5, 3], "t2": [1, 0]})
check("a placement exported as slots is the same document whichever ordinals it ran on",
      _ex_here == _ex_there, f"{_ex_here} != {_ex_there}")
check("  and a shared device is one slot with both its residents on it",
      {"tenant": "m1", "shard": 0} in _ex_here[0]["residents"]
      and {"tenant": "r1", "shard": 0} in _ex_here[0]["residents"], str(_ex_here[0]))
check("  and a tp2 tenant occupies two slots, one shard each, never two shards on one device",
      sum(1 for s in _ex_here for r in s["residents"] if r["tenant"] == "r2") == 2
      and all(len([r for r in s["residents"] if r["tenant"] == "r2"]) <= 1 for s in _ex_here))
check("  and its shards are numbered, so four slots hold four shards and not one shard four times",
      sorted(r["shard"] for s in _ex_here for r in s["residents"] if r["tenant"] in ("r2", "t2"))
      == [0, 0, 1, 1])
check("  and the slot count is the device count, which is what the result is scored in",
      len(_ex_here) == 5 and len(exportmod.disjoint_slots({"a": 3, "b": 1})) == 4,
      f"{len(_ex_here)} slot(s)")
check("  while the baseline's slots hold one resident each, because §0 forbids overlap",
      all(len(s["residents"]) == 1 for s in exportmod.disjoint_slots({"a": 3, "b": 1})))

# The emitter. Hand-rolled for the comments, so what it writes has to be checked against the reader
# that will parse it -- a document that only this file can read is the JSON it replaced.
_ex_awkward = {"plain": "ITL mean", "yes": "yes", "nullish": "null", "numeric": "1.0",
               "stamp": "2026-09-20T13:55:53+00:00", "colon": "a: b", "hash": "# x", "empty": "",
               "dash": "-5", "nested": {"list": [1, 2.5, None, True],
                                        "flow": exportmod.Flow({"a": 1, "b": "two"})},
               "slots": exportmod.slots_for({"m1": [0], "r1": [0]})}
check("the emitter's output parses back to the document it was given, awkward scalars and all",
      yaml.safe_load(exportmod.dump(_ex_awkward, comments={})) == _ex_awkward,
      str(yaml.safe_load(exportmod.dump(_ex_awkward, comments={}))))
check("  and a timestamp comes back as a string rather than as a date, which JSON has no type for",
      isinstance(yaml.safe_load(exportmod.dump(_ex_awkward, comments={}))["stamp"], str))
check("  and the comments the exporter writes survive in it, since they are why this is not JSON",
      exportmod.dump(_ex_build()).lstrip().startswith("# A co-location result in canonical form"))
check("  and every comment line is a comment, so a prose line cannot be read as a field",
      all(line.lstrip().startswith("#") or ":" in line or line.lstrip().startswith("- ")
          or not line.strip() for line in exportmod.dump(_ex_build()).splitlines()))

# The baseline, with nothing measured co-resident. This is the case a fleet is in before its first
# search, and exporting it is what makes the mode worth running then.
_ex_bare = _ex_build()
check("the baseline is exported with no search at all, because it is arithmetic on measured degrees",
      _ex_bare["baseline"]["devices"] == _ex_base.resolved_gpus
      and len(_ex_bare["baseline"]["slots"]) == _ex_base.resolved_gpus)
check("  and then there is no strict result and no relaxed one",
      _ex_bare["strict"] is None and _ex_bare["relaxed"] is None)
check("  and the absent relaxed result says why it is absent, rather than being a bare null",
      "no search has been run" in _ex_bare["relaxed_why_not"])
check("  and nothing is reported as rejected, since nothing was measured",
      _ex_bare["rejected"] == [])
check("  and the provenance says so too: no placement block, but the solo rows are still named",
      _ex_bare["provenance"]["placements"] is None
      and _ex_bare["provenance"]["baseline"]["per_tenant"]["m1"]["run_id"] == "profile-m1")
check("  and the two halves are addressed separately, because they are keyed differently",
      _ex_bare["fleet"]["validity_key_hash"] == "deadbeefcafe"
      and _ex_bare["provenance"]["baseline"]["profile_key_hash"] == "1234abcd5678")
check("  and the fleet is carried by reference: its path and the hash of its bytes, not a copy",
      _ex_bare["fleet"]["sha256"]
      == hashlib.sha256(_ex_fleet_file.read_bytes()).hexdigest())
check("  and no card's capacity appears anywhere in it",
      not [k for k, v in _ex_bare["fleet"].items() if isinstance(v, float)]
      and "device_total_gib" not in json.dumps(_ex_bare))

# The memory requirement. The trap this guards against is real and cost nothing only because it was
# caught while reading: `SoloProfile.envelope_gib` is what the tenant was GIVEN alone on a card,
# which is the whole card, and it sits one field away from what it needs.
_ex_required = {n: r["required_gib"] for n, r in _ex_bare["tenants"].items()}
check("the per-tenant memory requirement is the solver's envelope at the resolved degree",
      _ex_required == {t.name: round(t.envelope_at_resolved_gib, 3) for t in _ex_base.tenants},
      str(_ex_required))
check("  so it is well under a whole device, which is what the solo row would have said instead",
      all(0.0 < v < CARD_GIB for v in _ex_required.values()), str(_ex_required))

# A search, exported. The summary is the shape `_print_outcome` writes, reduced to what the exporter
# reads: a certified placement, a denser one that was measured and missed, and the relaxation table
# that says by how much.
_ex_summary = {
    "run_token": "tok", "verdict": "met", "baseline_gpus": 6, "stopped_because": "submitted",
    "reasoner": "sdk:test", "gpu_node": "node-a", "elapsed_s": 12.0,
    "strict": {"certified": True, "gpus": 4, "freed": 2, "run_id": "search-tok-001-dense",
               "assignment": {"m1": [0], "r1": [0], "r2": [1, 2], "t2": [3, 0]}},
    "relaxed": {"offered": False, "gpus": None, "freed": None, "run_id": None, "certified": False,
                "why_not": "no rejected placement is denser than what STRICT reports"},
    "closest_to_targets": {"run_id": "search-tok-001-dense"},
    "vs_solo": {"m1": {"solo_itl_ms": 10.0, "target_itl_ms": 25.0, "ratio": 2.0, "tp": 1,
                       "solo_run_id": "profile-m1"}},
    "attempts": [
        {"run_id": "search-tok-001-dense", "outcome": "measured", "gpus_used": 4,
         "assignment": {"m1": [0], "r1": [0], "r2": [1, 2], "t2": [3, 0]},
         "coresidency": {"m1": 2, "r1": 2, "r2": 0, "t2": 1},
         "envelope_gib": {"m1": 4.0, "r1": 4.0, "r2": 20.0, "t2": 36.0},
         "per_tenant_itl_ms": {"m1": 20.0, "r1": 12.0, "r2": 9.0, "t2": 11.0},
         "targets_met": {"m1": True, "r1": True, "r2": True, "t2": True}},
        {"run_id": "search-tok-002-denser", "outcome": "measured", "gpus_used": 3,
         "assignment": {"m1": [0], "r1": [0], "r2": [1, 2], "t2": [1, 2]},
         "per_tenant_itl_ms": {"m1": 40.0, "r1": 12.0, "r2": 9.0, "t2": 11.0},
         "targets_met": {"m1": False, "r1": True, "r2": True, "t2": False}},
        {"run_id": "search-tok-003-unready", "outcome": "not_ready", "gpus_used": 2,
         "assignment": {"m1": [0], "r1": [0], "r2": [1], "t2": [1]}, "targets_met": {}},
    ],
    "required_relaxation": {"search-tok-002-denser": {
        "blocked": [], "factor": 1.6,
        "per_tenant": {"m1": {"metric": "ITL mean", "ceiling": 25.0, "measured": 40.0,
                              "over_by": 15.0, "factor": 1.6, "tolerance_pct": 5.0,
                              "declared_tolerance": False, "within_tolerance": False}}}},
}
_ex_searched = _ex_build(_ex_summary)
check("a certified placement is exported as slots, freed devices, and what each tenant measured",
      _ex_searched["strict"]["devices"] == 4 and _ex_searched["strict"]["certified"] is True
      and len(_ex_searched["strict"]["slots"]) == 4)
check("  and the saving is scored against the baseline the summary recorded, not a recomputed one",
      _ex_searched["strict"]["scored_against_devices"] == 6
      and _ex_searched["strict"]["fraction_freed"] == round(2 / 6, 4))
check("  and a tenant's ratio against solo is computed for THIS placement, not copied from vs_solo",
      _ex_searched["strict"]["per_tenant"]["m1"]["ratio_vs_solo"] == 2.0)
check("  and the envelope it was granted is reported beside what it requires, never instead of it",
      _ex_searched["strict"]["per_tenant"]["t2"]["granted_gib"] == 36.0
      and _ex_searched["strict"]["per_tenant"]["t2"]["required_gib"] == _ex_required["t2"])
check("  and a certified placement carries no empty concession, which would read as one that was "
      "made and came to nothing", "concession" not in _ex_searched["strict"])
check("  and a relaxed result that was not offered is null, with the summary's own reason beside it",
      _ex_searched["relaxed"] is None
      and _ex_searched["relaxed_why_not"] == _ex_summary["relaxed"]["why_not"])
_ex_rejected = _ex_searched["rejected"]
check("the placements that were measured and missed are exported with their overshoot",
      [r["run_id"] for r in _ex_rejected] == ["search-tok-002-denser"], str(_ex_rejected))
check("  and the overshoot is against the ceiling as written, in ms and as a factor",
      _ex_rejected[0]["missed"]["m1"] == exportmod.Flow(
          {"metric": "ITL mean", "ceiling_ms": 25.0, "measured_ms": 40.0, "over_by_ms": 15.0,
           "over_by_factor": 1.6, "within_stated_tolerance": False}),
      str(_ex_rejected[0]["missed"]["m1"]))
check("  and a tenant that missed with no overshoot priced for it is still listed as a miss, "
      "because one absent from both lists reads as an oversight",
      _ex_rejected[0]["missed"]["t2"]["metric"] is None
      and "delivery gate" in _ex_rejected[0]["missed"]["t2"]["note"])
check("  and the tenants that met their targets in that same placement are named",
      _ex_rejected[0]["met"] == ["r1", "r2"])
check("  while a placement that never reached readiness is not exported as a rejection: it says "
      "something about that afternoon and nothing about the fleet",
      "search-tok-003-unready" not in json.dumps(_ex_rejected))
check("  and the certified placement is not listed as rejected as well",
      "search-tok-001-dense" not in [r["run_id"] for r in _ex_rejected])
check("  but the count of what was tried is kept, so 'nothing else was measured' stays "
      "distinguishable from 'nothing else is exported'",
      _ex_searched["provenance"]["placements"]["attempts"] == 3
      and _ex_searched["provenance"]["placements"]["measured"] == 2)

# A relaxed offer, which is the one block that carries a concession and the one whose saving may be
# scored against a different baseline than strict's.
_ex_offer = copy.deepcopy(_ex_summary)
_ex_offer["relaxed"] = {
    "offered": True, "gpus": 3, "freed": 2, "run_id": "search-tok-002-denser", "certified": False,
    "scored_against_baseline_gpus": 5, "baseline_moves_under_the_concession": True,
    "baseline_movement_checked": True,
    "concession": {"m1": {"metric": "ITL mean", "ceiling": 25.0, "measured": 26.0, "factor": 1.04,
                          "tolerance_pct": 5.0, "declared_tolerance": False}},
    "assignment": {"m1": [0], "r1": [0], "r2": [1, 2], "t2": [1, 2]}}
_ex_relaxed = _ex_build(_ex_offer)["relaxed"]
check("a relaxed offer is exported, and never as a certification",
      _ex_relaxed["devices"] == 3 and _ex_relaxed["certified"] is False)
check("  with the concession it rests on, per tenant, and whose tolerance it was",
      _ex_relaxed["concession"]["m1"]["over_by_factor"] == 1.04
      and _ex_relaxed["concession"]["m1"]["tolerance_declared"] is False)
check("  and scored against the baseline the same concession leaves standing, not against strict's: "
      "crediting co-location with a GPU the concession freed is the overstatement §0 exists for",
      _ex_relaxed["scored_against_devices"] == 5
      and _ex_relaxed["baseline_moves_under_the_concession"] is True)
check("  and its own latencies, not the ones vs_solo reports for the placement STRICT chose",
      _ex_relaxed["per_tenant"]["m1"]["itl_mean_ms"] == 40.0
      and _ex_build(_ex_offer)["strict"]["per_tenant"]["m1"]["itl_mean_ms"] == 20.0)

# Which stored search gets exported. More than one under a key is two answers, neither superseding
# the other, so the choice is the newest unless a token names one -- and a token that names nothing
# is an error that says what is there.
_ex_store = ObservationStore(_ex_dir / "store", "deadbeefcafe")
(_ex_store.key_dir / "searches").mkdir(parents=True)
for _name, _when in (("older", 1.0), ("newer", 2.0)):
    _p = _ex_store.key_dir / "searches" / f"{_name}.json"
    _p.write_text(json.dumps({"run_token": _name}))
    os.utime(_p, (_when, _when))
check("with no token, the newest search under the key is the one exported",
      exportmod.newest_search(_ex_store)[0]["run_token"] == "newer")
check("  and a token names one instead, since two searches under a key are two answers",
      exportmod.newest_search(_ex_store, "older")[0]["run_token"] == "older")
check("  and a key with no search at all is not an error: the baseline is still exportable",
      exportmod.newest_search(ObservationStore(_ex_dir / "store", "nothinghere")) == (None, None))
try:
    exportmod.newest_search(_ex_store, "absent")
    check("  while a token that names nothing is refused, listing what the key does hold", False)
except exportmod.ExportError as _exc:
    check("  while a token that names nothing is refused, listing what the key does hold",
          "newer" in str(_exc) and "older" in str(_exc), str(_exc))

# A search exports itself. The whole point of the canonical form is that the result somebody deploys
# from does not address a node that has been reclaimed, and a step nothing runs is a step that gets
# skipped: the mode exists for re-exporting, not for remembering. Read off the source because the path
# it is on needs a lease, and one call site is what makes the two paths the same document.
_ex_main = Path(agent_main.__file__).read_text()
_ex_search_src = _ex_main.split("\ndef _search(")[1].split("\ndef ")[0]
check("a search writes the canonical form as it finishes, rather than leaving it to be remembered",
      "_write_canonical(" in _ex_search_src)
check("  through the same call the `export` mode uses, so neither path is a second answer",
      _ex_main.count("def _write_canonical(") == 1 and _ex_main.count("_write_canonical(") == 3)
check("  and inside a try, because a failed write of the description must not discard the result",
      "try:" in _ex_search_src.split("_write_canonical(")[0][-300:])

# --- deploying an export on KAI fractions (kai/plan.py) ---------------------
# Offline, like everything above: the arithmetic that turns slots into fractions, the manifests, and
# the comparison that decides whether the binder reproduced the slots. The kubectl half is not here.
from colocation.kai import plan as kaiplan

_kai_file = _ex_dir / "kai-export.yaml"
_kai_file.write_text(exportmod.dump(_ex_build(_ex_summary)))
_kai_doc = kaiplan.load_export(_kai_file)
_kai = kaiplan.build(name="tok", result="strict", export_path=_kai_file, doc=_kai_doc, fleet=FLEET,
                     base=_ex_base, addressable_mib=ADDRESSABLE_MIB)
_kai_env = {t.name: t.envelope_at_resolved_gib for t in _ex_base.tenants}
check("a KAI plan re-derives exactly the envelopes the export states",
      kaiplan.envelope_mismatches(_kai_doc, FLEET, _ex_base) == [])
_kai_moved = copy.deepcopy(_kai_doc)
_kai_moved["tenants"]["m1"]["required_gib"] += 0.5
check("  and names the tenant whose required_gib no longer re-derives, so deploy can refuse it",
      [m.split(":")[0] for m in kaiplan.envelope_mismatches(_kai_moved, FLEET, _ex_base)] == ["m1"])
_kai_tp_only = [all(_kai.tenants[n].tp > 1 for n in s) for s in _kai.slots]
check("every slot's card is filled to the card less the margin, so binpack has one place to go",
      all(left == 0 for left, tp_only in zip(_kai.leftover_mib(), _kai_tp_only) if not tp_only),
      str(_kai.leftover_mib()))
check("  except a TP tenant's own card when a mate sits on its other one, since a TP fraction is "
      "one size on every device", any(_kai.leftover_mib()) and all(
          left == 0 or tp_only for left, tp_only in zip(_kai.leftover_mib(), _kai_tp_only)))
check("  and the card is the store's addressable MiB, not a figure in this package",
      _kai.card_mib == ADDRESSABLE_MIB - kaiplan.CARD_MARGIN_MIB)
check("a bare fraction is the envelope with the MPS reserve swapped for the KAI one",
      all(t.min_mib == math.ceil((_kai_env[n] + kaiplan.KAI_RESERVE_MARGIN_GIB) * 1024)
          for n, t in _kai.tenants.items()))
check("  and util hands vLLM the budget it was measured with, on whatever the fraction became",
      all(abs(t.util * t.mib / 1024 - (t.mib / 1024 - t.reserve_kai_gib) * _kai.headroom) < 1e-4
          for t in _kai.tenants.values()))
check("  which at the bare fraction is exactly the MPS run's (E - r_mps) * headroom",
      all(abs(((t.min_mib / 1024) - t.reserve_kai_gib) - (_kai_env[n] - t.reserve_mps_gib)) < 1 / 1024
          for n, t in _kai.tenants.items()))
_kai_multi = [n for n, t in _kai.tenants.items() if t.tp > 1]
_kai_mates = {n for n in _kai_multi
              if any(_kai.tenants[m].tp == 1 for i in _kai.tenants[n].slots for m in _kai.slots[i])}
check("a TP tenant beside a single-shard mate is not padded: the mate takes that card's slack",
      _kai_mates and all(_kai.tenants[n].mib == _kai.tenants[n].min_mib for n in _kai_mates),
      str({n: (_kai.tenants[n].min_mib, _kai.tenants[n].mib) for n in _kai_multi}))
check("  while one alone on its cards is padded to fill them",
      all(_kai.tenants[n].mib == _kai.card_mib for n in _kai_multi if n not in _kai_mates))
# Binpack takes the fullest card a fraction fits, so a part-empty TP card would attract anything
# submitted after it: cards that fill exactly go first, then TP tenants, then the mates on TP cards.
_kai_order = kaiplan.submission_order([["a"], ["t", "m"], ["t"], ["b", "c"]],
                                      {"a": 1, "b": 1, "c": 1, "m": 1, "t": 2})
check("slots with no TP tenant are submitted first, then TP tenants, then their mates",
      _kai_order == ["a", "b", "c", "t", "m"], str(_kai_order))
check("  and the plan submits in that order", _kai.order == kaiplan.submission_order(
    _kai.slots, {n: t.tp for n, t in _kai.tenants.items()}) and set(_kai.order) == set(_kai.tenants))
try:
    kaiplan.size([["a", "b"]], {"a": (1, 30.0, 1.0), "b": (1, 20.0, 1.0)}, 45000)
    check("a slot whose bare fractions overfill the card is refused, not squeezed", False)
except kaiplan.PlanError as _exc:
    check("a slot whose bare fractions overfill the card is refused, not squeezed",
          "slot 0" in str(_exc), str(_exc))
try:
    kaiplan.choose_slots(_kai_doc, "relaxed")
    check("an absent result is refused, naming the baseline as the one that always exists", False)
except kaiplan.PlanError as _exc:
    check("an absent result is refused, naming the baseline as the one that always exists",
          "--result baseline" in str(_exc))
_kai_capped = copy.deepcopy(_kai_doc)
_kai_cap_res = next(r for r in ("strict", "relaxed") if _kai_capped.get(r))
_kai_cap_t = sorted(_kai_capped[_kai_cap_res]["per_tenant"])[0]
_kai_capped[_kai_cap_res]["per_tenant"][_kai_cap_t]["active_thread_percentage"] = 25
_kai_capped_file = _ex_dir / "kai-export-capped.yaml"
_kai_capped_file.write_text(yaml.safe_dump(_kai_capped))
_kai_cap_plan = kaiplan.build(name="tok", result=_kai_cap_res, export_path=_kai_capped_file,
                              doc=_kai_capped, fleet=FLEET, base=_ex_base,
                              addressable_mib=ADDRESSABLE_MIB)
_kai_cap_env = {n: {e["name"]: e["value"] for e in kaiplan.tenant_values(_kai_cap_plan, n)[
    "tenant"]["env"]} for n in _kai_cap_plan.tenants}
check("a result measured with an SM cap is served on KAI with that cap in the capped pod's env",
      _kai_cap_env[_kai_cap_t].get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") == "25", str(_kai_cap_env))
check("  and no other pod, or any pod of an uncapped result, gets the variable",
      not any("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in v for n, v in _kai_cap_env.items()
              if n != _kai_cap_t) and not any(
          e["name"] == "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" for n in _kai.tenants
          for e in kaiplan.tenant_values(_kai, n)["tenant"]["env"]))
check("  and `render` prints the cap beside the capped tenant on each of its slots, and nowhere else",
      kaiplan.describe(_kai_cap_plan).count(" SM cap 25%")
      == len(_kai_cap_plan.tenants[_kai_cap_t].slots)
      and all(f"{_kai_cap_t}" in l for l in kaiplan.describe(_kai_cap_plan).splitlines()
              if "SM cap" in l)
      and "SM cap" not in kaiplan.describe(_kai), kaiplan.describe(_kai_cap_plan))
check("the plan survives the release it is kept in, which is all verify and teardown read",
      kaiplan.Plan.from_dict(json.loads(kaiplan.plan_values(_kai, "ns")["plan"]["planJson"])) == _kai)

_kai_val = {n: kaiplan.tenant_values(_kai, n)["tenant"] for n in _kai.tenants}
check("every tenant is a KAI fraction of exactly the planned size, across as many devices as it "
      "has shards", all(v["memory"] == f"{_kai.tenants[n].mib}Mi" and v["devices"] == _kai.tenants[n].tp
                        for n, v in _kai_val.items()))
check("  with vLLM told the utilisation computed for the fraction, not the card",
      all(f"--gpu-memory-utilization={_kai.tenants[n].util}" in v["args"]
          for n, v in _kai_val.items()))
check("  and a tenant after the first is pinned to the first one's node",
      kaiplan.tenant_values(_kai, "m1", node="node-a")["tenant"]["nodeSelector"]
      ["kubernetes.io/hostname"] == "node-a"
      and "kubernetes.io/hostname" not in _kai_val["m1"]["nodeSelector"])
check("  and every release is named as its objects are, the plan's first, then submission order",
      [r for r, _ in kaiplan.releases(_kai, "ns")]
      == [kaiplan.object_name("tok", n) for n in ["plan", *_kai.order]]
      and all(v["name"] == kaiplan.object_name("tok", n) for n, v in _kai_val.items()))

_kai_apart = kaiplan.tenant_values(_kai, "m1", avoid=["tok-n1"], zone="zone-x")["tenant"]


def _kai_anti(t):
    return t["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]


def _kai_zone(t):
    return (t["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
            ["nodeSelectorTerms"])


check("a later node's first tenant avoids the hosts of the earlier nodes' pods, in the first zone",
      _kai_anti(_kai_apart) == [{"labelSelector": {"matchExpressions": [
          {"key": kaiplan.LABEL_DEPLOYMENT, "operator": "In", "values": ["tok-n1"]}]},
          "topologyKey": "kubernetes.io/hostname"}]
      and _kai_zone(_kai_apart) == [{"matchExpressions": [
          {"key": kaiplan.ZONE_LABEL, "operator": "In", "values": ["zone-x"]}]}]
      and "affinity" not in _kai_val["m1"])
# Karpenter refuses to provision for a node affinity on the hostname label (2026-10-10), and this
# pod is the one that has to bring the next node up.
check("  and nothing a pod brings a node up with names the hostname label in a node affinity",
      "kubernetes.io/hostname" not in json.dumps(_kai_apart["affinity"].get("nodeAffinity", {})))

# A version 2 export deploys node by node, each node from its group's own file, which has to be the
# one the split export copied.
_kai_body = {k: v for k, v in _kai_doc.items() if k not in ("schema_version", "generated_at")}
_kai_v2 = {"schema_version": 2, "nodes": [{"export": str(_kai_file), **_kai_body}]}
check("a version 1 export is one node, its own file",
      kaiplan.node_exports(_kai_doc, _kai_file, Path("/")) == [(_kai_file, _kai_doc)])
check("  and a version 2 export's nodes are its groups' files, read back",
      kaiplan.node_exports(_kai_v2, _ex_dir / "split.yaml", Path("/")) == [(_kai_file, _kai_doc)])
_kai_v2_moved = copy.deepcopy(_kai_v2)
_kai_v2_moved["nodes"][0]["baseline"]["devices"] += 1
try:
    kaiplan.node_exports(_kai_v2_moved, _ex_dir / "split.yaml", Path("/"))
    check("  and a group file that is no longer the one copied is refused", False)
except kaiplan.PlanError as _exc:
    check("  and a group file that is no longer the one copied is refused",
          "exported again" in str(_exc), str(_exc))

# `deploy` of two nodes, with the cluster half replaced: what each install was asked for.
from colocation.kai import __main__ as kaimain   # noqa: E402
from colocation.kai import cluster as kaicluster  # noqa: E402
_kai_n2 = copy.deepcopy(_kai)
_kai_n2.name = "tok-n2"
_kai_installs, _kai_saved = [], {f: getattr(kaicluster, f)
                                for f in ("live_pods", "install", "wait_bound", "get_json")}
_kai_where = {"tok": "node-a", "tok-n2": "node-b"}
kaicluster.live_pods = lambda ns, name: []
kaicluster.install = lambda ns, name, release, values, chart: _kai_installs.append((name, values))
kaicluster.wait_bound = lambda ns, name, tenant, timeout: {
    "spec": {"nodeName": _kai_where[name]}, "metadata": {"annotations": {}}}
kaicluster.get_json = lambda argv, check=True: {
    "metadata": {"labels": {kaiplan.ZONE_LABEL: "zone-x"}}}
_kai_real_verify = kaimain._verify
kaimain._verify = lambda args, plan: 0
try:
    with contextlib.redirect_stdout(io.StringIO()):
        _kai_rc = kaimain._deploy(SimpleNamespace(namespace="ns", queue="q", chart="c",
                                                  bind_timeout=1, next_bind_timeout=1),
                                  [_kai, _kai_n2])
finally:
    for _f, _v in _kai_saved.items():
        setattr(kaicluster, _f, _v)
    kaimain._verify = _kai_real_verify
_kai_t2 = [v["tenant"] for name, v in _kai_installs if name == "tok-n2" and "tenant" in v]
_kai_t1 = [v["tenant"] for name, v in _kai_installs if name == "tok" and "tenant" in v]
check("deploying a split: the second node's first tenant avoids the first node, in its zone",
      _kai_rc == 0 and len(_kai_t2) == len(_kai.order)
      and _kai_anti(_kai_t2[0])[0]["labelSelector"]["matchExpressions"][0]["values"] == ["tok"]
      and _kai_zone(_kai_t2[0])[0]["matchExpressions"]
      == [{"key": kaiplan.ZONE_LABEL, "operator": "In", "values": ["zone-x"]}],
      json.dumps(_kai_t2[:1])[:300])
check("  and the rest of its tenants follow it to its own node, while the first node's are as before",
      all("affinity" not in t and t["nodeSelector"]["kubernetes.io/hostname"] == "node-b"
          for t in _kai_t2[1:])
      and all("affinity" not in t for t in _kai_t1)
      and all(t["nodeSelector"]["kubernetes.io/hostname"] == "node-a" for t in _kai_t1[1:]))

# The chart lays the values out; what it lays out is checked as manifests, because the binder reads
# annotation keys and the scheduler reads fields no values check can see. Needs helm and the chart,
# which a harness image has neither of.
_kai_chart = Path(kaiplan.__file__).resolve().parents[3] / kaiplan.CHART
if shutil.which("helm") is None or not _kai_chart.is_dir():
    print("NOT CHECKED here: the kai-vllm chart's manifests -- no helm on PATH, or no chart beside "
          "the source. Run the suite from the repository with helm installed.")
else:
    import subprocess as _kai_sp

    def _kai_template(values):
        out = _kai_sp.run(["helm", "template", "r", str(_kai_chart), "-n", "ns", "-f", "-"],
                          input=yaml.safe_dump(values), capture_output=True, text=True, check=True)
        return {(d["kind"], d["metadata"]["name"]): d
                for d in yaml.safe_load_all(out.stdout) if d}

    def _kai_rendered(plan):
        return {n: _kai_template(kaiplan.tenant_values(plan, n)) for n in plan.tenants}

    _kai_out = _kai_rendered(_kai)
    _kai_cap_out = _kai_rendered(_kai_cap_plan)
    _kai_dep = {n: o[("Deployment", kaiplan.object_name("tok", n))] for n, o in _kai_out.items()}
    _kai_pod = {n: d["spec"]["template"] for n, d in _kai_dep.items()}
    check("a tenant's release is one Deployment and one Service, named as the release is",
          all(set(o) == {("Deployment", kaiplan.object_name("tok", n)),
                         ("Service", kaiplan.object_name("tok", n))} for n, o in _kai_out.items()))
    check("every tenant is a KAI fraction of exactly the planned size, in sm-sharing",
          all(p["metadata"]["annotations"][kaiplan.REQUEST_ANNOTATION] == f"{_kai.tenants[n].mib}Mi"
              and p["metadata"]["annotations"][f"nvidia.com/container.{kaiplan.CONTAINER}.gpu-memory.limit"]
              == f"{_kai.tenants[n].mib}Mi"
              and p["metadata"]["annotations"][f"nvidia.com/container.{kaiplan.CONTAINER}.gpu-compute.mode"]
              == "sm-sharing"
              and p["spec"]["containers"][0]["name"] == kaiplan.CONTAINER
              and p["spec"]["schedulerName"] == "kai-scheduler" for n, p in _kai_pod.items()))
    check("  across as many devices as it has shards, and the annotation is absent at TP=1",
          all(p["metadata"]["annotations"].get("gpu-fraction-num-devices")
              == (None if _kai.tenants[n].tp == 1 else str(_kai.tenants[n].tp))
              for n, p in _kai_pod.items()) and any(t.tp > 1 for t in _kai.tenants.values()))
    check("  started without the NGC entrypoint, outside the mesh, and never two at once",
          all(p["spec"]["containers"][0]["command"] == ["python3"]
              and p["metadata"]["annotations"]["sidecar.istio.io/inject"] == "false"
              for p in _kai_pod.values())
          and all(d["spec"]["strategy"]["type"] == "Recreate" and d["spec"]["replicas"] == 1
                  for d in _kai_dep.values()))
    check("  with the values' args, env, image, selector and toleration, unchanged by the chart",
          all(p["spec"]["containers"][0]["args"] == _kai_val[n]["args"]
              and p["spec"]["containers"][0]["env"] == _kai_val[n]["env"]
              and p["spec"]["containers"][0]["image"] == _kai_val[n]["image"]
              and p["spec"]["nodeSelector"] == _kai_val[n]["nodeSelector"]
              and p["spec"]["tolerations"] == [{"key": kaiplan.TAINT, "operator": "Exists",
                                                "effect": "NoSchedule"}]
              for n, p in _kai_pod.items()))
    check("  on the FSx claim and an in-memory /dev/shm of the planned size",
          all({v["name"]: v for v in p["spec"]["volumes"]}["fsx"]["persistentVolumeClaim"]
              ["claimName"] == kaiplan.FSX_CLAIM
              and {v["name"]: v for v in p["spec"]["volumes"]}["dshm"]["emptyDir"]
              == {"medium": "Memory", "sizeLimit": f"{_kai.tenants[n].shm_size_gi}Gi"}
              for n, p in _kai_pod.items()))
    check("  and labelled so the cluster half finds its pods, behind a Service on the same labels",
          all(p["metadata"]["labels"] == {**kaiplan.labels("tok", n),
                                          kaiplan.LABEL_QUEUE: kaiplan.DEFAULT_QUEUE}
              and _kai_dep[n]["spec"]["selector"]["matchLabels"] == kaiplan.labels("tok", n)
              and _kai_out[n][("Service", kaiplan.object_name("tok", n))]["spec"]["selector"]
              == kaiplan.labels("tok", n) for n, p in _kai_pod.items()))
    check("a capped tenant's pod carries its cap through the chart, and no other pod does",
          all(({e["name"]: e["value"] for e in o[("Deployment", kaiplan.object_name("tok", n))]
                ["spec"]["template"]["spec"]["containers"][0]["env"]}
               .get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")) == ("25" if n == _kai_cap_t else None)
              for n, o in _kai_cap_out.items()))
    _kai_apart_pod = _kai_template({"tenant": _kai_apart})[
        ("Deployment", kaiplan.object_name("tok", "m1"))]["spec"]["template"]["spec"]
    check("  and the chart lays a later node's affinity out as given, and none where there is none",
          _kai_apart_pod["affinity"] == _kai_apart["affinity"]
          and all("affinity" not in p["spec"] for p in _kai_pod.values()))
    _kai_cm = _kai_template(kaiplan.plan_values(_kai, "ns", "node-a"))
    check("the plan's release is the one ConfigMap verify reads, with the plan and the node intact",
          list(_kai_cm) == [("ConfigMap", kaiplan.object_name("tok", "plan"))]
          and kaiplan.Plan.from_dict(json.loads(
              _kai_cm[("ConfigMap", kaiplan.object_name("tok", "plan"))]["data"]["plan.json"]))
          == _kai and _kai_cm[("ConfigMap", kaiplan.object_name("tok", "plan"))]["data"]["node"]
          == "node-a")


def _kai_bound(cards_by_slot, node="node-a"):
    """Pods as the binder leaves them: each tenant's cards, from per-slot card ids."""
    devs = {}
    for slot, card in zip(_kai.slots, cards_by_slot):
        for n in slot:
            devs.setdefault(n, []).append(card)
    return {n: {"node": node, "annotations": {
        kaiplan.REQUEST_ANNOTATION: f"{_kai.tenants[n].mib}Mi",
        kaiplan.DEVICES_ANNOTATION: ",".join(f"k8s.device-plugin.nvidia.com/gpu=GPU-{c}"
                                             for c in reversed(d))}} for n, d in devs.items()}


_kai_cards = [f"u{i}" for i in range(len(_kai.slots))]
check("the slots match whatever cards the binder chose, and in whatever shard order",
      kaiplan.compare(_kai, _kai_bound(list(reversed(_kai_cards)))) == [])
_kai_swapped = _kai_bound(_kai_cards)
_kai_single = [n for n in _kai.tenants if _kai.tenants[n].tp == 1 and len(_kai.slots[_kai.tenants[n].slots[0]]) > 1]
_kai_alone = [i for i, s in enumerate(_kai.slots) if _kai_single[0] not in s][0]
_kai_swapped[_kai_single[0]]["annotations"][kaiplan.DEVICES_ANNOTATION] = f"x=GPU-{_kai_cards[_kai_alone]}"
check("a tenant the binder put on another slot's card is a mismatch",
      any("where the slots say" in p for p in kaiplan.compare(_kai, _kai_swapped)))
_kai_split = _kai_bound(_kai_cards)
_kai_split[_kai_single[0]]["node"] = "node-b"
check("  and so is a fleet spread over two nodes",
      any("2 nodes" in p for p in kaiplan.compare(_kai, _kai_split)))
check("  and so is a card shared with a fraction this plan does not own",
      any("also holds" in p for p in kaiplan.compare(_kai, _kai_bound(_kai_cards),
                                                     {"GPU-u0": ["other/pod"], "u0": ["other/pod"]})))
_kai_unbound = _kai_bound(_kai_cards)
_kai_unbound[_kai_single[0]]["annotations"].pop(kaiplan.DEVICES_ANNOTATION)
check("  while a tenant not yet bound is reported as that, not as a misplacement",
      any("not been bound" in p for p in kaiplan.compare(_kai, _kai_unbound)))

# 2026-10-05: a Deployment's pod sat Pending as NonPreemptibleOverQuota in default-queue (quota 0).
def _kai_queue(quota):
    return {"spec": {"resources": {"gpu": {"quota": quota, "limit": -1}}}}


check("tenants go to the serving queue, whose quota a non-preemptible Deployment needs",
      all(v["podLabels"][kaiplan.LABEL_QUEUE] == kaiplan.DEFAULT_QUEUE != "default-queue"
          for v in _kai_val.values()))
check("  a queue with no GPU quota is refused, with or without a plan",
      kaiplan.queue_shortfall(_kai_queue(0), "q", len(_kai.slots)) is not None
      and kaiplan.queue_shortfall(_kai_queue(0), "q") is not None
      and kaiplan.queue_shortfall({"spec": {}}, "q") is not None)
check("  and so is one short of the plan's cards, while enough or unlimited (-1) is not",
      kaiplan.queue_shortfall(_kai_queue(len(_kai.slots) - 1), "q", len(_kai.slots)) is not None
      and kaiplan.queue_shortfall(_kai_queue(len(_kai.slots)), "q", len(_kai.slots)) is None
      and kaiplan.queue_shortfall(_kai_queue(-1), "q", len(_kai.slots)) is None)

# Drift: the real traffic and its latency against the fleet file, per tenant and per whole hour.
from colocation.kai import drift as kaidrift              # noqa: E402
from datetime import datetime as _dt, timedelta as _td, timezone as _tz  # noqa: E402

_dr_expr = kaidrift.expressions("ns", "tok")
check("every drift figure selects this deployment in its namespace, and sums by tenant",
      set(_dr_expr) == {"requests", "isl_tokens", "osl_tokens", "itl_mean_s", "ttft_p95_s"}
      and all('namespace="ns"' in e and f'{kaidrift.DEPLOYMENT}="tok"' in e
              and kaidrift.TENANT in e for e in _dr_expr.values()))
check("  ITL is each request's own mean token gap, the harness's itl_mean_ms, not the per-token one",
      "vllm:request_time_per_output_token_seconds_sum" in _dr_expr["itl_mean_s"]
      and "inter_token" not in _dr_expr["itl_mean_s"])
check("  and the chart's recording rules are those same expressions, so a graph and a record agree",
      kaiplan.plan_values(_kai, "ns")["plan"]["rules"] == kaidrift.recording_rules("ns", "tok")
      and [r["expr"] for r in kaidrift.recording_rules("ns", "tok")] == list(_dr_expr.values()))
check("  and Prometheus labels the series by the labels the ServiceMonitor copies onto them",
      kaidrift.TENANT == re.sub(r"[^a-zA-Z0-9_]", "_", kaiplan.LABEL_TENANT)
      and kaidrift.DEPLOYMENT == re.sub(r"[^a-zA-Z0-9_]", "_", kaiplan.LABEL_DEPLOYMENT)
      and kaiplan.tenant_values(_kai, "m1")["tenant"]["monitor"]["targetLabels"]
      == [kaiplan.LABEL_DEPLOYMENT, kaiplan.LABEL_TENANT])


def _dr_t(hh, mm=0, day=7):
    return _dt(2026, 10, day, hh, mm, tzinfo=_tz.utc)


check("only whole clock hours after the deploy are recorded, and only once settled",
      kaidrift.window_ends(_dr_t(10, 20), None, _dr_t(13, 3)) == [_dr_t(12)]
      and kaidrift.window_ends(_dr_t(10, 20), None, _dr_t(13, 6)) == [_dr_t(12), _dr_t(13)])
check("  never an hour already recorded, and none before Prometheus's retention",
      kaidrift.window_ends(_dr_t(10, 20), _dr_t(12), _dr_t(13, 3)) == []
      and kaidrift.window_ends(_dr_t(10, 20), _dr_t(12), _dr_t(13, 6)) == [_dr_t(13)]
      and kaidrift.window_ends(_dr_t(10) - _td(days=40), None, _dr_t(13, 6))[0]
      == _dr_t(14) - _td(days=30))
check("  and the last one recorded is read back from the file",
      kaidrift.last_end([json.dumps({"window_end": _dr_t(12).isoformat()}),
                         json.dumps({"window_end": _dr_t(11).isoformat()}), ""]) == _dr_t(12))
check("an idle hour's 0/0 is no value, not a number",
      kaidrift.series({"data": {"result": [{"metric": {kaidrift.TENANT: "a"},
                                            "values": [[3600, "NaN"], [7200, "0.5"]]}]}})
      == {"a": {3600: None, 7200: 0.5}})

_dr_names = list(_kai.order)
_dr_busy, _dr_thin = _dr_names[0], _dr_names[1]
_dr_doc = copy.deepcopy(_kai_doc)
_dr_doc["tenants"][_dr_busy]["targets"].update(itl_mean_ms=8.0, ttft_p95_ms=500.0)
_dr_refs = kaidrift.references(_dr_doc, "strict")
_dr_end = int(_dr_t(12).timestamp())
_dr_fleet = _dr_doc["tenants"][_dr_busy]


_dr_fig = {
    "requests": {kaiplan.dns(_dr_busy): {_dr_end: 1.5 * _dr_fleet["demand"]["rate_hz"] * 3600},
                 kaiplan.dns(_dr_thin): {_dr_end: 10.0}},
    "isl_tokens": {kaiplan.dns(_dr_busy): {_dr_end: float(_dr_fleet["demand"]["isl_tokens"])}},
    "osl_tokens": {kaiplan.dns(_dr_busy): {_dr_end: float(_dr_fleet["demand"]["osl_tokens"])}},
    "itl_mean_s": {kaiplan.dns(_dr_busy): {_dr_end: 2 * _dr_fleet["targets"]["itl_mean_ms"] / 1000},
                   kaiplan.dns(_dr_thin): {_dr_end: 0.004}},
    "ttft_p95_s": {kaiplan.dns(_dr_busy): {_dr_end: 0.25}, kaiplan.dns(_dr_thin): {_dr_end: 0.1}},
}
_dr_recs = {r["tenant"]: r for r in kaidrift.records(_kai, _dr_refs, _dr_fig, [_dr_t(12)])}
check("one record per tenant per hour, idle tenants included",
      set(_dr_recs) == set(_kai.tenants)
      and all(r["window_start"] == _dr_t(11).isoformat() for r in _dr_recs.values()))
_dr_b = _dr_recs[_dr_busy]
check("  the demand beside the fleet file's: half as many requests again is +50%, the same length 0%",
      _dr_b["status"] == kaidrift.MEASURED
      and _dr_b["demand"]["rate_hz"]["vs_fleet_pct"] == 50.0
      and _dr_b["demand"]["rate_hz"]["fleet"] == _dr_fleet["demand"]["rate_hz"]
      and _dr_b["demand"]["isl_tokens"]["vs_fleet_pct"] == 0.0, str(_dr_b["demand"]))
check("  and the latency beside both the target and what the search certified for this result",
      _dr_b["latency"]["itl_mean_ms"]["vs_target_pct"] == 100.0
      and _dr_b["latency"]["itl_mean_ms"]["certified"]
      == _kai_doc["strict"]["per_tenant"][_dr_busy]["itl_mean_ms"]
      and _dr_b["latency"]["ttft_p95_ms"]["observed"] == 250.0
      and _dr_b["latency"]["ttft_p95_ms"]["vs_target_pct"] == -50.0,
      str(_dr_b["latency"]))
_dr_th = _dr_recs[_dr_thin]
check("  a tenant with too few requests for a p95 gets its ITL and demand, and no p95",
      _dr_th["status"] == kaidrift.THIN and _dr_th["requests"] == 10
      and _dr_th["latency"]["itl_mean_ms"]["observed"] == 4.0
      and _dr_th["latency"]["ttft_p95_ms"]["observed"] is None
      and _dr_th["latency"]["ttft_p95_ms"]["vs_target_pct"] is None)
_dr_idle = [r for n, r in _dr_recs.items() if n not in (_dr_busy, _dr_thin)]
check("  and an hour with no requests is no traffic, with no figure in it, not zero drift",
      _dr_idle and all(r["status"] == kaidrift.IDLE and r["requests"] == 0
                       and all(v["observed"] is None
                               for v in (*r["demand"].values(),
                                         r["latency"]["itl_mean_ms"], r["latency"]["ttft_p95_ms"]))
                       for r in _dr_idle))
check("  each record names the export and result it is read against",
      all(r["export_sha256"] == _kai.export_sha256 and r["result"] == "strict"
          for r in _dr_recs.values()))
_dr_text = kaidrift.describe(list(_dr_recs.values()))
check("  and prints a line per tenant, saying the drift against the fleet file and the target",
      len(_dr_text.splitlines()) == len(_dr_recs) and "+50% vs fleet" in _dr_text
      and "+100% vs target" in _dr_text and "no traffic" in _dr_text, _dr_text)
_dr_base = kaidrift.references(_kai_doc, "baseline")
check("a baseline result is held against its solo ITL, and has no certified TTFT",
      all(_dr_base[n]["certified"]["ttft_p95_ms"] is None
          and _dr_base[n]["certified"]["itl_mean_ms"]
          == _kai_doc["baseline"]["per_tenant"][n]["solo_itl_ms"] for n in _dr_base))

if shutil.which("helm") is not None and _kai_chart.is_dir():
    def _kai_template_mon(values, api=True):
        argv = ["helm", "template", "r", str(_kai_chart), "-n", "ns", "-f", "-"]
        if api:
            argv += ["--api-versions", "monitoring.coreos.com/v1/ServiceMonitor",
                     "--api-versions", "monitoring.coreos.com/v1/PrometheusRule"]
        out = _kai_sp.run(argv, input=yaml.safe_dump(values), capture_output=True, text=True,
                          check=True)
        return {(d["kind"], d["metadata"]["name"]): d for d in yaml.safe_load_all(out.stdout) if d}

    _dr_sm = _kai_template_mon(kaiplan.tenant_values(_kai, "m1")).get(
        ("ServiceMonitor", kaiplan.object_name("tok", "m1")))
    check("where Prometheus's CRDs are, each tenant is scraped on its Service's port, with its "
          "deployment and tenant labels copied onto every series",
          _dr_sm is not None and _dr_sm["spec"]["selector"]["matchLabels"]
          == {kaiplan.LABEL_DEPLOYMENT: "tok", kaiplan.LABEL_TENANT: "m1"}
          and _dr_sm["spec"]["targetLabels"] == [kaiplan.LABEL_DEPLOYMENT, kaiplan.LABEL_TENANT]
          and _dr_sm["spec"]["endpoints"] == [{"port": "http", "path": "/metrics",
                                               "interval": kaidrift.SCRAPE_INTERVAL}])
    check("  and selected without app.kubernetes.io/managed-by, which Helm rewrites on install",
          "app.kubernetes.io/managed-by" not in _dr_sm["spec"]["selector"]["matchLabels"])
    _dr_pr = _kai_template_mon(kaiplan.plan_values(_kai, "ns")).get(
        ("PrometheusRule", kaiplan.object_name("tok", "plan")))
    check("  and the plan's release carries the recording rules, unchanged by the chart",
          _dr_pr is not None
          and _dr_pr["spec"]["groups"][0]["rules"] == kaidrift.recording_rules("ns", "tok"))
    check("  and where they are not, neither is rendered and the deploy still goes ahead",
          set(_kai_template_mon(kaiplan.tenant_values(_kai, "m1"), api=False))
          == {("Deployment", kaiplan.object_name("tok", "m1")),
              ("Service", kaiplan.object_name("tok", "m1"))}
          and set(_kai_template_mon(kaiplan.plan_values(_kai, "ns"), api=False))
          == {("ConfigMap", kaiplan.object_name("tok", "plan"))})

# --- a load test of a KAI deployment (kai/load.py) --------------------------
from colocation.kai import load as kaiload              # noqa: E402

_ld_root = Path(kaiplan.__file__).resolve().parents[3]


def _ld_scenario(text):
    p = _ex_dir / "scenario.yaml"
    p.write_text(text)
    return p


def _ld_refused(text, doc=None):
    try:
        s = kaiload.load_scenario(_ld_scenario(text))
        if doc is not None:
            kaiload.demand(s, doc)
    except kaiload.LoadError as exc:
        return str(exc)
    return None


_ld_fleet = _kai_doc["fleet"]["path"]
_ld_doc = copy.deepcopy(_kai_doc)
_ld_names = sorted(_ld_doc["tenants"])
_ld_a, _ld_b = _ld_names[0], _ld_names[1]
_ld_doc["tenants"][_ld_a]["max_model_len"] = (_ld_doc["tenants"][_ld_a]["demand"]["isl_tokens"]
                                              + _ld_doc["tenants"][_ld_a]["demand"]["osl_tokens"])
check("a scenario is strict: an unknown key, a rate given twice, a non-positive change, and hours "
      "that are not a whole number are each refused",
      all(_ld_refused(t) for t in (
          f"fleet: {_ld_fleet}\nhours: 2\nduration: 3\n",
          f"fleet: {_ld_fleet}\nhours: 2\ntenants: {{{_ld_a}: {{rate: 2, per_hour: 10}}}}\n",
          f"fleet: {_ld_fleet}\nhours: 2\ntenants: {{{_ld_a}: {{rate: 0}}}}\n",
          f"fleet: {_ld_fleet}\nhours: 2\ntenants: {{{_ld_a}: {{speed: 2}}}}\n",
          f"fleet: {_ld_fleet}\nhours: 1.5\n")))
check("  as is one for another fleet, a tenant the fleet does not have, and a request past the "
      "tenant's context",
      "exported from" in (_ld_refused("fleet: fleets/x.yaml\nhours: 1\n", _ld_doc) or "")
      and "no tenant nope" in (_ld_refused(f"fleet: {_ld_fleet}\nhours: 1\n"
                                           "tenants: {nope: {rate: 2}}\n", _ld_doc) or "")
      and "max_model_len" in (_ld_refused(f"fleet: {_ld_fleet}\nhours: 1\n"
                                          f"tenants: {{{_ld_a}: {{isl: 1.1}}}}\n", _ld_doc) or "")
      and _ld_refused(f"fleet: {_ld_fleet}\nhours: 1\n"
                      + "tenants: {" + ", ".join(f"{n}: off" for n in _ld_names) + "}\n",
                      _ld_doc) is not None)
_ld_s = kaiload.load_scenario(_ld_scenario(
    f"fleet: {_ld_fleet}\nhours: 2\ntenants:\n  {_ld_a}: {{rate: 2.0, osl: 0.5}}\n"
    f"  {_ld_b}: {{per_hour: 10}}\n" + "".join(f"  {n}: off\n" for n in _ld_names[3:])))
_ld_dem = kaiload.demand(_ld_s, _ld_doc)
_ld_fd = {n: _ld_doc["tenants"][n]["demand"] for n in _ld_names}
check("the traffic is the export's demand times the scenario's changes, a count replaces a rate, "
      "an unlisted tenant runs as its fleet says and `off` is not driven",
      _ld_dem[_ld_a] == (2 * _ld_fd[_ld_a]["rate_hz"], _ld_fd[_ld_a]["isl_tokens"],
                         round(_ld_fd[_ld_a]["osl_tokens"] * 0.5))
      and _ld_dem[_ld_b][0] == 10 / 3600
      and _ld_dem[_ld_names[2]] == tuple(_ld_fd[_ld_names[2]][k]
                                         for k in ("rate_hz", "isl_tokens", "osl_tokens"))
      and set(_ld_dem) == set(_ld_names[:3]), str(_ld_dem))
_ld_cfg = kaiload.run_config(_kai, _ld_doc, _ld_s, "ns", "run-1")
_ld_t = {t["name"]: t for t in _ld_cfg["tenants"]}
check("  each tenant is driven at its own Service, by its served name and its model's tokenizer, "
      "as vLLM's OpenAI server",
      _ld_t[_ld_a]["base_url"] == f"http://{kaiplan.object_name('tok', _ld_a)}.ns.svc:{kaiplan.PORT}"
      and _ld_t[_ld_a]["served_name"] == _kai.tenants[_ld_a].served_name
      and _ld_t[_ld_a]["tokenizer_path"] == _ld_doc["tenants"][_ld_a]["model_path"]
      and {t["engine"] for t in _ld_cfg["tenants"]} == {"vllm-openai"}, str(_ld_t[_ld_a]))
_ld_fields = {n.target.id for c in ast.parse((
    _ld_root / "containers/colocation-harness/harness/config.py").read_text()).body
    if isinstance(c, ast.ClassDef) and c.name == "TenantConfig"
    for n in c.body if isinstance(n, ast.AnnAssign)}
check("  in a run config the harness reads: every tenant field is one TenantConfig has",
      all(set(t) <= _ld_fields for t in _ld_cfg["tenants"]), str(set(_ld_cfg["tenants"][0])
                                                                 - _ld_fields))
_ld_cm, _ld_job = kaiload.manifests(
    yaml.safe_load((_ld_root / "containers/colocation-harness/deploy/job.yaml").read_text()),
    _kai, _ld_cfg, 2, "ns", "img:1", "zone-a", "x/demand-drift.yaml")
_ld_pod = _ld_job["spec"]["template"]
_ld_c = _ld_pod["spec"]["containers"][0]
check("the load Job is the harness Job running harness.soak for the scenario's hours, on its own "
      "run config, in the zone of the deployment",
      _ld_c["command"] == ["python", "-m", "harness.soak"]
      and _ld_c["args"] == ["--config=/config/run.yaml", "--hours=2"] and _ld_c["image"] == "img:1"
      and any(v.get("configMap", {}).get("name") == "run-1" for v in _ld_pod["spec"]["volumes"])
      and yaml.safe_load(_ld_cm["data"]["run.yaml"]) == _ld_cfg
      and _ld_pod["spec"]["nodeSelector"][kaiload.ZONE_LABEL] == "zone-a"
      and _ld_pod["spec"]["nodeSelector"]["karpenter.sh/nodepool"] == "cpu-compute"
      and _ld_job["spec"]["activeDeadlineSeconds"] > 2 * 3600)
check("  labelled as the deployment's so teardown takes it, while its pod is not counted a tenant",
      all(_ld_job["metadata"]["labels"][k] == v for k, v in kaiplan.labels("tok").items())
      and _ld_cm["metadata"]["labels"] == _ld_job["metadata"]["labels"]
      and kaiplan.LABEL_DEPLOYMENT not in _ld_pod["metadata"].get("labels", {}))

# Every example fleet has a folder of scenarios, at the same path under load-tests/examples, and
# every scenario there is one its fleet can run: its tenants exist and its requests fit.
_ld_fleets = sorted(p.relative_to(_ld_root / "fleets/examples").with_suffix("")
                    for p in (_ld_root / "fleets/examples").glob("*/*.yaml"))
_ld_dirs = sorted(p.relative_to(_ld_root / "load-tests/examples")
                  for p in (_ld_root / "load-tests/examples").glob("*/*") if p.is_dir())
check("every example fleet has a load-test folder, and every folder is a fleet's",
      _ld_fleets and _ld_fleets == _ld_dirs, f"{_ld_fleets} vs {_ld_dirs}")
_ld_bad = []
for _ld_rel in _ld_dirs:
    _ld_fdoc = yaml.safe_load((_ld_root / "fleets/examples" / f"{_ld_rel}.yaml").read_text())
    _ld_pseudo = {"fleet": {"path": f"fleets/examples/{_ld_rel}.yaml"},
                  "tenants": {t["name"]: {"demand": {k: t[k] for k in ("rate_hz", "isl_tokens",
                                                                       "osl_tokens")},
                                          "max_model_len": t.get("max_model_len")}
                              for t in _ld_fdoc["tenants"]}}
    _ld_here = sorted(p.name for p in (_ld_root / "load-tests/examples" / _ld_rel).glob("*.yaml"))
    if not {"as-fleet.yaml", "demand-drift.yaml"} <= set(_ld_here):
        _ld_bad.append(f"{_ld_rel}: has {_ld_here}")
    for _ld_name in _ld_here:
        try:
            kaiload.demand(kaiload.load_scenario(_ld_root / "load-tests/examples" / _ld_rel
                                                 / _ld_name), _ld_pseudo)
        except kaiload.LoadError as exc:
            _ld_bad.append(f"{_ld_rel}/{_ld_name}: {exc}")
check("  each with an as-fleet and a demand-drift scenario its fleet can run", not _ld_bad,
      "; ".join(_ld_bad))

# The count, not just the word, and this line is the only place it appears. This is a flat module:
# an exception anywhere in it aborts the run, and the abort is silent -- no FAIL, no traceback,
# just truncated output and exit 1. Reaching this line at all is the real signal, but a count is
# what makes a truncated run visible to someone who greps `PASS` and sees a healthy-looking number.
# On 2026-09-14 that was 505 of 879, and the missing 374 read as a pass.
print(f"all {len(ran)} self-tests passed")
