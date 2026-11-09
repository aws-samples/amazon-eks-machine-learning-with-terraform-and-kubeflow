#!/usr/bin/env python3
"""Choose a fleet's demand offline, before any GPU time is spent on it.

WHAT THIS IS FOR. `plan` answers "given this fleet file, what placements exist?". It does not answer
"what should `rate_hz` be?", and after 2026-09-09 that is the question in front of both fleets: the
prefix-caching defect (found in our own measurement runs) left fleet 2 measured infeasible at its stated demand and
fleet 1 unmeasured with one tenant almost certainly over its ceiling. Re-running `profile` at a
demand that cannot work spends 25 minutes of a leased g6e to learn arithmetic. This sweeps the demand
space on the laptop instead and names the candidates worth measuring.

WHAT IT COMPUTES, AND WHICH HALF IS ARITHMETIC. Two halves, and they are not equally trustworthy:

  - **Memory, which is arithmetic.** Envelopes, memory-floor TP degrees, share pairs and the densest
    feasible placement all come from `solver/` -- the same functions `plan` and `search` call, at the
    same `envelope_step_mib`. Nothing here re-derives them.
  - **Latency, which is a PREDICTION with a stated model and a stated calibration.** Every ITL below
    is marked `~`. The model is fitted per tenant-model from our own solo measurements on g6e.48xlarge,
    it is not a general predictor of anything, and its only job is to rank candidate demands so the
    GPU rounds start somewhere defensible. The method's baseline rule forbids treating a predicted
    latency as a result, and this tool does not produce results.

THE LATENCY MODEL, and why this shape. One decode step costs a fixed part -- reading the shard's
weights out of HBM, which does not depend on load -- plus a marginal part proportional to the token
work the engine is doing. So

    ITL(work) = a + b x work,   work = rate_hz x (isl_tokens + osl_tokens)  tok/s

`work` counts prefill AND decode because that is what the engine actually does once prefix caching
is off; the void measurements had `work = rate_hz x osl_tokens`, which is the whole defect in one
substitution. Sharding wider divides both terms by the degree, which holds for dense models on this
node (decode all-reduce over PCIe is ~0.3 ms against a ~35 ms step) and DOES NOT hold for the MoE
anchors -- GLM-4.5-Air at tp8 was measured worse than at tp4, so no degree above a measured one is
predicted for an MoE at all.

WHERE `a` AND `b` COME FROM, per model, all from our own measurement runs:

  - Two measurements at different `work` fit both: GLM-4.5-Air tp4 (54.891 ms with caching on at
    1024 tok/s, 73.536 ms with it off at 2048) and all four 0.6B rows.
  - One measurement plus a physical `a` for Qwen3-14B: 27.508 GiB/shard of bf16 weights read per step
    over an L40S's 864 GB/s is 34.19 ms, which leaves `b` from the single caching-on point. This one
    is checkable and checks out -- it reproduces the measured 49.060 ms to 0.5% -- which is the only
    reason a one-point fit is used at all.
  - Qwen3-30B-A3B is an MoE, so `a` is not computable: a step reads only the experts the batch
    touched. It gets one point instead -- the x1.34 caching-on-to-off ratio measured on the OTHER MoE
    at the same shape, rate and node -- and GLM-4.5-Air's own 49/51 split of fixed to marginal is
    transferred to divide it into (a, b). That transfer is the weakest step in this file and it is
    called out at the constant.

CO-RESIDENCY IS PRICED, because a solo prediction that just clears a target predicts a co-resident
miss. The factors are measured, from the void searches' interference tables, which survive as
decode-phase numbers: an anchor carrying one rider per shard cost 1.27x (30B-A3B, 22.0 -> 28.0 ms),
riders stacked three or four deep cost 1.6-2.0x. A candidate demand is only reported as landing if
the solo prediction times the role's factor still meets the target.

Offline in the strict sense: no cluster, no store writes, no network. It opens two stores
read-only -- the profile store, because the fleet loader resolves TP degrees from it, and tolerates
it being empty; and the hardware store, because the device's memory total is measured rather than
tabulated and there is no default to fall back on. If nothing has measured the card this fleet's
instance type carries, this refuses and says which figure is missing, the same way `plan` does: run
a mode that leases a node first. `--gpu-count`/`--gpu-product` bootstrap an instance type nobody has
leased, exactly as they do for `plan`.

    python3 tools/demand_sweep.py --fleet fleets/examples/g6e.48xlarge/qwen-anchor-riders.yaml
    python3 tools/demand_sweep.py --fleet fleets/examples/g6e.48xlarge/glm-air-anchor-riders.yaml --tenant reason
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from colocation.agent import fleetfile                                        # noqa: E402
from colocation.agent.hardware import UnknownHardware                         # noqa: E402
from colocation.agent.__main__ import AGENT_ENVELOPE_STEP_MIB                 # noqa: E402
from colocation.solver.baseline import (at_tp, candidate_tp_degrees, fleet_baseline,  # noqa: E402
                                        min_envelope_gib)
from colocation.solver.cpsat import Infeasible, PlacementSolver, Unproven      # noqa: E402
from colocation.store.hubfacts import HubFactsStore                           # noqa: E402
from colocation.store.hwfacts import Disagreement, HardwareIndex              # noqa: E402
from colocation.store.profile import ProfileStore                             # noqa: E402

LOCAL_STORE = Path.home() / ".colocation" / "store"

# --- the latency model's calibration, per model_id. See the docstring. ---
#
# `a` is the load-independent part of a decode step at tp=1, in ms; `b` is ms per (tok/s) of token
# work at tp=1. Both are divided by the TP degree when predicting a wider shard, which is why
# `tp_scales` exists: an MoE is not predicted above the degree it was measured at.
FIT = {
    # a, b fitted from two points at tp4 (36.24, 0.01821) and scaled back to tp1 for one table.
    "zai-org/GLM-4.5-Air-FP8": dict(a=144.96, b=0.07284, tp_scales=False, measured_tp=4,
                                    source="tp4 two-point fit: 54.891 ms @1024, 73.536 ms @2048"),
    # a physical (27.508 GiB / 864 GB/s), b from the one caching-on point. Reproduces it to 0.5%.
    "Qwen/Qwen3-14B": dict(a=34.19, b=0.01936, tp_scales=True, measured_tp=1,
                           source="a = weights/HBM bandwidth; b from 49.060 ms @768"),
    # MoE, one point, no computable `a`: a step reads only the experts the batch touched. The x1.34
    # caching-on-to-off ratio measured on GLM-4.5-Air -- the other MoE, same shape, same rate, same
    # node -- puts its caching-off ITL at 22.0 x 1.34 = 29.5 ms at 2048 tok/s, and GLM-4.5-Air's own
    # fit splits that 49% fixed / 51% marginal. Transferring the split is the weakest step in this
    # file; it survives only because the margin it has to explain is 51%, not 2%.
    "Qwen/Qwen3-30B-A3B-Base": dict(a=29.0, b=0.01464, tp_scales=False, measured_tp=2,
                                    source="22.0 ms x1.34 (GLM-4.5-Air's measured ratio), split 49/51"),
    # a, b fitted from route's two points; the four 0.6B rows agree to within 0.05 ms.
    "Qwen/Qwen3-0.6B": dict(a=2.3686, b=6.1e-6, tp_scales=True, measured_tp=1,
                            source="two-point fit: 2.369 ms @64, 2.394 ms @4160"),
}

# Measured interference, from the void searches' co-residency tables (decode-phase, still valid).
ROLE_FACTOR = {"anchor": 1.27, "rider": 2.04}


def work_tok_s(t) -> float:
    """Token work per second with real prefill. The substitution the defect got wrong."""
    return t.rate_hz * (t.isl_tokens + t.osl_tokens)


def predict_itl_ms(t, tp: int) -> tuple[float, str] | None:
    """Predicted solo ITL for this tenant at this degree, or None if the model refuses to guess."""
    fit = FIT.get(t.model_id)
    if fit is None:
        return None
    if not fit["tp_scales"] and int(tp) != fit["measured_tp"]:
        return None
    return fit["a"] / tp + fit["b"] / tp * work_tok_s(t), "a+b*work"


def role_of(t, tenants) -> str:
    """An anchor is a tenant whose memory forces sharding; everything else rides."""
    return "anchor" if t.model_id in ("zai-org/GLM-4.5-Air-FP8", "Qwen/Qwen3-30B-A3B-Base",
                                      "Qwen/Qwen3-14B") else "rider"


def predicted_minimal_tp(t, tenants, floor_tp: int, gpu_count: int) -> tuple:
    """The narrowest degree at or above the memory floor whose prediction meets the target.

    Returns (tp or None, note). None means no degree in range is predicted to meet it -- which is a
    reason not to spend a round, not a measurement that it cannot.
    """
    if t.slo is None or t.slo.itl_mean_ms is None:
        return floor_tp, "no ITL target"
    factor = ROLE_FACTOR[role_of(t, tenants)]
    for tp in candidate_tp_degrees(gpu_count):
        if tp < floor_tp:
            continue
        got = predict_itl_ms(t, tp)
        if got is None:
            return None, f"no prediction at tp{tp}"
        itl, _how = got
        if itl * factor <= t.slo.itl_mean_ms:
            return tp, f"~{itl:.1f} ms solo, ~{itl * factor:.1f} ms with x{factor} co-residency"
    return None, "no degree predicted to meet the target"


def cap_note(t) -> str:
    """Whether `max_num_seqs` can hold the decoders the target implies. A lower bound (§0)."""
    need = t.concurrency_at_target()
    if need is None:
        return ""
    cap = t.max_num_seqs or 64
    return f"cap {cap} vs {need:.1f} decoding at target ({100 * need / cap:.0f}%)"


def load(path: str, overrides: dict, *, gpu_count=None, gpu_product=None):
    """Load the fleet offline, with `rate_hz`/`osl_tokens` overridden per tenant."""
    index = HardwareIndex(LOCAL_STORE)
    pk = fleetfile.profile_key(path, index, gpu_count=gpu_count, gpu_product=gpu_product)
    store = ProfileStore(LOCAL_STORE, pk)
    # The same store the agent caches checkpoint manifests in, so a sweep asks the Hub nothing that a
    # `plan` on the same fleet has already asked it -- and prices the same weight bytes and the same
    # KV page the agent would, which is the only way the two can be compared at all.
    loaded = fleetfile.load(path, index=index, profiles=store,
                            hub=HubFactsStore(LOCAL_STORE),
                            gpu_count=gpu_count, gpu_product=gpu_product,
                            allow_unresolved_tp=True)
    fleet = loaded["fleet"]
    if overrides:
        tenants = tuple(replace(t, **overrides.get(t.name, {})) for t in fleet.tenants)
        fleet = replace(fleet, tenants=tenants)
    return fleet


def report(fleet, label: str) -> dict:
    node = fleet.node.node_spec
    gpu_count = len(node.gpu_ids)
    base = fleet_baseline(node, fleet.requests, envelope_step_mib=AGENT_ENVELOPE_STEP_MIB)
    print(f"\n=== {label} ===")
    print(f"{'tenant':<9} {'work':>7} {'floor':>5} {'pred':>4} {'env@pred':>9} {'pocket':>7}  note")
    total_floor, total_pred, unpredicted = 0, 0, []
    at_pred = []
    for t in fleet.tenants:
        tb = base.tenant(t.name)
        floor = tb.floor_tp
        total_floor += floor or 0
        tp, note = predicted_minimal_tp(t, fleet.tenants, floor or 1, gpu_count)
        if tp is None:
            unpredicted.append(t.name)
            env = pocket = float("nan")
        else:
            total_pred += tp
            req = at_tp(t.request, tp)
            at_pred.append(req)
            env = min_envelope_gib(req, tp)
            pocket = node.device_total_gib - env
        print(f"{t.name:<9} {work_tok_s(t):>7.0f} {str(floor):>5} {str(tp) if tp else '--':>4} "
              f"{env:>9.2f} {pocket:>7.2f}  {note}")
        if cap_note(t):
            print(f"{'':<9} {'':>7} {'':>5} {'':>4} {'':>9} {'':>7}  {cap_note(t)}")
    print(f"\nsum(memory floor tp) = {total_floor}      "
          f"sum(PREDICTED minimal tp) = {total_pred}"
          + (f"  + {', '.join(unpredicted)} unpredicted" if unpredicted else ""))
    print(f"can_coreside = {base.can_coreside}   "
          f"shareable pairs = {len(base.shareable)} of {len(base.pairs)}")

    used = None
    if unpredicted:
        print("no placement solved: a tenant with no predicted degree has no shard count to place, "
              "and quoting a saving against an incomplete baseline is the mistake §0 is about")
        return dict(floor=total_floor, predicted=total_pred, used=None, unpredicted=unpredicted)
    if total_pred > gpu_count:
        # The baseline itself does not fit. Worth saying loudly: co-location cannot rescue this,
        # because the disjoint arrangement it is scored against needs more GPUs than the node has.
        print(f"THE BASELINE DOES NOT FIT: {total_pred} shards on a {gpu_count}-GPU node. Nothing "
              "can be scored here -- the demand has to come down further, or a tenant leaves.")
        return dict(floor=total_floor, predicted=total_pred, used=None, unpredicted=unpredicted)
    # Solved at the PREDICTED degrees, which is the whole point: a placement solved at memory floors
    # while the baseline counts predicted degrees compares two different fleets.
    solver = PlacementSolver(node, tuple(at_pred), envelope_step_mib=AGENT_ENVELOPE_STEP_MIB)
    try:
        best = solver.solve(cap=1, max_seconds=10.0)[0]
        used = len({g for gpus in best.placement.assignment.values() for g in gpus})
        print(f"densest placement at the PREDICTED degrees: {used} GPUs   "
              + "  ".join(f"{n}:{','.join(str(g) for g in sorted(gs))}"
                          for n, gs in sorted(best.placement.assignment.items())))
        print(f"  baseline {total_pred} -> {used}: {total_pred - used} GPUs freed "
              f"({100 * (total_pred - used) / total_pred:.0f}%) IF the predictions hold")
    except Unproven as exc:
        # The sweep's whole job is "is this demand serveable at all", so the distinction matters
        # here as much as in `plan` (impl §14.2): a demand the arithmetic cannot place is not a
        # demand that has been ruled out, and this sweep's latency half is a prediction already.
        print(f"no placement at the predicted degrees, and it is NOT ruled out: {exc}")
    except Infeasible as exc:
        print(f"no placement at the predicted degrees: {exc}")
    return dict(floor=total_floor, predicted=total_pred, used=used, unpredicted=unpredicted)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="demand_sweep.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fleet", required=True)
    p.add_argument("--tenant", action="append", default=[],
                   help="tenant to sweep `rate_hz` for. Repeatable. Default: every tenant with an "
                        "ITL target whose prediction misses at its memory floor")
    p.add_argument("--rates", default="1.0,0.7,0.6,0.5,0.4",
                   help="multipliers applied to the swept tenants' stated rate_hz")
    p.add_argument("--set", action="append", default=[], metavar="TENANT.FIELD=VALUE",
                   help="a fixed override applied to every row, e.g. chat.rate_hz=1.5")
    p.add_argument("--gpu-count", type=int, default=None,
                   help="how many GPUs one node of this instance type has. Needed only to bootstrap "
                        "a type nobody has leased yet; see `plan --help`")
    p.add_argument("--gpu-product", default=None,
                   help="which GPU that type carries (l40s, a10g, h100), to price the sweep against "
                        "a card measured elsewhere. Only useful beside --gpu-count")
    args = p.parse_args(argv)

    fixed: dict = {}
    for item in args.set:
        lhs, _, value = item.partition("=")
        tenant, _, field = lhs.partition(".")
        fixed.setdefault(tenant, {})[field] = (float(value) if field.endswith("_hz")
                                               else int(value))

    try:
        base_fleet = load(args.fleet, fixed, gpu_count=args.gpu_count,
                          gpu_product=args.gpu_product)
    except (fleetfile.FleetFileError, UnknownHardware, Disagreement) as exc:
        # A missing device total is a refusal with an instruction in it, not a stack trace: this
        # tool exists to be run before anything is spent, so it has to say what to run instead.
        print(f"{args.fleet}: {exc}", file=sys.stderr)
        return 2
    report(base_fleet, f"{args.fleet} as written"
                       + (f", with {args.set}" if args.set else ""))

    swept = args.tenant or [t.name for t in base_fleet.tenants
                            if t.slo and t.slo.itl_mean_ms is not None]
    for mult in [float(x) for x in args.rates.split(",")]:
        if mult == 1.0:
            continue
        over = {k: dict(v) for k, v in fixed.items()}
        for name in swept:
            t = base_fleet.tenant(name)
            over.setdefault(name, {})["rate_hz"] = round(t.rate_hz * mult, 4)
        report(load(args.fleet, over, gpu_count=args.gpu_count, gpu_product=args.gpu_product),
               f"{', '.join(swept)} at x{mult:g} rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
