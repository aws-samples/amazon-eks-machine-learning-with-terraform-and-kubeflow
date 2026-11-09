from .baseline import (FleetBaseline, SharePair, TenantBaseline, WeightFloor, at_tp,
                       candidate_tp_degrees, capacity_steps, fleet_baseline, memory_floor_tp,
                       min_envelope_gib, share_pairs, weight_floor_tp)
from .cpsat import (Candidate, FeasibleSet, Infeasible, NodeSpec, Placement, PlacementSolver,
                    SolverBug, TenantRequest, Unproven, ENVELOPE_STEP_MIB_DEFAULT)
from .memory import (ACTIVATION_GIB_DEFAULT, HEADROOM_DEFAULT,
                     MPS_RESERVE_TP1_GIB, MPS_RESERVE_TPN_GIB, ModelMemory,
                     envelope_gib_for_kv_tokens, gpu_memory_utilization, kv_bytes_per_token,
                     kv_gib_for_envelope, kv_tokens_for_envelope, kv_tokens_intercept,
                     kv_tokens_per_mib)
from .topology import coresidency

# The tier vocabulary is NOT re-exported here. It used to be, through `identify.py`, which fitted a
# model's memory law to an engine's own report and was deleted with `onboard` (impl §14.4). It lives
# in `colocation/tiers.py`, above every package, and importing it from there is the only route.

__all__ = ["ACTIVATION_GIB_DEFAULT",
           "ENVELOPE_STEP_MIB_DEFAULT", "HEADROOM_DEFAULT",
           "Candidate", "FeasibleSet", "FleetBaseline",
           "Infeasible",
           "MPS_RESERVE_TP1_GIB", "MPS_RESERVE_TPN_GIB", "ModelMemory", "NodeSpec", "Placement",
           "PlacementSolver", "SharePair", "SolverBug", "TenantBaseline", "TenantRequest",
           "Unproven", "WeightFloor",
           "at_tp", "candidate_tp_degrees", "capacity_steps",
           "coresidency",
           "envelope_gib_for_kv_tokens",
           "fleet_baseline", "gpu_memory_utilization", "kv_bytes_per_token",
           "kv_gib_for_envelope",
           "kv_tokens_for_envelope", "kv_tokens_intercept", "kv_tokens_per_mib",
           "memory_floor_tp", "min_envelope_gib", "share_pairs", "weight_floor_tp"]
