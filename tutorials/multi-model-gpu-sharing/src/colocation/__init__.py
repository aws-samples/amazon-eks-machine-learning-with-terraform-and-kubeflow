"""Agent-side components for GPU co-location search (impl §§8-11).

Pure standard library throughout, except the solver, which needs OR-Tools. The store and the
memory arithmetic run on the ML Ops desktop and on the solver's critical path -- CPU-only
and deterministic (impl §9) -- so nothing here pulls in a numerical stack. Nothing here fits
anything either: impl §9 is prediction-free, which is why a numerical stack was never needed.
"""
