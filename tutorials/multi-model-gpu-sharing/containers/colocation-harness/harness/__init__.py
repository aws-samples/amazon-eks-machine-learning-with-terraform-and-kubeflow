"""Co-location measurement harness (impl §6).

Drives a fixed open-loop workload at a set of co-located vLLM tenants, waits for
steady state, measures one window, and writes a single observation record to the
store described in §8.

Entry point: `python -m harness --config run.yaml`.
"""

__all__ = ["batchmeans", "config", "controller", "load", "observation", "scrape", "workload"]
