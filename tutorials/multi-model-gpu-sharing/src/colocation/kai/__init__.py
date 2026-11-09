"""Deploying an exported result on the KAI scheduler's GPU fractions (`python -m colocation.kai`).

A separate step, run only when asked. It reads an export (`agent/export.py`) and the store it was
written from, and nothing in the pipeline calls it: a search measures under MPS on a leased node,
and this is one way of serving what it found. `plan.py` is the arithmetic and the manifests,
`cluster.py` the kubectl.
"""
