#!/usr/bin/env python3
"""PostToolUse: what has to be true after a measurement round, checked rather than remembered.

Runs only after a `profile` or `search` invocation. Three things it reports, all of which are
silent failures otherwise:

  the node is unheld    a search ends with its tenants still serving on purpose, and a holder that
                        cannot get every GPU stays Pending. A Pending holder holds nothing.
  degrees moved         `profile` settles minimal TP, so the baseline and every placement moved
                        with it, and nothing in the code makes `plan` run again.
  no canonical form     a search result addressed by GPU UUID is a result that expires with the
                        node. A search exports itself, so a summary with no export beside it is an
                        older result or a failed write -- either way, re-stating it costs nothing.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agentmode import agent_mode

NAMESPACE = os.environ.get("COLOCATION_NAMESPACE", "kubeflow-user-example-com")
STORE = Path(os.environ.get("COLOCATION_STORE") or Path.home() / ".colocation" / "store")


def kubectl(args):
    try:
        done = subprocess.run(["kubectl"] + args, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def holder_note():
    phase = kubectl(["get", "pod", "gpu-node-holder", "-n", NAMESPACE,
                     "-o", "jsonpath={.status.phase}"])
    if phase == "Running":
        return None
    daemon = kubectl(["get", "pods", "-n", NAMESPACE,
                      "-l", "app.kubernetes.io/name=mps-daemon",
                      "-o", "jsonpath={range .items[*]}{.status.phase} {end}"])
    if daemon and "Running" in daemon:
        return None
    if phase is None and daemon is None:
        return None  # no cluster to read; say nothing rather than guess
    if phase:
        return ("The GPU node is not safely held: gpu-node-holder is %s, not Running. It needs "
                "every GPU on the node, so a tenant pod still up leaves it Pending, and a Pending "
                "holder carries none of its own do-not-disrupt protection. Free the GPUs or say "
                "plainly in your report that the node is unheld." % phase)
    return ("Nothing is holding the GPU node: no gpu-node-holder pod, no running mps-daemon. It is "
            "reclaimed in about 600 s, and the replacement's GPU UUIDs make every stored placement "
            "un-reusable. Apply manifests/gpu-node-holder.yaml, or hand the user the command.")


def unexported(command):
    """A search summary in the store with no canonical form beside it, or None.

    Read off the store rather than off the run's own output, so a write that failed silently is still
    caught. The store root comes from the command's own `--store` when it names one, so a run pointed
    at a different store is not judged against the default one.

    Only summaries newer than the newest export count. A key whose fleet file has moved on cannot be
    exported at all -- its degrees are no longer settled, and the loader refuses, naming `profile` --
    so nagging about it on every later search would be asking for something that cannot be done. A
    result this run just produced is by definition the newest thing in the store, which is what makes
    that test the right one rather than a cutoff in days.
    """
    root = STORE
    named = re.search(r"--store[= ]+(\S+)", command)
    if named:
        root = Path(named.group(1).strip("\"'")).expanduser()
    keys = root / "observations"
    if not keys.is_dir():
        return None
    found, newest_export = [], 0.0
    for key in sorted(keys.iterdir()):
        for export in (key / "exports").glob("*.yaml"):
            newest_export = max(newest_export, export.stat().st_mtime)
        for summary in sorted((key / "searches").glob("*.json")):
            if not (key / "exports" / f"{summary.stem}.yaml").exists():
                found.append((summary.stat().st_mtime, f"{key.name}/{summary.stem}"))
    found = [(when, name) for when, name in found if when > newest_export]
    if not found:
        return None
    # Newest first, and only a few: a list long enough to scroll buries the one this run produced.
    missing = [name for _, name in sorted(found, reverse=True)[:3]]
    return ("No canonical form beside %s: %s. A search writes one as it finishes, so this is either "
            "an older result or a write that failed -- and the stored result addresses this node's "
            "GPUs by UUID and ordinal, both of which expire with the node. Export it before "
            "reporting it:\n\n"
            "    PYTHONPATH=src python3 -m colocation.agent export --fleet fleets/<name>.yaml\n\n"
            "Free and offline -- it re-states the same placement as slots, in absolute GiB, with the "
            "conditions carried by reference -- and it exports the baseline whether or not the "
            "search found anything." % ("that search" if len(missing) == 1 else "these searches",
                                        ", ".join(missing)))


def main():
    try:
        event = json.load(sys.stdin)
    except (ValueError, OSError):
        return
    command = (event.get("tool_input") or {}).get("command") or ""
    mode = agent_mode(command)
    if mode not in ("profile", "search"):
        return

    notes = []
    held = holder_note()
    if held:
        notes.append(held)
    if mode == "search":
        pending = unexported(command)
        if pending:
            notes.append(pending)
    if mode == "profile":
        notes.append("That was a `profile`, so run `plan` again before any `search`: the degrees it "
                     "settled move the baseline and every placement, and nothing in the code "
                     "sequences those two. If sum(minimal tp) now exceeds the node's GPU count "
                     "there is nothing to search — the arrangement being scored against does not "
                     "fit.")
    if not notes:
        return
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": "\n\n".join(notes),
        }
    }))


if __name__ == "__main__":
    main()
