"""Telling what a command would actually run apart from what it merely mentions.

Shared by both hooks, because both got this wrong in the same way and a rule this subtle drifts when
it is written twice. `guard.py` uses `agent_mode` to decide whether a harness image is required and
`strip_heredocs` before it reads a command for anything at all; `after-run.py` uses `agent_mode` to
decide whether a round just finished.

Both hooks are invoked as `python3 "<plugin>/hooks/<name>.py"`, so this module's directory is already
on the import path.
"""

import os
import re

# Commands that only ever read. Naming a mode inside one of these is asking a question about a run,
# not starting one.
INSPECTORS = ("pgrep", "ps", "grep", "egrep", "fgrep", "rg", "echo", "printf", "cat", "tail", "head",
              "less", "awk", "sed", "wc", "test")

HEREDOC_START = re.compile(r"<<-?\s*([\"']?)([A-Za-z_][A-Za-z0-9_]*)\1")
# The only consumers for which a heredoc body is itself a command line.
SHELLS = re.compile(r"(?:^|[;|&(]\s*)(?:bash|sh|zsh|dash|ksh)\b")


def strip_heredocs(command):
    """The command without its heredoc bodies, which are data rather than commands.

    A commit message is the case that made this necessary: prose describing what the guard refuses
    was itself refused, because `git commit -F - <<'MSG'` puts the words `helm uninstall` on the same
    command line as far as a flat read is concerned. Every rule in `guard.py` reads the line it is
    given, so the fix belongs here rather than in each rule.

    A body fed to a shell is kept, since those lines really do run.
    """
    lines = command.splitlines()
    kept, i = [], 0
    while i < len(lines):
        line = lines[i]
        kept.append(line)
        i += 1
        delimiters = [d for _, d in HEREDOC_START.findall(line)]
        if not delimiters:
            continue
        body = []
        while i < len(lines) and delimiters:
            if lines[i].strip() in delimiters:
                delimiters.remove(lines[i].strip())
            else:
                body.append(lines[i])
            i += 1
        if SHELLS.search(line):
            kept.extend(body)
    return "\n".join(kept)


def agent_mode(command):
    """The mode this command would RUN, or None.

    Two things make a mention an invocation, and both are required. The module has to be executed,
    which in this repo is always `-m colocation.agent <mode>`; and the segment doing it must not be a
    command that only reads. `pgrep -f "colocation.agent profile"` asks whether a round is still
    going -- denying that one for naming no harness image is the guard obstructing exactly the thing
    it exists to protect, and it happened during a live profile on 2026-09-19. The same mention made
    the post-run hook report that a round had finished when nothing had run.

    Segments are read separately so an inspector in one cannot cover a real invocation in another,
    and heredoc bodies are dropped first so a mode named in a message is not a mode being run.
    """
    flat = " ".join(strip_heredocs(command).split())
    for segment in re.split(r"\|\||&&|[;|]", flat):
        tokens = segment.split()
        if not tokens:
            continue
        if os.path.basename(tokens[0].strip("\"'")) in INSPECTORS:
            continue
        found = re.search(r"-m\s+colocation\.agent\s+(\w+)", segment)
        if found:
            return found.group(1)
    return None


def kai_mode(command):
    """The `colocation.kai` mode this command would RUN, or None. `agent_mode`'s rules, its module."""
    flat = " ".join(strip_heredocs(command).split())
    for segment in re.split(r"\|\||&&|[;|]", flat):
        tokens = segment.split()
        if not tokens or os.path.basename(tokens[0].strip("\"'")) in INSPECTORS:
            continue
        found = re.search(r"-m\s+colocation\.kai\s+(\w+)", segment)
        if found:
            return found.group(1)
    return None
