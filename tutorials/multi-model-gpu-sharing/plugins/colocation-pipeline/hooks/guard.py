#!/usr/bin/env python3
"""PreToolUse guard for the co-location pipeline.

Every rule here was a real incident first. The skill states them in prose; this file is what
makes them hold when the prose is not being read. Two outcomes:

  deny  the command is never the right one, and there is a stated alternative.
  ask   the command changes what a run measures or what it risks, so the person who owns the
        requirement decides, not the agent.

A third outcome is a note with no decision: the command is fine, but something about the cluster
now needs attention that the command itself will not print.

Nothing here is about what a run costs. GPU time is not this plugin's concern. Cluster *operation*
is the agent's to drive — putting a daemon in, taking it out, handing the holder back and forth —
so the rules below aim at shapes that destroy something unrecoverable, not at ordinary lifecycle.
"""

import json
import os
import re
import subprocess
import sys

from agentmode import agent_mode, kai_mode, strip_heredocs

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_SCRIPT = os.path.join(HERE, os.pardir, "scripts", "harness-image.sh")


ADVICE = []


def respond(decision, reason):
    """Answer the hook and stop. `allow` here means 'this guard has no objection'."""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        },
        "systemMessage": reason,
    }))
    sys.exit(0)


def advise(note):
    """Say something about the command without deciding anything about it.

    Held until the end so a deny that comes later is what gets read, not a note about a command
    that is not going to run.
    """
    ADVICE.append(note)


def no_objection():
    if ADVICE:
        print(json.dumps({"systemMessage": "\n".join(ADVICE)}))
    sys.exit(0)


# --------------------------------------------------------------------------------------------
# the harness image
# --------------------------------------------------------------------------------------------

def assigned_value(command, name):
    """The value assigned to `name` in a shell command, or None.

    Stops at the first unquoted whitespace outside any $(...), so an assignment from a command
    substitution comes back whole.
    """
    marker = name + "="
    at = command.find(marker)
    if at < 0:
        return None
    i = at + len(marker)
    depth, quote, out = 0, None, []
    while i < len(command):
        ch = command[i]
        if quote:
            if ch == quote:
                quote = None
            out.append(ch)
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif command.startswith("$(", i):
            depth += 1
            out.append(ch)
        elif ch == ")" and depth:
            depth -= 1
            out.append(ch)
        elif ch.isspace() and not depth:
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def source_tag():
    """The tag the current build context hashes to, or None if it cannot be computed."""
    try:
        done = subprocess.run(["bash", IMAGE_SCRIPT, "--tag-only"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


IMAGE_FIX = (
    "Derive it from the source instead, in the same command:\n"
    "  IMG=$(bash \"$CLAUDE_PLUGIN_ROOT/scripts/harness-image.sh\" --ensure) || exit 1\n"
    "  COLOCATION_HARNESS_IMAGE=\"$IMG\" PYTHONPATH=src python3 -m colocation.agent ...\n"
    "That script builds, self-tests and pushes only when the tag is absent, so it is free on "
    "every run after the first."
)


def check_harness_image(command, mode):
    value = assigned_value(command, "COLOCATION_HARNESS_IMAGE")
    if value is None:
        value = os.environ.get("COLOCATION_HARNESS_IMAGE")
        if not value:
            respond("deny", (
                "`%s` needs a harness image built from the current source, and this command "
                "names none.\n\n%s" % (mode, IMAGE_FIX)
            ))
    if "harness-image.sh" in command:
        no_objection()
    stripped = value.strip("\"'")
    if stripped.startswith("$"):
        respond("deny", (
            "This passes %s as the harness image, and nothing in the command shows where it came "
            "from. A tag that pulls is not a tag that measures: colocation-harness:v5 sat in ECR "
            "pulling perfectly while predating the per-tenant tokenizer, so it could not parse "
            "what render.py writes.\n\n%s" % (stripped, IMAGE_FIX)
        ))
    tag = stripped.rsplit(":", 1)[-1] if ":" in stripped else ""
    want = source_tag()
    if want is None:
        no_objection()
    if tag != want:
        respond("deny", (
            "The harness image in this command is tagged `%s`, but the build context hashes to "
            "`%s`. Either the image predates the current harness source, or the tag was chosen by "
            "hand — both have shipped a measuring defect before, and a measuring defect voids "
            "complete results rather than failing loudly.\n\n%s" % (tag or stripped, want, IMAGE_FIX)
        ))
    no_objection()


# --------------------------------------------------------------------------------------------
# bash
# --------------------------------------------------------------------------------------------

TOKEN_READERS = re.compile(
    r"\b(cat|echo|printf|head|tail|less|more|strings|base64|xxd|od|read|awk|sed|cut|tr|grep|"
    r"export|env|curl|wget)\b")

# The namespace every release this pipeline creates lives in. An uninstall aimed anywhere else is
# aimed at somebody else's work.
WORKING_NAMESPACE = os.environ.get("COLOCATION_NAMESPACE") or "kubeflow-user-example-com"

HELM_REMOVE_VERBS = ("uninstall", "delete", "del", "un")
# helm flags whose value is the next token, so the value is not mistaken for a release name.
HELM_VALUE_FLAGS = {
    "-n", "--namespace", "--kube-context", "--kubeconfig", "--description", "--timeout",
    "--burst-limit", "--qps", "--registry-config", "--repository-cache", "--repository-config",
    "--kube-apiserver", "--kube-as-user", "--kube-as-group", "--kube-token", "--kube-ca-file",
    "--kube-tls-server-name", "--cascade",
}


def helm_removals(flat):
    """Every helm uninstall in a command line, as (release names, namespace) pairs.

    `namespace` is None when the command does not say one -- which is not the same as saying the
    right one, because helm would then fall back to whatever the kubeconfig's context defaults to.
    Each shell segment is read separately so `a && b` cannot hide one inside the other.
    """
    found = []
    for segment in re.split(r"\|\||&&|[;|]", flat):
        tokens = segment.split()
        at = next((i for i, t in enumerate(tokens) if t == "helm" or t.endswith("/helm")), None)
        if at is None:
            continue
        tokens = tokens[at + 1:]
        releases, namespace, verb_seen, i = [], None, False, 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("-"):
                flag, eq, inline = token.partition("=")
                if flag in ("-n", "--namespace"):
                    namespace = inline if eq else (tokens[i + 1] if i + 1 < len(tokens) else "")
                if not eq and flag in HELM_VALUE_FLAGS:
                    i += 1
            elif not verb_seen:
                if token not in HELM_REMOVE_VERBS:
                    break            # some other helm subcommand; nothing is being removed
                verb_seen = True
            else:
                releases.append(token)
            i += 1
        if verb_seen:
            found.append(([r.strip("\"'") for r in releases],
                          namespace.strip("\"'") if namespace is not None else None))
    return found


def check_helm_removals(flat):
    """Uninstalling is the agent's to do. Uninstalling *blind* is not.

    Taking the daemon out and putting it back is ordinary operation of this pipeline, so an
    uninstall that names what it removes, in the namespace this pipeline works in, is allowed. The
    three shapes below are refused because none of them can be read back from the command: a bare
    `helm uninstall` with no release, a name that arrives from a substitution or a glob (which is
    how one command removes every release in a namespace), and a namespace that was left to the
    kubeconfig or pointed somewhere else.
    """
    for releases, namespace in helm_removals(flat):
        if not releases:
            respond("deny", (
                "This `helm uninstall` names no release. Name the releases to remove; `helm list "
                "-n %s` says what is there." % WORKING_NAMESPACE
            ))
        derived = [r for r in releases if re.search(r"[*?]|\$|`", r)]
        if derived:
            respond("deny", (
                "This removes releases named by %s rather than by hand. One expansion is how every "
                "release in a namespace gets uninstalled at once, including a lease another run is "
                "measuring under. List the releases explicitly instead -- `helm list -n %s` first "
                "if the names are not known." % (", ".join(derived), WORKING_NAMESPACE)
            ))
        if namespace is None:
            respond("deny", (
                "This `helm uninstall` does not say a namespace, so helm uses whatever the "
                "kubeconfig's context happens to be pointing at. Pass `-n %s` explicitly."
                % WORKING_NAMESPACE
            ))
        if namespace != WORKING_NAMESPACE:
            respond("deny", (
                "This uninstalls in namespace `%s`. Everything this pipeline installs lives in "
                "`%s`, so a release found anywhere else belongs to somebody else."
                % (namespace, WORKING_NAMESPACE)
            ))
        advise(
            "Uninstalling in %s is fine to do here. Remember what it does to the node: a fleet's "
            "tenants are default priority and the holder is -1000, so removing the daemon leaves "
            "the node held by nothing. The full release is daemon, tenants, then `kubectl apply -f "
            "manifests/gpu-node-holder.yaml` -- and the holder must be Running before the node is "
            "left alone, because an unheld node is reclaimed in about 600 s." % WORKING_NAMESPACE
        )


def check_bash(command):
    # Heredoc bodies go first: every rule below reads a flat line, and a commit message that
    # describes what these rules refuse would otherwise be refused by them.
    command = strip_heredocs(command)
    flat = " ".join(command.split())

    check_helm_removals(flat)

    if re.search(r"\bhelm\s+(--\S+\s+)*upgrade\b", flat) and "mps-daemon" in flat:
        respond("deny", (
            "`helm upgrade` on the mps-daemon restarts the control daemon, which tears down every "
            "MPS server and therefore every running tenant. If the daemon's configuration has to "
            "change, that is an uninstall and install between runs, not under one."
        ))

    if re.search(r"\b(pkill|killall)\b", flat) and re.search(r"mps|nvidia", flat):
        respond("deny", (
            "Do not pattern-kill the MPS processes. `pkill -f nvidia-cuda-mps-server` matches the "
            "`bash -c` wrapper's own command line and kills the shell (exit 143). Find the stray "
            "with `ps -eo pid,comm` — which truncates at 15 characters, so match on the prefix — "
            "and kill it by PID."
        ))

    if ".hf_token" in flat or re.search(r"\bHF_TOKEN\b", flat):
        reading = TOKEN_READERS.search(flat) or "$(" in command or "`" in command
        setting = re.search(r"--set(-json|-string)?\b", flat)
        if reading or setting:
            respond("deny", (
                "The Hugging Face token is never read, echoed or interpolated — not into a "
                "variable, not through helm --set, not to check it. It reaches Kubernetes as a "
                "file and nothing else:\n"
                "  kubectl create secret generic hf-token -n $NS --from-file=token=$HOME/.hf_token\n"
                "and a fleet references it by `hf_secret_name` / `hf_secret_key`. If the file "
                "itself needs fixing, hand the user the command prefixed with `!`."
            ))

    if "--no-holder" in flat:
        respond("deny", (
            "`--no-holder` leaves the GPU node unheld. An unheld node is reclaimed in about 600 s, "
            "and the replacement's GPU UUIDs make every stored placement un-reusable — a stale "
            "UUID then fails silently as an unbounded engine, which is the one failure mode with "
            "no record of itself."
        ))

    if re.search(r"\bkubectl\s+delete\b", flat) and "gpu-node-holder" in flat:
        advise(
            "That deletes the thing holding the GPU node. Something else has to be holding it "
            "within about 600 s: a daemon going in, or a replacement holder applied straight "
            "after. Deleting the holder is only a step, never the last one."
        )

    if "--release-node" in flat:
        advise(
            "`--release-node` hands the leased node back. The release re-applies "
            "manifests/gpu-node-holder.yaml, so the node stays held -- confirm the holder is "
            "Running afterwards rather than assuming it landed."
        )

    mode = agent_mode(flat)

    for flag, why in (
        ("--reprofile", "overwrites usable rows the store already holds, so it discards evidence "
                        "rather than adding to it"),
        ("--allow-disjoint-only", "scores a fleet with nothing co-resident, which answers a "
                                  "different question than the one the pipeline asks"),
        ("--relax-targets-pct", "changes what the report is allowed to offer past a stated "
                                "ceiling, and a ceiling belongs to whoever wrote it"),
    ):
        if flag in flat:
            respond("ask", "`%s` %s. Confirm with the user before it runs." % (flag, why))

    if mode in ("profile", "search"):
        check_harness_image(command, mode)

    kai = kai_mode(flat)
    if kai in KAI_WHAT and not invoked_by_user(kai):
        respond("ask", (
            "`colocation.kai %s` %s, and it is a step the user starts: nothing in the pipeline "
            "deploys, load-tests or removes a result. Confirm with the user, or have them run "
            "/fleet-kai-%s." % (kai, KAI_WHAT[kai], kai)
        ))

    no_objection()


# --------------------------------------------------------------------------------------------
# KAI deploys, load tests and teardowns: the user's to start
# --------------------------------------------------------------------------------------------

TRANSCRIPT = None
KAI_COMMAND = re.compile(r"<command-name>/?(?:[\w.-]+:)?fleet-kai-(deploy|load|teardown)</command-name>")
# A load test is here for the same reason as a deploy, not for its cost: it sends synthetic traffic
# to a serving deployment, which slows whoever else it serves and is counted in that deployment's
# drift records as if it were real demand.
KAI_WHAT = {
    "deploy": "creates serving Deployments on the KAI node pool, and may launch a GPU node",
    "load": "sends a scenario's synthetic traffic to a serving deployment for hours, which its "
            "drift records will count as real demand",
    "teardown": "removes a serving deployment from the KAI node pool",
}


def last_prompt(path):
    """The text of the newest message the person typed, or None. Tool results are not prompts."""
    try:
        with open(path) as f:
            lines = f.readlines()[-400:]
    except (OSError, TypeError):
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        # `isMeta` is what a command or skill expands into, written after the prompt that ran it.
        if entry.get("type") != "user" or entry.get("isMeta"):
            continue
        content = (entry.get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                continue
            return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return None


def invoked_by_user(kai):
    """Whether the person's newest prompt is the command for this mode.

    A deploy authorizes the teardown that precedes its own retry, so /fleet-kai-deploy covers both.
    """
    found = KAI_COMMAND.search(last_prompt(TRANSCRIPT) or "")
    return bool(found) and (found.group(1) == kai or found.group(1) == "deploy")


# --------------------------------------------------------------------------------------------
# fleet files
# --------------------------------------------------------------------------------------------

REQUIREMENT_KEYS = re.compile(
    r"tolerance_pct|\bslo\b|itl_mean_ms|itl_p95_ms|ttft_p95_ms|kv_bytes_per_token")


def check_edit(tool_input):
    path = tool_input.get("file_path", "")
    # Any depth under a `fleets/` directory: the shipped fleets sit under
    # `fleets/examples/<instance type>/`, and a pattern that only matched the top level would let
    # every one of them be edited without this rule applying.
    if not re.search(r"fleets/(?:[^/]+/)*[^/]+\.ya?ml$", path):
        no_objection()
    text = "".join(str(tool_input.get(k, "")) for k in ("new_string", "content", "old_string"))
    for edit in tool_input.get("edits") or []:
        text += str(edit.get("new_string", "")) + str(edit.get("old_string", ""))
    if not REQUIREMENT_KEYS.search(text):
        no_objection()
    respond("ask", (
        "This edits a requirement in a fleet file. A target decides what a run certifies, a "
        "tolerance decides what the report may offer, and a stated kv_bytes_per_token overrides "
        "what the checkpoint says about itself — none of the three is the agent's to move. When "
        "demand has to change, change the rate: relaxing a ceiling also raises the KV floor, so it "
        "moves the memory geometry at the same time as the goalposts."
    ))


def main():
    try:
        event = json.load(sys.stdin)
    except (ValueError, OSError):
        no_objection()
    global TRANSCRIPT
    TRANSCRIPT = event.get("transcript_path")
    tool = event.get("tool_name", "")
    tool_input = event.get("tool_input") or {}
    if tool == "Bash":
        check_bash(tool_input.get("command") or "")
    elif tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        check_edit(tool_input)
    no_objection()


if __name__ == "__main__":
    main()
