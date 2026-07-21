#!/usr/bin/env bash
#
# ML Ops Desktop bootstrap for the "Build Your Own SRE Agents with Claude"
# workshop. Runs during CloudFormation UserData execution as the ubuntu user
# after the parent repo has been cloned to
#   /home/ubuntu/amazon-eks-machine-learning-with-terraform-and-kubeflow
#
# Kept out of CFN UserData because UserData is capped at 16,384 bytes.
#
# Idempotent: safe to re-run.

set -uo pipefail

REPO_ROOT="/home/ubuntu/amazon-eks-machine-learning-with-terraform-and-kubeflow"
SRE_DIR="${REPO_ROOT}/sre-agent-workshop"
DEST_WORKSHOP="/home/ubuntu/sre-agent-workshop"
DEST_AGENT="/home/ubuntu/sre-agent"

echo "[sre-bootstrap] starting at $(date -u +%FT%TZ)"

# ─── Copy lab scripts into expected paths ───────────────────────────────────
# Participants BUILD the agent themselves with Claude Code — we do NOT ship or
# copy any agent Python code onto the desktop. The only thing we deliver is the
# lab-scenario scripts (the fault injectors), which are environment tooling the
# participant runs but does not author.
#
# Content pages reference:
#   ~/sre-agent-workshop/lab/scenarios/module-N-.../*.sh   (delivered here)
#   ~/sre-agent/...                                        (participant-created)

if [[ ! -d "${SRE_DIR}" ]]; then
    echo "[sre-bootstrap] WARNING: ${SRE_DIR} does not exist; the sibling repo may not include SRE workshop code yet."
    exit 0
fi

# ~/sre-agent/ is the participant's own workspace — create it empty (with a
# reports/ subdir) but do NOT populate it with any pre-written agent code.
mkdir -p "${DEST_AGENT}/reports"

# Deliver only the lab scenarios to ~/sre-agent-workshop/lab/.
mkdir -p "${DEST_WORKSHOP}/lab"
if [[ -d "${SRE_DIR}/lab" ]]; then
    cp -rn "${SRE_DIR}/lab/." "${DEST_WORKSHOP}/lab/"
fi

# ─── Python dependencies for the SRE agent ──────────────────────────────────
# The desktop's .bashrc runs `conda activate` at startup, so participants land
# in the miniconda `base` environment. Install into `base` so the SDK is
# importable from the interactive shell without extra activation.
CONDA_SH="/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate base
    pip install --quiet --upgrade \
        'claude-agent-sdk>=0.2.100,<0.3.0' \
        'mcp>=1.0.0' \
        'anthropic>=0.40.0' \
        'boto3>=1.34.0' \
        'pyyaml>=6.0' \
        'python-frontmatter>=1.0.0' \
      || echo "[sre-bootstrap] WARNING: pip install into conda base failed."
    conda deactivate
else
    # Fallback: no conda on the desktop. Install to user-site.
    python3 -m pip install --user --upgrade --quiet \
        'claude-agent-sdk>=0.2.100,<0.3.0' \
        'mcp>=1.0.0' \
        'anthropic>=0.40.0' \
        'boto3>=1.34.0' \
        'pyyaml>=6.0' \
        'python-frontmatter>=1.0.0' \
      || echo "[sre-bootstrap] WARNING: pip install failed."
fi

# ─── uv / uvx (used by the Claude Agent SDK to spawn MCP servers) ───────────
if ! command -v uvx >/dev/null 2>&1; then
    if ! [[ -x "${HOME}/.local/bin/uvx" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh \
          || echo "[sre-bootstrap] WARNING: uv install failed — the setup page walks participants through a manual install."
    fi
fi

# Make sure ~/.local/bin is on PATH for future shells.
if ! grep -q 'HOME/.local/bin' "${HOME}/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"
fi

# ─── Ownership ──────────────────────────────────────────────────────────────
chown -R ubuntu:ubuntu "${DEST_WORKSHOP}" "${DEST_AGENT}" 2>/dev/null || true

echo "[sre-bootstrap] complete at $(date -u +%FT%TZ)"
