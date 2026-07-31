#!/usr/bin/env bash
# setup_env_cpu.sh — clone the `ssp` (CPU) mamba environment from the frozen YAML.
#
# Usage:
#   bash 0_env_setup/setup_env_cpu.sh          # create the env if it doesn't exist
#   bash 0_env_setup/setup_env_cpu.sh --force  # remove and recreate
#
# The single source of truth for the env is `ssp_cpu.yaml` next to this script,
# exported from a known-good machine. No requirements.txt, no per-package pip
# install lines — mamba resolves the whole thing from the YAML.

set -euo pipefail

ENV_NAME="ssp"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML="${HERE}/ssp_cpu.yaml"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

if ! command -v mamba &>/dev/null; then
    echo "❌ mamba not found on PATH."
    echo "   Install Miniforge (bundles mamba):"
    echo "     curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh"
    echo "     bash miniforge.sh -b -p \"\${HOME}/miniforge3\""
    echo "     source \"\${HOME}/miniforge3/etc/profile.d/conda.sh\""
    echo "     source \"\${HOME}/miniforge3/etc/profile.d/mamba.sh\""
    echo "   (Add the two source lines to ~/.bashrc for persistence.)"
    echo "   If you have micromamba, an alias works: alias mamba=micromamba"
    exit 1
fi

if [[ ! -f "${YAML}" ]]; then
    echo "❌ ${YAML} not found."
    exit 1
fi

if mamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ "${FORCE}" -eq 1 ]]; then
        echo "→ removing existing '${ENV_NAME}' env (--force) …"
        mamba env remove -n "${ENV_NAME}" -y
    else
        echo "❌ env '${ENV_NAME}' already exists."
        echo "   To recreate: bash 0_env_setup/setup_env_cpu.sh --force"
        exit 1
    fi
fi

echo "→ creating '${ENV_NAME}' from ${YAML} …"
mamba env create -n "${ENV_NAME}" -f "${YAML}"

echo
echo "✅ done. To activate:"
echo "   mamba activate ${ENV_NAME}"
