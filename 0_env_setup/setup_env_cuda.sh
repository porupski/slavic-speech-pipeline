#!/usr/bin/env bash
# setup_env_cuda.sh — create the `ssp-cuda` mamba environment for slavic-speech-pipeline.
# Targets CUDA 12.4 (compatible with driver 550+).
#
# Usage:
#   bash 0_env_setup/setup_env_cuda.sh          # create the env if it doesn't exist
#   bash 0_env_setup/setup_env_cuda.sh --force  # remove and recreate
#
# This is a from-scratch installer with pinned versions. All conda-forge packages
# are pinned to match the CPU env (ssp_cpu.yaml). PyTorch is installed from the
# official pip CUDA 12.4 wheel index instead of conda-forge, so the GPU build is used.
#
# Once the GPU server is online and the env is stable, export a proper YAML:
#   mamba env export -n ssp-cuda --no-builds > 0_env_setup/ssp_cuda.yaml
# Then retire this script in favour of a YAML-based setup_env_cuda.sh (like the CPU one).

set -euo pipefail

ENV_NAME="ssp-cuda"
PY_VERSION="3.11"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

echo "→ Checking mamba is available..."
if ! command -v mamba &>/dev/null; then
    echo "❌ mamba not found on PATH."
    echo "   Install Miniforge:"
    echo "     curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh"
    echo "     bash miniforge.sh -b -p \"\${HOME}/miniforge3\""
    echo "     eval \"\$(\${HOME}/miniforge3/bin/mamba shell hook --shell bash)\""
    echo "   (Add the eval line to ~/.bashrc for persistence)"
    echo "   Or if you already have micromamba:"
    echo "     eval \"\$(micromamba shell hook --shell bash)\" && alias mamba=micromamba"
    exit 1
fi

if mamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    if [[ "${FORCE}" -eq 1 ]]; then
        echo "→ Removing existing '${ENV_NAME}' env (--force) ..."
        mamba env remove -n "${ENV_NAME}" -y
    else
        echo "❌ Env '${ENV_NAME}' already exists."
        echo "   To recreate: bash 0_env_setup/setup_env_cuda.sh --force"
        exit 1
    fi
fi

echo
echo "→ Creating env '${ENV_NAME}' with Python ${PY_VERSION}..."
mamba create -n "${ENV_NAME}" "python=${PY_VERSION}" -y

echo
echo "→ Installing scientific + audio stack from conda-forge (pinned)..."
# Versions pinned to match ssp_cpu.yaml for reproducibility.
mamba install -n "${ENV_NAME}" -c conda-forge -y \
    "numpy=2.4.6" \
    "pandas=3.0.3" \
    "scipy=1.17.1" \
    "librosa=0.11.0" \
    "pysoundfile=0.13.1" \
    "lxml=6.1.1" \
    "matplotlib=3.10.9" \
    "seaborn=0.13.2" \
    "scikit-learn=1.8.0" \
    "jupyter=1.1.1" \
    "ipykernel=7.2.0" \
    "tqdm=4.67.3" \
    "requests=2.34.2"

echo
echo "→ Installing PyTorch CUDA 12.4 build via pip..."
# cu124 index: https://download.pytorch.org/whl/cu124
# Latest available in the cu124 index as of env creation.
# Check available versions with:
#   pip index versions torch --index-url https://download.pytorch.org/whl/cu124
mamba run -n "${ENV_NAME}" pip install \
    "torch==2.6.0" \
    "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

echo
echo "→ Installing pip-only packages (pinned to match CPU env)..."
mamba run -n "${ENV_NAME}" pip install \
    "praatio==6.2.2" \
    "transformers==5.9.0" \
    "tokenizers==0.22.2" \
    "datasets==4.8.5" \
    "accelerate==1.13.0" \
    "huggingface_hub==1.16.4" \
    "jupytext==1.19.3"

echo
echo "→ Registering Jupyter kernel..."
mamba run -n "${ENV_NAME}" python -m ipykernel install --user \
    --name "${ENV_NAME}" \
    --display-name "Python (${ENV_NAME})"

echo
echo "✅ Done. To activate:"
echo "   mamba activate ${ENV_NAME}"
echo
echo "Verify GPU is visible:"
echo "   python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
