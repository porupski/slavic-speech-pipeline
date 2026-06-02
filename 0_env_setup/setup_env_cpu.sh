#!/usr/bin/env bash
# setup_env.sh — create the `ssp` mamba environment for slavic-speech-pipeline
# Run from the repo root: bash setup_env.sh

set -euo pipefail

ENV_NAME="ssp"
PY_VERSION="3.11"

echo "→ Checking mamba is available..."
if ! command -v mamba &> /dev/null; then
    echo "❌ mamba not found. Install miniforge first: https://github.com/conda-forge/miniforge"
    exit 1
fi
mamba --version

# Has the env already? If so, bail loudly rather than half-update it.
if mamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "❌ Env '${ENV_NAME}' already exists."
    echo "   To recreate: mamba env remove -n ${ENV_NAME} && bash setup_env.sh"
    exit 1
fi

echo
echo "→ Creating env '${ENV_NAME}' with Python ${PY_VERSION}..."
mamba create -n "${ENV_NAME}" python="${PY_VERSION}" -y

echo
echo "→ Installing scientific + audio stack from conda-forge..."
mamba install -n "${ENV_NAME}" -c conda-forge -y \
    numpy pandas scipy \
    pysoundfile librosa \
    lxml \
    matplotlib seaborn \
    scikit-learn \
    jupyter ipykernel tqdm requests

echo
echo "→ Installing PyTorch (CPU build by default — replace with CUDA build if needed)..."
# mamba install -n "${ENV_NAME}" -c pytorch -y \
#    pytorch cpuonly
mamba run -n "${ENV_NAME}" pip install torch --index-url https://download.pytorch.org/whl/cpu

echo
echo "→ Installing pip-only packages..."
mamba run -n "${ENV_NAME}" pip install praatio transformers datasets accelerate jupytext

echo
echo "→ Registering Jupyter kernel..."
mamba run -n "${ENV_NAME}" python -m ipykernel install --user \
    --name "${ENV_NAME}" \
    --display-name "Python (${ENV_NAME})"

echo
echo "✅ Done."
echo
echo "To activate:"
echo "   mamba activate ${ENV_NAME}"
echo
echo "To verify:"
echo "   python -c 'import numpy, pandas, soundfile, librosa, lxml, praatio, torch, transformers; print(\"ok\")'"
echo
echo "For chapter 3+ (training), see ENV_SETUP.md → 'Adding chapter 3+ dependencies'."
