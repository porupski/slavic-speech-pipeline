#!/usr/bin/env bash
# setup_env_cuda.sh — create the `ssp-cuda` mamba environment for slavic-speech-pipeline
# Targets CUDA 12.4 (compatible with driver 610 / CUDA runtime 13.x).
# Run from the repo root: bash setup_env_cuda.sh

set -euo pipefail

ENV_NAME="ssp-cuda"
PY_VERSION="3.11"

echo "→ Checking mamba is available..."
if ! command -v mamba &> /dev/null; then
    echo "❌ mamba not found."
    echo "   If you have micromamba, run: eval \"\$(micromamba shell hook --shell bash)\""
    exit 1
fi
mamba --version

if mamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "❌ Env '${ENV_NAME}' already exists."
    echo "   To recreate: mamba env remove -n ${ENV_NAME} && bash setup_env_cuda.sh"
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
echo "→ Installing PyTorch (CUDA 12.4 build)..."
mamba run -n "${ENV_NAME}" pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

echo
echo "→ Installing pip-only packages..."
mamba run -n "${ENV_NAME}" pip install \
    praatio transformers datasets accelerate jupytext

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
echo "   python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"