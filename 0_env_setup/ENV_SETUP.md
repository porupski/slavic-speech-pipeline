# Environment setup

Two environment variants: **CPU** (development, laptops) and **CUDA** (GPU servers).

## Prerequisites

Miniforge or micromamba. If `mamba` isn't found, initialize it first:

```bash
eval "$(micromamba shell hook --shell bash)"
```

Then alias it if needed: `alias mamba=micromamba`

---

## CPU environment (`ssp`)

For local development and data prep. No GPU required.

```bash
bash setup_env_cpu.sh
mamba activate ssp
```

---

## CUDA environment (`ssp-cuda`)

For training on GPU servers. Targets **CUDA 12.4** (compatible with driver 610+).

```bash
bash setup_env_cuda.sh
mamba activate ssp-cuda
```

Verify GPU is visible after activating:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## What the scripts install

| Layer | How | Why |
|---|---|---|
| numpy, pandas, scipy, librosa, lxml, etc. | conda-forge | C deps, best builds here |
| torch (CPU) | pip + `whl/cpu` index | conda pytorch channel has resolver bugs |
| torch (CUDA 12.4) | pip + `whl/cu124` index | same reason |
| transformers, datasets, accelerate, praatio | pip | pure Python, no conda benefit |

---

## Manual step-by-step (CPU)

```bash
mamba create -n ssp python=3.11 -y
mamba activate ssp

mamba install -c conda-forge -y \
    numpy pandas scipy pysoundfile librosa lxml \
    matplotlib seaborn scikit-learn \
    jupyter ipykernel tqdm requests

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install praatio transformers datasets accelerate jupytext

python -m ipykernel install --user --name ssp --display-name "Python (ssp)"
```

## Manual step-by-step (CUDA)

```bash
mamba create -n ssp-cuda python=3.11 -y
mamba activate ssp-cuda

mamba install -c conda-forge -y \
    numpy pandas scipy pysoundfile librosa lxml \
    matplotlib seaborn scikit-learn \
    jupyter ipykernel tqdm requests

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install praatio transformers datasets accelerate jupytext

python -m ipykernel install --user --name ssp-cuda --display-name "Python (ssp-cuda)"
```

---

## Starting fresh

```bash
mamba env remove -n ssp        # CPU
mamba env remove -n ssp-cuda   # CUDA
```

---

## Notes

- `ssp` and `ssp-cuda` are intentionally separate envs — don't try to upgrade one into the other.
- If the server doesn't have `nvcc`, that's fine — PyTorch from the pip wheel bundles its own CUDA runtime.
- The Jupyter kernel registration step is not optional. Skip it and notebooks won't see the env.
- `requirements_cuda.txt` documents dependencies but can't express the `--index-url` for torch; always use the script or the manual steps for the torch install.