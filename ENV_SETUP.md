# Environment setup

This project uses **mamba** (a fast drop-in replacement for conda) to manage its environment.

## Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) installed (ships with `mamba`), or any conda + mamba combination.
- Check: `mamba --version` should print something.

## Quick setup

```bash
bash setup_env.sh
```

This creates the `ssp` environment and installs everything for chapter 1.

To activate it later:

```bash
mamba activate ssp
```

## Manual setup (what the script does)

If you'd rather see what's happening step by step, or you want to install only some of it:

```bash
# 1. Create the env with Python 3.11
mamba create -n ssp python=3.11 -y

# 2. Activate
mamba activate ssp

# 3. Install scientific + audio stack from conda-forge
#    (these have C dependencies and are happiest from conda)
mamba install -n ssp -c conda-forge -y \
    numpy pandas scipy \
    pysoundfile librosa \
    lxml \
    jupyter ipykernel tqdm requests

# 4. Pip-install the rest (no C deps, plays nicely with pip)
pip install praatio

# 5. Register the kernel with Jupyter so notebooks can find it
python -m ipykernel install --user --name ssp --display-name "Python (ssp)"
```

## Adding chapter 3+ dependencies later

When you get to training (chapter 3), you'll need PyTorch and Hugging Face. Don't install these now — they're big and you don't need them for data prep.

When you do:

```bash
mamba activate ssp

# PyTorch — pick the right line for your hardware.
# CUDA 12.1:
mamba install -n ssp -c pytorch -c nvidia -y pytorch pytorch-cuda=12.1
# CPU only:
mamba install -n ssp -c pytorch -y pytorch cpuonly

# HF stack + sklearn + plotting
pip install transformers datasets accelerate scikit-learn matplotlib seaborn
```

## Removing the env (if you ever want to start fresh)

```bash
mamba deactivate
mamba env remove -n ssp
```

## Notes

- The env is named `ssp` (slavic-speech-pipeline). Change `setup_env.sh` if you want a different name.
- Conda-forge is set as the channel for the scientific stack because it has the best audio-stack builds. Pip is used for pure-Python packages that don't need to be conda-managed.
- The Jupyter kernel registration step matters: if you skip it, `jupyter notebook` won't see the env and you'll spend an hour confused. Don't skip it.
- All dependencies are intentionally pinned by minor version range in `requirements.txt` (used by collaborators who don't have mamba); the mamba flow above is the canonical path.
