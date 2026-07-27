# Environment setup — details

Two environment variants: **CPU** (development, laptops) and **CUDA** (GPU servers). The CPU env is cloned from a frozen `mamba env export` YAML. The CUDA env is a from-scratch installer with pinned versions until a YAML export from the GPU server is available.

## Prerequisites

You need `mamba` on PATH. The recommended route is **Miniforge** (bundles mamba + conda-forge by default):

```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh
bash miniforge.sh -b -p "${HOME}/miniforge3"
eval "$("${HOME}/miniforge3/bin/conda" shell.bash hook)"
```

Add the `eval` line to your `~/.bashrc` (or `~/.zshrc`) so mamba is available in every new shell. Then close and reopen the terminal.

If you already have **micromamba** but not mamba:

```bash
eval "$(micromamba shell hook --shell bash)"
alias mamba=micromamba
```

---

## CPU environment (`ssp`)

For local development and data prep. No GPU required.

```bash
bash 0_env_setup/setup_env_cpu.sh          # create, or bail if `ssp` already exists
bash 0_env_setup/setup_env_cpu.sh --force  # remove + recreate

mamba activate ssp
```

The script is a thin wrapper around:
```bash
mamba env create -n ssp -f 0_env_setup/ssp_cpu.yaml
```

`ssp_cpu.yaml` was exported with `--no-builds` from a known-good machine and is the single source of truth for the env.

---

## CUDA environment (`ssp-cuda`)

For GPU servers. Targets **CUDA 12.4** (compatible with driver 550+).

```bash
bash 0_env_setup/setup_env_cuda.sh          # create, or bail if `ssp-cuda` already exists
bash 0_env_setup/setup_env_cuda.sh --force  # remove + recreate

mamba activate ssp-cuda
```

Unlike the CPU env, this is a **from-scratch installer** (no YAML yet — that requires an export from a live GPU machine). The script installs conda-forge packages with versions pinned to match `ssp_cpu.yaml`, then installs PyTorch from the official CUDA 12.4 pip wheel index:

```bash
pip install torch==2.10.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is visible after activating:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Once the GPU server is online and the env is stable**, lock it down properly:

```bash
mamba env export -n ssp-cuda --no-builds > 0_env_setup/ssp_cuda.yaml
```

Then `setup_env_cuda.sh` becomes a thin YAML wrapper (same pattern as the CPU script) and this from-scratch approach is retired.

---

## Starting fresh

```bash
mamba env remove -n ssp        # CPU
mamba env remove -n ssp-cuda   # CUDA
```

Or `bash 0_env_setup/setup_env_cpu.sh --force` for the CPU env — one step.

---

## Exporting a new YAML

Whenever the env changes meaningfully and the new state should be shared:

```bash
mamba env export -n ssp      --no-builds > 0_env_setup/ssp_cpu.yaml
mamba env export -n ssp-cuda --no-builds > 0_env_setup/ssp_cuda.yaml
```

`--no-builds` drops per-machine build hashes (e.g. `numpy=1.26.4=py311hXXXX` becomes `numpy=1.26.4`), so the YAML replays on other Linux boxes without hitting "no such build" errors.

For torch specifically, mamba/conda-forge resolves the CPU / CUDA build cleanly in most cases. If a cross-machine replay ever fails on a `torch==X.Y.Z+cpu` line because pip can't find that build, hand-edit the exported YAML to add the appropriate PyTorch pip index above the `torch` line:

```yaml
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/cpu
    - torch==2.x.x+cpu
    - ...
```

## Notes

- `ssp` and `ssp-cuda` are intentionally separate envs — don't try to upgrade one into the other.
- If the server doesn't have `nvcc`, that's fine — PyTorch from the pip wheel bundles its own CUDA runtime.
- The Jupyter kernel registration is handled by mamba's env resolver via the `ipykernel` package in the YAML. If you don't see the "Python (ssp)" kernel in Jupyter, run once: `python -m ipykernel install --user --name ssp --display-name "Python (ssp)"`.
