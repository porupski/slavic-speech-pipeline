# Environment setup — details

Two environment variants: **CPU** (development, laptops) and **CUDA** (GPU servers). Both are cloned from a frozen `mamba env export` YAML — no per-package pip lines to keep in sync.

## Prerequisites

Miniforge or micromamba. If `mamba` isn't on PATH:

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

The YAML export is pending until the GPU server is next online. Until then, the last-known-good pre-YAML script lives at `0_env_setup/legacy/setup_env_cuda.sh` (kept locally, gitignored):

```bash
bash 0_env_setup/legacy/setup_env_cuda.sh
mamba activate ssp-cuda
```

That script pins **CUDA 12.4** — compatible with driver 550+. It uses:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Once the GPU server is up, re-export to a proper YAML:

```bash
mamba env export -n ssp-cuda --no-builds > 0_env_setup/ssp_cuda.yaml
```

Then create a matching `setup_env_cuda.sh` (mirror of the CPU one) and retire the legacy script.

Verify GPU is visible after activating:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

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
