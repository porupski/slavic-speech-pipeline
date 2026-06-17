# Chapter 0 — Environment setup

One-shot install of the Python env every other chapter runs in. Two variants — **CPU** for data prep and light development, **CUDA** for GPU training.

## Run

```bash
# CPU (laptops, data prep)
bash setup_env_cpu.sh
mamba activate ssp

# CUDA (GPU servers, training)
bash setup_env_cuda.sh
mamba activate ssp-cuda
```

Each script reads its matching `requirements_*.txt` and creates a mamba env named `ssp` or `ssp-cuda`.

## Prerequisites

`micromamba` or `miniforge`. If `mamba` is unavailable, activate micromamba first:

```bash
eval "$(micromamba shell hook --shell bash)"
alias mamba=micromamba
```

## Verify

After activation, check the install (and the GPU, for CUDA):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`ENV_SETUP.md` carries the full detail — pinned versions, CUDA 12.4 compatibility notes, the audio stack.
