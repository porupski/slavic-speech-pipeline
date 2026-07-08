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

`micromamba` or `miniforge`. If `mamba` isn't found, activate micromamba first:

```bash
eval "$(micromamba shell hook --shell bash)"
alias mamba=micromamba
```

## Verify

After activation, check the install (and the GPU, for CUDA):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CPU env: `torch.cuda.is_available()` returns `False`, which is expected. CUDA env on a GPU server: it should return `True` — if it doesn't, the CUDA build didn't take.

`ENV_SETUP.md` carries the full detail — pinned versions, CUDA 12.4 compatibility notes, the audio stack.

## Windows users

The scripts here are `bash` scripts. On Windows we recommend running them under **Git Bash** (comes with Git for Windows) rather than PowerShell or CMD. The rest of the pipeline is portable Python and runs the same way.

> The pipeline is developed and tested on Linux. The steps below are what a Windows user reported worked for them — treat them as an unverified starting point rather than official support. If any of it breaks, open an issue with what you tried.

Before running `setup_env_cpu.sh`, do this once:

1. **Enable long file paths** — mamba envs regularly hit Windows's 260-char path limit. Open PowerShell as Administrator and run:
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
                    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```
2. **Put miniforge on your Git Bash PATH.** Add this to `~/.bashrc`:
   ```bash
   export PATH="$HOME/miniforge3/bin:$HOME/miniforge3/Library/bin:$PATH"
   eval "$(mamba.exe shell hook --shell bash)"
   ```
3. **Run Git Bash as Administrator** for the setup script (env creation writes many files; UAC otherwise gets in the way).

If `setup_env_cpu.sh` still can't find mamba once you're in Git Bash, run the PATH export from step 2 in the current shell and try again.
