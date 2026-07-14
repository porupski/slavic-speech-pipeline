# Chapter 0 — Environment setup

One command per environment, driven by a frozen YAML. The YAML is exported from a known-good machine and lives next to this file — no requirements chatter, no per-package hand-holding, no drift.

## Run

**CPU** (data prep, chapter 1, light development):

```bash
bash 0_env_setup/setup_env_cpu.sh
mamba activate ssp
```

Add `--force` to remove and recreate an existing `ssp` env.

**CUDA** (GPU training): the `ssp_cuda.yaml` export is pending. The last-known-good `setup_env_cuda.sh` lives in `legacy/` (gitignored, but present locally) — use it when needed. A frozen CUDA YAML will replace it as soon as the GPU server is next online.

## Prerequisites

`micromamba` or `miniforge`. If `mamba` isn't on your PATH:

```bash
eval "$(micromamba shell hook --shell bash)"
alias mamba=micromamba
```

## Verify

After activation:

```bash
python -c "import torch, datasets, soundfile; print(torch.__version__, datasets.__version__)"
```

CPU: `torch.cuda.is_available()` returns `False` — expected. CUDA (when set up): should return `True`; if not, the CUDA build didn't take.

## Exporting a new YAML

When the env changes meaningfully and needs to be shared:

```bash
mamba env export -n ssp      --no-builds > 0_env_setup/ssp_cpu.yaml
mamba env export -n ssp-cuda --no-builds > 0_env_setup/ssp_cuda.yaml  # when it exists
```

`--no-builds` drops per-machine build hashes so the YAML replays on other Linux boxes without "no such build" errors.

## Windows users

The scripts here are `bash` scripts. On Windows we recommend running them under **Git Bash** (comes with Git for Windows) rather than PowerShell or CMD. The rest of the pipeline is portable Python and runs the same way.

> Developed and tested on Linux. The steps below are what a Windows user reported worked for them — unverified starting point rather than official support.

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

## `legacy/`

Holds the pre-YAML pair of `setup_env_*.sh` + `requirements_*.txt` files. Gitignored — kept locally as a fallback for the CUDA env until its YAML export lands. Not the intended install path anymore.
