# %% [markdown]
# # Frame-classification inference (chapter 6 · variant 63)
#
# Runs a **frame classification** model over a folder of audio, emits a
# per-file JSONL of events, optional TextGrids for visual QA in Praat, and a
# small plot of 3 randomly-sampled files. Same modular style as chapter 3:
# knobs live in `config.json`, the engine lives in **`utils_frame_infer.py`**,
# and each cell here imports just what it needs.
#
# Default model is `classla/wav2vecbert2-filledPause` — the Slavic filled-
# pause detector (w2v-BERT 2.0 backbone). Any HF repo or local checkpoint that
# implements `AutoModelForAudioFrameClassification` works; chapter-3
# utterance-level checkpoints will NOT work here (different head).
#
# Prefer hands-off? `run_63_frame_classification_inference.py` drives the same
# engine from the CLI — no interactive prompts, safe for tmux.

# %% [markdown]
# # Setup
#
# One visible block does three things **before** anything heavy loads:
#
# 1. Locates this chapter's folder and `config.json`.
# 2. Runs the **GPU guard** — same rule as chapter 3: `CUDA_VISIBLE_DEVICES`
#    must be set before torch's first CUDA touch, and `utils_frame_infer`
#    imports torch at module level.
# 3. Reserves the project GPU (from config) or falls back to CPU. Type `y` for
#    GPU, anything else = CPU.

# %%
import json
import os
import sys
from pathlib import Path

HERE = Path.cwd()
if HERE.name != "6_inference":
    candidate = HERE / "6_inference"
    if candidate.exists():
        HERE = candidate
sys.path.insert(0, str(HERE))          # utils_frame_infer.py lives next to this notebook
CONFIG_PATH = HERE / "config.json"

RESERVED_GPU = json.loads(CONFIG_PATH.read_text())["shared"].get("reserved_gpu", "2")
_env = os.environ.get("CONDA_DEFAULT_ENV", "")
print(f"conda env = {_env or '(none)'}")
print(f"⚠️  GPU mode will use PHYSICAL GPU {RESERVED_GPU} — and only GPU {RESERVED_GPU}.")
_choice = input(f"Use GPU {RESERVED_GPU}?  type 'y' for GPU {RESERVED_GPU}, anything else = CPU: ").strip().lower()
if _choice == "y":
    os.environ["CUDA_VISIBLE_DEVICES"] = RESERVED_GPU
    USE_CUDA = True
    print(f"🚀 GPU mode — CUDA_VISIBLE_DEVICES={RESERVED_GPU}  (physical GPU {RESERVED_GPU} → cuda:0 in-process)")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    USE_CUDA = False
    print("🖥️  CPU mode")

# %% [markdown]
# With the GPU pinned (or blanked), it's safe to import the engine. This is the
# heavy import — torch and transformers load here, and `HF_HOME` is redirected
# to the project-local `stock_models/` cache from inside the module.

# %%
import utils_frame_infer as ufi
from utils_frame_infer import (
    load_config, print_config_summary, resolve_device,
    run_inference, sample_plot_examples,
    N_PLOT_EXAMPLES,
)

print(f"PROJECT_ROOT = {ufi.PROJECT_ROOT}")
print(f"HF_HOME      = {os.environ['HF_HOME']}")

# %% [markdown]
# # Config
#
# Pick a run mode (`test` / `demo` / `full`) and a run name. The run name
# controls the output directory: `runs/{RUN_NAME}/`. Re-running with the same
# `RUN_NAME` **resumes** — audio files already covered in `inference.jsonl`
# are skipped. Delete the JSONL to force a full re-run.

# %%
RUN_MODE = "demo"                    # "test" | "demo" | "full"
RUN_NAME = "fp_bert2_demo"

cfg = load_config(CONFIG_PATH, run_mode=RUN_MODE)
# Honour the notebook's GPU guard decision above (overrides config's use_cuda).
cfg.use_cuda = USE_CUDA

device = resolve_device(cfg)
print_config_summary(cfg, device)

# %% [markdown]
# # Inference
#
# Walks `cfg.audio_dir`, chunks each file into 30 s non-overlapping segments,
# runs the model, collapses per-frame predictions into events, applies the
# postproc filters, and streams one JSONL line per file. TextGrids are written
# side-by-side under `runs/{RUN_NAME}/textgrids/` when `write_textgrids` is
# true.

# %%
result = run_inference(cfg, run_name=RUN_NAME)

# %% [markdown]
# # Sample plots
#
# Reload 3 random files from the JSONL and render `examples.png` in the run
# folder: waveform on top, red bars beneath marking detected filled-pause
# intervals — a quick visual check that the model is behaving.

# %%
sample_plot_examples(result["out_jsonl"], result["run_dir"],
                     n=N_PLOT_EXAMPLES, show=True)
