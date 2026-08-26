# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: ssp
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train frame — classification (chapter 4)
#
# Per-frame classification: feed an utterance (or a word sub-clip of one),
# predict a class per model frame. Token cross-entropy with
# `ignore_index=-100`. Shared engine lives in `utils_frame_train.py`; every
# knob lives in `config.json`. This notebook is a thin tutorial — each cell
# imports exactly the function it uses, so you can follow the pipeline step
# by step. For unattended full runs use `run_41_classification.py` in tmux.

# %% [markdown]
# # Setup
#
# One visible block runs BEFORE the heavy engine imports:
#
# 1. Picks this notebook's TASK_TYPE.
# 2. Finds the chapter folder and `config.json`.
# 3. GPU guard — pins CUDA_VISIBLE_DEVICES before torch is imported.
#    Type `y` to arm the reserved GPU; anything else = CPU.

# %%
TASK_TYPE = "classification"

import json
import os
import sys
from pathlib import Path

HERE = Path.cwd()
if HERE.name != "4_frame_models":
    candidate = HERE / "4_frame_models"
    if candidate.exists():
        HERE = candidate
sys.path.insert(0, str(HERE))
CONFIG_PATH = HERE / "config.json"

RESERVED_GPU = json.loads(CONFIG_PATH.read_text())["shared"].get("reserved_gpu", "2")
_env = os.environ.get("CONDA_DEFAULT_ENV", "")
print(f"conda env = {_env or '(none)'}")
print(f"⚠️  GPU mode will use PHYSICAL GPU {RESERVED_GPU} — and only GPU {RESERVED_GPU}.")
_choice = input(f"Use GPU {RESERVED_GPU}?  type 'y' for GPU {RESERVED_GPU}, anything else = CPU: ").strip().lower()
if _choice == "y":
    os.environ["CUDA_VISIBLE_DEVICES"] = RESERVED_GPU
    USE_CUDA = True
    print(f"🚀 GPU mode — CUDA_VISIBLE_DEVICES={RESERVED_GPU}")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    USE_CUDA = False
    print("🖥️  CPU mode")

# %% [markdown]
# Heavy import — torch + transformers + datasets all load here; `HF_HOME` is
# redirected to the project-local `stock_models/` cache inside the module.

# %%
import utils_frame_train as uft
from utils_frame_train import mark, print_project_info

mark("literal start")
print_project_info()

# %% [markdown]
# # Targets
#
# Frame-level TARGETS live in `utils_frame_train.py`. Pick by name in
# `config.json`; this cell just lists what is on the menu.

# %%
from utils_frame_train import print_target_menu

print_target_menu(TASK_TYPE)

# %% [markdown]
# # Config
#
# Everything tunable lives in **`config.json`** — you should never need to
# edit Python to change a run. Loader layers, in order: `Config` defaults →
# `shared` block → this task's block → the active run mode's overrides.
#
# Run modes:
# - `test` — tiny random model, a few dozen records; proves the plumbing only.
# - `demo` — real model, capped data (caps hit train/dev/test identically); ~1 h.
# - `full` — caps off, whole dataset. Full runs belong in the **py runner**.
#
# `RUN_MODE = None` takes the mode from `config.json`. Override with
# `"test" | "demo" | "full"` for a one-off.

# %%
from utils_frame_train import load_config, validate_config, resolve_device, print_config_summary

RUN_MODE = None

cfg, raw_config = load_config(CONFIG_PATH, task_type=TASK_TYPE, run_mode=RUN_MODE)
validate_config(cfg)
DEVICE = resolve_device(cfg, USE_CUDA)
print("✅ config valid\n")
print_config_summary(cfg, DEVICE)

# %% [markdown]
# # Data
#
# ## Load JSONL, filter, cap
#
# `load_and_cap_splits` reads every pooled JSONL once per split, keeping only
# records that carry a non-empty `label_key` sequence and (when
# `exclude_multistress=True`) skipping rows tagged
# `metadata.multistress=True`. Run-mode caps hit train/dev/test identically.

# %%
from utils_frame_train import load_and_cap_splits, print_rough_eta

mark("data prep")
train_records, dev_records, test_records = load_and_cap_splits(cfg)
print_rough_eta(len(train_records), len(dev_records), cfg)

# %% [markdown]
# ## Frame-rate guard + optional length cap
#
# Every frame record must declare `frame_rate_hz == 50` (source labels at any
# other rate are a hard error). When `enable_max_audio_seconds` is on, longer
# records are DROPPED (never truncated — truncation would desync the labels).

# %%
from utils_frame_train import validate_frame_rate, drop_long_records

for _split, _recs in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
    validate_frame_rate(_recs, cfg.required_frame_rate_hz)
print(f"✅ all records declare frame_rate_hz={cfg.required_frame_rate_hz}")

if cfg.enable_max_audio_seconds:
    train_records, n_tr = drop_long_records(train_records, cfg.max_audio_seconds)
    dev_records,   n_dv = drop_long_records(dev_records,   cfg.max_audio_seconds)
    test_records,  n_te = drop_long_records(test_records,  cfg.max_audio_seconds)
    print(f"dropped >{cfg.max_audio_seconds}s: train={n_tr} dev={n_dv} test={n_te}")

# %% [markdown]
# ## Labels & mappings
#
# Frame labels must all appear in `cfg.label_order`. Counts are FRAMES
# (tokens), not records — glance at the imbalance here; a steep positive/
# negative ratio is the usual culprit when macro-F1 lags.

# %%
from utils_frame_train import build_label_maps

label2id, id2label, num_labels = build_label_maps(cfg, train_records, dev_records, test_records)

# %% [markdown]
# # Training engine
#
# From here on every cell is a thin call into `utils_frame_train`. What
# `run_phase` wires together, in order:
#
# `prepare_dataset_dict` → `preprocess_function` (loads WAV, slices word
# span, aligns labels to the model's per-record output frame count) →
# `DataCollatorForFrame` (pads with IGNORE_INDEX) → `build_model` (dispatches
# on model family: Wav2Vec2 or Wav2Vec2Bert) → `compute_frame_metrics`
# (macro-F1, accuracy, positive-class F1) → `EpochCheckpointCallback`
# (per-epoch predictions + strip plot) → `best_epoch_of`.

# %% [markdown]
# ## Run directory
#
# One timestamped pair per run: `runs/<name>/` collects logs; `models/<name>/`
# receives the best model. The effective config is snapshotted into the run
# dir so every result stays reproducible.

# %%
from utils_frame_train import make_run_dirs

run_dir, model_dir, run_name = make_run_dirs(cfg, train_records)

# %% [markdown]
# ## Feature extractor
#
# `load_feature_extractor` forces `return_attention_mask=True`. wav2vec2-base
# ships with it off; without the mask the encoder mean-pools over padding,
# which degrades every downstream head.

# %%
from utils_frame_train import load_feature_extractor

feature_extractor = load_feature_extractor(cfg)

# %% [markdown]
# ## Phase 1 — TRAIN → DEV (development)
#
# Iterate hyperparameters here.

# %%
from utils_frame_train import run_phase

mark("model prep")
phase1_results, phase1_best = run_phase(
    phase_name="phase1_dev",
    train_records=train_records, eval_records=dev_records,
    eval_split_name="DEV", save_best_model=False,
    cfg=cfg, run_dir=run_dir, model_dir=model_dir,
    feature_extractor=feature_extractor,
    label2id=label2id, id2label=id2label, device=DEVICE,
)
mark("end phase 1")

# %% [markdown]
# ## Phase 2 — TRAIN + DEV → TEST (final)
#
# Same engine, final configuration. Only the best epoch's weights land in
# `models/<run>/best_model/`.

# %%
from utils_frame_train import print_recalibrated_eta

print_recalibrated_eta(len(train_records), len(dev_records), cfg)

phase2_results, phase2_best = run_phase(
    phase_name="phase2_test",
    train_records=train_records + dev_records, eval_records=test_records,
    eval_split_name="TEST", save_best_model=True,
    cfg=cfg, run_dir=run_dir, model_dir=model_dir,
    feature_extractor=feature_extractor,
    label2id=label2id, id2label=id2label, device=DEVICE,
)
mark("end phase 2")

# %% [markdown]
# ## Run summary + spot check

# %%
from utils_frame_train import print_run_summary, spot_check

print_run_summary(cfg, run_name, run_dir, model_dir, phase1_best, phase2_best)
spot_check(run_dir, phase2_best)

# %% [markdown]
# ## Stage timing

# %%
from utils_frame_train import print_stage_breakdown

mark("end script")
print_stage_breakdown()

# %% [markdown]
# ## What is next
#
# - Happy with a demo? Flip `run_mode` to `"full"` in `config.json` and hand
#   the job to `run_41_classification.py` in tmux — full runs do not belong
#   in a notebook kernel.
# - Different target: change `classification.target` in `config.json`.
# - Regression targets: deferred to `42_train_frame_regression` (future).
