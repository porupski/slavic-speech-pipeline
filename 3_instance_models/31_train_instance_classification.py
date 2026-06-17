# %% [markdown]
# # Train instance — classification (chapter 3)
#
# **Tutorial twin.** Trains a Wav2Vec2 classifier on utterance instances —
# gender, filled-pause presence/count, and the ROG classification targets. Its
# sibling `32_train_instance_regression` is the same pipeline with a regression
# head; the shared engine lives in **`utils_instance_train.py`**, all knobs live
# in **`config.json`**, and each cell below imports exactly the functions it
# uses, so you can follow the pipeline step by step.
#
# Prefer it hands-off? The py runner (`run_31_classification.py`) drives the same
# engine from the command line — that's the tool for full, unattended runs. The
# original self-contained notebook is preserved in `legacy/` if you want the
# whole engine inline.

# %% [markdown]
# # Setup
#
# One visible block does three things **before** anything heavy loads:
#
# 1. Picks this notebook's `TASK_TYPE` (the only difference between the twins' setup cells).
# 2. Finds the chapter folder and `config.json`.
# 3. Runs the **GPU guard** — the one piece of engine logic that *cannot* live in
#    `utils_instance_train.py`: the utils module imports torch, and
#    `CUDA_VISIBLE_DEVICES` must be pinned before torch's first CUDA touch.
#    Type `y` to arm the reserved GPU; anything else runs on CPU.

# %%
TASK_TYPE = "classification"   # this notebook's side of the twin pair

# ── GPU GUARD — must run BEFORE `utils_instance_train` is imported ────────────
# CUDA_VISIBLE_DEVICES only takes effect if set BEFORE torch's first CUDA call,
# and the utils module imports torch at the top — so this stays a visible,
# input-gated block here rather than living in utils. GPU 2 below is reserved
# for this project; we NEVER touch another GPU. No auto-arming: you must type
# 'y' to use the GPU. ENTER (or anything else) = CPU.
import json
import os
import sys
from pathlib import Path

HERE = Path.cwd()
if HERE.name != "3_instance_models":
    candidate = HERE / "3_instance_models"
    if candidate.exists():
        HERE = candidate
sys.path.insert(0, str(HERE))          # utils_instance_train.py lives next to this notebook
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
# heavy import of the session — torch, transformers, datasets all load here, and
# `HF_HOME` is redirected to the project-local `stock_models/` cache inside the
# module. Every later cell imports just the functions it uses from
# `utils_instance_train`, so each step shows you exactly where its machinery
# lives — if you want to change what a cell does, that named function is the
# place to look.

# %%
import utils_instance_train as uit   # heavy import: pulls in torch + transformers (HF_HOME is handled inside)
from utils_instance_train import mark

mark("literal start")
print(f"PROJECT_ROOT = {uit.PROJECT_ROOT}")
print(f"HF_HOME      = {os.environ['HF_HOME']}")

# %% [markdown]
# # Targets
#
# Training targets are **presets** in the `TARGETS` registry inside
# `utils_instance_train.py` — each one names a JSONL, a `label_key`, a task type,
# and (for classification) a canonical `label_order`. You pick one by name in
# `config.json`; this cell just lists what's on the menu for this notebook's
# task type.

# %%
from utils_instance_train import available_targets

print(f"{TASK_TYPE} targets:")
for t in available_targets(TASK_TYPE):
    print(f"   {t}")

# %% [markdown]
# # Config
#
# Everything tunable lives in **`config.json`** — you should never need to edit
# the Python to change a run. The loader layers, in order: `Config` defaults →
# the `shared` block → this task's block → the active run mode's overrides.
#
# **Run modes** (`run_mode` in config.json, or the `RUN_MODE` override below):
#
# - `test` — tiny random model, a few dozen records; proves the plumbing only.
# - `demo` — real model, capped data (the caps hit train/dev/**test** identically); a tangible number in ~1–2 h.
# - `full` — caps off, whole corpus. Full runs belong in the **py runner** (tmux, unattended), not the notebook.
#
# The commented override spot is for quick one-off experiments; anything you
# want to keep, promote into `config.json`.

# %%
from utils_instance_train import load_config, validate_config, resolve_device, print_config_summary

RUN_MODE = None   # None → take run_mode from config.json; or override here: "test" | "demo" | "full"

cfg, raw_config = load_config(CONFIG_PATH, task_type=TASK_TYPE, run_mode=RUN_MODE)

# ── experiment overrides (optional) ────────────────────────────────────────────
# For one-off experiments, tinker here instead of editing config.json, e.g.:
#   cfg.num_epochs = 5
#   cfg.target = "parlaspeech_rs_fp_present"; uit.resolve_target(cfg)   # re-resolve after changing target!

validate_config(cfg)
DEVICE = resolve_device(cfg, USE_CUDA)
print("✅ config valid\n")
print_config_summary(cfg, DEVICE)

# %% [markdown]
# # Data
#
# ## Load JSONL, filter, cap
#
# `load_and_cap_splits` reads the target's JSONL once per split, keeping only
# records that carry a non-null `label_key` and dropping anything longer than
# `max_duration_s` (long clips waste batch padding and can OOM the GPU). The run
# mode's caps are then applied **identically to train/dev/test** — a capped run
# never silently evaluates on a full-size TEST. The rough ETA is a seeded guess;
# it recalibrates from real throughput after phase 1.

# %%
from utils_instance_train import load_and_cap_splits, print_rough_eta

mark("data prep")
train_records, dev_records, test_records = load_and_cap_splits(cfg)
print_rough_eta(len(train_records), len(dev_records), cfg)

# %% [markdown]
# ## Labels & mappings
#
# Classification labels need a **canonical order**: `label_order` fixes which
# class gets which index (presets with `label_order: None` build it from the data
# union). `build_label_maps` constructs `label2id`/`id2label`, hard-fails on any
# label the order doesn't know, and prints the per-split class distribution —
# glance at it; a steep imbalance here is the usual culprit when macro-F1 lags.

# %%
from utils_instance_train import build_label_maps

label2id, id2label, num_labels = build_label_maps(cfg, train_records, dev_records, test_records)
normalizer = None   # regression-only; defined here so the engine cells below stay twin-identical

# %% [markdown]
# # Training engine
#
# From here on, every cell is a thin call into `utils_instance_train`. The pieces
# `run_phase` wires together, in order:
#
# `prepare_dataset_dict` (records → items with labels + provenance) →
# `preprocess_function` (WAV → input values) → `DataCollatorForInstance` (pads
# batches and threads the **attention mask** through) → `build_model` (HF audio-
# classification head on wav2vec2-base, CNN encoder frozen) →
# `compute_classification_metrics` (macro-F1, accuracy, Spearman) →
# `EpochCheckpointCallback` (per-epoch predictions + confusion matrices) →
# `best_epoch_of` (max `best_metric_classification`).

# %% [markdown]
# # Run
#
# ## Run directory
#
# One timestamped directory pair per run: `runs/<name>/` collects per-epoch logs
# and artifacts, `models/<name>/` receives the best model. The *effective* config
# (after target resolution and mode overrides) is snapshotted into the run dir,
# so every result stays reproducible.

# %%
from utils_instance_train import make_run_dirs

run_dir, model_dir, run_name = make_run_dirs(cfg, train_records, normalizer=normalizer)

# %% [markdown]
# ## Feature extractor
#
# `load_feature_extractor` forces `return_attention_mask=True`. wav2vec2-base
# ships with it **off**; without the mask the model mean-pools over batch
# padding, which collapses predictions toward a constant. The fix is forced
# inside utils.

# %%
from utils_instance_train import load_feature_extractor

feature_extractor = load_feature_extractor(cfg)

# %% [markdown]
# ## Phase 1 — TRAIN → DEV (development)
#
# `run_phase` is the heart of the engine: build datasets → preprocess audio →
# train with a per-epoch checkpoint callback (metrics + predictions + plots for
# every epoch) → pick the best epoch → flush the GPU. Phase 1 trains on TRAIN and
# evaluates on DEV — this is where you'd iterate on hyperparameters.

# %%
from utils_instance_train import run_phase

mark("model prep")
phase1_results, phase1_best = run_phase(
    phase_name="phase1_dev",
    train_records=train_records, eval_records=dev_records,
    eval_split_name="DEV", save_best_model=False,
    cfg=cfg, run_dir=run_dir, model_dir=model_dir,
    feature_extractor=feature_extractor,
    label2id=label2id, id2label=id2label, device=DEVICE,
    normalizer=normalizer,
)
mark("end phase 1")

# %% [markdown]
# ## Phase 2 — TRAIN + DEV → TEST (final)
#
# Same engine, final configuration: train on TRAIN∪DEV, evaluate on the untouched
# TEST split, and save the best model (weights + feature extractor + that epoch's
# artifacts) into `models/<run>/best_model/`. The ETA line re-estimates phase 2
# from phase 1's measured records-per-second.

# %%
from utils_instance_train import print_recalibrated_eta

print_recalibrated_eta(len(train_records), len(dev_records), cfg)

phase2_results, phase2_best = run_phase(
    phase_name="phase2_test",
    train_records=train_records + dev_records, eval_records=test_records,
    eval_split_name="TEST", save_best_model=True,
    cfg=cfg, run_dir=run_dir, model_dir=model_dir,
    feature_extractor=feature_extractor,
    label2id=label2id, id2label=id2label, device=DEVICE,
    normalizer=normalizer,
)
mark("end phase 2")

# %% [markdown]
# ## Run summary
#
# The headline numbers: best DEV epoch (phase 1) and best TEST epoch (phase 2),
# plus where everything was written.

# %%
from utils_instance_train import print_run_summary

print_run_summary(cfg, run_name, run_dir, model_dir, phase1_best, phase2_best, normalizer=normalizer)

# %% [markdown]
# ## Confusion matrix (TEST, best epoch)
#
# The best phase-2 epoch on TEST, stacked: raw counts on top, row-normalized
# percentages below (each row sums to 100 — read it as "of the true X, what did
# the model call them?").

# %%
from utils_instance_train import plot_test_confusion

plot_test_confusion(run_dir, phase2_best, label2id, cfg.label_order)

# %% [markdown]
# ## Inference spot-check
#
# Five random TEST examples from the best epoch's saved predictions — a quick
# gut-check that the numbers above correspond to sane per-instance behavior.

# %%
from utils_instance_train import spot_check

spot_check(run_dir, phase2_best, cfg.task_type)

# %% [markdown]
# ## Stage timing
#
# Where the wall-clock went, stage by stage. Partial-run safe — it prints
# whatever marks exist.

# %%
from utils_instance_train import print_stage_breakdown

mark("end script")
print_stage_breakdown()

# %% [markdown]
# ## What's next
#
# - Happy with a demo? Flip `run_mode` to `"full"` in `config.json` and hand it
#   to the py runner — full runs don't belong in a notebook kernel.
# - Different target: change `classification.target` in `config.json`
#   (the menu is in the Targets cell above).
# - Regression targets (age, sentiment logit) live in the twin:
#   `32_train_instance_regression.ipynb`.
