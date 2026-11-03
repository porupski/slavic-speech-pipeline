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
# # Train frame — regression (chapter 4)
#
# **⚠️  SCAFFOLD — 2026-11-02. Not yet tested end to end.**
#
# This file is the regression twin of `41_train_frame_classification`. It is
# shipped for structural completeness so chapter 4 shows both heads. No frame
# regression target exists in the TARGETS registry yet (see
# `utils_frame_train.py :: _add_*_targets`), and the frame-training engine in
# `utils_frame_train.py` is wired for classification only. Running this
# notebook top-to-bottom today will stop at the `load_config` call with a hard
# error that points you here.
#
# When the first continuous frame-level target lands, port the regression
# pieces from chapter 3 (`utils_instance_train.py` already supports both
# heads) into `utils_frame_train.py`. The port plan sits at the bottom of
# this notebook under "What to wire in the engine".
#
# What already works in the scaffold (will run until `load_config`):
#   * GPU guard + heavy import
#   * `print_target_menu("regression")` — prints nothing yet (empty registry)
#   * `load_config` layering (classification + regression blocks in 4_config.json)
#
# What stops at `load_config`:
#   * `load_config` raises when `task_type="regression"` by design, with a
#     dated port-plan message. Remove that guard once the engine is ready.
#
# **Tutorial twin pattern.** Shared engine lives in `utils_frame_train.py`;
# every knob lives in `4_config.json`. For unattended full runs, write a
# `run_42_regression.py` once the engine is wired (mirror `run_41_classification.py`).

# %% [markdown]
# # Setup
#
# One visible block runs BEFORE the heavy engine imports:
#
# 1. Picks this notebook's TASK_TYPE.
# 2. Finds the chapter folder and `4_config.json`.
# 3. GPU guard — pins CUDA_VISIBLE_DEVICES before torch is imported.
#    Type `y` to arm the reserved GPU; anything else = CPU.

# %%
TASK_TYPE = "regression"   # this notebook's side of the twin pair

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
CONFIG_PATH = HERE / "4_config.json"

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
# `4_config.json`; this cell just lists what is on the menu for regression.
#
# **Scaffold note (2026-11-02):** no regression frame targets are registered
# yet. The call below will print an empty menu. Add a regression target with
# `_add_*_targets` once a continuous frame-level annotation exists — e.g.
# per-frame prominence score, per-frame pitch, per-frame energy derivative.

# %%
from utils_frame_train import print_target_menu

print_target_menu(TASK_TYPE)

# %% [markdown]
# # Config
#
# Everything tunable lives in **`4_config.json`** — you should never need to
# edit Python to change a run. Loader layers, in order: `Config` defaults →
# `shared` block → this task's block (`regression`) → the active run mode's
# overrides.
#
# **Scaffold stop point.** `load_config` raises a dated port-plan error
# because the engine still hard-blocks regression. See the error message for
# what to port from `utils_instance_train.py`. The `regression` block in
# `4_config.json` carries placeholder knobs mirroring chapter 3.

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
# records that carry a non-empty `label_key` sequence. Run-mode caps hit
# train/dev/test identically.

# %%
from utils_frame_train import load_and_cap_splits, print_rough_eta

mark("data prep")
train_records, dev_records, test_records = load_and_cap_splits(cfg)
print_rough_eta(len(train_records), len(dev_records), cfg)

# %% [markdown]
# ## Frame-rate guard + optional length cap
#
# Every frame record must declare `frame_rate_hz == 50`. When
# `enable_max_audio_seconds` is on, longer records are DROPPED (never
# truncated — truncation would desync the targets).

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
# ## Target normalization
#
# **Scaffold stub (2026-11-02) — port from chapter 3.**
#
# Regression targets are z-scored on TRAIN-only statistics so raw magnitudes
# do not blow MSE/MAE up and so Spearman does not collapse. The port template
# is `utils_instance_train.fit_normalizer` (instance-level, one scalar per
# record). For frame regression the normalizer must act per-element over the
# target sequence; masking rules must match the IGNORE_INDEX convention used
# in classification. Decode back to real units before metrics, saved
# predictions, and plots — so MSE/MAE read in natural units.

# %%
# normalizer = uft.fit_normalizer(cfg, train_records, dev_records, test_records)
# label2id, id2label = None, None  # classification-only; defined here so the engine cells stay twin-identical
raise NotImplementedError(
    "2026-11-02 scaffold: fit_normalizer not yet implemented in utils_frame_train. "
    "Port from utils_instance_train.fit_normalizer; adapt to per-frame targets."
)

# %% [markdown]
# # Training engine
#
# From here on every cell is a thin call into `utils_frame_train`. What
# `run_phase` wires together, in order:
#
# `prepare_dataset_dict` → `preprocess_function` (loads WAV, slices word
# span, aligns targets to the model's per-record output frame count) →
# `DataCollatorForFrame` (pads with IGNORE_INDEX; already handles
# `task_type="regression"` dtype=float32) → **`build_model`** (needs a
# `Wav2Vec2ForFrameRegression` + `Wav2Vec2BertForFrameRegression`; the
# instance-level `Wav2Vec2ForRegression` in chapter 3 is the port template)
# → **`make_regression_metrics`** (per-frame MSE / MAE / Spearman computed in
# real units) → `EpochCheckpointCallback` (per-epoch predictions + scatter /
# strip plot) → `best_epoch_of` (min for mse/mae, max for spearman).

# %% [markdown]
# ## Run directory

# %%
from utils_frame_train import make_run_dirs

# run_dir, model_dir, run_name = make_run_dirs(cfg, train_records, normalizer=normalizer)
run_dir, model_dir, run_name = make_run_dirs(cfg, train_records)

# %% [markdown]
# ## Feature extractor

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
    label2id=None, id2label=None, device=DEVICE,
    # normalizer=normalizer,   # port once fit_normalizer lands
)
mark("end phase 1")

# %% [markdown]
# ## Phase 2 — TRAIN + DEV → TEST (final)

# %%
from utils_frame_train import print_recalibrated_eta

print_recalibrated_eta(len(train_records), len(dev_records), cfg)

phase2_results, phase2_best = run_phase(
    phase_name="phase2_test",
    train_records=train_records + dev_records, eval_records=test_records,
    eval_split_name="TEST", save_best_model=True,
    cfg=cfg, run_dir=run_dir, model_dir=model_dir,
    feature_extractor=feature_extractor,
    label2id=None, id2label=None, device=DEVICE,
    # normalizer=normalizer,   # port once fit_normalizer lands
)
mark("end phase 2")

# %% [markdown]
# ## Run summary + spot check

# %%
from utils_frame_train import print_run_summary, spot_check

# print_run_summary(cfg, run_name, run_dir, model_dir, phase1_best, phase2_best, normalizer=normalizer)
print_run_summary(cfg, run_name, run_dir, model_dir, phase1_best, phase2_best)
spot_check(run_dir, phase2_best)

# %% [markdown]
# ## Scatter / strip plot (TEST, best epoch)
#
# **Scaffold stub (2026-11-02) — port + adapt.**
#
# For instance regression chapter 3 uses `plot_test_scatter` (gold-vs-pred
# scatter + distribution overlay). Frame regression needs a per-frame view:
# either a time-aligned gold-vs-pred strip (matches the classification strip
# plot) or a flattened scatter. Pick when porting; the strip is more useful
# for debugging mask/alignment bugs.

# %%
# from utils_frame_train import plot_test_frame_regression
# plot_test_frame_regression(run_dir, phase2_best, n_per_tier=cfg.n_examples_to_plot)
raise NotImplementedError(
    "2026-11-02 scaffold: plot_test_frame_regression not yet implemented. "
    "Mirror plot_test_example_predictions (strip layout) but encode continuous "
    "gold / pred values via colour intensity, not binary cell fill."
)

# %% [markdown]
# ## Stage timing

# %%
from utils_frame_train import print_stage_breakdown

mark("end script")
print_stage_breakdown()

# %% [markdown]
# ## What is next
#
# - Different target: change `regression.target` in `4_config.json`
#   (the menu is in the Targets cell above — empty until a regression target
#   is registered).
# - Classification targets (filled pauses, primary stress) live in the twin:
#   `41_train_frame_classification.ipynb`.

# %% [markdown]
# ## What to wire in the engine
#
# **Port plan — 2026-11-02.** The scaffold runs top-to-bottom today until
# `load_config` raises. To make 42 train end to end, extend
# `utils_frame_train.py` with the following, modelled on
# `utils_instance_train.py`:
#
# 1. **Remove the two hard guards** in `load_config` and `validate_config`
#    that reject `task_type != "classification"`. Replace with regression-
#    aware validation (`best_metric_regression` ∈ {`mse`, `mae`, `spearman`};
#    `loss_function` ∈ {`mse`, `l1`}; `normalize` ∈ {`zscore`, `none`}).
# 2. **Add Config fields** mirroring chapter 3: `best_metric_regression`,
#    `normalize`, `loss_function`, `label_scale`.
# 3. **Register at least one frame regression target** in TARGETS. Candidate
#    shapes: per-frame prominence score, per-frame log-pitch, per-frame
#    energy. Set `"task_type": "regression"` on the preset.
# 4. **Normalizer.** Port `fit_normalizer` from chapter 3. Adapt to per-frame
#    targets — fit over all TRAIN frames pooled, decode per batch.
# 5. **Model classes.** `Wav2Vec2ForFrameRegression` and
#    `Wav2Vec2BertForFrameRegression` — Linear(1) head over the encoder's
#    per-frame hidden states, masked MSE / L1 loss (reuse the IGNORE_INDEX
#    convention from the classifier; mark ignored frames with NaN or a mask).
# 6. **Metrics.** `compute_frame_regression_metrics` returns per-frame MSE,
#    MAE, Spearman computed in real units via the normalizer.
# 7. **Checkpoint callback.** Mirror `EpochCheckpointCallback` but write
#    regression predictions (gold / pred floats + mask) per epoch.
# 8. **Plot.** `plot_test_frame_regression` — gold / pred strip or scatter
#    on the best phase-2 epoch.
# 9. **`run_phase`.** Accept `normalizer` arg; dispatch model / loss /
#    metrics on `cfg.task_type`.
# 10. **CLI runner.** `run_42_regression.py` mirrors `run_41_classification.py`.
