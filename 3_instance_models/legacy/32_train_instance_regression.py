# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (ssp-cuda)
#     language: python
#     name: ssp-cuda
# ---

# %% [markdown]
# # Train instance — regression (chapter 3)
#
# One trainer for **instance-level regression** on Wav2Vec2-base. Twin of
# `31_train_classification.ipynb` (gender + filled-pause): the run engine
# (`run_phase`, the two-phase loop, run dirs, the stage-timing harness) is
# **identical** across the two — only the task-specific cells differ. That
# redundancy collapses into a shared utils module in phase E.
#
# **Two-phase training:** phase 1 trains TRAIN→DEV every epoch (no model saved);
# phase 2 retrains TRAIN∪DEV→TEST and saves the best epoch's model.
#
# **Run tiers** — one knob, `RUN_MODE` (see the Run-mode cell below):
# - **test** — tiny random model, a handful of records, 1 epoch. Proves the
#   plumbing end-to-end; produces no real result.
# - **demo** — real model, all three splits capped (20k/4k/4k), 2 epochs. A fast,
#   tangible result.
# - **full** — caps off, whole corpus. You and the GPU.
#
# **Regression-only knobs:** `normalize` (z-score the target on TRAIN stats,
# invert before metrics) and `loss_function` (`mse` | `l1`).

# %% [markdown]
# # Setup

# %%
import time

# ── Stage timing ───────────────────────────────────────────────────────────────
# Identical harness across 31/32 (lifts to utils in phase E). mark() stamps a
# milestone; the final cell prints a per-stage breakdown. Stdlib-only, cheap,
# partial-run safe (prints whatever marks exist).
STAGE_TIMES: dict[str, float] = {}

def mark(stage: str) -> None:
    STAGE_TIMES[stage] = time.time()

def fmt_mmss(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    return f"{s // 60:02d}:{s % 60:02d}"

def print_stage_breakdown(times: dict[str, float]) -> None:
    items = list(times.items())
    if not items:
        print("no timing recorded")
        return
    width = max(len(k) for k, _ in items)
    print("stage timing (delta from previous mark)")
    print("-" * (width + 11))
    prev = items[0][1]
    for name, t in items:
        print(f"  {name:<{width}}  {fmt_mmss(t - prev)}")
        prev = t
    print("-" * (width + 11))
    print(f"  {'TOTAL':<{width}}  {fmt_mmss(items[-1][1] - items[0][1])}")

mark("literal start")

import os
import sys
from pathlib import Path

# Find PROJECT_ROOT via utils_dataprep (chapter 1 already does this).
HERE = Path.cwd()
if HERE.name != "3_instance_models":
    candidate = HERE / "3_instance_models"
    if candidate.exists():
        HERE = candidate
sys.path.insert(0, str(HERE.parent / "1_data_prep"))

import utils_dataprep as udp
PROJECT_ROOT = udp.PROJECT_ROOT

os.environ["HF_HOME"] = str(PROJECT_ROOT / "stock_models")
# Cache HF pretrained models in a project-local folder (gitignored) instead of
# ~/.cache/huggingface, so the download survives across machines / reclones.
# MUST be set BEFORE any `from transformers import ...` — HF reads HF_HOME at
# import time and caches the resolved path internally.

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"Chapter dir  = {HERE}")
print(f"HF_HOME      = {os.environ['HF_HOME']}")

# ── GPU GUARD — MUST run before torch is imported (next cell) ─────────────────
# CUDA_VISIBLE_DEVICES only takes effect if set BEFORE the first torch.cuda call.
# GPU 2 is reserved for this project; we NEVER touch another GPU. No auto-arming:
# you must type 'y' to use the GPU. ENTER (or anything else) = CPU.
RESERVED_GPU = "2"
_env = os.environ.get("CONDA_DEFAULT_ENV", "")
print(f"\nconda env = {_env or '(none)'}")
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
# Standard third-party imports.

# %%
import gc
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import torch
import torch.nn as nn
from datasets import Dataset
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    AutoConfig, AutoFeatureExtractor, Trainer, TrainerCallback, TrainingArguments,
    Wav2Vec2Model, Wav2Vec2PreTrainedModel,
)

plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# # Targets
#
# What combinations of *(dataset, label)* this trainer can handle. Pick one via
# `Config.target`; the resolver fills `jsonl_path`, `label_key`, `task_type`,
# `label_order`. To add a target, add an entry here — `Config.target` is the only
# downstream knob.

# %%
TARGETS: dict = {
    # ROG has no regression target yet — ParlaSpeech targets generated below.
    # ---- ParlaSpeech utterance_instance (generated per lang below) ------
}


def _add_parlaspeech_targets(targets: dict, langs=("hr", "rs", "pl", "cz")) -> None:
    """Regression utterance_instance targets per ParlaSpeech lang.
    (gender + filled-pause classification targets live in 31_classification.)"""
    for l in langs:
        path = f"data/processed_jsonl/parlaspeech_{l}_utterance_instance.jsonl"
        targets[f"parlaspeech_{l}_sentiment"] = {
            "jsonl_path": path, "label_key": "sentiment_logit",
            "task_type": "regression", "label_order": None,
        }
        targets[f"parlaspeech_{l}_age"] = {
            "jsonl_path": path, "label_key": "speaker_age",
            "task_type": "regression", "label_order": None,
        }


_add_parlaspeech_targets(TARGETS)


def resolve_target(cfg, targets: dict) -> None:
    """Overwrite jsonl_path/label_key/task_type/label_order from the picked
    preset. Mutates cfg in place. Raises if cfg.target isn't a known key."""
    if cfg.target not in targets:
        raise ValueError(
            f"Config.target={cfg.target!r} not in TARGETS. Known: {sorted(targets)}"
        )
    t = targets[cfg.target]
    cfg.jsonl_path  = t["jsonl_path"]
    cfg.label_key   = t["label_key"]
    cfg.task_type   = t["task_type"]
    cfg.label_order = t["label_order"]   # may be None — built from data later


print(f"available targets: {sorted(TARGETS)}")

# %% [markdown]
# # Run mode
#
# One knob decides how much work runs. `RUN_MODE` picks a tier; the base `Config`
# below holds the real/full defaults, and `MODES` lists each tier's overrides on
# top of it. `apply_mode` layers the active tier in. This is the one place to flip
# between "does it even run" and "train for real".
#
# - **test** — plumbing only. Tiny random model, a handful of records, 1 epoch,
#   writes under `runs/test` + `models/test`. Answers *"does it run end-to-end?"*
# - **demo** — real model + real task, **all three splits capped** (train/dev/test
#   = 20k/4k/4k), 2 epochs. A fast, semi-working model with a tangible curve.
# - **full** — every cap off, real everything.
#
# `DEMO_SAMPLING` only matters when pooling >1 language: `proportional` keeps the
# corpus balance (plain random sample across the pool); `balanced` draws ~equally
# per language. (Chapter 3 targets are single-language, so it's inert here until
# multi-lang pooling lands — kept for parity with the frame trainer / phase-E utils.)

# %%
RUN_MODE = "demo"               # "test" | "demo" | "full"
DEMO_SAMPLING = "proportional"  # "proportional" | "balanced" — only when pooling >1 lang

# Each entry overrides the base (full) Config. Reading all three side by side
# tells you exactly what each tier changes; everything unlisted stays at its full
# default. (This block is a good candidate to lift to utils in phase E.)
MODES: dict = {
    "test": {
        "model_name": "hf-internal-testing/tiny-random-wav2vec2",
        "cap_train": 64, "cap_dev": 16, "cap_test": 16,
        "num_epochs": 1, "batch_size": 2, "grad_accum": 1,
        "runs_dir": "runs/test", "models_dir": "models/test",
    },
    "demo": {
        "cap_train": 20_000, "cap_dev": 4_000, "cap_test": 4_000,
        "num_epochs": 2,
    },
    "full": {
        "cap_train": None, "cap_dev": None, "cap_test": None,
    },
}


def apply_mode(cfg, overrides: dict) -> None:
    """Layer a mode's overrides onto the base (full) Config, in place. Every key
    must name a real Config field — a typo'd knob is a hard error, not a silent
    no-op."""
    valid = {f.name for f in fields(cfg)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"mode overrides name unknown Config fields: {sorted(unknown)}")
    for k, v in overrides.items():
        setattr(cfg, k, v)


def cap_split(records: list[dict], n, seed: int, sampling: str = "proportional") -> list[dict]:
    """Down-sample `records` to `n`, or return them unchanged when `n is None` or
    the split is already small enough. Applied identically to train/dev/test.
    JSONL is speaker-grouped, so we always shuffle before slicing.

    `sampling` only bites when >1 language is pooled:
      - proportional: plain random sample across the pool (true to corpus balance).
      - balanced: round-robin per language -> ~equal counts (draws whatever's left
        once a smaller corpus is exhausted)."""
    if n is None or len(records) <= n:
        return records
    rng = random.Random(seed)
    if sampling == "proportional":
        out = records[:]
        rng.shuffle(out)
        return out[:n]
    if sampling == "balanced":
        by_lang: dict = defaultdict(list)
        for r in records:
            by_lang[r["dataset"]].append(r)
        for recs in by_lang.values():
            rng.shuffle(recs)
        langs = sorted(by_lang)
        out: list = []
        idx = {l: 0 for l in langs}
        while len(out) < n:
            progressed = False
            for l in langs:
                if idx[l] < len(by_lang[l]):
                    out.append(by_lang[l][idx[l]]); idx[l] += 1; progressed = True
                    if len(out) >= n:
                        break
            if not progressed:
                break
        return out
    raise ValueError(f"DEMO_SAMPLING must be 'proportional' or 'balanced', got {sampling!r}")


# %% [markdown]
# # Config
#
# All knobs are here.
#
# - `task_type`: regression-only here (`31_classification` handles classification).
# - `label_key`: which key inside `labels` to train on (e.g. `speaker_age`,
#   `sentiment_logit`).
# - `normalize`: **regression-only.** `"none"` trains on raw labels; `"zscore"`
#   standardizes the target using **TRAIN-only** mean/std and inverts before all
#   metrics + saved predictions, so reported MSE/MAE/scatter stay in real units.
#   (Essential for raw-magnitude targets like age; harmless for small ones.)
# - `label_scale`: optional class→float map; if None, labels must already be numeric.
# - `model_name`: `facebook/wav2vec2-base` (~95M). Test mode swaps in a tiny random model.
# - `use_cuda`: honored from the GPU guard above; never touches a GPU other than 2.
#
# **Output layout**
# - `runs/<run_name>/` — per-epoch logs, predictions, plots, config snapshot. Light (~MB).
# - `models/<run_name>/best_model/` — final weights + best epoch's artifacts. Heavy.
#
# **⚠️ CPU training of wav2vec2-base needs a few GB of working RAM** on top of the
# OS + IDE. If RAM-bound: drop `batch_size` to 2 / `grad_accum` to 1, or train on GPU.

# %%
@dataclass
class Config:
    # -- Target preset (resolver overwrites the data fields below) -----------
    # target: str = "parlaspeech_hr_sentiment"   # ParlaSent logit — small-magnitude demo
    target: str = "parlaspeech_hr_age"           # speaker age (years) — demo target

    # -- Data (overwritten by resolve_target) -------------------------------
    jsonl_path:  str  = ""
    label_key:   str  = ""
    task_type:   str  = "regression"     # regression-only in this notebook
    label_order: list | None = None

    # Regression: optional class→float map. If None, labels must already be numeric.
    label_scale: dict | None = None
    # Regression target normalization: "none" | "zscore" (fit on TRAIN only).
    normalize:   str  = "zscore"

    # -- Run-mode caps (set by apply_mode from MODES[RUN_MODE]) --------------
    # None = no cap (full). cap_split applies these identically to train/dev/test,
    # so a capped run never silently leaves TEST at full size.
    cap_train: int | None = None
    cap_dev:   int | None = None
    cap_test:  int | None = None

    # -- Model ---------------------------------------------------------------
    model_name: str              = "facebook/wav2vec2-base"
    freeze_feature_encoder: bool = True
    loss_function: str           = "mse"   # regression only: "mse" | "l1"

    # -- Training ------------------------------------------------------------
    batch_size: int      = 16
    grad_accum: int      = 1
    learning_rate: float = 1e-5
    num_epochs: int      = 3      # full-run default; modes override (test=1, demo=2)
    max_grad_norm: float = 1.0
    warmup_ratio: float  = 0.10
    logging_steps: int   = 100

    # Rough ETA seed: train records processed per second (base model on GPU).
    # A deliberate guess — recalibrated from phase 1's real duration.
    eta_rec_per_s_guess: float = 40.0

    # -- Output --------------------------------------------------------------
    runs_dir: str   = "runs"
    models_dir: str = "models"

    # -- Best-epoch selection ------------------------------------------------
    best_metric_classification: str = "macro_f1"    # | "accuracy" | "spearman"
    best_metric_regression:     str = "spearman"    # | "mse" | "mae"

    # -- Preprocessing -------------------------------------------------------
    preprocess_batch_size: int  = 32
    dataloader_num_workers: int = 16   # drop to 0–2 on the CPU box if RAM-bound
    map_num_proc: int = 8              # parallel workers for the .map() feature pass
    max_duration_s: float = 15.0       # drop instances longer than this (OOM guard)

    # -- Hardware ------------------------------------------------------------
    # Honored from the GPU guard (cell above). GPU 2 reserved; never touch others.
    use_cuda: bool   = True

cfg = Config()
resolve_target(cfg, TARGETS)

# Layer the active run mode onto the base (full) config.
if RUN_MODE not in MODES:
    raise ValueError(f"RUN_MODE={RUN_MODE!r} not in {sorted(MODES)}")
apply_mode(cfg, MODES[RUN_MODE])
if RUN_MODE != "full":
    udp.banner(f"🔧 RUN_MODE = {RUN_MODE}", char="-")

# Device resolution. GPU selection + CUDA_VISIBLE_DEVICES already happened in the
# setup cell (input-gated) BEFORE torch was imported — the only point where
# pinning works. Here we just honor it; we do NOT touch CUDA_VISIBLE_DEVICES.
cfg.use_cuda = USE_CUDA
if cfg.use_cuda and torch.cuda.is_available():
    DEVICE = "cuda"   # = cuda:0 in-process, pinned to physical GPU 2
elif cfg.use_cuda and not torch.cuda.is_available():
    print("⚠️  GPU selected but torch.cuda.is_available()==False; falling back to CPU")
    DEVICE = "cpu"
else:
    DEVICE = "cpu"

print(f"target      = {cfg.target}")
print(f"run_mode    = {RUN_MODE}  (caps train/dev/test = {cfg.cap_train}/{cfg.cap_dev}/{cfg.cap_test}, sampling={DEMO_SAMPLING})")
print(f"jsonl_path  = {cfg.jsonl_path}")
print(f"label_key   = {cfg.label_key}")
print(f"task_type   = {cfg.task_type}")
print(f"normalize   = {cfg.normalize}")
print(f"device      = {DEVICE}")
if DEVICE == "cuda":
    print(f"✓ visible devices : {torch.cuda.device_count()}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    print(f"✓ device name     : {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Validate the config
#
# Fail loud here so you don't burn hours of training only to learn a knob was wrong.

# %%
def validate_config(cfg: Config) -> None:
    if RUN_MODE not in MODES:
        raise ValueError(f"RUN_MODE invalid: {RUN_MODE!r} (choose {sorted(MODES)})")
    if DEMO_SAMPLING not in ("proportional", "balanced"):
        raise ValueError(f"DEMO_SAMPLING invalid: {DEMO_SAMPLING!r} (proportional|balanced)")
    if cfg.task_type != "regression":
        raise ValueError(f"32_ is regression-only; got task_type={cfg.task_type!r}. "
                         f"Use 31_train_classification for classification targets.")
    if cfg.loss_function not in ("mse", "l1"):
        raise ValueError(f"loss_function must be 'mse' or 'l1', got {cfg.loss_function!r}")
    if cfg.best_metric_regression not in ("spearman", "mse", "mae"):
        raise ValueError(f"best_metric_regression: invalid {cfg.best_metric_regression!r}")
    if cfg.normalize not in ("none", "zscore"):
        raise ValueError(f"normalize must be 'none' or 'zscore', got {cfg.normalize!r}")

validate_config(cfg)
print("✅ config valid")

# %% [markdown]
# # Data

# %% [markdown]
# ## Load JSONL, filter to records that carry `label_key`
#
# Records missing the target label (or longer than `max_duration_s`) are dropped.
# `DEMO_*` caps shrink TRAIN/DEV for a bounded demo run; TEST stays whole for a
# trustworthy final number. Set a cap to `None` for the full run.

# %%
mark("data prep")

CAP_SEED = 1234   # deterministic shuffle seed for cap_split


def load_split(jsonl_path: str, split: str, label_key: str) -> list[dict]:
    out, n_long = [], 0
    for r in udp.iter_jsonl(jsonl_path):
        if r["split"] != split:
            continue
        if label_key not in r.get("labels", {}):
            continue
        if r["labels"][label_key] is None:
            continue
        if r.get("metadata", {}).get("audio_length", 0.0) > cfg.max_duration_s:
            n_long += 1
            continue
        out.append(r)
    if n_long:
        print(f"  dropped {n_long} {split} records > {cfg.max_duration_s}s")
    return out


train_records = load_split(cfg.jsonl_path, "train", cfg.label_key)
dev_records   = load_split(cfg.jsonl_path, "dev",   cfg.label_key)
test_records  = load_split(cfg.jsonl_path, "test",  cfg.label_key)

# One knob, applied identically to every split. None caps (full mode) are no-ops.
_pre = (len(train_records), len(dev_records), len(test_records))
train_records = cap_split(train_records, cfg.cap_train, CAP_SEED, DEMO_SAMPLING)
dev_records   = cap_split(dev_records,   cfg.cap_dev,   CAP_SEED, DEMO_SAMPLING)
test_records  = cap_split(test_records,  cfg.cap_test,  CAP_SEED, DEMO_SAMPLING)


def _capline(name: str, pre: int, post: int, cap) -> None:
    tag = f"capped→{cap}" if (cap is not None and post < pre) else "uncapped"
    print(f"   {name:<6} {post:>9d}   (of {pre:>9d}, {tag})")


print(f"run_mode={RUN_MODE}  sampling={DEMO_SAMPLING}")
_capline("train", _pre[0], len(train_records), cfg.cap_train)
_capline("dev",   _pre[1], len(dev_records),   cfg.cap_dev)
_capline("test",  _pre[2], len(test_records),  cfg.cap_test)
if not train_records or not dev_records or not test_records:
    raise ValueError("one of the splits is empty after filtering for label_key — check the JSONL")


def rough_eta_seconds(n_train: int, n_dev: int, cfg: Config, rec_per_s: float) -> float:
    """Coarse wall-clock estimate: both phases train (TRAIN, then TRAIN∪DEV) for
    num_epochs at ~rec_per_s training records/second. Eval + preprocessing extra."""
    train_recs = (n_train * cfg.num_epochs) + ((n_train + n_dev) * cfg.num_epochs)
    return train_recs / max(1e-9, rec_per_s)


_eta = rough_eta_seconds(len(train_records), len(dev_records), cfg, cfg.eta_rec_per_s_guess)
print(f"\n⏱  rough ETA ~{fmt_mmss(_eta)} for {cfg.num_epochs} epoch(s) × 2 phases "
      f"(guess {cfg.eta_rec_per_s_guess:.0f} train-rec/s — approximate; "
      f"recalibrates after phase 1)")

# %% [markdown]
# ## Labels & normalization
#
# Regression has no label space, so we keep `(label2id, id2label, num_labels)` as
# `(None, None, 1)` purely so `run_phase` stays identical to 31. We assert the
# target is numeric (or `label_scale`-mappable), then fit the normalizer on the
# **TRAIN labels only** — never dev/test, to avoid leakage.

# %%
label2id, id2label, num_labels = None, None, 1

for r in train_records + dev_records + test_records:
    v = r["labels"][cfg.label_key]
    if isinstance(v, (int, float)):
        continue
    if cfg.label_scale is not None and v in cfg.label_scale:
        continue
    raise ValueError(
        f"{r['instance_id']}: regression label is {v!r} (type {type(v).__name__}), "
        f"not numeric and no label_scale mapping. Provide label_scale or fix the data."
    )


def raw_label(r: dict, cfg: Config) -> float:
    """Raw target in real units, applying label_scale if present."""
    v = r["labels"][cfg.label_key]
    if cfg.label_scale is not None and v in cfg.label_scale:
        return float(cfg.label_scale[v])
    return float(v)


@dataclass
class LabelNormalizer:
    """Standardize a regression target. encode(): real → training space;
    decode(): training space → real. 'none' is the identity."""
    kind: str = "none"
    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def fit(cls, values, kind: str) -> "LabelNormalizer":
        if kind == "none":
            return cls("none", 0.0, 1.0)
        if kind == "zscore":
            arr = np.asarray(list(values), dtype=float)
            mean = float(arr.mean())
            std = float(arr.std())
            if std < 1e-8:
                std = 1.0   # guard against a constant target
            return cls("zscore", mean, std)
        raise ValueError(f"unknown normalize kind {kind!r}")

    def encode(self, y):
        if self.kind == "none":
            return y
        return (np.asarray(y, dtype=float) - self.mean) / self.std

    def decode(self, y):
        if self.kind == "none":
            return y
        return np.asarray(y, dtype=float) * self.std + self.mean


# Fit on TRAIN ONLY (no dev/test leakage).
normalizer = LabelNormalizer.fit((raw_label(r, cfg) for r in train_records), cfg.normalize)
print(f"Regression target: '{cfg.label_key}'  | normalize: {normalizer.kind}"
      + (f"  (train mean={normalizer.mean:.3f}, std={normalizer.std:.3f})"
         if normalizer.kind != "none" else ""))

# %% [markdown]
# # Training engine
#
# Everything from here to just before the two phase calls is the shared engine:
# feature extraction, collator, model, metrics, per-epoch artifacts, and
# `run_phase`. This block is byte-identical to 31 except the task-specific bodies
# (regression metrics / scatter plots / float labels), so it lifts cleanly to
# utils in phase E.

# %% [markdown]
# ## Audio loading & feature extraction
#
# `prepare_dataset_dict` turns canonical records into the list-of-dicts
# `Dataset.from_list` wants. Provenance (`file_id`, `start_t`, `end_t`,
# `audio_length`) is read from `metadata.*` so it actually lands in predictions.json.
# `label_to_value` emits the **normalized** training target.
# `preprocess_function` loads each WAV (mono/16 kHz guaranteed by chapter 1) and
# runs the feature extractor *without* padding — the collator pads per batch.

# %%
def label_to_value(r: dict, cfg: Config, label2id: dict | None) -> float:
    """Real-unit label → normalized training target (identity if normalize='none')."""
    return float(normalizer.encode(raw_label(r, cfg)))


def prepare_dataset_dict(records: list[dict], cfg: Config, label2id: dict | None) -> list[dict]:
    items = []
    for r in records:
        md = r.get("metadata", {})
        items.append({
            "instance_id":  r["instance_id"],
            "file_id":      r.get("file_id", ""),
            "start_t":      md.get("audio_start"),
            "end_t":        md.get("audio_end"),
            "audio_length": md.get("audio_length"),
            "audio_path":   str(udp.from_project_relative(r["audio_path"])),
            "label":        label_to_value(r, cfg, label2id),
            "label_class":  r["labels"][cfg.label_key],   # original value for reporting
        })
    return items


def preprocess_function(examples, feature_extractor):
    """Load each WAV via soundfile. Chapter-1 splitter guarantees 16 kHz mono PCM-16;
    we sanity-check and resample as a defensive fallback."""
    audio_arrays = []
    for path in examples["audio_path"]:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        audio_arrays.append(data)
    inputs = feature_extractor(
        audio_arrays, sampling_rate=16000, return_tensors=None, padding=False,
    )
    return {"input_values": inputs["input_values"], "labels": examples["label"]}

# %% [markdown]
# ## Data collator (pad audio within each batch)
#
# The feature extractor doesn't pad; this does — **and emits the `attention_mask`**
# so the model's masked mean-pooling knows which frames are real vs padding.
# (wav2vec2-base ships `return_attention_mask=False`; we force it on at load, and
# pass it explicitly here. Without it the pooling silently averages over padding.)
# Labels are floats for regression.

# %%
class DataCollatorForInstance:
    """Pads audio within each batch and threads the attention mask through.
    Keeps the `task_type` arg for signature parity with 31_classification so
    `run_phase` stays identical across the two notebooks."""
    def __init__(self, feature_extractor, task_type: str = "regression"):
        self.feature_extractor = feature_extractor
        self.task_type = task_type

    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]
        batch = self.feature_extractor.pad(
            {"input_values": input_values},
            padding=True, return_attention_mask=True, return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch

# %% [markdown]
# ## Model factory
#
# `Wav2Vec2ForRegression` = wav2vec2 + masked mean-pooling + `Linear(1)`. The
# pretrained `wav2vec2` submodule is loaded into it. `freeze_feature_encoder`
# freezes the CNN front-end. Pooling uses the collator's `attention_mask`, so
# padded frames are excluded from the mean.

# %%
class Wav2Vec2ForRegression(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 + masked mean-pooling + Linear(1). MSE or L1 loss."""

    def __init__(self, config, loss_type: str = "mse"):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.regression_head = nn.Linear(config.hidden_size, 1)
        self.loss_type = loss_type
        self.post_init()

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        stride = attention_mask.shape[1] / feature_vector_length
        indices = (torch.arange(feature_vector_length, device=attention_mask.device) * stride).long()
        indices = torch.clamp(indices, max=attention_mask.shape[1] - 1)
        return torch.index_select(attention_mask, 1, indices)

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        if attention_mask is not None:
            sub_mask = self._get_feature_vector_attention_mask(last_hidden.shape[1], attention_mask)
            mask = sub_mask.unsqueeze(-1).float()
            sum_h = (last_hidden * mask).sum(dim=1)
            cnt = torch.clamp(mask.sum(dim=1), min=1e-9)
            hidden = sum_h / cnt
        else:
            hidden = last_hidden.mean(dim=1)
        logits = self.regression_head(hidden)
        loss = None
        if labels is not None:
            fct = nn.MSELoss() if self.loss_type == "mse" else nn.L1Loss()
            loss = fct(logits.view(-1), labels.view(-1))
        return {"loss": loss, "logits": logits.view(-1)}


def build_model(cfg: Config, num_labels: int, label2id, id2label):
    config_obj = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
    model = Wav2Vec2ForRegression(config_obj, loss_type=cfg.loss_function)
    model.wav2vec2 = Wav2Vec2Model.from_pretrained(
        cfg.model_name, config=config_obj, ignore_mismatched_sizes=True,
    )
    if cfg.freeze_feature_encoder:
        model.wav2vec2.freeze_feature_encoder()
        print("🔒 feature encoder (CNN) frozen")
    return model

# %% [markdown]
# ## Metrics
#
# MSE, MAE, Spearman (+ p-value), all in **real units** — preds and labels are
# de-normalized via the fitted `normalizer` before scoring, so the numbers read
# as years / sentiment points regardless of `normalize`.

# %%
def compute_regression_metrics(eval_pred):
    preds, labels = eval_pred
    preds  = np.asarray(normalizer.decode(np.asarray(preds).reshape(-1)), dtype=float)
    labels = np.asarray(normalizer.decode(np.asarray(labels).reshape(-1)), dtype=float)
    mse = float(mean_squared_error(labels, preds))
    mae = float(mean_absolute_error(labels, preds))
    if len(set(labels.tolist())) > 1 and len(set(preds.tolist())) > 1:
        rho, p = spearmanr(labels, preds)
        rho = float(rho) if not np.isnan(rho) else float("nan")
        p = float(p) if not np.isnan(p) else float("nan")
    else:
        rho, p = float("nan"), float("nan")
    return {"mse": mse, "mae": mae, "spearman": rho, "spearman_p_value": p}


def get_compute_metrics(cfg: Config):
    return compute_regression_metrics

# %% [markdown]
# ## Per-epoch artifacts
#
# `predictions.json` (real units, correct provenance), a gold-vs-pred scatter, and
# gold/pred distribution histograms — one set per epoch.

# %%
def save_predictions_json(predictions, labels, items, out_path: Path, task_type: str,
                          id2label: dict | None):
    out = []
    for i, item in enumerate(items):
        gold = float(labels[i])
        pred = float(predictions[i])
        out.append({
            "instance_id": item["instance_id"],
            "file_id":     item.get("file_id", ""),
            "start_t":     item.get("start_t"),
            "end_t":       item.get("end_t"),
            "gold_label":  gold,
            "pred_label":  pred,
            "gold_raw":    gold,
            "pred_raw":    pred,
        })
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))


def plot_scatter(gold, pred, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gold, pred, alpha=0.4)
    lo = float(min(gold.min(), pred.min())); hi = float(max(gold.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax.set_xlabel("gold"); ax.set_ylabel("pred"); ax.set_title("gold vs pred")
    plt.tight_layout(); fig.savefig(out_path); plt.close(fig)


def plot_distribution(gold, pred, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = 30
    ax.hist(gold, bins=bins, alpha=0.5, label="gold")
    ax.hist(pred, bins=bins, alpha=0.5, label="pred")
    ax.legend(); ax.set_title("label distribution: gold vs pred")
    plt.tight_layout(); fig.savefig(out_path); plt.close(fig)

# %% [markdown]
# ## `EpochCheckpointCallback`
#
# Evaluates the eval set every epoch and writes per-epoch logs. Predictions and
# plots are de-normalized to real units. Does **not** save weights — only the best
# epoch's model is saved, and only in phase 2.

# %%
class EpochCheckpointCallback(TrainerCallback):
    def __init__(self, phase_dir: Path, eval_dataset, eval_items,
                 compute_metrics, data_collator, cfg: Config,
                 label_order=None, id2label=None):
        self.phase_dir = phase_dir
        self.logs_dir = phase_dir / "epoch_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dataset = eval_dataset
        self.eval_items = eval_items
        self.compute_metrics = compute_metrics
        self.data_collator = data_collator
        self.cfg = cfg
        self.label_order = label_order
        self.id2label = id2label
        self.epoch_results: list[dict] = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        print(f"\n💾 logging epoch {epoch}…")
        epoch_dir = self.logs_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        eval_trainer = Trainer(
            model=model, args=args,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator,
        )
        out = eval_trainer.predict(self.eval_dataset)

        # Normalize "test_..." → "eval_..." (HF predict uses 'test_' prefix)
        metrics = {k.replace("test_", "eval_"): v for k, v in out.metrics.items()}

        train_loss = None
        for log in reversed(state.log_history):
            if "loss" in log:
                train_loss = log["loss"]; break

        epoch_info = {"epoch": epoch, "train_loss": train_loss, **metrics}
        self.epoch_results.append(epoch_info)

        # De-normalize to real units for saved predictions + plots.
        gold = np.asarray(normalizer.decode(np.asarray(out.label_ids).reshape(-1)), dtype=float)
        pred = np.asarray(normalizer.decode(np.asarray(out.predictions).reshape(-1)), dtype=float)

        save_predictions_json(
            pred, gold, self.eval_items, epoch_dir / "predictions.json",
            task_type=self.cfg.task_type, id2label=self.id2label,
        )
        plot_scatter(gold, pred, epoch_dir / "scatter_plot.png")
        plot_distribution(gold, pred, epoch_dir / "distribution_plot.png")

        (epoch_dir / "epoch_summary.json").write_text(json.dumps(epoch_info, indent=2))

        print(f"   epoch={epoch}  mse={metrics.get('eval_mse', 0):.4f}  "
              f"mae={metrics.get('eval_mae', 0):.4f}  "
              f"rho={metrics.get('eval_spearman', float('nan')):.4f}")

# %% [markdown]
# ## `run_phase` — train + evaluate + save logs
#
# One function for both phases. Differences are only *which* records train, which
# split evaluates, and whether the best model is saved. Frees the model + CUDA
# cache on the way out so the next phase starts clean. **Identical across 31/32.**

# %%
def best_epoch_of(epoch_results: list[dict], cfg: Config) -> dict:
    m = cfg.best_metric_regression
    if m in ("mse", "mae"):
        return min(epoch_results, key=lambda r: r.get(f"eval_{m}", float("inf")))
    return max(epoch_results, key=lambda r: (r.get(f"eval_{m}", float("-inf"))
                                             if r.get(f"eval_{m}") is not None
                                             and not np.isnan(r.get(f"eval_{m}", float("nan")))
                                             else float("-inf")))


def run_phase(*, phase_name: str, train_records: list[dict], eval_records: list[dict],
              eval_split_name: str, save_best_model: bool,
              cfg: Config, run_dir: Path, model_dir: Path, feature_extractor,
              label2id, id2label) -> tuple[list[dict], dict]:
    udp.banner(f"PHASE: {phase_name}  (train→{eval_split_name})")
    phase_dir = run_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Build items + datasets
    train_items = prepare_dataset_dict(train_records, cfg, label2id)
    eval_items  = prepare_dataset_dict(eval_records,  cfg, label2id)
    train_ds = Dataset.from_list(train_items)
    eval_ds  = Dataset.from_list(eval_items)

    print(f"preprocessing {len(train_ds)} train + {len(eval_ds)} eval (batch_size={cfg.preprocess_batch_size})…")
    train_ds = train_ds.map(
        lambda x: preprocess_function(x, feature_extractor),
        batched=True, batch_size=cfg.preprocess_batch_size,
        remove_columns=train_ds.column_names, num_proc=cfg.map_num_proc,
    )
    eval_ds = eval_ds.map(
        lambda x: preprocess_function(x, feature_extractor),
        batched=True, batch_size=cfg.preprocess_batch_size,
        remove_columns=eval_ds.column_names, num_proc=cfg.map_num_proc,
    )
    label_dtype = "torch.long" if cfg.task_type == "classification" else "torch.float32"
    train_ds.set_format(type="torch", columns=["input_values", "labels"])
    eval_ds.set_format(type="torch", columns=["input_values", "labels"])

    print(f"building model: {cfg.model_name}")
    model = build_model(cfg, num_labels=len(cfg.label_order) if cfg.task_type == "classification" else 1,
                        label2id=label2id, id2label=id2label)

    data_collator = DataCollatorForInstance(feature_extractor, cfg.task_type)
    compute_metrics = get_compute_metrics(cfg)

    steps_per_epoch = max(1, len(train_ds) // (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(phase_dir / "trainer_tmp"),
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        report_to="none",
        label_names=["labels"],
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        max_grad_norm=cfg.max_grad_norm,
        remove_unused_columns=False,
        use_cpu=(DEVICE == "cpu"),
        bf16=(DEVICE == "cuda"),
        tf32=True,
        dataloader_num_workers=cfg.dataloader_num_workers,
    )

    callback = EpochCheckpointCallback(
        phase_dir=phase_dir, eval_dataset=eval_ds, eval_items=eval_items,
        compute_metrics=compute_metrics, data_collator=data_collator, cfg=cfg,
        label_order=cfg.label_order if cfg.task_type == "classification" else None,
        id2label=id2label,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator, callbacks=[callback],
    )
    print(f"🚀 training {cfg.num_epochs} epochs (bs={cfg.batch_size} ga={cfg.grad_accum} lr={cfg.learning_rate})")
    trainer.train()

    (phase_dir / "all_epochs_summary.json").write_text(
        json.dumps(callback.epoch_results, indent=2)
    )
    best = best_epoch_of(callback.epoch_results, cfg)
    print(f"\n🏆 best epoch in {phase_name}: {best['epoch']}")
    for k, v in best.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    if save_best_model:
        best_dir = model_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir)
        feature_extractor.save_pretrained(best_dir)
        src = phase_dir / "epoch_logs" / f"epoch_{best['epoch']}"
        if src.exists():
            for f in src.iterdir():
                shutil.copy(f, best_dir / f.name)
        info_lines = [
            f"run_name: {run_dir.name}",
            f"run_dir:  {run_dir}",
            f"phase:    {phase_name}",
            f"epoch:    {best['epoch']}",
        ]
        (best_dir / "run_info.txt").write_text("\n".join(info_lines) + "\n")
        print(f"   saved best model → {best_dir.relative_to(PROJECT_ROOT)}")

    # Clean up Trainer's tmp dir
    shutil.rmtree(phase_dir / "trainer_tmp", ignore_errors=True)

    # Free model + CUDA cache before the next phase allocates a fresh one.
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return callback.epoch_results, best

# %% [markdown]
# # Run

# %% [markdown]
# ## Run directory
#
# `runs/{dataset}_{label_key}_{task_type}_{timestamp}/`. Also snapshots the
# resolved Config and the fitted label-normalization stats for reproducibility.

# %%
dataset_name = train_records[0]["dataset"]
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"{dataset_name}_{cfg.label_key}_{cfg.task_type}_{ts}"

run_dir   = udp.from_project_relative(cfg.runs_dir)   / run_name
model_dir = udp.from_project_relative(cfg.models_dir) / run_name
run_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
print(f"run_dir   = {run_dir.relative_to(PROJECT_ROOT)}    (per-epoch logs)")
print(f"model_dir = {model_dir.relative_to(PROJECT_ROOT)}  (best_model goes here)")

from dataclasses import asdict as _asdict
(run_dir / "config.json").write_text(json.dumps(_asdict(cfg), indent=2, default=str))
(run_dir / "label_normalization.json").write_text(
    json.dumps({"kind": normalizer.kind, "mean": normalizer.mean, "std": normalizer.std}, indent=2)
)

# %% [markdown]
# ## Feature extractor
#
# Single load, reused across both phases. We force `return_attention_mask=True`
# so the collator can hand the model a real mask (see the collator note above).

# %%
print(f"loading feature extractor: {cfg.model_name}")
feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name)
feature_extractor.return_attention_mask = True
print(f"   sampling_rate         = {feature_extractor.sampling_rate}")
print(f"   return_attention_mask = {feature_extractor.return_attention_mask}")

# %% [markdown]
# ## Phase 1 — TRAIN → DEV (development)
#
# Train on TRAIN, evaluate on DEV every epoch. No model saved.

# %%
mark("model prep")
phase1_results, phase1_best = run_phase(
    phase_name="phase1_dev",
    train_records=train_records, eval_records=dev_records,
    eval_split_name="DEV", save_best_model=False,
    cfg=cfg, run_dir=run_dir, model_dir=model_dir,
    feature_extractor=feature_extractor,
    label2id=label2id, id2label=id2label,
)
mark("end phase 1")

# %% [markdown]
# ## Phase 2 — TRAIN + DEV → TEST (final)
#
# Re-train on TRAIN ∪ DEV, evaluate on TEST every epoch. Best epoch's model saved.
# The ETA is recalibrated from phase 1's real training rate before kicking off.

# %%
_p1_secs = STAGE_TIMES["end phase 1"] - STAGE_TIMES["model prep"]
_rate = (len(train_records) * cfg.num_epochs) / max(1e-9, _p1_secs)
_p2_secs = (len(train_records) + len(dev_records)) * cfg.num_epochs / max(1e-9, _rate)
print(f"⏱  phase 1 took {fmt_mmss(_p1_secs)} → ~{_rate:.0f} train-rec/s  |  "
      f"phase 2 rough ETA ~{fmt_mmss(_p2_secs)} (approximate)\n")

phase2_results, phase2_best = run_phase(
    phase_name="phase2_test",
    train_records=train_records + dev_records, eval_records=test_records,
    eval_split_name="TEST", save_best_model=True,
    cfg=cfg, run_dir=run_dir, model_dir=model_dir,
    feature_extractor=feature_extractor,
    label2id=label2id, id2label=id2label,
)
mark("end phase 2")

# %% [markdown]
# ## Run summary
#
# `phase2_best` is the headline (TEST); `phase1_best` is informational (DEV).

# %%
udp.banner(f"RUN SUMMARY: {run_name}")
print(f"task        : {cfg.task_type}")
print(f"label_key   : {cfg.label_key}")
print(f"normalize   : {normalizer.kind}")
print(f"model       : {cfg.model_name}")
print(f"epochs      : {cfg.num_epochs}")
print(f"run_dir     : {run_dir.relative_to(PROJECT_ROOT)}    (logs)")
print(f"model_dir   : {model_dir.relative_to(PROJECT_ROOT)}  (best model)\n")

print(f"Phase 1 best (DEV)  — epoch {phase1_best['epoch']}:")
for k, v in phase1_best.items():
    if isinstance(v, float):
        print(f"   {k}: {v:.4f}")

print(f"\nPhase 2 best (TEST) — epoch {phase2_best['epoch']}:")
for k, v in phase2_best.items():
    if isinstance(v, float):
        print(f"   {k}: {v:.4f}")

print(f"\nbest model: {(model_dir / 'best_model').relative_to(PROJECT_ROOT)}")

# %% [markdown]
# ## Inline scatter + distribution (TEST, best epoch)
#
# Reads `predictions.json` from disk (real units) so this re-runs independently of
# training.

# %%
best_epoch_dir = run_dir / "phase2_test" / "epoch_logs" / f"epoch_{phase2_best['epoch']}"
preds_path = best_epoch_dir / "predictions.json"
if not preds_path.exists():
    raise FileNotFoundError(f"Expected predictions at {preds_path}, but it's missing.")

preds_data = json.loads(preds_path.read_text())
gold = np.array([p["gold_raw"] for p in preds_data], dtype=float)
pred = np.array([p["pred_raw"] for p in preds_data], dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes[0].scatter(gold, pred, alpha=0.4)
lo = float(min(gold.min(), pred.min())); hi = float(max(gold.max(), pred.max()))
axes[0].plot([lo, hi], [lo, hi], "--", linewidth=1)
axes[0].set_xlabel("gold"); axes[0].set_ylabel("pred")
axes[0].set_title(f"gold vs pred — epoch {phase2_best['epoch']} on TEST")

bins = 30
axes[1].hist(gold, bins=bins, alpha=0.5, label="gold")
axes[1].hist(pred, bins=bins, alpha=0.5, label="pred")
axes[1].legend(); axes[1].set_title("label distribution: gold vs pred")

plt.tight_layout()
plot_png = run_dir / "scatter_distribution_test.png"
fig.savefig(plot_png, dpi=120, bbox_inches="tight")
print(f"saved {plot_png.relative_to(PROJECT_ROOT)}")
plt.show()

# %% [markdown]
# ## Inference spot-check
#
# Eyeball 5 random TEST predictions: provenance + gold vs pred (real units) + the
# absolute error. A quick sanity read on what the model is actually doing —
# complements the aggregate metrics above.

# %%
import random as _rnd

_rows = json.loads(preds_path.read_text())
_sample = _rnd.Random(0).sample(_rows, k=min(5, len(_rows)))

udp.banner("INFERENCE SPOT-CHECK — 5 random TEST examples")
for p in _sample:
    g, pr = p.get("gold_raw"), p.get("pred_raw")
    err = abs(g - pr) if (g is not None and pr is not None) else float("nan")
    g_s  = f"{g:.3f}"  if g  is not None else "None"
    pr_s = f"{pr:.3f}" if pr is not None else "None"
    print(f"  {p['instance_id']}")
    print(f"     file={p.get('file_id', '')}  span={p.get('start_t')}–{p.get('end_t')}")
    print(f"     gold={g_s}   pred={pr_s}   |err|={err:.3f}")
    print()

# %% [markdown]
# ## Stage timing
#
# Wall-clock per stage (delta from the previous mark) + total, mm:ss. Prints
# whatever marks exist, so a partial run still reports cleanly.

# %%
mark("end script")
print_stage_breakdown(STAGE_TIMES)

# %% [markdown]
# ## What's next
#
# This run wrote per-epoch logs + a saved best model under `runs/`. Chapter 5
# (`5_analysis/`) loads run directories like this one for error analysis and
# cross-run comparison.
#
# Same dataset, second target: change `Config.target` and re-run. For ROG
# sentiment as regression, set a target with `task_type="regression"` and provide
# `label_scale={...}` mapping classes → a numeric ramp. `31_train_classification`
# is the twin for gender / filled-pause classification.
