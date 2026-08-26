# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train frame — classification (chapter 4)
#
# Per-frame classification: feed an utterance WAV, predict a class **per model
# frame** (`(B, T, num_labels)`), token-CE with `ignore_index=-100`. First frame
# task: **FP frames** — binary FP / not-FP over the whole utterance on
# ParlaSpeech-HR `utterance_frame` (the 50 Hz `filled_pause` sequence emitted by
# `11c`).
#
# The run engine (GPU guard, stage timer + ETA, demo/test/full tiers, two-phase
# loop, attention-mask handling, GPU flush, inference spot-check) mirrors the
# chapter-3 twins `31`/`32`. The frame-specific pieces are: the per-frame head,
# token-CE, **per-record label alignment to the model's real CNN output length**
# (~49 Hz vs the nominal 50), and frame-flavored metrics/plots. `42_train_frame
# _regression` will be the twin once a continuous per-frame target exists.

# %% [markdown]
# # Setup

# %%
import time

# ── Stage timing ───────────────────────────────────────────────────────────────
# Identical harness across 31/32/41. mark() stamps a milestone; the final cell
# prints a per-stage breakdown. Stdlib-only, cheap, partial-run safe (prints
# whatever marks exist).
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
if HERE.name != "4_frame_models":
    candidate = HERE / "4_frame_models"
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
import matplotlib.colors as mcolors
import seaborn as sns
import soundfile as sf
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoConfig, AutoFeatureExtractor,
    Trainer, TrainerCallback, TrainingArguments,
    Wav2Vec2Model, Wav2Vec2PreTrainedModel,
)

plt.rcParams["figure.dpi"] = 100

# Token-CE pad sentinel. Labels at this value are ignored by the loss and masked
# out of every metric. Used by the collator, the head, metrics, and plots.
IGNORE_INDEX = -100

# %% [markdown]
# # Targets
#
# What this trainer can handle — lang-agnostic. Pick a task via `Config.target`;
# `Config.langs` then selects which ParlaSpeech languages to pool (default: all
# supported langs that have a JSONL on disk). The resolver fills `label_key`,
# `task_type`, `label_order`, and the list of `jsonl_paths` to concatenate.
# Splits are speaker-grouped within each lang and speakers never cross languages,
# so pooling is leakage-free concatenation; the Trainer's shuffle interleaves
# langs. Frame-flavored mirror of the chapter-3 `TARGETS` registry — same shape,
# plus a `level: "frame"` tag. To add a target, add an entry below;
# `Config.target` is the only downstream knob.

# %%
TARGETS: dict = {}


def _add_parlaspeech_frame_targets(targets: dict) -> None:
    """Task-keyed frame targets (lang-agnostic):
    - `fp_frames`: binary filled_pause over the whole utterance (`utterance_frame`);
      HR/RS/PL/CZ — FP is acoustically comparable across all four.
    - `primary_stress_frames`: rung 6, the north star — the WORD is the instance
      (`word_frame`, sliced in memory via start_t/end_t). HR/RS only: they carry
      `primary_stress` and are close Štokavian pitch-accent langs, so mixing is
      sound. (Polish/Czech fixed stress would be a different phenomenon — but they
      have no stress annotation anyway.)"""
    targets["parlaspeech_fp_frames"] = {
        "jsonl_template": "data/processed_jsonl/parlaspeech_{lang}_utterance_frame.jsonl",
        "langs":       ("hr", "rs", "pl", "cz"),
        "label_key":   "filled_pause",
        "task_type":   "classification",
        "level":       "frame",
        "label_order": [0, 1],   # 0 = no FP, 1 = FP
    }
    targets["parlaspeech_primary_stress_frames"] = {
        "jsonl_template": "data/processed_jsonl/parlaspeech_{lang}_word_frame.jsonl",
        "langs":       ("hr", "rs"),
        "label_key":   "primary_stress",
        "task_type":   "classification",
        "level":       "frame",
        "label_order": [0, 1],   # 0 = unstressed frame, 1 = primary stress
    }


def _add_nejc_slo_stress_targets(targets: dict) -> None:
    """Slovenian primary-stress `word_frame` from Nejc's annotated TGs
    (built by `5_tg_minter/54_stress_tg_to_jsonl.py`). Independent target —
    not pooled with the HR/RS ParlaSpeech one; Slovenian stress is a different
    phenomenon from Štokavian pitch accent."""
    targets["si_primary_stress_frames"] = {
        "jsonl_path":  "data/processed_jsonl/si_primary_stress_word_frame.jsonl",
        "label_key":   "primary_stress",
        "task_type":   "classification",
        "level":       "frame",
        "label_order": [0, 1],
    }


_add_parlaspeech_frame_targets(TARGETS)
_add_nejc_slo_stress_targets(TARGETS)


def resolve_target(cfg, targets: dict) -> None:
    """Fill label_key/task_type/label_order and `cfg.jsonl_paths` (the list of
    JSONLs to pool) from the picked task preset and `cfg.langs`. Mutates cfg.

    Two shapes are supported:
    - Multi-lang template: preset carries `jsonl_template` + `langs`.
      `cfg.langs=()` → every supported lang that has a JSONL on disk (missing
      ones skipped with a note). `cfg.langs=("hr",)` → exactly those (must be
      supported; a requested-but-missing JSONL is a hard error).
    - Single-file target: preset carries `jsonl_path`. `cfg.langs` is ignored
      (a warning is printed if it is set); the one path must exist on disk.
    """
    if cfg.target not in targets:
        raise ValueError(
            f"Config.target={cfg.target!r} not in TARGETS. Known: {sorted(targets)}"
        )
    t = targets[cfg.target]
    cfg.label_key   = t["label_key"]
    cfg.task_type   = t["task_type"]
    cfg.label_order = t["label_order"]

    # ── Single-file target ────────────────────────────────────────────────
    if "jsonl_path" in t:
        if cfg.langs:
            print(f"  ⚠️  cfg.langs={cfg.langs} ignored for single-file target {cfg.target!r}")
        p = t["jsonl_path"]
        if not udp.from_project_relative(p).exists():
            raise FileNotFoundError(
                f"target {cfg.target!r} expects {p} on disk — build it first"
            )
        cfg.jsonl_paths = [p]
        return

    # ── Multi-lang template target ────────────────────────────────────────
    supported = t["langs"]
    chosen = tuple(l.lower() for l in cfg.langs) if cfg.langs else supported
    unknown = [l for l in chosen if l not in supported]
    if unknown:
        raise ValueError(
            f"cfg.langs {unknown} not supported by {cfg.target!r} (supports {list(supported)})"
        )
    paths = []
    for l in chosen:
        p = t["jsonl_template"].format(lang=l)
        if udp.from_project_relative(p).exists():
            paths.append(p)
        elif cfg.langs:                       # explicitly requested but absent → loud
            raise FileNotFoundError(f"requested lang {l!r} but {p} is missing — run 11c")
        else:                                 # auto (all): skip what isn't there yet
            print(f"  ⏭  {l}: {p} not found — skipping")
    if not paths:
        raise FileNotFoundError(
            f"no JSONL on disk for {cfg.target!r} (langs={list(chosen)}). Run 11c first."
        )
    cfg.jsonl_paths = paths


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
# per language. Orthogonal to `cfg.langs`: mode = *how much*, langs = *which
# languages*.

# %%
RUN_MODE = "demo"               # "test" | "demo" | "full"
DEMO_SAMPLING = "proportional"  # "proportional" | "balanced" — only when pooling >1 lang

# Each entry overrides the base (full) Config. Reading all three side by side
# tells you exactly what each tier changes; everything unlisted stays at its full
# default.
MODES: dict = {
    "test": {
        "model_name": "hf-internal-testing/tiny-random-wav2vec2",
        "cap_train": 64, "cap_dev": 16, "cap_test": 16,
        "num_epochs": 1, "batch_size": 2, "grad_accum": 1,
        "runs_dir": "runs/test", "models_dir": "models/test",
    },
    "demo": {
        "cap_train": 20_000, "cap_dev": 4_000, "cap_test": 4_000,
        "num_epochs": 2, "batch_size": 128,
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
      - balanced: round-robin per language → ~equal counts (draws whatever's left
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
# - `task_type`: classification-only in v1; `"regression"` raises (frame
#   regression is deferred — `42_train_frame_regression` is the future twin).
# - `label_key`: which frame sequence inside `labels` to train on (e.g.
#   `filled_pause`). Filled by the resolver.
# - `label_order`: **required** canonical class order → `id2label`. Binary FP =
#   `[0, 1]`. No alphabetical fallback.
# - `required_frame_rate_hz`: hard-locked to 50. Labels are aligned per-record to
#   the model's actual CNN output length (~49 Hz) at preprocess time; any other
#   *source* rate is a hard error (resampling source labels is deferred).
# - `use_cuda`: honored from the GPU guard above; never touches a GPU other than 2.
#
# **Output layout**
# - `runs/<run_name>/` — per-epoch logs, predictions, plots, config snapshot. Light.
# - `models/<run_name>/best_model/` — final weights + best epoch's artifacts. Heavy.

# %%
@dataclass
class Config:
    # -- Target preset (resolver overwrites the data fields below) -----------
    target: str = "parlaspeech_primary_stress_frames" #"parlaspeech_fp_frames"
    # Languages to pool. () = all langs the target supports that have a JSONL on
    # disk; e.g. ("hr",) for Croatian only, ("hr", "rs") for a HR+RS mix.
    langs: tuple = ()

    # -- Run-mode caps (set by apply_mode from MODES[RUN_MODE]) --------------
    # None = no cap (full). cap_split applies these identically to train/dev/test,
    # so a capped run never silently leaves TEST at full size.
    cap_train: int | None = None
    cap_dev:   int | None = None
    cap_test:  int | None = None

    # -- Data (overwritten by resolve_target) -------------------------------
    jsonl_paths: list = field(default_factory=list)   # one per pooled lang
    label_key:   str = ""
    task_type:   str = "classification"   # classification-only in this notebook
    label_order: list = field(default_factory=lambda: [0, 1])

    # -- Frame alignment -----------------------------------------------------
    # Wav2Vec2 strides 320 samples @ 16 kHz → nominal 50 Hz output. Source labels
    # must be 50 Hz; per-record alignment to the model's true frame count happens
    # in preprocess. Any other source rate is a hard error.
    required_frame_rate_hz: int = 50

    # -- Model ---------------------------------------------------------------
    model_name: str              = "facebook/wav2vec2-base"
    freeze_feature_encoder: bool = True
    head_dropout: float          = 0.1

    # -- Training ------------------------------------------------------------
    batch_size: int      = 16
    grad_accum: int      = 1
    learning_rate: float = 1e-5
    num_epochs: int      = 3      # full-run default; modes override (test=1, demo=2)
    max_grad_norm: float = 1.0
    warmup_ratio: float  = 0.10
    logging_steps: int   = 50

    # Rough ETA seed: train records processed per second (base model on GPU).
    # A deliberate guess — recalibrated from phase 1's real duration.
    eta_rec_per_s_guess: float = 25.0

    # -- Output --------------------------------------------------------------
    runs_dir: str   = "runs"
    models_dir: str = "models"

    # -- Best-epoch selection ------------------------------------------------
    # "frame_macro_f1" | "frame_accuracy" | "frame_f1_positive" (all higher-better)
    best_metric: str = "frame_macro_f1"

    # -- Audio length cap ----------------------------------------------------
    # OFF by default; when ON, longer records are DROPPED (never truncated).
    enable_max_audio_seconds: bool = False
    max_audio_seconds: float       = 15.0

    # -- Preprocessing -------------------------------------------------------
    preprocess_batch_size: int  = 32
    dataloader_num_workers: int = 8    # drop to 0–2 on the CPU box if RAM-bound
    map_num_proc: int = 8              # parallel workers for the .map() feature pass

    # -- Visualization -------------------------------------------------------
    # How many eval records to render as gold/pred frame strips per epoch.
    n_examples_to_plot: int = 6

    # -- Hardware ------------------------------------------------------------
    # Honored from the GPU guard (cell above). GPU 2 reserved; never touch others.
    use_cuda: bool = True


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
print(f"langs       = {cfg.langs or '(all available)'} → {len(cfg.jsonl_paths)} JSONL(s)")
for _p in cfg.jsonl_paths:
    print(f"   • {_p}")
print(f"label_key   = {cfg.label_key}")
print(f"task_type   = {cfg.task_type}")
print(f"label_order = {cfg.label_order}")
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
    if cfg.task_type == "regression":
        raise NotImplementedError(
            "Frame-level regression is deferred to a future chapter. "
            "v1 supports task_type='classification' only — see 42_train_frame_regression (future)."
        )
    if cfg.task_type != "classification":
        raise ValueError(f"task_type must be 'classification', got {cfg.task_type!r}")
    if not cfg.label_order or len(cfg.label_order) < 2:
        raise ValueError(
            "Config.label_order is REQUIRED and must have ≥2 entries. Binary FP = [0, 1]."
        )
    if len(set(cfg.label_order)) != len(cfg.label_order):
        raise ValueError(f"label_order has duplicates: {cfg.label_order}")
    if cfg.required_frame_rate_hz != 50:
        raise ValueError(
            f"required_frame_rate_hz must be 50 in v1 (Wav2Vec2 stride = 320 @ 16 kHz). "
            f"Got {cfg.required_frame_rate_hz}. Resampling source labels is deferred."
        )
    if cfg.best_metric not in ("frame_macro_f1", "frame_accuracy", "frame_f1_positive"):
        raise ValueError(f"best_metric: invalid {cfg.best_metric!r}")

validate_config(cfg)
print("✅ config valid")

# %% [markdown]
# # Data

# %% [markdown]
# ## Load JSONL, filter to records that carry `label_key`
#
# A frame record qualifies only if `labels[label_key]` is a **non-empty list**.
# `cap_split` then trims each split to the active mode's cap (`None` = whole
# split) — applied identically to train, dev, **and** test, so a capped run never
# leaves TEST at full size. Sampling (`DEMO_SAMPLING`) only matters when >1 lang
# is pooled.

# %%
mark("data prep")

CAP_SEED = 1234   # deterministic shuffle seed for cap_split


def load_split(jsonl_paths: list, split: str, label_key: str) -> list[dict]:
    """Concatenate `split` records carrying a non-empty `label_key` sequence across
    every pooled lang JSONL. Records keep their own `dataset` tag for per-lang
    analysis downstream."""
    out = []
    for jp in jsonl_paths:
        for r in udp.iter_jsonl(jp):
            if r["split"] != split:
                continue
            v = r.get("labels", {}).get(label_key)
            if v is None or not isinstance(v, list) or len(v) == 0:
                continue
            out.append(r)
    return out


train_records = load_split(cfg.jsonl_paths, "train", cfg.label_key)
dev_records   = load_split(cfg.jsonl_paths, "dev",   cfg.label_key)
test_records  = load_split(cfg.jsonl_paths, "test",  cfg.label_key)

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
if len(cfg.jsonl_paths) > 1:
    _mix = Counter(r["dataset"] for r in train_records)
    print(f"   train lang mix: {dict(_mix)}")
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
# ## Frame-rate guard
#
# Every frame record must declare `frame_rate_hz == 50`. Source labels at any
# other rate are a hard error (resampling source labels is deferred); the
# *model*-frame alignment to ~49 Hz happens later, in preprocess.

# %%
def validate_frame_rate(records: list[dict], required_hz: int) -> None:
    missing = [r["instance_id"] for r in records if r.get("frame_rate_hz") is None]
    bad = [(r["instance_id"], r["frame_rate_hz"]) for r in records
           if r.get("frame_rate_hz") is not None and r["frame_rate_hz"] != required_hz]
    if missing:
        raise ValueError(
            f"{len(missing)} records missing frame_rate_hz. Examples: {missing[:5]}. "
            f"Every frame-level record must declare its rate."
        )
    if bad:
        rates = sorted({b[1] for b in bad})
        raise ValueError(
            f"{len(bad)} records have frame_rate_hz != {required_hz} (rates seen: {rates}). "
            f"Examples: {bad[:5]}. v1 supports only {required_hz} Hz."
        )


for _split, _recs in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
    validate_frame_rate(_recs, cfg.required_frame_rate_hz)
print(f"✅ all records carry frame_rate_hz={cfg.required_frame_rate_hz}")

# %% [markdown]
# ## Optional length cap
#
# OFF by default. When `enable_max_audio_seconds` is on, records whose WAV is
# longer than `max_audio_seconds` are **dropped** (never truncated — truncation
# would desync the frame labels).

# %%
def drop_long_records(records: list[dict], max_s: float) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for r in records:
        try:
            dur = udp.get_wav_duration(r["audio_path"])
        except Exception as e:
            print(f"⚠️  could not read duration for {r['instance_id']}: {e}; keeping")
            kept.append(r)
            continue
        if dur > max_s:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


if cfg.enable_max_audio_seconds:
    udp.banner(f"applying max_audio_seconds = {cfg.max_audio_seconds}s", char="-")
    train_records, n_tr = drop_long_records(train_records, cfg.max_audio_seconds)
    dev_records,   n_dv = drop_long_records(dev_records,   cfg.max_audio_seconds)
    test_records,  n_te = drop_long_records(test_records,  cfg.max_audio_seconds)
    print(f"dropped (train={n_tr}, dev={n_dv}, test={n_te})")
    print(f"kept    (train={len(train_records)}, dev={len(dev_records)}, test={len(test_records)})")
    if not train_records or not dev_records or not test_records:
        raise ValueError("a split became empty after applying max_audio_seconds")
else:
    print("max_audio_seconds: disabled (cfg.enable_max_audio_seconds=False)")

# %% [markdown]
# ## Labels & mappings
#
# `label_order` is the source of truth: `label2id[label] = index`. Any frame
# label in the data that isn't in `label_order` is a hard error. The per-split
# table counts **frames** (tokens), not records — this is where you eyeball the
# FP-frame class imbalance (positives are rare).

# %%
label2id = {lab: i for i, lab in enumerate(cfg.label_order)}
id2label = {i: lab for i, lab in enumerate(cfg.label_order)}
num_labels = len(cfg.label_order)

seen = set()
for r in train_records + dev_records + test_records:
    seen.update(r["labels"][cfg.label_key])
unknown = seen - set(label2id)
if unknown:
    raise ValueError(
        f"Found frame labels not in Config.label_order: {sorted(unknown)}. "
        f"Either add them to label_order or fix the data."
    )


def frame_counts(records: list[dict], label_key: str) -> Counter:
    c = Counter()
    for r in records:
        c.update(r["labels"][label_key])
    return c


ftr = frame_counts(train_records, cfg.label_key)
fdv = frame_counts(dev_records,   cfg.label_key)
fte = frame_counts(test_records,  cfg.label_key)
print(f"Frame labels ({num_labels}, canonical order) — counts are FRAMES, not records:")
print(f"   {'class':>10}  {'train':>12}  {'dev':>12}  {'test':>12}")
for lab in cfg.label_order:
    print(f"   {str(lab):>10}  {ftr.get(lab, 0):>12d}  {fdv.get(lab, 0):>12d}  {fte.get(lab, 0):>12d}")
print(f"   {'TOTAL':>10}  {sum(ftr.values()):>12d}  {sum(fdv.values()):>12d}  {sum(fte.values()):>12d}")

# %% [markdown]
# # Training engine
#
# Everything from here to just before the two phase calls is the shared engine:
# label alignment, feature extraction, collator, the per-frame head, metrics,
# per-epoch artifacts, and `run_phase`. The run-loop scaffolding (`run_phase`,
# two-phase split, GPU flush, timer) matches 31/32; the task-specific bodies are
# the per-frame head + token-CE + frame metrics/plots.

# %% [markdown]
# ## Label alignment to model frames
#
# Wav2Vec2's CNN front-end maps raw samples → frames via `out = (in - k)//s + 1`
# per conv layer. Nominal rate is 50 Hz but the real output is ~49 Hz, and it
# differs per record by a frame or two. We compute **each record's** true output
# length from the feature-extracted input length and resample its 50 Hz label
# sequence to that length by nearest-neighbor — so labels and model frames line
# up exactly before collation. (`align_labels_to_frames` handles both expansion
# and contraction.)

# %%
def compute_feat_extract_output_length(input_length: int,
                                       conv_kernel: list, conv_stride: list) -> int:
    """Replicate Wav2Vec2Model._get_feat_extract_output_lengths from config alone.
    For each CNN layer: out = (in - kernel) // stride + 1."""
    out = int(input_length)
    for k, s in zip(conv_kernel, conv_stride):
        out = (out - int(k)) // int(s) + 1
    return max(out, 0)


def align_labels_to_frames(label_seq, n_frames: int) -> list:
    """Resize a label sequence to length `n_frames` by nearest-neighbor index.
    Source labels (50 Hz) and model frames span the same duration; map model
    frame j → source index (j * L) // n_frames."""
    L = len(label_seq)
    if L == 0 or n_frames == 0:
        return []
    if L == n_frames:
        return list(label_seq)
    out = []
    for j in range(n_frames):
        i = (j * L) // n_frames
        out.append(label_seq[min(i, L - 1)])
    return out


def prepare_dataset_dict(records: list[dict], cfg: Config, label2id: dict) -> list[dict]:
    """Canonical records → list-of-dicts for Dataset.from_list. Frame labels are
    mapped through label2id (no-op for binary 0/1 with label_order [0,1])."""
    items = []
    for r in records:
        raw = r["labels"][cfg.label_key]
        items.append({
            "instance_id": r["instance_id"],
            "file_id":     r.get("file_id", ""),
            "audio_path":  str(udp.from_project_relative(r["audio_path"])),
            # Sub-clip bounds within audio_path (word_frame); None → whole file.
            "start_t":     r.get("start_t"),
            "end_t":       r.get("end_t"),
            "labels":      [label2id[v] for v in raw],
        })
    return items


def make_preprocess_function(feature_extractor, conv_kernel: list, conv_stride: list):
    """Factory: bind the model's CNN geometry into preprocess_function."""

    def preprocess_function(examples):
        """Load each WAV, run the feature extractor (no pad), align labels to the
        model's per-record frame count. If a record carries start_t/end_t (e.g.
        word_frame), slice that sub-clip from audio_path in memory first — no word
        WAVs on disk."""
        audio_arrays = []
        for path, st, et in zip(examples["audio_path"], examples["start_t"], examples["end_t"]):
            data, sr = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            if sr != 16000:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            if st is not None and et is not None:
                a = max(0, int(round(st * 16000)))
                b = min(len(data), int(round(et * 16000)))
                data = data[a:b]
            audio_arrays.append(data)

        inputs = feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors=None, padding=False,
        )

        aligned_labels = []
        for input_values, label_seq in zip(inputs["input_values"], examples["labels"]):
            n_frames = compute_feat_extract_output_length(
                len(input_values), conv_kernel, conv_stride,
            )
            aligned_labels.append(align_labels_to_frames(label_seq, n_frames))

        return {"input_values": inputs["input_values"], "labels": aligned_labels}

    return preprocess_function

# %% [markdown]
# ## Data collator (pad audio + frame labels within each batch)
#
# The feature extractor doesn't pad; this does — **and emits the
# `attention_mask`** so the encoder ignores padded audio frames. (wav2vec2-base
# ships `return_attention_mask=False`; we force it on at load and pass it here.)
# Frame labels are padded to the batch's longest sequence with `IGNORE_INDEX`,
# which the loss and every metric mask out. The `task_type` arg keeps the
# signature aligned with the future `42_train_frame_regression` twin.

# %%
class DataCollatorForFrame:
    def __init__(self, feature_extractor, task_type: str = "classification"):
        self.feature_extractor = feature_extractor
        self.task_type = task_type

    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.feature_extractor.pad(
            {"input_values": input_values},
            padding=True, return_attention_mask=True, return_tensors="pt",
        )

        max_len = max(len(l) for l in labels)
        padded = []
        for l in labels:
            l = l.tolist() if torch.is_tensor(l) else list(l)
            padded.append(l + [IGNORE_INDEX] * (max_len - len(l)))
        label_dtype = torch.long if self.task_type == "classification" else torch.float32
        batch["labels"] = torch.tensor(padded, dtype=label_dtype)
        return batch

# %% [markdown]
# ## Per-frame model head
#
# `Wav2Vec2Model` body + dropout + a `Linear(hidden, num_labels)` applied at
# **every frame** → logits `(B, T, num_labels)`. Token cross-entropy with
# `ignore_index=-100`. The forward pass re-aligns label length to the model's
# actual output `T` (truncate/pad with `-100`) as a belt-and-suspenders guard on
# top of the per-record preprocess alignment — normally a no-op.

# %%
class Wav2Vec2ForFrameClassification(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 + dropout + per-frame Linear classifier. Token-CE loss."""

    def __init__(self, config, head_dropout: float = 0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden = self.dropout(outputs.last_hidden_state)   # (B, T, H)
        logits = self.classifier(hidden)                   # (B, T, num_labels)

        loss = None
        if labels is not None:
            T_model, T_lab = logits.shape[1], labels.shape[1]
            if T_lab > T_model:
                labels = labels[:, :T_model]
            elif T_lab < T_model:
                pad = torch.full(
                    (labels.shape[0], T_model - T_lab),
                    IGNORE_INDEX, dtype=labels.dtype, device=labels.device,
                )
                labels = torch.cat([labels, pad], dim=1)
            fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            loss = fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))
        return {"loss": loss, "logits": logits}


def build_model(cfg: Config, num_labels: int, label2id, id2label):
    # HF transformers 5.x requires label2id keys to be str on from_pretrained.
    hf_label2id = {str(k): int(v) for k, v in label2id.items()}
    hf_id2label = {int(k): str(v) for k, v in id2label.items()}

    config_obj = AutoConfig.from_pretrained(
        cfg.model_name, num_labels=num_labels,
        label2id=hf_label2id, id2label=hf_id2label,
    )
    model = Wav2Vec2ForFrameClassification(config_obj, head_dropout=cfg.head_dropout)
    # Load pretrained body weights into the `wav2vec2` submodule.
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
# Frame-level, computed over **all non-pad frames** flattened across the batch:
# accuracy, macro-F1, and positive-class F1 (the last id in `label_order` — class
# 1 for binary FP). Positive-class F1 is the one to watch given the imbalance.

# %%
def compute_frame_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)          # (B, T)
    mask = labels != IGNORE_INDEX
    y_true = labels[mask]
    y_pred = preds[mask]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    n_classes = int(logits.shape[-1])
    per_class_f1 = f1_score(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0,
    )
    f1_pos = float(per_class_f1[n_classes - 1])   # last class = positive (1 for binary)

    return {
        "frame_accuracy": acc,
        "frame_macro_f1": macro_f1,
        "frame_f1_positive": f1_pos,
    }

# %% [markdown]
# ## Per-epoch artifacts
#
# `predictions.json` (per-record gold/pred frame sequences, pad-trimmed) and a
# `example_predictions.png` strip plot (gold over pred, per record) — one set per
# epoch.

# %%
def save_predictions_json(predictions, labels, items, out_path: Path, id2label: dict):
    """predictions: (B, T, num_labels) logits → argmax. labels: (B, T) w/ -100 pad.
    Also stores per-frame positive-class probability (`prob_pos`, softmax of the
    last/positive class) for QC: peak = max(prob_pos) tells you how confident the
    model's chosen stress frame is, and the list shows where that peak sits."""
    pred_idx = np.argmax(predictions, axis=-1) if predictions.ndim == 3 else predictions
    prob_pos_all = None
    if predictions.ndim == 3:
        z = predictions - predictions.max(axis=-1, keepdims=True)
        ez = np.exp(z)
        prob_pos_all = (ez / ez.sum(axis=-1, keepdims=True))[..., -1]   # positive = last class
    out = []
    for i, item in enumerate(items):
        valid = labels[i] != IGNORE_INDEX
        gold_seq = labels[i][valid].tolist()
        pred_seq = pred_idx[i][valid].tolist()
        rec = {
            "instance_id": item["instance_id"],
            "file_id":     item.get("file_id", ""),
            "n_frames":    int(len(gold_seq)),
            "gold":        [id2label[int(g)] for g in gold_seq],
            "pred":        [id2label[int(p)] for p in pred_seq],
            "gold_raw":    [int(g) for g in gold_seq],
            "pred_raw":    [int(p) for p in pred_seq],
        }
        if prob_pos_all is not None:
            rec["prob_pos"] = [round(float(x), 4) for x in prob_pos_all[i][valid]]
        out.append(rec)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))


# A simple categorical palette. Extend if num_labels > 8.
_PALETTE = ["#dddddd", "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
            "#ff7f0e", "#8c564b", "#e377c2"]


def _row_to_image(seq, n_classes: int):
    """Turn a 1-D label sequence into a (1, T, 4) RGBA image via palette."""
    pal = (_PALETTE * ((n_classes // len(_PALETTE)) + 1))[:n_classes]
    rgba = np.zeros((1, len(seq), 4), dtype=float)
    for t, v in enumerate(seq):
        rgba[0, t] = mcolors.to_rgba(pal[int(v)])
    return rgba, pal


def plot_example_predictions(predictions, labels, items, out_path: Path,
                             id2label: dict, n_examples: int):
    pred_idx = np.argmax(predictions, axis=-1) if predictions.ndim == 3 else predictions
    n = min(n_examples, len(items))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(10, max(2, 0.9 * n)), squeeze=False)
    axes = axes[:, 0]
    n_classes = len(id2label)
    pal = None

    for i in range(n):
        valid = labels[i] != IGNORE_INDEX
        gold_img, pal = _row_to_image(labels[i][valid], n_classes)
        pred_img, _   = _row_to_image(pred_idx[i][valid], n_classes)
        strip = np.concatenate([gold_img, pred_img], axis=0)   # (2, T, 4)
        ax = axes[i]
        ax.imshow(strip, aspect="auto", interpolation="nearest")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["gold", "pred"], fontsize=8)
        ax.set_xticks([])
        title = items[i]["instance_id"]
        ax.set_title(("..." + title[-57:]) if len(title) > 60 else title,
                     fontsize=8, loc="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=pal[k]) for k in range(n_classes)]
    fig.legend(handles, [str(id2label[k]) for k in range(n_classes)],
               loc="lower center", ncol=min(n_classes, 6), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## `EpochCheckpointCallback`
#
# Evaluates the eval set every epoch and writes per-epoch logs + plots. Does
# **not** save weights — only the best epoch's model is saved, and only in phase 2.

# %%
class EpochCheckpointCallback(TrainerCallback):
    def __init__(self, phase_dir: Path, eval_dataset, eval_items,
                 compute_metrics, data_collator, cfg: Config,
                 label_order: list, id2label: dict):
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
            compute_metrics=self.compute_metrics, data_collator=self.data_collator,
        )
        out = eval_trainer.predict(self.eval_dataset)

        # HF predict uses 'test_' prefix; normalize to 'eval_'.
        metrics = {k.replace("test_", "eval_"): v for k, v in out.metrics.items()}

        train_loss = None
        for log in reversed(state.log_history):
            if "loss" in log:
                train_loss = log["loss"]; break

        epoch_info = {"epoch": epoch, "train_loss": train_loss, **metrics}
        self.epoch_results.append(epoch_info)

        save_predictions_json(
            out.predictions, out.label_ids, self.eval_items,
            epoch_dir / "predictions.json", id2label=self.id2label,
        )
        plot_example_predictions(
            out.predictions, out.label_ids, self.eval_items,
            epoch_dir / "example_predictions.png",
            id2label=self.id2label, n_examples=self.cfg.n_examples_to_plot,
        )
        (epoch_dir / "epoch_summary.json").write_text(json.dumps(epoch_info, indent=2))

        print(f"   epoch={epoch}  "
              f"macroF1={metrics.get('eval_frame_macro_f1', 0):.4f}  "
              f"acc={metrics.get('eval_frame_accuracy', 0):.4f}  "
              f"F1+={metrics.get('eval_frame_f1_positive', 0):.4f}")

# %% [markdown]
# ## `run_phase` — train + evaluate + save logs
#
# One function for both phases. Differences are only *which* records train, which
# split evaluates, and whether the best model is saved. Frees the model + CUDA
# cache on the way out so the next phase starts clean. Same scaffolding as 31/32.

# %%
def best_epoch_of(epoch_results: list[dict], cfg: Config) -> dict:
    m = cfg.best_metric   # all frame metrics are higher-is-better
    return max(epoch_results,
               key=lambda r: (r.get(f"eval_{m}", float("-inf"))
                              if r.get(f"eval_{m}") is not None else float("-inf")))


def run_phase(*, phase_name: str, train_records: list[dict], eval_records: list[dict],
              eval_split_name: str, save_best_model: bool,
              cfg: Config, run_dir: Path, model_dir: Path, feature_extractor,
              label2id, id2label) -> tuple[list[dict], dict]:
    udp.banner(f"PHASE: {phase_name}  (train→{eval_split_name})")
    phase_dir = run_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    # CNN geometry (static config fields) → drives per-record frame alignment.
    print(f"loading config: {cfg.model_name}")
    model_config = AutoConfig.from_pretrained(cfg.model_name)
    conv_kernel = list(model_config.conv_kernel)
    conv_stride = list(model_config.conv_stride)
    print(f"   conv_kernel = {conv_kernel}")
    print(f"   conv_stride = {conv_stride}")

    train_items = prepare_dataset_dict(train_records, cfg, label2id)
    eval_items  = prepare_dataset_dict(eval_records,  cfg, label2id)
    train_ds = Dataset.from_list(train_items)
    eval_ds  = Dataset.from_list(eval_items)

    preprocess_fn = make_preprocess_function(feature_extractor, conv_kernel, conv_stride)
    print(f"preprocessing {len(train_ds)} train + {len(eval_ds)} eval (batch_size={cfg.preprocess_batch_size})…")
    train_ds = train_ds.map(
        preprocess_fn, batched=True, batch_size=cfg.preprocess_batch_size,
        remove_columns=train_ds.column_names, num_proc=cfg.map_num_proc,
    )
    eval_ds = eval_ds.map(
        preprocess_fn, batched=True, batch_size=cfg.preprocess_batch_size,
        remove_columns=eval_ds.column_names, num_proc=cfg.map_num_proc,
    )
    train_ds.set_format(type="torch", columns=["input_values", "labels"])
    eval_ds.set_format(type="torch",  columns=["input_values", "labels"])

    print(f"building model: {cfg.model_name}")
    model = build_model(cfg, num_labels=len(cfg.label_order),
                        label2id=label2id, id2label=id2label)

    data_collator = DataCollatorForFrame(feature_extractor, cfg.task_type)
    compute_metrics = compute_frame_metrics

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
        tf32=(DEVICE == "cuda"),
        dataloader_num_workers=cfg.dataloader_num_workers,
    )

    callback = EpochCheckpointCallback(
        phase_dir=phase_dir, eval_dataset=eval_ds, eval_items=eval_items,
        compute_metrics=compute_metrics, data_collator=data_collator, cfg=cfg,
        label_order=cfg.label_order, id2label=id2label,
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
# `runs/{dataset}_{label_key}_frame_{timestamp}/`. Snapshots the resolved Config.

# %%
_langs_in_run = sorted({r["dataset"].split("-")[-1].lower() for r in train_records})
dataset_name = "ParlaSpeech-" + "+".join(_langs_in_run)   # e.g. ParlaSpeech-hr+rs
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"{dataset_name}_{cfg.label_key}_frame_{ts}"

run_dir   = udp.from_project_relative(cfg.runs_dir)   / run_name
model_dir = udp.from_project_relative(cfg.models_dir) / run_name
run_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
print(f"run_dir   = {run_dir.relative_to(PROJECT_ROOT)}    (per-epoch logs)")
print(f"model_dir = {model_dir.relative_to(PROJECT_ROOT)}  (best_model goes here)")

from dataclasses import asdict as _asdict
(run_dir / "config.json").write_text(json.dumps(_asdict(cfg), indent=2, default=str))

# %% [markdown]
# ## Feature extractor
#
# Single load, reused across both phases. We force `return_attention_mask=True`
# so the collator hands the encoder a real mask — without it, padded audio frames
# leak into the per-frame representations.

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
print(f"task        : frame {cfg.task_type}")
print(f"label_key   : {cfg.label_key}")
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
# ## Inline frame confusion matrix (TEST, best epoch)
#
# Aggregated over **all TEST frames** (every non-pad frame across every record).
# Stacked: counts on top, row-normalized % below. Reads `predictions.json` from
# disk so this re-runs independently of training. For binary FP this is the 2×2
# that exposes the class imbalance head-on.

# %%
best_epoch_dir = run_dir / "phase2_test" / "epoch_logs" / f"epoch_{phase2_best['epoch']}"
preds_path = best_epoch_dir / "predictions.json"
if not preds_path.exists():
    raise FileNotFoundError(f"Expected predictions at {preds_path}, but it's missing.")

preds_data = json.loads(preds_path.read_text())
# Flatten every frame across every record.
y_true_idx, y_pred_idx = [], []
for p in preds_data:
    y_true_idx.extend(p["gold_raw"])
    y_pred_idx.extend(p["pred_raw"])

str_labels = [str(x) for x in cfg.label_order]
n = len(cfg.label_order)
cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(n)))
row_sums = cm.sum(axis=1, keepdims=True)
cm_rel = np.divide(cm.astype(float), row_sums,
                   out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

fig, axes = plt.subplots(2, 1, figsize=(max(6, n * 1.1), max(9, n * 1.8)))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
            xticklabels=str_labels, yticklabels=str_labels, ax=axes[0])
axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true")
axes[0].set_title(f"frame counts — epoch {phase2_best['epoch']} on TEST")

sns.heatmap(cm_rel, annot=True, fmt=".1f", cmap="viridis",
            xticklabels=str_labels, yticklabels=str_labels, ax=axes[1])
axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true")
axes[1].set_title(f"row-normalized %  — epoch {phase2_best['epoch']} on TEST")

plt.tight_layout()
cm_png = run_dir / "frame_confusion_matrix_test.png"
fig.savefig(cm_png, dpi=120, bbox_inches="tight")
print(f"saved {cm_png.relative_to(PROJECT_ROOT)}")
plt.show()

# %% [markdown]
# ## Inference spot-check
#
# A flat random sample is ~all negatives (FP frames are ~1–2% of frames), and a
# positive-frame *count* says nothing about whether predictions land in the right
# place. So instead: stratified selection — random positive- and negative-event
# records, plus the best/worst positive-event predictions by `SPOTCHECK_RANK_BY`
# — rendered as **gold-over-pred frame strips**, each stretched to equal width so
# misalignment is visible at a glance. Best/worst are ranked among positive-event
# records only (negatives are trivially near-perfect under the imbalance).
#
# `SPOTCHECK_RANK_BY` is a string switch over `_RANKERS` (`"pos_f1"` |
# `"accuracy"`; add more by extending the dict). Note: accuracy is frame-aligned
# but imbalance-dominated and barely moves under a few-frame shift — true
# misalignment-aware (tolerance-windowed) scoring is deferred to a later chapter.

# %%
import random as _rnd

# ── Spot-check selection ──────────────────────────────────────────────────────
SPOTCHECK_RANK_BY = "pos_f1"   # "pos_f1" | "accuracy"  — extend _RANKERS to add more
N_POS, N_NEG, N_BEST, N_WORST = 3, 3, 2, 2
SPOT_SEED = 0

_pos_id = cfg.label_order[-1]   # positive class (1 for binary FP)


def _acc(g, p, pos):
    return float((g == p).mean()) if len(g) else 0.0


def _pos_f1(g, p, pos):
    tp = int(((g == pos) & (p == pos)).sum())
    fp = int(((g != pos) & (p == pos)).sum())
    fn = int(((g == pos) & (p != pos)).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


# Best/worst rankers (higher = better). NB: accuracy is frame-aligned but
# imbalance-dominated — a few-frame shift barely moves it. Misalignment-aware
# scoring (tolerance-windowed events) is deferred to a later chapter; this panel
# is an eyeball aid, not a metric.
_RANKERS = {"pos_f1": _pos_f1, "accuracy": _acc}
if SPOTCHECK_RANK_BY not in _RANKERS:
    raise ValueError(f"SPOTCHECK_RANK_BY={SPOTCHECK_RANK_BY!r} not in {sorted(_RANKERS)}")
_rank = _RANKERS[SPOTCHECK_RANK_BY]

_rows = json.loads(preds_path.read_text())
for r in _rows:
    r["_g"] = np.asarray(r["gold_raw"]); r["_p"] = np.asarray(r["pred_raw"])
    r["_gold_pos"] = int((r["_g"] == _pos_id).sum())
    r["_pred_pos"] = int((r["_p"] == _pos_id).sum())
    r["_acc"] = _acc(r["_g"], r["_p"], _pos_id)
    r["_score"] = _rank(r["_g"], r["_p"], _pos_id)

pos_events = [r for r in _rows if r["_gold_pos"] > 0]
neg_events = [r for r in _rows if r["_gold_pos"] == 0]

_rng = _rnd.Random(SPOT_SEED)
_order = {"POS": 0, "NEG": 1, "BEST": 2, "WORST": 3}
picks: dict = {}   # instance_id -> {"rec": r, "tags": [...]}


def _add(recs, tag):
    for r in recs:
        picks.setdefault(r["instance_id"], {"rec": r, "tags": []})["tags"].append(tag)


_add(_rng.sample(pos_events, k=min(N_POS, len(pos_events))), "POS")
_add(_rng.sample(neg_events, k=min(N_NEG, len(neg_events))), "NEG")
_ranked = sorted(pos_events, key=lambda r: r["_score"], reverse=True)
_add(_ranked[:N_BEST], "BEST")
_add(_ranked[-N_WORST:], "WORST")

selected = sorted(picks.values(), key=lambda d: min(_order[t] for t in d["tags"]))


def _tagstr(tags):
    return "+".join(sorted(set(tags), key=lambda t: _order[t]))


udp.banner(f"INFERENCE SPOT-CHECK — {len(selected)} TEST examples "
           f"(best/worst by {SPOTCHECK_RANK_BY}, among positive-events)")
if not pos_events:
    print("⚠️  no positive-event records in TEST — best/worst panel is empty")
for d in selected:
    r = d["rec"]
    print(f"  [{_tagstr(d['tags']):<14}] {r['instance_id']}")
    print(f"     file={r.get('file_id','')}  frames={r['n_frames']}  "
          f"gold_pos={r['_gold_pos']}  pred_pos={r['_pred_pos']}  "
          f"acc={r['_acc']:.4f}  {SPOTCHECK_RANK_BY}={r['_score']:.4f}")

# ── Strips: gold (top) over pred (bottom), one record per row, equal width so
# alignment is visible regardless of length. Reuses _row_to_image / _PALETTE.
n = len(selected)
if n:
    n_classes = len(id2label)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(2, 0.85 * n)), squeeze=False)
    axes = axes[:, 0]
    # Row-specific palettes: negatives grey, gold positives orange, pred positives blue.
    _GOLD_PAL = ["#dddddd", "#FFA500"]
    _PRED_PAL = ["#dddddd", "#1f77b4"]

    def _img(seq, pal):
        rgba = np.zeros((1, len(seq), 4))
        for t, v in enumerate(seq):
            rgba[0, t] = mcolors.to_rgba(pal[int(v)])
        return rgba

    for i, d in enumerate(selected):
        r = d["rec"]
        gold_img = _img(r["_g"], _GOLD_PAL)
        pred_img = _img(r["_p"], _PRED_PAL)
        strip = np.concatenate([gold_img, pred_img], axis=0)   # (2, T, 4)
        ax = axes[i]
        ax.imshow(strip, aspect="auto", interpolation="nearest")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["gold", "pred"], fontsize=8)
        ax.set_xticks([])
        iid = r["instance_id"]
        iid = ("..." + iid[-44:]) if len(iid) > 47 else iid
        ax.set_title(f"[{_tagstr(d['tags'])}] {iid}  · {r['n_frames']}f · "
                     f"{SPOTCHECK_RANK_BY}={r['_score']:.3f}", fontsize=8, loc="left")
    
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#FFA500", "#1f77b4", "#dddddd")]
    fig.legend(handles, ["gold positive", "pred positive", "negative"],
               loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    strips_png = run_dir / "spot_check_strips.png"
    fig.savefig(strips_png, dpi=120, bbox_inches="tight")
    print(f"\nsaved {strips_png.relative_to(PROJECT_ROOT)}")
    plt.show()

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
# This run wrote per-epoch logs + a saved best model under `runs/`.
#
# - **Different target:** change `Config.target` — `parlaspeech_fp_frames` for
#   filled-pause frames, `parlaspeech_primary_stress_frames` (rung 6, the north
#   star) for HR/RS primary stress. The full menu prints from the Targets cell.
# - **Subset of languages:** set `cfg.langs = ("hr",)` (or any tuple of supported
#   codes); `()` pools every language the target supports that has a JSONL on
#   disk.
# - `42_train_frame_regression` is the deferred twin (no continuous per-frame
#   target yet).
