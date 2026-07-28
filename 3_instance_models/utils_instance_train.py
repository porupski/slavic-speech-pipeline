"""utils_instance_train — the chapter-3 instance-training engine.

Shared by:
  * 31_train_instance_classification.ipynb / run_31_classification.py
  * 32_train_instance_regression.ipynb     / run_32_regression.py

Everything here was lifted verbatim from the (now frozen) standalone twin
notebooks; only module hygiene changed — globals that the notebooks captured by
closure (``cfg``, ``normalizer``) are explicit parameters here.

IMPORT-ORDER CONTRACT (read this before importing):
  1. The GPU guard must run BEFORE this module is imported. This module imports
     torch at the top, and ``CUDA_VISIBLE_DEVICES`` only takes effect if set
     before torch's first CUDA touch. Consumers keep a tiny visible guard block
     (set ``CUDA_VISIBLE_DEVICES`` to the reserved GPU or ``""``), then import.
  2. ``HF_HOME`` is handled HERE (project-local ``stock_models/``), before the
     transformers import below — consumers don't need to set it.
  3. Headless runners should set ``MPLBACKEND=Agg`` before importing (notebooks
     don't need to — the inline backend is already active).

Config flow: ``load_config(path, task_type, run_mode=...)`` reads config.json
(shared block + per-task block + mode overrides), resolves the TARGETS preset,
and returns a validated-ready ``Config``. The notebooks/runners never define
hyperparameters in code.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path

# ── PROJECT_ROOT + HF_HOME (must precede the transformers import) ─────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_data_prep"))
import utils_dataprep as udp

PROJECT_ROOT = udp.PROJECT_ROOT
# Cache HF pretrained models in a project-local folder (gitignored) instead of
# ~/.cache/huggingface, so the download survives across machines / reclones.
# MUST be set BEFORE `from transformers import ...` — HF reads HF_HOME at
# import time and caches the resolved path internally.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "stock_models"))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import torch
import torch.nn as nn
from datasets import Dataset
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error,
)
from transformers import (
    AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification,
    Trainer, TrainerCallback, TrainingArguments,
    Wav2Vec2Model, Wav2Vec2PreTrainedModel,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage timing
# ═══════════════════════════════════════════════════════════════════════════════
# mark() stamps a milestone; print_stage_breakdown() prints a per-stage delta
# table. Stdlib-only, cheap, partial-run safe (prints whatever marks exist).

STAGE_TIMES: dict[str, float] = {}


def mark(stage: str) -> None:
    STAGE_TIMES[stage] = time.time()


def fmt_mmss(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    return f"{s // 60:02d}:{s % 60:02d}"


def print_stage_breakdown(times: dict[str, float] | None = None) -> None:
    items = list((STAGE_TIMES if times is None else times).items())
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


# ═══════════════════════════════════════════════════════════════════════════════
# TARGETS registry
# ═══════════════════════════════════════════════════════════════════════════════
# Union of the twins' registries: classification (31) + regression (32) presets
# in one dict; each entry carries its task_type, and load_config() enforces
# that the picked preset matches the notebook/runner you're in.
#
# A preset MAY optionally declare per-target constraints (see
# TARGET_CONSTRAINT_FIELDS below) — currently just `max_duration_s`. When
# present, the value overrides the Config default and any shared/task-block
# value in config.json; mode-level overrides still win.

TARGETS: dict = {
    # ---- ROG-Art (rog_instance.jsonl) ----------------------------------
    "rog_art_filled_pause": {
        "jsonl_path": "data/processed_jsonl/rog_instance.jsonl",
        "label_key":  "filled_pause_present",
        "task_type":  "classification",
        "label_order": [0, 1],
    },
    "rog_art_disfluency_count": {
        "jsonl_path": "data/processed_jsonl/rog_instance.jsonl",
        "label_key":  "disfluency_count",
        "task_type":  "classification",
        "label_order": [0, 1, 2],   # class 2 is ~0.5% of rows; macro-F1 will be brutal
    },

    # ---- ROG-Dialog (rog_dialog_instance.jsonl) ------------------------
    "rog_dia_sentiment": {
        "jsonl_path": "data/processed_jsonl/rog_dialog_instance.jsonl",
        "label_key":  "sentiment",
        "task_type":  "classification",
        "label_order": [
            "predominantlyNegative", "mixedNegative", "neutralNegative",
            "neutralPositive", "mixedPositive", "predominantlyPositive",
        ],
    },
    "rog_dia_sentiment_annotated": {
        "jsonl_path": "data/processed_jsonl/rog_dialog_instance.jsonl",
        "label_key":  "sentiment_annotated",
        "task_type":  "classification",
        "label_order": [
            "predominantlyNegative", "mixedNegative", "neutralNegative",
            "neutralPositive", "mixedPositive", "predominantlyPositive",
        ],
    },
    "rog_dia_dialogue_act_dimension": {
        "jsonl_path": "data/processed_jsonl/rog_dialog_instance.jsonl",
        "label_key":  "dialogue_act_dimension",
        "task_type":  "classification",
        # ~8 categories; non-ordinal, so Spearman is uninterpretable here.
        "label_order": [
            "task", "discourseStructuring", "autoFeedback", "timeManagement",
            "alloFeedback", "turnManagement", "ownCommunicationManagement",
            "unspecifiedDimension",
        ],
    },
    "rog_dia_dialogue_act_function": {
        "jsonl_path": "data/processed_jsonl/rog_dialog_instance.jsonl",
        "label_key":  "dialogue_act_function",
        "task_type":  "classification",
        # 30+ categories with a long tail; loader builds label_order from data.
        "label_order": None,
    },

    # ---- ParlaSpeech utterance_instance (generated per lang below) ------
}


def _add_parlaspeech_targets(targets: dict, langs=("hr", "rs", "pl", "cz")) -> None:
    """ParlaSpeech utterance_instance presets per lang: classification
    (gender + filled-pause presence/count) and regression (sentiment, age)."""
    for l in langs:
        path = f"data/processed_jsonl/parlaspeech_{l}_utterance_instance.jsonl"
        targets[f"parlaspeech_{l}_gender"] = {
            "jsonl_path": path, "label_key": "speaker_gender",
            "task_type": "classification", "label_order": ["M", "F"],
        }
        targets[f"parlaspeech_{l}_fp_present"] = {
            "jsonl_path": path, "label_key": "filled_pause_present",
            "task_type": "classification", "label_order": [0, 1],
        }
        targets[f"parlaspeech_{l}_fp_count"] = {
            "jsonl_path": path, "label_key": "filled_pause_count",
            "task_type": "classification", "label_order": None,  # built from data
        }
        targets[f"parlaspeech_{l}_sentiment"] = {
            "jsonl_path": path, "label_key": "sentiment_logit",
            "task_type": "regression", "label_order": None,
        }
        targets[f"parlaspeech_{l}_age"] = {
            "jsonl_path": path, "label_key": "speaker_age",
            "task_type": "regression", "label_order": None,
        }


_add_parlaspeech_targets(TARGETS)

def _add_hr_benchmark_v1_targets(targets: dict) -> None:
    """ParlaSpeech-HR-benchmark-v1 presets (11e output): one JSONL per task,
    splits baked in by the benchmark construction. ALL FOUR are classification —
    v1's age is a young/old group, not a continuous value (and there's no
    orientation tier). Labels are normalized to v3's casing in 11e, so M/F and
    Coalition/Opposition orders are shared with the v3 family."""
    base = "data/processed_jsonl/parlaspeech_hr_bench_v1_"
    targets["hr_bench_v1_gender"] = {
        "jsonl_path": f"{base}gender.jsonl", "label_key": "speaker_gender",
        "task_type": "classification", "label_order": ["M", "F"],
    }
    targets["hr_bench_v1_speaker_id"] = {
        "jsonl_path": f"{base}speaker_id.jsonl", "label_key": "speaker_name",
        "task_type": "classification", "label_order": None,  # 50 classes, built from data
    }
    targets["hr_bench_v1_power_status"] = {
        "jsonl_path": f"{base}power_status.jsonl", "label_key": "power_status",
        "task_type": "classification", "label_order": ["Coalition", "Opposition"],
    }
    targets["hr_bench_v1_age"] = {
        "jsonl_path": f"{base}age.jsonl", "label_key": "speaker_age_group",
        "task_type": "classification", "label_order": ["young", "old"],
    }

_add_hr_benchmark_v1_targets(TARGETS)


def _add_hr_benchmark_v3_targets(targets: dict) -> None:
    """ParlaSpeech-HR-benchmark-v3 presets (11d output): one JSONL per task,
    splits baked in by the benchmark construction. gender/speaker_id/power_status
    are classification; age/orientation are regression (v3 ships continuous age)."""
    base = "data/processed_jsonl/parlaspeech_hr_bench_v3_"
    targets["hr_bench_v3_gender"] = {
        "jsonl_path": f"{base}gender.jsonl", "label_key": "speaker_gender",
        "task_type": "classification", "label_order": ["M", "F"],
    }
    targets["hr_bench_v3_speaker_id"] = {
        "jsonl_path": f"{base}speaker_id.jsonl", "label_key": "speaker_name",
        "task_type": "classification", "label_order": None,  # 50 classes, built from data
    }
    targets["hr_bench_v3_power_status"] = {
        "jsonl_path": f"{base}power_status.jsonl", "label_key": "power_status",
        "task_type": "classification", "label_order": ["Coalition", "Opposition"],
    }
    targets["hr_bench_v3_age"] = {
        "jsonl_path": f"{base}age.jsonl", "label_key": "speaker_age",
        "task_type": "regression", "label_order": None,
    }
    targets["hr_bench_v3_orientation"] = {
        "jsonl_path": f"{base}orientation.jsonl", "label_key": "orientation",
        "task_type": "regression", "label_order": None,
    }

_add_hr_benchmark_v3_targets(TARGETS)


def available_targets(task_type: str | None = None) -> list[str]:
    """Sorted preset names, optionally filtered to one task_type."""
    if task_type is None:
        return sorted(TARGETS)
    return sorted(k for k, t in TARGETS.items() if t["task_type"] == task_type)


# Optional per-target constraint fields. If a TARGETS entry sets one,
# resolve_target applies it to cfg in addition to the data fields above —
# mode-level overrides in config.json still win (they apply later in
# load_config). Add to this tuple to wire more constraints through.
TARGET_CONSTRAINT_FIELDS: tuple = ("max_duration_s",)


def resolve_target(cfg, targets: dict = TARGETS) -> None:
    """Overwrite jsonl_path/label_key/task_type/label_order from the picked
    preset, and apply any optional per-target constraints declared by the preset
    (see ``TARGET_CONSTRAINT_FIELDS``). Mutates cfg in place. Raises if
    cfg.target isn't a known key."""
    if cfg.target not in targets:
        raise ValueError(
            f"Config.target={cfg.target!r} not in TARGETS. Known: {sorted(targets)}"
        )
    t = targets[cfg.target]
    cfg.jsonl_path  = t["jsonl_path"]
    cfg.label_key   = t["label_key"]
    cfg.task_type   = t["task_type"]
    cfg.label_order = t["label_order"]   # may be None — built from data later
    for key in TARGET_CONSTRAINT_FIELDS:
        if key in t:
            setattr(cfg, key, t[key])


# ═══════════════════════════════════════════════════════════════════════════════
# Config — one dataclass for both twins; config.json fills it
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # -- Target preset (resolve_target overwrites the data fields below) -----
    target: str = ""

    # -- Data (overwritten by resolve_target) -------------------------------
    jsonl_path:  str  = ""
    label_key:   str  = ""
    task_type:   str  = "classification"   # "classification" | "regression"
    label_order: list | None = None

    # -- Regression-only knobs (ignored by classification) -------------------
    # Optional class→float map. If None, labels must already be numeric.
    label_scale: dict | None = None
    # Target normalization: "none" | "zscore" (fit on TRAIN only).
    normalize:   str  = "zscore"
    loss_function: str = "mse"   # "mse" | "l1"

    # -- Run mode (set from config.json / CLI; apply_mode layers the caps) ----
    run_mode: str = "full"                  # "test" | "demo" | "full"
    demo_sampling: str = "proportional"     # "proportional" | "balanced" — only when pooling >1 lang
    cap_seed: int = 1234                    # deterministic shuffle seed for cap_split

    # -- Run-mode caps (set by apply_mode from the modes block) --------------
    # None = no cap (full). cap_split applies these identically to train/dev/test,
    # so a capped run never silently leaves TEST at full size.
    cap_train: int | None = None
    cap_dev:   int | None = None
    cap_test:  int | None = None

    # -- Model ---------------------------------------------------------------
    model_name: str              = "facebook/wav2vec2-base"
    freeze_feature_encoder: bool = True

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
    # GPU pinning happens in the consumer's guard BEFORE this module is
    # imported; use_cuda just records the outcome (see resolve_device).
    reserved_gpu: str = "2"            # physical GPU reserved for this project
    use_cuda: bool   = True


def apply_mode(cfg: Config, overrides: dict) -> None:
    """Layer a mode's overrides onto the base (full) Config, in place. Every key
    must name a real Config field — a typo'd knob is a hard error, not a silent
    no-op."""
    valid = {f.name for f in fields(cfg)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"mode overrides name unknown Config fields: {sorted(unknown)}")
    for k, v in overrides.items():
        setattr(cfg, k, v)


def load_config(path: str | Path, task_type: str,
                run_mode: str | None = None) -> tuple[Config, dict]:
    """Build a Config from config.json for one task_type.

    Layering order (later wins): Config defaults → ``shared`` block →
    ``classification``/``regression`` block → ``modes[run_mode]`` overrides.
    ``run_mode`` argument (CLI / notebook) overrides the file's ``run_mode``.

    Returns ``(cfg, raw)`` — raw is the parsed json, kept for provenance.
    Every json key must name a real Config field (same typo guard as
    apply_mode). The picked target preset must match ``task_type``.
    """
    raw = json.loads(Path(path).read_text())
    if task_type not in ("classification", "regression"):
        raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}")

    cfg = Config(task_type=task_type)
    apply_mode(cfg, raw.get("shared", {}))          # reuses the typo guard
    apply_mode(cfg, raw.get(task_type, {}))

    cfg.run_mode = run_mode if run_mode is not None else raw.get("run_mode", cfg.run_mode)
    modes = raw.get("modes", {})
    if cfg.run_mode not in modes:
        raise ValueError(f"run_mode={cfg.run_mode!r} not in config modes: {sorted(modes)}")

    resolve_target(cfg, TARGETS)
    if cfg.task_type != task_type:
        raise ValueError(
            f"target {cfg.target!r} is a {cfg.task_type} preset, but this is the "
            f"{task_type} pipeline. Pick a {task_type} target "
            f"(see available_targets({task_type!r}))."
        )

    apply_mode(cfg, modes[cfg.run_mode])
    return cfg, raw


def validate_config(cfg: Config) -> None:
    """Union of the twins' validators, branched by task_type."""
    if cfg.run_mode not in ("test", "demo", "full"):
        raise ValueError(f"run_mode invalid: {cfg.run_mode!r} (choose test|demo|full)")
    if cfg.demo_sampling not in ("proportional", "balanced"):
        raise ValueError(f"demo_sampling invalid: {cfg.demo_sampling!r} (proportional|balanced)")
    if cfg.task_type == "classification":
        # label_order=None is allowed — it signals "build from data union after load".
        if cfg.label_order is not None and len(set(cfg.label_order)) != len(cfg.label_order):
            raise ValueError(f"label_order has duplicates: {cfg.label_order}")
        if cfg.best_metric_classification not in ("macro_f1", "accuracy", "spearman"):
            raise ValueError(f"best_metric_classification: invalid {cfg.best_metric_classification!r}")
    elif cfg.task_type == "regression":
        if cfg.loss_function not in ("mse", "l1"):
            raise ValueError(f"loss_function must be 'mse' or 'l1', got {cfg.loss_function!r}")
        if cfg.best_metric_regression not in ("spearman", "mse", "mae"):
            raise ValueError(f"best_metric_regression: invalid {cfg.best_metric_regression!r}")
        if cfg.normalize not in ("none", "zscore"):
            raise ValueError(f"normalize must be 'none' or 'zscore', got {cfg.normalize!r}")
    else:
        raise ValueError(f"task_type invalid: {cfg.task_type!r}")


def resolve_device(cfg: Config, use_cuda: bool) -> str:
    """Honor the consumer's GPU guard. CUDA_VISIBLE_DEVICES was already pinned
    (or blanked) BEFORE this module imported torch — the only point where
    pinning works. We do NOT touch CUDA_VISIBLE_DEVICES here."""
    cfg.use_cuda = use_cuda
    if cfg.use_cuda and torch.cuda.is_available():
        device = "cuda"   # = cuda:0 in-process, pinned to the reserved physical GPU
    elif cfg.use_cuda and not torch.cuda.is_available():
        print("⚠️  GPU selected but torch.cuda.is_available()==False; falling back to CPU")
        device = "cpu"
    else:
        device = "cpu"
    return device


def print_config_summary(cfg: Config, device: str) -> None:
    print(f"target      = {cfg.target}")
    print(f"run_mode    = {cfg.run_mode}  (caps train/dev/test = {cfg.cap_train}/{cfg.cap_dev}/{cfg.cap_test}, sampling={cfg.demo_sampling})")
    print(f"jsonl_path  = {cfg.jsonl_path}")
    print(f"label_key   = {cfg.label_key}")
    print(f"task_type   = {cfg.task_type}")
    if cfg.task_type == "classification":
        print(f"label_order = {cfg.label_order}")
    else:
        print(f"normalize   = {cfg.normalize}")
    print(f"device      = {device}")
    if device == "cuda":
        print(f"✓ visible devices : {torch.cuda.device_count()}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        print(f"✓ device name     : {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Data — load splits, cap them, rough ETA
# ═══════════════════════════════════════════════════════════════════════════════

def load_split(jsonl_path: str, split: str, label_key: str, max_duration_s: float) -> list[dict]:
    """Records of one split that carry a non-null ``label_key``, dropping
    instances longer than ``max_duration_s`` (padding-waste OOM guard)."""
    out, n_long = [], 0
    for r in udp.iter_jsonl(jsonl_path):
        if r["split"] != split:
            continue
        if label_key not in r.get("labels", {}):
            continue
        if r["labels"][label_key] is None:
            continue
        if r.get("metadata", {}).get("audio_length", 0.0) > max_duration_s:
            n_long += 1
            continue
        out.append(r)
    if n_long:
        print(f"  dropped {n_long} {split} records > {max_duration_s}s")
    return out


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
    raise ValueError(f"demo_sampling must be 'proportional' or 'balanced', got {sampling!r}")


def load_and_cap_splits(cfg: Config) -> tuple[list[dict], list[dict], list[dict]]:
    """Load train/dev/test for cfg's JSONL + label_key, then apply the run-mode
    caps identically to every split. Prints the cap report; raises if any split
    comes back empty."""
    train_records = load_split(cfg.jsonl_path, "train", cfg.label_key, cfg.max_duration_s)
    dev_records   = load_split(cfg.jsonl_path, "dev",   cfg.label_key, cfg.max_duration_s)
    test_records  = load_split(cfg.jsonl_path, "test",  cfg.label_key, cfg.max_duration_s)

    # One knob, applied identically to every split. None caps (full mode) are no-ops.
    _pre = (len(train_records), len(dev_records), len(test_records))
    train_records = cap_split(train_records, cfg.cap_train, cfg.cap_seed, cfg.demo_sampling)
    dev_records   = cap_split(dev_records,   cfg.cap_dev,   cfg.cap_seed, cfg.demo_sampling)
    test_records  = cap_split(test_records,  cfg.cap_test,  cfg.cap_seed, cfg.demo_sampling)

    def _capline(name: str, pre: int, post: int, cap) -> None:
        tag = f"capped→{cap}" if (cap is not None and post < pre) else "uncapped"
        print(f"   {name:<6} {post:>9d}   (of {pre:>9d}, {tag})")

    print(f"run_mode={cfg.run_mode}  sampling={cfg.demo_sampling}")
    _capline("train", _pre[0], len(train_records), cfg.cap_train)
    _capline("dev",   _pre[1], len(dev_records),   cfg.cap_dev)
    _capline("test",  _pre[2], len(test_records),  cfg.cap_test)
    if not train_records or not dev_records or not test_records:
        raise ValueError("one of the splits is empty after filtering for label_key — check the JSONL")
    return train_records, dev_records, test_records


def rough_eta_seconds(n_train: int, n_dev: int, cfg: Config, rec_per_s: float) -> float:
    """Coarse wall-clock estimate: both phases train (TRAIN, then TRAIN∪DEV) for
    num_epochs at ~rec_per_s training records/second. Eval + preprocessing extra."""
    train_recs = (n_train * cfg.num_epochs) + ((n_train + n_dev) * cfg.num_epochs)
    return train_recs / max(1e-9, rec_per_s)


def print_rough_eta(n_train: int, n_dev: int, cfg: Config) -> None:
    eta = rough_eta_seconds(n_train, n_dev, cfg, cfg.eta_rec_per_s_guess)
    print(f"\n⏱  rough ETA ~{fmt_mmss(eta)} for {cfg.num_epochs} epoch(s) × 2 phases "
          f"(guess {cfg.eta_rec_per_s_guess:.0f} train-rec/s — approximate; "
          f"recalibrates after phase 1)")


def print_recalibrated_eta(n_train: int, n_dev: int, cfg: Config) -> None:
    """Between phases: re-estimate phase 2 from phase 1's real rate.
    Reads the 'model prep' and 'end phase 1' marks."""
    p1_secs = STAGE_TIMES["end phase 1"] - STAGE_TIMES["model prep"]
    rate = (n_train * cfg.num_epochs) / max(1e-9, p1_secs)
    p2_secs = (n_train + n_dev) * cfg.num_epochs / max(1e-9, rate)
    print(f"⏱  phase 1 took {fmt_mmss(p1_secs)} → ~{rate:.0f} train-rec/s  |  "
          f"phase 2 rough ETA ~{fmt_mmss(p2_secs)} (approximate)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Labels — classification maps / regression normalizer
# ═══════════════════════════════════════════════════════════════════════════════

def build_label_maps(cfg: Config, train_records, dev_records, test_records):
    """Classification only. If the target preset left label_order=None, build it
    from the data union; then construct label2id/id2label, validate that every
    seen label is known, and print the per-split distribution.

    Returns (label2id, id2label, num_labels)."""
    if cfg.label_order is None:
        seen_vals = sorted({
            r["labels"][cfg.label_key]
            for r in (train_records + dev_records + test_records)
        }, key=str)
        cfg.label_order = seen_vals
        print(f"📌 auto-built label_order from data ({len(seen_vals)} classes): {seen_vals}")

    label2id = {lab: i for i, lab in enumerate(cfg.label_order)}
    id2label = {i: lab for i, lab in enumerate(cfg.label_order)}
    num_labels = len(cfg.label_order)

    # Validate all seen labels are in label_order
    seen = {r["labels"][cfg.label_key] for r in (train_records + dev_records + test_records)}
    unknown = seen - set(label2id)
    if unknown:
        raise ValueError(
            f"Found labels not in Config.label_order: {sorted(map(str, unknown))}. "
            f"Either add them to label_order or fix the data."
        )

    # Print per-split distribution
    print(f"Labels ({num_labels}, canonical order):")
    for lab in cfg.label_order:
        n_tr = sum(1 for r in train_records if r["labels"][cfg.label_key] == lab)
        n_dv = sum(1 for r in dev_records   if r["labels"][cfg.label_key] == lab)
        n_te = sum(1 for r in test_records  if r["labels"][cfg.label_key] == lab)
        print(f"   {str(lab):<28} train={n_tr:>4}  dev={n_dv:>4}  test={n_te:>4}")
    return label2id, id2label, num_labels


def raw_label(r: dict, cfg: Config) -> float:
    """Regression: raw target in real units, applying label_scale if present."""
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


def fit_normalizer(cfg: Config, train_records: list[dict],
                   *other_record_lists) -> LabelNormalizer:
    """Regression only. Validate that every label (train + any extra lists,
    typically dev/test) is numeric or label_scale-mapped — exactly the legacy
    up-front check — then fit the normalizer on TRAIN ONLY (no leakage)."""
    for records in (train_records, *other_record_lists):
        for r in records:
            v = r["labels"][cfg.label_key]
            if isinstance(v, (int, float)):
                continue
            if cfg.label_scale is not None and v in cfg.label_scale:
                continue
            raise ValueError(
                f"{r['instance_id']}: regression label is {v!r} (type {type(v).__name__}), "
                f"not numeric and no label_scale mapping. Provide label_scale or fix the data."
            )
    normalizer = LabelNormalizer.fit((raw_label(r, cfg) for r in train_records), cfg.normalize)
    print(f"Regression target: '{cfg.label_key}'  | normalize: {normalizer.kind}"
          + (f"  (train mean={normalizer.mean:.3f}, std={normalizer.std:.3f})"
             if normalizer.kind != "none" else ""))
    return normalizer


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset construction + preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def label_to_value(r: dict, cfg: Config, label2id: dict | None = None,
                   normalizer: LabelNormalizer | None = None):
    """Classification: original label → index in label_order.
    Regression: real-unit label → normalized training target."""
    if cfg.task_type == "classification":
        return label2id[r["labels"][cfg.label_key]]
    return float(normalizer.encode(raw_label(r, cfg)))


def prepare_dataset_dict(records: list[dict], cfg: Config, label2id: dict | None = None,
                         normalizer: LabelNormalizer | None = None) -> list[dict]:
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
            "label":        label_to_value(r, cfg, label2id, normalizer),
            "label_class":  r["labels"][cfg.label_key],   # original value for reporting
        })
    return items


def preprocess_function(examples, feature_extractor):
    """Load each WAV via soundfile. Chapter-1 splitter guarantees 16 kHz mono PCM-16;
    we sanity-check and resample as a defensive fallback.

    Works for both wav2vec2-style models (input_values, raw waveform) and
    SeamlessM4T-style models (input_features, log-mel spectrogram). The right
    key is read from feature_extractor.model_input_names[0].

    SeamlessM4T is processed one clip at a time: batched processing with
    padding=False causes numpy to produce a 1-D object array (sequences have
    different frame counts), which then fails shape unpacking inside the
    extractor. wav2vec2 is still processed in one batched call."""
    audio_key = feature_extractor.model_input_names[0]
    audio_arrays = []
    for path in examples["audio_path"]:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        audio_arrays.append(data)

    if audio_key == "input_values":
        inputs = feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors=None, padding=False,
        )
        processed = inputs[audio_key]
    else:
        processed = []
        for audio in audio_arrays:
            out = feature_extractor(
                [audio], sampling_rate=16000, return_tensors=None, padding=False,
            )
            processed.append(out[audio_key][0])

    return {audio_key: processed, "labels": examples["label"]}


def load_feature_extractor(cfg: Config):
    """AutoFeatureExtractor with the attention mask FORCED on. wav2vec2-base
    ships return_attention_mask=False; without the mask the model mean-pools
    over padding → near-constant predictions (the regression collapse)."""
    print(f"loading feature extractor: {cfg.model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name)
    feature_extractor.return_attention_mask = True
    print(f"   sampling_rate         = {feature_extractor.sampling_rate}")
    print(f"   return_attention_mask = {feature_extractor.return_attention_mask}")
    return feature_extractor


class DataCollatorForInstance:
    """Pads audio within each batch and threads the attention mask through.
    Label dtype follows the task: long for classification, float32 for
    regression. Works for both wav2vec2 (input_values) and SeamlessM4T
    (input_features) — the key is read from model_input_names."""
    def __init__(self, feature_extractor, task_type: str):
        self.feature_extractor = feature_extractor
        self.task_type = task_type
        self.audio_key = feature_extractor.model_input_names[0]

    def __call__(self, features):
        audio_seqs = [f[self.audio_key] for f in features]
        labels = [f["labels"] for f in features]
        batch = self.feature_extractor.pad(
            {self.audio_key: audio_seqs},
            padding=True, return_attention_mask=True, return_tensors="pt",
        )
        dtype = torch.long if self.task_type == "classification" else torch.float32
        batch["labels"] = torch.tensor(labels, dtype=dtype)
        return batch


# ═══════════════════════════════════════════════════════════════════════════════
# Models — classification head (HF stock) / regression head (custom)
# ═══════════════════════════════════════════════════════════════════════════════

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
    """Dispatch on task_type. Classification: HF stock audio-classification
    head (transformers 5.x requires str label2id keys at the from_pretrained
    boundary — stringified here; the in-process maps keep native types).
    Regression: the custom masked-mean-pool head above."""
    if cfg.task_type == "classification":
        hf_label2id = {str(k): int(v) for k, v in label2id.items()}
        hf_id2label = {int(k): str(v) for k, v in id2label.items()}
        model = AutoModelForAudioClassification.from_pretrained(
            cfg.model_name,
            num_labels=num_labels,
            label2id=hf_label2id, id2label=hf_id2label,
            ignore_mismatched_sizes=True,
        )
    else:
        config_obj = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
        model = Wav2Vec2ForRegression(config_obj, loss_type=cfg.loss_function)
        model.wav2vec2 = Wav2Vec2Model.from_pretrained(
            cfg.model_name, config=config_obj, ignore_mismatched_sizes=True,
        )
    if cfg.freeze_feature_encoder:
        model.wav2vec2.freeze_feature_encoder()
        print("🔒 feature encoder (CNN) frozen")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_classification_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    if len(set(labels)) > 1 and len(set(preds)) > 1:
        rho, _ = spearmanr(labels, preds)
        rho = float(rho) if not np.isnan(rho) else float("nan")
    else:
        rho = float("nan")
    return {"macro_f1": macro_f1, "accuracy": acc, "spearman": rho}


def make_regression_metrics(normalizer: LabelNormalizer):
    """Metrics in REAL units: de-normalize preds + labels before MSE/MAE.
    (The legacy notebook captured `normalizer` as a global; here it's an
    explicit closure.)"""
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
    return compute_regression_metrics


def get_compute_metrics(cfg: Config, normalizer: LabelNormalizer | None = None):
    if cfg.task_type == "classification":
        return compute_classification_metrics
    return make_regression_metrics(normalizer)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-epoch artifacts
# ═══════════════════════════════════════════════════════════════════════════════

def save_predictions_json(predictions, labels, items, out_path: Path, task_type: str,
                          id2label: dict | None):
    """Classification: predictions are logits → argmax indices, labelled via
    id2label. Regression: predictions/labels arrive already de-normalized
    (real units)."""
    out = []
    for i, item in enumerate(items):
        if task_type == "classification":
            gold = int(labels[i])
            pred = int(np.argmax(predictions[i]))
            row = {
                "gold_label":  (id2label[gold] if id2label else gold),
                "pred_label":  (id2label[pred] if id2label else pred),
                "gold_raw":    float(gold),   # class index
                "pred_raw":    float(pred),   # class index (predicted)
            }
        else:
            gold = float(labels[i])
            pred = float(predictions[i])
            row = {
                "gold_label":  gold,
                "pred_label":  pred,
                "gold_raw":    gold,
                "pred_raw":    pred,
            }
        out.append({
            "instance_id": item["instance_id"],
            "file_id":     item.get("file_id", ""),
            "start_t":     item.get("start_t"),
            "end_t":       item.get("end_t"),
            **row,
        })
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))


def plot_confusion_matrix(y_true_idx, y_pred_idx, label_order: list, out_dir: Path):
    # Stringify labels for plot/report (label_order may be ints like [0,1])
    str_labels = [str(x) for x in label_order]
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(label_order))))
    # Absolute
    fig, ax = plt.subplots(figsize=(max(6, len(label_order)*1.2), max(5, len(label_order))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=str_labels, yticklabels=str_labels, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("confusion matrix (counts)")
    plt.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png"); plt.close(fig)
    # Relative
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_rel = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float),
                       where=row_sums != 0) * 100.0
    fig, ax = plt.subplots(figsize=(max(6, len(label_order)*1.2), max(5, len(label_order))))
    sns.heatmap(cm_rel, annot=True, fmt=".1f", cmap="viridis",
                xticklabels=str_labels, yticklabels=str_labels, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("confusion matrix (%)")
    plt.tight_layout(); fig.savefig(out_dir / "confusion_matrix_relative.png"); plt.close(fig)
    # Report
    report = classification_report(y_true_idx, y_pred_idx,
                                   labels=list(range(len(label_order))),
                                   target_names=str_labels, digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report)


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


# ═══════════════════════════════════════════════════════════════════════════════
# Training engine — epoch callback, best-epoch pick, run_phase
# ═══════════════════════════════════════════════════════════════════════════════

class EpochCheckpointCallback(TrainerCallback):
    """After every epoch: predict on the eval split, log metrics, and write the
    per-epoch artifacts (predictions.json + task-specific plots) into
    ``phase_dir/epoch_logs/epoch_N/``."""

    def __init__(self, phase_dir: Path, eval_dataset, eval_items,
                 compute_metrics, data_collator, cfg: Config,
                 label_order=None, id2label=None,
                 normalizer: "LabelNormalizer | None" = None):
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
        self.normalizer = normalizer
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

        if self.cfg.task_type == "classification":
            save_predictions_json(
                out.predictions, out.label_ids, self.eval_items,
                epoch_dir / "predictions.json",
                task_type=self.cfg.task_type, id2label=self.id2label,
            )
            pred_idx = np.argmax(out.predictions, axis=-1)
            plot_confusion_matrix(out.label_ids, pred_idx, self.label_order, epoch_dir)
            line = (f"   epoch={epoch}  f1={metrics.get('eval_macro_f1', 0):.4f}  "
                    f"acc={metrics.get('eval_accuracy', 0):.4f}  "
                    f"rho={metrics.get('eval_spearman', float('nan')):.4f}")
        else:
            # De-normalize to real units for saved predictions + plots.
            gold = np.asarray(self.normalizer.decode(np.asarray(out.label_ids).reshape(-1)), dtype=float)
            pred = np.asarray(self.normalizer.decode(np.asarray(out.predictions).reshape(-1)), dtype=float)
            save_predictions_json(
                pred, gold, self.eval_items, epoch_dir / "predictions.json",
                task_type=self.cfg.task_type, id2label=self.id2label,
            )
            plot_scatter(gold, pred, epoch_dir / "scatter_plot.png")
            plot_distribution(gold, pred, epoch_dir / "distribution_plot.png")
            line = (f"   epoch={epoch}  mse={metrics.get('eval_mse', 0):.4f}  "
                    f"mae={metrics.get('eval_mae', 0):.4f}  "
                    f"rho={metrics.get('eval_spearman', float('nan')):.4f}")

        (epoch_dir / "epoch_summary.json").write_text(json.dumps(epoch_info, indent=2))
        print(line)


def best_epoch_of(epoch_results: list[dict], cfg: Config) -> dict:
    """Pick the best epoch by the task's metric. Lower-is-better for mse/mae;
    higher-is-better otherwise (NaN-safe)."""
    if cfg.task_type == "classification":
        m = cfg.best_metric_classification
    else:
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
              label2id, id2label, device: str,
              normalizer: "LabelNormalizer | None" = None) -> tuple[list[dict], dict]:
    """One full train→eval phase: build datasets, preprocess, train with the
    per-epoch callback, pick the best epoch, optionally save the best model,
    and flush the GPU. Returns (epoch_results, best_epoch)."""
    udp.banner(f"PHASE: {phase_name}  (train→{eval_split_name})")
    phase_dir = run_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Build items + datasets
    train_items = prepare_dataset_dict(train_records, cfg, label2id, normalizer)
    eval_items  = prepare_dataset_dict(eval_records,  cfg, label2id, normalizer)
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
    audio_key = feature_extractor.model_input_names[0]
    train_ds.set_format(type="torch", columns=[audio_key, "labels"])
    eval_ds.set_format(type="torch", columns=[audio_key, "labels"])

    print(f"building model: {cfg.model_name}")
    model = build_model(cfg, num_labels=len(cfg.label_order) if cfg.task_type == "classification" else 1,
                        label2id=label2id, id2label=id2label)

    data_collator = DataCollatorForInstance(feature_extractor, cfg.task_type)
    compute_metrics = get_compute_metrics(cfg, normalizer)

    steps_per_epoch = max(1, len(train_ds) // (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    _is_ampere = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8

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
        use_cpu=(device == "cpu"),
        bf16=(device == "cuda"),
        tf32=_is_ampere,
        dataloader_num_workers=cfg.dataloader_num_workers,
    )

    callback = EpochCheckpointCallback(
        phase_dir=phase_dir, eval_dataset=eval_ds, eval_items=eval_items,
        compute_metrics=compute_metrics, data_collator=data_collator, cfg=cfg,
        label_order=cfg.label_order if cfg.task_type == "classification" else None,
        id2label=id2label, normalizer=normalizer,
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


# ═══════════════════════════════════════════════════════════════════════════════
# Run setup + reporting
# ═══════════════════════════════════════════════════════════════════════════════

def make_run_dirs(cfg: Config, train_records: list[dict],
                  normalizer: "LabelNormalizer | None" = None) -> tuple[Path, Path, str]:
    """Create the timestamped run + model directories, snapshot the effective
    config (and the normalizer stats for regression). Returns
    (run_dir, model_dir, run_name)."""
    dataset_name = train_records[0]["dataset"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{dataset_name}_{cfg.label_key}_{cfg.task_type}_{ts}"

    run_dir   = udp.from_project_relative(cfg.runs_dir)   / run_name
    model_dir = udp.from_project_relative(cfg.models_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir   = {run_dir.relative_to(PROJECT_ROOT)}    (per-epoch logs)")
    print(f"model_dir = {model_dir.relative_to(PROJECT_ROOT)}  (best_model goes here)")

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    if normalizer is not None:
        (run_dir / "label_normalization.json").write_text(
            json.dumps({"kind": normalizer.kind, "mean": normalizer.mean, "std": normalizer.std}, indent=2)
        )
    return run_dir, model_dir, run_name


def print_run_summary(cfg: Config, run_name: str, run_dir: Path, model_dir: Path,
                      phase1_best: dict, phase2_best: dict,
                      normalizer: "LabelNormalizer | None" = None) -> None:
    udp.banner(f"RUN SUMMARY: {run_name}")
    print(f"task        : {cfg.task_type}")
    print(f"label_key   : {cfg.label_key}")
    print(f"model       : {cfg.model_name}")
    print(f"epochs      : {cfg.num_epochs}")
    if normalizer is not None:
        print(f"normalize   : {normalizer.kind}")
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


def best_test_predictions_path(run_dir: Path, phase2_best: dict) -> Path:
    """Path to predictions.json of the best phase-2 (TEST) epoch; raises if missing."""
    best_epoch_dir = run_dir / "phase2_test" / "epoch_logs" / f"epoch_{phase2_best['epoch']}"
    preds_path = best_epoch_dir / "predictions.json"
    if not preds_path.exists():
        raise FileNotFoundError(f"Expected predictions at {preds_path}, but it's missing.")
    return preds_path


def plot_test_confusion(run_dir: Path, phase2_best: dict, label2id: dict,
                        label_order: list, show: bool = True):
    """Classification: stacked TEST confusion matrices (counts on top,
    row-norm % below) for the best phase-2 epoch; saved into run_dir."""
    preds_path = best_test_predictions_path(run_dir, phase2_best)
    preds_data = json.loads(preds_path.read_text())
    y_true_idx = [int(p["gold_raw"]) for p in preds_data]
    y_pred_idx = [label2id[p["pred_label"]] for p in preds_data]

    str_labels = [str(x) for x in label_order]
    n = len(label_order)
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(n)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_rel = np.divide(cm.astype(float), row_sums,
                       out=np.zeros_like(cm, dtype=float),
                       where=row_sums != 0) * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(max(6, n * 1.1), max(9, n * 1.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=str_labels, yticklabels=str_labels, ax=axes[0])
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true")
    axes[0].set_title(f"counts — epoch {phase2_best['epoch']} on TEST")

    sns.heatmap(cm_rel, annot=True, fmt=".1f", cmap="viridis",
                xticklabels=str_labels, yticklabels=str_labels, ax=axes[1])
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("true")
    axes[1].set_title(f"row-normalized %  — epoch {phase2_best['epoch']} on TEST")

    plt.tight_layout()
    cm_png = run_dir / "confusion_matrix_test.png"
    fig.savefig(cm_png, dpi=120, bbox_inches="tight")
    print(f"saved {cm_png.relative_to(PROJECT_ROOT)}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_test_scatter(run_dir: Path, phase2_best: dict, show: bool = True):
    """Regression: side-by-side gold-vs-pred scatter + distribution histogram
    for the best phase-2 (TEST) epoch; saved into run_dir."""
    preds_path = best_test_predictions_path(run_dir, phase2_best)
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
    if show:
        plt.show()
    else:
        plt.close(fig)


def spot_check(run_dir: Path, phase2_best: dict, task_type: str, k: int = 5,
               seed: int = 0) -> None:
    """Print k random TEST examples from the best epoch's predictions.
    Classification: gold/pred labels + hit marker. Regression: real-unit
    gold/pred + absolute error."""
    preds_path = best_test_predictions_path(run_dir, phase2_best)
    rows = json.loads(preds_path.read_text())
    sample = random.Random(seed).sample(rows, k=min(k, len(rows)))

    udp.banner(f"INFERENCE SPOT-CHECK — {min(k, len(rows))} random TEST examples")
    for p in sample:
        print(f"  {p['instance_id']}")
        print(f"     file={p.get('file_id', '')}  span={p.get('start_t')}–{p.get('end_t')}")
        if task_type == "classification":
            gold, pred = p.get("gold_label"), p.get("pred_label")
            hit = "✓" if gold == pred else "✗"
            print(f"     gold={gold}   pred={pred}   {hit}")
        else:
            g, pr = p.get("gold_raw"), p.get("pred_raw")
            err = abs(g - pr) if (g is not None and pr is not None) else float("nan")
            g_s  = f"{g:.3f}"  if g  is not None else "None"
            pr_s = f"{pr:.3f}" if pr is not None else "None"
            print(f"     gold={g_s}   pred={pr_s}   |err|={err:.3f}")
        print()
