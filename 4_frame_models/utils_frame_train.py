"""utils_frame_train — the chapter-4 frame-training engine.

Shared by:
  * 41_train_frame_classification.ipynb / run_41_classification.py
  * (future) 42_train_frame_regression.ipynb / run_42_regression.py

Mirrors chapter 3's `utils_instance_train.py` shape as closely as possible.
The differences from chapter 3 are frame-level, not instance-level:
  * per-frame classifier head with IGNORE_INDEX-aware cross-entropy
  * per-record label alignment to the model's actual output frame count
  * two model families supported at once — Wav2Vec2 (raw waveform, CNN
    front-end) and Wav2Vec2Bert (log-mel filterbanks, adapter downsampling)
  * word_frame records slice a sub-clip from the utterance WAV via
    start_t/end_t during preprocess (no per-word WAVs on disk)
  * an `exclude_multistress` filter that drops rows tagged
    metadata.multistress=True (SI primary-stress target's north star is
    one stressed syllable per word)

IMPORT-ORDER CONTRACT (read this before importing):
  1. GPU guard must run BEFORE this module is imported (this module imports
     torch at the top; CUDA_VISIBLE_DEVICES only takes effect if set before
     torch's first CUDA touch).
  2. HF_HOME is handled HERE (project-local ``stock_models/``), before the
     transformers import below.
  3. Headless runners should set MPLBACKEND=Agg before importing.

Config flow: ``load_config(path, task_type, run_mode=...)`` reads
`config.json` (shared block + task block + mode overrides), resolves the
TARGETS preset, and returns a validated Config. Consumers never define
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
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
from pathlib import Path

# ── PROJECT_ROOT + HF_HOME (must precede the transformers import) ─────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_data_prep"))
import utils_dataprep as udp

PROJECT_ROOT = udp.PROJECT_ROOT
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "stock_models"))

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

# Wav2Vec2Bert may not exist in older transformers versions — guard the import
# so the file still loads for wav2vec2-only environments. build_model raises a
# clear error if someone picks a w2v-bert model without the classes available.
try:
    from transformers import Wav2Vec2BertModel, Wav2Vec2BertPreTrainedModel  # type: ignore
    _HAS_W2V_BERT = True
except ImportError:
    Wav2Vec2BertModel = None            # type: ignore
    Wav2Vec2BertPreTrainedModel = None  # type: ignore
    _HAS_W2V_BERT = False


# Token-CE pad sentinel. Labels at this value are ignored by loss and masked
# out of every metric.
IGNORE_INDEX = -100


# ═══════════════════════════════════════════════════════════════════════════════
# Stage timing
# ═══════════════════════════════════════════════════════════════════════════════

STAGE_TIMES: dict[str, float] = {}


def mark(stage: str) -> None:
    STAGE_TIMES[stage] = time.time()


def fmt_mmss(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    return f"{s // 60:02d}:{s % 60:02d}"


def print_stage_breakdown(times: dict[str, float] | None = None) -> None:
    items = list((STAGE_TIMES if times is None else times).items())
    if not items:
        print("no timing recorded"); return
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
# TARGETS registry — frame-level presets
# ═══════════════════════════════════════════════════════════════════════════════
# Two preset shapes:
#   * single-file target : preset carries ``jsonl_path``
#   * multi-lang target  : preset carries ``jsonl_template`` + ``langs``
# resolve_target() branches on which field is present.

TARGETS: dict = {}


def _add_parlaspeech_frame_targets(targets: dict) -> None:
    """Chapter-4 north-star targets from ParlaSpeech (built by 11c).
    - fp_frames: binary filled_pause over the utterance (HR/RS/PL/CZ).
    - primary_stress_frames: word-level (HR/RS only)."""
    targets["parlaspeech_fp_frames"] = {
        "jsonl_template": "data/processed_jsonl/parlaspeech_{lang}_utterance_frame.jsonl",
        "langs":       ("hr", "rs", "pl", "cz"),
        "label_key":   "filled_pause",
        "task_type":   "classification",
        "level":       "frame",
        "label_order": [0, 1],
    }
    targets["parlaspeech_primary_stress_frames"] = {
        "jsonl_template": "data/processed_jsonl/parlaspeech_{lang}_word_frame.jsonl",
        "langs":       ("hr", "rs"),
        "label_key":   "primary_stress",
        "task_type":   "classification",
        "level":       "frame",
        "label_order": [0, 1],
    }


def _add_nejc_slo_stress_targets(targets: dict) -> None:
    """Slovenian primary-stress `word_frame` from Nejc's annotated TGs (built
    by `5_tg_minter/54_stress_tg_to_jsonl.py`). Independent of the HR/RS
    pool — Slovenian stress is a different phenomenon."""
    targets["si_primary_stress_frames"] = {
        "jsonl_path":  "data/processed_jsonl/si_primary_stress_word_frame.jsonl",
        "label_key":   "primary_stress",
        "task_type":   "classification",
        "level":       "frame",
        "label_order": [0, 1],
    }


_add_parlaspeech_frame_targets(TARGETS)
_add_nejc_slo_stress_targets(TARGETS)


def available_targets(task_type: str | None = None) -> list[str]:
    if task_type is None:
        return sorted(TARGETS)
    return sorted(k for k, v in TARGETS.items() if v["task_type"] == task_type)


def print_target_menu(task_type: str | None = None) -> None:
    for name in available_targets(task_type):
        t = TARGETS[name]
        loc = t.get("jsonl_path") or f"{t.get('jsonl_template')}  (langs={list(t.get('langs', ()))})"
        print(f"  {name:<40} → {loc}")


def resolve_target(cfg, targets: dict = TARGETS) -> None:
    """Overwrite label_key / task_type / label_order and populate
    ``cfg.jsonl_paths`` (a list, always) from the picked preset.

    Two shapes:
    - Single-file: preset carries ``jsonl_path``. ``cfg.langs`` is ignored.
    - Multi-lang:  preset carries ``jsonl_template`` + ``langs``. ``cfg.langs=()``
      → every supported lang that has a JSONL on disk (missing ones skipped);
      ``cfg.langs=("hr",)`` → exactly those (must exist)."""
    if cfg.target not in targets:
        raise ValueError(
            f"Config.target={cfg.target!r} not in TARGETS. Known: {sorted(targets)}"
        )
    t = targets[cfg.target]
    cfg.label_key   = t["label_key"]
    cfg.task_type   = t["task_type"]
    cfg.label_order = t["label_order"]

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
        elif cfg.langs:
            raise FileNotFoundError(f"requested lang {l!r} but {p} is missing — run 11c")
        else:
            print(f"  ⏭  {l}: {p} not found — skipping")
    if not paths:
        raise FileNotFoundError(
            f"no JSONL on disk for {cfg.target!r} (langs={list(chosen)}). Run 11c first."
        )
    cfg.jsonl_paths = paths


# ═══════════════════════════════════════════════════════════════════════════════
# Config — one dataclass, config.json fills it
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # -- Target preset (resolve_target overwrites the data fields below) -----
    target: str = ""
    langs:  tuple = ()

    # -- Data (overwritten by resolve_target) --------------------------------
    jsonl_paths: list = field(default_factory=list)
    label_key:   str = ""
    task_type:   str = "classification"
    label_order: list = field(default_factory=lambda: [0, 1])

    # -- Frame filters -------------------------------------------------------
    # Skip rows tagged metadata.multistress=True (SI primary-stress).
    # Harmless on targets that don't carry the flag.
    exclude_multistress: bool = True

    # -- Run mode ------------------------------------------------------------
    run_mode: str = "full"
    demo_sampling: str = "proportional"
    cap_seed: int = 1234

    cap_train: int | None = None
    cap_dev:   int | None = None
    cap_test:  int | None = None

    # -- Frame rate ----------------------------------------------------------
    # v1 hard-lock: label sequences must be 50 Hz. Model-frame alignment
    # (~50 Hz for wav2vec2, ~25 Hz for wav2vec-BERT with adapter) is per-record
    # in preprocess.
    required_frame_rate_hz: int = 50

    # -- Model ---------------------------------------------------------------
    model_name: str              = "facebook/wav2vec2-base"
    freeze_feature_encoder: bool = True
    head_dropout: float          = 0.1

    # -- Training ------------------------------------------------------------
    batch_size: int      = 16
    grad_accum: int      = 1
    learning_rate: float = 1e-5
    num_epochs: int      = 3
    max_grad_norm: float = 1.0
    warmup_ratio: float  = 0.10
    logging_steps: int | str = "auto"

    eta_rec_per_s_guess: float = 25.0

    # -- Output --------------------------------------------------------------
    runs_dir: str   = "runs"
    models_dir: str = "models"

    # -- Best-epoch selection ------------------------------------------------
    # "frame_macro_f1" | "frame_accuracy" | "frame_f1_positive"
    best_metric_classification: str = "frame_macro_f1"

    # -- Preprocessing -------------------------------------------------------
    preprocess_batch_size: int  = 32
    dataloader_num_workers: int = 8
    map_num_proc: int = 8

    # Optional length cap — DROPS records (never truncates; truncation would
    # desync the frame labels). Off by default.
    enable_max_audio_seconds: bool = False
    max_audio_seconds: float       = 15.0

    # -- Visualization -------------------------------------------------------
    n_examples_to_plot: int = 6

    # -- Hardware ------------------------------------------------------------
    reserved_gpu: str = "2"
    use_cuda: bool   = True


def apply_mode(cfg: Config, overrides: dict) -> None:
    """Layer overrides onto Config. Every key must name a real field — a typo
    is a hard error, not a silent no-op. `langs` is coerced tuple(list)."""
    valid = {f.name for f in fields(cfg)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"overrides name unknown Config fields: {sorted(unknown)}")
    for k, v in overrides.items():
        if k == "langs" and isinstance(v, list):
            v = tuple(v)
        setattr(cfg, k, v)


def load_config(path: str | Path, task_type: str,
                run_mode: str | None = None) -> tuple[Config, dict]:
    """Layered loader (later wins): Config defaults → ``shared`` → task block
    (``classification``) → ``modes[run_mode]``. Returns (cfg, raw)."""
    raw = json.loads(Path(path).read_text())
    if task_type != "classification":
        raise ValueError(
            f"chapter-4 v1 supports only 'classification', got {task_type!r} "
            "(frame regression → future 42_train_frame_regression)"
        )
    cfg = Config(task_type=task_type)
    apply_mode(cfg, raw.get("shared", {}))
    apply_mode(cfg, raw.get(task_type, {}))
    cfg.run_mode = run_mode if run_mode is not None else raw.get("run_mode", cfg.run_mode)
    modes = raw.get("modes", {})
    if cfg.run_mode not in modes:
        raise ValueError(f"run_mode={cfg.run_mode!r} not in modes: {sorted(modes)}")

    resolve_target(cfg, TARGETS)
    if cfg.task_type != task_type:
        raise ValueError(
            f"target {cfg.target!r} is {cfg.task_type}, but this pipeline is {task_type}"
        )
    apply_mode(cfg, modes[cfg.run_mode])
    return cfg, raw


def validate_config(cfg: Config) -> None:
    if cfg.run_mode not in ("test", "demo", "full"):
        raise ValueError(f"run_mode invalid: {cfg.run_mode!r}")
    if cfg.demo_sampling not in ("proportional", "balanced"):
        raise ValueError(f"demo_sampling invalid: {cfg.demo_sampling!r}")
    if cfg.task_type != "classification":
        raise ValueError(f"v1 supports classification only, got {cfg.task_type!r}")
    if not cfg.label_order or len(cfg.label_order) < 2:
        raise ValueError("label_order must have ≥2 entries. Binary = [0, 1].")
    if len(set(cfg.label_order)) != len(cfg.label_order):
        raise ValueError(f"label_order has duplicates: {cfg.label_order}")
    if cfg.required_frame_rate_hz != 50:
        raise ValueError(
            f"required_frame_rate_hz must be 50 in v1, got {cfg.required_frame_rate_hz}"
        )
    if cfg.best_metric_classification not in (
        "frame_macro_f1", "frame_accuracy", "frame_f1_positive"
    ):
        raise ValueError(f"best_metric_classification invalid: {cfg.best_metric_classification!r}")


def resolve_device(cfg: Config, use_cuda: bool) -> str:
    cfg.use_cuda = use_cuda
    if cfg.use_cuda and torch.cuda.is_available():
        return "cuda"
    if cfg.use_cuda and not torch.cuda.is_available():
        print("⚠️  GPU selected but torch.cuda.is_available()==False; falling back to CPU")
    return "cpu"


def print_project_info(verbose: bool = False) -> None:
    if verbose:
        print(f"PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"HF_HOME      = {os.environ['HF_HOME']}")
    else:
        print(f"repo         = {PROJECT_ROOT.name}/")
        print(f"HF cache     = <repo>/stock_models/  (via HF_HOME, project-local)")


def print_config_summary(cfg: Config, device: str) -> None:
    print(f"target      = {cfg.target}")
    print(f"run_mode    = {cfg.run_mode}  "
          f"(caps train/dev/test = {cfg.cap_train}/{cfg.cap_dev}/{cfg.cap_test}, "
          f"sampling={cfg.demo_sampling})")
    print(f"langs       = {cfg.langs or '(target default)'} → {len(cfg.jsonl_paths)} JSONL(s)")
    for p in cfg.jsonl_paths:
        print(f"   • {p}")
    print(f"label_key   = {cfg.label_key}")
    print(f"label_order = {cfg.label_order}")
    print(f"model       = {cfg.model_name}")
    print(f"epochs      = {cfg.num_epochs}  batch_size={cfg.batch_size}  lr={cfg.learning_rate}")
    print(f"exclude_multistress = {cfg.exclude_multistress}")
    print(f"device      = {device}")
    if device == "cuda":
        print(f"✓ visible devices : {torch.cuda.device_count()}  "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        print(f"✓ device name     : {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Data — load splits, cap
# ═══════════════════════════════════════════════════════════════════════════════

def load_split(jsonl_paths: list, split: str, label_key: str,
               *, exclude_multistress: bool) -> tuple[list[dict], int]:
    """Return (records, n_multistress_dropped) for `split` across every pooled
    JSONL. Records without the multistress tag are unaffected."""
    out, n_multi = [], 0
    for jp in jsonl_paths:
        for r in udp.iter_jsonl(jp):
            if r["split"] != split:
                continue
            v = r.get("labels", {}).get(label_key)
            if v is None or not isinstance(v, list) or len(v) == 0:
                continue
            if exclude_multistress and r.get("metadata", {}).get("multistress"):
                n_multi += 1
                continue
            out.append(r)
    return out, n_multi


def cap_split(records: list[dict], n, seed: int, sampling: str = "proportional") -> list[dict]:
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
                    if len(out) >= n: break
            if not progressed:
                break
        return out
    raise ValueError(f"demo_sampling must be proportional|balanced, got {sampling!r}")


def load_and_cap_splits(cfg: Config) -> tuple[list[dict], list[dict], list[dict]]:
    """Load train/dev/test, filter by label_key + multistress, then apply the
    active mode's caps identically to every split."""
    tr, n_mtr = load_split(cfg.jsonl_paths, "train", cfg.label_key,
                           exclude_multistress=cfg.exclude_multistress)
    dv, n_mdv = load_split(cfg.jsonl_paths, "dev",   cfg.label_key,
                           exclude_multistress=cfg.exclude_multistress)
    te, n_mte = load_split(cfg.jsonl_paths, "test",  cfg.label_key,
                           exclude_multistress=cfg.exclude_multistress)

    _pre = (len(tr), len(dv), len(te))
    tr = cap_split(tr, cfg.cap_train, cfg.cap_seed, cfg.demo_sampling)
    dv = cap_split(dv, cfg.cap_dev,   cfg.cap_seed, cfg.demo_sampling)
    te = cap_split(te, cfg.cap_test,  cfg.cap_seed, cfg.demo_sampling)

    def _capline(name, pre, post, cap):
        tag = f"capped→{cap}" if (cap is not None and post < pre) else "uncapped"
        print(f"   {name:<6} {post:>9d}   (of {pre:>9d}, {tag})")

    print(f"run_mode={cfg.run_mode}  sampling={cfg.demo_sampling}")
    _capline("train", _pre[0], len(tr), cfg.cap_train)
    _capline("dev",   _pre[1], len(dv), cfg.cap_dev)
    _capline("test",  _pre[2], len(te), cfg.cap_test)
    if cfg.exclude_multistress and (n_mtr + n_mdv + n_mte):
        print(f"   multistress excluded: train={n_mtr}  dev={n_mdv}  test={n_mte}")
    if len(cfg.jsonl_paths) > 1:
        print(f"   train lang mix: {dict(Counter(r['dataset'] for r in tr))}")
    if not tr or not dv or not te:
        raise ValueError("a split became empty after filtering — check the JSONL")
    return tr, dv, te


def rough_eta_seconds(n_train: int, n_dev: int, cfg: Config, rec_per_s: float) -> float:
    train_recs = (n_train * cfg.num_epochs) + ((n_train + n_dev) * cfg.num_epochs)
    return train_recs / max(1e-9, rec_per_s)


def print_rough_eta(n_train: int, n_dev: int, cfg: Config) -> None:
    eta = rough_eta_seconds(n_train, n_dev, cfg, cfg.eta_rec_per_s_guess)
    print(f"\n⏱  rough ETA ~{fmt_mmss(eta)} for {cfg.num_epochs} epoch(s) × 2 phases "
          f"(guess {cfg.eta_rec_per_s_guess:.0f} train-rec/s — recalibrates after phase 1)")


def print_recalibrated_eta(n_train: int, n_dev: int, cfg: Config) -> None:
    p1_secs = STAGE_TIMES["end phase 1"] - STAGE_TIMES["model prep"]
    rate = (n_train * cfg.num_epochs) / max(1e-9, p1_secs)
    p2_secs = (n_train + n_dev) * cfg.num_epochs / max(1e-9, rate)
    print(f"⏱  phase 1 took {fmt_mmss(p1_secs)} → ~{rate:.0f} train-rec/s  |  "
          f"phase 2 rough ETA ~{fmt_mmss(p2_secs)}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Labels & mappings, frame-rate guard
# ═══════════════════════════════════════════════════════════════════════════════

def build_label_maps(cfg: Config, train_records, dev_records, test_records):
    """Frame-flavored label map. Frame labels must all appear in
    ``cfg.label_order``. The per-split counts are FRAMES (tokens), not records."""
    label2id = {lab: i for i, lab in enumerate(cfg.label_order)}
    id2label = {i: lab for i, lab in enumerate(cfg.label_order)}
    num_labels = len(cfg.label_order)

    seen = set()
    for r in train_records + dev_records + test_records:
        seen.update(r["labels"][cfg.label_key])
    unknown = seen - set(label2id)
    if unknown:
        raise ValueError(
            f"frame labels not in label_order: {sorted(unknown)}. "
            f"Fix label_order or fix the data."
        )

    def _fc(records):
        c = Counter()
        for r in records: c.update(r["labels"][cfg.label_key])
        return c
    ftr = _fc(train_records); fdv = _fc(dev_records); fte = _fc(test_records)
    print(f"Frame labels ({num_labels}, canonical order) — counts are FRAMES:")
    print(f"   {'class':>10}  {'train':>12}  {'dev':>12}  {'test':>12}")
    for lab in cfg.label_order:
        print(f"   {str(lab):>10}  {ftr.get(lab,0):>12d}  {fdv.get(lab,0):>12d}  {fte.get(lab,0):>12d}")
    print(f"   {'TOTAL':>10}  {sum(ftr.values()):>12d}  {sum(fdv.values()):>12d}  {sum(fte.values()):>12d}")
    return label2id, id2label, num_labels


def validate_frame_rate(records: list[dict], required_hz: int) -> None:
    missing = [r["instance_id"] for r in records if r.get("frame_rate_hz") is None]
    bad = [(r["instance_id"], r["frame_rate_hz"]) for r in records
           if r.get("frame_rate_hz") is not None and r["frame_rate_hz"] != required_hz]
    if missing:
        raise ValueError(f"{len(missing)} records missing frame_rate_hz. "
                         f"Examples: {missing[:5]}")
    if bad:
        rates = sorted({b[1] for b in bad})
        raise ValueError(f"{len(bad)} records have frame_rate_hz != {required_hz} "
                         f"(seen: {rates}). Examples: {bad[:5]}")


def drop_long_records(records: list[dict], max_s: float) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for r in records:
        try:
            dur = udp.get_wav_duration(r["audio_path"])
        except Exception as e:
            print(f"⚠️  could not read duration for {r['instance_id']}: {e}; keeping")
            kept.append(r); continue
        if dur > max_s:
            dropped += 1; continue
        kept.append(r)
    return kept, dropped


# ═══════════════════════════════════════════════════════════════════════════════
# Model-family detection + output-length formulas
# ═══════════════════════════════════════════════════════════════════════════════

def detect_model_family(hf_config) -> str:
    """Return 'wav2vec2' or 'wav2vec2_bert'. Raises on other families."""
    mt = getattr(hf_config, "model_type", "unknown")
    if mt == "wav2vec2":
        return "wav2vec2"
    if mt == "wav2vec2-bert":
        return "wav2vec2_bert"
    raise ValueError(
        f"unsupported model_type={mt!r}. Supported families: wav2vec2, wav2vec2-bert "
        f"(add a compute_output_length branch + a head class to extend)"
    )


def compute_output_length(hf_config, input_length: int, family: str) -> int:
    """Model-frame count for one input.

    - wav2vec2: input is raw audio samples; CNN stride formula.
    - wav2vec2_bert: input is filterbank frames; only the (optional) adapter
      downsamples. Encoder itself preserves the frame count."""
    if family == "wav2vec2":
        out = int(input_length)
        for k, s in zip(hf_config.conv_kernel, hf_config.conv_stride):
            out = (out - int(k)) // int(s) + 1
        return max(out, 0)
    if family == "wav2vec2_bert":
        out = int(input_length)
        if getattr(hf_config, "add_adapter", False):
            num_layers = int(getattr(hf_config, "num_adapter_layers", 1))
            kernel = int(getattr(hf_config, "adapter_kernel_size", 3))
            stride = int(getattr(hf_config, "adapter_stride", 2))
            for _ in range(num_layers):
                out = (out - kernel) // stride + 1
        return max(out, 0)
    raise ValueError(f"unknown family {family!r}")


def align_labels_to_frames(label_seq, n_frames: int) -> list:
    """Nearest-neighbor resize of `label_seq` to `n_frames`. 50 Hz source →
    per-record model-frame count."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset prep + preprocessing (both families)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_dataset_dict(records: list[dict], cfg: Config, label2id: dict) -> list[dict]:
    """Canonical records → list-of-dicts for Dataset.from_list. Frame labels
    mapped through label2id (no-op for binary [0, 1])."""
    items = []
    for r in records:
        raw = r["labels"][cfg.label_key]
        items.append({
            "instance_id": r["instance_id"],
            "file_id":     r.get("file_id", ""),
            "audio_path":  str(udp.from_project_relative(r["audio_path"])),
            "start_t":     r.get("start_t"),      # word-frame: sub-clip within audio_path
            "end_t":       r.get("end_t"),
            "labels":      [label2id[v] for v in raw],
        })
    return items


def make_preprocess_function(feature_extractor, hf_config, family: str):
    """Factory: bind the model config + family into a preprocess function that
    (a) loads and slices audio, (b) runs feature extraction, and (c) aligns
    the 50 Hz label sequence to the model's per-record output frame count."""
    audio_key = feature_extractor.model_input_names[0]

    def preprocess_function(examples):
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

        if audio_key == "input_values":
            inputs = feature_extractor(
                audio_arrays, sampling_rate=16000, return_tensors=None, padding=False,
            )
            features = inputs[audio_key]
        else:
            # SeamlessM4T-style: variable-length filterbanks. Pad within this
            # preprocess batch so numpy can stack, then unpack via attention_mask
            # so the DataCollator can re-pad to the training-batch length.
            inputs = feature_extractor(
                audio_arrays, sampling_rate=16000, return_tensors="np", padding="longest",
            )
            feats = inputs[audio_key]         # (B, max_frames, feat_dim)
            mask = inputs["attention_mask"]   # (B, max_frames)
            features = [feats[i, : int(mask[i].sum())] for i in range(len(feats))]

        aligned_labels = []
        for feat, label_seq in zip(features, examples["labels"]):
            n_frames = compute_output_length(hf_config, len(feat), family)
            aligned_labels.append(align_labels_to_frames(label_seq, n_frames))
        return {audio_key: features, "labels": aligned_labels}
    return preprocess_function


def load_feature_extractor(cfg: Config):
    """AutoFeatureExtractor with attention_mask FORCED on. wav2vec2-base ships
    return_attention_mask=False; without the mask the encoder mean-pools over
    padding, which degrades every downstream head."""
    print(f"loading feature extractor: {cfg.model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name)
    feature_extractor.return_attention_mask = True
    print(f"   sampling_rate         = {feature_extractor.sampling_rate}")
    print(f"   return_attention_mask = {feature_extractor.return_attention_mask}")
    print(f"   model_input_names[0]  = {feature_extractor.model_input_names[0]}")
    return feature_extractor


class DataCollatorForFrame:
    """Pads audio within each batch, threads attention_mask through, and pads
    frame labels to the batch's longest sequence with IGNORE_INDEX (loss and
    metrics ignore those). Works for both `input_values` and `input_features`."""
    def __init__(self, feature_extractor, task_type: str = "classification"):
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
        max_len = max(len(l) for l in labels)
        padded = []
        for l in labels:
            l = l.tolist() if torch.is_tensor(l) else list(l)
            padded.append(l + [IGNORE_INDEX] * (max_len - len(l)))
        dtype = torch.long if self.task_type == "classification" else torch.float32
        batch["labels"] = torch.tensor(padded, dtype=dtype)
        return batch


# ═══════════════════════════════════════════════════════════════════════════════
# Model — per-frame heads for both families, with ignore_index=-100
# ═══════════════════════════════════════════════════════════════════════════════

def _frame_cls_forward(self, hidden, labels):
    """Shared per-frame classifier tail: dropout → Linear → token-CE with
    ignore_index. Belt-and-suspenders truncate/pad if labels and logits ever
    drift by a frame (preprocess should have aligned them exactly)."""
    hidden = self.dropout(hidden)
    logits = self.classifier(hidden)      # (B, T, num_labels)
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


class Wav2Vec2ForFrameCLS(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 body + per-frame classifier. Token-CE with ignore_index=-100."""

    def __init__(self, config, head_dropout: float = 0.1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_values, attention_mask=None, labels=None):
        out = self.wav2vec2(input_values, attention_mask=attention_mask)
        return _frame_cls_forward(self, out.last_hidden_state, labels)


if _HAS_W2V_BERT:
    class Wav2Vec2BertForFrameCLS(Wav2Vec2BertPreTrainedModel):
        """Wav2Vec2Bert body + per-frame classifier. Same head + loss as the
        wav2vec2 variant; the body accepts input_features, not input_values."""

        def __init__(self, config, head_dropout: float = 0.1):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.wav2vec2_bert = Wav2Vec2BertModel(config)
            self.dropout = nn.Dropout(head_dropout)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.post_init()

        def forward(self, input_features, attention_mask=None, labels=None):
            out = self.wav2vec2_bert(input_features, attention_mask=attention_mask)
            return _frame_cls_forward(self, out.last_hidden_state, labels)
else:
    Wav2Vec2BertForFrameCLS = None   # sentinel; build_model raises if picked


def build_model(cfg: Config, num_labels: int, label2id, id2label):
    """Dispatch on model family. transformers 5.x requires str label2id keys
    at the from_pretrained boundary — stringified here; the in-process maps
    keep native types.

    For fine-tuned checkpoints (e.g. classla/Wav2Vec2BertPrimaryStress...),
    load with ignore_mismatched_sizes=True so a body-only match still works
    even if the checkpoint's classifier shape differs."""
    hf_label2id = {str(k): int(v) for k, v in label2id.items()}
    hf_id2label = {int(k): str(v) for k, v in id2label.items()}
    hf_config = AutoConfig.from_pretrained(
        cfg.model_name, num_labels=num_labels,
        label2id=hf_label2id, id2label=hf_id2label,
    )
    family = detect_model_family(hf_config)

    if family == "wav2vec2":
        model = Wav2Vec2ForFrameCLS.from_pretrained(
            cfg.model_name, config=hf_config, head_dropout=cfg.head_dropout,
            ignore_mismatched_sizes=True,
        )
    elif family == "wav2vec2_bert":
        if Wav2Vec2BertForFrameCLS is None:
            raise ImportError(
                "wav2vec-BERT support requires transformers with Wav2Vec2Bert*. "
                "Upgrade transformers or pick a wav2vec2 model_name."
            )
        model = Wav2Vec2BertForFrameCLS.from_pretrained(
            cfg.model_name, config=hf_config, head_dropout=cfg.head_dropout,
            ignore_mismatched_sizes=True,
        )
    else:
        raise ValueError(f"unhandled family {family!r}")

    if cfg.freeze_feature_encoder:
        backbone_name = model.base_model_prefix
        backbone = getattr(model, backbone_name, None)
        if backbone is not None and hasattr(backbone, "freeze_feature_encoder"):
            backbone.freeze_feature_encoder()
            print(f"🔒 feature encoder frozen on backbone '{backbone_name}'")
        else:
            # wav2vec-BERT 2.0 has no CNN feature encoder (input is filterbanks).
            print(f"ℹ️  '{backbone_name}' has no feature_encoder to freeze — skipping")

    return model, family, hf_config


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_frame_metrics(eval_pred):
    """Frame-level metrics, non-pad frames only, flattened across the batch."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != IGNORE_INDEX
    y_true = labels[mask]; y_pred = preds[mask]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    n_classes = int(logits.shape[-1])
    per_class_f1 = f1_score(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0,
    )
    f1_pos = float(per_class_f1[n_classes - 1])   # last class = positive
    return {"frame_accuracy": acc, "frame_macro_f1": macro_f1, "frame_f1_positive": f1_pos}


# ═══════════════════════════════════════════════════════════════════════════════
# Per-epoch artifacts (predictions.json + gold/pred strip plot)
# ═══════════════════════════════════════════════════════════════════════════════

def save_predictions_json(predictions, labels, items, out_path: Path, id2label: dict):
    pred_idx = np.argmax(predictions, axis=-1) if predictions.ndim == 3 else predictions
    prob_pos_all = None
    if predictions.ndim == 3:
        z = predictions - predictions.max(axis=-1, keepdims=True)
        ez = np.exp(z)
        prob_pos_all = (ez / ez.sum(axis=-1, keepdims=True))[..., -1]
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


_PALETTE = ["#dddddd", "#d62728", "#1f77b4", "#2ca02c", "#9467bd",
            "#ff7f0e", "#8c564b", "#e377c2"]


def _row_to_image(seq, n_classes: int):
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
    n_classes = len(id2label); pal = None
    for i in range(n):
        valid = labels[i] != IGNORE_INDEX
        gold_img, pal = _row_to_image(labels[i][valid], n_classes)
        pred_img, _   = _row_to_image(pred_idx[i][valid], n_classes)
        strip = np.concatenate([gold_img, pred_img], axis=0)
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


# ═══════════════════════════════════════════════════════════════════════════════
# EpochCheckpointCallback + best-epoch selection
# ═══════════════════════════════════════════════════════════════════════════════

class EpochCheckpointCallback(TrainerCallback):
    """After every epoch: predict on eval, log metrics + per-epoch artifacts.
    Does NOT save weights — only the best epoch's weights are saved, at the end
    of phase 2."""

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
        print(f"   epoch={epoch}  macroF1={metrics.get('eval_frame_macro_f1', 0):.4f}  "
              f"acc={metrics.get('eval_frame_accuracy', 0):.4f}  "
              f"F1+={metrics.get('eval_frame_f1_positive', 0):.4f}")


def best_epoch_of(epoch_results: list[dict], cfg: Config) -> dict:
    """All frame metrics are higher-is-better; NaN-safe max."""
    m = cfg.best_metric_classification
    return max(
        epoch_results,
        key=lambda r: (r.get(f"eval_{m}", float("-inf"))
                       if r.get(f"eval_{m}") is not None
                       and not (isinstance(r.get(f"eval_{m}"), float) and np.isnan(r.get(f"eval_{m}")))
                       else float("-inf"))
    )


def release_gpu(verbose: bool = True) -> None:
    reserved_before_mb = None
    if torch.cuda.is_available():
        reserved_before_mb = torch.cuda.memory_reserved() // (1024 * 1024)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try: torch.cuda.ipc_collect()
        except Exception: pass
        if verbose:
            reserved_after_mb = torch.cuda.memory_reserved() // (1024 * 1024)
            print(f"🧹 released model + emptied CUDA cache "
                  f"(reserved: {reserved_before_mb} MB → {reserved_after_mb} MB)")
    elif verbose:
        print("🧹 released Python refs (no CUDA device)")


# ═══════════════════════════════════════════════════════════════════════════════
# run_phase — train + evaluate + save best model (phase 2 only)
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase(*, phase_name: str, train_records: list[dict], eval_records: list[dict],
              eval_split_name: str, save_best_model: bool,
              cfg: Config, run_dir: Path, model_dir: Path, feature_extractor,
              label2id, id2label, device: str) -> tuple[list[dict], dict]:
    udp.banner(f"PHASE: {phase_name}  (train→{eval_split_name})")
    phase_dir = run_dir / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)

    print(f"building model config: {cfg.model_name}")
    # We need hf_config + family BEFORE preprocess to compute per-record output
    # length. build_model is called for the actual weights next.
    hf_config_probe = AutoConfig.from_pretrained(cfg.model_name)
    family = detect_model_family(hf_config_probe)
    print(f"   model_type = {hf_config_probe.model_type}  → family = {family}")

    train_items = prepare_dataset_dict(train_records, cfg, label2id)
    eval_items  = prepare_dataset_dict(eval_records,  cfg, label2id)
    train_ds = Dataset.from_list(train_items)
    eval_ds  = Dataset.from_list(eval_items)

    cpu_max = os.cpu_count() or 1
    map_num_proc = min(cfg.map_num_proc, cpu_max)
    dataloader_num_workers = min(cfg.dataloader_num_workers, cpu_max)
    if map_num_proc < cfg.map_num_proc or dataloader_num_workers < cfg.dataloader_num_workers:
        print(f"⚙️  capped CPU workers to os.cpu_count()={cpu_max}: "
              f"map_num_proc {cfg.map_num_proc}→{map_num_proc}, "
              f"dataloader_num_workers {cfg.dataloader_num_workers}→{dataloader_num_workers}")

    preprocess_fn = make_preprocess_function(feature_extractor, hf_config_probe, family)
    print(f"preprocessing {len(train_ds)} train + {len(eval_ds)} eval "
          f"(batch_size={cfg.preprocess_batch_size})…")
    train_ds = train_ds.map(
        preprocess_fn, batched=True, batch_size=cfg.preprocess_batch_size,
        remove_columns=train_ds.column_names, num_proc=map_num_proc,
    )
    eval_ds = eval_ds.map(
        preprocess_fn, batched=True, batch_size=cfg.preprocess_batch_size,
        remove_columns=eval_ds.column_names, num_proc=map_num_proc,
    )
    audio_key = feature_extractor.model_input_names[0]
    train_ds.set_format(type="torch", columns=[audio_key, "labels"])
    eval_ds.set_format(type="torch",  columns=[audio_key, "labels"])

    print(f"building model: {cfg.model_name}")
    model, family, _ = build_model(cfg, num_labels=len(cfg.label_order),
                                   label2id=label2id, id2label=id2label)

    data_collator = DataCollatorForFrame(feature_extractor, cfg.task_type)
    compute_metrics = compute_frame_metrics

    steps_per_epoch = max(1, len(train_ds) // (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    if isinstance(cfg.logging_steps, str) and cfg.logging_steps == "auto":
        logging_steps = max(1, steps_per_epoch // 10)
        print(f"📝 logging_steps=auto → {logging_steps} (~10/epoch, "
              f"steps_per_epoch={steps_per_epoch})")
    else:
        logging_steps = int(cfg.logging_steps)

    _is_ampere = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8

    training_args = TrainingArguments(
        output_dir=str(phase_dir / "trainer_tmp"),
        eval_strategy="no", save_strategy="no",
        logging_strategy="steps", logging_steps=logging_steps,
        report_to="none", label_names=["labels"],
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps, lr_scheduler_type="linear",
        max_grad_norm=cfg.max_grad_norm,
        remove_unused_columns=False,
        use_cpu=(device == "cpu"),
        bf16=(device == "cuda"), tf32=_is_ampere,
        dataloader_num_workers=dataloader_num_workers,
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
    print(f"🚀 training {cfg.num_epochs} epochs "
          f"(bs={cfg.batch_size} ga={cfg.grad_accum} lr={cfg.learning_rate})")
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
        (best_dir / "run_info.txt").write_text(
            f"run_name: {run_dir.name}\nrun_dir:  {run_dir}\n"
            f"phase:    {phase_name}\nepoch:    {best['epoch']}\n"
        )
        print(f"   saved best model → {best_dir.relative_to(PROJECT_ROOT)}")

    shutil.rmtree(phase_dir / "trainer_tmp", ignore_errors=True)
    del trainer, model
    release_gpu(verbose=True)
    return callback.epoch_results, best


# ═══════════════════════════════════════════════════════════════════════════════
# Run setup + summary
# ═══════════════════════════════════════════════════════════════════════════════

def make_run_dirs(cfg: Config, train_records: list[dict]) -> tuple[Path, Path, str]:
    dataset_name = train_records[0]["dataset"]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = cfg.model_name.replace("/", "_")
    run_name = f"{dataset_name}_{cfg.label_key}_{safe_model}_{ts}"
    run_dir   = udp.from_project_relative(cfg.runs_dir)   / run_name
    model_dir = udp.from_project_relative(cfg.models_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir   = {run_dir.relative_to(PROJECT_ROOT)}    (per-epoch logs)")
    print(f"model_dir = {model_dir.relative_to(PROJECT_ROOT)}  (best_model goes here)")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    return run_dir, model_dir, run_name


def print_run_summary(cfg: Config, run_name: str, run_dir: Path, model_dir: Path,
                      phase1_best: dict, phase2_best: dict) -> None:
    udp.banner(f"RUN SUMMARY: {run_name}")
    print(f"task        : {cfg.task_type}")
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


def _best_test_predictions_path(run_dir: Path, phase2_best: dict) -> Path:
    """predictions.json for the best phase-2 (TEST) epoch. Raises if missing."""
    p = (run_dir / "phase2_test" / "epoch_logs"
         / f"epoch_{phase2_best['epoch']}" / "predictions.json")
    if not p.exists():
        raise FileNotFoundError(f"expected predictions at {p}")
    return p


def plot_test_confusion(run_dir: Path, phase2_best: dict, label2id: dict,
                        label_order: list, show: bool = True) -> None:
    """Frame-level confusion matrix on the best phase-2 (TEST) epoch, stacked:
    absolute counts on top, row-normalized percentages below (each row sums to
    100 — read it as ``of the true X frames, what did the model call them?``).
    Saves to ``<run_dir>/confusion_matrix_test.png``."""
    preds_data = json.loads(_best_test_predictions_path(run_dir, phase2_best).read_text())
    y_true = [g for p in preds_data for g in p["gold_raw"]]
    y_pred = [pr for p in preds_data for pr in p["pred_raw"]]

    str_labels = [str(x) for x in label_order]
    n = len(label_order)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_rel = np.divide(cm.astype(float), row_sums,
                       out=np.zeros_like(cm, dtype=float),
                       where=row_sums != 0) * 100.0

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
    out = run_dir / "confusion_matrix_test.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out.relative_to(PROJECT_ROOT)}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_test_example_predictions(run_dir: Path, phase2_best: dict, id2label: dict,
                                  n_examples: int = 6, seed: int = 0,
                                  show: bool = True) -> None:
    """Gold-over-pred frame-strip plot for `n_examples` random TEST records
    from the best phase-2 epoch. Saves to
    ``<run_dir>/example_predictions_test.png``."""
    preds_data = json.loads(_best_test_predictions_path(run_dir, phase2_best).read_text())
    n = min(n_examples, len(preds_data))
    if n == 0:
        print("⚠️  no predictions to plot"); return
    sample = random.Random(seed).sample(preds_data, k=n)

    fig, axes = plt.subplots(n, 1, figsize=(10, max(2, 0.9 * n)), squeeze=False)
    axes = axes[:, 0]
    n_classes = len(id2label); pal = None
    for i, p in enumerate(sample):
        gold_img, pal = _row_to_image(p["gold_raw"], n_classes)
        pred_img, _   = _row_to_image(p["pred_raw"], n_classes)
        strip = np.concatenate([gold_img, pred_img], axis=0)
        ax = axes[i]
        ax.imshow(strip, aspect="auto", interpolation="nearest")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["gold", "pred"], fontsize=8)
        ax.set_xticks([])
        title = p["instance_id"]
        ax.set_title(("..." + title[-57:]) if len(title) > 60 else title,
                     fontsize=8, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=pal[k]) for k in range(n_classes)]
    fig.legend(handles, [str(id2label[k]) for k in range(n_classes)],
               loc="lower center", ncol=min(n_classes, 6), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = run_dir / "example_predictions_test.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out.relative_to(PROJECT_ROOT)}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def spot_check(run_dir: Path, phase2_best: dict, k: int = 5, seed: int = 0) -> None:
    epoch_dir = run_dir / "phase2_test" / "epoch_logs" / f"epoch_{phase2_best['epoch']}"
    preds_path = epoch_dir / "predictions.json"
    if not preds_path.exists():
        print(f"⚠️  {preds_path} missing — nothing to spot-check")
        return
    rows = json.loads(preds_path.read_text())
    sample = random.Random(seed).sample(rows, k=min(k, len(rows)))
    udp.banner(f"INFERENCE SPOT-CHECK — {len(sample)} random TEST examples")
    for p in sample:
        print(f"  {p['instance_id']}   ({p['n_frames']} frames)")
        gold = p["gold_raw"]; pred = p["pred_raw"]
        matched = sum(1 for g, pr in zip(gold, pred) if g == pr)
        print(f"     frame-match  {matched}/{len(gold)}  "
              f"({100.0 * matched / max(1, len(gold)):.1f}%)")
        if "prob_pos" in p and p["prob_pos"]:
            peak = max(p["prob_pos"])
            print(f"     peak prob_pos = {peak:.3f}")
        print()
