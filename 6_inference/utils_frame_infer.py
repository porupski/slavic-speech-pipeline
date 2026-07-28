"""utils_frame_infer — the chapter-6 frame-classification inference engine.

Shared by:
  * 63_frame_classification_inference.ipynb / run_63_frame_classification_inference.py

Frame-classification model = one logit vector per ~20 ms audio frame. Runs of
identical predicted class along the frame axis collapse into events; the events
are then filtered with the drop_short / drop_initial / drop_final postproc rules
documented on the FP-BERT model card.

IMPORT-ORDER CONTRACT (same as chapter 3 — read before importing):
  1. The GPU guard must run BEFORE this module is imported. This module imports
     torch at the top, and ``CUDA_VISIBLE_DEVICES`` only takes effect if set
     before torch's first CUDA touch. Consumers keep a tiny visible guard block
     (set ``CUDA_VISIBLE_DEVICES`` to the reserved GPU or ``""``), then import.
  2. ``HF_HOME`` is handled HERE (project-local ``stock_models/``), before the
     transformers import below — consumers don't need to set it.
  3. Headless runners should set ``MPLBACKEND=Agg`` before importing (notebooks
     don't need to — the inline backend is already active).

Config flow: ``load_config(path, run_mode=...)`` reads config.json (shared block
+ mode overrides) and returns a validated ``Config``.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── PROJECT_ROOT + HF_HOME (must precede the transformers import) ─────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_data_prep"))
import utils_dataprep as udp

PROJECT_ROOT = udp.PROJECT_ROOT
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "stock_models"))

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification


# Number of random audio files rendered in the end-of-run examples plot.
N_PLOT_EXAMPLES = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # -- Run mode ------------------------------------------------------------
    run_mode: str = "full"                 # "test" | "demo" | "full"
    cap_files: int | None = None           # limit input audio files

    # -- Model ---------------------------------------------------------------
    # HF repo id OR local dir containing a frame-classification checkpoint
    # (AutoModelForAudioFrameClassification-compatible). Ch3 utterance-level
    # checkpoints are NOT compatible — different head.
    model_name: str = "classla/wav2vecbert2-filledPause"

    # -- Input / output ------------------------------------------------------
    audio_dir: str = "data/inference_input"
    runs_dir:  str = "runs"

    # -- Inference knobs -----------------------------------------------------
    batch_size:      int   = 8
    chunk_length_s:  float = 30.0   # 0 disables chunking (must be pre-segmented)
    background_class_id: int | None = 0   # None → emit events for all classes

    # -- Postprocessing (per FP-BERT model card) ----------------------------
    postproc_drop_short:     bool  = True
    postproc_drop_initial:   bool  = True
    postproc_drop_final:     bool  = True
    postproc_short_cutoff_s: float = 0.08

    # -- Output extras -------------------------------------------------------
    keep_frame_labels: bool = False   # append raw per-frame 0/1 array in JSONL
    write_textgrids:   bool = True

    # -- Hardware ------------------------------------------------------------
    reserved_gpu: str = "2"
    use_cuda:     bool = True


def apply_mode(cfg: Config, overrides: dict) -> None:
    """Layer a mode's overrides onto the base Config, in place."""
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"Config has no attribute {k!r}")
        setattr(cfg, k, v)


def load_config(path: str | Path, run_mode: str = "full") -> Config:
    """Read config.json, apply the mode overrides, return a Config."""
    with open(path) as f:
        raw = json.load(f)
    shared = raw.get("shared", {})
    modes  = raw.get("modes", {})
    if run_mode not in modes:
        raise KeyError(f"unknown run_mode {run_mode!r}; known: {list(modes)}")

    cfg = Config()
    for k, v in shared.items():
        if k == "postproc" and isinstance(v, dict):
            for pk, pv in v.items():
                fld = f"postproc_{pk}"
                if not hasattr(cfg, fld):
                    raise KeyError(f"unknown postproc field {pk!r}")
                setattr(cfg, fld, pv)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise KeyError(f"unknown shared config key {k!r}")

    apply_mode(cfg, modes[run_mode])
    cfg.run_mode = run_mode
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Device + summary
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_device(cfg: Config) -> str:
    if cfg.use_cuda and torch.cuda.is_available():
        return "cuda"
    if cfg.use_cuda:
        print("⚠️  GPU selected but torch.cuda.is_available()==False; falling back to CPU")
    return "cpu"


def print_config_summary(cfg: Config, device: str) -> None:
    print(f"model          = {cfg.model_name}")
    print(f"audio_dir      = {cfg.audio_dir}")
    print(f"run_mode       = {cfg.run_mode}  (cap_files={cfg.cap_files})")
    print(f"chunk_length_s = {cfg.chunk_length_s}")
    print(f"batch_size     = {cfg.batch_size}")
    print(f"device         = {device}")
    if device == "cuda":
        print(f"✓ device name  : {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Model + feature extractor
# ═══════════════════════════════════════════════════════════════════════════════

def load_feature_extractor(cfg: Config):
    """AutoFeatureExtractor. `model_input_names[0]` is read at call sites so the
    same code works with both wav2vec2 (input_values) and wav2vec-BERT
    (input_features) frame models."""
    fe = AutoFeatureExtractor.from_pretrained(cfg.model_name)
    fe.return_attention_mask = True
    print(f"loaded feature extractor: {cfg.model_name}")
    print(f"   input key : {fe.model_input_names[0]}")
    return fe


def load_model(cfg: Config, device: str):
    """AutoModelForAudioFrameClassification + move to device + eval mode.
    Returns (model, id2label dict with int keys)."""
    model = AutoModelForAudioFrameClassification.from_pretrained(cfg.model_name)
    model.to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"loaded model: {cfg.model_name} ({len(id2label)} classes: {id2label})")
    return model, id2label


# ═══════════════════════════════════════════════════════════════════════════════
# Audio IO + chunking
# ═══════════════════════════════════════════════════════════════════════════════

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")


def iter_audio_files(audio_dir: Path):
    """Recursively yield audio files under audio_dir, sorted for determinism."""
    audio_dir = Path(audio_dir)
    yield from sorted(p for p in audio_dir.rglob("*")
                      if p.suffix.lower() in AUDIO_EXTS)


def read_audio(path: Path):
    """Read audio → mono float32 16 kHz. Resamples via librosa if the source
    isn't already 16 kHz (defensive fallback; Chapter-1 clips are already 16k)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    return data, sr


def chunk_audio(data: np.ndarray, sr: int, chunk_length_s: float):
    """Split waveform into non-overlapping fixed-length chunks. Returns a list of
    (start_s, end_s, chunk_array). chunk_length_s<=0 or duration<=chunk_length_s
    → single chunk covering the whole clip."""
    n = len(data)
    total_s = n / sr
    if chunk_length_s <= 0 or total_s <= chunk_length_s:
        return [(0.0, total_s, data)]
    chunk_size = int(round(chunk_length_s * sr))
    out = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out.append((start / sr, end / sr, data[start:end]))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Forward pass
# ═══════════════════════════════════════════════════════════════════════════════

def run_chunks(chunk_arrays, model, fe, device, batch_size):
    """Run the frame classifier on a list of chunk waveforms.

    Returns two parallel lists (one entry per chunk):
      preds  — 1-D int array of per-frame class ids, length = real frame count
      probs  — 1-D float array of the softmax probability of the winning class
    Attention-mask sums are used to trim padding introduced by batch padding."""
    all_preds, all_probs = [], []
    for i in range(0, len(chunk_arrays), batch_size):
        batch = chunk_arrays[i:i + batch_size]
        inputs = fe(
            batch, sampling_rate=16000, return_tensors="pt", padding="longest",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits              # (B, T, C)
        probs_all = torch.softmax(logits, dim=-1)        # (B, T, C)
        preds = probs_all.argmax(dim=-1)                 # (B, T)
        # Softmax prob of the argmax class per frame:
        winner_probs = probs_all.gather(-1, preds.unsqueeze(-1)).squeeze(-1)  # (B, T)

        preds_np = preds.cpu().numpy()
        winner_np = winner_probs.cpu().numpy()
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask_np = mask.cpu().numpy()
        for j in range(len(batch)):
            if mask is not None:
                # attention_mask is at input-frame resolution, which for these
                # models matches the classifier output-frame resolution.
                real = int(mask_np[j].sum())
                # Guard against mask/output rate mismatch: never index beyond
                # the actual output sequence length.
                real = min(real, preds_np.shape[1])
                all_preds.append(preds_np[j, :real])
                all_probs.append(winner_np[j, :real])
            else:
                all_preds.append(preds_np[j])
                all_probs.append(winner_np[j])
    return all_preds, all_probs


# ═══════════════════════════════════════════════════════════════════════════════
# Frames → events + postprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def frames_to_events(frame_preds, frame_probs, id2label, frame_s,
                     background_class_id=None, chunk_start_s=0.0):
    """Collapse a per-frame prediction sequence into events (contiguous runs of
    the same class). Skips background if its class id is given. Returns a list
    of dicts with keys: start_s, end_s, label, mean_prob."""
    preds = np.asarray(frame_preds)
    probs = np.asarray(frame_probs)
    if len(preds) == 0:
        return []
    # np.diff-based run-start detection; prepend a sentinel so the first index
    # always counts as a change.
    change_idx = np.where(np.diff(preds, prepend=preds[0] - 1))[0]
    change_idx = np.append(change_idx, len(preds))
    events = []
    for start, end in zip(change_idx[:-1], change_idx[1:]):
        label_id = int(preds[start])
        if background_class_id is not None and label_id == background_class_id:
            continue
        events.append({
            "start_s":   round(chunk_start_s + float(start * frame_s), 3),
            "end_s":     round(chunk_start_s + float(end   * frame_s), 3),
            "label":     id2label[label_id],
            "mean_prob": round(float(probs[start:end].mean()), 4),
        })
    return events


def postprocess_events(events, cfg: Config, file_duration_s: float):
    """Apply the FP-model card's postproc: drop events starting at 0.0,
    ending at file end, or shorter than short_cutoff_s. Each flag is
    independently toggleable via config."""
    out = list(events)
    if cfg.postproc_drop_initial:
        out = [e for e in out if e["start_s"] > 0.0]
    if cfg.postproc_drop_final:
        # Compare with a small epsilon: with float rounding an "end at file end"
        # can land a hair below duration_s. Use 20 ms as an epsilon — one frame.
        out = [e for e in out if e["end_s"] < file_duration_s - 0.02]
    if cfg.postproc_drop_short:
        out = [e for e in out if (e["end_s"] - e["start_s"]) >= cfg.postproc_short_cutoff_s]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# TextGrid writer
# ═══════════════════════════════════════════════════════════════════════════════

def _events_to_intervals(events, duration_s):
    """praatio needs full gap-fill; return a list of (start, end, label)
    covering [0, duration_s] with empty labels for gaps."""
    intervals = []
    cursor = 0.0
    for e in events:
        s, t = e["start_s"], e["end_s"]
        if s > cursor:
            intervals.append((cursor, s, ""))
        intervals.append((s, t, e["label"]))
        cursor = t
    if cursor < duration_s:
        intervals.append((cursor, duration_s, ""))
    return intervals


def write_textgrid(path: Path, duration_s: float, raw_events, postproc_events):
    """Two-tier TextGrid: tier 1 raw events, tier 2 postproc events."""
    from praatio import textgrid
    from praatio.utilities.constants import Interval

    tg = textgrid.Textgrid(minTimestamp=0.0, maxTimestamp=duration_s)
    for tier_name, evs in [("raw_events", raw_events),
                           ("postproc_events", postproc_events)]:
        entries = [Interval(s, t, lbl)
                   for (s, t, lbl) in _events_to_intervals(evs, duration_s)]
        tier = textgrid.IntervalTier(tier_name, entries, minT=0.0, maxT=duration_s)
        tg.addTier(tier)
    tg.save(str(path), format="short_textgrid", includeBlankSpaces=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting — waveform + event bars, Praat-like
# ═══════════════════════════════════════════════════════════════════════════════

def plot_examples(examples, out_png, show=False):
    """Render one row per example: waveform on top, red bars beneath marking
    the (post-processed) event intervals."""
    n = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, ex in zip(axes, examples):
        data, sr = ex["waveform"], ex["sr"]
        t = np.arange(len(data)) / sr
        ax.plot(t, data, color="steelblue", lw=0.4)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, len(data) / sr if len(data) else 1)
        ax.set_xlabel("time (s)")
        ax.set_title(ex["title"], fontsize=9, loc="left")
        # Event overlay — thin red bar at the bottom of the axis.
        for e in ex["events"]:
            ax.axvspan(e["start_s"], e["end_s"], ymin=0.0, ymax=0.10,
                       color="tab:red", alpha=0.7)
    fig.savefig(out_png, dpi=120)
    if not show:
        plt.close(fig)
    return fig


def sample_plot_examples(out_jsonl: Path, run_dir: Path,
                         n: int = N_PLOT_EXAMPLES, seed: int = 1234,
                         show: bool = True):
    """Read n random JSONL lines, reload audio, render examples.png."""
    out_jsonl = Path(out_jsonl)
    if not out_jsonl.exists():
        print(f"no inference jsonl at {out_jsonl.relative_to(PROJECT_ROOT)}; skipping plots")
        return None
    lines = out_jsonl.read_text().splitlines()
    if not lines:
        print("no examples to plot")
        return None
    rng = random.Random(seed)
    sample = rng.sample(lines, min(n, len(lines)))
    examples = []
    for line in sample:
        rec = json.loads(line)
        audio_path = PROJECT_ROOT / rec["audio_path"]
        try:
            data, sr = read_audio(audio_path)
        except Exception as exc:
            print(f"⚠️  skipping plot for {rec['audio_path']}: {exc}")
            continue
        events = rec.get("postproc_events", [])
        examples.append({
            "waveform": data, "sr": sr,
            "events": events,
            "title": f"{rec['audio_path']}   ·   {rec['duration_s']}s   ·   {len(events)} FP events",
        })
    if not examples:
        return None
    out_png = run_dir / "examples.png"
    plot_examples(examples, out_png, show=show)
    print(f"saved {out_png.relative_to(PROJECT_ROOT)}")
    return out_png


# ═══════════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════════

def _model_revision(model) -> str:
    """Best-effort HF revision hash. Falls back to 'unknown'."""
    return getattr(model.config, "_commit_hash", None) or "unknown"


def run_inference(cfg: Config, run_name: str) -> dict:
    """Walk cfg.audio_dir, run the frame classifier, stream results to
    runs/{run_name}/inference.jsonl. Writes TextGrids if requested. Returns a
    dict with run_dir / out_jsonl / counts."""
    t0 = time.time()
    device = resolve_device(cfg)
    print_config_summary(cfg, device)

    # Prepare output dirs.
    run_dir = udp.from_project_relative(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = run_dir / "inference.jsonl"
    tg_dir = run_dir / "textgrids"
    if cfg.write_textgrids:
        tg_dir.mkdir(exist_ok=True)

    # Resume support: skip audio_paths already covered by an existing JSONL.
    already_done: set[str] = set()
    if out_jsonl.exists():
        with open(out_jsonl) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    already_done.add(rec.get("audio_path"))
                except json.JSONDecodeError:
                    continue
        if already_done:
            print(f"resume: {len(already_done)} file(s) already covered → "
                  f"{out_jsonl.relative_to(PROJECT_ROOT)}")

    # Discover audio.
    audio_dir_abs = udp.from_project_relative(cfg.audio_dir)
    if not audio_dir_abs.exists():
        raise FileNotFoundError(f"audio_dir does not exist: "
                                f"{audio_dir_abs.relative_to(PROJECT_ROOT)}")
    audio_files = list(iter_audio_files(audio_dir_abs))
    print(f"found {len(audio_files)} audio files under "
          f"{audio_dir_abs.relative_to(PROJECT_ROOT)}")

    todo = [p for p in audio_files
            if str(p.resolve().relative_to(PROJECT_ROOT)) not in already_done]
    if cfg.cap_files is not None:
        todo = todo[:cfg.cap_files]
    print(f"queue: {len(todo)} file(s) to process  (cap_files={cfg.cap_files})")

    if not todo:
        print("nothing to do.")
        return {"run_dir": run_dir, "out_jsonl": out_jsonl,
                "processed": 0, "total_events": 0, "total_event_time": 0.0}

    # Load model + FE.
    fe = load_feature_extractor(cfg)
    model, id2label = load_model(cfg, device)
    model_rev = _model_revision(model)

    processed = 0
    total_events = 0
    total_event_time = 0.0

    # Prefer tqdm if available; fall back to a plain loop otherwise.
    try:
        from tqdm import tqdm
        it = tqdm(todo, desc="inference", unit="file")
    except ImportError:
        it = todo

    with open(out_jsonl, "a") as fout:
        for path in it:
            rel_path = str(path.resolve().relative_to(PROJECT_ROOT))
            try:
                data, sr = read_audio(path)
            except Exception as exc:
                print(f"⚠️  failed to read {rel_path}: {exc}")
                continue

            duration_s = len(data) / sr
            chunks = chunk_audio(data, sr, cfg.chunk_length_s)
            preds_list, probs_list = run_chunks(
                [c[2] for c in chunks], model, fe, device, cfg.batch_size,
            )

            # Derive frame_s empirically from chunk 0 (works for any frame rate).
            first_chunk_dur = chunks[0][1] - chunks[0][0]
            first_chunk_frames = max(1, len(preds_list[0]))
            frame_s = first_chunk_dur / first_chunk_frames

            raw_events, frame_dump = [], []
            for (c_start, c_end, _), preds, probs in zip(chunks, preds_list, probs_list):
                raw_events.extend(frames_to_events(
                    preds, probs, id2label, frame_s,
                    background_class_id=cfg.background_class_id,
                    chunk_start_s=c_start,
                ))
                if cfg.keep_frame_labels:
                    frame_dump.extend(int(x) for x in preds)

            postproc_events = postprocess_events(raw_events, cfg, duration_s)

            rec: dict = {
                "audio_path":       rel_path,
                "duration_s":       round(duration_s, 3),
                "model":            cfg.model_name,
                "model_revision":   model_rev,
                "frame_ms":         round(1000 * frame_s, 3),
                "id2label":         {str(k): v for k, v in id2label.items()},
                "chunks": [
                    {"start_s": round(cs, 3), "end_s": round(ce, 3)}
                    for cs, ce, _ in chunks
                ],
                "raw_events":       raw_events,
                "postproc_events":  postproc_events,
                "postproc_applied": {
                    "drop_short":     cfg.postproc_drop_short,
                    "drop_initial":   cfg.postproc_drop_initial,
                    "drop_final":     cfg.postproc_drop_final,
                    "short_cutoff_s": cfg.postproc_short_cutoff_s,
                },
            }
            if cfg.keep_frame_labels:
                # Keep it at the end so the first lines of a JSONL preview
                # remain readable — a full frame dump can be thousands of ints.
                rec["frames"] = frame_dump

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if cfg.write_textgrids:
                try:
                    tg_path = tg_dir / (path.stem + ".TextGrid")
                    write_textgrid(tg_path, duration_s, raw_events, postproc_events)
                except Exception as exc:
                    print(f"⚠️  textgrid write failed for {rel_path}: {exc}")

            processed += 1
            total_events += len(postproc_events)
            total_event_time += sum(e["end_s"] - e["start_s"] for e in postproc_events)

    elapsed = time.time() - t0
    summary_lines = [
        f"processed files       = {processed}",
        f"total postproc events = {total_events}",
        f"total event time (s)  = {total_event_time:.2f}",
        f"elapsed (s)           = {elapsed:.1f}",
        f"model                 = {cfg.model_name}  (rev {model_rev})",
        f"output jsonl          = {out_jsonl.relative_to(PROJECT_ROOT)}",
    ]
    if cfg.write_textgrids:
        summary_lines.append(f"textgrids dir         = {tg_dir.relative_to(PROJECT_ROOT)}")
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (run_dir / "run_summary.txt").write_text(summary + "\n")

    return {
        "run_dir": run_dir,
        "out_jsonl": out_jsonl,
        "processed": processed,
        "total_events": total_events,
        "total_event_time": total_event_time,
        "elapsed_s": elapsed,
    }
