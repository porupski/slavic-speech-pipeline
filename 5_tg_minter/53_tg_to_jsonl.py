# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: ssp
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 53 — Annotated TextGrids → canonical JSONL
#
# Once an annotator has filled in the `fp-annotation` tier on the TextGrids
# minted by `51`, this notebook turns the annotated folder into a canonical
# pipeline JSONL ready for chapter 3 — rung 3 in the BLUEPRINT ladder
# (per-FP-event instance classification: Vocal / Nasal / V-N / Other /
# FalsePositive).
#
# **Two modes:**
#
# - **Merge mode (default)** — the TextGrid stem is matched against the
#   canonical 11c output (`parlaspeech_<lang>_utterance_instance.jsonl`) by the
#   `audio_path` basename. The full canonical record is recovered (file_id,
#   speaker, speaker_info, splits, the utterance's session-absolute bounds) and
#   each annotated FP event becomes one event-instance record whose `start_t` /
#   `end_t` carry through the utterance offset. Splits come from the parent
#   utterance — no separate `assign_splits` call.
# - **As-is mode** — emit only what the TextGrid plus its sibling audio file
#   carry (`instance_id` from the stem, `audio_path` to the per-event cut,
#   `labels.fp_type`). All records land in `train`; the user is expected to
#   re-split later if needed.
#
# **One record = one annotated interval on the annotation tier** (event
# instance, `cut=True` — per-event 16 kHz mono WAV written under
# `data/cut_audio/<dataset_tag>/<file_id>/`). Intervals that still carry the
# pending placeholder (`?` by default) are emitted with `labels.fp_type =
# null` — the trainer drops those per-target, so they cost only themselves.
# Empty intervals are ignored.

# %% [markdown]
# ## ▼ USER SETTINGS — edit this cell, then run the rest

# %%
from pathlib import Path

# --- WHERE ------------------------------------------------------------------
# Absolute paths are safest.
TG_DIR        = Path("path/to/your/annotated_textgrids")   # *.merged.TextGrid live here
UTT_AUDIO_DIR = Path("path/to/your/utterance_wavs")        # 16 kHz mono utterance WAVs
                                                            # (basename = TG stem)
OUTPUT_JSONL  = "data/processed_jsonl/parlaspeech_hr_fp_events.jsonl"  # project-relative

# --- MERGE WITH PSv3 / 11c CANONICAL ---------------------------------------
#   When True, look up each TG's source utterance in SOURCE_JSONL by
#   audio_path basename. The recovered record provides file_id, speaker,
#   speaker_info, split, and the utterance's session bounds. When False, the
#   notebook emits minimal records (no metadata enrichment, split = "train").
MERGE_PSV3   = True
SOURCE_JSONL = "data/processed_jsonl/parlaspeech_hr_utterance_instance.jsonl"

# --- ANNOTATION TIER --------------------------------------------------------
ANNOTATION_TIER = "fp-annotation"             # name from 51's mint
PENDING_LABEL   = "?"                          # rewritten to None on output
VALID_LABELS    = ("Vocal", "Nasal", "V-N", "Other", "FalsePositive")

# --- OUTPUT NAMING ----------------------------------------------------------
DATASET_TAG   = "ParlaSpeech-HR-FP"            # ends up in record["dataset"]
EVENT_CUT_DIR = "data/cut_audio/ParlaSpeech-HR-FP"  # project-relative; one
                                                     # sub-folder per file_id

# --- RUN OPTIONS ------------------------------------------------------------
PROCESS_LIMIT = None        # cap how many TGs to process (None = all)
OVERWRITE_CUTS = False      # re-write existing per-event WAVs?
# ----------------------------------------------------------------------------

# %% [markdown]
# ## Imports & helpers

# %%
import json
import sys
from collections import Counter
from dataclasses import asdict

import pandas as pd
from praatio import textgrid as tgio
from praatio.utilities.constants import INTERVAL_TIER

# Chapter-1 utils: canonical JSONL I/O + the audio cutter used by 11c.
HERE = Path.cwd()
if HERE.name != "5_tg_minter":
    cand = HERE / "5_tg_minter"
    if cand.exists():
        HERE = cand
sys.path.insert(0, str(HERE.parent / "1_data_prep"))

import utils_dataprep as udp

PROJECT_ROOT = udp.PROJECT_ROOT
print(f"PROJECT_ROOT = {PROJECT_ROOT}")


def read_tg(path: Path):
    return tgio.openTextgrid(str(path), includeEmptyIntervals=True,
                             reportingMode="silence", duplicateNamesMode="rename")


def tg_stem(path: Path) -> str:
    """'<base>.merged.TextGrid' -> '<base>'. Robust to dots inside base."""
    name = path.name
    suffix = ".merged.TextGrid"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def annotation_intervals(tg, tier_name: str) -> list[tuple[float, float, str]]:
    """Return [(start, end, raw_label), ...] for the annotation tier.
    Skips intervals whose label is whitespace-only (truly empty)."""
    if tier_name not in tg.tierNames:
        return []
    tier = tg.getTier(tier_name)
    if tier.tierType != INTERVAL_TIER:
        raise ValueError(f"annotation tier '{tier_name}' is not an interval tier")
    out: list[tuple[float, float, str]] = []
    for e in tier.entries:
        lab = (e.label or "").strip()
        if not lab:
            continue
        out.append((float(e.start), float(e.end), lab))
    return out


def normalize_label(raw: str) -> str | None:
    """`?` (or whatever PENDING_LABEL is) -> None. Anything else -> as-is
    (the inventory cell reports any out-of-vocabulary strings before emit)."""
    if raw == PENDING_LABEL:
        return None
    return raw


# %% [markdown]
# ## Step 1 — Load TGs (and the merge source)
#
# Glob the annotated folder, parse the stem off each `<base>.merged.TextGrid`,
# and (when merging) build a basename → utterance-record index from the
# source canonical JSONL. A TG with no match in merge mode is a loud warning,
# not a crash — annotation passes often run on a subset and we want to surface
# missing matches rather than skip them silently.

# %%
TG_DIR = Path(TG_DIR)
tg_paths = sorted(TG_DIR.glob("*.merged.TextGrid"))
if PROCESS_LIMIT is not None:
    tg_paths = tg_paths[:PROCESS_LIMIT]
print(f"found {len(tg_paths)} annotated TextGrids in {TG_DIR}")

if MERGE_PSV3:
    source_records = udp.read_jsonl(SOURCE_JSONL)
    by_basename: dict[str, dict] = {}
    for r in source_records:
        ap = r.get("audio_path")
        if not ap:
            continue
        by_basename[Path(ap).stem] = r
    print(f"indexed {len(by_basename)} utterance records from {SOURCE_JSONL}")
else:
    source_records = []
    by_basename = {}
    print("merge disabled — records will carry only what the TGs themselves provide")

# %% [markdown]
# ## Step 2 — Inventory: label distribution and join coverage
#
# Walk every TG, pull the annotation-tier intervals, count raw labels. In merge
# mode also report how many TG stems matched the source JSONL. Use this to
# spot:
#
# - out-of-vocabulary labels (a typo, or a label the recipe doesn't know yet);
# - a high `?` count (annotation still in progress);
# - missing source utterances (the join key is off, or the merge source is the
#   wrong language).

# %%
raw_counter: Counter = Counter()
events_per_tg: list[int] = []
tg_with_tier = 0
matched, unmatched = 0, []

for p in tg_paths:
    tg = read_tg(p)
    if ANNOTATION_TIER in tg.tierNames:
        tg_with_tier += 1
    ints = annotation_intervals(tg, ANNOTATION_TIER)
    raw_counter.update(lab for _, _, lab in ints)
    events_per_tg.append(len(ints))
    if MERGE_PSV3:
        if tg_stem(p) in by_basename:
            matched += 1
        else:
            unmatched.append(p.name)

print(f"TGs with '{ANNOTATION_TIER}' tier : {tg_with_tier} / {len(tg_paths)}")
print(f"annotated intervals (raw)        : {sum(raw_counter.values())}")
print(f"events per TG                    : "
      f"min={min(events_per_tg, default=0)}  "
      f"median={int(pd.Series(events_per_tg).median()) if events_per_tg else 0}  "
      f"max={max(events_per_tg, default=0)}")

print("\nraw label counts:")
for lab, n in raw_counter.most_common():
    marker = "" if (lab == PENDING_LABEL or lab in VALID_LABELS) else "  <- OUT OF VOCAB"
    print(f"  {lab:<20} {n:>8}{marker}")

if MERGE_PSV3:
    print(f"\njoin coverage                    : {matched}/{len(tg_paths)} matched")
    if unmatched:
        print(f"unmatched TG stems (first 5)     : {unmatched[:5]}")

# %% [markdown]
# ## Step 3 — Cut event clips + write JSONL
#
# For each annotated TG:
#
# 1. Locate the parent utterance WAV (`UTT_AUDIO_DIR / <base>.wav` — in merge
#    mode we cross-check against the source record's `audio_path`).
# 2. For each annotated interval on the annotation tier, cut a 16 kHz mono
#    event clip into `data/cut_audio/<DATASET_TAG>/<file_id>/<event>.wav` via
#    `utils_dataprep.resample_to_16k_mono` (which also handles the resample if
#    the parent isn't 16 kHz).
# 3. Emit one canonical record per event. In merge mode `start_t` / `end_t`
#    are session-absolute (utterance start + event offset); without merge they
#    are relative to the parent WAV.

# %%
def event_record(*, tg_base: str, ev_idx: int, ev_start: float, ev_end: float,
                 raw_label: str, event_wav: Path, audio_length: float,
                 parent: dict | None) -> dict:
    """Assemble one canonical event-instance record."""
    label = normalize_label(raw_label)

    if parent is not None:
        utt_start = float(parent.get("start_t", 0.0))
        utt_end   = float(parent.get("end_t",   utt_start + audio_length))
        file_id   = parent.get("file_id", tg_base)
        speaker   = parent.get("speaker")
        split     = parent.get("split", "train")
        parent_md = parent.get("metadata", {}) or {}
        instance_id = f"{parent['instance_id']}_ev{ev_idx:02d}_{ev_start:.3f}-{ev_end:.3f}"
        start_t = utt_start + ev_start
        end_t   = utt_start + ev_end
    else:
        file_id   = tg_base
        speaker   = None
        split     = "train"
        parent_md = {}
        instance_id = udp.make_instance_id(DATASET_TAG, tg_base, None, ev_start, ev_end)
        instance_id = f"{instance_id}_ev{ev_idx:02d}"
        start_t = ev_start
        end_t   = ev_end

    record = {
        "instance_id": instance_id,
        "dataset":     DATASET_TAG,
        "file_id":     file_id,
        "audio_path":  udp.to_project_relative(event_wav),
        "split":       split,
        "start_t":     round(start_t, 3),
        "end_t":       round(end_t,   3),
        "labels":      {"fp_type": label},
        "metadata": {
            "source_audio":         parent.get("audio_path") if parent else str(event_wav.parent),
            "source_utterance_id":  parent["instance_id"] if parent else None,
            "event_index":          ev_idx,
            "event_start_in_utt_s": round(ev_start, 3),
            "event_end_in_utt_s":   round(ev_end,   3),
            "audio_length":         round(audio_length, 3),
            "annotation_tier":      ANNOTATION_TIER,
            "raw_label":            raw_label,
            "speaker_info":         parent_md.get("speaker_info") if parent_md else None,
        },
    }
    if speaker is not None:
        record["speaker"] = speaker
    return record


written_records: list[dict] = []
written_cuts = 0
skipped_no_parent_wav = 0
skipped_no_merge_match = 0

UTT_AUDIO_DIR = Path(UTT_AUDIO_DIR)
cut_root = udp.from_project_relative(EVENT_CUT_DIR)
cut_root.mkdir(parents=True, exist_ok=True)

for p in tg_paths:
    base = tg_stem(p)
    parent = by_basename.get(base) if MERGE_PSV3 else None
    if MERGE_PSV3 and parent is None:
        skipped_no_merge_match += 1
        continue

    # Locate the source utterance WAV. Merge mode trusts the canonical path
    # in the parent record; as-is mode falls back to UTT_AUDIO_DIR/<base>.wav.
    if parent is not None:
        src_wav = udp.from_project_relative(parent["audio_path"])
    else:
        src_wav = UTT_AUDIO_DIR / f"{base}.wav"
    if not src_wav.exists():
        skipped_no_parent_wav += 1
        print(f"  parent WAV missing for {base}: {src_wav}")
        continue

    tg = read_tg(p)
    ints = annotation_intervals(tg, ANNOTATION_TIER)
    file_id = (parent.get("file_id") if parent else base) or base
    event_dir = cut_root / file_id
    event_dir.mkdir(parents=True, exist_ok=True)

    for idx, (ev_start, ev_end, raw_label) in enumerate(ints):
        ev_name = f"{base}__ev{idx:02d}__{ev_start:.3f}-{ev_end:.3f}.wav"
        ev_path = event_dir / ev_name
        if (not ev_path.exists()) or OVERWRITE_CUTS:
            udp.resample_to_16k_mono(src_wav, ev_path,
                                     start_t=ev_start, end_t=ev_end)
            written_cuts += 1
        audio_length = max(0.0, ev_end - ev_start)
        written_records.append(event_record(
            tg_base=base, ev_idx=idx, ev_start=ev_start, ev_end=ev_end,
            raw_label=raw_label, event_wav=ev_path, audio_length=audio_length,
            parent=parent,
        ))

print(f"\nemitted records : {len(written_records)}")
print(f"event WAVs cut  : {written_cuts} (existing kept unless OVERWRITE_CUTS)")
if skipped_no_merge_match:
    print(f"skipped (no merge match) : {skipped_no_merge_match}")
if skipped_no_parent_wav:
    print(f"skipped (parent WAV missing) : {skipped_no_parent_wav}")

n_written = udp.write_jsonl(written_records, OUTPUT_JSONL)
print(f"\nwrote {n_written} records -> {OUTPUT_JSONL}")

# %% [markdown]
# ## Step 4 — Verify
#
# Re-read the output, run it through the canonical-schema validator, and print
# a short summary: per-split counts, per-label counts (with pending = `null`
# shown alongside), and a peek at one record.

# %%
records = udp.read_jsonl(OUTPUT_JSONL)
n_total, n_valid, errs = udp.validate_jsonl(records)
print(f"validated : {n_valid}/{n_total} records")
for e in errs[:5]:
    print(f"  {e}")

split_counts = Counter(r["split"] for r in records)
label_counts = Counter("<null>" if r["labels"]["fp_type"] is None
                       else r["labels"]["fp_type"] for r in records)
print(f"\nsplit counts  : {dict(split_counts)}")
print(f"label counts  : {dict(label_counts)}")

if records:
    print("\nfirst record:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))
