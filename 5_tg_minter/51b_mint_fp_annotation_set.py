# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 51b — Mint FP Annotation Set
#
# Build a **balanced, numbered annotation pack** of filled-pause utterances for a
# human annotator. A selection front-end bolted onto the TextGrid minter (`51`),
# sibling to the word cherry-picker (`52`) — but instead of picking by target word
# and bucketing by stress, this picks by **filled pauses** and balances by
# **gender + speaker variety**:
#
# 1. stream the master **JSONL**, keep utterances with at least
#    `REQUIRE_MIN_FP` filled-pause events (+ optional duration filters);
# 2. **balance** to a fixed per-gender quota (default 1000 M / 1000 F), spreading
#    across as many distinct **`Speaker_ID`s** as possible (wide before deep). The
#    per-speaker ideal is only a target — the cap relaxes automatically so the
#    quota is always hit if the pool allows;
# 3. **number** the selected clips `0000…N-1`, interleaving genders so the set is
#    order-blind, and split them into **`batch_00 … batch_KK`** folders of
#    `PER_FOLDER` clips each — bite-sized for the annotator;
# 4. **mint** each clip's merged TextGrid (the `51` recipe + a blank
#    `fp-annotation` tier mirroring FilledPauses + an instructions tier) and copy
#    its **`.wav`** (converted from FLAC, resampled) right beside it;
# 5. write a top-level **`_README.txt`** and a **`_manifest.tsv` / `.jsonl`** that
#    map every index back to speaker / gender / FP count — the ground-truth key.
#
# Output layout (gender is **not** in the filename — only in the manifest):
# ```
# OUT_DIR/
#   _README.txt   _manifest.tsv   _manifest.jsonl
#   batch_00/
#     0000_<hash>_<range>_fp3.merged.TextGrid  + 0000_<hash>_<range>_fp3.wav
#     0001_…_fp1.merged.TextGrid               + 0001_…_fp1.wav
#   batch_01/  …
# ```
#
# **`DRY_RUN = True` by default**: it does the whole scan / balance / numbering,
# checks that every selected clip has its layers and audio on disk, and writes the
# manifest of *what it would mint* — but skips the expensive merging / converting /
# copying. Eyeball the plan, then set `DRY_RUN = False` to actually mint.

# %% [markdown]
# ## ▼ USER SETTINGS — edit this cell, then run the rest

# %%
import os
from pathlib import Path

# --- WHERE (defaults are Ivan's paths; override with env vars of the same name) ---
# Master JSONL: the picker reads filled pauses + speaker info from here.
JSONL_PATH = Path(os.environ.get(
    "JSONL_PATH",
    "/cache/ivanp/projects/slavic-speech-pipeline/data/unpacked/ParlaSpeech-HR/ParlaSpeech-HR.v3.0/ParlaSpeech-HR.v3.0.jsonl"))

# Per-layer TextGrids (<base>.align.TextGrid, <base>.pause.TextGrid, …). One flat
# dir holding ALL of them (millions of files). Never listed whole — sampled via
# os.scandir with a hard cap. A doubled archive (…/X.textgrid/X.textgrid/) is
# auto-resolved if present.
IN_DIR = Path(os.environ.get(
    "IN_DIR",
    "/cache/ivanp/projects/slavic-speech-pipeline/data/unpacked/ParlaSpeech-HR/ParlaSpeech-HR.v3.0.textgrid"))

# Where the numbered, bucketed annotation pack is written.
OUT_DIR = Path(os.environ.get("OUT_DIR", "fp_annotation_set"))

# Audio. The shard subfolders (…part1 … part6) under AUDIO_DIR each hold {hash}
# folders containing the source clips as .flac:  {AUDIO_DIR}/{shard}/{hash}/{stem}.flac
# (the JSONL "audio" field already names {hash}/{stem}.flac). Sources are converted
# to .wav (resampled to TARGET_SR) and written into each batch folder — Praat-ready.
AUDIO_DIR = Path(os.environ.get(
    "AUDIO_DIR",
    "/cache/ivanp/projects/slavic-speech-pipeline/data/unpacked/ParlaSpeech-HR-audio"))
SHARD_GLOB       = "ParlaSpeech-HR.v2.0.part*"   # shard subfolders directly under AUDIO_DIR
AUDIO_SRC_EXTS   = [".flac", ".wav"]             # source extensions to look for, in order
TARGET_SR        = 16000                         # resample to this rate; None = keep source rate
COPY_AUDIO       = True                          # write a (converted) .wav into each batch folder
OUTPUT_AUDIO_EXT = ".wav"                         # always written as wav

# --- SELECTION: the filled-pause floor + optional filters --------------------
REQUIRE_MIN_FP   = 1        # keep only utterances with >= this many filled-pause events
MIN_DURATION_S   = None     # drop clips shorter than this (seconds); None = off
MAX_DURATION_S   = None     # drop clips longer  than this (seconds); None = off

# --- BALANCE: per-gender quota + speaker variety -----------------------------
#   Quota drives everything (total minted = sum of quotas). Selection spreads
#   across distinct Speaker_IDs (wide before deep). MAX_PER_SPEAKER is only the
#   *ideal* ceiling used for the report — it relaxes automatically so the quota
#   is always hit when the eligible pool is large enough.
GENDER_QUOTAS      = {"M": 1000, "F": 1000}
MAX_PER_SPEAKER    = 5      # ideal max clips per speaker (soft; relaxes to hit quota)
SPEAKER_ID_FIELD   = "Speaker_ID"          # variety grouping key (falls back to Speaker_name)
GENDER_FIELD       = "Speaker_gender"      # speaker_info field carrying gender
GENDER_MAP = {                              # normalise raw values -> quota keys
    "m": "M", "male": "M", "muski": "M", "muški": "M",
    "f": "F", "female": "F", "z": "F", "zenski": "F", "ženski": "F",
}
SEED = 42                   # deterministic shuffle -> reproducible set (re-mint is idempotent)

# --- NUMBERING & FOLDERING ---------------------------------------------------
PER_FOLDER         = 100    # clips per batch folder (2000 / 100 = 20 folders)
FOLDER_PREFIX      = "batch_"
FOLDER_WIDTH       = 2      # batch_00 … batch_19
INDEX_WIDTH        = 4      # 0000 … 1999
INTERLEAVE_GENDERS = True   # alternate M/F in the numbering so the set is order-blind

# --- WHAT TO COMBINE  (top-to-bottom = tier order in the output) ------------
#   (\"align\", \"all\") | (\"align\", [\"W\",\"G\"]) | (\"align\",\"W\",\"G\") | (\"pause\",\"FP\")
MERGE_RECIPE = [
    ("align", "all"),
    ("pause", "FilledPauses"),
]

# --- ADD A BLANK ANNOTATION TIER (mimics FilledPauses' boundaries) ----------
#   This is the tier the annotator fills in. Wherever FilledPauses had text, the
#   new interval is pre-marked with ANNOT_PRE_LABEL; the rest are blank.
ADD_MIMIC_TIER     = True
MIMIC_SOURCE_LAYER = "pause"
MIMIC_SOURCE_TIER  = "FilledPauses"   # exact tier name, or None = that layer's FIRST tier
MIMIC_NEW_TIER     = "fp-annotation"
ANNOT_PRE_LABEL    = "?"

# --- ADD AN INSTRUCTIONS TIER (label legend, spans the whole utterance) -----
ADD_INSTRUCTION_TIER  = True
INSTRUCTION_TIER_NAME = "instructions"
INSTRUCTION_TEXT      = "Valid labels: Vocal(aeiou@), Nasal(nm), V+N, Other(O), FalsePositive(F)"

# --- README handed to the annotator (top of OUT_DIR) ------------------------
WRITE_README = True
README_TEXT = """FILLED-PAUSE ANNOTATION SET
===========================

Each numbered item is one utterance: a merged .TextGrid + its .wav, side by side.
Open the .TextGrid in Praat with its matching .wav.

Folders
  batch_00, batch_01, … each hold {per_folder} items. Work through one folder at a
  time — they are independent bites.

File names
  <index>_<clip-id>_fp<N>.merged.TextGrid   (+ the same name with .wav)
  - <index>  : global running number, 0000 upward (unique across the whole set)
  - <clip-id>: source clip id (recording hash + time range) — do not edit
  - fp<N>    : how many filled-pause events this clip contains

Tiers (top to bottom)
  - alignment tiers (words / graphemes) — reference only, do not edit
  - FilledPauses                        — the automatically detected pauses
  - {mimic_tier}                        — YOUR tier. Every '{pre_label}' marks a
                                          detected pause; replace it with a label.
  - {instr_tier}                        — the label legend, for glance-down

Labels
  {instr_text}

The _manifest.tsv maps every <index> back to speaker, gender, duration and FP
count. Please do not rename or move files — the index is how everything is tracked.
"""

# --- RUN OPTIONS ------------------------------------------------------------
INVENTORY_SAMPLE = 30        # files to sample when reporting which tiers exist
DRY_RUN          = True      # True = plan only (no merge/convert/copy); flip to False to mint
# ----------------------------------------------------------------------------

# %% [markdown]
# ## Imports & helpers
#
# Minter helpers are carried over verbatim from `51`/`52`; the audio helpers are
# carried from `52`. The selection helpers (filled-pause filter, gender/speaker
# balancing, numbering) are new and live below them.

# %%
import csv
import glob
import json
import random
import itertools
from collections import defaultdict, Counter

import pandas as pd
from praatio import textgrid as tgio
from praatio.utilities.constants import Interval, Point, INTERVAL_TIER


# ---- minter helpers (verbatim from 51/52) ----------------------------------
def read_tg(path):
    return tgio.openTextgrid(str(path), includeEmptyIntervals=True,
                             reportingMode="silence", duplicateNamesMode="rename")


def save_tg(tg, path):
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True,
            reportingMode="silence")


def parse_tg_name(path):
    """'<base>.<layer>.TextGrid' -> (base, layer). Robust to dots inside base."""
    name = Path(path).name
    if not name.endswith(".TextGrid"):
        return None
    stem = name[:-len(".TextGrid")]
    if "." not in stem:
        return None
    base, layer = stem.rsplit(".", 1)
    return base, layer


def make_mimic_tier(src, new_name, pre_label):
    if src.tierType == INTERVAL_TIER:
        entries = [Interval(e.start, e.end,
                            pre_label if (e.label or "").strip() else "")
                   for e in src.entries]
        return tgio.IntervalTier(new_name, entries, src.minTimestamp, src.maxTimestamp)
    else:
        entries = [Point(e.time, pre_label if (e.label or "").strip() else "")
                   for e in src.entries]
        return tgio.PointTier(new_name, entries, src.minTimestamp, src.maxTimestamp)


def make_instruction_tier(name, text, t0, t1):
    return tgio.IntervalTier(name, [Interval(t0, t1, text)], t0, t1)


def normalize_recipe(recipe):
    norm_ = []
    for row in recipe:
        if not row:
            continue
        layer, rest = row[0], row[1:]
        if len(rest) == 0:
            spec = "all"
        elif len(rest) == 1:
            r = rest[0]
            if r == "all":
                spec = "all"
            elif isinstance(r, (list, tuple)):
                spec = list(r)
            else:
                spec = [r]
        else:
            spec = list(rest)
        norm_.append((layer, spec))
    return norm_


def required_layers():
    layers = [lay for lay, _ in normalize_recipe(MERGE_RECIPE)]
    if ADD_MIMIC_TIER and MIMIC_SOURCE_LAYER not in layers:
        layers.append(MIMIC_SOURCE_LAYER)
    return layers


def merge_one(layer_paths):
    """layer_paths: {layer: Path} for one base. Returns (merged_Textgrid, warnings)."""
    warnings = []
    need = required_layers()
    layer_tgs = {lay: read_tg(p) for lay, p in layer_paths.items() if lay in need}

    doms = [(tg.minTimestamp, tg.maxTimestamp) for tg in layer_tgs.values()]
    t0 = min(d[0] for d in doms)
    t1 = max(d[1] for d in doms)
    if len({(round(a, 3), round(b, 3)) for a, b in doms}) > 1:
        warnings.append(f"layer time-domains differ: {doms}")

    merged = tgio.Textgrid(minTimestamp=t0, maxTimestamp=t1)
    used = set()
    for layer, want in normalize_recipe(MERGE_RECIPE):
        if layer not in layer_tgs:
            warnings.append(f"layer '{layer}' missing; skipped")
            continue
        src = layer_tgs[layer]
        names = list(src.tierNames) if want == "all" else list(want)
        for nm in names:
            if nm not in src.tierNames:
                warnings.append(f"tier '{nm}' not in layer '{layer}'; skipped")
                continue
            out_name = nm if nm not in used else f"{layer}_{nm}"
            used.add(out_name)
            merged.addTier(src.getTier(nm).new(name=out_name))

    if ADD_MIMIC_TIER:
        src = layer_tgs.get(MIMIC_SOURCE_LAYER)
        if src is None:
            warnings.append(f"mimic source layer '{MIMIC_SOURCE_LAYER}' missing")
        else:
            tier_name = MIMIC_SOURCE_TIER or src.tierNames[0]
            merged.addTier(make_mimic_tier(src.getTier(tier_name),
                                           MIMIC_NEW_TIER, ANNOT_PRE_LABEL))

    if ADD_INSTRUCTION_TIER:
        merged.addTier(make_instruction_tier(INSTRUCTION_TIER_NAME,
                                             INSTRUCTION_TEXT, t0, t1))

    return merged, warnings


# ---- audio + path helpers (verbatim from 52) -------------------------------
def sanitise(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def norm(tok: str) -> str:
    return (tok or "").strip().lower()


def stem_from_audio(audio_rel: str):
    """'I27W0FVLD2I/I27W0FVLD2I_27158.08-27159.66.flac' -> (rec, stem=base)."""
    base = os.path.basename(audio_rel)
    stem = os.path.splitext(base)[0]
    rec = audio_rel.split("/")[0] if "/" in audio_rel else stem.split("_")[0]
    return rec, stem


def resolve_layers_dir(path: Path):
    """Descend into a doubled archive (…/X/X/) until the dir holding .TextGrid files."""
    if not path:
        return None
    cur = Path(path)
    for _ in range(4):
        subdirs = []
        try:
            with os.scandir(cur) as it:
                for e in it:
                    if e.is_file() and e.name.endswith(".TextGrid"):
                        return cur
                    if e.is_dir():
                        subdirs.append(e.name)
        except FileNotFoundError:
            return cur
        same = cur / cur.name
        if same.is_dir():
            cur = same
        elif len(subdirs) == 1:
            cur = cur / subdirs[0]
        else:
            break
    return cur


def layers_for_base(layers_dir: Path, base: str, need):
    """{layer: Path} for the needed layers of one base, via direct existence checks."""
    out = {}
    for lay in need:
        p = layers_dir / f"{base}.{lay}.TextGrid"
        if p.exists():
            out[lay] = p
    return out


def audio_roots(audio_dir: Path):
    """Shard subfolders (…part1 … part6) directly under AUDIO_DIR. Falls back to
    treating audio_dir itself as a root (plus siblings) if no child shards match."""
    if not audio_dir:
        return []
    audio_dir = str(audio_dir)
    child = [s for s in sorted(glob.glob(os.path.join(audio_dir, SHARD_GLOB)))
             if os.path.isdir(s)]
    if child:
        return child
    roots = [audio_dir]
    parent = os.path.dirname(audio_dir.rstrip("/"))
    for s in sorted(glob.glob(os.path.join(parent, SHARD_GLOB))):
        if os.path.isdir(s) and s not in roots:
            roots.append(s)
    return roots


def find_audio_source(roots, rec: str, stem: str):
    """Locate the source clip for one base across shards. Tries
    {shard}/{hash}/{stem}.{ext}, then {shard}/{hash}/wav/{stem}.{ext}."""
    for root in roots:
        for ext in AUDIO_SRC_EXTS:
            for cand in (os.path.join(root, rec, stem + ext),
                         os.path.join(root, rec, "wav", stem + ext)):
                if os.path.exists(cand):
                    return cand
    return None


_RESAMPLER_NOTE = {"printed": False}


def _resample(data, sr, target_sr):
    if not target_sr or sr == target_sr:
        return data, sr
    try:
        import soxr
        return soxr.resample(data, sr, target_sr), target_sr
    except Exception:
        pass
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(target_sr))
        return resample_poly(data, target_sr // g, sr // g, axis=0), target_sr
    except Exception:
        pass
    if not _RESAMPLER_NOTE["printed"]:
        print("  ! no resampler (soxr/scipy) found — keeping source sample rate")
        _RESAMPLER_NOTE["printed"] = True
    return data, sr


def make_wav(src, dst, target_sr):
    """Read src (flac/wav), optionally resample, write 16-bit PCM wav to dst. Returns sr."""
    import soundfile as sf
    data, sr = sf.read(src)
    data, sr = _resample(data, sr, target_sr)
    sf.write(dst, data, sr, subtype="PCM_16")
    return sr


# ---- selection helpers (new) -----------------------------------------------
def gender_of(info: dict):
    """Normalise the raw speaker_info gender to a quota key, or None if unknown."""
    return GENDER_MAP.get(norm(info.get(GENDER_FIELD)))


def speaker_of(info: dict):
    """Variety grouping key: Speaker_ID, falling back to Speaker_name."""
    return (info.get(SPEAKER_ID_FIELD)
            or info.get("Speaker_name")
            or "UNKNOWN")


MANIFEST_COLUMNS = [
    "annot_index", "batch_folder", "output_stem",
    "base", "utterance_id", "duration_s", "fp_count",
    "speaker_id", "speaker_name", "speaker_gender", "speaker_party", "party_status",
    "sentiment_3", "sentiment_6",
    "layers_found", "merged_tg", "audio_copied", "audio_src", "audio_rel", "minted", "text",
]


def find_eligibles(jsonl_path):
    """Stream the JSONL once; return (pool, stats). One record per eligible utterance
    (>= REQUIRE_MIN_FP filled pauses, passes duration filters, known gender). No file
    I/O — layer/audio resolution and minting happen later."""
    pool, stats = [], Counter()
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["lines"] += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue

            fps = entry.get("filled_pauses") or []
            if len(fps) < REQUIRE_MIN_FP:
                continue
            stats["has_min_fp"] += 1

            dur = entry.get("audio_length")
            try:
                dur_f = float(dur)
            except (TypeError, ValueError):
                dur_f = None
            if dur_f is not None:
                if MIN_DURATION_S is not None and dur_f < MIN_DURATION_S:
                    stats["skipped_short"] += 1
                    continue
                if MAX_DURATION_S is not None and dur_f > MAX_DURATION_S:
                    stats["skipped_long"] += 1
                    continue

            info = entry.get("speaker_info") or {}
            gender = gender_of(info)
            if gender not in GENDER_QUOTAS:
                stats["skipped_gender_unknown_or_offquota"] += 1
                continue
            spk = speaker_of(info)
            if not info.get(SPEAKER_ID_FIELD):
                stats["no_speaker_id_fell_back"] += 1

            rec, base = stem_from_audio(entry.get("audio", ""))
            sent = entry.get("sentiment") or {}
            pool.append({
                "base": base, "rec": rec, "gender": gender, "speaker": spk,
                "fp_count": len(fps),
                "row": {
                    "base": base, "utterance_id": entry.get("id", ""),
                    "duration_s": entry.get("audio_length", ""), "fp_count": len(fps),
                    "speaker_id": info.get(SPEAKER_ID_FIELD, ""),
                    "speaker_name": info.get("Speaker_name", ""),
                    "speaker_gender": gender,
                    "speaker_party": info.get("Speaker_party", ""),
                    "party_status": info.get("Party_status", ""),
                    "sentiment_3": sent.get("ParlaSent_3", ""),
                    "sentiment_6": sent.get("ParlaSent_6", ""),
                    "layers_found": "", "merged_tg": "", "audio_copied": "",
                    "audio_src": "", "audio_rel": entry.get("audio", ""),
                    "minted": False, "text": entry.get("text", ""),
                },
            })
            stats["eligible"] += 1
            stats[f"eligible::{gender}"] += 1
    return pool, stats


def select_balanced(pool, quotas, ideal_max_per_speaker, seed):
    """Per-gender round-robin over shuffled speakers (wide before deep). The cap is
    only an ideal — deeper passes run until the quota is met or the pool is exhausted.
    Returns (per_gender_ordered, report)."""
    rng = random.Random(seed)
    per_gender, report = {}, {}
    for g, quota in quotas.items():
        by_spk = defaultdict(list)
        for u in pool:
            if u["gender"] == g:
                by_spk[u["speaker"]].append(u)
        speakers = list(by_spk)
        rng.shuffle(speakers)
        for s in speakers:
            rng.shuffle(by_spk[s])

        chosen, taken = [], Counter()
        level = 0
        while len(chosen) < quota:
            level += 1
            progressed = False
            for s in speakers:
                if len(chosen) >= quota:
                    break
                if taken[s] < level and taken[s] < len(by_spk[s]):
                    chosen.append(by_spk[s][taken[s]])
                    taken[s] += 1
                    progressed = True
            if not progressed:
                break  # pool exhausted before quota

        per_gender[g] = chosen
        counts = [c for c in taken.values() if c > 0]
        report[g] = {
            "eligible": sum(len(v) for v in by_spk.values()),
            "speakers_in_pool": len(speakers),
            "selected": len(chosen), "quota": quota,
            "quota_met": len(chosen) >= quota,
            "speakers_used": len(counts),
            "max_per_speaker": max(counts) if counts else 0,
            "median_per_speaker": float(pd.Series(counts).median()) if counts else 0.0,
            "over_ideal_speakers": sum(1 for c in counts if c > ideal_max_per_speaker),
        }
    return per_gender, report


def order_and_number(per_gender, interleave, per_folder, folder_prefix,
                     folder_width, index_width):
    """Flatten per-gender lists into one numbered, foldered sequence. Interleaving
    alternates genders so the set is order-blind. Stamps annot_index / batch_folder /
    output_stem onto each record. Returns the ordered list."""
    if interleave:
        lists = [per_gender[g] for g in per_gender]
        ordered = [x for x in itertools.chain.from_iterable(itertools.zip_longest(*lists))
                   if x is not None]
    else:
        ordered = [x for g in per_gender for x in per_gender[g]]
    for i, u in enumerate(ordered):
        u["annot_index"] = i
        u["batch_folder"] = f"{folder_prefix}{i // per_folder:0{folder_width}d}"
        u["output_stem"] = f"{i:0{index_width}d}_{u['base']}_fp{u['fp_count']}"
    return ordered


# %% [markdown]
# ## Step A — Scan the JSONL, balance, number
#
# Streams the master JSONL for filled-pause utterances, balances to the per-gender
# quota across distinct speakers, then numbers + folders the result. Prints the
# balance report so you can eyeball speaker variety before minting.

# %%
if not JSONL_PATH.exists():
    print(f"!! JSONL not found: {JSONL_PATH}\n   Set JSONL_PATH (env var or settings cell).")
    POOL, POOL_STATS, SELECTED, BAL_REPORT = [], Counter(), [], {}
else:
    print(f"Streaming {JSONL_PATH} …")
    POOL, POOL_STATS = find_eligibles(JSONL_PATH)
    print(f"  lines scanned        : {POOL_STATS['lines']:,}")
    print(f"  >= {REQUIRE_MIN_FP} filled pause(s) : {POOL_STATS['has_min_fp']:,}")
    if POOL_STATS.get("skipped_short") or POOL_STATS.get("skipped_long"):
        print(f"  duration-filtered    : {POOL_STATS.get('skipped_short', 0):,} short, "
              f"{POOL_STATS.get('skipped_long', 0):,} long")
    if POOL_STATS.get("skipped_gender_unknown_or_offquota"):
        print(f"  gender unknown/off-quota: {POOL_STATS['skipped_gender_unknown_or_offquota']:,}")
    if POOL_STATS.get("no_speaker_id_fell_back"):
        print(f"  ! {POOL_STATS['no_speaker_id_fell_back']:,} had no {SPEAKER_ID_FIELD}; "
              f"fell back to Speaker_name")
    print(f"  eligible pool        : {POOL_STATS['eligible']:,}  "
          + "  ".join(f"{g}={POOL_STATS.get(f'eligible::{g}', 0):,}" for g in GENDER_QUOTAS))

    PER_GENDER, BAL_REPORT = select_balanced(POOL, GENDER_QUOTAS, MAX_PER_SPEAKER, SEED)
    SELECTED = order_and_number(PER_GENDER, INTERLEAVE_GENDERS, PER_FOLDER,
                                FOLDER_PREFIX, FOLDER_WIDTH, INDEX_WIDTH)

    print("\n  balance report:")
    for g, r in BAL_REPORT.items():
        flag = "" if r["quota_met"] else "  << QUOTA NOT MET (pool too small)"
        print(f"    {g}: {r['selected']:,}/{r['quota']:,} from {r['speakers_used']:,} speakers"
              f"  (max {r['max_per_speaker']}/speaker, median {r['median_per_speaker']:g}, "
              f"{r['over_ideal_speakers']} over ideal {MAX_PER_SPEAKER}){flag}")
    n_batches = (len(SELECTED) + PER_FOLDER - 1) // PER_FOLDER
    print(f"\n  total selected: {len(SELECTED):,}  ->  {n_batches} folder(s) of up to {PER_FOLDER}")
    if SELECTED:
        print(f"  first: {SELECTED[0]['batch_folder']}/{SELECTED[0]['output_stem']}")
        print(f"  last : {SELECTED[-1]['batch_folder']}/{SELECTED[-1]['output_stem']}")

# %% [markdown]
# ## Step 1 — Inventory: what tiers are in stock?
#
# Samples up to `INVENTORY_SAMPLE` files per layer to show tier names/types, so you
# can fill in `MERGE_RECIPE` / `MIMIC_SOURCE_TIER`. (Cheap — no full folder listing.)

# %%
LAYERS_DIR = resolve_layers_dir(IN_DIR)
print(f"Layer dir resolved to: {LAYERS_DIR}")

# The flat archive can hold millions of files; never list it whole. Walk os.scandir
# lazily and stop after SCAN_CAP directory entries (the per-layer sample fills well
# before then).
SCAN_CAP = INVENTORY_SAMPLE * 100
seen = defaultdict(lambda: defaultdict(lambda: {"type": set(), "counts": []}))
layer_file_counts = Counter()
scanned = bad_files = 0
try:
    with os.scandir(LAYERS_DIR) as it:
        for e in it:
            if scanned >= SCAN_CAP:
                break
            if not (e.is_file() and e.name.endswith(".TextGrid")):
                continue
            scanned += 1
            parsed = parse_tg_name(e.name)
            if not parsed:
                continue
            base, layer = parsed
            if layer_file_counts[layer] >= INVENTORY_SAMPLE:
                continue
            try:
                tg = read_tg(e.path)
            except Exception:
                bad_files += 1
                continue
            layer_file_counts[layer] += 1
            for name in tg.tierNames:
                t = tg.getTier(name)
                info = seen[layer][name]
                info["type"].add(t.tierType)
                info["counts"].append(len(t.entries))
except FileNotFoundError:
    print(f"  !! layer dir not found: {LAYERS_DIR}")
print(f"  scanned {scanned} dir entries (cap {SCAN_CAP})" +
      (f", skipped {bad_files} unreadable" if bad_files else ""))

rows = []
for layer in sorted(seen):
    for tname, info in seen[layer].items():
        cnts = info["counts"]
        rows.append({"layer": layer, "tier": tname,
                     "type": "/".join(sorted(info["type"])),
                     "n_sampled": len(cnts),
                     "median_intervals": int(pd.Series(cnts).median()) if cnts else 0})
inventory = pd.DataFrame(rows)
print("Layers sampled:", dict(layer_file_counts))
if not inventory.empty:
    print(inventory.to_string(index=False))

# %% [markdown]
# ## Step 2 — Preview the recipe on one selected clip

# %%
need = required_layers()
preview = next((u for u in SELECTED
                if all(l in layers_for_base(LAYERS_DIR, u["base"], need) for l in need)), None)
if preview is None:
    print(f"!! No selected clip has all required layers {need}. "
          f"Check MERGE_RECIPE / IN_DIR (selected: {len(SELECTED)}).")
else:
    try:
        merged, warns = merge_one(layers_for_base(LAYERS_DIR, preview["base"], need))
    except Exception as ex:
        merged, warns = None, [f"merge failed: {ex}"]
    if merged is None:
        print(f"Preview {preview['base']} failed to merge: {warns}")
    else:
        print(f"Preview clip : {preview['batch_folder']}/{preview['output_stem']}")
        print(f"Speaker/gender: {preview['speaker']} / {preview['gender']}  (fp={preview['fp_count']})")
        print("Output tiers (in order):")
        for i, name in enumerate(merged.tierNames, 1):
            t = merged.getTier(name)
            print(f"  {i}. {name}  [{t.tierType}, {len(t.entries)} intervals]")
        if ADD_MIMIC_TIER and MIMIC_NEW_TIER in merged.tierNames:
            ann = merged.getTier(MIMIC_NEW_TIER)
            pre = sum(1 for e in ann.entries if (e.label or "").strip())
            print(f"\n  '{MIMIC_NEW_TIER}': {pre} interval(s) pre-marked '{ANNOT_PRE_LABEL}'")
        if warns:
            print("  warnings:", warns)

# %% [markdown]
# ## Step 3 — Mint into numbered batch folders (+ copy wav)
#
# **`DRY_RUN = True`** plans only: it checks each selected clip's layers + audio on
# disk, computes the destination paths, and writes the manifest of what it *would*
# mint — but does not merge, convert, or copy. Set `DRY_RUN = False` to actually
# write the TextGrids and wavs. Re-running is idempotent (existing files are kept).

# %%
OUT_DIR.mkdir(parents=True, exist_ok=True)
roots = audio_roots(AUDIO_DIR) if COPY_AUDIO else []
if roots:
    print(f"Audio shards ({len(roots)}): {roots[0]}"
          + (f" (+{len(roots)-1} more)" if len(roots) > 1 else ""))
print(("DRY RUN — planning only, no files minted. "
       "Set DRY_RUN=False to write.") if DRY_RUN else "MINTING for real.")

need = required_layers()
tally = Counter()
warn_summary = defaultdict(list)
sample_paths = []

for u in SELECTED:
    lp = layers_for_base(LAYERS_DIR, u["base"], need)
    missing = [l for l in need if l not in lp]
    audio_src = find_audio_source(roots, u["rec"], u["base"]) if roots else None
    bdir = OUT_DIR / u["batch_folder"]
    tg_path = bdir / f"{u['output_stem']}.merged.TextGrid"
    wav_path = bdir / f"{u['output_stem']}{OUTPUT_AUDIO_EXT}"

    row = u["row"]
    row["annot_index"] = u["annot_index"]
    row["batch_folder"] = u["batch_folder"]
    row["output_stem"] = u["output_stem"]
    row["layers_found"] = ";".join(sorted(lp))
    row["audio_src"] = audio_src or ""
    row["merged_tg"] = str(tg_path)
    row["audio_copied"] = wav_path.name if audio_src else ""
    row["minted"] = False

    if missing:
        tally["missing_layers"] += 1
        warn_summary["missing:" + "+".join(missing)].append(u["base"])
        continue
    tally["layers_ok"] += 1
    tally["audio_found" if audio_src else "audio_missing"] += 1
    if len(sample_paths) < 5:
        sample_paths.append(f"{u['batch_folder']}/{u['output_stem']}"
                            + ("  (+wav)" if audio_src else "  (NO AUDIO)"))

    if DRY_RUN:
        tally["would_mint"] += 1
        continue

    try:
        merged, warns = merge_one(lp)
    except Exception as ex:
        warn_summary["merge_error"].append(f"{u['base']}: {ex}")
        continue
    for w in warns:
        warn_summary[w.split(":")[0]].append(u["base"])

    bdir.mkdir(parents=True, exist_ok=True)
    if not tg_path.exists():
        save_tg(merged, tg_path)
    if audio_src:
        try:
            if not wav_path.exists():
                make_wav(audio_src, wav_path, TARGET_SR)
        except Exception as ex:
            warn_summary["audio_convert_error"].append(f"{u['base']}: {ex}")
            row["audio_copied"] = ""
    row["minted"] = True
    tally["minted"] += 1

print(f"\nLayers OK    : {tally['layers_ok']:,}    missing layers: {tally['missing_layers']:,}")
if roots:
    print(f"Audio found  : {tally['audio_found']:,}    missing: {tally['audio_missing']:,}"
          + ("  (none matched — check AUDIO_DIR/shard/ext)" if tally['audio_found'] == 0 else ""))
if DRY_RUN:
    print(f"Would mint   : {tally['would_mint']:,}  ->  {OUT_DIR}")
    if sample_paths:
        print("Sample destinations:")
        for s in sample_paths:
            print(f"  {s}")
else:
    print(f"Minted       : {tally['minted']:,}  ->  {OUT_DIR}")
if warn_summary:
    print("Warning summary (type -> #clips):")
    for k, v in warn_summary.items():
        print(f"  {k}: {len(v)}   e.g. {v[:2]}")

# manifest — always written (it IS the plan); merged_tg/audio_copied paths are
# populated, `minted` marks whether files were actually written this run.
tsv_path = OUT_DIR / "_manifest.tsv"
with open(tsv_path, "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, delimiter="\t", extrasaction="ignore")
    w.writeheader()
    for u in SELECTED:
        w.writerow(u["row"])
with open(OUT_DIR / "_manifest.jsonl", "w", encoding="utf-8") as fh:
    for u in SELECTED:
        fh.write(json.dumps(u["row"], ensure_ascii=False) + "\n")
print(f"Manifest     : {tsv_path}  ({len(SELECTED):,} rows)")

# README — only on a real mint (it's part of the deliverable, not the plan).
if WRITE_README and not DRY_RUN:
    (OUT_DIR / "_README.txt").write_text(
        README_TEXT.format(per_folder=PER_FOLDER, mimic_tier=MIMIC_NEW_TIER,
                           pre_label=ANNOT_PRE_LABEL, instr_tier=INSTRUCTION_TIER_NAME,
                           instr_text=INSTRUCTION_TEXT),
        encoding="utf-8")
    print(f"README       : {OUT_DIR / '_README.txt'}")

# %% [markdown]
# ## Step 4 — Verify outputs + final balance
#
# The balance summary is derived from the selection, so it prints in dry-run too.
# On a real mint it also spot-checks that files are readable and wav-paired.

# %%
if BAL_REPORT:
    print("Final balance (from selection):")
    for g, r in BAL_REPORT.items():
        print(f"  {g}: {r['selected']:,} clips, {r['speakers_used']:,} distinct speakers "
              f"(max {r['max_per_speaker']}/speaker)")
    per_folder_counts = Counter(u["batch_folder"] for u in SELECTED)
    print("Per-folder counts:",
          ", ".join(f"{k}={v}" for k, v in sorted(per_folder_counts.items())))

if not DRY_RUN:
    n = 0
    for p in sorted(OUT_DIR.glob("*/*.merged.TextGrid")):
        tg = read_tg(p)
        rel = p.relative_to(OUT_DIR)
        wav = (p.parent / p.name.replace(".merged.TextGrid", OUTPUT_AUDIO_EXT))
        tag = "+wav" if wav.exists() else "no-wav"
        print(f"{rel}: {list(tg.tierNames)}  [{tag}]")
        with open(p, encoding="utf-8") as f:
            f.read()
        n += 1
        if n >= 5:
            break
    print(f"\nVerified {n} merged file(s) readable, UTF-8." if n else "No merged files to verify.")
else:
    print("\n(dry run — no files on disk to verify; flip DRY_RUN=False to mint)")
