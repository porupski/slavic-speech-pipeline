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
# # 54 — Nejc's annotated Slovenian TextGrids → `word_frame` JSONL
#
# Turns Nejc's ~81 annotated `.TextGrid` files into one canonical JSONL that
# `41_train_frame_classification` can train on. One JSONL row per multisyllabic
# word that carries a primary-stress mark. The frame sequence is 50 Hz binary
# (`primary_stress = 1` where the stressed vowel lives, else `0`) over the word
# span. The trainer slices the word from the utterance WAV in memory at load
# time via `start_t` / `end_t` — no per-word WAV files are written.
#
# ### Inputs
# - TextGrids: `data/nejc_slo_stress/final_textgrids/*.TextGrid` (UTF-16 BE).
# - Utterance WAVs: `data/nejc_slo_stress/final_wavs/*.wav`. WAV names are the
#   same as TG names, but decimals use `.` in place of `_` (e.g. TG
#   `..._82_407-218_235.TextGrid` ↔ WAV `..._82.407-218.235.wav`).
#
# ### Output
# - `data/processed_jsonl/si_primary_stress_word_frame.jsonl` — one row per
#   stressed word. Schema mirrors `11c_prep_parlaspeech`'s `word_frame`.
#
# ### Extra pieces
# - Words with more than one `"1"` mark are **kept** and tagged
#   `metadata.multistress = True`. All `"1"` spans become positive frames in
#   the same 0/1 label vector. A future primary-stress-only target can filter
#   these out; a future multi-stress target can use them as-is. (The stress
#   tier in this dataset uses only `"0"` / `"1"` — there is no secondary-stress
#   class to separate the peaks into distinct labels.)
# - Any `"1"` mark that falls outside every `wordAlign` interval is dropped
#   and logged.
# - Splits come from `SPLIT_OVERRIDE_CSV` (built by
#   `data/nejc_slo_stress/build_split_csv.py` from Nejc's Excel). Anything the
#   CSV does not cover falls back to a speaker-grouped hash (80/10/10) — no
#   leakage, deterministic.

# %% [markdown]
# ## ▼ USER SETTINGS — edit this cell, then run the rest

# %%
from pathlib import Path

# --- WHERE ------------------------------------------------------------------
TG_DIR       = "data/nejc_slo_stress/final_textgrids"
WAV_DIR      = "data/nejc_slo_stress/final_wavs"
OUTPUT_JSONL = "data/processed_jsonl/si_primary_stress_word_frame.jsonl"

# --- SPLITS -----------------------------------------------------------------
# Placeholder: deterministic hash grouped by speaker until Nejc's Excel arrives.
SPLIT_RATIOS       = (0.8, 0.1, 0.1)
SPLIT_GROUP_KEY    = "speaker"
SPLIT_SEED         = "nejc-slo-stress-v1"
SPLIT_OVERRIDE_CSV = "data/nejc_slo_stress/split.csv"   # None to fall back to hash

# --- FRAME RATE -------------------------------------------------------------
FRAME_RATE_HZ = 50   # hard-locked project-wide; matches chapter 4.

# --- DATASET TAG ------------------------------------------------------------
DATASET_TAG = "nejc-slo-stress"

# --- RUN OPTIONS ------------------------------------------------------------
PROCESS_LIMIT = None    # cap TGs processed (None = all 81)
MATCH_TOLERANCE_S = 1e-3   # tolerance for stress ↔ word span alignment
# ----------------------------------------------------------------------------

# %% [markdown]
# ## Imports & helpers

# %%
import json
import re
import sys
import tempfile
import wave
from collections import Counter, defaultdict

from praatio import textgrid as tgio

HERE = Path.cwd()
if HERE.name != "5_tg_minter":
    cand = HERE / "5_tg_minter"
    if cand.exists():
        HERE = cand
sys.path.insert(0, str(HERE.parent / "1_data_prep"))

import utils_dataprep as udp

PROJECT_ROOT = udp.PROJECT_ROOT
print(f"PROJECT_ROOT = {PROJECT_ROOT}")


# %% [markdown]
# ### Filename parser
#
# TG name pattern: `{source_id}_{speaker}_{start_int}_{start_frac}-{end_int}_{end_frac}.TextGrid`.
# Decimals use `_` (filesystem-safe). WAV sibling uses `.` in the same slots.
# For GOS the `source_id` itself carries an internal `_` (e.g. `GosVL08_droge`);
# the rule "everything before the last `_` in the prefix is `source_id`, the
# last piece is `speaker`" covers both corpora.

# %%
_TG_NAME_RE = re.compile(
    r"^(?P<pre>.+)_(?P<s_int>\d+)_(?P<s_frac>\d+)-(?P<e_int>\d+)_(?P<e_frac>\d+)\.TextGrid$"
)


def parse_tg_filename(name: str) -> dict | None:
    m = _TG_NAME_RE.match(name)
    if not m:
        return None
    pre = m["pre"]
    parts = pre.rsplit("_", 1)
    source_id, speaker = (parts[0], parts[1]) if len(parts) == 2 else (pre, "")
    start = float(f"{m['s_int']}.{m['s_frac']}")
    end = float(f"{m['e_int']}.{m['e_frac']}")
    wav_stem = f"{pre}_{m['s_int']}.{m['s_frac']}-{m['e_int']}.{m['e_frac']}"
    if source_id.startswith("GosVL"):
        corpus = "GOS"
    elif source_id.startswith("Artur-"):
        corpus = "ROG-Art"
    else:
        corpus = "other"
    return {
        "source_id": source_id,
        "speaker":   speaker,
        "start":     start,
        "end":       end,
        "wav_stem":  wav_stem,
        "corpus":    corpus,
    }


# %% [markdown]
# ### Encoding-aware TG loader
#
# Nejc's TextGrids are UTF-16 BE with BOM. praatio's `openTextgrid` expects a
# path, so we decode once, write a UTF-8 sidecar to a tempfile, and load that.

# %%
def read_tg_any_encoding(path: Path):
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".TextGrid",
                                     encoding="utf-8", delete=False) as f:
        f.write(text)
        tmp = Path(f.name)
    try:
        return tgio.openTextgrid(
            str(tmp),
            includeEmptyIntervals=True,
            reportingMode="silence",
            duplicateNamesMode="rename",
        )
    finally:
        tmp.unlink(missing_ok=True)


# %% [markdown]
# ### Tier extraction
#
# `wordAlign` gives the word intervals (non-empty text only). `stress` is sparse
# — non-empty intervals contain literally `"0"` or `"1"`, each matching a vowel
# in `phoneAlign`. Only `"1"` marks feed `primary_stress`. Absence of any `"1"`
# for a word means the word does not become a training row.

# %%
def _find_word_for_stress(words: list[dict], s: float, e: float,
                          tol: float) -> int | None:
    """Return the words_align index whose span contains [s, e]. Prefers full
    containment; falls back to midpoint-inside-word."""
    for i, w in enumerate(words):
        if w["time_s"] - tol <= s and e <= w["time_e"] + tol:
            return i
    mid = 0.5 * (s + e)
    for i, w in enumerate(words):
        if w["time_s"] - tol <= mid <= w["time_e"] + tol:
            return i
    return None


def extract_words_and_stress(tg, *, tol: float = MATCH_TOLERANCE_S
                             ) -> tuple[list[dict], list[dict], list[tuple]]:
    """Return (words_align, primary_stress, notes).
    `primary_stress` keeps **every** valid `"1"` mark (multi-stress words too).
    `notes` is a list of `(kind, *args)` tuples for the per-file report:
      - `stress_outside_word` — mark dropped (does not appear in `primary_stress`)
      - `multiple_primary_per_word` — mark kept; word will be tagged
        `metadata.multistress = True` in the emitted row.
    """
    if "wordAlign" not in tg.tierNames:
        raise ValueError("TG has no 'wordAlign' tier")
    if "stress" not in tg.tierNames:
        raise ValueError("TG has no 'stress' tier")

    words: list[dict] = []
    for e in tg.getTier("wordAlign").entries:
        text = (e.label or "").strip()
        if not text:
            continue
        words.append({
            "text":   text,
            "time_s": float(e.start),
            "time_e": float(e.end),
        })

    ps: list[dict] = []
    notes: list[tuple] = []
    for e in tg.getTier("stress").entries:
        lab = (e.label or "").strip()
        if lab != "1":
            continue
        s, ee = float(e.start), float(e.end)
        idx = _find_word_for_stress(words, s, ee, tol)
        if idx is None:
            notes.append(("stress_outside_word", round(s, 3), round(ee, 3)))
            continue
        ps.append({"words_align_idx": idx, "raw": [s, ee]})

    # Note (do not drop) words that carry >1 "1" mark. build_word_frame_rows
    # aggregates all marks for that word into one row and tags it multistress.
    idx_count = Counter(p["words_align_idx"] for p in ps)
    for i, n in sorted(idx_count.items()):
        if n > 1:
            notes.append(("multiple_primary_per_word", i, words[i]["text"], n))
    return words, ps, notes


# %% [markdown]
# ### `build_word_frame_rows`
#
# **Intentionally copied from `11c_prep_parlaspeech.build_word_frame_rows`** —
# project rule is no cross-notebook imports. Keep this in lock-step if `11c`
# ever changes shape.

# %%
def build_word_frame_rows(r, hz):
    """Expand one utterance record → one row per stress-annotated word.
    `audio_path` stays the utterance WAV; `start_t`/`end_t` mark the word span
    (41 slices in memory). The 0/1 label sequence covers the word span; each
    `"1"` interval becomes a run of positive frames. A word with multiple `"1"`
    marks gets ONE row with multiple positive runs and `metadata.multistress =
    True`, so a primary-only target can filter it out and a future multi-stress
    target can use it as-is.

    Deviates from `11c`'s `build_word_frame_rows`: 11c emits one row per
    ParlaSpeech `primary_stress` entry (never >1 per word); here we group by
    word so multi-stress words come out as a single tagged row."""
    md = r["metadata"]
    wa = md.get("words_align")
    ps_list = md.get("primary_stress")
    if not wa or not ps_list:
        return []
    # Group all stress marks by their target word.
    by_word: dict[int, list] = {}
    for ps in ps_list:
        wai = ps.get("words_align_idx")
        if wai is None or not (0 <= wai < len(wa)):
            continue
        by_word.setdefault(wai, []).append(ps)
    base = {k: r[k] for k in ("dataset", "file_id", "speaker", "split")}
    rows = []
    for wai in sorted(by_word):
        marks = by_word[wai]
        word = wa[wai]
        w_s, w_e = float(word["time_s"]), float(word["time_e"])
        word_dur = round(w_e - w_s, 3)
        n = round(word_dur * hz)
        if n <= 0:
            continue
        labels = [0] * n
        for ps in marks:
            raw = ps.get("raw")
            if (isinstance(raw, (list, tuple)) and len(raw) == 2
                    and all(isinstance(x, (int, float)) for x in raw)):
                s = max(0, round((float(raw[0]) - w_s) * hz))
                e = min(n, round((float(raw[1]) - w_s) * hz))
                for i in range(s, e):
                    labels[i] = 1
        rows.append({
            **base,
            "instance_id":  f"{r['instance_id']}#w{wai}",
            "audio_path":   r["audio_path"],
            "text":         word.get("text", ""),
            "start_t":      round(w_s, 3),
            "end_t":        round(w_e, 3),
            "frame_rate_hz": hz,
            "labels":       {"primary_stress": labels},
            "metadata": {
                "audio_length":     word_dur,
                "lang":             md.get("lang"),
                "source_utterance": r["instance_id"],
                "words_align_idx":  wai,
                "multistress":      len(marks) > 1,
                "n_stress_marks":   len(marks),
            },
        })
    return rows


# %% [markdown]
# ### WAV duration (stdlib, no soundfile dependency here)

# %%
def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# %% [markdown]
# ## Step 1 — Enumerate TGs, pair with WAVs, inventory

# %%
tg_root = udp.from_project_relative(TG_DIR)
wav_root = udp.from_project_relative(WAV_DIR)
tg_paths = sorted(tg_root.glob("*.TextGrid"))
if PROCESS_LIMIT is not None:
    tg_paths = tg_paths[:PROCESS_LIMIT]

print(f"found {len(tg_paths)} TextGrids under {TG_DIR}")

parsed_names: list[dict] = []
bad_names: list[str] = []
wav_missing: list[str] = []
corpus_counter: Counter = Counter()

for p in tg_paths:
    meta = parse_tg_filename(p.name)
    if meta is None:
        bad_names.append(p.name)
        continue
    corpus_counter[meta["corpus"]] += 1
    wav_path = wav_root / f"{meta['wav_stem']}.wav"
    if not wav_path.exists():
        wav_missing.append(meta["wav_stem"])
        continue
    meta["tg_path"] = p
    meta["wav_path"] = wav_path
    parsed_names.append(meta)

print(f"parsed OK: {len(parsed_names)} / {len(tg_paths)}")
print(f"corpus mix: {dict(corpus_counter)}")
if bad_names:
    udp.banner("filename parse failures", char="-")
    for n in bad_names:
        print(f"  {n}")
if wav_missing:
    udp.banner("missing WAV siblings", char="-")
    for s in wav_missing:
        print(f"  {s}.wav")

# %% [markdown]
# ## Step 2 — Parse tiers, build utterance-level records
#
# One record per TG. Metadata carries `words_align` and `primary_stress` for
# `build_word_frame_rows` to expand in Step 4.

# %%
utterance_records: list[dict] = []
note_totals: Counter = Counter()
per_file_notes: list[dict] = []
dur_mismatch_warns: list[str] = []

for meta in parsed_names:
    try:
        tg = read_tg_any_encoding(meta["tg_path"])
    except Exception as e:
        print(f"⚠️  TG parse failed for {meta['tg_path'].name}: {e}")
        continue

    try:
        words, ps, notes = extract_words_and_stress(tg)
    except ValueError as e:
        print(f"⚠️  tier missing in {meta['tg_path'].name}: {e}")
        continue

    for kind, *args in notes:
        note_totals[kind] += 1
    if notes:
        per_file_notes.append({
            "file":  meta["tg_path"].name,
            "notes": notes,
        })

    tg_xmax = float(tg.maxTimestamp)
    wav_dur = wav_duration_s(meta["wav_path"])
    filename_dur = meta["end"] - meta["start"]
    # TG's own clock is utterance-relative (xmin=0); compare xmax to WAV
    # duration and to the filename-derived span.
    if abs(tg_xmax - wav_dur) > 0.05:
        dur_mismatch_warns.append(
            f"{meta['tg_path'].name}: TG xmax={tg_xmax:.3f}s vs WAV={wav_dur:.3f}s")
    if abs(tg_xmax - filename_dur) > 0.05:
        dur_mismatch_warns.append(
            f"{meta['tg_path'].name}: TG xmax={tg_xmax:.3f}s vs filename span={filename_dur:.3f}s")

    audio_path_rel = f"{WAV_DIR}/{meta['wav_stem']}.wav"
    instance_id = f"{DATASET_TAG}__{meta['wav_stem']}"

    utterance_records.append({
        "instance_id": instance_id,
        "dataset":     DATASET_TAG,
        "file_id":     meta["source_id"],
        "audio_path":  audio_path_rel,
        "speaker":     meta["speaker"],
        "metadata": {
            "audio_length": round(wav_dur, 3),
            "lang":         "sl",
            "corpus":       meta["corpus"],
            "session_start_t": meta["start"],   # position within the session
            "session_end_t":   meta["end"],
            "words_align":     words,
            "primary_stress":  ps,
        },
    })

print(f"utterance records: {len(utterance_records)}")
print(f"note totals: {dict(note_totals)}")
print("  (multiple_primary_per_word = kept, tagged metadata.multistress=True; "
      "stress_outside_word = dropped)")
if dur_mismatch_warns:
    udp.banner("duration mismatches (>50 ms)", char="-")
    for w in dur_mismatch_warns[:20]:
        print(f"  {w}")
    if len(dur_mismatch_warns) > 20:
        print(f"  ... and {len(dur_mismatch_warns) - 20} more")

# Per-corpus word / stress-mark tallies for the inventory.
n_words_per_corpus: dict[str, int] = defaultdict(int)
n_stress_per_corpus: dict[str, int] = defaultdict(int)
for r in utterance_records:
    c = r["metadata"]["corpus"]
    n_words_per_corpus[c] += len(r["metadata"]["words_align"])
    n_stress_per_corpus[c] += len(r["metadata"]["primary_stress"])
print("per-corpus tallies:")
for c in sorted(set(n_words_per_corpus) | set(n_stress_per_corpus)):
    print(f"  {c:>10}  words={n_words_per_corpus.get(c, 0):>6}  "
          f"primary_stress rows={n_stress_per_corpus.get(c, 0):>6}")

# %% [markdown]
# ## Step 3 — Assign splits
#
# Splits come from `SPLIT_OVERRIDE_CSV` (canonical: built by
# `data/nejc_slo_stress/build_split_csv.py` from Nejc's Excel). CSV shape:
# columns `file_id` (or `wav_stem`) + `split` where split ∈ {train, dev, test}.
# Any TG whose `file_id` is not covered by the CSV falls back to a speaker-
# grouped hash so nothing gets silently dropped.

# %%
import csv as _csv


def _load_split_override(csv_path: str | None) -> dict[str, str]:
    if not csv_path:
        return {}
    p = udp.from_project_relative(csv_path)
    if not p.exists():
        print(f"⚠️  split override {csv_path} not found — using hash splits")
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    key_col = "file_id" if "file_id" in fields else "wav_stem"
    if key_col not in fields or "split" not in fields:
        raise ValueError(
            f"split override CSV must have a '{key_col}' column and a 'split' "
            f"column; got {fields}"
        )
    mapping = {r[key_col].strip(): r["split"].strip().lower() for r in rows
               if r.get(key_col)}
    valid = {"train", "dev", "test"}
    bad = {k: v for k, v in mapping.items() if v not in valid}
    if bad:
        raise ValueError(f"invalid split values in override: {bad}")
    return mapping


split_override = _load_split_override(SPLIT_OVERRIDE_CSV)
if split_override:
    print(f"split override: {len(split_override)} rows")
    for r in utterance_records:
        key_file_id = r["file_id"]
        key_stem = Path(r["audio_path"]).stem
        if key_file_id in split_override:
            r["split"] = split_override[key_file_id]
        elif key_stem in split_override:
            r["split"] = split_override[key_stem]

# Fill in hash-based split for anything the override did not cover.
udp.assign_splits(
    utterance_records,
    ratios=SPLIT_RATIOS,
    group_key=SPLIT_GROUP_KEY,
    seed=SPLIT_SEED,
    overwrite=False,
)
print(f"utterance splits: {udp.split_summary(utterance_records)}")

# %% [markdown]
# ## Step 4 — Expand to `word_frame` rows

# %%
rows: list[dict] = []
for r in utterance_records:
    rows.extend(build_word_frame_rows(r, FRAME_RATE_HZ))

print(f"word_frame rows: {len(rows)}")
row_split_counts = Counter(r["split"] for r in rows)
print(f"row split counts: {dict(row_split_counts)}")

n_multi = sum(1 for r in rows if r["metadata"].get("multistress"))
print(f"multistress rows: {n_multi} / {len(rows)} "
      f"({100.0 * n_multi / len(rows) if rows else 0.0:.2f}%)  "
      "— tagged metadata.multistress=True; a primary-only target filters these out")

# Positive-frame ratio for a quick sanity check.
tot_frames = sum(len(r["labels"]["primary_stress"]) for r in rows)
tot_pos = sum(sum(r["labels"]["primary_stress"]) for r in rows)
pos_pct = 100.0 * tot_pos / tot_frames if tot_frames else 0.0
print(f"positive frames: {tot_pos}/{tot_frames} ({pos_pct:.2f}%)")

# %% [markdown]
# ## Step 5 — Write JSONL + verify
#
# `validate_jsonl` is the generic instance validator. It accepts optional
# `frame_rate_hz` and does check the sequence-labels invariant, so it fits
# these word_frame rows.

# %%
n_written = udp.write_jsonl(rows, OUTPUT_JSONL)
print(f"wrote {n_written} rows → {OUTPUT_JSONL}")

reread = udp.read_jsonl(OUTPUT_JSONL)
n_tot, n_valid, errs = udp.validate_jsonl(reread)
tag = "✅" if not errs else "⚠️ "
print(f"{tag} validated: {n_valid}/{n_tot}")
for e in errs[:5]:
    print(f"   {e}")

# Peek at one record.
if reread:
    udp.banner("first record", char="-")
    print(json.dumps(reread[0], indent=2, ensure_ascii=False))

# %% [markdown]
# ## Step 6 — Per-file notes
#
# Two kinds of note appear here:
# - `multiple_primary_per_word` — word carries >1 `"1"` mark. **Kept** in the
#   JSONL and tagged `metadata.multistress = True`. Compare with Nejc's
#   `slo_stress_summary.jsonl` (`multiple_primary_stresses` counts should match
#   roughly).
# - `stress_outside_word` — `"1"` mark that does not fall inside any word span.
#   **Dropped** — no row emitted.

# %%
if per_file_notes:
    udp.banner("per-file notes", char="-")
    for entry in per_file_notes[:20]:
        print(f"  {entry['file']}")
        for kind, *args in entry["notes"][:10]:
            print(f"     {kind}: {args}")
    if len(per_file_notes) > 20:
        print(f"  ... and {len(per_file_notes) - 20} more files")

# %% [markdown]
# ## Next
#
# - Train: `4_frame_models/41_train_frame_classification.ipynb` — set
#   `Config.target = "si_primary_stress_frames"`. The target entry is
#   already registered.
