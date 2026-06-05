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
# # Prep ParlaSpeech — chapter 1 (variant c)
#
# Parse ParlaSpeech JSONL(s) into canonical pipeline JSONLs, convert per-utterance
# FLACs to 16 kHz mono WAVs, and emit **one file per instance-shape** (not per task
# — multiple label keys live in one file).
#
# A single run processes **every** ParlaSpeech-{LANG} found under `data/unpacked/`
# (or one pinned `cfg.lang`). Each section below defines a function; section 12
# loops over the languages and runs them.
#
# **Recipes emitted here** (both whole-utterance, no slicing):
# - `utterance_instance` — scalar labels: `speaker_gender`, `filled_pause_present`,
#   `filled_pause_count`, `sentiment_logit`, `sentiment_6`. Trainer picks one `label_key`.
# - `utterance_frame` — a 50 Hz `filled_pause` label sequence.
# - `word_frame` — one record per primary-stress-annotated word (HR/RS only); a
#   50 Hz `primary_stress` sequence over the word, sliced in memory by the trainer
#   from the utterance WAV via `start_t`/`end_t`.
#
# Both share the same WAVs and speaker-grouped splits, so nothing leaks and the
# splits agree across flavors.
#
# **Not done here** (future recipes, registry stubs below): `event_instance`
# (FP-quality, needs the annotator deliverable) and `word_frame` (primary stress,
# HR/RS only).
#
# ---
#
# ## 0. Setup

# %%
import time

# ── Stage timing ──────────────────────────────────────────────
# Tiny wall-clock harness: mark(stage) stamps a milestone; the final cell
# prints a per-stage breakdown. Stdlib-only, cheap, survives a partial run.
STAGE_TIMES: dict[str, float] = {}
def mark(stage: str) -> None:
    STAGE_TIMES[stage] = time.time()

mark("start")

import sys
from pathlib import Path

HERE = Path.cwd()
if HERE.name != "1_data_prep":
    candidate = HERE / "1_data_prep"
    if candidate.exists():
        HERE = candidate
sys.path.insert(0, str(HERE))

import utils_dataprep as udp
import utils_audio_splitter as uas

PROJECT_ROOT = udp.PROJECT_ROOT
print(f"PROJECT_ROOT = {PROJECT_ROOT}")

# %% [markdown]
# Standard imports.

# %%
import json
from collections import Counter
from dataclasses import dataclass, field, replace

from tqdm.auto import tqdm

# %% [markdown]
# ---
#
# ## 1. Config
#
# - `lang` — `""` → process every ParlaSpeech-{LANG} under `data/unpacked/`; set
#   (e.g. `"RS"`) to pin one.
# - `recipes` — which instance-shapes to emit.
# - `convert_audio` — FLAC → 16 kHz mono WAV. `False` re-writes JSONLs against
#   already-converted WAVs without touching audio.
# - `num_workers` — parallel FLAC→WAV (each FLAC independent → threads win).
# - `audio_index_cache` — persistent `{basename → path}` index, per lang; built
#   once, reused after. Delete the file to force a rescan.
#
# `configure(lang)` derives all per-lang paths + the recipe registry; the section-12
# driver calls it once per language.

# %%
KNOWN_LANGS = ["HR", "RS", "PL", "CZ"]

@dataclass
class Config:
    lang: str = ""                       # "" → all found; or pin e.g. "HR"
    recipes: tuple = ("utterance_instance", "utterance_frame", "word_frame")

    output_dir:    str = "data/processed_jsonl"
    convert_audio: bool = True
    num_workers:   int  = 8
    cache_index:   bool = True

    frame_rate_hz: int = 50

    split_ratios: tuple = (0.8, 0.1, 0.1)
    split_seed:   str   = "parlaspeech-v1"
    split_group:  str   = "speaker"      # group splits by speaker (no leakage)

    min_duration_s: float = 0.1

    test_mode:      bool = False                                       ############ TEST MODE
    test_n_records: int  = 500

cfg = Config()

# ── Resolve the set of languages to process ───────────────────────────────────
if cfg.lang:
    LANGS = [cfg.lang]
else:
    LANGS = [L for L in KNOWN_LANGS
             if (PROJECT_ROOT / f"data/unpacked/ParlaSpeech-{L}").exists()]
    if not LANGS:
        raise FileNotFoundError(
            "cfg.lang empty and no ParlaSpeech-{LANG} dirs under data/unpacked/. "
            "Run 10_download_data.ipynb first.")

if cfg.test_mode:
    udp.banner("🧪 TEST MODE", char="-")
print(f"languages to process: {LANGS}")

def configure(lang: str) -> dict:
    """Per-lang view: paths + recipe registry. (build_recipes defined in §2.)"""
    L, low = lang, lang.lower()
    DATASET = f"ParlaSpeech-{L}"
    OUT_DIR = cfg.output_dir if not cfg.test_mode else "data/test_processed_jsonl"
    WAV_DIR = (f"data/cut_audio/ParlaSpeech-{L}" if not cfg.test_mode
               else f"data/cut_audio/test/ParlaSpeech-{L}")
    ctx = dict(
        L=L, l=low, DATASET=DATASET, OUT_DIR=OUT_DIR, WAV_DIR=WAV_DIR,
        jsonl_path=f"data/unpacked/ParlaSpeech-{L}/ParlaSpeech-{L}.v3.0/ParlaSpeech-{L}.v3.0.jsonl",
        audio_base_dir=f"data/unpacked/ParlaSpeech-{L}-audio",
        audio_index_cache=f"data/processed_jsonl/parlaspeech_{low}_audio_index.json",
        RECIPES=build_recipes(OUT_DIR, low),
    )
    for name in cfg.recipes:
        if name not in ctx["RECIPES"]:
            raise ValueError(f"unknown recipe {name!r}. Known: {sorted(ctx['RECIPES'])}")
    return ctx


# %% [markdown]
# ---
#
# ## 2. Recipe registry
#
# A *recipe* is an instance-shape: `(unit × cut × instance|frame)`. Multiple label
# keys can live in one recipe — task type is a downstream parameter, not a reason
# to split files. This mirrors the `TARGETS` dict in `30_train_instance.ipynb`.
#
# `requires` gates corpus-specific tiers (e.g. `primary_stress` is HR/RS only) so a
# recipe fails loudly on a corpus that lacks them rather than emitting garbage.

# %%
def build_recipes(OUT_DIR: str, l: str) -> dict:
    return {
        # ---- emitted here (whole utterance, no cut) -------------------------
        "utterance_instance": dict(
            unit="utterance", level="instance", cut=False, requires=(),
            out=f"{OUT_DIR}/parlaspeech_{l}_utterance_instance.jsonl"),
        "utterance_frame": dict(
            unit="utterance", level="frame", cut=False, requires=("filled_pauses",),
            label_key="filled_pause",
            out=f"{OUT_DIR}/parlaspeech_{l}_utterance_frame.jsonl"),

        # ---- word-as-instance, sliced in memory by 41 (no word WAVs) --------
        # One record per primary-stress-annotated (multisyllabic) word. audio_path
        # is the utterance WAV; start_t/end_t mark the word's span within it, which
        # the frame trainer slices at load time. HR/RS only (needs primary_stress
        # + words_align); langs without them are skipped at write time.
        "word_frame": dict(
            unit="word", level="frame", cut=False,
            requires=("primary_stress", "words_align"),
            label_key="primary_stress",
            out=f"{OUT_DIR}/parlaspeech_{l}_word_frame.jsonl"),

        # ---- future (need cut=True via utils_audio_splitter) ----------------
        # "event_instance": one record per FP event; label = FP quality
        #   (vowel / vowel+nasal / nasal / other / NA). Needs annotator. unit="event".
    }


# %% [markdown]
# ---
#
# ## 3. Locate the JSONL

# %%
def locate_jsonl(ctx: dict) -> Path:
    p = PROJECT_ROOT / ctx["jsonl_path"]
    if not p.exists():
        raise FileNotFoundError(
            f"JSONL not found: {p}\nRun 10_download_data.ipynb for {ctx['DATASET']} first.")
    print(f"✅ {p.relative_to(PROJECT_ROOT)}  ({p.stat().st_size/1e6:.0f} MB)")
    return p


# %% [markdown]
# ---
#
# ## 4. Preflight — peek at one record
#
# Confirms which tiers this language actually has (sentiment/words/`_align` vary).

# %%
def preflight(jsonl_path: Path) -> None:
    with open(jsonl_path, encoding="utf-8") as f:
        ex = json.loads(f.readline())
    si = ex.get("speaker_info", {})
    fps = ex.get("filled_pauses")
    fp_state = "failed" if fps is None else ("none" if fps == [] else f"{len(fps)} FP")
    print(f"  e.g. {ex['id']}  | {si.get('Speaker_gender','?')} / {si.get('Lang','?')} "
          f"| FP={fp_state} | sentiment={'sentiment' in ex} "
          f"| words={'words' in ex} | aligns={'words_align' in ex} "
          f"| stress={'primary_stress' in ex}")


# %% [markdown]
# ---
#
# ## 5. Parse — build canonical records
#
# One canonical record per utterance, carrying **every** label key. `None` means
# "not available for this utterance" (FP inference failed, or gender `"-"`); the
# trainer drops `None` per-task, so one failed tier never costs the others.
#
# Keeps the `words` tier (needed for future word-level recipes). The two `_align`
# tiers are large and HR/RS-only — extraction lines present but commented out.

# %%
def gender_label(si):
    g = si.get("Speaker_gender")
    return g if g in ("M", "F") else None   # "-"/missing → None (dropped per-task)

def speaker_age(si):
    """Age at recording = year(Date) − Speaker_birth. None if either is missing/"-"."""
    try:
        return int(si.get("Date", "")[:4]) - int(si.get("Speaker_birth"))
    except (TypeError, ValueError):
        return None   # "-", "", or None → not computable
    
def parse_records(ctx: dict, jsonl_path: Path) -> list[dict]:
    DATASET, WAV_DIR = ctx["DATASET"], ctx["WAV_DIR"]

    def parse_record(r):
        dur = round(float(r.get("audio_length", 0.0)), 3)
        if dur < cfg.min_duration_s:
            return None
        si  = r.get("speaker_info", {})
        fps = r.get("filled_pauses")                 # None = failed; [] = none
        sent = r.get("sentiment") or {}
        raw_audio = r.get("audio")
        file_hash = Path(raw_audio).parts[0] if raw_audio else None
        stem      = Path(raw_audio).stem if raw_audio else r["id"]
        return {
            "instance_id": r["id"],
            "dataset":     DATASET,
            "file_id":     file_hash,
            "audio_path":  f"{WAV_DIR}/{file_hash}/{stem}.wav",
            "speaker":     si.get("Speaker_ID", "unknown"),
            "text":        r.get("text"),
            "labels": {
                "speaker_gender":       gender_label(si),
                "speaker_age":          speaker_age(si),
                "filled_pause_present": None if fps is None else int(bool(fps)),
                "filled_pause_count":   None if fps is None else len(fps),
                "sentiment_logit":      sent.get("ParlaSent_logit"),
                "sentiment_6":          sent.get("ParlaSent_6"),
            },
            "metadata": {
                "source_audio": raw_audio,
                "audio_length": dur,
                "lang":         si.get("Lang"),
                "speaker_info": si,
                "words":        r.get("words"),
                # "words_align": r.get("words_align"),   # captured below for word_frame
                # "chars_align": r.get("chars_align"),   # HR/RS only, bulky
                "filled_pauses": fps,                    # scratch (for frame recipe; stripped on write)
                "words_align":   r.get("words_align"),   # scratch (HR/RS; word_frame; stripped on write)
                "primary_stress": r.get("primary_stress"),  # scratch (HR/RS; word_frame; stripped on write)
            },
        }

    records, n_total, n_short = [], 0, 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"parsing {DATASET}", unit=" lines", leave=False):
            n_total += 1
            if cfg.test_mode and n_total > cfg.test_n_records:
                break
            rec = parse_record(json.loads(line))
            if rec is None:
                n_short += 1
                continue
            records.append(rec)
    print(f"  parsed {n_total} lines  kept {len(records)}  dropped(short) {n_short}")
    return records


# %% [markdown]
# ---
#
# ## 6. Stats

# %%
def print_stats(records: list[dict]) -> None:
    gender = Counter(r["labels"]["speaker_gender"] for r in records)
    fp_known = [r for r in records if r["labels"]["filled_pause_present"] is not None]
    n_fp_pos = sum(r["labels"]["filled_pause_present"] for r in fp_known)
    sent_known = sum(1 for r in records if r["labels"]["sentiment_logit"] is not None)
    speakers = {r["speaker"] for r in records}
    print(f"  speakers: {len(speakers)}  gender: {dict(gender)}")
    if fp_known:
        print(f"  FP labelled: {len(fp_known)} (failed {len(records)-len(fp_known)})  "
              f"present: {n_fp_pos} ({100*n_fp_pos/len(fp_known):.1f}%)")
    print(f"  sentiment logits: {sent_known}")


# %% [markdown]
# ---
#
# ## 7. Assign splits — grouped by speaker
#
# Deterministic; the same speaker always lands in the same split, so no speaker
# leaks between train/dev/test.

# %%
def do_splits(records: list[dict]) -> None:
    udp.assign_splits(records, ratios=cfg.split_ratios,
                      group_key=cfg.split_group, seed=cfg.split_seed, overwrite=True)
    print(f"  splits: {udp.split_summary(records)}")


# %% [markdown]
# ---
#
# ## 8. Convert FLAC → WAV
#
# Whole-file convert via `utils_audio_splitter`. The basename-index resolver finds
# each FLAC regardless of `partX/` nesting; the index is cached so re-runs skip the
# scan. Unresolved records are dropped. Prints where the cut audio landed; per-lang
# stats only when something was missing/failed/skipped.

# %%
def convert_audio(ctx: dict, records: list[dict]) -> list[dict]:
    WAV_DIR = ctx["WAV_DIR"]
    if not cfg.convert_audio:
        print(f"  convert_audio=False — assuming WAVs under {WAV_DIR}")
        return records
    resolver = uas.make_flac_index_resolver(
        ctx["audio_base_dir"],
        record_key_path=("metadata", "source_audio"),
        index_cache_path=(ctx["audio_index_cache"] if cfg.cache_index else None),
    )
    records, stats = uas.cut_dataset(records, resolver, num_workers=cfg.num_workers)
    msg = f"🔊 cut audio → {WAV_DIR}  (kept {stats['kept']})"
    extra = [f"{k} {stats[k]}" for k in ("missing_source", "cut_failed", "skipped_existing")
             if stats[k]]
    if extra:
        msg += "  [" + ", ".join(extra) + "]"
    print(msg)
    return records


# %% [markdown]
# ---
#
# ## 9. Frame label helper
#
# 50 Hz binary sequence from the `filled_pauses` intervals.

# %%
def compute_frame_labels(filled_pauses, dur, hz):
    n = round(dur * hz)
    labels = [0] * n
    for fp in (filled_pauses or []):
        s = max(0, round(fp["time_s"] * hz))
        e = min(n, round(fp["time_e"] * hz))
        for i in range(s, e):
            labels[i] = 1
    return labels


# Scratch metadata fields used to build recipes; stripped from every written row.
_SCRATCH = ("filled_pauses", "words_align", "primary_stress")


def _strip_scratch(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if k not in _SCRATCH}


def build_word_frame_rows(r, hz):
    """Expand one utterance → one record per primary-stress-annotated word.

    The instance is the WORD: audio_path stays the utterance WAV and start_t/end_t
    mark the word's [time_s, time_e] span within it (41 slices in memory). The
    `primary_stress` frame sequence spans just the word; positive frames come from
    each annotation's `raw` interval (utterance-relative, same clock as the word
    bounds), offset to word-local and clamped to the word."""
    md = r["metadata"]
    wa = md.get("words_align")
    ps_list = md.get("primary_stress")
    if not wa or not ps_list:
        return []
    base = {k: r[k] for k in ("dataset", "file_id", "speaker", "split")}
    rows = []
    for ps in ps_list:
        wai = ps.get("words_align_idx")
        if wai is None or not (0 <= wai < len(wa)):
            continue
        word = wa[wai]
        w_s, w_e = float(word["time_s"]), float(word["time_e"])
        word_dur = round(w_e - w_s, 3)
        n = round(word_dur * hz)
        if n <= 0:
            continue
        labels = [0] * n
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
            "start_t":      round(w_s, 3),   # word span within the utterance WAV
            "end_t":        round(w_e, 3),
            "frame_rate_hz": hz,
            "labels":       {"primary_stress": labels},
            "metadata": {
                "audio_length":     word_dur,
                "lang":             md.get("lang"),
                "source_utterance": r["instance_id"],
                "words_align_idx":  wai,
            },
        })
    return rows


# %% [markdown]
# ---
#
# ## 10. Write recipe JSONLs
#
# `utterance_instance` carries the scalar labels. `utterance_frame` carries the
# 50 Hz `filled_pause` sequence and only includes utterances where FP inference
# succeeded. `word_frame` expands each utterance into one record per primary-
# stress-annotated word. All scratch fields are stripped from every output.

# %%
def build_instance(r):
    out = {k: r[k] for k in ("instance_id", "dataset", "file_id",
                             "audio_path", "speaker", "text", "split")}
    out["labels"]   = dict(r["labels"])
    out["metadata"] = _strip_scratch(r["metadata"])
    return out

def build_frame(r):
    fps = r["metadata"]["filled_pauses"]
    dur = r["metadata"]["audio_length"]
    out = {k: r[k] for k in ("instance_id", "dataset", "file_id",
                             "audio_path", "speaker", "text", "split")}
    out["frame_rate_hz"] = cfg.frame_rate_hz
    out["labels"] = {"filled_pause": compute_frame_labels(fps, dur, cfg.frame_rate_hz)}
    out["metadata"] = _strip_scratch(r["metadata"])
    return out

def write_recipes(ctx: dict, records: list[dict]) -> dict:
    written = {}
    for name in cfg.recipes:
        spec = ctx["RECIPES"][name]
        if spec["level"] == "instance":
            rows = [build_instance(r) for r in records]
        elif spec["unit"] == "word":
            rows = [wr for r in records
                    for wr in build_word_frame_rows(r, cfg.frame_rate_hz)]
        else:  # utterance frame
            rows = [build_frame(r) for r in records
                    if r["metadata"][spec["requires"][0]] is not None]
        if not rows:
            req = "+".join(spec.get("requires", ()))
            print(f"  ⏭  {name}: 0 rows for {ctx['L']} (corpus lacks {req}) — skipped")
            continue
        n = udp.write_jsonl(rows, spec["out"])
        written[name] = (spec["out"], n)
        print(f"  ✅ {name}: {n} → {spec['out']}")
    return written


# %% [markdown]
# ---
#
# ## 11. Sanity checks

# %%
def sanity(ctx: dict, written: dict) -> None:
    for name, (path, n) in written.items():
        rows = udp.read_jsonl(path)
        n_tot, n_valid, errs = udp.validate_jsonl(rows)
        tag = "✅" if not errs else "⚠️ "
        print(f"  {tag} {name}: {n_valid}/{n_tot} valid")
        for e in errs[:3]:
            print(f"       {e}")
        if ctx["RECIPES"][name]["level"] == "frame":
            key = ctx["RECIPES"][name]["label_key"]
            bad = sum(1 for r in rows
                      if abs(len(r["labels"][key])
                             - round(r["metadata"]["audio_length"] * cfg.frame_rate_hz)) > 1)
            n_frames = sum(len(r["labels"][key]) for r in rows)
            n_pos    = sum(sum(r["labels"][key]) for r in rows)
            pos_pct  = (100.0 * n_pos / n_frames) if n_frames else 0.0
            print(f"       {key}: frame-length vs duration {bad} mismatches (>1 frame) "
                  f"| positive frames {n_pos}/{n_frames} ({pos_pct:.2f}%)")
        else:
            present = Counter(r["labels"]["filled_pause_present"] for r in rows)
            null_wav = sum(1 for r in rows if not r.get("audio_path"))
            print(f"       filled_pause_present: {dict(present)}  | audio_path null: {null_wav}")


# %% [markdown]
# ---
#
# ## 12. Run — all languages
#
# One pass per language; each is parsed, split, converted, and written before the
# next, so records don't accumulate across languages.

# %%
summary = {}
for LANG in LANGS:
    udp.banner(f"▶  {LANG}", char="=")
    mark(f"{LANG}: begin")
    ctx = configure(LANG)
    jsonl_path = locate_jsonl(ctx)
    preflight(jsonl_path)
    records = parse_records(ctx, jsonl_path)
    print_stats(records)
    do_splits(records)
    records = convert_audio(ctx, records)
    written = write_recipes(ctx, records)
    sanity(ctx, written)
    summary[LANG] = {name: n for name, (p, n) in written.items()}
    del records                      # free before the next language
    mark(f"{LANG}: done")

udp.banner("ALL LANGUAGES DONE", char="=")
for lng, w in summary.items():
    print(f"  {lng}: " + ", ".join(f"{k}={v}" for k, v in w.items()))

mark("end")


# %% [markdown]
# ---
#
# ## Timing
#
# Per-stage wall-clock, each stage measured as the delta from the previous
# mark, plus a total. Prints whatever marks exist, so a partial run still
# reports cleanly.

# %%
def print_stage_breakdown(times: dict[str, float]) -> None:
    items = list(times.items())
    if not items:
        print("no timing recorded")
        return
    def mmss(s: float) -> str:
        s = int(round(max(0.0, s)))
        return f"{s // 60:02d}:{s % 60:02d}"
    width = max(len(k) for k, _ in items)
    print("stage timing (delta from previous mark)")
    print("-" * (width + 11))
    prev = items[0][1]
    for name, t in items:
        print(f"  {name:<{width}}  {mmss(t - prev)}")
        prev = t
    print("-" * (width + 11))
    print(f"  {'TOTAL':<{width}}  {mmss(items[-1][1] - items[0][1])}")

print_stage_breakdown(STAGE_TIMES)


# %% [markdown]
# ---
#
# ## Next
#
# - **Chapter 2** — `20_sniff_dataset.ipynb` at either JSONL.
# - **Chapter 3** — add a `TARGETS` entry: `parlaspeech_{lang}_utterance_instance.jsonl`,
#   `label_key` ∈ `{speaker_gender, filled_pause_present, filled_pause_count}`
#   (classification) or `sentiment_logit` (regression).
# - **Chapter 4** — frame trainer at `..._utterance_frame.jsonl`.
