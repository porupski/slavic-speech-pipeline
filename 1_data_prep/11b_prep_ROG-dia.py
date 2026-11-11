# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ssp
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prep ROG-Dialog — chapter 1
#
# Convert ROG-Dialog EXB files into the canonical JSONL **and** cut the
# per-segment 16 kHz mono WAVs in one pass, using `utils_audio_splitter`.
#
# **Input**
# - `data/unpacked/ROG-Dialog/ROG-Dialog/...` — annotation files from `ROG-Dialog.zip`
#   - `.exb` files (EXMARaLDA Partitur Editor format, the source we parse here)
#   - These contain: speaker tiers (`colloq`, `norm`), dialogue act tiers (`dialogueActsIsoDimension`, `dialogueActsIsoFunction`), and sentiment tiers (`sentimentCurated`, `sentimentAnnotated`).
# - `data/unpacked/ROG-Dialog/ROG-Dialog_audio/...` — the session WAVs. The splitter
#   indexes this tree by stem (matched against each record's `metadata.source_file`)
#   and slices with the record's own `start_t` / `end_t`.
#
# **Output**
# - `data/processed_audio/ROG-Dialog/{file_id}_{speaker}_{start}_{end}.wav` —
#   one 16 kHz mono PCM-16 WAV per segment.
# - `data/processed_jsonl/rog_dialog_instance.jsonl` — one canonical-JSONL line per
#   `colloq` segment whose WAV was cut (records whose source audio could not be
#   resolved are dropped). `audio_path` points at the file that now exists.
#
# **What this notebook does NOT do**
# - Choose a target label (sentiment vs. dialogue act vs. ...). All labels are written into the `labels` dict; the trainer picks one via its config.
#
# ---
#
# ## 0. Setup

# %%
import sys
from pathlib import Path

HERE = Path.cwd()
if HERE.name != "1_data_prep":
    candidate = HERE / "1_data_prep"
    if candidate.exists():
        HERE = candidate
sys.path.insert(0, str(HERE))

import utils_dataprep as udp
PROJECT_ROOT = udp.PROJECT_ROOT
print(f"PROJECT_ROOT = {PROJECT_ROOT}")

# %% [markdown]
# ---
#
# ## 1. Config
#
# All knobs are here. Run with `test_mode=True` first to process a handful of files end-to-end.

# %%
from dataclasses import dataclass

@dataclass
class Config:
    # Which ROG-Dialog release to prepare. Default is v2.0 (hdl 11356/2637), which bundles
    # WAVs and EXBs inside a single EXMARaLDA zip. Set to "ROG-Dialog" for the legacy v1.0
    # (hdl 11356/2073) which ships WAVs and EXBs in two separate archives.
    source_name: str = "ROG-Dialog-v2"

    # Where to find unpacked EXB files. Recursive glob under the unpack dir. `{source_name}`
    # is substituted at runtime. v2's EXMARaLDA.zip nests its contents under
    # `ROG-Dialog-2.0-EXMARaLDA/...`; v1 nests under `ROG-Dialog/...`. The `**` keeps us
    # agnostic to the exact inner layout.
    exb_glob: str = "data/unpacked/{source_name}/**/*.exb"

    # Session-WAV root. The splitter indexes this tree recursively by stem (matched against
    # each record's `metadata.source_file`). v2 bundles audio in the EXMARaLDA zip (same tree
    # as EXBs); v1 has a separate `{source_name}_audio/` subdir.
    audio_source_dir: str = "data/unpacked/{source_name}"
    audio_source_pattern: str = "*.wav"

    # Where the cut WAVs land (per-instance 16 kHz mono PCM-16).
    processed_audio_subdir: str = "data/processed_audio/{source_name}"

    # Final canonical JSONL (post-cut). audio_path points at files that exist.
    # v2 → rog_dialog_v2_instance.jsonl; v1 → rog_dialog_instance.jsonl.
    output_jsonl: str = "data/processed_jsonl/{source_name_jsonl}_instance.jsonl"

    # Cut toggle — set False to rewrite the JSONL against an already-cut WAV cache
    # without touching audio (useful when iterating on label logic).
    cut_audio: bool = True

    # num_workers for the splitter. 0 = sequential + LRU source cache, which is
    # the right choice for ROG (many segments share one session WAV).
    num_workers: int = 0

    # Split ratios (train, dev, test). Grouped by file_id so no recording leaks across splits.
    split_ratios: tuple = (0.8, 0.1, 0.1)

    # Drop segments whose `norm` text is missing or shorter than this many chars.
    min_text_chars: int = 2

    # Tolerance for matching tier timestamps. The EXB extractor rounds to 3 decimals,
    # so 0.001 is the natural tolerance for exact-match lookups.
    timestamp_tolerance: float = 0.001

    # Test mode
    test_mode: bool = False                                                                 ################### TEST MODE
    test_n_files: int = 3   # cap on number of EXB files in test mode

cfg = Config()

# Resolve `{source_name}` / `{source_name_jsonl}` placeholders in path fields.
# source_name_jsonl: "ROG-Dialog-v2" → "rog_dialog_v2", "ROG-Dialog" → "rog_dialog".
_source_jsonl = cfg.source_name.lower().replace("-", "_")
for _f in ("exb_glob", "audio_source_dir", "processed_audio_subdir", "output_jsonl"):
    _v = getattr(cfg, _f)
    if "{" in _v:
        setattr(cfg, _f, _v.format(source_name=cfg.source_name, source_name_jsonl=_source_jsonl))

print(cfg)

# Tier categories we ask for if present in ROG-Dialog EXBs.
WANTED_CATEGORIES = [
    "colloq", "norm",
    "dialogueActsIsoDimension", "dialogueActsIsoFunction",
    "sentimentCurated", "sentimentAnnotated",
]

# %% [markdown]
# ---
#
# ## 2. Find input EXB files
#
# We glob the unpacked ROG-Dialog folder. If nothing turns up, you probably haven't run `download_data.ipynb` yet, or the unpacked layout is different from what we expect — check the path the glob is hitting.

# %%
exb_files = sorted((PROJECT_ROOT).glob(cfg.exb_glob))
print(f"Found {len(exb_files)} EXB files matching {cfg.exb_glob}")
if cfg.test_mode:
    exb_files = exb_files[: cfg.test_n_files]
    print(f"\U0001f9ea TEST MODE: capped to {len(exb_files)} files")

for p in exb_files[:5]:
    print(f"  {p.relative_to(PROJECT_ROOT)}")
if len(exb_files) > 5:
    print(f"  ... and {len(exb_files) - 5} more")

if not exb_files:
    raise FileNotFoundError(
        f"No EXB files found at {PROJECT_ROOT / cfg.exb_glob}. "
        "Run download_data.ipynb with dataset='ROG-Dialog' first, then re-check the glob path."
    )

# %% [markdown]
# ---
#
# ## 3. Preflight: which tiers exist across all files?
#
# ROG-Dialog carries sentiment and dialogue-act annotations on all files. This cell confirms that and shows which categories are present. If `sentimentCurated` or the dialogue-act tiers show 0 files, something is wrong with the glob or the unpacked layout.

# %%
from lxml import etree
from collections import Counter, defaultdict

def _quick_subcorpus(doc):
    """Return SUBCORPUS metadata if present."""
    for ud in doc.findall(".//meta-information//ud-information"):
        if ud.attrib.get("attribute-name") == "SUBCORPUS":
            return ud.text
    return "unknown"


category_to_subcorpora = defaultdict(Counter)  # {category: Counter(subcorpus: n_files_having_it)}
subcorpus_counts = Counter()
files_with_category = Counter()                # {category: n_files_having_it}

for exb_path in exb_files:
    try:
        doc = etree.parse(str(exb_path))
    except Exception as e:
        print(f"\u26a0\ufe0f  could not parse {exb_path.name}: {e}")
        continue
    sub = _quick_subcorpus(doc)
    subcorpus_counts[sub] += 1
    cats_here = {t.attrib.get("category") for t in doc.findall(".//tier")}
    for cat in cats_here:
        if cat is None:
            continue
        files_with_category[cat] += 1
        category_to_subcorpora[cat][sub] += 1

print(f"Scanned {len(exb_files)} files\n")
print(f"Subcorpora seen:")
for sub, n in subcorpus_counts.most_common():
    print(f"  {sub:30s} {n} files")

print(f"\nCategory presence across files:")
print(f"  {'category':35s}  files   in_subcorpora")
for cat, n in files_with_category.most_common():
    subs = ", ".join(f"{s}={k}" for s, k in category_to_subcorpora[cat].most_common())
    flag = "  \u2190  wanted" if cat in WANTED_CATEGORIES else ""
    print(f"  {cat:35s}  {n:5d}   {subs}{flag}")

print(f"\nWanted categories that are MISSING from every file:")
missing = [c for c in WANTED_CATEGORIES if files_with_category[c] == 0]
if not missing:
    print("  (none \u2014 all wanted categories appear somewhere)")
else:
    for c in missing:
        print(f"  \u26a0\ufe0f  {c}")

# %% [markdown]
# ---
#
# ## 4. EXB parsing primitives
#
# These are EXB-format helpers shared in spirit with `prep_ROG.ipynb`. ROG-Dialog EXBs follow the same EXMARaLDA structure:
# - A `<timeline>` block maps TLI ids to seconds.
# - Multiple `<tier>` blocks per speaker and category.
# - Each tier contains `<event start=".." end="..">text</event>` items.
#
# ROG-Dialog is a dialogue corpus so each EXB file will contain tiers for **multiple speakers** (unlike ROG-Art monologues). The speaker-scoped tier structure handles this naturally — we collect events per `(start_t, end_t, speaker)` key.

# %%
from lxml import etree
from collections import Counter

def get_timeline(doc):
    """Map TLI id \u2192 time in seconds."""
    return {
        tli.attrib["id"]: float(tli.attrib["time"])
        for tli in doc.findall(".//tli")
        if "time" in tli.attrib
    }


def list_tier_categories(doc):
    """Return Counter of {category: n_tiers} for every <tier> in the doc."""
    return Counter(t.attrib.get("category", "<none>") for t in doc.findall(".//tier"))


def extract_tier_intervals(doc, timeline, category):
    """Pull all events for tiers with the given category. Returns list of dicts."""
    intervals = []
    for tier in doc.findall(f'.//tier[@category="{category}"]'):
        speaker = tier.attrib.get("speaker")
        for event in tier.findall("event"):
            sid = event.attrib.get("start")
            eid = event.attrib.get("end")
            st = timeline.get(sid)
            et = timeline.get(eid)
            intervals.append({
                "start_t": round(st, 3) if st is not None else None,
                "end_t":   round(et, 3) if et is not None else None,
                "text":    (event.text or "").strip(),
                "speaker": speaker,
            })
    return intervals


def extract_speaker_metadata(doc):
    """Pull speaker block info (gender, age, etc.)."""
    out = {}
    for sp in doc.findall(".//speaker"):
        sid = sp.attrib.get("id")
        entry = {
            "sex": sp.find("sex").attrib.get("value") if sp.find("sex") is not None else None,
            "abbreviation": sp.findtext("abbreviation"),
        }
        for ud in sp.findall(".//ud-information"):
            name = ud.attrib.get("attribute-name")
            if name:
                entry[name] = ud.text
        out[sid] = entry
    return out


def parse_exb(exb_path: Path):
    """Parse one EXB. Returns (file_id, speakers, tiers_by_category, all_categories)."""
    doc = etree.parse(str(exb_path))
    file_id = exb_path.stem
    timeline = get_timeline(doc)
    speakers = extract_speaker_metadata(doc)
    all_categories = list_tier_categories(doc)
    tiers = {
        cat: extract_tier_intervals(doc, timeline, cat)
        for cat in WANTED_CATEGORIES
    }
    return file_id, speakers, tiers, all_categories


# Smoke test on the first file
if exb_files:
    file_id, speakers, tiers, all_cats = parse_exb(exb_files[0])
    print(f"file_id  = {file_id}")
    print(f"speakers = {list(speakers.keys())}")
    print(f"\nAll tier categories present in this EXB:")
    for cat, n in all_cats.most_common():
        marker = " \u2713 (wanted)" if cat in WANTED_CATEGORIES else ""
        print(f"  {cat:35s} {n:3d} tiers{marker}")
    print(f"\nWanted categories \u2014 interval counts:")
    for cat, intervals in tiers.items():
        print(f"  {cat:35s} {len(intervals):5d} intervals")


# %% [markdown]
# ---
#
# ## 5. Stitching tiers into per-segment records
#
# ROG-Dialog's six tiers split into two groups:
#
# 1. **Aligned tiers** (`colloq`, `norm`, `dialogueActsIsoDimension`, `dialogueActsIsoFunction`) — these share exact time spans for the same speaker. We join them on `(start_t, end_t, speaker)`.
# 2. **Overlap tiers** (`sentimentCurated`, `sentimentAnnotated`) — these have their own boundaries that *contain* one or more `colloq` segments. We find which sentiment interval each `colloq` segment falls inside.
#
# The result is one record per `colloq` segment, with every available label attached.

# %%
def _make_key(interval):
    s, e, sp = interval.get("start_t"), interval.get("end_t"), interval.get("speaker")
    if s is None or e is None:
        return None
    return (round(s, 3), round(e, 3), sp)


def _find_containing_sentiment(seg, sentiment_intervals, tol):
    """Return the sentiment label whose span contains seg, or None."""
    s, e = seg.get("start_t"), seg.get("end_t")
    if s is None or e is None:
        return None
    for sent in sentiment_intervals:
        ss, ee = sent.get("start_t"), sent.get("end_t")
        if ss is None or ee is None:
            continue
        if ss - tol <= s <= ee + tol and ss - tol <= e <= ee + tol:
            return sent.get("text") or None
    return None


def stitch_segments(file_id, speakers, tiers, tol=0.001):
    """Build one record per colloq segment, with all matching labels."""
    norm_lookup = {k: v for v in tiers["norm"] if (k := _make_key(v))}
    dim_lookup  = {k: v for v in tiers["dialogueActsIsoDimension"] if (k := _make_key(v))}
    func_lookup = {k: v for v in tiers["dialogueActsIsoFunction"] if (k := _make_key(v))}

    out = []
    for colloq in tiers["colloq"]:
        key = _make_key(colloq)
        if key is None:
            continue

        norm = norm_lookup.get(key)
        dim  = dim_lookup.get(key)
        func = func_lookup.get(key)

        out.append({
            "file_id": file_id,
            "speaker": colloq["speaker"],
            "start_t": colloq["start_t"],
            "end_t":   colloq["end_t"],
            "colloq_text": colloq["text"],
            "norm_text":   norm["text"] if norm else None,
            "dialogue_act_dimension": (dim["text"] if dim else None) or None,
            "dialogue_act_function":  (func["text"] if func else None) or None,
            "sentiment_curated":   _find_containing_sentiment(colloq, tiers["sentimentCurated"], tol),
            "sentiment_annotated": _find_containing_sentiment(colloq, tiers["sentimentAnnotated"], tol),
            "speaker_metadata": speakers.get(colloq["speaker"], {}),
        })
    return out


# Smoke test
if exb_files:
    file_id, speakers, tiers, _all_cats = parse_exb(exb_files[0])
    segs = stitch_segments(file_id, speakers, tiers, tol=cfg.timestamp_tolerance)
    print(f"Stitched {len(segs)} segments from {file_id}")
    if segs:
        import json as _json
        print("First segment:")
        print(_json.dumps(segs[0], ensure_ascii=False, indent=2))


# %% [markdown]
# ---
#
# ## 6. Mapping to canonical JSONL records
#
# Take the stitched segments and emit canonical-JSONL dicts:
# - `instance_id` via `udp.make_instance_id`
# - `audio_path` predicted (the splitter will create this file later)
# - `labels` dict carries everything we extracted
# - `metadata` dict carries speaker info and source provenance

# %%
def to_canonical(segment: dict, cfg: Config) -> dict:
    speaker = segment["speaker"]
    file_id = segment["file_id"]
    st, et = segment["start_t"], segment["end_t"]

    iid = udp.make_instance_id("ROG-Dialog", file_id, speaker, st, et)

    # Predict the future cut-WAV path so the splitter knows where to write.
    cut_name = f"{file_id}_{speaker}_{st:.3f}_{et:.3f}.wav"
    audio_path = f"{cfg.processed_audio_subdir}/{cut_name}"

    # Use norm if present, else colloq. Records with neither are dropped later by clean().
    text = segment.get("norm_text") or segment.get("colloq_text") or ""

    labels = {}
    if segment.get("sentiment_curated"):
        labels["sentiment"] = segment["sentiment_curated"]
    elif segment.get("sentiment_annotated"):
        # Fall back to the annotator value if curated is missing
        labels["sentiment"] = segment["sentiment_annotated"]
    if segment.get("sentiment_annotated"):
        labels["sentiment_annotated"] = segment["sentiment_annotated"]
    if segment.get("dialogue_act_function"):
        labels["dialogue_act_function"] = segment["dialogue_act_function"]
    if segment.get("dialogue_act_dimension"):
        labels["dialogue_act_dimension"] = segment["dialogue_act_dimension"]

    metadata = {
        "source_file": f"{file_id}.exb",
        "source_format": "EXB",
        "speaker_info": segment.get("speaker_metadata", {}),
        "colloq_text": segment.get("colloq_text"),  # keep original colloquial form for reference
    }

    return {
        "instance_id": iid,
        "dataset": "ROG-Dialog",
        "file_id": file_id,
        "audio_path": audio_path,
        # split is set later by udp.assign_splits \u2014 placeholder for now
        "split": "train",
        "speaker": speaker,
        "start_t": st,
        "end_t":   et,
        "text": text,
        "labels": labels,
        "metadata": metadata,
    }


# Smoke test
if exb_files:
    sample = to_canonical(segs[0], cfg)
    errs = udp.validate_instance(sample)
    if errs:
        print("\u26a0\ufe0f  validation errors on sample:")
        for e in errs:
            print(f"   - {e}")
    else:
        print("\u2705 sample validates against canonical schema")
    import json as _json
    print(_json.dumps(sample, ensure_ascii=False, indent=2))

# %% [markdown]
# ---
#
# ## 7. Run over all files
#
# Process every EXB, stitch, map to canonical, accumulate.

# %%
records = []
file_stats = []

for exb_path in exb_files:
    try:
        file_id, speakers, tiers, _all_cats = parse_exb(exb_path)
        segs = stitch_segments(file_id, speakers, tiers, tol=cfg.timestamp_tolerance)
        these = [to_canonical(s, cfg) for s in segs]
        records.extend(these)
        file_stats.append({
            "file_id": file_id,
            "n_segments": len(these),
            "n_with_sentiment": sum(1 for r in these if "sentiment" in r["labels"]),
            "n_with_dialogue_act": sum(1 for r in these if "dialogue_act_function" in r["labels"]),
        })
        print(f"  {file_id:50s} {len(these):4d} segs   sent={file_stats[-1]['n_with_sentiment']:4d}   da={file_stats[-1]['n_with_dialogue_act']:4d}")
    except Exception as e:
        print(f"  \u274c {exb_path.name}: {e}")

print(f"\nTotal records before cleaning: {len(records)}")
print(f"Of those: with sentiment    = {sum(s['n_with_sentiment'] for s in file_stats)}")
print(f"          with dialogue act = {sum(s['n_with_dialogue_act'] for s in file_stats)}")

# %% [markdown]
# ---
#
# ## 8. Clean and validate
#
# `udp.clean` runs the standard pipeline: drop invalid records, drop duplicates by `instance_id`, drop entries with empty text. We do NOT drop by missing label here — that's the trainer's job at load time (it knows which `label_key` it wants).

# %%
records = udp.clean(records, require_text=True, verbose=True)

# Filter by min text length too
before = len(records)
records = [r for r in records if len(r.get("text", "")) >= cfg.min_text_chars]
print(f"  min_text_chars={cfg.min_text_chars}: -{before - len(records)} ({len(records)} left)")

# Final full validation pass
n_total, n_valid, errs = udp.validate_jsonl(records, max_report=10)
print(f"\nValidation: {n_valid}/{n_total} valid")
if errs:
    print("First errors:")
    for e in errs:
        print(f"  - {e}")

# %% [markdown]
# ---
#
# ## 9. Assign splits
#
# Splits are deterministic (md5 over file_id) and grouped by file_id so no recording leaks across train/dev/test.

# %%
udp.assign_splits(records, ratios=cfg.split_ratios, group_key="file_id", overwrite=True)
counts = udp.split_summary(records)
print(f"Split counts: {counts}")

# Sanity check: same file_id must always have the same split
groups = {}
for r in records:
    groups.setdefault(r["file_id"], set()).add(r["split"])
leaks = {f: s for f, s in groups.items() if len(s) > 1}
print(f"Files with leakage across splits: {len(leaks)}")
assert not leaks, f"split leakage detected: {leaks}"
print("\u2705 no split leakage")

# %% [markdown]
# ---
#
# ## 10. Label distributions (quick peek)
#
# Unlike ROG-Art, ROG-Dialog should have non-empty sentiment and dialogue-act distributions here. If everything is still `None`, check the preflight output in cell 3 and inspect a raw EXB.

# %%
from collections import Counter

def show_label(records, key):
    c = Counter(r["labels"].get(key) for r in records)
    print(f"\n{key}:")
    for k, v in c.most_common():
        bar = "\u2588" * min(40, v * 40 // max(1, max(c.values())))
        print(f"  {str(k):35s} {v:5d}  {bar}")

show_label(records, "sentiment")
show_label(records, "dialogue_act_function")
show_label(records, "dialogue_act_dimension")

# %% [markdown]
# ---
#
# ## 11. Cut per-segment WAVs via `utils_audio_splitter`
#
# A stem-scan resolver indexes `audio_source_dir` by WAV stem; records are
# looked up via `metadata.source_file` (the `.exb` filename, whose stem matches
# the session WAV). Each record's absolute `start_t` / `end_t` are the slice
# bounds. Records the resolver can't match are dropped.
#
# Sequential + LRU (`num_workers=0`) is the right default here: many segments
# share one session WAV, so the cache pays for itself.

# %%
import utils_audio_splitter as uas

if cfg.cut_audio:
    resolver = uas.make_stem_scan_resolver(
        cfg.audio_source_dir,
        source_file_key="source_file",
        pattern=cfg.audio_source_pattern,
    )
    records, cut_stats = uas.cut_dataset(
        records, resolver, num_workers=cfg.num_workers,
    )
    print(f"\nkept {len(records)} records after cut")
else:
    print(f"cut_audio=False — assuming WAVs under {cfg.processed_audio_subdir}")

# %% [markdown]
# ---
#
# ## 12. Write the canonical JSONL

# %%
out_path = PROJECT_ROOT / cfg.output_jsonl
if cfg.test_mode:
    # Don't pollute real outputs in test mode
    out_path = out_path.with_name("test_" + out_path.name)

n_written = udp.write_jsonl(records, out_path)
print(f"✅ wrote {n_written} records to {out_path.relative_to(PROJECT_ROOT)}")

# Quick re-read sanity check
roundtripped = udp.read_jsonl(out_path)
assert len(roundtripped) == n_written
print(f"✅ round-trip read confirms {len(roundtripped)} records")

# %% [markdown]
# ---
#
# ## Next
#
# Point `2_data_analysis/sniff_dataset.py` at `rog_dialog_instance.jsonl` to
# inspect label distributions. For training, set `classification.target` in
# `4_instance_models/4_config.json` to one of the `rog_dia_*` presets
# (`sentiment`, `sentiment_annotated`, `dialogue_act_dimension`,
# `dialogue_act_function`) and run `run_41_classification.py`.
#
# If sentiment or dialogue-act coverage looks unexpectedly low, open a raw EXB
# in EXMARaLDA — the source is the ground truth, not this notebook.
