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
#     display_name: Python (ssp)
#     language: python
#     name: ssp
# ---

# %% [markdown]
# # Prep ROG-Art — chapter 1
#
# Parse ROG-Art EXB files into the canonical JSONL. **Two outputs** from one pass:
#
# 1. **`rog_instance.raw.jsonl`** — one record per `colloq` segment.
#    Carries `sentiment`, `dialogue_act_*`, and now `filled_pause_present` (binary, derived from `vocalDisfluency`).
#    `audio_path` is the *predicted* future cut-WAV location; `audio_splitter.ipynb` actually creates the cuts and produces the final `rog_instance.jsonl`.
#
# 2. **`rog_frame.raw.jsonl`** — one record per source WAV.
#    Carries `labels.filled_pause` as a **50 Hz frame sequence** covering the whole file.
#    `audio_path` points at a 16 kHz mono normalized copy of the source (created by this notebook in the final cell).
#    Chapter 4 (`train_frame.ipynb`) consumes this directly — no splitter step.
#
# **Inputs**
# - `data/unpacked/ROG/ROG/ROG/ROG-Art/EXB/*.exb` — annotation files (the triple-nested `ROG/ROG/ROG` matches the layout the ZIP unpacks into)
# - `data/unpacked/ROG/ROG-Art.wav/ROG/ROG-Art/WAV/*.wav` — source 44.1 kHz mono WAVs
#
# **Why two outputs, one notebook?** The EXB pass is expensive and the tier-stitching logic is shared. We do it once and emit both flavors. Chapter 3 reads the instance JSONL, chapter 4 reads the frame JSONL — same canonical schema.
#
# **What this notebook does NOT do**
# - Cut per-instance audio. That's `audio_splitter.ipynb`'s job (currently only configured for ROG-Dialog; generalizing it to ROG-Art is a follow-up).
# - Choose a target label. All labels are written; the trainer picks one via `cfg.label_key`.
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
# All knobs at the top. `test_mode=True` runs against the first few EXBs only — and for the frame output, caps each record to the first `test_max_seconds` of audio so verification finishes in seconds.
#
# **The path glob.** ROG.zip unpacks into `ROG/ROG/ROG/ROG-Art/`. The original glob assumed one fewer level — that's why the default in the previous version of this notebook missed everything. Fixed.
#
# **The tier list.** `include_tiers` declares every tier category we pull from the EXBs. The preflight cell below lists *all* tier categories present across your files — copy interesting ones from there into this list and re-run.

# %%
from dataclasses import dataclass, field

@dataclass
class Config:
    # -- Inputs --------------------------------------------------------------
    # The ROG.zip unpacks into a triple-nested ROG/ROG/ROG/ROG-Art/EXB tree.
    # (Yes, three levels of "ROG". That's just how the archive is built.)
    exb_glob: str = "data/unpacked/ROG/ROG/ROG/ROG-Art/EXB/*.exb"
    # Source WAVs (44.1 kHz mono in ROG-Art).
    source_wav_glob: str = "data/unpacked/ROG/ROG-Art.wav/ROG/ROG-Art/WAV/*.wav"

    # -- Tiers we extract ----------------------------------------------------
    # The preflight cell lists every tier category present in your EXBs.
    # Add interesting categories here and re-run.
    include_tiers: list = field(default_factory=lambda: [
        # Text tiers (stitched onto colloq spans):
        "colloq", "norm",
        # Dialogue acts (aligned on colloq spans):
        "dialogueActsIsoDimension", "dialogueActsIsoFunction",
        # Sentiment (overlap tiers — contain one or more colloq spans):
        "sentimentCurated", "sentimentAnnotated",
        # Disfluency events (sub-second, do NOT align with colloq):
        "vocalDisfluency",
    ])

    # -- Instance-level: which vocalDisfluency values flag presence on a colloq segment
    # A colloq segment gets filled_pause_present=1 if any vocalDisfluency event
    # whose `text` is in this set overlaps the segment in time.
    disfluency_instance_targets: list = field(default_factory=lambda: ["filledPause"])

    # -- Frame-level: which vocalDisfluency value(s) become class 1 in the frame sequence
    # Key is the label name written into labels.<key>; value is the set of
    # vocalDisfluency event texts that mark class 1.
    frame_label_targets: dict = field(default_factory=lambda: {
        "filled_pause": ["filledPause"],
        # Add others later, e.g. "any_disfluency": ["filledPause", "silentPause", "lengthening"]
    })

    # -- Frame rate (locked at 50 Hz; see chapter 4 README for why) ----------
    frame_rate_hz: int = 50

    # -- Output paths --------------------------------------------------------
    output_instance_jsonl: str = "data/processed_jsonl/rog_instance.raw.jsonl"
    output_frame_jsonl:    str = "data/processed_jsonl/rog_frame.raw.jsonl"

    # Where to write the 16 kHz mono normalized full-file WAVs (for chapter 4).
    # The frame JSONL's audio_path points here.
    frame_audio_subdir: str = "data/processed_audio/ROG-Art-Full"

    # Where the per-instance cut WAVs will eventually live (audio_splitter writes here).
    processed_audio_subdir: str = "data/processed_audio/ROG-Art"

    # -- Cleaning ------------------------------------------------------------
    min_text_chars: int = 2
    timestamp_tolerance: float = 0.001

    # -- Splits --------------------------------------------------------------
    split_ratios: tuple = (0.8, 0.1, 0.1)

    # -- Audio normalize-and-copy (full-file WAVs for frame model) -----------
    target_sample_rate: int = 16000
    skip_existing_wavs: bool = True   # if a normalized WAV already exists, skip re-encoding

    # -- Test mode -----------------------------------------------------------
    test_mode: bool      = False                                                                 ############ TEST MODE
    test_n_files: int    = 2
    # In test mode, only the first test_max_seconds of audio are kept for the
    # frame output (labels truncated, normalized WAV truncated). Real runs use
    # the full file. Set to None to disable the cap even in test mode.
    test_max_seconds: float | None = 5.0


cfg = Config()
print(cfg)

# %% [markdown]
# TODO: fix test mode: does only first cut of the first two audios, then the full run fails on them because they "exist"

# %% [markdown]
# ---
#
# ## 2. Find input EXB files
#
# Glob the unpacked directory. If you get zero hits, either the ROG ZIP isn't unpacked yet, or the glob doesn't match your layout — adjust `cfg.exb_glob`.

# %%
exb_files = sorted(PROJECT_ROOT.glob(cfg.exb_glob))
print(f"Found {len(exb_files)} EXB files matching {cfg.exb_glob}")
if cfg.test_mode:
    exb_files = exb_files[: cfg.test_n_files]
    print(f"🧪 TEST MODE: capped to {len(exb_files)} files")

for p in exb_files[:5]:
    print(f"  {p.relative_to(PROJECT_ROOT)}")
if len(exb_files) > 5:
    print(f"  ... and {len(exb_files) - 5} more")

if not exb_files:
    raise FileNotFoundError(
        f"No EXB files found at {PROJECT_ROOT / cfg.exb_glob}. "
        "Run download_data.ipynb with dataset='ROG' first, then check the glob path."
    )

# %% [markdown]
# ---
#
# ## 3. Preflight — what tier categories exist?
#
# Scan every EXB and report which tier categories appear where (and in which subcorpus). **This is the source of truth for `cfg.include_tiers`** — if you spot a category here you'd like to keep, add it to the Config and re-run.
#
# For ROG: `vocalDisfluency` is the tier we care about for filled-pause work. Its event texts are values like `filledPause`, `silentPause`, `lengthening`.

# %%
from lxml import etree
from collections import Counter, defaultdict

def _quick_subcorpus(doc):
    for ud in doc.findall(".//meta-information//ud-information"):
        if ud.attrib.get("attribute-name") == "SUBCORPUS":
            return ud.text
    return "unknown"


category_to_subcorpora = defaultdict(Counter)
subcorpus_counts = Counter()
files_with_category = Counter()
disfluency_value_counter = Counter()   # specific to vocalDisfluency: what values do we see?

for exb_path in exb_files:
    try:
        doc = etree.parse(str(exb_path))
    except Exception as e:
        print(f"⚠️  could not parse {exb_path.name}: {e}")
        continue
    sub = _quick_subcorpus(doc)
    subcorpus_counts[sub] += 1
    cats_here = set()
    for tier in doc.findall(".//tier"):
        cat = tier.attrib.get("category")
        if cat is None:
            continue
        cats_here.add(cat)
        # Snoop on vocalDisfluency values across the corpus
        if cat == "vocalDisfluency":
            for ev in tier.findall("event"):
                v = (ev.text or "").strip()
                if v:
                    disfluency_value_counter[v] += 1
    for cat in cats_here:
        files_with_category[cat] += 1
        category_to_subcorpora[cat][sub] += 1

print(f"Scanned {len(exb_files)} files\n")

print(f"Subcorpora seen:")
for sub, n in subcorpus_counts.most_common():
    print(f"  {sub:30s} {n} files")

print(f"\nAll tier categories (sorted by file count, descending):")
print(f"  {'category':35s}  files   in_subcorpora")
wanted = set(cfg.include_tiers)
for cat, n in files_with_category.most_common():
    subs = ", ".join(f"{s}={k}" for s, k in category_to_subcorpora[cat].most_common())
    flag = "  ←  in include_tiers" if cat in wanted else ""
    print(f"  {cat:35s}  {n:5d}   {subs}{flag}")

missing = [c for c in cfg.include_tiers if files_with_category[c] == 0]
if missing:
    print(f"\n⚠️  Categories in include_tiers MISSING from every scanned file:")
    for c in missing:
        print(f"     {c}")

if disfluency_value_counter:
    print(f"\nvocalDisfluency event-text distribution:")
    for v, n in disfluency_value_counter.most_common():
        flag = "  ←  in disfluency_instance_targets" if v in cfg.disfluency_instance_targets else ""
        flag += "  ←  in frame_label_targets" if any(v in vs for vs in cfg.frame_label_targets.values()) else ""
        print(f"  {v:25s} {n:6d}{flag}")

# %% [markdown]
# ---
#
# ## 4. EXB parsing primitives
#
# ROG-specific helpers; if a second EXB-based dataset shows up later, we lift them out into a shared module. Until then they live here.
#
# Structure recap:
# - `<common-timeline>` — list of `<tli id="..." time="seconds"/>`. Timeline IDs are mixed style (some look like `<file>.tNNN.wM`, some like `TN`); all that matters is that we map id → seconds.
# - One or more `<tier>` blocks per `(speaker, category)`. Each contains `<event start="..." end="...">text</event>` items.
# - Most ROG-Art files are single-speaker. ROG-Dialog has multiple — we union across speakers downstream.

# %%
from lxml import etree
from collections import Counter

def get_timeline(doc):
    """Map TLI id → time in seconds."""
    return {
        tli.attrib["id"]: float(tli.attrib["time"])
        for tli in doc.findall(".//tli")
        if "time" in tli.attrib
    }


def list_tier_categories(doc):
    return Counter(t.attrib.get("category", "<none>") for t in doc.findall(".//tier"))


def extract_tier_intervals(doc, timeline, category):
    """All events for tiers with the given category, with timestamps resolved."""
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
    out = {}
    for sp in doc.findall(".//speaker"):
        sid = sp.attrib.get("id")
        entry = {
            "sex":          sp.find("sex").attrib.get("value") if sp.find("sex") is not None else None,
            "abbreviation": sp.findtext("abbreviation"),
        }
        for ud in sp.findall(".//ud-information"):
            name = ud.attrib.get("attribute-name")
            if name:
                entry[name] = ud.text
        out[sid] = entry
    return out


def get_subcorpus(doc):
    for ud in doc.findall(".//meta-information//ud-information"):
        if ud.attrib.get("attribute-name") == "SUBCORPUS":
            return ud.text
    return None


def get_referenced_wav(doc) -> str | None:
    """Pull the `<referenced-file url=...>` value (relative path to the source WAV)."""
    ref = doc.find(".//meta-information/referenced-file")
    if ref is None:
        return None
    return ref.attrib.get("url")


def parse_exb(exb_path: Path, include_tiers: list[str]):
    """Parse one EXB. Returns dict with everything downstream cells need."""
    doc = etree.parse(str(exb_path))
    return {
        "file_id":   exb_path.stem,
        "subcorpus": get_subcorpus(doc),
        "speakers":  extract_speaker_metadata(doc),
        "wav_ref":   get_referenced_wav(doc),
        "all_cats":  list_tier_categories(doc),
        "tiers":     {
            cat: extract_tier_intervals(doc, get_timeline(doc), cat)
            for cat in include_tiers
        },
    }


# Smoke test on file #0
if exb_files:
    parsed = parse_exb(exb_files[0], cfg.include_tiers)
    print(f"file_id   = {parsed['file_id']}")
    print(f"subcorpus = {parsed['subcorpus']}")
    print(f"speakers  = {list(parsed['speakers'].keys())}")
    print(f"wav_ref   = {parsed['wav_ref']}")
    print(f"\nIncluded tiers — interval counts:")
    for cat, intervals in parsed["tiers"].items():
        print(f"  {cat:35s} {len(intervals):5d} intervals")


# %% [markdown]
# ---
#
# ## 5. Stitching tiers into per-segment records (instance flavor)
#
# ROG's tiers split into two groups:
#
# 1. **Aligned tiers** (`colloq`, `norm`, `dialogueActsIso*`) — share exact spans for the same speaker. Join on `(start_t, end_t, speaker)`.
# 2. **Overlap tiers** (`sentimentCurated`, `sentimentAnnotated`, `vocalDisfluency`) — have their own spans that contain or overlap colloq spans.
#
# Result: one record per colloq segment with every available label attached. **New here**: `filled_pause_present` (binary, derived from `vocalDisfluency` overlap with `cfg.disfluency_instance_targets`).

# %%
def _make_key(interval):
    s, e, sp = interval.get("start_t"), interval.get("end_t"), interval.get("speaker")
    if s is None or e is None:
        return None
    return (round(s, 3), round(e, 3), sp)


def _find_containing_sentiment(seg, sentiment_intervals, tol):
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


def _disfluency_events_overlapping(seg, disfluency_intervals, targets, tol):
    """Return list of disfluency events that overlap this segment in time AND
    whose text is in `targets`. Same-speaker by default — for multi-speaker
    files we still attribute to the colloq's speaker."""
    s, e = seg["start_t"], seg["end_t"]
    sp = seg["speaker"]
    hits = []
    for d in disfluency_intervals:
        if d["speaker"] != sp:
            continue
        if d.get("text") not in targets:
            continue
        ds, de = d.get("start_t"), d.get("end_t")
        if ds is None or de is None:
            continue
        # Any overlap counts
        if de + tol >= s and ds - tol <= e:
            hits.append(d)
    return hits


def stitch_instance_segments(parsed: dict, cfg: Config) -> list[dict]:
    """Build one record per colloq segment."""
    file_id = parsed["file_id"]
    speakers = parsed["speakers"]
    tiers = parsed["tiers"]
    tol = cfg.timestamp_tolerance

    norm_lookup = {k: v for v in tiers.get("norm", []) if (k := _make_key(v))}
    dim_lookup  = {k: v for v in tiers.get("dialogueActsIsoDimension", []) if (k := _make_key(v))}
    func_lookup = {k: v for v in tiers.get("dialogueActsIsoFunction", []) if (k := _make_key(v))}

    sent_curated   = tiers.get("sentimentCurated", [])
    sent_annotated = tiers.get("sentimentAnnotated", [])
    disfluency     = tiers.get("vocalDisfluency", [])

    out = []
    for colloq in tiers.get("colloq", []):
        key = _make_key(colloq)
        if key is None:
            continue

        norm = norm_lookup.get(key)
        dim  = dim_lookup.get(key)
        func = func_lookup.get(key)

        # Filled-pause presence (any disfluency event whose text matches targets
        # that overlaps this colloq's time span — same speaker).
        fp_events = _disfluency_events_overlapping(
            colloq, disfluency, set(cfg.disfluency_instance_targets), tol,
        )

        out.append({
            "file_id":   file_id,
            "subcorpus": parsed["subcorpus"],
            "speaker":   colloq["speaker"],
            "start_t":   colloq["start_t"],
            "end_t":     colloq["end_t"],
            "colloq_text": colloq["text"],
            "norm_text":   norm["text"] if norm else None,
            "dialogue_act_dimension": (dim["text"] if dim else None) or None,
            "dialogue_act_function":  (func["text"] if func else None) or None,
            "sentiment_curated":   _find_containing_sentiment(colloq, sent_curated, tol),
            "sentiment_annotated": _find_containing_sentiment(colloq, sent_annotated, tol),
            "filled_pause_present": int(len(fp_events) > 0),
            "disfluency_count":     len(fp_events),
            "speaker_metadata":     speakers.get(colloq["speaker"], {}),
        })
    return out


if exb_files:
    segs = stitch_instance_segments(parsed, cfg)
    print(f"Stitched {len(segs)} instance segments from {parsed['file_id']}")
    if segs:
        import json as _json
        print("\nFirst segment:")
        print(_json.dumps(segs[0], ensure_ascii=False, indent=2))

# %% [markdown]
# ---
#
# ## 6. Building the frame-level label sequence
#
# For each EXB we produce **one frame record per file**, with a per-frame label sequence at `cfg.frame_rate_hz` (default 50 Hz) covering the whole audio. We union across speakers — frame *t* is 1 if *any* speaker has a matching disfluency event spanning that frame.
#
# Mapping a continuous span `[start_t, end_t]` to discrete frames: frame *t* covers `[t/50, (t+1)/50)` seconds. We mark frame *t* as 1 if the span has any overlap with that frame's time window. Robust to off-by-one at boundaries.
#
# The frame record carries `frame_rate_hz` so chapter 4's hard-fail validator passes.

# %%
import math


def file_duration_from_timeline(doc) -> float | None:
    """Use the maximum timeline time as the file's duration. EXB writers
    typically include a terminal TLI at the audio's end."""
    times = [float(t.attrib["time"]) for t in doc.findall(".//tli") if "time" in t.attrib]
    return max(times) if times else None


def spans_to_frame_sequence(spans: list[tuple[float, float]],
                            duration_s: float, fps: int) -> list[int]:
    """Convert a list of (start_s, end_s) spans to a 0/1 sequence of length
    floor(duration * fps). Frame t covers [t/fps, (t+1)/fps)."""
    n_frames = int(math.floor(duration_s * fps))
    seq = [0] * n_frames
    for s, e in spans:
        if e <= s:
            continue
        # Frame indices whose window [j/fps, (j+1)/fps) intersects [s, e].
        j_lo = max(0, int(math.floor(s * fps)))
        j_hi = min(n_frames, int(math.ceil(e * fps)))
        for j in range(j_lo, j_hi):
            seq[j] = 1
    return seq


def build_frame_record(parsed: dict, exb_path: Path, cfg: Config) -> dict | None:
    """Build the per-file frame record. Returns None if no audio duration
    can be determined or no disfluency events are present."""
    doc = etree.parse(str(exb_path))
    duration_s = file_duration_from_timeline(doc)
    if duration_s is None or duration_s <= 0:
        print(f"⚠️  {parsed['file_id']}: cannot determine duration; skipping frame record")
        return None

    # Apply test-mode duration cap
    if cfg.test_mode and cfg.test_max_seconds is not None:
        duration_s = min(duration_s, float(cfg.test_max_seconds))

    disfluency = parsed["tiers"].get("vocalDisfluency", [])
    if not disfluency:
        # File has no vocalDisfluency tier (or no events). Still emit the
        # record — labels become all-zero, which is valid signal.
        pass

    labels = {}
    for label_name, target_values in cfg.frame_label_targets.items():
        target_set = set(target_values)
        spans = []
        for d in disfluency:
            if d.get("text") not in target_set:
                continue
            ds, de = d.get("start_t"), d.get("end_t")
            if ds is None or de is None:
                continue
            spans.append((ds, de))
        labels[label_name] = spans_to_frame_sequence(spans, duration_s, cfg.frame_rate_hz)

    # Note: audio_path is set to the future normalized-WAV location.
    # The final cell of this notebook actually creates that WAV.
    normalized_wav_name = f"{parsed['file_id']}.wav"
    audio_path = f"{cfg.frame_audio_subdir}/{normalized_wav_name}"

    iid = udp.make_instance_id("ROG-Art", parsed["file_id"], None, 0.0, duration_s)

    rec = {
        "instance_id":  iid,
        "dataset":      "ROG-Art",
        "file_id":      parsed["file_id"],
        "audio_path":   audio_path,
        "split":        "train",                  # set later by udp.assign_splits
        "start_t":      0.0,
        "end_t":        round(duration_s, 3),
        "frame_rate_hz": cfg.frame_rate_hz,
        "labels":       labels,
        "metadata": {
            "source_file":   exb_path.name,
            "source_format": "EXB",
            "subcorpus":     parsed["subcorpus"],
            "n_frames":      {k: len(v) for k, v in labels.items()},
            "n_speakers":    len(parsed["speakers"]),
        },
    }
    return rec


# Smoke test on file #0
if exb_files:
    frame_rec = build_frame_record(parsed, exb_files[0], cfg)
    if frame_rec is not None:
        print(f"frame record for {parsed['file_id']}:")
        print(f"  duration = {frame_rec['end_t']}s")
        print(f"  frame_rate_hz = {frame_rec['frame_rate_hz']}")
        for k, seq in frame_rec["labels"].items():
            n_pos = sum(seq)
            print(f"  labels.{k}: len={len(seq)}, positive_frames={n_pos} ({100*n_pos/max(1,len(seq)):.1f}%)")


# %% [markdown]
# ---
#
# ## 7. Map instance segments to canonical JSONL
#
# Take stitched segments and emit canonical-JSONL dicts. New label keys this version: `filled_pause_present`, `disfluency_count`.

# %%
def to_canonical_instance(segment: dict, cfg: Config) -> dict:
    speaker = segment["speaker"]
    file_id = segment["file_id"]
    st, et = segment["start_t"], segment["end_t"]

    iid = udp.make_instance_id("ROG-Art", file_id, speaker, st, et)

    cut_name = f"{file_id}_{speaker}_{st:.3f}_{et:.3f}.wav"
    audio_path = f"{cfg.processed_audio_subdir}/{cut_name}"

    text = segment.get("norm_text") or segment.get("colloq_text") or ""

    labels = {}
    if segment.get("sentiment_curated"):
        labels["sentiment"] = segment["sentiment_curated"]
    elif segment.get("sentiment_annotated"):
        labels["sentiment"] = segment["sentiment_annotated"]
    if segment.get("sentiment_annotated"):
        labels["sentiment_annotated"] = segment["sentiment_annotated"]
    if segment.get("dialogue_act_function"):
        labels["dialogue_act_function"] = segment["dialogue_act_function"]
    if segment.get("dialogue_act_dimension"):
        labels["dialogue_act_dimension"] = segment["dialogue_act_dimension"]

    # New disfluency labels
    labels["filled_pause_present"] = segment["filled_pause_present"]
    labels["disfluency_count"]     = segment["disfluency_count"]

    metadata = {
        "source_file":   f"{file_id}.exb",
        "source_format": "EXB",
        "subcorpus":     segment.get("subcorpus"),
        "speaker_info":  segment.get("speaker_metadata", {}),
        "colloq_text":   segment.get("colloq_text"),
    }

    return {
        "instance_id": iid,
        "dataset":     "ROG-Art",
        "file_id":     file_id,
        "audio_path":  audio_path,
        "split":       "train",
        "speaker":     speaker,
        "start_t":     st,
        "end_t":       et,
        "text":        text,
        "labels":      labels,
        "metadata":    metadata,
    }


if exb_files and segs:
    sample = to_canonical_instance(segs[0], cfg)
    errs = udp.validate_instance(sample)
    if errs:
        print("⚠️  validation errors on sample:")
        for e in errs: print(f"   - {e}")
    else:
        print("✅ sample validates against canonical schema")
    import json as _json
    print(_json.dumps(sample, ensure_ascii=False, indent=2))

# %% [markdown]
# ---
#
# ## 8. Run over all files (single EXB pass, two outputs)
#
# Process every EXB once. From each file we emit:
# - N instance records (one per colloq segment) → `instance_records`
# - 1 frame record (or 0 if no duration could be determined) → `frame_records`

# %%
instance_records: list[dict] = []
frame_records: list[dict] = []
file_stats = []

for exb_path in exb_files:
    try:
        parsed = parse_exb(exb_path, cfg.include_tiers)
    except Exception as e:
        print(f"❌ parse failed for {exb_path.name}: {e}")
        continue

    # Instance
    try:
        segs = stitch_instance_segments(parsed, cfg)
        canon = [to_canonical_instance(s, cfg) for s in segs]
        instance_records.extend(canon)
    except Exception as e:
        print(f"❌ instance stitch failed for {exb_path.name}: {e}")
        canon = []

    # Frame
    try:
        fr = build_frame_record(parsed, exb_path, cfg)
        if fr is not None:
            frame_records.append(fr)
    except Exception as e:
        print(f"❌ frame build failed for {exb_path.name}: {e}")
        fr = None

    n_fp_instance = sum(r["labels"].get("filled_pause_present", 0) for r in canon)
    fr_n_pos = sum(fr["labels"]["filled_pause"]) if fr else 0
    fr_n_tot = len(fr["labels"]["filled_pause"]) if fr else 0
    file_stats.append({
        "file_id": parsed["file_id"],
        "n_instance_records": len(canon),
        "n_with_fp": n_fp_instance,
        "frame_pos": fr_n_pos,
        "frame_tot": fr_n_tot,
    })
    print(f"  {parsed['file_id']:40s} "
          f"inst={len(canon):4d} (fp={n_fp_instance:3d})  "
          f"frame={fr_n_pos:5d}/{fr_n_tot:5d} pos ({100*fr_n_pos/max(1,fr_n_tot):.1f}%)")

print(f"\nTotals: {len(instance_records)} instance records, {len(frame_records)} frame records")

# %% [markdown]
# ---
#
# ## 9. Clean + validate
#
# `udp.clean` drops invalid records, dedupes by `instance_id`, drops empty-text records (instance only — frame records have no `text`).
#
# Then a strict full-validation pass for visibility.

# %%
# Instance: standard cleaning
instance_records = udp.clean(instance_records, require_text=True, verbose=True)
before = len(instance_records)
instance_records = [r for r in instance_records if len(r.get("text", "")) >= cfg.min_text_chars]
print(f"  min_text_chars={cfg.min_text_chars}: -{before - len(instance_records)} ({len(instance_records)} left)")

# Frame: no text requirement; dedupe by instance_id manually
seen_ids = set()
deduped = []
for r in frame_records:
    if r["instance_id"] in seen_ids:
        continue
    seen_ids.add(r["instance_id"])
    deduped.append(r)
print(f"\nFrame records: {len(frame_records)} → {len(deduped)} after dedupe")
frame_records = deduped

# Validate both
print("\nInstance validation:")
n_tot, n_val, errs = udp.validate_jsonl(instance_records, max_report=5)
print(f"  {n_val}/{n_tot} valid")
for e in errs: print(f"    - {e}")

print("Frame validation:")
n_tot, n_val, errs = udp.validate_jsonl(frame_records, max_report=5)
print(f"  {n_val}/{n_tot} valid")
for e in errs: print(f"    - {e}")

# %% [markdown]
# ---
#
# ## 10. Assign splits — grouped by `file_id` so no recording leaks
#
# Both JSONLs use the same `file_id`-based grouping, with the **same seed**. So a given source file ends up in the same split across both flavors — handy for sanity-checking downstream.

# %%
SPLIT_SEED = "rog-art-prep-v1"

udp.assign_splits(
    instance_records,
    ratios=cfg.split_ratios, group_key="file_id",
    overwrite=True, seed=SPLIT_SEED,
)
udp.assign_splits(
    frame_records,
    ratios=cfg.split_ratios, group_key="file_id",
    overwrite=True, seed=SPLIT_SEED,
)

print("Instance split counts:", udp.split_summary(instance_records))
print("Frame split counts:   ", udp.split_summary(frame_records))

# Sanity: same file_id → same split, in both JSONLs
def assert_no_leak(records, name):
    groups = {}
    for r in records:
        groups.setdefault(r["file_id"], set()).add(r["split"])
    leaks = {f: s for f, s in groups.items() if len(s) > 1}
    assert not leaks, f"{name} split leakage: {leaks}"

assert_no_leak(instance_records, "instance")
assert_no_leak(frame_records,    "frame")
print("✅ no split leakage in either JSONL")

# Cross-check: file_id splits agree between instance and frame
inst_split = {r["file_id"]: r["split"] for r in instance_records}
frame_split = {r["file_id"]: r["split"] for r in frame_records}
for fid in inst_split:
    if fid in frame_split and inst_split[fid] != frame_split[fid]:
        print(f"⚠️  {fid}: instance={inst_split[fid]}, frame={frame_split[fid]} — should not happen")
print("✅ instance and frame splits agree on shared file_ids")

# %% [markdown]
# ---
#
# ## 11. Label distribution peek
#
# Just so we can see what we've got. Real sniffing happens in chapter 2.

# %%
from collections import Counter

def show_label(records, key):
    vals = [r["labels"].get(key) for r in records if key in r["labels"]]
    if not vals:
        print(f"\n{key}: (no records carry this label)")
        return
    c = Counter(vals)
    print(f"\n{key} ({len(vals)} records):")
    mx = max(c.values())
    for k, v in c.most_common():
        bar = "█" * min(40, v * 40 // max(1, mx))
        print(f"  {str(k):35s} {v:5d}  {bar}")


print("--- Instance labels ---")
for key in ["sentiment", "dialogue_act_function", "dialogue_act_dimension", "filled_pause_present"]:
    show_label(instance_records, key)

print("\n--- Frame label totals (positive-frame counts per file) ---")
for key in cfg.frame_label_targets:
    totals = []
    for r in frame_records:
        seq = r["labels"].get(key, [])
        totals.append(sum(seq))
    if totals:
        print(f"  {key}: per-file positive frames: min={min(totals)}, "
              f"max={max(totals)}, sum={sum(totals)} (across {len(totals)} files)")


# %% [markdown]
# ---
#
# ## 12. Write both JSONLs
#
# Test mode prefixes outputs with `test_` so we don't pollute real outputs.

# %%
def maybe_test_prefix(path: Path, test_mode: bool) -> Path:
    return path.with_name("test_" + path.name) if test_mode else path


inst_out  = maybe_test_prefix(PROJECT_ROOT / cfg.output_instance_jsonl, cfg.test_mode)
frame_out = maybe_test_prefix(PROJECT_ROOT / cfg.output_frame_jsonl,    cfg.test_mode)

n_inst  = udp.write_jsonl(instance_records, inst_out)
n_frame = udp.write_jsonl(frame_records,    frame_out)

print(f"✅ wrote {n_inst:6d} instance records → {inst_out.relative_to(PROJECT_ROOT)}")
print(f"✅ wrote {n_frame:6d} frame records    → {frame_out.relative_to(PROJECT_ROOT)}")

# Quick roundtrip
assert len(udp.read_jsonl(inst_out))  == n_inst
assert len(udp.read_jsonl(frame_out)) == n_frame
print("✅ roundtrip OK for both")

# %% [markdown]
# ---
#
# ## 13. Normalize source WAVs to 16 kHz mono (one-time, for the frame model)
#
# Chapter 4 reads the WAV pointed at by each frame record's `audio_path`. We do the 44.1 → 16 kHz mono conversion **once, here**, into `data/processed_audio/ROG-Art-Full/<file_id>.wav`. Skip-if-exists by default, so re-runs are cheap.
#
# In test mode each WAV is truncated to `cfg.test_max_seconds` to keep verification fast. Real runs convert the whole file.
#
# If a WAV is already 16 kHz mono on disk (unlikely for ROG-Art, but possible), the resampler is a no-op and we still copy to keep the path layout uniform.

# %%
from pathlib import Path
import soundfile as sf

def find_source_wav(file_id: str, source_glob_pattern: str) -> Path | None:
    """The EXB's referenced-file points relatively — resolve via glob over WAVs."""
    candidates = list(PROJECT_ROOT.glob(source_glob_pattern))
    by_stem = {p.stem: p for p in candidates}
    return by_stem.get(file_id)


def normalize_wav(src: Path, dst: Path, target_sr: int,
                  max_seconds: float | None) -> tuple[float, bool]:
    """Read src (any rate, mono or stereo), optionally truncate, resample to
    target_sr mono, write dst. Returns (output_duration_s, did_convert)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)   # to mono
    if max_seconds is not None:
        data = data[: int(max_seconds * sr)]
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, target_sr)
            data = resample_poly(data, target_sr // g, sr // g).astype("float32")
    sf.write(str(dst), data, target_sr, subtype="PCM_16")
    return (len(data) / target_sr, sr != target_sr)


udp.banner(f"normalizing WAVs → {cfg.target_sample_rate} Hz mono", char="-")

max_s = float(cfg.test_max_seconds) if (cfg.test_mode and cfg.test_max_seconds is not None) else None
n_done, n_skipped, n_missing = 0, 0, 0

for r in frame_records:
    file_id = r["file_id"]
    dst = PROJECT_ROOT / cfg.frame_audio_subdir / f"{file_id}.wav"
    if cfg.skip_existing_wavs and dst.exists():
        n_skipped += 1
        continue
    src = find_source_wav(file_id, cfg.source_wav_glob)
    if src is None:
        print(f"⚠️  source WAV not found for {file_id} (skipping)")
        n_missing += 1
        continue
    try:
        out_dur, did_convert = normalize_wav(src, dst, cfg.target_sample_rate, max_s)
        n_done += 1
        if n_done <= 5 or n_done % 10 == 0:
            converted_tag = "(resampled)" if did_convert else "(copy)"
            print(f"  {file_id:40s} {out_dur:6.1f}s {converted_tag} → {dst.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"❌ {file_id}: {e}")

print(f"\nDone. converted={n_done}  skipped_existing={n_skipped}  missing_source={n_missing}")

# %% [markdown]
# ---
#
# ## 14. Sanity: frame label length matches the normalized WAV duration
#
# Per-record check that the label sequence length matches the audio duration to within one frame. If anything's off here, chapter 4 will hard-fail on the same record — better to catch it now.
#
# (In test mode we capped both audio and labels to `test_max_seconds`, so they line up by construction.)

# %%
import soundfile as sf

mismatches = []
for r in frame_records:
    audio_p = PROJECT_ROOT / r["audio_path"]
    if not audio_p.exists():
        continue   # normalize step may have skipped it
    info = sf.info(str(audio_p))
    audio_frames = int(info.duration * cfg.frame_rate_hz)
    for key, seq in r["labels"].items():
        if abs(len(seq) - audio_frames) > 1:
            mismatches.append((r["file_id"], key, len(seq), audio_frames))

if not mismatches:
    print("✅ frame-label lengths match normalized WAV durations (±1 frame)")
else:
    print(f"⚠️  {len(mismatches)} mismatch(es):")
    for fid, key, l_seq, l_aud in mismatches[:10]:
        print(f"  {fid}  {key}: labels={l_seq}  audio_frames={l_aud}")

# %% [markdown]
# ---
#
# ## Next
#
# - **Chapter 2** — point `sniff_dataset.ipynb` at either JSONL to look at distributions.
# - **Chapter 3 (instance models)** — load `rog_instance.raw.jsonl` (well, the cut version after running `audio_splitter.ipynb`), set `label_key="filled_pause_present"` or `"sentiment"` or whatever.
# - **Chapter 4 (frame model)** — load `rog_frame.raw.jsonl`, set `label_key="filled_pause"`. The frame WAVs created in cell 13 are at the path each record's `audio_path` points at, so chapter 4 reads them directly with no further prep.
#
# Currently the audio splitter (`audio_splitter.ipynb`) is configured only for ROG-Dialog. Generalizing it to ROG-Art is a small follow-up before training instance models on this corpus.
