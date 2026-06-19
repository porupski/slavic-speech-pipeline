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
# # Explore dataset — chapter 2
#
# Point at a canonical instance JSONL produced by chapter 1, get back a
# data-science-style sanity report: record counts, split balance, speaker
# coverage, audio-duration distribution, per-label distributions, missing
# fields. The notebook prints every result inline; at the end it stitches the
# tables and plots into a single markdown file under `data/reports/`.
#
# **Input.** One JSONL (`data/processed_jsonl/...`). Instance-shape only —
# frame JSONLs have per-record sequence labels that need their own treatment;
# this notebook prints a clear skip message and stops if it sees one.
#
# **Output.**
# - Cell-by-cell tables and figures inline.
# - PNGs and one consolidated `<stem>_report.md` next to each other under
#   `data/reports/` (or `data/test_reports/` in test mode).
#
# **What to look for.** A balanced split, no speaker leakage, audio durations
# inside the trainer's `max_duration_s`, label distributions you can live with,
# zero missing required fields. If any of those look off the fix lives in
# chapter 1 — re-run prep with adjusted knobs.

# %% [markdown]
# ## 0. Setup
#
# Locate `PROJECT_ROOT` via chapter 1's `utils_dataprep`, then bring in the
# data-science imports. `matplotlib` is a runtime dep from chapter 2 onwards;
# the conda envs in `0_env_setup/` already include it.

# %%
import sys
from pathlib import Path

HERE = Path.cwd()
if HERE.name != "2_data_analysis":
    candidate = HERE / "2_data_analysis"
    if candidate.exists():
        HERE = candidate
sys.path.insert(0, str(HERE.parent / "1_data_prep"))

import utils_dataprep as udp

PROJECT_ROOT = udp.PROJECT_ROOT
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"chapter dir  = {HERE}")

# %%
import wave
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["grid.alpha"] = 0.3

# %% [markdown]
# ## 1. Config
#
# One dataclass at the top, no buried constants. Change `jsonl_path` to point
# at any canonical instance JSONL; the rest are display knobs.
#
# - `imbalance_max_ratio` — warn when max-class / min-class exceeds this.
# - `imbalance_rare_pct` — warn on any class below this percentage of its split.
# - `max_duration_s` — the trainer's default cap; the duration section reports
#   how many records would be dropped at this length.
# - `top_n_speakers` / `top_n_file_ids` — how many to show in the bar charts.
# - `test_mode` — load only the first `test_n` records and write to
#   `data/test_reports/`. Useful while iterating on the notebook itself.

# %%
@dataclass
class Config:
    jsonl_path: str = "data/processed_jsonl/parlaspeech_hr_utterance_instance.jsonl"
    output_dir: str = "data/reports"

    imbalance_max_ratio: float = 20.0
    imbalance_rare_pct:  float = 1.0

    max_duration_s: float = 15.0

    top_n_speakers: int = 15
    top_n_file_ids: int = 10
    duration_hist_bins: int = 40
    numeric_hist_bins:  int = 30

    test_mode: bool = False
    test_n:    int  = 200


cfg = Config()
print(cfg)
if cfg.test_mode:
    udp.banner("TEST MODE", char="-")

# %% [markdown]
# ## 2. Load and validate
#
# Read the JSONL, run it through the canonical-schema validator from chapter 1.
# Any structural problem is a chapter-1 bug — fail loudly here rather than make
# the plots lie. Then split into instance vs frame records: this chapter only
# handles instance shape, so if the file is frame-only we print a clear note
# and stop.

# %%
records = udp.read_jsonl(cfg.jsonl_path)
n_total, n_valid, errs = udp.validate_jsonl(records)
print(f"loaded {n_total} records, {n_valid} valid")
if errs:
    print("first validation errors:")
    for e in errs[:5]:
        print(f"  {e}")
    raise ValueError("input JSONL failed canonical-schema validation; fix in chapter 1")

if cfg.test_mode:
    records = records[: cfg.test_n]
    print(f"capped to {len(records)} records (test mode)")

out_dir_str = (cfg.output_dir if not cfg.test_mode
               else cfg.output_dir.replace("data/reports", "data/test_reports"))
out_dir = udp.from_project_relative(out_dir_str)
out_dir.mkdir(parents=True, exist_ok=True)

jsonl_stem = Path(cfg.jsonl_path).stem
print(f"output dir = {out_dir}")
print(f"stem       = {jsonl_stem}")

# %%
def _is_frame_record(r: dict) -> bool:
    return any(isinstance(v, list) for v in r.get("labels", {}).values())

instance_records = [r for r in records if not _is_frame_record(r)]
frame_count = len(records) - len(instance_records)

if frame_count and not instance_records:
    print(f"this JSONL is frame-shape ({frame_count} records). "
          "Use a frame-aware explorer; this notebook handles instance JSONLs.")
    raise SystemExit(0)
if frame_count:
    print(f"note: {frame_count} frame-shape records present — skipped. "
          f"Continuing with {len(instance_records)} instance records.")

records = instance_records
SPLITS = ["train", "dev", "test"]

# %% [markdown]
# ## 3. Identity & shape
#
# What is this file. The dataset tag (`r["dataset"]`) and the label keys present
# tell you which corpus you loaded and which targets the trainer will see.

# %%
datasets = Counter(r["dataset"] for r in records)
label_keys_present = sorted({k for r in records for k in r.get("labels", {})})

print(f"records          : {len(records)}")
print(f"datasets         : {dict(datasets)}")
print(f"label keys       : {label_keys_present}")
print(f"first instance_id: {records[0]['instance_id']}")
print(f"first audio_path : {records[0]['audio_path']}")

# %% [markdown]
# ## 4. Splits
#
# Sizes per split + percent of total. Anything zero is a prep bug. A wildly
# skewed test split usually means the speaker grouping put too many records
# into one bucket.

# %%
split_counts = Counter(r["split"] for r in records)

print(f"{'split':<8}{'count':>10}{'pct':>10}")
print("-" * 28)
for s in SPLITS:
    n = split_counts.get(s, 0)
    pct = 100 * n / max(1, len(records))
    print(f"{s:<8}{n:>10}{pct:>9.1f}%")
print("-" * 28)
print(f"{'total':<8}{len(records):>10}")

# Bar chart of split sizes — a quick visual that the eye picks up faster than a
# table when something is wildly off.
fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
ax.bar(SPLITS, [split_counts.get(s, 0) for s in SPLITS])
ax.set_ylabel("records")
ax.set_title("records per split")
split_png = out_dir / f"{jsonl_stem}_splits.png"
fig.savefig(split_png)
plt.show()
plt.close(fig)
print(f"saved {split_png.relative_to(PROJECT_ROOT)}")

# %% [markdown]
# ## 5. Speakers
#
# Three things to check:
#
# 1. **Unique speakers per split.** A single-speaker dev or test split makes
#    the metric unstable.
# 2. **Cross-split overlap.** Speaker-grouped splits should give zero overlap
#    for ParlaSpeech; ROG uses `file_id` grouping, so some speaker overlap
#    across splits is expected for multi-speaker recordings.
# 3. **Top speakers by utterance count.** A few speakers dominating skews
#    everything downstream.

# %%
speakers_by_split: dict[str, set] = {s: set() for s in SPLITS}
for r in records:
    if "speaker" in r:
        speakers_by_split.setdefault(r["split"], set()).add(r["speaker"])

print("unique speakers per split:")
for s in SPLITS:
    print(f"  {s:<6}: {len(speakers_by_split[s])}")

print("\ncross-split overlap (speakers in both):")
for i, a in enumerate(SPLITS):
    for b in SPLITS[i + 1:]:
        ov = speakers_by_split[a] & speakers_by_split[b]
        print(f"  {a} ∩ {b}: {len(ov)}")

# %%
speaker_counts_all = Counter(r["speaker"] for r in records if "speaker" in r)
top_speakers = speaker_counts_all.most_common(cfg.top_n_speakers)

print(f"\ntop {cfg.top_n_speakers} speakers by utterance count:")
print(f"{'speaker':<35}{'count':>10}{'pct':>10}")
print("-" * 55)
total_with_speaker = sum(speaker_counts_all.values()) or 1
for spk, n in top_speakers:
    print(f"{spk[:35]:<35}{n:>10}{100*n/total_with_speaker:>9.1f}%")

if top_speakers:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    names = [s for s, _ in top_speakers][::-1]   # bottom-up for horizontal bar
    vals  = [n for _, n in top_speakers][::-1]
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("records")
    ax.set_title(f"top {cfg.top_n_speakers} speakers by utterance count")
    spk_png = out_dir / f"{jsonl_stem}_top_speakers.png"
    fig.savefig(spk_png)
    plt.show()
    plt.close(fig)
    print(f"saved {spk_png.relative_to(PROJECT_ROOT)}")
else:
    spk_png = None

# %% [markdown]
# Utterance-count-per-speaker distribution: how steep the long tail is. Most
# speech corpora have a handful of speakers contributing thousands of clips
# and a long tail of one-offs.

# %%
if speaker_counts_all:
    arr = np.array(list(speaker_counts_all.values()))
    print(f"speakers total    : {len(arr)}")
    print(f"records / speaker : "
          f"min={arr.min()}  p50={int(np.percentile(arr, 50))}  "
          f"p90={int(np.percentile(arr, 90))}  p99={int(np.percentile(arr, 99))}  "
          f"max={arr.max()}")

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    ax.hist(arr, bins=40, log=True)
    ax.set_xlabel("records per speaker")
    ax.set_ylabel("speakers (log)")
    ax.set_title("utterances-per-speaker distribution")
    spk_hist_png = out_dir / f"{jsonl_stem}_speaker_hist.png"
    fig.savefig(spk_hist_png)
    plt.show()
    plt.close(fig)
    print(f"saved {spk_hist_png.relative_to(PROJECT_ROOT)}")
else:
    spk_hist_png = None
    print("no `speaker` field on records — skipping speaker distribution")

# %% [markdown]
# Top `file_id`s per split. A single source recording dominating a split makes
# everything downstream correlated; this is the place to notice.

# %%
print(f"top {cfg.top_n_file_ids} file_ids per split:")
for s in SPLITS:
    f_counts = Counter(r["file_id"] for r in records
                       if r["split"] == s and "file_id" in r)
    print(f"\n  {s}:")
    for fid, n in f_counts.most_common(cfg.top_n_file_ids):
        print(f"    {fid[:50]:<50}{n:>6}")

# %% [markdown]
# ## 6. Audio durations (from disk)
#
# Open each WAV's header to read its real on-disk duration — cheap (no decode)
# and doubles as a splitter sanity check. We compare against `end_t - start_t`
# when present; differences over 50 ms point at a stale cut. Then we report
# the distribution and count how many records exceed the trainer's
# `max_duration_s` cap (the trainer would truncate or drop those).

# %%
def _wav_duration(p: Path) -> float | None:
    try:
        with wave.open(str(p), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


durations_by_split: dict[str, list[float]] = {s: [] for s in SPLITS}
duration_mismatches: list[dict] = []
unreadable = 0

for r in tqdm(records, desc="WAV headers", unit="rec"):
    p = udp.from_project_relative(r["audio_path"])
    d = _wav_duration(p)
    if d is None:
        unreadable += 1
        continue
    durations_by_split.setdefault(r["split"], []).append(d)
    if "start_t" in r and "end_t" in r:
        expected = r["end_t"] - r["start_t"]
        if abs(d - expected) > 0.050:
            duration_mismatches.append({"instance_id": r["instance_id"],
                                        "expected_s": expected, "actual_s": d})

durations_by_split = {s: np.array(v, dtype=float) for s, v in durations_by_split.items()}

if unreadable:
    print(f"WARNING: {unreadable} WAVs could not be read — check audio_path resolution")
if duration_mismatches:
    print(f"WARNING: {len(duration_mismatches)} records: on-disk duration "
          f"differs from end_t - start_t by >50 ms. First few:")
    for m in duration_mismatches[:5]:
        print(f"  {m['instance_id']}: expected {m['expected_s']:.3f}s, "
              f"actual {m['actual_s']:.3f}s")

# %%
print(f"\n{'split':<6}{'n':>8}{'min':>8}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>10}{'>'+str(cfg.max_duration_s)+'s':>12}")
print("-" * 78)
total_over_cap = 0
for s in SPLITS:
    d = durations_by_split.get(s, np.array([]))
    if d.size == 0:
        print(f"{s:<6}{'-':>8}")
        continue
    over_cap = int((d > cfg.max_duration_s).sum())
    total_over_cap += over_cap
    print(f"{s:<6}{d.size:>8}{d.min():>8.2f}{np.percentile(d,50):>8.2f}"
          f"{np.percentile(d,90):>8.2f}{np.percentile(d,99):>8.2f}{d.max():>10.2f}{over_cap:>12}")
print("-" * 78)
print(f"records over {cfg.max_duration_s}s: {total_over_cap} "
      f"({100*total_over_cap/max(1, len(records)):.2f}% of total)")

# %%
fig, axes = plt.subplots(1, len(SPLITS), figsize=(4 * len(SPLITS), 3.5),
                         sharex=True, sharey=False, constrained_layout=True)
fig.suptitle("audio duration (seconds) — per split")
for ax, s in zip(axes, SPLITS):
    d = durations_by_split.get(s, np.array([]))
    if d.size == 0:
        ax.set_title(f"{s} (empty)")
        continue
    ax.hist(d, bins=cfg.duration_hist_bins)
    ax.axvline(np.percentile(d, 50), linestyle="--", linewidth=1, label="p50")
    ax.axvline(np.percentile(d, 90), linestyle=":",  linewidth=1, label="p90")
    ax.axvline(cfg.max_duration_s, color="red", linewidth=1, label=f"cap={cfg.max_duration_s}s")
    ax.set_title(f"{s} (n={d.size})")
    ax.set_xlabel("seconds")
    ax.legend(fontsize=7, loc="upper right")
dur_hist_png = out_dir / f"{jsonl_stem}_duration_hist.png"
fig.savefig(dur_hist_png)
plt.show()
plt.close(fig)
print(f"saved {dur_hist_png.relative_to(PROJECT_ROOT)}")

# Boxplot — quick visual of shape comparison between splits.
fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
data = [durations_by_split.get(s, np.array([])) for s in SPLITS]
labels_used = [s for s, d in zip(SPLITS, data) if d.size > 0]
data        = [d for d in data if d.size > 0]
if data:
    ax.boxplot(data, tick_labels=labels_used, showfliers=True)
    ax.axhline(cfg.max_duration_s, color="red", linewidth=1,
               label=f"max_duration_s={cfg.max_duration_s}s")
    ax.set_ylabel("seconds")
    ax.set_title("audio duration boxplot")
    ax.legend(fontsize=8)
    dur_box_png = out_dir / f"{jsonl_stem}_duration_box.png"
    fig.savefig(dur_box_png)
    plt.show()
    plt.close(fig)
    print(f"saved {dur_box_png.relative_to(PROJECT_ROOT)}")
else:
    dur_box_png = None
    plt.close(fig)

# %% [markdown]
# ## 7. Labels
#
# Auto-detect every scalar `labels.<key>` and report:
#
# - **Categorical** keys (strings or small-cardinality ints) get a per-split
#   count table and a side-by-side bar chart, plus imbalance warnings.
# - **Numeric** keys (floats, or ints with many distinct values) get
#   summary stats and a per-split histogram.
#
# Null-valued labels are counted but excluded from the distribution — the
# trainer drops nulls per-target so they cost only themselves.

# %%
def _scalar_label_keys(records: list[dict]) -> list[str]:
    keys: set[str] = set()
    for r in records:
        for k, v in r.get("labels", {}).items():
            if not isinstance(v, list):
                keys.add(k)
    return sorted(keys)


def _is_numeric_label(values: list) -> bool:
    """A label is treated as numeric if its non-null values are all numbers
    AND it shows more than 12 distinct ones — small-int label_orders like
    `filled_pause_count` should plot as categorical bars."""
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) != len([v for v in values if v is not None]):
        return False
    return len(set(nums)) > 12


def _imbalance_warnings(counts: Counter, *, max_ratio: float, rare_pct: float) -> list[str]:
    if not counts:
        return []
    total = sum(counts.values())
    items = counts.most_common()
    biggest, smallest = items[0][1], items[-1][1]
    out: list[str] = []
    if smallest > 0 and biggest / smallest > max_ratio:
        out.append(f"max/min ratio = {biggest/smallest:.1f}x (threshold {max_ratio:.0f}x)")
    for cls, n in counts.items():
        pct = 100 * n / total
        if pct < rare_pct:
            out.append(f"rare class {cls!r}: {n} ({pct:.2f}%)")
    return out


scalar_keys = _scalar_label_keys(records)
print(f"scalar label keys to plot: {scalar_keys}\n")

label_plot_paths: dict[str, Path] = {}
label_warnings:   dict[str, list[str]] = {}
label_kind:       dict[str, str] = {}        # "categorical" | "numeric"
null_counts:      dict[str, int] = {}

for key in scalar_keys:
    udp.banner(f"label: {key}", char="-")
    values_by_split: dict[str, list] = {s: [] for s in SPLITS}
    n_null = 0
    for r in records:
        v = r.get("labels", {}).get(key, None)
        if v is None:
            n_null += 1
            continue
        values_by_split[r["split"]].append(v)
    null_counts[key] = n_null
    all_values = [v for vs in values_by_split.values() for v in vs]
    if not all_values:
        print(f"  no non-null values — skipping plot")
        continue

    numeric = _is_numeric_label(all_values)
    label_kind[key] = "numeric" if numeric else "categorical"
    print(f"  kind   : {label_kind[key]}")
    print(f"  nulls  : {n_null} ({100*n_null/len(records):.2f}% of records)")

    # ---- Numeric: per-split histogram ---------------------------------------
    if numeric:
        arr_all = np.array(all_values, dtype=float)
        print(f"  values : n={arr_all.size}  min={arr_all.min():.3f}  "
              f"p50={np.percentile(arr_all,50):.3f}  "
              f"p90={np.percentile(arr_all,90):.3f}  max={arr_all.max():.3f}  "
              f"mean={arr_all.mean():.3f}")

        fig, axes = plt.subplots(1, len(SPLITS), figsize=(4 * len(SPLITS), 3.5),
                                 sharex=True, sharey=False, constrained_layout=True)
        fig.suptitle(f"label '{key}' (numeric) — per-split histogram")
        for ax, s in zip(axes, SPLITS):
            vs = np.array(values_by_split[s], dtype=float)
            if vs.size == 0:
                ax.set_title(f"{s} (empty)")
                continue
            ax.hist(vs, bins=cfg.numeric_hist_bins)
            ax.set_title(f"{s} (n={vs.size})")
            ax.set_xlabel(key)
        png = out_dir / f"{jsonl_stem}_label_{key}.png"
        fig.savefig(png)
        plt.show()
        plt.close(fig)
        label_plot_paths[key] = png
        print(f"  saved {png.relative_to(PROJECT_ROOT)}")
        continue

    # ---- Categorical: per-split bar + imbalance warnings -------------------
    counts_by_split: dict[str, Counter] = {s: Counter(values_by_split[s]) for s in SPLITS}
    classes_order = [c for c, _ in
                     sum((Counter(v) for v in values_by_split.values()), Counter()).most_common()]

    header = f"  {'class':<24}" + "".join(f"{s:>10}" for s in SPLITS) + f"{'total':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    grand = Counter()
    for cls in classes_order:
        row = f"  {str(cls)[:24]:<24}"
        tot = 0
        for s in SPLITS:
            n = counts_by_split[s].get(cls, 0)
            tot += n
            row += f"{n:>10}"
            grand[cls] += n
        row += f"{tot:>10}"
        print(row)
    warns = _imbalance_warnings(grand, max_ratio=cfg.imbalance_max_ratio,
                                rare_pct=cfg.imbalance_rare_pct)
    label_warnings[key] = warns
    for w in warns:
        print(f"  WARN: {w}")

    fig, axes = plt.subplots(1, len(SPLITS), figsize=(4 * len(SPLITS), 3.5),
                             sharey=True, constrained_layout=True)
    fig.suptitle(f"label '{key}' — per-split distribution")
    for ax, s in zip(axes, SPLITS):
        c = counts_by_split[s]
        vals = [c.get(cls, 0) for cls in classes_order]
        ax.bar(range(len(classes_order)), vals)
        ax.set_xticks(range(len(classes_order)))
        ax.set_xticklabels([str(c) for c in classes_order],
                           rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{s} (n={sum(vals)})")
        ax.set_ylabel("count")
    png = out_dir / f"{jsonl_stem}_label_{key}.png"
    fig.savefig(png)
    plt.show()
    plt.close(fig)
    label_plot_paths[key] = png
    print(f"  saved {png.relative_to(PROJECT_ROOT)}")

# %% [markdown]
# ## 8. Missing fields
#
# Required-key absence is impossible if validation passed in §2. This sweep is
# for the optional canonical keys — `file_id`, `start_t`/`end_t`, `speaker`,
# `text`, `frame_rate_hz` — and for `metadata.source_audio`, which the splitter
# uses for resolution. High percentages here are normal for some corpora (ROG
# has no `frame_rate_hz` on instance records) — the column is informational.

# %%
OPTIONAL_KEYS = ["file_id", "start_t", "end_t", "speaker", "text", "frame_rate_hz"]
META_KEYS     = ["source_audio"]

missing: dict[str, int] = {k: 0 for k in OPTIONAL_KEYS + [f"metadata.{m}" for m in META_KEYS]}
for r in records:
    for k in OPTIONAL_KEYS:
        if k not in r:
            missing[k] += 1
    md = r.get("metadata", {}) or {}
    for m in META_KEYS:
        if not md.get(m):
            missing[f"metadata.{m}"] += 1

print(f"{'field':<26}{'missing':>10}{'pct':>10}")
print("-" * 46)
for k, n in missing.items():
    pct = 100 * n / max(1, len(records))
    print(f"{k:<26}{n:>10}{pct:>9.1f}%")

# %% [markdown]
# ## 9. Markdown report
#
# Everything printed above, gathered into a single `<stem>_report.md` next to
# the PNGs. Open it in any markdown viewer or commit it alongside experiment
# notes when a dataset version changes.

# %%
def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_report() -> Path:
    L: list[str] = []
    L.append(f"# Dataset report — `{cfg.jsonl_path}`")
    L.append("")
    L.append(f"- Records: **{len(records)}**")
    L.append(f"- Datasets: {dict(datasets)}")
    L.append(f"- Label keys: {label_keys_present}")
    L.append("")

    L.append("## Splits")
    L.append("")
    L.extend(_md_table(
        ["split", "count", "pct"],
        [[s, str(split_counts.get(s, 0)),
          f"{100*split_counts.get(s, 0)/max(1, len(records)):.1f}%"] for s in SPLITS],
    ))
    L.append("")
    L.append(f"![splits]({split_png.name})")
    L.append("")

    L.append("## Speakers")
    L.append("")
    L.extend(_md_table(
        ["split", "unique speakers"],
        [[s, str(len(speakers_by_split[s]))] for s in SPLITS],
    ))
    L.append("")
    if spk_png is not None:
        L.append(f"![top_speakers]({spk_png.name})")
        L.append("")
    if spk_hist_png is not None:
        L.append(f"![speaker_hist]({spk_hist_png.name})")
        L.append("")

    L.append("## Audio durations")
    L.append("")
    rows = []
    for s in SPLITS:
        d = durations_by_split.get(s, np.array([]))
        if d.size == 0:
            rows.append([s, "0", "-", "-", "-", "-", "-", "-"])
            continue
        rows.append([
            s, str(d.size), f"{d.min():.2f}", f"{np.percentile(d,50):.2f}",
            f"{np.percentile(d,90):.2f}", f"{np.percentile(d,99):.2f}",
            f"{d.max():.2f}", str(int((d > cfg.max_duration_s).sum())),
        ])
    L.extend(_md_table(
        ["split", "n", "min", "p50", "p90", "p99", "max", f">{cfg.max_duration_s}s"],
        rows,
    ))
    L.append("")
    L.append(f"![duration_hist]({dur_hist_png.name})")
    L.append("")
    if dur_box_png is not None:
        L.append(f"![duration_box]({dur_box_png.name})")
        L.append("")
    if duration_mismatches:
        L.append(f"**WARN: {len(duration_mismatches)} records have on-disk duration "
                 ">50 ms off from `end_t - start_t`.**")
        L.append("")

    L.append("## Labels")
    L.append("")
    for key in scalar_keys:
        L.append(f"### `{key}` ({label_kind.get(key, 'unknown')})")
        L.append("")
        L.append(f"- nulls: {null_counts.get(key, 0)} "
                 f"({100*null_counts.get(key, 0)/max(1, len(records)):.2f}%)")
        for w in label_warnings.get(key, []):
            L.append(f"- WARN: {w}")
        L.append("")
        if key in label_plot_paths:
            L.append(f"![label_{key}]({label_plot_paths[key].name})")
            L.append("")

    L.append("## Missing optional fields")
    L.append("")
    L.extend(_md_table(
        ["field", "missing", "pct"],
        [[k, str(n), f"{100*n/max(1, len(records)):.1f}%"]
         for k, n in missing.items()],
    ))
    L.append("")

    p = out_dir / f"{jsonl_stem}_report.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


report_path = write_report()
print(f"wrote {report_path.relative_to(PROJECT_ROOT)}")
