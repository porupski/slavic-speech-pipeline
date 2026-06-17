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
# # Prep ParlaSpeech-HR benchmark **v1** — chapter 1 (variant e)
#
# Parse the pre-built **ParlaSpeech-HR-benchmark-v1** (built against ParlaSpeech-HR
# **v1.0**, placed under `data/benchmarking/`) into canonical pipeline JSONLs —
# **one small file per task**, because the benchmark's splits are *per-task*: the
# same utterance can be gender/train and age/dev, so a shared top-level `split`
# cannot exist. Within each task file the canonical shape holds and `31` consumes
# it unchanged.
#
# **Sibling to `11d` (the v3 benchmark prep), with v1's quirks:**
# - splits come from the `benchmark` key — **no `assign_splits`** (the benchmark
#   construction already guarantees speaker-disjointness, and hash-disjointness
#   for `speaker_id`);
# - **no audio conversion** — the build script already wrote ready 16 kHz mono
#   WAVs under `audio/<hash>/<stem>.wav`; `audio_path` points straight there;
# - the v1 record has **no `id`/`audio`/`audio_length` fields**. Identity comes
#   from `path` (`seg.<hash>_<start>-<end>.flac`): the stem (minus `seg.`) is the
#   `instance_id` and names the WAV; the hash is the `file_id`; duration is
#   `end - start`. The original `path` tags along in metadata for tracking.
# - **all four tasks are classification** (v1's `age` is a `young`/`old` group, not
#   a continuous age) — unlike v3, which had two regression tasks.
#
# **Tasks emitted** (each → `parlaspeech_hr_bench_v1_<task>.jsonl`):
#
# | task | label_key | type | labels |
# |---|---|---|---|
# | `gender` | `speaker_gender` | classification | M / F |
# | `speaker_id` | `speaker_name` | classification | 50-class |
# | `power_status` | `power_status` | classification | Coalition / Opposition |
# | `age` | `speaker_age_group` | classification | young / old |
#
# Labels are normalized to v3's casing where they overlap (`m`→`M`, `coalition`→
# `Coalition`) so a single chapter-3 target family can span both benchmarks.
#
# ---
#
# ## 0. Setup

# %%
import time

# ── Stage timing ──────────────────────────────────────────────
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

PROJECT_ROOT = udp.PROJECT_ROOT
print(f"PROJECT_ROOT = {PROJECT_ROOT}")

# %% [markdown]
# Standard imports.

# %%
import json
from collections import Counter, defaultdict
from dataclasses import dataclass

from tqdm.auto import tqdm

# %% [markdown]
# ---
#
# ## 1. Config
#
# - `benchmark_dir` — where the pre-built benchmark lives (jsonl + `audio/`).
#   **Read-only** — this notebook never writes back into it; the derived files
#   go to `output_dir`.
# - `tasks` — which task files to emit.
# - `check_audio` — verify every referenced WAV exists on disk (cheap; the
#   benchmark is small).

# %%
@dataclass
class Config:
    benchmark_dir: str = "data/benchmarking/ParlaSpeech-HR-benchmark-v1"
    benchmark_jsonl: str = "ParlaSpeech-HR-benchmark-v1.jsonl"
    dataset_name: str = "ParlaSpeech-HR-benchmark-v1"

    output_dir: str = "data/processed_jsonl"
    tasks: tuple = ("gender", "speaker_id", "power_status", "age")

    check_audio: bool = True

cfg = Config()


# %% [markdown]
# ---
#
# ## 2. Task registry
#
# Mirrors the `TARGETS` registry the trainer uses: each task names its
# `label_key`, task type, and output file. `transform` normalizes the raw
# benchmark label into the value the trainer expects — here it also fixes v1's
# lowercased labels (`m`/`coalition`) to v3's casing (`M`/`Coalition`) so the two
# benchmarks share a target family. `young`/`old` and speaker names pass through.

# %%
def build_tasks(out_dir: str) -> dict:
    return {
        "gender": dict(
            label_key="speaker_gender", task_type="classification",
            transform=lambda x: str(x).upper(),          # m/f → M/F
            out=f"{out_dir}/parlaspeech_hr_bench_v1_gender.jsonl"),
        "speaker_id": dict(
            label_key="speaker_name", task_type="classification",
            transform=str,                               # "Surname, Name" verbatim
            out=f"{out_dir}/parlaspeech_hr_bench_v1_speaker_id.jsonl"),
        "power_status": dict(
            label_key="power_status", task_type="classification",
            transform=lambda x: str(x).capitalize(),     # coalition → Coalition
            out=f"{out_dir}/parlaspeech_hr_bench_v1_power_status.jsonl"),
        "age": dict(
            label_key="speaker_age_group", task_type="classification",
            transform=str,                               # young / old verbatim
            out=f"{out_dir}/parlaspeech_hr_bench_v1_age.jsonl"),
    }

TASKS = build_tasks(cfg.output_dir)
for name in cfg.tasks:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}. Known: {sorted(TASKS)}")
print(f"tasks to emit: {list(cfg.tasks)}")


# %% [markdown]
# ---
#
# ## 3. Locate the benchmark

# %%
BENCH_DIR = PROJECT_ROOT / cfg.benchmark_dir
BENCH_JSONL = BENCH_DIR / cfg.benchmark_jsonl
AUDIO_DIR = BENCH_DIR / "audio"

for p, what in ((BENCH_JSONL, "benchmark jsonl"), (AUDIO_DIR, "audio dir")):
    if not p.exists():
        raise FileNotFoundError(
            f"{what} not found: {p}\n"
            f"Place the ParlaSpeech-HR-benchmark-v1 bundle under "
            f"{cfg.benchmark_dir} and re-run.")
print(f"✅ {BENCH_JSONL.relative_to(PROJECT_ROOT)}  "
      f"({BENCH_JSONL.stat().st_size/1e6:.0f} MB)")

# %% [markdown]
# ---
#
# ## 4. Preflight — peek at one record
#
# v1 records carry word alignments and a top-level corpus `split`; we use neither
# (the trainer wants utterance audio + a per-task split). Confirms the fields we
# *do* rely on are present.

# %%
def stem_of(path_field: str) -> str:
    """`seg.<hash>_<s>-<e>.flac` → `<hash>_<s>-<e>` (the WAV stem + instance_id)."""
    stem = Path(path_field).stem            # drops the .flac suffix
    return stem[4:] if stem.startswith("seg.") else stem

def hash_of(stem: str) -> str:
    """Split the YouTube hash off the right; the hash itself may contain `_`."""
    return stem.rpartition("_")[0]

with open(BENCH_JSONL, encoding="utf-8") as f:
    ex = json.loads(f.readline())
si = ex.get("speaker_info", {})
_stem = stem_of(ex["path"])
print(f"  e.g. {_stem}  | hash {hash_of(_stem)} "
      f"| {si.get('Speaker_gender','?')} | {si.get('Speaker_name','?')} "
      f"| dur {round(ex['end'] - ex['start'], 3)}s")
print(f"  benchmark tasks on this record: {list(ex.get('benchmark', {}))}")

# %% [markdown]
# ---
#
# ## 5. Parse — one canonical row per (record × selecting task)
#
# Each benchmark record carries `benchmark: {task: {label, split}}` for the tasks
# that selected it (only ~10% of instances are shared across tasks, so most carry
# a single task). We emit one canonical row per entry, with that task's split and
# a single label under the task's `label_key`. `speaker` is `Speaker_name` (v1 has
# no `Speaker_ID`); metadata keeps the original `path`/`orig_file`, the utterance
# bounds, the full `speaker_info`, and the full `benchmark` dict for traceability.

# %%
def parse_rows(path: Path) -> dict[str, list[dict]]:
    per_task: dict[str, list[dict]] = defaultdict(list)
    n_in = 0
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="parsing benchmark", unit=" lines", leave=False):
            n_in += 1
            r = json.loads(line)
            si = r.get("speaker_info", {})
            stem = stem_of(r["path"])
            file_hash = hash_of(stem)
            start, end = float(r["start"]), float(r["end"])
            base = {
                "instance_id": stem,
                "dataset":     cfg.dataset_name,
                "file_id":     file_hash,
                "audio_path":  f"{cfg.benchmark_dir}/audio/{file_hash}/{stem}.wav",
                "speaker":     si.get("Speaker_name", "unknown"),
                "text":        " ".join(r.get("words", [])),
                "metadata": {
                    "path":         r["path"],            # original v1 field, tracking
                    "orig_file":    r.get("orig_file"),
                    "source_audio": r["path"],
                    "audio_length": round(end - start, 3),
                    "start":        start,
                    "end":          end,
                    "speaker_info": si,
                    "benchmark":    r.get("benchmark", {}),
                },
            }
            for task, entry in r.get("benchmark", {}).items():
                if task not in cfg.tasks:
                    continue
                spec = TASKS[task]
                per_task[task].append({
                    **base,
                    "split":  entry["split"],
                    "labels": {spec["label_key"]: spec["transform"](entry["label"])},
                })
    print(f"  parsed {n_in} benchmark records")
    for task in cfg.tasks:
        print(f"  {task}: {len(per_task[task])} rows")
    return per_task


mark("parse")
per_task = parse_rows(BENCH_JSONL)

# %% [markdown]
# ---
#
# ## 6. Audio check
#
# The benchmark ships its own WAVs; this just confirms every referenced file is
# actually there (e.g. the tarball unpacked fully).

# %%
if cfg.check_audio:
    paths = {row["audio_path"] for rows in per_task.values() for row in rows}
    missing = [p for p in sorted(paths) if not (PROJECT_ROOT / p).exists()]
    if missing:
        print(f"⚠️  {len(missing)}/{len(paths)} WAVs missing, e.g.:")
        for m in missing[:5]:
            print(f"   {m}")
        raise FileNotFoundError("benchmark audio incomplete — re-run the build/download")
    print(f"✅ all {len(paths)} referenced WAVs present")
else:
    print("check_audio=False — skipped")

# %% [markdown]
# ---
#
# ## 7. Write task JSONLs

# %%
mark("write")
written = {}
for task in cfg.tasks:
    spec = TASKS[task]
    n = udp.write_jsonl(per_task[task], spec["out"])
    written[task] = (spec["out"], n)
    print(f"  ✅ {task}: {n} → {spec['out']}")

# %% [markdown]
# ---
#
# ## 8. Sanity checks
#
# Beyond schema validation, re-verify the benchmark invariants per task: split
# sizes, no null labels, label distributions, **speaker-disjoint** splits
# (gender/age/power) and **same-speakers + hash-disjoint** splits (speaker_id).

# %%
def sanity(task: str, path: str) -> None:
    spec = TASKS[task]
    rows = udp.read_jsonl(path)
    n_tot, n_valid, errs = udp.validate_jsonl(rows)
    tag = "✅" if not errs else "⚠️ "
    print(f"{tag} {task}: {n_valid}/{n_tot} schema-valid")
    for e in errs[:3]:
        print(f"     {e}")

    key = spec["label_key"]
    n_null = sum(1 for r in rows if r["labels"].get(key) is None)
    splits = Counter(r["split"] for r in rows)
    print(f"     splits: {dict(splits)}  | null labels: {n_null}")

    dist = Counter(r["labels"][key] for r in rows)
    top = dict(sorted(dist.items(), key=lambda kv: -kv[1])[:4])
    print(f"     labels: {len(dist)} classes, top {top}")

    spk_by_split = defaultdict(set)
    hash_by_split = defaultdict(set)
    for r in rows:
        spk_by_split[r["split"]].add(r["speaker"])
        hash_by_split[r["split"]].add(r["file_id"])
    if task == "speaker_id":
        same = spk_by_split["train"] == spk_by_split["dev"] == spk_by_split["test"]
        h_leak = (hash_by_split["train"] & hash_by_split["dev"]) \
               | (hash_by_split["train"] & hash_by_split["test"]) \
               | (hash_by_split["dev"] & hash_by_split["test"])
        print(f"     same speakers in all splits: {'✅' if same else '❌'}  "
              f"| hash leaks: {'✅ none' if not h_leak else f'❌ {len(h_leak)}'}")
    else:
        leak = (spk_by_split["train"] & spk_by_split["dev"]) \
             | (spk_by_split["train"] & spk_by_split["test"]) \
             | (spk_by_split["dev"] & spk_by_split["test"])
        print(f"     speaker leak across splits: "
              f"{'✅ none' if not leak else f'❌ {sorted(leak)[:3]}'}")


for task, (path, _) in written.items():
    sanity(task, path)

# %% [markdown]
# ---
#
# ## Timing

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

mark("end")
print_stage_breakdown(STAGE_TIMES)

# %% [markdown]
# ---
#
# ## Next
#
# - **Chapter 3** — `31` trains on these files via the existing `hr_bench_v1_*`
#   presets in `utils_instance_train.py`'s `TARGETS` (gender, speaker_id,
#   power_status, age — all classification).
