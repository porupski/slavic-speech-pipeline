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
# # Prep ParlaSpeech-HR benchmark — chapter 1 (variant d)
#
# Parse the pre-built **ParlaSpeech-HR-benchmark-v3** (placed under
# `data/benchmarking/`) into canonical pipeline JSONLs — **one small file per
# task**, because the benchmark's splits are *per-task*: the same utterance can
# be gender/train and age/dev, so a shared top-level `split` cannot exist.
# Within each task file, the canonical shape holds and 31/32 consume it
# unchanged.
#
# **Differences from `11c_prep_parlaspeech`:**
# - splits come from the `benchmark` key — **no `assign_splits`** (the benchmark
#   construction already guarantees speaker-disjointness, and hash-disjointness
#   for `speaker_id`);
# - **no audio conversion** — the benchmark ships ready 16 kHz mono WAVs under
#   `data/benchmarking/ParlaSpeech-HR-benchmark-v3/audio/`; `audio_path` points
#   straight there.
#
# **Tasks emitted** (each → `parlaspeech_hr_bench_<task>.jsonl`):
#
# | task | label_key | type |
# |---|---|---|
# | `gender` | `speaker_gender` | classification (M/F) |
# | `speaker_id` | `speaker_name` | classification (50-class) |
# | `power_status` | `power_status` | classification (Coalition/Opposition) |
# | `age` | `speaker_age` | regression (years at recording) |
# | `orientation` | `orientation` | regression (−3 … +3) |
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
# - `tasks` — which task files to emit.
# - `check_audio` — verify every referenced WAV exists on disk (cheap; the
#   benchmark is small).

# %%
@dataclass
class Config:
    benchmark_dir: str = "data/benchmarking/ParlaSpeech-HR-benchmark-v3"
    benchmark_jsonl: str = "ParlaSpeech-HR-benchmark-v3.jsonl"
    dataset_name: str = "ParlaSpeech-HR-benchmark-v3"

    output_dir: str = "data/processed_jsonl"
    tasks: tuple = ("gender", "speaker_id", "power_status", "age", "orientation")

    check_audio: bool = True

cfg = Config()


# %% [markdown]
# ---
#
# ## 2. Task registry
#
# Mirrors the `TARGETS` registry the trainers use: each task names its
# `label_key`, task type, and output file. `transform` normalizes the raw
# benchmark label into the type the trainer expects.

# %%
def build_tasks(out_dir: str) -> dict:
    return {
        "gender": dict(
            label_key="speaker_gender", task_type="classification",
            transform=str,
            out=f"{out_dir}/parlaspeech_hr_bench_v3_gender.jsonl"),
        "speaker_id": dict(
            label_key="speaker_name", task_type="classification",
            transform=str,
            out=f"{out_dir}/parlaspeech_hr_bench_v3_speaker_id.jsonl"),
        "power_status": dict(
            label_key="power_status", task_type="classification",
            transform=str,
            out=f"{out_dir}/parlaspeech_hr_bench_v3_power_status.jsonl"),
        "age": dict(
            label_key="speaker_age", task_type="regression",
            transform=int,
            out=f"{out_dir}/parlaspeech_hr_bench_v3_age.jsonl"),
        "orientation": dict(
            label_key="orientation", task_type="regression",
            transform=float,
            out=f"{out_dir}/parlaspeech_hr_bench_v3_orientation.jsonl"),
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
            f"Place the ParlaSpeech-HR-benchmark-v3 bundle under "
            f"{cfg.benchmark_dir} and re-run.")
print(f"✅ {BENCH_JSONL.relative_to(PROJECT_ROOT)}  "
      f"({BENCH_JSONL.stat().st_size/1e6:.0f} MB)")

# %% [markdown]
# ---
#
# ## 4. Preflight — peek at one record

# %%
with open(BENCH_JSONL, encoding="utf-8") as f:
    ex = json.loads(f.readline())
si = ex.get("speaker_info", {})
print(f"  e.g. {ex['id']}  | {si.get('Speaker_gender','?')} "
      f"| {si.get('Speaker_name','?')} | dur {ex.get('audio_length')}s")
print(f"  benchmark tasks on this record: {list(ex.get('benchmark', {}))}")

# %% [markdown]
# ---
#
# ## 5. Parse — one canonical row per (record × selecting task)
#
# Each benchmark record carries `benchmark: {task: {label, split}}` for the
# tasks that selected it. We emit one canonical row per entry, with that task's
# split and a single label under the task's `label_key`. Metadata keeps the full
# `benchmark` dict for traceability, plus the usual provenance.

# %%
def parse_rows(path: Path) -> dict[str, list[dict]]:
    per_task: dict[str, list[dict]] = defaultdict(list)
    n_in = 0
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="parsing benchmark", unit=" lines", leave=False):
            n_in += 1
            r = json.loads(line)
            si = r.get("speaker_info", {})
            raw_audio = r.get("audio")
            file_hash = Path(raw_audio).parts[0]
            stem = Path(raw_audio).stem
            base = {
                "instance_id": r["id"],
                "dataset":     cfg.dataset_name,
                "file_id":     file_hash,
                "audio_path":  f"{cfg.benchmark_dir}/audio/{file_hash}/{stem}.wav",
                "speaker":     si.get("Speaker_ID", "unknown"),
                "text":        r.get("text"),
                "metadata": {
                    "source_audio": raw_audio,
                    "audio_length": round(float(r.get("audio_length", 0.0)), 3),
                    "lang":         si.get("Lang"),
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
# Beyond schema validation, re-verify the benchmark invariants per task:
# split sizes, no null labels, label distributions, **speaker-disjoint** splits
# (gender/age/power/orientation) and **same-speakers + hash-disjoint** splits
# (speaker_id).

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

    if spec["task_type"] == "classification":
        dist = Counter(r["labels"][key] for r in rows)
        top = dict(sorted(dist.items(), key=lambda kv: -kv[1])[:4])
        print(f"     labels: {len(dist)} classes, top {top}")
    else:
        vals = [r["labels"][key] for r in rows]
        print(f"     label range: [{min(vals)}, {max(vals)}]  "
              f"mean {sum(vals)/len(vals):.2f}")

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
# - **Chapter 3** — `31`/`32` train on these files via the existing
#   `hr_bench_v3_*` presets in `utils_instance_train.py`'s `TARGETS`
#   (gender, speaker_id, power_status — classification; age, orientation —
#   regression).
