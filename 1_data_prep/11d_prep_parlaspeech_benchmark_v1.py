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
# # Prep ParlaSpeech-HR benchmark **v1** — chapter 1 (variant d)
#
# Pull the ParlaSpeech-HR benchmark v1 from **Hugging Face** and emit canonical pipeline
# JSONLs — **one file per task**. The benchmark's splits are *per-task* (the same utterance
# can be `gender/train` and `age/dev`), so we write one small file per task; within each,
# the canonical shape holds and `31` consumes it unchanged.
#
# **What changed vs. the old flow:** the source is `load_dataset("porupski/ParlaSpeech-HR-benchmark_v1")`
# instead of a local `.jsonl` + `audio/` folder. Audio bytes come inline in the parquet;
# we write them to disk as 16 kHz PCM_16 WAVs so `audio_path` in the emitted JSONL still
# points at a real file on disk — identical schema, identical downstream contract.
#
# **All four v1 tasks are classification** (v1's age is a `young`/`old` group, not continuous):
#
# | task | label_key | type | labels |
# |---|---|---|---|
# | `gender` | `speaker_gender` | classification | M / F |
# | `speaker_id` | `speaker_name` | classification | 50-class |
# | `power_status` | `power_status` | classification | Coalition / Opposition |
# | `age` | `speaker_age_group` | classification | young / old |
#
# Labels are normalized to v3's casing where they overlap (`m`→`M`, `coalition`→`Coalition`).
#
# ---
#
# ## 0. Setup

# %%
import time

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
import io
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm.auto import tqdm


def read_audio_bytes(entry: dict) -> tuple[np.ndarray, int]:
    """Decode an ``Audio(decode=False)`` row-entry with soundfile.

    ``entry`` is ``{"bytes": <raw file bytes or None>, "path": <str or None>}``.
    We prefer the inlined bytes (present after ``push_to_hub``); fall back to
    ``path`` if only that's set. Using soundfile everywhere sidesteps
    ``datasets``' torchcodec dependency, which brings in CUDA runtime libs
    even on CPU-only PyTorch and often breaks fresh envs.
    """
    b = entry.get("bytes")
    if b is not None:
        arr, sr = sf.read(io.BytesIO(b))
    else:
        arr, sr = sf.read(entry["path"])
    return arr, int(sr)

# %% [markdown]
# ---
#
# ## 1. Config
#
# - `hf_repo` — where the benchmark lives on HF. Cached under `~/.cache/huggingface/`.
# - `audio_dir` — where to write the extracted WAVs. Layout: `<audio_dir>/<hash>/<instance_id>.wav`.
# - `limit` — 0 = all rows. Set small for a quick sanity check.
# - `force_audio` — overwrite existing WAVs. Default off; already-written files are skipped.

# %%
@dataclass
class Config:
    hf_repo: str = "porupski/ParlaSpeech-HR-benchmark_v1"
    dataset_name: str = "ParlaSpeech-HR-benchmark-v1"

    output_dir: str = "data/processed_jsonl"
    audio_dir:  str = "data/benchmarking/ParlaSpeech-HR-benchmark-v1/audio"

    tasks: tuple = ("gender", "speaker_id", "power_status", "age")

    limit: int = 0
    force_audio: bool = False

cfg = Config()

# %% [markdown]
# ---
#
# ## 2. Task registry
#
# Same shape as before: each task names its `label_key`, task type, and output file.
# `transform` normalizes the raw benchmark label (v1's lowercased forms → v3-consistent casing).

# %%
def build_tasks(out_dir: str) -> dict:
    return {
        "gender": dict(
            label_key="speaker_gender", task_type="classification",
            transform=lambda x: None if x is None else str(x).upper(),          # m/f → M/F
            out=f"{out_dir}/parlaspeech_hr_bench_v1_gender.jsonl"),
        "speaker_id": dict(
            label_key="speaker_name", task_type="classification",
            transform=lambda x: None if x is None else str(x),
            out=f"{out_dir}/parlaspeech_hr_bench_v1_speaker_id.jsonl"),
        "power_status": dict(
            label_key="power_status", task_type="classification",
            transform=lambda x: None if x is None else str(x).capitalize(),     # coalition → Coalition
            out=f"{out_dir}/parlaspeech_hr_bench_v1_power_status.jsonl"),
        "age": dict(
            label_key="speaker_age_group", task_type="classification",
            transform=lambda x: None if x is None else str(x),                  # young / old
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
# ## 3. Load the HF dataset
#
# One call, cached automatically. First run downloads parquet shards; subsequent runs are
# instant. If you Ctrl-C and re-run, `datasets` resumes from cache.

# %%
mark("load")
ds = load_dataset(cfg.hf_repo, "default", split="train")
# Turn off datasets' built-in audio decoding — we hand the raw bytes to
# soundfile ourselves (see `read_audio_bytes` above).
ds = ds.cast_column("audio", Audio(decode=False))
if cfg.limit:
    ds = ds.select(range(min(cfg.limit, len(ds))))
print(f"✅ loaded {len(ds)} rows from {cfg.hf_repo}")

# Sanity peek
ex = ds[0]
_arr, _sr = read_audio_bytes(ex["audio"])
print(f"  e.g. {ex['instance_id']}  | {ex['speaker_gender']} | {ex['speaker_name']} "
      f"| audio shape {_arr.shape} @ {_sr} Hz")

# %% [markdown]
# ---
#
# ## 4. Extract audio to disk + build canonical rows
#
# For each row: write the inlined audio bytes to `<audio_dir>/<hash>/<instance_id>.wav`
# (16 kHz mono PCM_16), then assemble the canonical base row. For each task the row
# participates in (non-null label + split), append to that task's bucket.
#
# `file_id` is the YouTube hash, parsed from `instance_id` (`<hash>_<start>-<end>`).

# %%
def hash_of(instance_id: str) -> str:
    """v1 instance_id shape: `<hash>_<start>-<end>`. The hash itself may contain `_`."""
    return instance_id.rpartition("_")[0]


mark("audio+build")
audio_root = PROJECT_ROOT / cfg.audio_dir
audio_root.mkdir(parents=True, exist_ok=True)

per_task: dict[str, list[dict]] = defaultdict(list)
n_written = 0
n_skipped = 0

for row in tqdm(ds, desc="processing", unit=" rows"):
    instance_id = row["instance_id"]
    file_hash   = hash_of(instance_id)

    audio_arr, sr = read_audio_bytes(row["audio"])

    audio_rel = f"{cfg.audio_dir}/{file_hash}/{instance_id}.wav"
    audio_abs = PROJECT_ROOT / audio_rel
    audio_abs.parent.mkdir(parents=True, exist_ok=True)
    if cfg.force_audio or not audio_abs.exists():
        sf.write(str(audio_abs), audio_arr, sr, subtype="PCM_16")
        n_written += 1
    else:
        n_skipped += 1

    base = {
        "instance_id": instance_id,
        "dataset":     cfg.dataset_name,
        "file_id":     file_hash,
        "audio_path":  audio_rel,
        "speaker":     row["speaker_name"] or "unknown",
        "text":        row["text"],
        "metadata": {
            "path":         row["path"],
            "orig_file":    row["orig_file"],
            "source_audio": row["path"],
            "audio_length": row["audio_length"],
            "start":        row["start"],
            "end":          row["end"],
            "speaker_info": {
                "Speaker_role":       row["speaker_role"],
                "Speaker_type":       row["speaker_type"],
                "Speaker_party":      row["speaker_party"],
                "Speaker_party_name": row["speaker_party_name"],
                "Party_status":       row["party_status"],
                "Speaker_name":       row["speaker_name"],
                "Speaker_gender":     row["speaker_gender"],
                "Speaker_birth":      row["speaker_birth"],
            },
            "benchmark": {
                task: {
                    "label": row[f"benchmark_{task}_label"],
                    "split": row[f"benchmark_{task}_split"],
                }
                for task in cfg.tasks
                if row.get(f"benchmark_{task}_split")
            },
        },
    }
    for task in cfg.tasks:
        split = row.get(f"benchmark_{task}_split")
        label = row.get(f"benchmark_{task}_label")
        if not split or label is None:
            continue
        spec = TASKS[task]
        per_task[task].append({
            **base,
            "split":  split,
            "labels": {spec["label_key"]: spec["transform"](label)},
        })

print(f"✅ audio: {n_written} written, {n_skipped} skipped (already on disk)")
for task in cfg.tasks:
    print(f"  {task}: {len(per_task[task])} rows")

# %% [markdown]
# ---
#
# ## 5. Write per-task JSONLs

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
# ## 6. Sanity checks
#
# Schema validation, split sizes, no null labels, label distributions, **speaker-disjoint**
# splits (gender/age/power) and **same-speakers + hash-disjoint** splits (speaker_id).

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
    top  = dict(sorted(dist.items(), key=lambda kv: -kv[1])[:4])
    print(f"     labels: {len(dist)} classes, top {top}")

    spk_by_split  = defaultdict(set)
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
        print("no timing recorded"); return
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
# **Chapter 3** — `31` trains on these files via the existing `hr_bench_v1_*` presets in
# `utils_instance_train.py`'s `TARGETS` (gender, speaker_id, power_status, age — all
# classification).
