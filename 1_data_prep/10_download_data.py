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
# # Download data — Chapter 1
#
# Fetch the source corpora the rest of the pipeline expects on disk. Two kinds of source:
#
# - **CLARIN.SI** archives (ParlaSpeech, ROG, GOS) — plain HTTP downloads, then unpacked.
# - **Hugging Face Hub** snapshots (ParlaSpeech-HR benchmark v1 / v3) — grabbed via
#   `huggingface_hub`, already laid out as `<name>/audio/<hash>/…` so no unpack step.
#
# The full catalogue lives in [`10a_dataset_registry.json`](10a_dataset_registry.json); the
# helpers live in [`utils_download.py`](utils_download.py). This notebook is meant to stay short
# — pick what you want, glance at the safety switches, run.
#
# **Default target: ParlaSpeech-HR-benchmark-v3.** ~9 GB, ships ready-to-train audio and
# per-task splits — the smallest self-contained thing that lets you finish chapter 1 and move
# on to chapter 3. Anything larger, you opt into deliberately below.
#
# ---
#
# ## 0. Imports and project root

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
import utils_download as udl

PROJECT_ROOT = udp.PROJECT_ROOT
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"Chapter dir  = {HERE}")

REGISTRY = udl.load_registry(HERE)
print(f"Registry     = {HERE / udl.REGISTRY_FILENAME}")
print(f"Datasets available: {sorted(REGISTRY['datasets'])}")
print(f"Shorthands:         {sorted(REGISTRY['shorthands'])}")

# %% [markdown]
# ---
#
# ## 1. What to download
#
# `cfg.datasets` is the list of registry keys (or shorthands) to fetch. Examples:
#
# - `["ParlaSpeech-HR-benchmark-v3"]` — the default; ~9 GB, ready to train.
# - `["ParlaSpeech-HR-benchmark-v1"]` — the older HR benchmark (~9 GB).
# - `["ParlaSpeech-RS", "ParlaSpeech-RS-audio"]` — Serbian annotations + audio; smallest of the
#   full ParlaSpeech releases.
# - `["ParlaSpeech-benchmarks"]` — shorthand: both HR benchmarks.
# - `["ParlaSpeech"]` — all four ParlaSpeech annotation sets (no audio).
# - `["ParlaSpeech-audio"]` — all four ParlaSpeech audio sets. **Hundreds of GB.** Be sure.

# %%
from dataclasses import dataclass, field

@dataclass
class Config:
    datasets: list = field(default_factory=lambda: ["ParlaSpeech-HR-benchmark-v3"])
    confirm_large: bool = True   # allow files/snapshots marked is_large
    force: bool = False          # re-download things already on disk
    dry_run: bool = False        # plan only, do not touch disk
    download_only: bool = False  # download but skip the unpack step

cfg = Config()
print(cfg)

# %% [markdown]
# ---
#
# ## 2. ⚠️  Safety switches — read before running
#
# Four boolean knobs sit above and control what actually happens. Defaults are cautious.
# Change them in the cell below **only if you know what you're doing**.
#
# | Switch | Default | Effect |
# |---|---|---|
# | `confirm_large` | `True`  | Allow files/snapshots marked `is_large`. Set to `False` and the plan will *skip* every large item — safe way to explore the catalogue. |
# | `force`         | `False` | Wipe partial downloads and re-fetch from scratch. Leave `False` and interrupted downloads resume where they left off. |
# | `dry_run`       | `False` | Print the plan and stop. Nothing touches disk. Great for a first pass. |
# | `download_only` | `False` | Fetch archives but skip the unpack step. Useful when moving data between machines. |
#
# **Resume is on by default** — Ctrl-C the notebook mid-download and just re-run this cell.
# CLARIN sources continue via HTTP `Range` on the `.part` file; HF snapshots continue via
# `snapshot_download`'s built-in `.incomplete` handling. `force=True` overrides both.
#
# **If you're just here to poke around, set `dry_run = True` in the cell below.**

# %%
cfg.confirm_large = True
cfg.force         = False
cfg.dry_run       = False
cfg.download_only = False

print(f"  confirm_large = {cfg.confirm_large}")
print(f"  force         = {cfg.force}")
print(f"  dry_run       = {cfg.dry_run}")
print(f"  download_only = {cfg.download_only}")

# %% [markdown]
# ---
#
# ## 3. Resolve names
#
# Expands shorthands, deduplicates, and validates every requested key against the registry.

# %%
target_datasets = udl.resolve_datasets(cfg.datasets, REGISTRY)
print(f"Requested: {cfg.datasets}")
print(f"Resolved:  {target_datasets}")

# %% [markdown]
# ---
#
# ## 4. Plan
#
# One line per downloadable item. `is_large` items are skipped when `confirm_large=False`, and
# the whole thing is a no-op when `dry_run=True`. Nothing has touched disk yet.

# %%
plan = []

for ds_name in target_datasets:
    spec = REGISTRY["datasets"][ds_name]

    udp.banner(f"Download plan — {ds_name}", char="-")
    print(f"source: {spec['source']}   note: {spec['note']}\n")

    items = udl.iter_files(spec)
    for item in items:
        is_large = item.get("is_large", False)
        size_mb  = item.get("size_mb", 0)

        if spec["source"] == "clarin":
            dest    = PROJECT_ROOT / spec["target_dir"] / item["name"]
            url     = f"{spec['base_url']}/{item['name']}"
            part    = dest.with_suffix(dest.suffix + ".part")
            done    = dest.exists() and dest.stat().st_size > 0
            partial = part.exists() and part.stat().st_size > 0
        else:  # hf
            dest    = PROJECT_ROOT / spec["target_dir"]
            url     = f"hf://{spec['hf_repo']}"
            # HF: can't cheaply know if the snapshot is fully complete without
            # asking the hub, so we always defer to snapshot_download itself
            # (idempotent — skips files already fully on disk).
            done    = False
            partial = dest.exists() and any(dest.iterdir())

        if cfg.dry_run:
            action, reason = "skip", "dry_run"
        elif is_large and not cfg.confirm_large:
            action, reason = "skip", f"large (~{size_mb} MB) — set confirm_large=True"
        elif done and not cfg.force:
            action, reason = "skip", "already on disk"
        elif partial and not cfg.force:
            action, reason = "resume", "picking up partial"
        elif cfg.force:
            action, reason = "download", "force re-download"
        else:
            action, reason = "download", "ok"

        plan.append({
            "dataset":  ds_name,
            "spec":     spec,
            "item":     item,
            "dest":     dest,
            "url":      url,
            "size_mb":  size_mb,
            "is_large": is_large,
            "action":   action,
            "reason":   reason,
        })

        flag = {"download": "📥", "resume": "⏯️ ", "skip": "⏭️ "}[action]
        print(f"  {flag} {item['name']:60s} ~{size_mb:>7} MB   [{action}: {reason}]")

    ds_plans   = [p for p in plan if p["dataset"] == ds_name]
    ds_on_disk = sum(p["size_mb"] for p in ds_plans if p["reason"] == "already on disk")
    ds_active  = sum(p["size_mb"] for p in ds_plans if p["action"] in ("download", "resume"))
    print(f"\n  on disk: ~{ds_on_disk:,} MB   to fetch: ~{ds_active:,} MB\n")

total_on_disk = sum(p["size_mb"] for p in plan if p["reason"] == "already on disk")
total_active  = sum(p["size_mb"] for p in plan if p["action"] in ("download", "resume"))
print(f"Totals — on disk: ~{total_on_disk:,} MB  |  "
      f"to fetch: ~{total_active:,} MB  (~{total_active/1024:.1f} GB)")

# %% [markdown]
# ---
#
# ## 5. Execute
#
# CLARIN items stream to a `.part` sibling and get renamed on success — an aborted download
# never leaves a half-finished file. HF snapshots go through `huggingface_hub.snapshot_download`
# with allow/ignore patterns so we grab the JSONL + audio (+ textgrids for v3) and skip the
# auto-generated parquet.

# %%
if cfg.dry_run:
    print("🧪 dry_run=True — skipping all downloads")
else:
    for p in plan:
        if p["action"] not in ("download", "resume"):
            print(f"⏭️  [{p['dataset']}] {p['item']['name']}  ({p['reason']})")
            continue
        verb = "📥 downloading" if p["action"] == "download" else "⏯️  resuming"
        print(f"{verb} [{p['dataset']}] {p['item']['name']}  ←  {p['url']}")
        try:
            if p["spec"]["source"] == "clarin":
                udl.download_clarin_file(p["url"], p["dest"], force=cfg.force)
                size_mb = p["dest"].stat().st_size / 1e6
                print(f"   ✅ wrote {p['dest'].relative_to(PROJECT_ROOT)}  ({size_mb:.1f} MB)")
            else:  # hf
                udl.download_hf_snapshot(p["spec"], PROJECT_ROOT, force=cfg.force)
                print(f"   ✅ snapshot at {p['dest'].relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"   ❌ failed: {e}")

print("\nDone.")

# %% [markdown]
# ---
#
# ## 6. Unpack (CLARIN only)
#
# HF snapshots are already directories; nothing to unpack. Everything else is either a `.zip`,
# a `.tgz`/`.tar.gz`, or a `.gz`. Idempotent — an already-unpacked target is skipped.

# %%
if cfg.download_only:
    print("⏭️  download_only=True — skipping unpack")
elif cfg.dry_run:
    print("🧪 dry_run=True — skipping unpack")
else:
    for ds_name in target_datasets:
        spec = REGISTRY["datasets"][ds_name]
        if spec["source"] != "clarin":
            continue
        udp.banner(f"Unpacking — {ds_name}", char="-")
        unpacked_dir = PROJECT_ROOT / spec["unpack_dir"]
        unpacked_dir.mkdir(parents=True, exist_ok=True)
        for p in [x for x in plan if x["dataset"] == ds_name]:
            if not p["dest"].exists():
                print(f"  ⏭️  {p['item']['name']}  (not on disk, can't unpack)")
                continue
            try:
                udl.unpack(p["dest"], unpacked_dir, PROJECT_ROOT)
            except Exception as e:
                print(f"     ❌ unpack failed for {p['item']['name']}: {e}")

print("\nDone.")

# %% [markdown]
# ---
#
# ## 7. What's on disk now

# %%
for ds_name in target_datasets:
    spec = REGISTRY["datasets"][ds_name]
    print(f"\n=== {ds_name} ===")
    if spec["source"] == "clarin":
        print("raw:")
        udl.show_tree(PROJECT_ROOT / spec["target_dir"], PROJECT_ROOT)
        print("unpacked:")
        udl.show_tree(PROJECT_ROOT / spec["unpack_dir"], PROJECT_ROOT)
    else:  # hf
        print("snapshot:")
        udl.show_tree(PROJECT_ROOT / spec["target_dir"], PROJECT_ROOT)

# %% [markdown]
# ---
#
# ## Next
#
# - **ParlaSpeech-HR benchmark v1** → `11d_prep_parlaspeech_benchmark_v1.ipynb`
# - **ParlaSpeech-HR benchmark v3** → `11e_prep_parlaspeech_benchmark_v3.ipynb`
# - **ParlaSpeech-{HR,RS,PL,CZ}** → `11c_prep_parlaspeech.ipynb` (`cfg.lang` pins one language;
#   empty = every ParlaSpeech-{LANG} under `data/unpacked/`).
# - **ROG** → `11a_prep_ROG-art.ipynb`
# - **ROG-Dialog** → `11b_prep_ROG-dia.ipynb`
# - **GOS** → arrange restricted audio access first.
#
# Re-running with the same config just prints skip lines — idempotent.
