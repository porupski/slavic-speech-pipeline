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
# # Download data — chapter 1
#
# This notebook downloads the source datasets used by `slavic-speech-pipeline` from the CLARIN.SI repository.
#
# **What it does**
#
# - Download one or more datasets into `data/raw/<dataset>/`, sequentially.
# - Unpack archives into `data/unpacked/<dataset>/`.
# - Idempotent: skips files that already exist and are non-empty.
# - Refuses to download multi-GB files unless `confirm_large=True`.
#
# **What it does *not* do**
#
# - Convert anything into canonical JSONL — that's what the `11*_prep_*.ipynb`
#   notebooks do.
# - Cut WAVs — that happens inline inside the prep notebooks via
#   `utils_audio_splitter`.
# - Manage CLARIN auth for restricted resources (e.g. GOS audio). Those need a manual step.
#
# ---
#
# ## 0. Imports and project root setup

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
print(f"Chapter dir  = {HERE}")

# %% [markdown]
# ---
#
# ## 1. Config
#
# Set `datasets` to what you want to download, then Run All.
#
# Examples:
# - `["ROG"]` — just ROG
# - `["ParlaSpeech-HR", "ParlaSpeech-HR-audio"]` — HR annotations + audio
# - `["ParlaSpeech"]` — all 4 PS annotation sets (no audio)
# - `["ParlaSpeech-audio"]` — all 4 PS audio sets (hundreds of GB, be sure)

# %%
from dataclasses import dataclass, field

@dataclass
class Config:
    # Which dataset(s) to fetch.
    # Registry keys: "ROG-Dialog", "ROG", "GOS",
    #   Annotations: "ParlaSpeech-HR/RS/PL/CZ"  or shorthand "ParlaSpeech"
    #   Audio:       "ParlaSpeech-HR/RS/PL/CZ-audio"  or shorthand "ParlaSpeech-audio"
    datasets: list = field(default_factory=lambda: ["ROG-Dialog", "ParlaSpeech", "ParlaSpeech-audio"])

    # Allow files marked is_large=True to download. Safety switch: set False to refuse
    # multi-GB files (audio parts are all is_large).
    confirm_large: bool = True

    # Force re-download even if file already exists.
    force: bool = False

    # Skip the unpack step.
    download_only: bool = False

    # Test mode: plan without fetching.
    test_mode: bool = False

cfg = Config()
print(cfg)

# %% [markdown]
# ---
#
# ## 2. Dataset registry
#
# Single source of truth for what can be downloaded. Sizes are approximate.
#
# **ParlaSpeech split across two releases per language:**
# - `ParlaSpeech-{LANG}` → v3.0 annotations (JSONL + vert + TextGrid, handle `11356/1833`)
# - `ParlaSpeech-{LANG}-audio` → audio FLACs (v2.0 for HR, v1.0 for RS/PL/CZ, separate handles)
#
# Shorthands: `"ParlaSpeech"` → all 4 annotation sets. `"ParlaSpeech-audio"` → all 4 audio sets.

# %%
DATASETS = {
    # ── Slovenian ────────────────────────────────────────────────────────────
    "ROG-Dialog": {
        "handle":   "http://hdl.handle.net/11356/2073",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2073",
        "files": [
            # (filename, approx_size_mb, is_large)
            ("ROG-Dialog.zip",       5,    True),
            ("ROG-Dialog_audio.zip", 1220, True),
        ],
        "notes": (
            "Dialogue corpus — sentiment, dialogue-act, filled-pause annotations. "
            "25 speakers, 5.2 h, EXB/TRS/TXT. "
            "ROG-Dialog_audio.zip (~1.2 GB) required for audio tasks."
        ),
    },
    "ROG": {
        "handle":   "http://hdl.handle.net/11356/2062",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2062",
        "files": [
            ("ROG.zip",         30,   False),
            ("ROG-Art.wav.zip", 1400, True),
        ],
        "notes": "Read-speech + arts subcorpora. ROG-Art.wav.zip (~1.4 GB) required for audio.",
    },
    "GOS": {
        "handle":   "http://hdl.handle.net/11356/1863",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1863",
        "files": [
            ("Gos.TEI.zip",  60, False),
            ("Gos.TRS.zip",  50, False),
            ("Gos.TXT.zip",  10, False),
            ("Gos.vert.zip", 30, False),
        ],
        "notes": (
            "Spontaneous speech transcriptions only. "
            "Audio is on restricted handle http://hdl.handle.net/11356/1973 — manual step required."
        ),
    },
    # ── ParlaSpeech v3.0 — annotations (JSONL + vert + TextGrid) ─────────────
    "ParlaSpeech-HR": {
        "handle":   "http://hdl.handle.net/11356/1833",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1833",
        "files": [
            ("ParlaSpeech-HR.v3.0.jsonl.gz",     2580,  False),   # 2.58 GB
            ("ParlaSpeech-HR.v3.0.vert.gz",       260,  False),   # 259 MB
            ("ParlaSpeech-HR.v3.0.textgrid.tgz", 4330,  True),    # 4.33 GB
        ],
        "notes": "Croatian parliamentary speech — v3.0 annotations only. Pair with ParlaSpeech-HR-audio.",
    },
    "ParlaSpeech-RS": {
        "handle":   "http://hdl.handle.net/11356/1833",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1833",
        "files": [
            ("ParlaSpeech-RS.v3.0.jsonl.gz",     586,  False),    # 586 MB
            ("ParlaSpeech-RS.v3.0.vert.gz",       73,  False),    # 73 MB
            ("ParlaSpeech-RS.v3.0.textgrid.tgz", 1270, True),     # 1.27 GB
        ],
        "notes": "Serbian parliamentary speech — v3.0 annotations only. Pair with ParlaSpeech-RS-audio.",
    },
    "ParlaSpeech-PL": {
        "handle":   "http://hdl.handle.net/11356/1833",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1833",
        "files": [
            ("ParlaSpeech-PL.v3.0.jsonl.gz", 393, False),         # 393 MB
            ("ParlaSpeech-PL.v3.0.vert.gz",  103, False),         # 103 MB
        ],
        "notes": "Polish parliamentary speech — v3.0 annotations only. No TextGrid. Pair with ParlaSpeech-PL-audio.",
    },
    "ParlaSpeech-CZ": {
        "handle":   "http://hdl.handle.net/11356/1833",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1833",
        "files": [
            ("ParlaSpeech-CZ.v3.0.jsonl.gz", 565, False),         # 565 MB
            ("ParlaSpeech-CZ.v3.0.vert.gz",  129, False),         # 129 MB
        ],
        "notes": "Czech parliamentary speech — v3.0 annotations only. No TextGrid. Pair with ParlaSpeech-CZ-audio.",
    },
    # ── ParlaSpeech audio — FLAC, separate handles ───────────────────────────
    "ParlaSpeech-HR-audio": {
        "handle":   "http://hdl.handle.net/11356/1914",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1914",
        "files": [
            ("ParlaSpeech-HR.v2.0.part1.tgz", 30480, True),       # 30.48 GB
            ("ParlaSpeech-HR.v2.0.part2.tgz", 42370, True),       # 42.37 GB
            ("ParlaSpeech-HR.v2.0.part3.tgz", 37610, True),       # 37.61 GB
            ("ParlaSpeech-HR.v2.0.part4.tgz", 41480, True),       # 41.48 GB
            ("ParlaSpeech-HR.v2.0.part5.tgz", 50130, True),       # 50.13 GB
            ("ParlaSpeech-HR.v2.0.part6.tgz",  4910, True),       # 4.91 GB
            ("README.txt",                          1, False),
        ],
        "notes": "Croatian FLAC audio — ~207 GB in 6 parts. Pair with ParlaSpeech-HR (v3.0 annotations).",
    },
    "ParlaSpeech-RS-audio": {
        "handle":   "http://hdl.handle.net/11356/1834",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1834",
        "files": [
            ("ParlaSpeech-RS.v1.0.part1.tgz", 36410, True),       # 36.41 GB
            ("ParlaSpeech-RS.v1.0.part2.tgz", 26620, True),       # 26.62 GB
            ("README.txt",                         1, False),
        ],
        "notes": "Serbian FLAC audio — ~63 GB in 2 parts. Pair with ParlaSpeech-RS (v3.0 annotations).",
    },
    "ParlaSpeech-PL-audio": {
        "handle":   "http://hdl.handle.net/11356/1686",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1686",
        "files": [
            ("ParlaSpeech-PL.v1.0.part1.tgz", 27870, True),       # 27.87 GB
            ("ParlaSpeech-PL.v1.0.part2.tgz", 30740, True),       # 30.74 GB
            ("README.txt",                         1, False),
        ],
        "notes": "Polish FLAC audio — ~59 GB in 2 parts. Pair with ParlaSpeech-PL (v3.0 annotations).",
    },
    "ParlaSpeech-CZ-audio": {
        "handle":   "http://hdl.handle.net/11356/1785",
        "base_url": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1785",
        "files": [
            ("ParlaSpeech-CZ.v1.0.part1.tgz", 46330, True),       # 46.33 GB
            ("ParlaSpeech-CZ.v1.0.part2.tgz", 40610, True),       # 40.61 GB
            ("ParlaSpeech-CZ.v1.0.part3.tgz", 43620, True),       # 43.62 GB
            ("ParlaSpeech-CZ.v1.0.part4.tgz", 22070, True),       # 22.07 GB
        ],
        "notes": "Czech FLAC audio — ~153 GB in 4 parts. Pair with ParlaSpeech-CZ (v3.0 annotations).",
    },
}

# ── Shorthands ────────────────────────────────────────────────────────────────
PARLASPEECH_LANGS       = ["ParlaSpeech-HR", "ParlaSpeech-RS", "ParlaSpeech-PL", "ParlaSpeech-CZ"]
PARLASPEECH_AUDIO_LANGS = ["ParlaSpeech-HR-audio", "ParlaSpeech-RS-audio",
                            "ParlaSpeech-PL-audio", "ParlaSpeech-CZ-audio"]

def resolve_datasets(requested):
    """Expand shorthands to full registry keys. Deduplicates preserving order."""
    out = []
    for name in requested:
        if name == "ParlaSpeech":
            out.extend(PARLASPEECH_LANGS)
        elif name == "ParlaSpeech-audio":
            out.extend(PARLASPEECH_AUDIO_LANGS)
        else:
            out.append(name)
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]

target_datasets = resolve_datasets(cfg.datasets)
print(f"Available: {list(DATASETS.keys())}")
print(f"Requested: {cfg.datasets}")
print(f"Resolved:  {target_datasets}")

for name in target_datasets:
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Valid keys: {list(DATASETS)}, or shorthands 'ParlaSpeech' / 'ParlaSpeech-audio'."
        )

# %% [markdown]
# ---
#
# ## 3. Plan the download
#
# Lists every file per dataset, marks what will be skipped.
# Nothing touches disk yet.

# %%
plan = []

for ds_name in target_datasets:
    spec    = DATASETS[ds_name]
    raw_dir = PROJECT_ROOT / "data" / "raw" / ds_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    udp.banner(f"Download plan — {ds_name}", char="-")
    print(f"notes: {spec['notes']}\n")

    for fname, size_mb, is_large in spec["files"]:
        dest    = raw_dir / fname
        url     = f"{spec['base_url']}/{fname}"
        already = dest.exists() and dest.stat().st_size > 0

        if already and not cfg.force:
            action, reason = "skip", "already downloaded"
        elif is_large and not cfg.confirm_large:
            action, reason = "skip", f"large ({size_mb} MB) — set confirm_large=True"
        elif cfg.test_mode:
            action, reason = "skip", "test_mode"
        else:
            action, reason = "download", "ok"

        plan.append({
            "dataset":  ds_name,
            "raw_dir":  raw_dir,
            "filename": fname,
            "url":      url,
            "dest":     dest,
            "size_mb":  size_mb,
            "is_large": is_large,
            "action":   action,
            "reason":   reason,
        })

        flag = "📥" if action == "download" else "⏭️ "
        print(f"  {flag} {fname:55s} ~{size_mb:>7} MB   [{action}: {reason}]")

    ds_plans   = [p for p in plan if p["dataset"] == ds_name]
    ds_on_disk = sum(p["size_mb"] for p in ds_plans if p["reason"] == "already downloaded")
    ds_to_dl   = sum(p["size_mb"] for p in ds_plans if p["action"] == "download")
    print(f"\n  on disk: ~{ds_on_disk:,} MB   to download: ~{ds_to_dl:,} MB   → {raw_dir}\n")

total_on_disk = sum(p["size_mb"] for p in plan if p["reason"] == "already downloaded")
total_to_dl   = sum(p["size_mb"] for p in plan if p["action"] == "download")
print(f"Totals — on disk: ~{total_on_disk:,} MB  |  to download: ~{total_to_dl:,} MB  (~{total_to_dl/1024:.1f} GB)")

# %% [markdown]
# ---
#
# ## 4. Download helper
#
# Streams to a `.part` file, renames on success — aborted download never leaves a half-finished file.

# %%
import requests
from tqdm.auto import tqdm

def download_file(url: str, dest, *, chunk_size: int = 1 << 20):
    """Download url to dest. Streams via .part file for atomicity."""
    from pathlib import Path
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with part.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name, leave=True,
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    part.rename(dest)
    return dest


# %% [markdown]
# ---
#
# ## 5. Execute the plan

# %%
if cfg.test_mode:
    print("🧪 TEST MODE: skipping all downloads")
else:
    for p in plan:
        if p["action"] != "download":
            print(f"⏭️  [{p['dataset']}] {p['filename']}  ({p['reason']})")
            continue
        print(f"📥 [{p['dataset']}] {p['filename']}  ←  {p['url']}")
        try:
            download_file(p["url"], p["dest"])
            print(f"   ✅ wrote {p['dest']}  ({p['dest'].stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"   ❌ failed: {e}")

print("\nDone.")

# %% [markdown]
# ---
#
# ## 6. Unpack archives
#
# `.zip` → unzip, `.tgz`/`.tar.gz` → tar, `.jsonl.gz`/`.gz` → single-file decompress.
#
# Idempotent: if the unpacked directory exists and is non-empty, skip.

# %%
import zipfile, tarfile, gzip, shutil

def unpack(archive, dest_dir) -> None:
    from pathlib import Path
    archive  = Path(archive)
    dest_dir = Path(dest_dir)
    name = archive.name
    stem = name
    for suf in (".tar.gz", ".tgz", ".jsonl.gz", ".zip", ".gz"):
        if name.endswith(suf):
            stem = name[: -len(suf)]
            break
    out = dest_dir / stem

    if out.exists() and any(out.iterdir()):
        print(f"  ⏭️  {name}  already unpacked → {out.relative_to(PROJECT_ROOT)}")
        return

    out.mkdir(parents=True, exist_ok=True)
    print(f"  📦 unpacking {name} → {out.relative_to(PROJECT_ROOT)}")

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out)
    elif name.endswith(".jsonl.gz") or name.endswith(".gz"):
        decomp_name = name[: -len(".gz")]
        with gzip.open(archive, "rb") as f_in, (out / decomp_name).open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        print(f"     ⚠️  no unpacker for {name}, skipping")
        return

    print(f"     ✅ unpacked")


if cfg.download_only:
    print("⏭️  download_only=True — skipping unpack")
elif cfg.test_mode:
    print("🧪 TEST MODE: skipping unpack")
else:
    for ds_name in target_datasets:
        udp.banner(f"Unpacking — {ds_name}", char="-")
        unpacked_dir = PROJECT_ROOT / "data" / "unpacked" / ds_name
        unpacked_dir.mkdir(parents=True, exist_ok=True)
        for p in [x for x in plan if x["dataset"] == ds_name]:
            if not p["dest"].exists():
                print(f"  ⏭️  {p['filename']}  (not on disk, can't unpack)")
                continue
            try:
                unpack(p["dest"], unpacked_dir)
            except Exception as e:
                print(f"     ❌ unpack failed for {p['filename']}: {e}")

print("\nDone.")


# %% [markdown]
# ---
#
# ## 7. What's on disk now

# %%
def show_tree(root, max_depth=3, max_files_per_dir=4, max_dirs_at_depth=4):
    from pathlib import Path
    root = Path(root)
    if not root.exists():
        print(f"  (no such dir: {root})")
        return

    def _walk(d, depth, indent):
        if depth > max_depth:
            return
        try:
            children = sorted(d.iterdir())
        except PermissionError:
            return
        dirs  = [c for c in children if c.is_dir()]
        files = [c for c in children if c.is_file()]

        # Cap dir listing at deeper levels (hash folders explode)
        cap_dirs   = depth >= 2
        shown_dirs = dirs[:max_dirs_at_depth] if cap_dirs else dirs
        for sd in shown_dirs:
            print(f"{indent}📁 {sd.name}/")
            _walk(sd, depth + 1, indent + "  ")
        if cap_dirs and len(dirs) > max_dirs_at_depth:
            print(f"{indent}... and {len(dirs) - max_dirs_at_depth} more dirs")

        for f in files[:max_files_per_dir]:
            print(f"{indent}📄 {f.name}  {f.stat().st_size / 1e6:.1f} MB")
        if len(files) > max_files_per_dir:
            print(f"{indent}... and {len(files) - max_files_per_dir} more files")

    print(f"  {root.relative_to(PROJECT_ROOT)}/")
    _walk(root, depth=1, indent="    ")

for ds_name in target_datasets:
    print(f"\n=== {ds_name} ===")
    print("raw:")
    show_tree(PROJECT_ROOT / "data" / "raw" / ds_name)
    print("unpacked:")
    show_tree(PROJECT_ROOT / "data" / "unpacked" / ds_name)

# %% [markdown]
# ---
#
# ## Next
#
# - **ROG** → `11a_prep_ROG-art.ipynb`
# - **ROG-Dialog** → `11b_prep_ROG-dia.ipynb`
# - **ParlaSpeech-{HR,RS,PL,CZ}** → `11c_prep_parlaspeech.ipynb` (`cfg.lang` pins
#   one language; empty = every ParlaSpeech-{LANG} under `data/unpacked/`).
# - **ParlaSpeech-HR benchmark** → `11d_prep_parlaspeech_benchmark_v3.ipynb` (v3)
#   or `11e_prep_parlaspeech_benchmark_v1.ipynb` (v1); each consumes a pre-built
#   benchmark bundle under `data/benchmarking/`.
# - **GOS** → arrange restricted audio access first.
#
# Re-running with the same config just prints skip lines — idempotent.
