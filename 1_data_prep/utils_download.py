"""
utils_download.py
=================
Helpers behind ``10_download_data``. The notebook stays thin — config, plan,
execute — and imports everything real from here.

Two source types are supported:

- ``clarin`` — direct HTTP to the CLARIN.SI repository. Files are streamed to
  a ``.part`` sibling and renamed on success (aborted downloads leave no
  half-finished file).
- ``hf``     — Hugging Face Hub. ``snapshot_download`` is used with
  ``allow_patterns``/``ignore_patterns`` so we grab audio + JSONL and skip the
  auto-generated parquet.

The registry itself lives in ``10a_dataset_registry.json``.
"""

from __future__ import annotations

import gzip
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import requests

# Force text-mode progress bars everywhere, including inside huggingface_hub.
# tqdm.auto defaults to ipywidgets in a Jupyter kernel, which VS Code's Jupyter
# frontend rejects with "Cannot read properties of undefined
# (reading 'ipywidgetsKernel')" — annoying, and it hides the actual progress.
import tqdm.auto as _tqdm_auto
from tqdm.std import tqdm as _std_tqdm
_tqdm_auto.tqdm = _std_tqdm
from tqdm.auto import tqdm  # noqa: E402  — text-mode via the patch above


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

REGISTRY_FILENAME = "10a_dataset_registry.json"


def load_registry(chapter_dir: Path) -> dict[str, Any]:
    """Load the JSON registry that sits next to ``10_download_data``."""
    path = chapter_dir / REGISTRY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"registry not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_datasets(requested: list[str], registry: dict[str, Any]) -> list[str]:
    """Expand shorthands (e.g. ``ParlaSpeech``) into concrete registry keys.

    Deduplicates while preserving the caller's order. Unknown names raise.
    """
    shorthands = registry.get("shorthands", {})
    datasets   = registry["datasets"]

    out: list[str] = []
    for name in requested:
        if name in shorthands:
            out.extend(shorthands[name])
        elif name in datasets:
            out.append(name)
        else:
            valid = sorted(list(datasets) + list(shorthands))
            raise ValueError(f"Unknown dataset {name!r}. Valid keys: {valid}")

    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

def iter_files(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Yield the individual downloadable items for a dataset entry.

    - ``clarin`` datasets expose one item per file in ``spec['files']``.
    - ``hf`` datasets expose one virtual item representing the whole snapshot.
    """
    if spec["source"] == "clarin":
        return list(spec["files"])
    if spec["source"] == "hf":
        return [{
            "name":     f"{spec['hf_repo']} (HF snapshot)",
            "size_mb":  spec.get("size_mb", 0),
            "is_large": spec.get("is_large", True),
        }]
    raise ValueError(f"Unknown source: {spec['source']!r}")


# --------------------------------------------------------------------------- #
# CLARIN downloads
# --------------------------------------------------------------------------- #

def download_clarin_file(url: str, dest: Path, *,
                         force: bool = False, chunk_size: int = 1 << 20) -> Path:
    """Stream ``url`` to ``dest`` via a ``.part`` sibling for atomicity.

    Resume-friendly: if ``dest.part`` already exists and ``force`` is False, an
    HTTP ``Range`` request continues from the partial file's current size. If
    the server ignores ``Range`` (returns 200 instead of 206), we discard the
    partial and restart. ``force=True`` deletes both the partial and the final
    file first.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if force:
        part.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)

    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()

        # Server ignored our Range and served the whole file → discard partial.
        if resume_from > 0 and r.status_code == 200:
            print(f"   ⚠️  server didn't honor Range on {dest.name}; restarting from scratch")
            part.unlink(missing_ok=True)
            resume_from = 0

        remaining = int(r.headers.get("content-length", 0))
        total = remaining + resume_from
        mode  = "ab" if resume_from > 0 else "wb"
        with part.open(mode) as f, tqdm(
            initial=resume_from, total=total,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name, leave=True,
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    part.rename(dest)
    return dest


# --------------------------------------------------------------------------- #
# Hugging Face downloads
# --------------------------------------------------------------------------- #

def download_hf_snapshot(spec: dict[str, Any], project_root: Path, *,
                         force: bool = False, max_workers: int = 16) -> Path:
    """Download a Hugging Face repo snapshot into ``spec['target_dir']``.

    ``snapshot_download`` is already idempotent and resume-friendly: files
    already fully on disk are skipped, partial ``.incomplete`` fetches are
    resumed. ``force=True`` forwards ``force_download`` to redownload
    everything. ``max_workers`` bumps HF's default parallelism (8) — helpful
    when the snapshot is thousands of small files.

    ``allow_patterns``/``ignore_patterns`` are read from the registry entry so
    we grab the JSONL + audio (+ textgrids for v3) and skip the auto-generated
    parquet under ``data/``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is not installed. Add it via requirements_cpu.txt "
            "(or requirements_cuda.txt) and re-run the setup script, or "
            "`pip install huggingface_hub` into the active env."
        ) from exc

    target = project_root / spec["target_dir"]
    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=spec["hf_repo"],
        repo_type=spec.get("hf_repo_type", "dataset"),
        local_dir=str(target),
        allow_patterns=spec.get("allow_patterns"),
        ignore_patterns=spec.get("ignore_patterns"),
        force_download=force,
        max_workers=max_workers,
    )
    return target


# --------------------------------------------------------------------------- #
# Unpack
# --------------------------------------------------------------------------- #

_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".jsonl.gz", ".zip", ".gz")


def unpack(archive: Path, dest_dir: Path, project_root: Path) -> None:
    """Unpack a single archive into ``dest_dir/<stem>``.

    Idempotent: if the target dir exists and is non-empty, skip.
    """
    name = archive.name
    stem = name
    for suf in _ARCHIVE_SUFFIXES:
        if name.endswith(suf):
            stem = name[: -len(suf)]
            break
    out = dest_dir / stem

    if out.exists() and any(out.iterdir()):
        print(f"  ⏭️  {name}  already unpacked → {out.relative_to(project_root)}")
        return

    out.mkdir(parents=True, exist_ok=True)
    print(f"  📦 unpacking {name} → {out.relative_to(project_root)}")

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out)
    elif name.endswith((".jsonl.gz", ".gz")):
        decomp_name = name[: -len(".gz")]
        with gzip.open(archive, "rb") as f_in, (out / decomp_name).open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        print(f"     ⚠️  no unpacker for {name}, skipping")
        return

    print("     ✅ unpacked")


# --------------------------------------------------------------------------- #
# Disk-tree pretty-print
# --------------------------------------------------------------------------- #

def show_tree(root: Path, project_root: Path, *,
              max_depth: int = 3, max_files_per_dir: int = 4,
              max_dirs_at_depth: int = 4) -> None:
    """Compact tree view of a directory. Caps to keep hash-folder corpora sane."""
    if not root.exists():
        print(f"  (no such dir: {root})")
        return

    def _walk(d: Path, depth: int, indent: str) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(d.iterdir())
        except PermissionError:
            return
        dirs  = [c for c in children if c.is_dir()]
        files = [c for c in children if c.is_file()]

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

    print(f"  {root.relative_to(project_root)}/")
    _walk(root, depth=1, indent="    ")
