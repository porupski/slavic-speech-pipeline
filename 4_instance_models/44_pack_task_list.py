#!/usr/bin/env python3
"""44_pack_task_list — slim-pack chapter-4 run dirs for off-host download.

Walks one or more ``runs/<task_list>/`` directories and emits a
``.tar.gz`` that drops the heavy per-task weights (``_model/``) and raw
prediction dumps (``predictions.json``) while keeping everything the
analyzer and the HTML write-ups need: the task-list CSV + summary PNG,
``tasks.json`` / ``args.json`` provenance, and per-task ``config.json``,
``task.json``, ``epoch_summary.json`` + top-level confusion / scatter PNGs.

Default excludes shrink a typical mixed-recipe task list from ~40 GB to
~5 MB — enough to scp to a laptop and feed into
``45_analyze_task_list.py``.

Usage:
  # Pack one run dir to /tmp:
  python 44_pack_task_list.py ../runs/2026-11-21_143634_task_list

  # Pack multiple dirs into one tarball:
  python 44_pack_task_list.py ../runs/2026-11-21_143634_task_list \\
                               ../runs/2026-11-22_011427_task_list \\
                               --out /tmp/p43_results.tar.gz

  # Dry-run (list what would be packed + size, don't write the tar):
  python 44_pack_task_list.py --dry_run ../runs/2026-11-21_143634_task_list

  # Keep predictions.json (needed for CORN / threshold-sweep analyses):
  python 44_pack_task_list.py --keep_predictions ../runs/*_task_list

Exclude policy:
  - Any directory named exactly ``_model`` or ``best_model`` (weights).
  - Any file named ``predictions.json`` (large per-epoch preds dump).
  - Any file larger than ``--max_mb`` (default 20 MB; PNGs are usually
    well under this, so the cap just bites ``predictions.json`` edge cases).
"""

import argparse
import sys
import tarfile
from pathlib import Path


DEFAULT_EXCLUDE_DIRS = {"_model", "best_model"}
DEFAULT_EXCLUDE_FILES = {"predictions.json"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dirs", nargs="+",
                   help="one or more runs/<task_list>/ directories")
    p.add_argument("--out", default=None,
                   help="output tarball path (default: /tmp/<basename>.tar.gz "
                        "or /tmp/<first_basename>_plus.tar.gz for multiple inputs)")
    p.add_argument("--max_mb", type=float, default=20.0,
                   help="skip any file larger than this (in MB)")
    p.add_argument("--keep_predictions", action="store_true",
                   help="keep predictions.json files (default: strip)")
    p.add_argument("--keep_weights", action="store_true",
                   help="keep _model/ and best_model/ dirs (default: strip)")
    p.add_argument("--dry_run", action="store_true",
                   help="list what would be packed + estimated size, no output")
    return p.parse_args()


def should_skip(path: Path, root: Path, exclude_dirs: set, exclude_files: set,
                max_bytes: int) -> tuple[bool, str]:
    """Return (skip?, reason). ``path`` is absolute; ``root`` is the run_dir."""
    rel = path.relative_to(root.parent)
    parts = set(rel.parts)
    if parts & exclude_dirs:
        return True, "weights dir"
    if path.is_file() and path.name in exclude_files:
        return True, "prediction dump"
    if path.is_file():
        try:
            if path.stat().st_size > max_bytes:
                return True, f"too large ({path.stat().st_size / 1e6:.1f} MB)"
        except OSError:
            return True, "stat failed"
    return False, ""


def walk_kept(root: Path, exclude_dirs: set, exclude_files: set,
              max_bytes: int):
    """Yield (abs_path, arcname) pairs that pass the exclude filter. Directories
    whose name is in exclude_dirs are pruned so we don't walk into them."""
    for p in root.rglob("*"):
        rel = p.relative_to(root.parent)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        skip, _ = should_skip(p, root, exclude_dirs, exclude_files, max_bytes)
        if skip:
            continue
        yield p, str(rel)


def summarize(root: Path, kept: list[tuple[Path, str]]) -> dict:
    kept_files = [(p, arc) for p, arc in kept if p.is_file()]
    kept_bytes = sum(p.stat().st_size for p, _ in kept_files)
    # Rough "original" size for comparison (full subtree, ignoring errors).
    total_bytes = 0
    for p in root.rglob("*"):
        if p.is_file():
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
    return {
        "root": str(root),
        "files_kept": len(kept_files),
        "bytes_kept": kept_bytes,
        "bytes_total": total_bytes,
        "pct_kept": (100.0 * kept_bytes / total_bytes) if total_bytes else 0.0,
    }


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def default_out_path(run_dirs: list[Path]) -> Path:
    if len(run_dirs) == 1:
        return Path("/tmp") / f"{run_dirs[0].name}.tar.gz"
    return Path("/tmp") / f"{run_dirs[0].name}_plus{len(run_dirs) - 1}.tar.gz"


def main() -> None:
    args = parse_args()
    run_dirs = [Path(d).resolve() for d in args.run_dirs]
    missing = [d for d in run_dirs if not d.exists()]
    if missing:
        print(f"ERROR: missing run dir(s): {missing}", file=sys.stderr); sys.exit(2)

    exclude_dirs = set() if args.keep_weights else set(DEFAULT_EXCLUDE_DIRS)
    exclude_files = set() if args.keep_predictions else set(DEFAULT_EXCLUDE_FILES)
    max_bytes = int(args.max_mb * 1024 * 1024)

    print(f"excluding dirs : {sorted(exclude_dirs) or '(none)'}")
    print(f"excluding files: {sorted(exclude_files) or '(none)'}")
    print(f"max file size  : {args.max_mb:.1f} MB")

    all_kept: list[tuple[Path, str]] = []
    totals = []
    for rd in run_dirs:
        kept = list(walk_kept(rd, exclude_dirs, exclude_files, max_bytes))
        all_kept.extend(kept)
        s = summarize(rd, kept)
        totals.append(s)
        print(f"\n{rd.name}: kept {s['files_kept']} files, "
              f"{human(s['bytes_kept'])} of {human(s['bytes_total'])} "
              f"({s['pct_kept']:.1f}%)")

    total_kept = sum(s["bytes_kept"] for s in totals)
    total_src = sum(s["bytes_total"] for s in totals)
    print(f"\nTOTAL: {human(total_kept)} of {human(total_src)} "
          f"({100.0 * total_kept / max(1, total_src):.1f}%) across "
          f"{len(run_dirs)} run dirs")

    if args.dry_run:
        print("\n(dry_run — no tarball written)")
        return

    out = Path(args.out) if args.out else default_out_path(run_dirs)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {out} …")
    with tarfile.open(out, "w:gz") as tar:
        for p, arc in all_kept:
            tar.add(p, arcname=arc, recursive=False)
    print(f"✓ wrote {human(out.stat().st_size)} → {out}")


if __name__ == "__main__":
    main()
