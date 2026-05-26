"""
utils_dataprep.py
=================
Shared utilities for chapter 1 (data preparation).

Everything in here is dataset-agnostic. The per-dataset scripts
(prep_ROG, prep_GOS, prep_ParlaSpeech) import from here.

Contents
--------
- Path helpers (PROJECT_ROOT resolution, relative-path conversion)
- Canonical JSONL I/O (read_jsonl, write_jsonl)
- Schema validation (validate_instance, validate_jsonl)
- Cleaning (drop_invalid, drop_duplicates, drop_missing_label)
- Split assignment (assign_splits, group-aware)
- Audio helpers (resample_to_16k_mono, get_wav_duration)
- Logging helpers (banner, test_mode_clamp)

Conventions
-----------
- All paths in the canonical JSONL are project-relative strings.
- We use Path objects internally and convert to/from strings at the boundaries.
- Failure modes are loud: validators print offending instance_ids.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


# --------------------------------------------------------------------------- #
# 1. Paths
# --------------------------------------------------------------------------- #

def find_project_root(start: Path | None = None) -> Path:
    """
    Walk up from `start` (default: this file's location) looking for the
    project root, identified by the presence of `BLUEPRINT.md` or `README.md`
    next to a `data/` directory.

    Falls back to `Path.cwd()` if nothing matches — useful for notebooks
    started in odd places.
    """
    start = (start or Path(__file__).resolve()).resolve()
    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / "BLUEPRINT.md").exists() or ((p / "README.md").exists() and (p / "data").exists()):
            return p
    # Last-resort fallback: cwd. Notebook users should set this explicitly.
    return Path.cwd().resolve()


PROJECT_ROOT: Path = find_project_root()


def to_project_relative(path: str | Path) -> str:
    """
    Convert an absolute path into a project-relative POSIX-style string.
    If the path is already relative, return its POSIX form unchanged.
    If the path doesn't live under PROJECT_ROOT, return it as-is (string)
    and print a warning — callers should not rely on this.
    """
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        print(f"⚠️  to_project_relative: path is outside PROJECT_ROOT, keeping absolute: {p}", file=sys.stderr)
        return p.as_posix()


def from_project_relative(path: str | Path) -> Path:
    """Resolve a project-relative string into an absolute Path."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


# --------------------------------------------------------------------------- #
# 2. JSONL I/O
# --------------------------------------------------------------------------- #

def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file fully into memory as a list of dicts."""
    path = from_project_relative(path)
    out = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} is not valid JSON ({e})") from e
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream a JSONL file line by line. Use for large files."""
    path = from_project_relative(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} is not valid JSON ({e})") from e


def write_jsonl(records: Iterable[dict], path: str | Path, *, ensure_parent: bool = True) -> int:
    """Write an iterable of dicts as JSONL. Returns the number of lines written."""
    path = from_project_relative(path)
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# 3. Canonical schema validation
# --------------------------------------------------------------------------- #

_REQUIRED_KEYS: dict[str, type | tuple[type, ...]] = {
    "instance_id": str,
    "dataset":     str,
    "audio_path":  str,
    "split":       str,
    "labels":      dict,
    "metadata":    dict,
}

_OPTIONAL_KEYS: dict[str, type | tuple[type, ...]] = {
    "file_id":        str,
    "start_t":        (int, float),
    "end_t":          (int, float),
    "speaker":        str,
    "text":           str,
    "frame_rate_hz":  int,
}

_VALID_SPLITS = {"train", "dev", "test"}


def validate_instance(obj: dict) -> list[str]:
    """
    Check a single instance dict against the canonical schema.
    Returns a list of error strings (empty list = valid).
    Does not raise.
    """
    errors: list[str] = []

    if not isinstance(obj, dict):
        return [f"not a dict: got {type(obj).__name__}"]

    iid = obj.get("instance_id", "<no instance_id>")

    # Required keys present and correctly typed
    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in obj:
            errors.append(f"{iid}: missing required key '{key}'")
            continue
        if not isinstance(obj[key], expected_type):
            actual = type(obj[key]).__name__
            errors.append(f"{iid}: '{key}' has wrong type (expected {expected_type}, got {actual})")

    # Optional keys: if present, must have correct type
    for key, expected_type in _OPTIONAL_KEYS.items():
        if key in obj and not isinstance(obj[key], expected_type):
            actual = type(obj[key]).__name__
            errors.append(f"{iid}: optional '{key}' has wrong type (expected {expected_type}, got {actual})")

    # Split value
    if obj.get("split") not in _VALID_SPLITS:
        errors.append(f"{iid}: split must be one of {_VALID_SPLITS}, got {obj.get('split')!r}")

    # start_t / end_t consistency
    if "start_t" in obj and "end_t" in obj:
        if obj["end_t"] < obj["start_t"]:
            errors.append(f"{iid}: end_t ({obj['end_t']}) < start_t ({obj['start_t']})")

    # frame_rate_hz only meaningful if at least one label is a list
    if "frame_rate_hz" in obj:
        labels = obj.get("labels", {})
        has_sequence_label = any(isinstance(v, list) for v in labels.values())
        if not has_sequence_label:
            errors.append(f"{iid}: frame_rate_hz set but no sequence labels in 'labels'")

    return errors


def validate_jsonl(
    records: Iterable[dict],
    *,
    strict: bool = False,
    max_report: int = 20,
) -> tuple[int, int, list[str]]:
    """
    Validate an iterable of instance dicts.

    Returns
    -------
    n_total, n_valid, errors (list of error strings, truncated to max_report)

    If strict=True, raises ValueError on the first error.
    """
    n_total = 0
    n_valid = 0
    all_errors: list[str] = []
    for obj in records:
        n_total += 1
        errs = validate_instance(obj)
        if errs:
            if strict:
                raise ValueError(errs[0])
            all_errors.extend(errs)
        else:
            n_valid += 1
    return n_total, n_valid, all_errors[:max_report]


# --------------------------------------------------------------------------- #
# 4. Cleaning
# --------------------------------------------------------------------------- #

def drop_invalid(records: Iterable[dict]) -> tuple[list[dict], int]:
    """
    Drop records that fail schema validation.
    Returns (kept_records, n_dropped).
    """
    kept = []
    dropped = 0
    for r in records:
        if not validate_instance(r):
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


def drop_duplicates(records: Iterable[dict], *, key: str = "instance_id") -> tuple[list[dict], int]:
    """Drop records with duplicate `key` values, keeping the first."""
    seen: set[str] = set()
    kept = []
    dropped = 0
    for r in records:
        v = r.get(key)
        if v is None or v in seen:
            dropped += 1
            continue
        seen.add(v)
        kept.append(r)
    return kept, dropped


def drop_missing_label(records: Iterable[dict], label_key: str) -> tuple[list[dict], int]:
    """Drop records whose labels dict lacks `label_key` or has None for it."""
    kept = []
    dropped = 0
    for r in records:
        v = r.get("labels", {}).get(label_key)
        if v is None or (isinstance(v, str) and v.strip() == ""):
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def drop_empty_text(records: Iterable[dict]) -> tuple[list[dict], int]:
    """Drop records whose `text` field is missing, empty, or whitespace-only."""
    kept = []
    dropped = 0
    for r in records:
        t = r.get("text", "")
        if not isinstance(t, str) or not t.strip():
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def clean(
    records: list[dict],
    *,
    label_key: str | None = None,
    require_text: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the standard cleaning pipeline. Each step prints how many it dropped
    so you can see where data is being lost.
    """
    n_in = len(records)

    records, n_inv = drop_invalid(records)
    if verbose: print(f"  drop_invalid:        -{n_inv} ({len(records)} left)")

    records, n_dup = drop_duplicates(records)
    if verbose: print(f"  drop_duplicates:     -{n_dup} ({len(records)} left)")

    if label_key is not None:
        records, n_lab = drop_missing_label(records, label_key)
        if verbose: print(f"  drop_missing_label:  -{n_lab} ({len(records)} left)")

    if require_text:
        records, n_txt = drop_empty_text(records)
        if verbose: print(f"  drop_empty_text:     -{n_txt} ({len(records)} left)")

    if verbose:
        print(f"  clean: {n_in} → {len(records)} ({n_in - len(records)} dropped total)")
    return records


# --------------------------------------------------------------------------- #
# 5. Split assignment
# --------------------------------------------------------------------------- #

def _stable_hash_float(s: str) -> float:
    """
    Map a string to a float in [0, 1) deterministically across Python runs.
    `hash()` is randomised by PYTHONHASHSEED, so we use md5.
    """
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    # First 8 hex chars give 32 bits — plenty for a fraction.
    return int(h[:8], 16) / 0xFFFFFFFF


def assign_splits(
    records: list[dict],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    group_key: str = "file_id",
    seed: str = "speech-ml-pipeline",
    overwrite: bool = False,
) -> list[dict]:
    """
    Assign train/dev/test splits to each record, deterministically.

    - Instances sharing the same `group_key` value land in the same split
      (so no leakage across e.g. files or speakers).
    - If a record already has a valid `split`, it is left alone unless
      `overwrite=True`.
    - If `group_key` is missing on a record, it falls back to grouping by
      `instance_id` (effectively per-instance random split).

    Returns the same list, mutated in place. Also returns it for chaining.
    """
    train_r, dev_r, test_r = ratios
    if abs(train_r + dev_r + test_r - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    train_cut = train_r
    dev_cut = train_r + dev_r  # rest is test

    group_to_split: dict[str, str] = {}

    for r in records:
        if not overwrite and r.get("split") in _VALID_SPLITS:
            continue
        group_val = r.get(group_key) or r.get("instance_id") or ""
        if group_val not in group_to_split:
            x = _stable_hash_float(seed + "::" + str(group_val))
            if x < train_cut:
                group_to_split[group_val] = "train"
            elif x < dev_cut:
                group_to_split[group_val] = "dev"
            else:
                group_to_split[group_val] = "test"
        r["split"] = group_to_split[group_val]

    return records


def split_summary(records: Iterable[dict]) -> dict[str, int]:
    """Count how many records are in each split."""
    counts = {"train": 0, "dev": 0, "test": 0, "other": 0}
    for r in records:
        s = r.get("split", "other")
        counts[s if s in counts else "other"] = counts.get(s if s in counts else "other", 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# 6. instance_id construction
# --------------------------------------------------------------------------- #

_INSTANCE_ID_SANITISE_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def _sanitise(part: str) -> str:
    """Make a string safe to use inside an instance_id (no spaces/slashes/etc)."""
    return _INSTANCE_ID_SANITISE_RE.sub("-", str(part))


def make_instance_id(
    dataset: str,
    file_id: str,
    speaker: str | None = None,
    start_t: float | None = None,
    end_t: float | None = None,
) -> str:
    """
    Build a canonical instance_id from its parts.

    Examples
    --------
    >>> make_instance_id("ROG", "Art-S01-V01", "SPK0", 2.190, 4.580)
    'ROG_Art-S01-V01_SPK0_2.190_4.580'
    >>> make_instance_id("GOS", "Artur-N-G0001")
    'GOS_Artur-N-G0001'
    """
    parts = [_sanitise(dataset), _sanitise(file_id)]
    if speaker is not None:
        parts.append(_sanitise(speaker))
    if start_t is not None:
        parts.append(f"{float(start_t):.3f}")
    if end_t is not None:
        parts.append(f"{float(end_t):.3f}")
    return "_".join(parts)


# --------------------------------------------------------------------------- #
# 7. Audio helpers
# --------------------------------------------------------------------------- #

def get_wav_duration(path: str | Path) -> float:
    """Return WAV duration in seconds using the stdlib only (no extra deps)."""
    import wave
    p = from_project_relative(path)
    with wave.open(str(p), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate)


def resample_to_16k_mono(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    start_t: float | None = None,
    end_t: float | None = None,
) -> Path:
    """
    Read `src_path`, optionally trim to [start_t, end_t], resample to 16 kHz
    mono, write to `dst_path`. Uses soundfile + numpy + (optional) librosa.

    Lazy imports so importing utils_dataprep doesn't drag in librosa unless
    you actually cut audio.
    """
    import numpy as np
    import soundfile as sf

    src = from_project_relative(src_path)
    dst = from_project_relative(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    data, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)  # to mono

    if start_t is not None or end_t is not None:
        s = int((start_t or 0.0) * sr)
        e = int((end_t if end_t is not None else len(data) / sr) * sr)
        data = data[s:e]

    if sr != 16000:
        # Prefer librosa for high-quality resampling; fall back to scipy.
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, 16000)
            data = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
        sr = 16000

    sf.write(str(dst), data, sr, subtype="PCM_16")
    return dst


# --------------------------------------------------------------------------- #
# 8. Logging helpers
# --------------------------------------------------------------------------- #

def banner(title: str, *, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(title)
    print(char * width)


@dataclass
class TestMode:
    """
    Standard test-mode clamp. Every runnable script should accept one.

    Usage
    -----
    >>> tm = TestMode(enabled=True, n=10)
    >>> records = tm.maybe_truncate(records)
    """
    enabled: bool = False
    n: int = 10
    epochs: int = 1
    batch_size: int = 2

    def maybe_truncate(self, items: list) -> list:
        if not self.enabled:
            return items
        print(f"🧪 TEST MODE: truncating {len(items)} → {self.n}")
        return items[: self.n]

    def banner(self) -> None:
        if self.enabled:
            banner("🧪 TEST MODE ENABLED", char="-")


# --------------------------------------------------------------------------- #
# 9. Self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    """Quick sanity check. Run with: python utils_dataprep.py"""
    banner("utils_dataprep self-test")
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")

    sample = {
        "instance_id": make_instance_id("ROG", "Art-S01-V01", "SPK0", 2.190, 4.580),
        "dataset": "ROG",
        "audio_path": "data/cut_audio/ROG/example.wav",
        "split": "train",
        "labels": {"sentiment": "neutralPositive"},
        "metadata": {},
        "file_id": "Art-S01-V01",
        "start_t": 2.190,
        "end_t": 4.580,
        "speaker": "SPK0",
        "text": "Tudi.",
    }
    errs = validate_instance(sample)
    assert not errs, f"sample should validate, got: {errs}"
    print(f"✅ sample validates: {sample['instance_id']}")

    bad = {"instance_id": "x", "dataset": "X", "audio_path": "x", "split": "invalid",
           "labels": {}, "metadata": {}}
    errs = validate_instance(bad)
    assert errs, "bad sample should fail"
    print(f"✅ bad sample fails as expected: {errs[0]}")

    # split assignment
    recs = [
        {**sample, "instance_id": f"X_{i}", "file_id": f"file_{i // 3}", "split": "train"}
        for i in range(9)
    ]
    for r in recs:
        r["split"] = "train"  # all start as train
    assign_splits(recs, overwrite=True, ratios=(0.6, 0.2, 0.2))
    counts = split_summary(recs)
    print(f"✅ split assignment: {counts}")
    # All instances with the same file_id should share a split
    groups: dict[str, set[str]] = {}
    for r in recs:
        groups.setdefault(r["file_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in groups.values()), "leak across splits!"
    print("✅ no split leakage across file_id groups")

    print("\nAll good.")


if __name__ == "__main__":
    _selftest()
