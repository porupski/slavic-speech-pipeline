"""
utils_audio_splitter.py
=======================
Reusable audio cutting for chapter-1 prep notebooks.

The contract is deliberately dumb: a prep notebook *decides* the cuts and writes
`start_t` / `end_t` (rounded to 3 dp) into each record. This module just
*executes* what the JSONL says — resolve each record to a source audio file
(+ optional slice bounds), produce a 16 kHz mono PCM-16 WAV at the record's
`audio_path`, drop records it can't cut.

What varies between corpora — where the source lives, whether to slice or
whole-file convert, absolute vs relative time — lives in a **resolver**:
a callable `record -> SourceRef | None`. The cutting core never knows which
corpus it's looking at.

    record  ──resolver──▶  SourceRef(path, start_t, end_t)  ──core──▶  16k mono WAV

Two resolver factories ship here:
- `make_stem_scan_resolver`   — session-WAV corpora (ROG): scan a tree once,
                                look up by `metadata.source_file` stem, slice
                                using the record's own start_t/end_t.
- `make_record_path_resolver` — pre-cut corpora (ParlaSpeech): source path is a
                                field on the record; whole-file convert by
                                default, or slice if you name slice fields.

Prep notebooks can write their own resolver — it's just a function.

Performance:
- `num_workers == 0` (default): sequential + an LRU source cache, records sorted
  by source path. Best when many records share one big source (ROG).
- `num_workers > 0`: a pool of workers, no cache. Best when every record has its
  own small source (ParlaSpeech's hundreds of thousands of FLACs).
  `parallel_backend="process"` (default) gives *true* multi-core parallelism —
  decode + resample are CPU-bound and `librosa.resample` spends most of its time
  holding the GIL, so threads barely scale on this workload. Processes do.
  `parallel_backend="thread"` is kept for I/O-bound or fork-unfriendly cases.
  Workers receive only `(instance_id, audio_path, SourceRef)` — small and
  picklable — because resolution happens up front in the parent, so the big
  resolver index is never shipped across the process boundary.

  Note: the process backend relies on the default `fork` start method (Linux),
  so it works inside Jupyter without a `__main__` guard. On spawn-only platforms
  (macOS/Windows notebooks) use `parallel_backend="thread"`.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import utils_dataprep as udp

# A resolver maps one record to its source audio (+ optional slice), or None
# if the source can't be found (record then gets dropped).
Resolver = Callable[[dict], "SourceRef | None"]

_TIME_DP = 3  # canonical rounding for all start/end timestamps


def round_time(t: float | int | None) -> float | None:
    """Round a timestamp to the canonical 3 dp. None passes through."""
    return None if t is None else round(float(t), _TIME_DP)


# --------------------------------------------------------------------------- #
# SourceRef — the resolver's output
# --------------------------------------------------------------------------- #

@dataclass
class SourceRef:
    """
    Where a record's audio comes from.

    - `path`: absolute Path to the source audio (any format soundfile reads).
    - `start_t` / `end_t`: slice bounds **relative to that source file**, in
      seconds. Both None ⇒ whole-file convert (no slicing).
    """
    path: Path
    start_t: float | None = None
    end_t: float | None = None

    @property
    def is_slice(self) -> bool:
        return self.start_t is not None or self.end_t is not None


# --------------------------------------------------------------------------- #
# Source cache (sequential mode only)
# --------------------------------------------------------------------------- #

class SourceCache:
    """Tiny LRU of decoded source WAVs (data, sr) keyed by absolute path str."""

    def __init__(self, max_size: int = 4) -> None:
        self.max_size = max(1, int(max_size))
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()

    def get(self, path: Path):
        import soundfile as sf
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data, sr = sf.read(key, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)  # to mono
        self._cache[key] = (data, sr)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return data, sr


def _resample_to_16k(data, sr: int):
    """Resample a 1-D float32 array to 16 kHz. librosa, scipy fallback."""
    if sr == 16000:
        return data
    try:
        import librosa
        return librosa.resample(data, orig_sr=sr, target_sr=16000)
    except ImportError:
        from math import gcd
        import numpy as np
        from scipy.signal import resample_poly
        g = gcd(sr, 16000)
        return resample_poly(data, 16000 // g, sr // g).astype(np.float32)


# --------------------------------------------------------------------------- #
# The cut itself
# --------------------------------------------------------------------------- #

def _cut_cached(ref: SourceRef, dst: Path, cache: SourceCache) -> None:
    """Slice (or whole-file) from a cached source array, resample, write."""
    import soundfile as sf
    data, sr = cache.get(ref.path)
    if ref.is_slice:
        st = round_time(ref.start_t) or 0.0
        et = round_time(ref.end_t) if ref.end_t is not None else len(data) / sr
        s = max(0, int(st * sr))
        e = min(len(data), int(et * sr))
        if e <= s:
            raise ValueError(f"empty slice after sample-conversion (s={s}, e={e})")
        data = data[s:e]
    data = _resample_to_16k(data, sr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), data, 16000, subtype="PCM_16")


def _cut_independent(ref: SourceRef, dst: Path) -> None:
    """No cache: lean on udp.resample_to_16k_mono (re-reads source each call)."""
    udp.resample_to_16k_mono(
        ref.path, dst,
        start_t=round_time(ref.start_t),
        end_t=round_time(ref.end_t),
    )


def _maybe_tqdm(iterable, *, total: int, desc: str, enable: bool):
    """Wrap an iterable in a tqdm bar if available + enabled; else pass through."""
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, total=total, desc=desc, unit=" rec", leave=False)
    except Exception:  # noqa: BLE001 — progress is cosmetic, never fatal
        return iterable


def _process_one(
    instance_id: str,
    audio_path: str,
    ref: "SourceRef | None",
    *,
    cache: "SourceCache | None" = None,
    force: bool,
    short_cut_warn_ms: int,
) -> tuple[str | None, str, str | None]:
    """
    Cut a single record's audio. Returns (normalised_path | None, outcome, note).

    outcome ∈ {"kept", "skipped_existing", "missing_source", "cut_failed"}.
    note is a short warning string (or None) for the caller to surface.

    Takes only the id + destination path + resolved SourceRef — never the full
    record — so it stays tiny to ship across a process boundary. The orchestrator
    owns the record dict and writes the returned `audio_path` back onto it.
    """
    if ref is None:
        return None, "missing_source", None

    dst = udp.from_project_relative(audio_path)

    # Idempotency: a non-empty destination is left alone unless force=True.
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return udp.to_project_relative(dst), "skipped_existing", None

    try:
        if cache is not None:
            _cut_cached(ref, dst, cache)
        else:
            _cut_independent(ref, dst)
    except Exception as ex:  # noqa: BLE001 — we want to drop & report, not crash
        return None, "cut_failed", f"{instance_id}: {ex}"

    note = None
    if ref.is_slice and short_cut_warn_ms > 0 and ref.end_t is not None:
        dur_ms = (round_time(ref.end_t) - (round_time(ref.start_t) or 0.0)) * 1000.0
        if dur_ms < short_cut_warn_ms:
            note = f"short cut ({dur_ms:.0f} ms): {instance_id}"

    return udp.to_project_relative(dst), "kept", note


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def cut_dataset(
    records: list[dict],
    resolver: Resolver,
    *,
    force: bool = False,
    source_cache_size: int = 4,
    short_cut_warn_ms: int = 100,
    num_workers: int = 0,
    parallel_backend: str = "process",
    chunksize: int | None = None,
    progress: bool = True,
    max_warn_print: int = 25,
) -> tuple[list[dict], dict]:
    """
    Cut every record per `resolver`. Returns (kept_records, stats).

    num_workers == 0 → sequential + LRU cache (best for shared big sources, ROG).
    num_workers  > 0 → worker pool, no cache (best for many small sources):
        parallel_backend="process" (default) → true multi-core (CPU-bound cut).
        parallel_backend="thread"            → I/O-bound / fork-unfriendly cases.
    chunksize → tasks per dispatch in process mode (None → auto from len/workers).

    Records whose source can't be resolved, or whose cut fails, are dropped.
    `audio_path` on kept records is normalised to project-relative.
    """
    if parallel_backend not in ("process", "thread"):
        raise ValueError(
            f"parallel_backend must be 'process' or 'thread', got {parallel_backend!r}")

    stats = {"input": len(records), "missing_source": 0, "cut_failed": 0,
             "skipped_existing": 0, "kept": 0, "short_warned": 0}
    kept: list[dict] = []
    warns_shown = 0

    def _surface(outcome: str, note: str | None) -> None:
        nonlocal warns_shown
        stats[outcome] = stats.get(outcome, 0) + 1
        if outcome == "kept" and note:
            stats["short_warned"] += 1
        if note and warns_shown < max_warn_print:
            print(f"   {note}")
            warns_shown += 1

    # Resolve once, up front. Keeps the (possibly huge) resolver index in the
    # parent only — workers receive just the small SourceRef per record.
    pairs = [(rec, resolver(rec)) for rec in records]

    if num_workers and num_workers > 0:
        ids   = [rec.get("instance_id", "?") for rec, _ in pairs]
        paths = [rec["audio_path"] for rec, _ in pairs]
        refs  = [ref for _, ref in pairs]
        worker = partial(_process_one, cache=None,
                         force=force, short_cut_warn_ms=short_cut_warn_ms)
        if chunksize is None:
            chunksize = max(1, len(pairs) // (num_workers * 64) or 1)

        Pool = ProcessPoolExecutor if parallel_backend == "process" else ThreadPoolExecutor
        desc = f"cutting ({parallel_backend} ×{num_workers})"
        with Pool(max_workers=num_workers) as ex:
            # map preserves input order, so results zip straight back to pairs.
            results = ex.map(worker, ids, paths, refs, chunksize=chunksize)
            results = _maybe_tqdm(results, total=len(pairs), desc=desc, enable=progress)
            for (rec, _ref), (new_path, outcome, note) in zip(pairs, results):
                _surface(outcome, note)
                if outcome in ("kept", "skipped_existing"):
                    rec["audio_path"] = new_path
                    kept.append(rec)
    else:
        cache = SourceCache(max_size=source_cache_size)
        # Sort by source path so all cuts of one source are back-to-back.
        pairs.sort(key=lambda p: str(p[1].path) if p[1] else "")
        iterator = _maybe_tqdm(pairs, total=len(pairs),
                               desc="cutting (sequential)", enable=progress)
        for rec, ref in iterator:
            new_path, outcome, note = _process_one(
                rec.get("instance_id", "?"), rec["audio_path"], ref,
                cache=cache, force=force, short_cut_warn_ms=short_cut_warn_ms)
            _surface(outcome, note)
            if outcome in ("kept", "skipped_existing"):
                rec["audio_path"] = new_path
                kept.append(rec)

    return kept, stats


# --------------------------------------------------------------------------- #
# Resolver factories
# --------------------------------------------------------------------------- #

def make_stem_scan_resolver(
    source_audio_root: str | Path,
    *,
    source_file_key: str = "source_file",
    pattern: str = "*.wav",
) -> Resolver:
    """
    For session-WAV corpora (ROG). Scans `source_audio_root` once, indexing every
    matching file by stem. Resolves each record via `metadata[source_file_key]`
    (stripped to its stem) and slices using the record's own start_t/end_t —
    i.e. the timestamps are absolute within the session WAV.

    Duplicate stems under the tree are reported; first one wins.
    """
    root = udp.from_project_relative(source_audio_root)
    if not root.exists():
        raise FileNotFoundError(f"source_audio_root does not exist: {root}")

    index: dict[str, Path] = {}
    collisions: list[tuple[str, Path, Path]] = []
    for f in root.rglob(pattern):
        if f.stem in index:
            collisions.append((f.stem, index[f.stem], f))
            continue
        index[f.stem] = f

    print(f"source index: {len(index)} files under {udp.to_project_relative(root)}")
    if collisions:
        print(f"⚠️  {len(collisions)} duplicate stems; keeping first:")
        for stem, kept, dropped in collisions[:5]:
            print(f"   - {stem}: kept {kept.name}, ignored {dropped.name}")
        if len(collisions) > 5:
            print(f"   ... and {len(collisions) - 5} more")

    def resolver(rec: dict) -> SourceRef | None:
        src_file = rec.get("metadata", {}).get(source_file_key)
        if not src_file:
            return None
        path = index.get(Path(src_file).stem)
        if path is None:
            return None
        return SourceRef(path, rec.get("start_t"), rec.get("end_t"))

    return resolver


def make_record_path_resolver(
    *,
    path_key: str = "audio_path_raw",
    audio_root: str | Path | None = None,
    slice_start_key: str | None = None,
    slice_end_key: str | None = None,
) -> Resolver:
    """
    For pre-cut corpora (ParlaSpeech). The source path comes straight off the
    record at `path_key` (e.g. the raw `audio` field), joined under `audio_root`
    if given.

    Whole-file convert by default (FLAC → 16 kHz mono WAV). To slice instead
    (e.g. word/event cuts from a per-utterance FLAC), name the record fields
    holding the slice bounds via `slice_start_key` / `slice_end_key` — those
    times are interpreted relative to the source file.
    """
    root = udp.from_project_relative(audio_root) if audio_root else None

    def resolver(rec: dict) -> SourceRef | None:
        raw = rec.get(path_key)
        if not raw:
            return None
        path = (root / raw) if root is not None else udp.from_project_relative(raw)
        if not path.exists():
            return None
        start = rec.get(slice_start_key) if slice_start_key else None
        end = rec.get(slice_end_key) if slice_end_key else None
        return SourceRef(path, start, end)

    return resolver


def _dig(d: dict, path: tuple[str, ...]):
    """Walk a nested dict by a key path; None if any hop is missing."""
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def make_flac_index_resolver(
    audio_root: str | Path,
    *,
    record_key_path: tuple[str, ...] = ("metadata", "source_audio"),
    index_cache_path: str | Path | None = None,
    pattern: str = "*.flac",
) -> Resolver:
    """
    For ParlaSpeech-style corpora where the JSONL audio field's directory nesting
    is NOT reproducible and varies by corpus (HR/RS: partX/{hash}/, CZ:
    partX/audio/psp/YYYY/MM/DD/). One recursive scan of `audio_root` builds a
    {basename -> absolute Path} index — the filename `{hash}_{start}-{end}.flac`
    is unique across a corpus, so directory layout is irrelevant. Each record is
    resolved by the basename of `record[*record_key_path]`. Whole-file convert
    (no slicing).

    `index_cache_path` makes the scan persistent: if the cache file exists it is
    loaded instead of rescanning (huge win for HR's ~1.4M files); otherwise the
    scan runs once and writes it. Delete the cache file to force a rescan.
    """
    import json

    root = udp.from_project_relative(audio_root)
    if not root.exists():
        raise FileNotFoundError(f"audio_root does not exist: {root}")

    cache = udp.from_project_relative(index_cache_path) if index_cache_path else None
    index: dict[str, Path] = {}

    if cache is not None and cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        index = {name: udp.from_project_relative(rel) for name, rel in raw.items()}
        print(f"loaded audio index from cache: {len(index):,} files "
              f"({udp.to_project_relative(cache)})")
    else:
        collisions = 0
        for f in root.rglob(pattern):
            if f.name in index:
                collisions += 1
                continue
            index[f.name] = f
        msg = f"scanned {len(index):,} files under {udp.to_project_relative(root)}"
        if collisions:
            msg += f"  ({collisions} duplicate basenames ignored)"
        print(msg)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(
                json.dumps({n: udp.to_project_relative(p) for n, p in index.items()},
                           ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  cached index → {udp.to_project_relative(cache)}")

    def resolver(rec: dict) -> SourceRef | None:
        raw = _dig(rec, record_key_path)
        if not raw:
            return None
        path = index.get(Path(raw).name)
        return SourceRef(path, None, None) if path is not None else None

    return resolver


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _selftest() -> None:
    """Synthesize a sine, cut a slice + a whole-file, verify. `python utils_audio_splitter.py`"""
    import tempfile
    import numpy as np
    import soundfile as sf

    udp.banner("utils_audio_splitter self-test")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "tone.wav"
        sr_native = 44100
        t = np.linspace(0, 3.0, int(3.0 * sr_native), endpoint=False)
        sf.write(str(src), (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"),
                 sr_native, subtype="PCM_16")

        out_slice = td / "slice.wav"
        out_whole = td / "whole.wav"
        recs = [
            {"instance_id": "t_slice", "audio_path": str(out_slice),
             "start_t": 0.500, "end_t": 1.250, "metadata": {"source_file": "tone.wav"}},
            {"instance_id": "t_whole", "audio_path": str(out_whole),
             "metadata": {"source_file": "tone.wav"}},
        ]

        # Resolver: slice record carries times, whole-file record doesn't.
        def resolver(rec):
            return SourceRef(src, rec.get("start_t"), rec.get("end_t"))

        kept, stats = cut_dataset(recs, resolver, num_workers=0)
        assert stats["kept"] == 2, stats
        d1, s1 = sf.read(str(out_slice))
        d2, s2 = sf.read(str(out_whole))
        assert s1 == 16000 and s2 == 16000, (s1, s2)
        assert abs(len(d1) / s1 - 0.750) < 0.01, len(d1) / s1
        assert abs(len(d2) / s2 - 3.000) < 0.01, len(d2) / s2
        print(f"✅ sequential: sliced {len(d1)/s1:.3f}s + whole {len(d2)/s2:.3f}s at 16 kHz")

        # Idempotent re-run via the parallel (process) path.
        _, stats2 = cut_dataset(recs, resolver, num_workers=2, parallel_backend="process")
        assert stats2["skipped_existing"] == 2, stats2
        print("✅ process backend: idempotent re-run skipped existing")

        # Process backend actually cuts, into a fresh destination.
        pslice, pwhole = td / "p_slice.wav", td / "p_whole.wav"
        precs = [
            {"instance_id": "p_slice", "audio_path": str(pslice),
             "start_t": 0.500, "end_t": 1.250},
            {"instance_id": "p_whole", "audio_path": str(pwhole)},
        ]
        kept_p, stats_p = cut_dataset(precs, resolver, num_workers=2,
                                      parallel_backend="process", progress=False)
        assert stats_p["kept"] == 2, stats_p
        dp1, sp1 = sf.read(str(pslice))
        dp2, sp2 = sf.read(str(pwhole))
        assert abs(len(dp1) / sp1 - 0.750) < 0.01, len(dp1) / sp1
        assert abs(len(dp2) / sp2 - 3.000) < 0.01, len(dp2) / sp2
        # Order preserved → kept records line up with input order.
        assert [r["instance_id"] for r in kept_p] == ["p_slice", "p_whole"], kept_p
        print(f"✅ process backend: cut {len(kept_p)} fresh files, order preserved")

        # Thread backend still works (kept for fork-unfriendly platforms).
        trecs = [{"instance_id": "t_whole2", "audio_path": str(td / "t_whole2.wav")}]
        _, stats_t = cut_dataset(trecs, resolver, num_workers=2,
                                 parallel_backend="thread", progress=False)
        assert stats_t["kept"] == 1, stats_t
        print("✅ thread backend: still cuts")

    print("\nAll good.")


if __name__ == "__main__":
    _selftest()