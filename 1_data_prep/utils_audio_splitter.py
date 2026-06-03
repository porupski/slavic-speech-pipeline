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
- `num_workers > 0`: thread pool, no cache. Best when every record has its own
  small source (ParlaSpeech's 290k FLACs). Threads win here because soundfile
  decode and the resample release the GIL.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


def _process_one(
    rec: dict, ref: SourceRef | None, cache: SourceCache | None,
    *, force: bool, short_cut_warn_ms: int,
) -> tuple[dict | None, str, str | None]:
    """
    Cut a single record. Returns (kept_record_or_None, outcome, note).

    outcome ∈ {"kept", "skipped_existing", "missing_source", "cut_failed"}.
    note is a short warning string (or None) for the caller to surface.
    """
    if ref is None:
        return None, "missing_source", None

    dst = udp.from_project_relative(rec["audio_path"])

    # Idempotency: a non-empty destination is left alone unless force=True.
    if dst.exists() and dst.stat().st_size > 0 and not force:
        rec["audio_path"] = udp.to_project_relative(dst)
        return rec, "skipped_existing", None

    try:
        if cache is not None:
            _cut_cached(ref, dst, cache)
        else:
            _cut_independent(ref, dst)
    except Exception as ex:  # noqa: BLE001 — we want to drop & report, not crash
        return None, "cut_failed", f"{rec.get('instance_id', '?')}: {ex}"

    note = None
    if ref.is_slice and short_cut_warn_ms > 0 and ref.end_t is not None:
        dur_ms = (round_time(ref.end_t) - (round_time(ref.start_t) or 0.0)) * 1000.0
        if dur_ms < short_cut_warn_ms:
            note = f"short cut ({dur_ms:.0f} ms): {rec.get('instance_id', '?')}"

    rec["audio_path"] = udp.to_project_relative(dst)
    return rec, "kept", note


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
    max_warn_print: int = 25,
) -> tuple[list[dict], dict]:
    """
    Cut every record per `resolver`. Returns (kept_records, stats).

    num_workers == 0 → sequential + LRU cache (best for shared big sources).
    num_workers  > 0 → thread pool, no cache (best for many small sources).

    Records whose source can't be resolved, or whose cut fails, are dropped.
    `audio_path` on kept records is normalised to project-relative.
    """
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

    # Resolve once, up front (also needed to sort by source in sequential mode).
    pairs = [(rec, resolver(rec)) for rec in records]

    if num_workers and num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(_process_one, rec, ref, None,
                          force=force, short_cut_warn_ms=short_cut_warn_ms)
                for rec, ref in pairs
            ]
            for fut in as_completed(futures):
                rec_out, outcome, note = fut.result()
                _surface(outcome, note)
                if rec_out is not None and outcome in ("kept", "skipped_existing"):
                    kept.append(rec_out)
    else:
        cache = SourceCache(max_size=source_cache_size)
        # Sort by source path so all cuts of one source are back-to-back.
        pairs.sort(key=lambda p: str(p[1].path) if p[1] else "")
        for rec, ref in pairs:
            rec_out, outcome, note = _process_one(
                rec, ref, cache, force=force, short_cut_warn_ms=short_cut_warn_ms)
            _surface(outcome, note)
            if rec_out is not None and outcome in ("kept", "skipped_existing"):
                kept.append(rec_out)

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
        print(f"✅ sliced {len(d1)/s1:.3f}s + whole {len(d2)/s2:.3f}s at 16 kHz")

        # Idempotency + parallel path
        _, stats2 = cut_dataset(recs, resolver, num_workers=2)
        assert stats2["skipped_existing"] == 2, stats2
        print("✅ idempotent re-run skipped existing (parallel path)")

    print("\nAll good.")


if __name__ == "__main__":
    _selftest()