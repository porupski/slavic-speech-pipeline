#!/usr/bin/env python3
"""11c_prep_subset_parlaspeech — parametric subset of a full ParlaSpeech JSONL.

Takes the full ``parlaspeech_v4_<lang>_utterance_instance.jsonl`` written by 11c
and emits a subset JSONL with user-chosen caps and balance controls. The output
sits next to the source under ``data/processed_jsonl/`` and is drop-in for the
chapter-4 TARGETS registry (same schema, same audio paths — no re-prep needed).

Parameters (all optional, sensible defaults):

  --target_hours H         Cap the subset to ~H hours of audio (post-filter).
  --target_n N             Cap by utterance count instead of hours.
  --split_ratio a,b,c      Train/dev/test proportions (default 0.8,0.1,0.1).
                           Respects the source JSONL's existing ``split`` when
                           present AND the resulting per-split hour budgets are
                           feasible; otherwise re-assigns via speaker buckets.
  --speaker_disjoint       Default on. Each speaker belongs to one split only.
                           --no_speaker_disjoint disables (random per-row split).
  --duration_range lo,hi   Keep only utterances with hi ≥ audio_length ≥ lo
                           seconds (default 0.0,30.0 — matches v4 corpus prep).
  --balance_sentiment_6    Equalize ParlaSent_6 class counts in TRAIN by
                           downsampling the majority classes. Dev/test stay at
                           the natural distribution. Default off.
  --balance_gender         Equalize M/F counts in TRAIN by downsampling the
                           majority gender. Default off (PS is ~70/30).
  --require_sentiment      Drop rows missing ``sentiment_logit`` (default on —
                           we use PS for sentiment transfer, so unlabeled rows
                           are useless).
  --seed INT               RNG seed (default 1234).
  --merge_with PATH        Also emit a merged JSONL (shuffled with --seed) that
                           concatenates ROG and the PS subset. Both corpora
                           keep their native split keys.
  --merge_tag NAME         Suffix for the merged file (default: ``rog_plus``).
  --name NAME              Subset name tag — default is auto from caps
                           (e.g. ``200h_speakerDisjoint``). Shows up in the
                           output filename.
  --out_dir PATH           Override output directory (default: source dir).
  --input PATH             Source JSONL (default:
                           data/processed_jsonl/parlaspeech_v4_si_utterance_instance.jsonl).

Usage:
  python 11c_prep_subset_parlaspeech.py --target_hours 200 --name 200h
  python 11c_prep_subset_parlaspeech.py --target_hours 500 --name 500h \\
       --merge_with data/processed_jsonl/rog_dialog_v2_instance.jsonl \\
       --merge_tag PR500h

Outputs:
  data/processed_jsonl/parlaspeech_v4_<lang>_subset_<name>.jsonl
  data/processed_jsonl/parlaspeech_v4_<lang>_subset_<name>_report.md
  data/processed_jsonl/<merge_tag>_merged_<name>.jsonl   (when --merge_with set)
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import utils_dataprep as udp   # noqa: E402

PROJECT_ROOT = udp.PROJECT_ROOT
DEFAULT_INPUT = PROJECT_ROOT / "data/processed_jsonl/parlaspeech_v4_si_utterance_instance.jsonl"

# Same mapping the trainer uses for rog_dia_sentiment_regr: 6-cls → [0,5] float.
ROG_LABEL_SCALE = {
    "predominantlyNegative": 0.0,
    "mixedNegative":         1.0,
    "neutralNegative":       2.0,
    "neutralPositive":       3.0,
    "mixedPositive":         4.0,
    "predominantlyPositive": 5.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="source PS JSONL (full-corpus output of 11c)")
    p.add_argument("--target_hours", type=float, default=None,
                   help="cap by total subset hours")
    p.add_argument("--target_n", type=int, default=None,
                   help="cap by total utterance count (ignored if --target_hours set)")
    p.add_argument("--split_ratio", default="0.8,0.1,0.1",
                   help="train,dev,test proportions (comma-separated, sum ~= 1)")
    p.add_argument("--speaker_disjoint", action="store_true", default=True,
                   help="each speaker goes to exactly one split (default on)")
    p.add_argument("--no_speaker_disjoint", dest="speaker_disjoint",
                   action="store_false", help="disable speaker disjointness")
    p.add_argument("--duration_range", default="0.0,30.0",
                   help="lo,hi audio-length range in seconds")
    p.add_argument("--balance_sentiment_6", action="store_true",
                   help="equalize ParlaSent_6 classes in TRAIN")
    p.add_argument("--balance_gender", action="store_true",
                   help="equalize M/F counts in TRAIN")
    p.add_argument("--require_sentiment", action="store_true", default=True,
                   help="drop rows missing sentiment_logit (default on)")
    p.add_argument("--no_require_sentiment", dest="require_sentiment",
                   action="store_false")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--merge_with", default=None,
                   help="path to another JSONL (usually ROG) to concatenate + shuffle")
    p.add_argument("--merge_tag", default="rog_plus",
                   help="filename prefix for the merged output")
    p.add_argument("--name", default=None,
                   help="subset name tag (default: auto from caps)")
    p.add_argument("--out_dir", default=None,
                   help="override output directory (default: source dir)")
    return p.parse_args()


def _parse_range(s: str) -> tuple[float, float]:
    lo, hi = (float(x) for x in s.split(","))
    if lo < 0 or hi <= lo:
        raise ValueError(f"invalid range {s!r}: expect '0,N' with hi>lo>=0")
    return lo, hi


def _parse_ratio(s: str) -> tuple[float, float, float]:
    a, b, c = (float(x) for x in s.split(","))
    tot = a + b + c
    if tot <= 0:
        raise ValueError(f"invalid split_ratio {s!r}")
    return a / tot, b / tot, c / tot


def _auto_name(args) -> str:
    bits = []
    if args.target_hours is not None:
        bits.append(f"{int(round(args.target_hours))}h")
    elif args.target_n is not None:
        bits.append(f"{args.target_n}n")
    else:
        bits.append("full")
    if args.speaker_disjoint:
        bits.append("spkDisj")
    if args.balance_sentiment_6:
        bits.append("balS6")
    if args.balance_gender:
        bits.append("balG")
    return "_".join(bits)


def load_source(path: Path, dur_range: tuple[float, float],
                require_sentiment: bool) -> list[dict]:
    """Stream the source JSONL, apply duration + null filters. Returns the kept rows."""
    lo, hi = dur_range
    out, n_total, n_short, n_long, n_nosent = [], 0, 0, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n_total += 1
            r = json.loads(line)
            dur = r.get("metadata", {}).get("audio_length", 0.0) or 0.0
            if dur < lo:
                n_short += 1; continue
            if dur > hi:
                n_long += 1; continue
            if require_sentiment and r.get("labels", {}).get("sentiment_logit") is None:
                n_nosent += 1; continue
            out.append(r)
    print(f"source: {n_total} rows, kept {len(out)}")
    print(f"  dropped: {n_short} shorter than {lo}s, {n_long} longer than {hi}s, "
          f"{n_nosent} missing sentiment_logit")
    return out


def assign_splits_speaker_disjoint(rows: list[dict], ratio: tuple[float, float, float],
                                    rng: random.Random) -> None:
    """Group rows by speaker, shuffle groups, then assign speakers round-robin to
    train/dev/test until each split's cumulative row count meets its quota.
    Mutates rows in place by setting the top-level ``split`` key."""
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_spk[r.get("speaker", "unknown")].append(r)
    speakers = list(by_spk)
    rng.shuffle(speakers)

    quotas = {
        "train": int(round(ratio[0] * len(rows))),
        "dev":   int(round(ratio[1] * len(rows))),
        "test":  int(round(ratio[2] * len(rows))),
    }
    quotas["test"] = len(rows) - quotas["train"] - quotas["dev"]  # absorb rounding
    counts = {"train": 0, "dev": 0, "test": 0}
    for spk in speakers:
        under = sorted(counts, key=lambda s: counts[s] / max(1, quotas[s]))
        tgt = under[0]
        for r in by_spk[spk]:
            r["split"] = tgt
        counts[tgt] += len(by_spk[spk])
    print(f"speaker-disjoint split counts: {counts}  (quotas {quotas})")


def assign_splits_random(rows: list[dict], ratio: tuple[float, float, float],
                         rng: random.Random) -> None:
    """Flat per-row sampling. Speaker leakage is possible (use for ablations)."""
    r0, r1, _ = ratio
    for r in rows:
        u = rng.random()
        r["split"] = "train" if u < r0 else ("dev" if u < r0 + r1 else "test")


def _downsample_class(rows: list[dict], key: str,
                      extractor, rng: random.Random) -> list[dict]:
    """Equalize classes under ``key`` by downsampling to the smallest non-empty
    class. ``extractor`` maps a row to its class label (or None to skip)."""
    buckets: dict = defaultdict(list)
    for r in rows:
        v = extractor(r)
        if v is None:
            continue
        buckets[v].append(r)
    if not buckets:
        return rows
    min_c = min(len(v) for v in buckets.values())
    kept: list[dict] = []
    for cls, bucket in buckets.items():
        rng.shuffle(bucket)
        kept.extend(bucket[:min_c])
    print(f"  TRAIN balanced by {key}: {len(kept)} rows, "
          f"{min_c}/class across {len(buckets)} classes "
          f"(was {sum(len(v) for v in buckets.values())})")
    rng.shuffle(kept)
    return kept


def cap_subset(rows: list[dict], target_hours: float | None,
               target_n: int | None, rng: random.Random) -> list[dict]:
    """Downsample to the target. Preserves the existing split proportions so a
    hour cap bites train/dev/test uniformly."""
    if target_hours is None and target_n is None:
        return rows
    total_h = sum(r["metadata"]["audio_length"] for r in rows) / 3600.0
    if target_hours is not None:
        if total_h <= target_hours:
            print(f"cap: target {target_hours}h, source has {total_h:.1f}h — keeping all")
            return rows
        keep_frac = target_hours / total_h
    else:
        if len(rows) <= target_n:
            print(f"cap: target {target_n} rows, source has {len(rows)} — keeping all")
            return rows
        keep_frac = target_n / len(rows)

    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    out: list[dict] = []
    for sp, bucket in by_split.items():
        rng.shuffle(bucket)
        k = max(1, int(round(len(bucket) * keep_frac)))
        out.extend(bucket[:k])
    print(f"cap: kept {len(out)} rows across splits "
          f"(target {target_hours}h / {target_n} rows, keep_frac={keep_frac:.3f})")
    return out


def balance_train(rows: list[dict], balance_sentiment_6: bool, balance_gender: bool,
                  rng: random.Random) -> list[dict]:
    """Apply class-balance downsampling to the TRAIN split only.
    Dev/test keep their natural distributions (evaluation fidelity)."""
    if not (balance_sentiment_6 or balance_gender):
        return rows
    train = [r for r in rows if r["split"] == "train"]
    rest = [r for r in rows if r["split"] != "train"]
    if balance_sentiment_6:
        train = _downsample_class(train, "sentiment_6",
                                  lambda r: r.get("labels", {}).get("sentiment_6"), rng)
    if balance_gender:
        train = _downsample_class(train, "speaker_gender",
                                  lambda r: r.get("labels", {}).get("speaker_gender"), rng)
    return train + rest


def add_sentiment_f(rows: list[dict]) -> None:
    """Add a unified regression target ``sentiment_f`` on the [0, 5] axis.
    - ROG rows: look up labels.sentiment in ROG_LABEL_SCALE.
    - PS rows: use labels.sentiment_6 directly (already 0..5 int).
    Idempotent — overwrites any pre-existing sentiment_f."""
    for r in rows:
        labels = r.setdefault("labels", {})
        sent6 = labels.get("sentiment_6")
        rog_v = labels.get("sentiment")
        if isinstance(sent6, (int, float)):
            labels["sentiment_f"] = float(sent6)
        elif isinstance(rog_v, str) and rog_v in ROG_LABEL_SCALE:
            labels["sentiment_f"] = ROG_LABEL_SCALE[rog_v]
        else:
            labels["sentiment_f"] = None


def merge_with(subset_rows: list[dict], merge_path: Path,
               rng: random.Random) -> list[dict]:
    """Concatenate the subset with another JSONL (usually ROG) and shuffle."""
    other: list[dict] = []
    with open(merge_path, encoding="utf-8") as f:
        for line in f:
            other.append(json.loads(line))
    print(f"merge: {len(other)} rows from {merge_path.name}, "
          f"{len(subset_rows)} from subset → {len(other) + len(subset_rows)} total")
    pooled = subset_rows + other
    rng.shuffle(pooled)
    add_sentiment_f(pooled)
    return pooled


def dump_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows → {path.relative_to(PROJECT_ROOT)}")


def write_report(rows: list[dict], path: Path, args, meta: dict) -> None:
    by_split = Counter(r["split"] for r in rows)
    hours_by_split = defaultdict(float)
    sent_by_split: dict = defaultdict(Counter)
    gender_by_split: dict = defaultdict(Counter)
    for r in rows:
        sp = r["split"]
        hours_by_split[sp] += r["metadata"]["audio_length"] / 3600.0
        sent_by_split[sp][r["labels"].get("sentiment_6")] += 1
        gender_by_split[sp][r["labels"].get("speaker_gender")] += 1

    lines = [
        f"# Subset report — {meta['name']}",
        "",
        f"Source JSONL : `{Path(args.input).name}`",
        f"Seed         : {args.seed}",
        f"Params       : target_hours={args.target_hours}  target_n={args.target_n}  "
        f"split_ratio={args.split_ratio}  speaker_disjoint={args.speaker_disjoint}  "
        f"duration_range={args.duration_range}  balance_sentiment_6={args.balance_sentiment_6}  "
        f"balance_gender={args.balance_gender}",
        "",
        "## Split counts",
        "",
        "| split | rows | hours |",
        "|---|---:|---:|",
    ]
    for sp in ("train", "dev", "test"):
        lines.append(f"| {sp} | {by_split[sp]} | {hours_by_split[sp]:.2f} |")

    lines += ["", "## sentiment_6 by split", "",
              "| split | " + " | ".join(str(k) for k in sorted(
                  {k for d in sent_by_split.values() for k in d})) + " |"]
    cols = sorted({k for d in sent_by_split.values() for k in d}, key=lambda x: (x is None, x))
    header_sep = "|---|" + "---:|" * len(cols)
    lines.append(header_sep)
    for sp in ("train", "dev", "test"):
        row = [str(sent_by_split[sp].get(c, 0)) for c in cols]
        lines.append(f"| {sp} | " + " | ".join(row) + " |")

    lines += ["", "## speaker_gender by split", "",
              "| split | M | F | None |", "|---|---:|---:|---:|"]
    for sp in ("train", "dev", "test"):
        c = gender_by_split[sp]
        lines.append(f"| {sp} | {c.get('M', 0)} | {c.get('F', 0)} | {c.get(None, 0)} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report → {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    dur_range = _parse_range(args.duration_range)
    ratio = _parse_ratio(args.split_ratio)

    src = Path(args.input)
    if not src.is_absolute():
        src = (PROJECT_ROOT / src).resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    print(f"source: {src.relative_to(PROJECT_ROOT)}")

    name = args.name or _auto_name(args)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    stem = src.stem.replace("_utterance_instance", "")
    out_jsonl = out_dir / f"{stem}_subset_{name}.jsonl"
    out_report = out_dir / f"{stem}_subset_{name}_report.md"

    rows = load_source(src, dur_range, args.require_sentiment)

    if args.speaker_disjoint:
        assign_splits_speaker_disjoint(rows, ratio, rng)
    else:
        assign_splits_random(rows, ratio, rng)

    rows = cap_subset(rows, args.target_hours, args.target_n, rng)
    rows = balance_train(rows, args.balance_sentiment_6, args.balance_gender, rng)
    add_sentiment_f(rows)
    rng.shuffle(rows)

    dump_jsonl(rows, out_jsonl)
    write_report(rows, out_report, args, {"name": name})

    if args.merge_with:
        merge_src = Path(args.merge_with)
        if not merge_src.is_absolute():
            merge_src = (PROJECT_ROOT / merge_src).resolve()
        pooled = merge_with(rows, merge_src, rng)
        merged_out = out_dir / f"{args.merge_tag}_merged_{name}.jsonl"
        dump_jsonl(pooled, merged_out)
        print(f"merged file for mixtures: {merged_out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
