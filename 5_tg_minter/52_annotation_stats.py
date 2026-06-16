# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 52 — Annotation Stats
#
# Point this at a folder of annotated TextGrids and get a per-tier overview
# (how many intervals, how many carry text) and the **label distribution** on
# your annotation tier — **auto-collected**, so whatever labels you used
# (Vocal / Nasal / V-N / Other / FalsePositive / `?` / anything) just appear.
#
# Optional interpretation layers: `PENDING_LABELS` (still to classify, e.g. `?`)
# and `INVALID_LABELS` (machine false-positives) split marks into valid / pending
# / false-positive without hiding any raw label. Uses **praatio**.

# %% [markdown]
# ## ▼ USER SETTINGS

# %%
from pathlib import Path

# --- WHERE ------------------------------------------------------------------
IN_DIR = Path("/cache/ivanp/projects/slavic-speech-pipeline/data/"
              "workshop_examples/merged")     # folder of annotated *.merged.TextGrid

# --- WHICH TIER(S) TO ANALYZE -----------------------------------------------
#   "auto"              -> every tier found
#   ["fp-annotation"]   -> only these named tiers
ANALYZE_TIERS = ["fp-annotation"]

# --- LABEL INTERPRETATION (adaptive; nothing is hardcoded away) -------------
EMPTY_VALUES   = {""}        # texts counted as "no mark at all"
PENDING_LABELS = {"?"}       # marked but not yet classified
INVALID_LABELS = {"FalsePositive", "F"}   # machine false-positives (still listed)

# --- OPTIONAL: join speaker/gender from the extractor manifest --------------
#   Join key is the filename label (e.g. F_BabicAnte_2fp_001). None = skip.
MANIFEST_CSV = None
# MANIFEST_CSV = Path(".../workshop_examples/manifest.csv")

# --- OUTPUT -----------------------------------------------------------------
OUT_DIR = IN_DIR.parent / "annotation_stats"
# ----------------------------------------------------------------------------

# %% [markdown]
# ## Imports & helpers

# %%
import json
from collections import Counter, defaultdict

import pandas as pd
from praatio import textgrid as tgio


def read_tg(path):
    return tgio.openTextgrid(str(path), includeEmptyIntervals=True,
                             reportingMode="silence", duplicateNamesMode="rename")


def file_id(path):
    """Strip '.merged.TextGrid' or '.TextGrid' -> the label used in the manifest."""
    n = Path(path).name
    for suf in (".merged.TextGrid", ".TextGrid"):
        if n.endswith(suf):
            return n[:-len(suf)]
    return Path(path).stem


def is_empty(label):
    return (label or "").strip() in {e.strip() for e in EMPTY_VALUES}


files = sorted(IN_DIR.glob("*.TextGrid"))
print(f"{len(files):,} TextGrid file(s) in {IN_DIR}")

# %% [markdown]
# ## Step 1 — Tier overview (all tiers)
#
# Per tier across the folder: total intervals, how many carry text, distinct
# labels. The annotation tier stands out here.

# %%
tier_tot = defaultdict(int)
tier_nonempty = defaultdict(int)
tier_labels = defaultdict(set)
tier_files = defaultdict(int)

for p in files:
    tg = read_tg(p)
    for name in tg.tierNames:
        tier_files[name] += 1
        for e in tg.getTier(name).entries:
            tier_tot[name] += 1
            if not is_empty(e.label):
                tier_nonempty[name] += 1
                tier_labels[name].add(e.label.strip())

overview = pd.DataFrame([{
    "tier": name,
    "files": tier_files[name],
    "intervals_total": tier_tot[name],
    "intervals_annotated": tier_nonempty[name],
    "distinct_labels": len(tier_labels[name]),
} for name in tier_tot]).sort_values("intervals_annotated", ascending=False)
overview

# %% [markdown]
# ## Step 2 — Label distribution on the analyzed tier(s)
#
# Auto-collected counts for every label, plus the valid / pending / false-positive
# breakdown driven by the settings above.

# %%
targets = (list(tier_tot.keys()) if ANALYZE_TIERS == "auto" else ANALYZE_TIERS)

labels = Counter()
n_total = n_empty = 0
per_file = []

for p in files:
    tg = read_tg(p)
    fc = Counter()
    for tname in targets:
        if tname not in tg.tierNames:
            continue
        for e in tg.getTier(tname).entries:
            n_total += 1
            if is_empty(e.label):
                n_empty += 1
            else:
                lab = e.label.strip()
                labels[lab] += 1
                fc[lab] += 1
    per_file.append({"file": file_id(p), "annotated": sum(fc.values()), **dict(fc)})

annotated = sum(labels.values())
pending = sum(c for l, c in labels.items() if l in PENDING_LABELS)
invalid = sum(c for l, c in labels.items() if l in INVALID_LABELS)
valid = annotated - pending - invalid

dist = pd.DataFrame(
    [{"label": l, "count": c, "share": c / annotated if annotated else 0}
     for l, c in labels.most_common()]
)

print(f"Analyzed tier(s): {targets}")
print(f"  intervals total       : {n_total}")
print(f"  empty (no mark)       : {n_empty}")
print(f"  annotated (any text)   : {annotated}")
print(f"    valid                : {valid}")
print(f"    pending {sorted(PENDING_LABELS)} : {pending}")
print(f"    false-pos {sorted(INVALID_LABELS)} : {invalid}")
print("\nLabel distribution:")
dist

# %% [markdown]
# ## Step 3 — Per-file counts

# %%
per_file_df = pd.DataFrame(per_file).fillna(0)
for c in per_file_df.columns:
    if c != "file":
        per_file_df[c] = per_file_df[c].astype(int)
per_file_df.head(20)

# %% [markdown]
# ## Step 4 — (optional) slice by speaker / gender via the manifest

# %%
if MANIFEST_CSV:
    man = pd.read_csv(MANIFEST_CSV)
    merged = per_file_df.merge(man[["label", "gender", "speaker_id", "speaker_name"]],
                               left_on="file", right_on="label", how="left")
    label_cols = [c for c in per_file_df.columns if c not in ("file", "annotated")]
    by_gender = merged.groupby("gender")[["annotated"] + label_cols].sum()
    print("Annotated marks by gender:")
    print(by_gender.to_string())
    print("\nTop speakers by annotated marks:")
    print(merged.groupby(["gender", "speaker_name"])["annotated"]
                .sum().sort_values(ascending=False).head(10).to_string())
else:
    print("MANIFEST_CSV not set — skipping speaker/gender slice.")

# %% [markdown]
# ## Step 5 — Save

# %%
OUT_DIR.mkdir(parents=True, exist_ok=True)
overview.to_csv(OUT_DIR / "tier_overview.csv", index=False, encoding="utf-8")
dist.to_csv(OUT_DIR / "label_distribution.csv", index=False, encoding="utf-8")
per_file_df.to_csv(OUT_DIR / "per_file_counts.csv", index=False, encoding="utf-8")

summary = {
    "in_dir": str(IN_DIR),
    "files": len(files),
    "analyzed_tiers": targets,
    "intervals_total": int(n_total),
    "intervals_empty": int(n_empty),
    "annotated": int(annotated),
    "valid": int(valid),
    "pending": int(pending),
    "pending_labels": sorted(PENDING_LABELS),
    "false_positive": int(invalid),
    "invalid_labels": sorted(INVALID_LABELS),
    "label_counts": dict(labels),
}
with open(OUT_DIR / "annotation_stats.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"Saved to {OUT_DIR}:")
print("  tier_overview.csv, label_distribution.csv, per_file_counts.csv, annotation_stats.json")
