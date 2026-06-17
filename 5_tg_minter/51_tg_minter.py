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
# # 51 — TextGrid Minter
#
# Combine the split per-layer TextGrids (`<base>.align.TextGrid`,
# `<base>.pause.TextGrid`, `<base>.stress.TextGrid`, ...) into one long-format
# `<base>.merged.TextGrid`. Uses **praatio** (reads/writes UTF-8, auto-detecting
# the source encoding).
#
# You pick **which layers** go in and **in what order**, can append a blank
# **`fp-annotation`** tier that mimics another tier's boundaries (pre-marked with
# `?` wherever the source had text), and an **instructions** tier carrying the
# label legend so an annotator can glance down for the current task.
#
# Same code runs on the small workshop folder or the full corpus folder — just
# slower on the big one. The **inventory** cell shows what tiers are in stock
# before you commit to a recipe.

# %% [markdown]
# ## ▼ USER SETTINGS — edit this cell, then run the rest

# %%
from pathlib import Path

# --- WHERE ------------------------------------------------------------------
# Point these at your own folders. Absolute paths are safest.
IN_DIR  = Path("path/to/your/textgrid_layers")     # raw <base>.<layer>.TextGrid live here
OUT_DIR = Path("path/to/your/merged_textgrids")    # <base>.merged.TextGrid written here

# --- WHAT TO COMBINE  (top-to-bottom = tier order in the output) ------------
#   Each row names a layer, then which tiers to take from it. Accepted forms:
#     ("align", "all")                        -> every tier from that layer
#     ("align", ["WordAlign", "GraphAlign"])  -> those tiers, in that order
#     ("align", "WordAlign", "GraphAlign")    -> same, shorthand (no brackets)
#     ("pause", "FilledPauses")               -> a single tier
#   layer = the ".<layer>." token in the filename. Comment a row out to skip it.
#   ("all" is reserved; a tier literally named "all" isn't supported.)
MERGE_RECIPE = [
    ("align", "all"),
    ("pause", "all"),
    # ("stress", "all"),
]

# --- ADD A BLANK ANNOTATION TIER (mimics another tier's boundaries) ---------
#   Copies the source tier's interval structure. Wherever the source had ANY
#   text, the new interval is pre-filled with ANNOT_PRE_LABEL; the rest are blank.
ADD_MIMIC_TIER     = True
MIMIC_SOURCE_LAYER = "pause"          # which layer's TextGrid to read the source tier from
MIMIC_SOURCE_TIER  = None             # exact tier name, or None = that layer's FIRST tier
MIMIC_NEW_TIER     = "fp-annotation"  # name of the tier you'll hand-annotate
ANNOT_PRE_LABEL    = "?"              # placed where the source tier had text

# --- ADD AN INSTRUCTIONS TIER (label legend, spans the whole utterance) -----
#   A single interval from start to end of each TextGrid, holding a reminder of
#   the valid labels. Purely a glance-down aid for the annotator.
ADD_INSTRUCTION_TIER  = True
INSTRUCTION_TIER_NAME = "instructions"
INSTRUCTION_TEXT      = "Valid labels: Vocal, Nasal, V-N, Other, FalsePositive"

# --- RUN OPTIONS ------------------------------------------------------------
INVENTORY_SAMPLE = 30        # files to sample when reporting which tiers exist
PROCESS_LIMIT    = None      # cap how many utterances to mint (None = all)
# ----------------------------------------------------------------------------

# %% [markdown]
# ## Imports & helpers

# %%
import itertools
from collections import defaultdict, Counter

import pandas as pd
from praatio import textgrid as tgio
from praatio.utilities.constants import Interval, Point, INTERVAL_TIER


def read_tg(path):
    # includeEmptyIntervals=True is essential: blank intervals carry the structure
    # the merge and mimic depend on. praatio auto-handles UTF-8/UTF-16 on read.
    return tgio.openTextgrid(str(path), includeEmptyIntervals=True,
                             reportingMode="silence", duplicateNamesMode="rename")


def save_tg(tg, path):
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True,
            reportingMode="silence")


def parse_tg_name(path):
    """'<base>.<layer>.TextGrid' -> (base, layer). Robust to dots inside base."""
    name = Path(path).name
    if not name.endswith(".TextGrid"):
        return None
    stem = name[:-len(".TextGrid")]
    if "." not in stem:
        return None
    base, layer = stem.rsplit(".", 1)
    return base, layer


def make_mimic_tier(src, new_name, pre_label):
    """Copy boundary structure; pre_label where src had text, blank elsewhere."""
    if src.tierType == INTERVAL_TIER:
        entries = [Interval(e.start, e.end,
                            pre_label if (e.label or "").strip() else "")
                   for e in src.entries]
        return tgio.IntervalTier(new_name, entries, src.minTimestamp, src.maxTimestamp)
    else:  # point tier
        entries = [Point(e.time, pre_label if (e.label or "").strip() else "")
                   for e in src.entries]
        return tgio.PointTier(new_name, entries, src.minTimestamp, src.maxTimestamp)


def make_instruction_tier(name, text, t0, t1):
    return tgio.IntervalTier(name, [Interval(t0, t1, text)], t0, t1)


def group_all(folder):
    """Return {base: {layer: Path}}. Lists the whole folder (slow on the big one)."""
    groups = defaultdict(dict)
    for p in folder.glob("*.TextGrid"):
        parsed = parse_tg_name(p)
        if parsed:
            base, layer = parsed
            groups[base][layer] = p
    return groups


def normalize_recipe(recipe):
    """Accept any of these row shapes and return [(layer, "all" | [tiers]), ...]:
         ("align", "all")
         ("align", ["WordAlign", "GraphAlign"])   # list of tiers
         ("align", "WordAlign", "GraphAlign")     # shorthand, no brackets
         ("pause", "FilledPauses")                # single tier
         ("pause",)                               # bare layer -> all tiers
    "all" is reserved to mean every tier.
    """
    norm = []
    for row in recipe:
        if not row:
            continue
        layer, rest = row[0], row[1:]
        if len(rest) == 0:
            spec = "all"
        elif len(rest) == 1:
            r = rest[0]
            if r == "all":
                spec = "all"
            elif isinstance(r, (list, tuple)):
                spec = list(r)
            else:
                spec = [r]                      # single bare tier name
        else:
            spec = list(rest)                   # ("align", "A", "B", ...)
        norm.append((layer, spec))
    return norm


def required_layers():
    layers = [lay for lay, _ in normalize_recipe(MERGE_RECIPE)]
    if ADD_MIMIC_TIER and MIMIC_SOURCE_LAYER not in layers:
        layers.append(MIMIC_SOURCE_LAYER)
    return layers


def merge_one(layer_paths):
    """layer_paths: {layer: Path} for one base. Returns (merged_Textgrid, warnings)."""
    warnings = []
    need = required_layers()
    layer_tgs = {lay: read_tg(p) for lay, p in layer_paths.items() if lay in need}

    doms = [(tg.minTimestamp, tg.maxTimestamp) for tg in layer_tgs.values()]
    t0 = min(d[0] for d in doms)
    t1 = max(d[1] for d in doms)
    if len({(round(a, 3), round(b, 3)) for a, b in doms}) > 1:
        warnings.append(f"layer time-domains differ: {doms}")

    merged = tgio.Textgrid(minTimestamp=t0, maxTimestamp=t1)
    used = set()
    for layer, want in normalize_recipe(MERGE_RECIPE):
        if layer not in layer_tgs:
            warnings.append(f"layer '{layer}' missing; skipped")
            continue
        src = layer_tgs[layer]
        names = list(src.tierNames) if want == "all" else list(want)
        for nm in names:
            if nm not in src.tierNames:
                warnings.append(f"tier '{nm}' not in layer '{layer}'; skipped")
                continue
            out_name = nm if nm not in used else f"{layer}_{nm}"  # avoid collisions
            used.add(out_name)
            merged.addTier(src.getTier(nm).new(name=out_name))

    if ADD_MIMIC_TIER:
        src = layer_tgs.get(MIMIC_SOURCE_LAYER)
        if src is None:
            warnings.append(f"mimic source layer '{MIMIC_SOURCE_LAYER}' missing")
        else:
            tier_name = MIMIC_SOURCE_TIER or src.tierNames[0]
            merged.addTier(make_mimic_tier(src.getTier(tier_name),
                                           MIMIC_NEW_TIER, ANNOT_PRE_LABEL))

    if ADD_INSTRUCTION_TIER:
        merged.addTier(make_instruction_tier(INSTRUCTION_TIER_NAME,
                                             INSTRUCTION_TEXT, t0, t1))

    return merged, warnings


# %% [markdown]
# ## Step 1 — Inventory: what tiers are in stock?
#
# Samples up to `INVENTORY_SAMPLE` files per layer (no full folder listing) and
# reports the tier names, type, and a typical interval count. Use it to fill in
# `MERGE_RECIPE` / `MIMIC_SOURCE_TIER` above.

# %%
seen = defaultdict(lambda: defaultdict(lambda: {"type": set(), "counts": []}))
layer_file_counts = Counter()

for p in itertools.islice(IN_DIR.glob("*.TextGrid"), INVENTORY_SAMPLE * 4):
    parsed = parse_tg_name(p)
    if not parsed:
        continue
    base, layer = parsed
    if layer_file_counts[layer] >= INVENTORY_SAMPLE:
        continue
    layer_file_counts[layer] += 1
    tg = read_tg(p)
    for name in tg.tierNames:
        t = tg.getTier(name)
        info = seen[layer][name]
        info["type"].add(t.tierType)
        info["counts"].append(len(t.entries))

rows = []
for layer in sorted(seen):
    for tname, info in seen[layer].items():
        cnts = info["counts"]
        rows.append({
            "layer": layer,
            "tier": tname,
            "type": "/".join(sorted(info["type"])),
            "n_sampled": len(cnts),
            "median_intervals": int(pd.Series(cnts).median()) if cnts else 0,
        })
inventory = pd.DataFrame(rows)
print("Layers sampled:", dict(layer_file_counts))
print("Tip: set MERGE_RECIPE / MIMIC_SOURCE_TIER from the 'tier' names below.\n")
inventory

# %% [markdown]
# ## Step 2 — Preview the recipe on one utterance

# %%
need = required_layers()
groups = group_all(IN_DIR)
print(f"{len(groups):,} bases found in {IN_DIR}")

preview_base = next((b for b, lp in groups.items()
                     if all(l in lp for l in need)), None)
if preview_base is None:
    print(f"!! No base has all required layers {need}. Check MERGE_RECIPE / folder.")
else:
    merged, warns = merge_one(groups[preview_base])
    print(f"\nPreview base: {preview_base}")
    print("Output tiers (in order):")
    for i, name in enumerate(merged.tierNames, 1):
        t = merged.getTier(name)
        print(f"  {i}. {name}  [{t.tierType}, {len(t.entries)} intervals]")
    if ADD_MIMIC_TIER and MIMIC_NEW_TIER in merged.tierNames:
        ann = merged.getTier(MIMIC_NEW_TIER)
        pre = sum(1 for e in ann.entries if (e.label or "").strip())
        print(f"\n  '{MIMIC_NEW_TIER}': {pre} interval(s) pre-marked '{ANNOT_PRE_LABEL}'")
    if warns:
        print("  warnings:", warns)

# %% [markdown]
# ## Step 3 — Batch mint
#
# Mints every base (or the first `PROCESS_LIMIT`) that has the required layers,
# writing `<base>.merged.TextGrid` as long-format UTF-8.

# %%
OUT_DIR.mkdir(parents=True, exist_ok=True)

bases = sorted(groups)
if PROCESS_LIMIT is not None:
    bases = bases[:PROCESS_LIMIT]

minted, skipped = 0, 0
all_warnings = defaultdict(list)
for base in bases:
    lp = groups[base]
    if not all(l in lp for l in need):
        skipped += 1
        all_warnings["missing_required_layers"].append(base)
        continue
    merged, warns = merge_one(lp)
    for w in warns:
        all_warnings[w.split(":")[0]].append(base)
    save_tg(merged, OUT_DIR / f"{base}.merged.TextGrid")
    minted += 1

print(f"Minted  : {minted}  ->  {OUT_DIR}")
print(f"Skipped : {skipped} (missing required layers)")
if all_warnings:
    print("\nWarning summary (type -> #bases):")
    for k, v in all_warnings.items():
        print(f"  {k}: {len(v)}   e.g. {v[:2]}")

# %% [markdown]
# ## Step 4 — Verify outputs

# %%
for p in itertools.islice(OUT_DIR.glob("*.merged.TextGrid"), 3):
    tg = read_tg(p)
    print(f"{p.name}: {list(tg.tierNames)}")
    with open(p, encoding="utf-8") as f:   # confirm UTF-8 readable
        f.read()
print("\nVerified readable, UTF-8.")
