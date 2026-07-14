# Chapter 2 — Explore dataset

Point at a canonical instance JSONL → a data-science-style sanity report:
split balance, speaker coverage, audio-duration distribution, per-label
distributions (categorical bars, numeric histograms), missing-field sweep.
Tables inline, plots saved as PNGs, everything stitched into one markdown
file under `data/reports/` at the end. No GPU, no model.

## Run

```bash
mamba activate ssp
jupyter lab 2_data_analysis/20_explore_dataset.ipynb
```

Edit the top `Config` (`jsonl_path` is the only field you usually change), run
cells top-to-bottom.

## Output

`data/reports/<jsonl_stem>_report.md` plus the PNGs it references
(splits / top speakers / speaker histogram / duration histogram + boxplot /
one figure per scalar label). In test mode the same artifacts mirror under
`data/test_reports/`.

## Scope

Instance-shape JSONLs only. Frame-shape files (per-record sequence labels)
need their own treatment — the notebook detects them and stops with a clear
message. Frame analysis lives in chapter 4 territory.

## Notebook source

`20_explore_dataset.py` is the jupytext percent-format source of truth.
Regenerate the `.ipynb` after edits with `jupytext --to ipynb 20_explore_dataset.py`.
The pre-overhaul notebook is preserved as `20_sniff_dataset.BACKUP.ipynb` for
reference.
