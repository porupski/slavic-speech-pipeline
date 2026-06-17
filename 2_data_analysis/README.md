# Chapter 2 — Dataset analysis (semi-stub)

Point at any canonical JSONL → stats + a short markdown report. Works for instance and frame files. Nothing here touches a GPU or a model.

Treated as a semi-stub: the planned overhaul replaces this with a real pass over label distributions, audio-duration histograms, and speaker/split summaries — and drops the "sniff" framing. Until then, the current notebook is useful for a quick sanity check.

## Run

```bash
mamba activate ssp
jupyter lab 2_data_analysis/20_sniff_dataset.ipynb
```

Edit the top `Config` to point at your JSONL, run cells.

## Output

A markdown report next to the JSONL: label counts, split sizes, basic schema sanity.
