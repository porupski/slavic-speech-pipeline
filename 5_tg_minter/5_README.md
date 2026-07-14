# Chapter 5 — TextGrid minter

Turn ParlaSpeech v3 utterances into TextGrids an annotator can edit, then read the annotated TGs back into stats. No GPU, no model — pure data shaping. Uses **praatio** for TextGrid I/O (UTF-8, auto-detects source encoding).

## Run

```bash
mamba activate ssp
cd 5_tg_minter
jupyter lab 51_tg_minter.ipynb        # mint annotation-ready TextGrids
jupyter lab 52_annotation_stats.ipynb # read annotated TextGrids back
jupyter lab 53_tg_to_jsonl.ipynb      # emit annotated TGs as canonical event-instance JSONL
```

Each notebook starts with a `USER SETTINGS` cell — edit paths, tier selection, and the annotation tier name, then run the rest.

## Notebooks

- **`51_tg_minter.ipynb`** — merges the split per-layer TextGrids (`<base>.align.TextGrid`, `<base>.pause.TextGrid`, `<base>.stress.TextGrid`, …) into one long-format `<base>.merged.TextGrid`. You pick which tiers go in and in what order, can append a blank `fp-annotation` tier seeded with `?` wherever a source tier had text, and an `instructions` tier carrying the label legend for the annotator. Same code runs on the small workshop folder or the full corpus folder — just slower on the latter. The inventory cell shows what tiers are in stock before you commit to a recipe.
- **`52_annotation_stats.ipynb`** — point at a folder of annotated TextGrids. Per-tier overview (interval counts, intervals with text), auto-collected label distribution on your annotation tier (whatever labels were used — Vocal, Nasal, V-N, Other, FalsePositive, `?`, …). Optional `PENDING_LABELS` and `INVALID_LABELS` split marks into valid / pending / false-positive without hiding any raw label.
- **`53_tg_to_jsonl.ipynb`** — annotated TGs → canonical event-instance JSONL (rung 3 in the BLUEPRINT ladder: per-FP-event classification). Default **merge mode** joins each TG against the chapter-1 canonical JSONL by `audio_path` basename, recovering `file_id`, `speaker`, `speaker_info`, split, and the utterance's session bounds; events inherit those and carry their own offsets. **As-is mode** emits minimal records from the TG + sibling audio alone. One 16 kHz mono WAV is cut per annotated interval (`utils_dataprep.resample_to_16k_mono`); pending `?` intervals come through with `labels.fp_type = null`. The output JSONL feeds chapter 3 like any other instance file.

All three notebooks have paired `.py` files (jupytext percent — source of truth). Regenerate the `.ipynb` with `jupytext --to ipynb <file>.py`.
