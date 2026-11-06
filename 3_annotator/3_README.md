# Chapter 5 — TextGrid minter

Turn ParlaSpeech v3 utterances into TextGrids an annotator can edit, then read the annotated TGs back into stats and canonical JSONL. No GPU, no model — pure data shaping. Uses **praatio** for TextGrid I/O (UTF-8, auto-detects source encoding).

## Run

```bash
mamba activate ssp
cd 3_annotator
jupyter lab 31_tg_minter.ipynb         # mint annotation-ready TextGrids
jupyter lab 31b_mint_fp_annotation_set.ipynb  # balanced, numbered FP annotation pack
jupyter lab 32_annotation_stats.ipynb  # read annotated TGs back for stats
jupyter lab 33_tg_to_jsonl.ipynb       # annotated FP TGs → canonical event-instance JSONL
jupyter lab 34_stress_tg_to_jsonl.ipynb # Slovenian primary-stress TGs → word_frame JSONL
```

Each notebook starts with a `USER SETTINGS` cell — edit paths, tier selection, and the annotation tier name, then run the rest.

All notebooks have paired `.py` files (jupytext percent — source of truth). Regenerate the `.ipynb` with `jupytext --to ipynb <file>.py`.

## Files

| File | Purpose |
|---|---|
| `31_tg_minter.py` / `.ipynb` | Merge split per-layer TextGrids (`<base>.align.TextGrid`, `<base>.pause.TextGrid`, `<base>.stress.TextGrid`, …) into one long-format `<base>.merged.TextGrid`. Pick tiers in and in order; append a blank `fp-annotation` tier seeded with `?` where a source tier had text; append an `instructions` tier carrying the label legend for the annotator. Same code runs on a small workshop folder or the full corpus. |
| `31b_mint_fp_annotation_set.py` / `.ipynb` | Build a balanced, numbered annotation pack for a human annotator. Streams the master JSONL, keeps utterances with ≥ `REQUIRE_MIN_FP` filled-pause events, balances to a per-gender quota (default 1000 M / 1000 F) spread across as many distinct speakers as possible, numbers the selected clips `0000…N-1` and splits them into `batch_00 … batch_KK` folders. Writes a top-level `_README.txt`, `_manifest.tsv`, and `.jsonl` ground-truth key. |
| `32_annotation_stats.py` / `.ipynb` | Point at a folder of annotated TGs. Per-tier overview (interval counts, intervals with text), label distribution on the annotation tier (whatever labels were used — Vocal, Nasal, V-N, Other, FalsePositive, `?`, …). Optional `PENDING_LABELS` and `INVALID_LABELS` split marks into valid / pending / false-positive without hiding any raw label. |
| `33_tg_to_jsonl.py` / `.ipynb` | Annotated FP TGs → canonical event-instance JSONL (rung 3 of the BLUEPRINT ladder: per-FP-event classification). Default merge mode joins each TG against the chapter-1 canonical JSONL by `audio_path` basename, recovering `file_id`, `speaker`, `speaker_info`, split, and session bounds; events inherit those and carry their own offsets. As-is mode emits minimal records from the TG + sibling audio alone. One 16 kHz mono WAV is cut per annotated interval. Pending `?` intervals come through with `labels.fp_type = null`. |
| `34_stress_tg_to_jsonl.py` / `.ipynb` | Nejc's annotated Slovenian TGs (`data/nejc_slo_stress/final_textgrids/*.TextGrid`) → `data/processed_jsonl/si_primary_stress_word_frame.jsonl`. One row per multisyllabic word carrying a primary-stress mark; 50 Hz binary frame sequence over the word span. Multi-stress words are kept and tagged `metadata.multistress=True`. Each row also carries a `graph_align` list — the `graphAlign` tier sliced to the word's span (utterance-relative times rebased to word-local seconds) — for the vowel-nucleus accuracy metric in chapter 5 / 6. |
| `3_README.md` | This file. |
