# Chapter 1 — Data prep

Take any source corpus and emit a **canonical JSONL** that every downstream chapter consumes. If chapter 1 is right, the rest of the pipeline is dataset-agnostic.

The canonical schema (instance and frame shapes, required keys, audio conventions, split rules) is defined once in `BLUEPRINT.md` §2 — it isn't re-stated here.

## Run

```bash
mamba activate ssp                  # see 0_env_setup
cd 1_data_prep
```

Then, in order:

1. `10_download_data.ipynb` — fetch + unpack the corpora you need. `cfg.datasets` selects; the cell is idempotent and skips what's already on disk.
2. The matching prep notebook for each corpus (see below).
3. Optional: `2_data_analysis/20_explore_dataset.ipynb` to eyeball the output.

Each prep notebook writes one or more files under `data/processed_jsonl/` and cut 16 kHz mono WAVs under `data/cut_audio/`.

## Prep notebooks

- **`11a_prep_ROG-art.ipynb`** — ROG-Art (EXB), dual output (instance + frame).
- **`11b_prep_ROG-dia.ipynb`** — ROG-Dialog (EXB), instance.
- **`11c_prep_parlaspeech.ipynb`** — the workhorse. Reads ParlaSpeech v3 JSONL per language, converts per-utterance FLAC to 16 kHz mono WAV via the multicore splitter, and emits up to three recipes per language: `utterance_instance` (scalar labels — gender, FP presence/count, sentiment, age), `utterance_frame` (50 Hz `filled_pause` sequence), and `word_frame` (one record per primary-stress-annotated word, HR/RS only). One file per instance-shape — multiple label keys live in the same file; the trainer picks one via `label_key`.
- **`11d_prep_parlaspeech_benchmark_v3.ipynb`** — parse the pre-built ParlaSpeech-HR benchmark v3 (fetched by `10`) into per-task JSONLs. Splits come from the benchmark's per-task `benchmark` key (no `assign_splits`); audio is already cut.
- **`11e_prep_parlaspeech_benchmark_v1.ipynb`** — same idea, ParlaSpeech-HR benchmark v1. Sibling to `11d`.

Each prep file with a paired `.py` (jupytext percent format) treats the `.py` as the source of truth — regenerate the `.ipynb` with `jupytext --to ipynb <file>.py`.

## Shared modules

- **`utils_dataprep.py`** — JSONL I/O, schema validation, deterministic speaker-grouped splits, `make_instance_id`, `PROJECT_ROOT` resolver.
- **`utils_audio_splitter.py`** — the cutter/converter. Resolver-driven (stem-scan / record-path / FLAC basename-index + persistent cache), multicore by default (process pool — `librosa.resample` is CPU-bound and barely scales on threads). Used inline by every prep notebook; the old standalone `12_audio_splitter` is retired.

## Gotchas

- **ParlaSpeech audio nesting differs by language and is not derivable from the JSONL `audio` field** (HR/RS: `partX/{hash}/`; CZ: `partX/audio/psp/Y/M/D/`). `utils_audio_splitter` resolves by basename via one recursive scan, cached to `..._audio_index.json` next to your processed JSONLs. HR's ~1.4M files make the cache matter; RS's ~290k is fine either way.
- **Splits are speaker-grouped for ParlaSpeech** (gender leaks badly otherwise) and file-grouped for ROG.
- **ROG source WAVs are 44.1 kHz**; the splitter resamples to 16 kHz on the way out.
- **Annotations and per-language audio live on separate CLARIN handles** — see the table below.

## Data sources and citations

### ParlaSpeech 3.0 — annotations

- Handle: `http://hdl.handle.net/11356/1833`
- Project page: <https://clarinsi.github.io/parlaspeech/>
- Citation: Ljubešić, Nikola; et al., 2025, *Spoken corpora of parliamentary debates ParlaSpeech 3.0*, CLARIN.SI, ISSN 2820-4042, http://hdl.handle.net/11356/1833.
- Enrichment layers used here: **filled pauses** (HR, RS, PL, CZ), **primary stress** (HR, RS only).

### ParlaSpeech audio — per-language releases

| Lang | Handle | Size | Citation |
|---|---|---|---|
| HR | `http://hdl.handle.net/11356/1914` | ~207 GB | Ljubešić, Nikola; Koržinek, Danijel; Rupnik, Peter, 2024, *Parliamentary spoken corpus of Croatian ParlaSpeech-HR 2.0*, CLARIN.SI, http://hdl.handle.net/11356/1914. |
| RS | `http://hdl.handle.net/11356/1834` | ~63 GB | Ljubešić, Nikola; Rupnik, Peter; Koržinek, Danijel, 2024, *Parliamentary spoken corpus of Serbian ParlaSpeech-RS 1.0*, CLARIN.SI, http://hdl.handle.net/11356/1834. |
| PL | `http://hdl.handle.net/11356/1686` | ~59 GB | Koržinek, Danijel; Ljubešić, Nikola, 2024, *Parliamentary spoken corpus of Polish ParlaSpeech-PL 1.0*, CLARIN.SI, http://hdl.handle.net/11356/1686. |
| CZ | `http://hdl.handle.net/11356/1785` | ~153 GB | Kopp, Matyáš; Ljubešić, Nikola, 2024, *Parliamentary spoken corpus of Czech ParlaSpeech-CZ 1.0*, CLARIN.SI, http://hdl.handle.net/11356/1785. |

### ROG 1.1 — Slovenian spoken corpus

- Handle: `http://hdl.handle.net/11356/2062`
