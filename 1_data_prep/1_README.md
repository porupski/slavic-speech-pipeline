# Chapter 1 — Data prep

Take any source corpus and emit a **canonical JSONL** that every downstream chapter consumes. If chapter 1 is right, the rest of the pipeline is dataset-agnostic.

The canonical schema is stable across instance and frame flavors — only the `labels` shape differs. See `utils_dataprep.py` for the validator; every prep notebook writes rows that pass it.

## Run

```bash
mamba activate ssp
cd 1_data_prep
```

There are two entry points depending on what you want:

- **Fastest path to a working model** — go straight to `11d`/`11e` (the ParlaSpeech-HR benchmarks). They pull ~9 GB of pre-built audio + labels from Hugging Face via `load_dataset(...)`, no separate download step. One `python 11e_prep_parlaspeech_benchmark_v3.py` and you have five per-task JSONLs ready for chapter 3.
- **Full corpora** — start with `10_download_data`, then the matching prep notebook. This is the CLARIN track (ParlaSpeech-{HR,RS,PL,CZ}, ROG, GOS) — large downloads, more control.

Every prep writes:
- one or more files under `data/processed_jsonl/` (the canonical JSONL that every downstream chapter reads), and
- 16 kHz mono WAVs under `data/cut_audio/` (full corpora) or `data/benchmarking/<name>/audio/` (benchmarks).

Then optionally `2_data_analysis/20_explore_dataset.ipynb` to eyeball the output.

## Prep notebooks

- **`11a_prep_ROG-art.ipynb`** — ROG-Art (EXB), dual output (instance + frame).
- **`11b_prep_ROG-dia.ipynb`** — ROG-Dialog (EXB), instance.
- **`11c_prep_parlaspeech.ipynb`** — the workhorse for full ParlaSpeech per language. Reads the v3 JSONL, converts per-utterance FLAC → 16 kHz mono WAV via the multicore splitter, and emits up to three recipes per language: `utterance_instance` (scalar labels — gender, FP presence/count, sentiment, age), `utterance_frame` (50 Hz `filled_pause` sequence), and `word_frame` (one record per primary-stress-annotated word, HR/RS only). One file per instance-shape — multiple label keys live in the same file; the trainer picks one via `label_key`.
- **`11d_prep_parlaspeech_benchmark_v1.ipynb`** — pull the HR benchmark **v1** from HF (`porupski/ParlaSpeech-HR-benchmark_v1`), extract audio to disk, emit four per-task classification JSONLs (gender / speaker-id / power-status / age). Splits come from the benchmark's per-task columns — no `assign_splits`.
- **`11e_prep_parlaspeech_benchmark_v3.ipynb`** — same idea, benchmark **v3** (`porupski/ParlaSpeech-HR-benchmark_v3`). Three classification tasks + two regression tasks (age, orientation).

Each prep file with a paired `.py` (jupytext percent format) treats the `.py` as the source of truth — regenerate the `.ipynb` with `jupytext --to ipynb <file>.py`.

## Downloading full corpora — `10_download_data`

The download catalogue for CLARIN sources lives in **`10a_dataset_registry.json`** (kept out of the notebook so the notebook stays skimmable), and the fetch/unpack helpers live in **`utils_download.py`**. Two source types are supported by the registry:

- **CLARIN.SI** direct HTTP — all four ParlaSpeech language corpora + audio, ROG, GOS. Streams to a `.part` sibling and renames on success, so an aborted download never leaves a half-finished file. Resume via HTTP `Range`.
- **Hugging Face Hub** raw file snapshots — supported by the registry, currently unused because the HR benchmarks are consumed via `load_dataset` in the prep step instead of raw snapshots.

Shorthands `ParlaSpeech` / `ParlaSpeech-audio` expand to their per-language sets.

**To add another dataset** — either CLARIN or an HF raw-file snapshot — see the `_readme.how_to_extend` note inside `10a_dataset_registry.json` for the exact fields required.

## Shared modules

- **`utils_download.py`** — registry loader, CLARIN downloader, HF snapshot downloader, unpacker, tree pretty-print.
- **`utils_dataprep.py`** — JSONL I/O, schema validation, deterministic speaker-grouped splits, `make_instance_id`, `PROJECT_ROOT` resolver.
- **`utils_audio_splitter.py`** — the cutter/converter. Resolver-driven (stem-scan / record-path / FLAC basename-index + persistent cache), multicore by default (process pool — `librosa.resample` is CPU-bound and barely scales on threads). Used inline by every prep notebook that touches raw ParlaSpeech / ROG audio.

## Gotchas

- **ParlaSpeech audio nesting differs by language and is not derivable from the JSONL `audio` field** (HR/RS: `partX/{hash}/`; CZ: `partX/audio/psp/Y/M/D/`). `utils_audio_splitter` resolves by basename via one recursive scan, cached to `..._audio_index.json` next to your processed JSONLs. HR's ~1.4M files make the cache matter; RS's ~290k is fine either way.
- **Splits are speaker-grouped for ParlaSpeech** (gender leaks badly otherwise) and file-grouped for ROG.
- **ROG source WAVs are 44.1 kHz**; the splitter resamples to 16 kHz on the way out.
- **Benchmark audio may be `.wav` or `.flac` on disk** — the JSONL's `audio` field lists an extension, but the on-disk file can differ (v1 lists `.flac`, ships `.wav`). The prep scripts detect the on-disk extension once and use that.
- **Annotations and per-language audio live on separate CLARIN handles** — see the table below.

## Data sources and citations

### ParlaSpeech 3.0 — annotations

- Handle: `http://hdl.handle.net/11356/1833`
- Project page: <https://clarinsi.github.io/parlaspeech/>
- Citation: Ljubešić, Nikola; et al., 2025, *Spoken corpora of parliamentary debates ParlaSpeech 3.0*, CLARIN.SI, ISSN 2820-4042, http://hdl.handle.net/11356/1833.
- Enrichment layers used here: **filled pauses** (HR, RS, PL, CZ), **primary stress** (HR, RS only).

### ParlaSpeech-HR benchmarks — Hugging Face

- v1: <https://huggingface.co/datasets/porupski/ParlaSpeech-HR-benchmark_v1>
- v3: <https://huggingface.co/datasets/porupski/ParlaSpeech-HR-benchmark_v3>

### ParlaSpeech audio — per-language releases

| Lang | Handle | Size | Citation |
|---|---|---|---|
| HR | `http://hdl.handle.net/11356/1914` | ~207 GB | Ljubešić, Nikola; Koržinek, Danijel; Rupnik, Peter, 2024, *Parliamentary spoken corpus of Croatian ParlaSpeech-HR 2.0*, CLARIN.SI, http://hdl.handle.net/11356/1914. |
| RS | `http://hdl.handle.net/11356/1834` | ~63 GB  | Ljubešić, Nikola; Rupnik, Peter; Koržinek, Danijel, 2024, *Parliamentary spoken corpus of Serbian ParlaSpeech-RS 1.0*, CLARIN.SI, http://hdl.handle.net/11356/1834. |
| PL | `http://hdl.handle.net/11356/1686` | ~59 GB  | Koržinek, Danijel; Ljubešić, Nikola, 2024, *Parliamentary spoken corpus of Polish ParlaSpeech-PL 1.0*, CLARIN.SI, http://hdl.handle.net/11356/1686. |
| CZ | `http://hdl.handle.net/11356/1785` | ~153 GB | Kopp, Matyáš; Ljubešić, Nikola, 2024, *Parliamentary spoken corpus of Czech ParlaSpeech-CZ 1.0*, CLARIN.SI, http://hdl.handle.net/11356/1785. |

### ROG 1.1 — Slovenian spoken corpus

- Handle: `http://hdl.handle.net/11356/2062`
