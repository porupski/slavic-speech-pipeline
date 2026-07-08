# slavic-speech-pipeline

A modular pipeline for fine-tuning Wav2Vec2 on Slavic speech corpora. Instance- and frame-level classification and regression share the same engine — one config flag switches task type. The working corpus is **ParlaSpeech** (HR / RS / CZ / PL), with two pre-built **ParlaSpeech-HR benchmark** bundles (v1 and v3) for a smaller entry point. ROG-Art and ROG-Dialog are still supported. The north star is a primary-stress frame model: feed one word, the model marks which frames carry the primary stress.

## Quick start

```bash
# Install the CPU env (data prep, light development):
bash 0_env_setup/setup_env_cpu.sh
mamba activate ssp

# Or the CUDA env on a GPU server (training):
bash 0_env_setup/setup_env_cuda.sh
mamba activate ssp-cuda
```

Then walk the chapters in order. Each has its own `README.md` and one-knob notebooks. If you just want to see the whole thing move end-to-end quickly, download the default target (`ParlaSpeech-HR benchmark v3`, ~9 GB) in chapter 1 and follow the trail from there.

## Chapters

- **`0_env_setup/`** — CPU and CUDA mamba envs, pinned requirements.
- **`1_data_prep/`** — source corpora → canonical JSONL. ParlaSpeech (HR / RS / CZ / PL) and ROG (Art / Dialog), plus the ParlaSpeech-HR v1 and v3 benchmark preps.
- **`2_data_analysis/`** — describe a canonical JSONL. Semi-stub; full overhaul pending.
- **`3_instance_models/`** — train an instance classifier (gender, filled pauses, benchmark tasks) or regressor (age, sentiment). Shared engine in `utils_instance_train.py` + one `config.json`.
- **`4_frame_models/`** — train a frame classifier (filled-pause frames; primary-stress frames).
- **`5_tg_minter/`** — turn ParlaSpeech v3 utterances into TextGrids for annotation, read annotated TGs back.

## How this repo is meant to be used

Every chapter ships three parallel entry points, so the same code fits three audiences:

- **`.ipynb`** — the beginner-friendly path. Rich markdown between cells, small config at the top, run cells one at a time. Use these first: they explain *why* each step exists, not just *what* it does.
- **`.py`** — the same notebook expressed as a plain script in Jupytext's *percent* format. Same cell boundaries, no JSON noise, easy to diff and edit in any editor. If you're comfortable in Python but don't need Jupyter overhead, pick these.
- **`run_*.py`** (chapter 3, more to come) — thin CLI wrappers around the shared engine (`utils_*.py`) driven by `config.json`. Made for `tmux`, long GPU runs, and batch experiments. No prompts, no notebook state.

The `.py` and `.ipynb` files are paired by Jupytext — edit either one and regenerate the other with `jupytext --to ipynb file.py` (or `--to py:percent file.ipynb`). The runners and the shared engine are the source of truth for chapter 3+; the notebooks import from them cell by cell.

Pick whichever fits how you work. The pipeline behaves the same either way.

## What you need on disk

`data/` (gitignored) holds raw downloads, unpacked corpora, cut 16 kHz mono WAVs, and the canonical JSONLs. Chapter 1 lays that out. Disk costs are real — ParlaSpeech-HR audio alone is ~207 GB across six tarballs — so the download notebook defaults to the **ParlaSpeech-HR benchmark v3** bundle (~9 GB, ready to train). Anything larger is opt-in.

## Datasets

- **ParlaSpeech 3.0** — Croatian, Serbian, Czech, Polish parliamentary speech. Filled-pause and (HR/RS) primary-stress layers. [Project page](https://clarinsi.github.io/parlaspeech/) · CLARIN handle `http://hdl.handle.net/11356/1833`.
- **ParlaSpeech-HR benchmark v1 / v3** — pre-built speaker-disjoint splits for gender / speaker-id / power-status / age (+ orientation on v3). Hosted on Hugging Face: `porupski/ParlaSpeech-HR-benchmark_v1` and `_v3`.
- **ROG 1.1** — Slovenian spoken corpus, sentiment + dialogue acts. CLARIN handle `http://hdl.handle.net/11356/2062`.

Per-language ParlaSpeech audio handles and citations are in `1_data_prep/README.md`.

## License

MIT. See `LICENSE`.
