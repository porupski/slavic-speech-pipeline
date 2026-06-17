# slavic-speech-pipeline

A modular pipeline for fine-tuning Wav2Vec2 on Slavic speech corpora. Instance- and frame-level classification and regression share the same engine — one config flag switches task type. The working corpus is **ParlaSpeech** (HR / RS / CZ / PL); ROG-Art and ROG-Dialog are still supported. The north star is a primary-stress frame model: feed one word, the model marks which frames carry the primary stress.

Design and rationale live in `BLUEPRINT.md` — read it once end-to-end before touching code.

## Quick start

```bash
# Install the CPU env (data prep, light development):
bash 0_env_setup/setup_env_cpu.sh
mamba activate ssp

# Or the CUDA env on a GPU server (training):
bash 0_env_setup/setup_env_cuda.sh
mamba activate ssp-cuda
```

Then walk the chapters in order; each has its own `README.md` and one-knob notebooks.

## Chapters

- **`0_env_setup/`** — CPU and CUDA conda envs, pinned requirements.
- **`1_data_prep/`** — source corpora → canonical JSONL. ParlaSpeech (HR / RS / CZ / PL) and ROG (Art / Dialog), plus the ParlaSpeech-HR v1 and v3 benchmark preps.
- **`2_data_analysis/`** — describe a canonical JSONL. Semi-stub; full overhaul pending.
- **`3_instance_models/`** — train an instance classifier (gender, filled pauses, benchmark tasks) or regressor (age, sentiment). Phase-E lifted: shared `utils_instance_train.py` + `config.json` + py runners.
- **`4_frame_models/`** — train a frame classifier (filled-pause frames; primary-stress frames).
- **`5_tg_minter/`** — turn ParlaSpeech v3 utterances into TextGrids for annotation, read annotated TGs back.

The ladder (instance → frame, secure each rung) is in `BLUEPRINT.md` §1.

## What you need on disk

`data/` (gitignored) holds raw downloads, unpacked corpora, cut 16 kHz mono WAVs, and the canonical JSONLs. Chapter 1 lays that out. Disk costs are real — ParlaSpeech-HR audio alone is ~207 GB across six tarballs; pick languages deliberately.

## Datasets

- **ParlaSpeech 3.0** — Croatian, Serbian, Czech, Polish parliamentary speech. Filled-pause and (HR/RS) primary-stress layers. [Project page](https://clarinsi.github.io/parlaspeech/) · CLARIN handle `http://hdl.handle.net/11356/1833`.
- **ROG 1.1** — Slovenian spoken corpus, sentiment + dialogue acts. CLARIN handle `http://hdl.handle.net/11356/2062`.

Per-language ParlaSpeech audio handles and citations are in `1_data_prep/README.md`.

## License

MIT. See `LICENSE`.
