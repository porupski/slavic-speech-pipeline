# slavic-speech-pipeline

A modular, plug-and-play pipeline for fine-tuning speech models on instance- and frame-level classification and regression tasks. Built around Slavic speech corpora (Slovenian ROG and GOS, ParlaSpeech for HR / RS / CZ / PL), but every component is dataset-agnostic.

## What this is

- **Lego, not monolith.** Each stage (data prep, analysis, training, inference, post-hoc analysis) is a self-contained module connected through a single canonical JSONL format.
- **One job per script.** No script does two things.
- **`.py` + `.ipynb` pairs.** Notebooks for step-by-step development and learning. `.py` runners for production once a notebook is stable. Both import from the same `utils_<chapter>.py`, so logic is never duplicated.
- **Task type is a parameter.** Classification and regression share the same trainer; one config flag switches between them.

## Repository layout

```
slavic-speech-pipeline/
├── BLUEPRINT.md             ← the design doc, read this first
├── README.md                ← you are here
├── requirements.txt
│
├── 1_data_prep/             ← any source format → canonical JSONL
├── 2_data_analysis/         ← descriptive stats + sniff reports
├── 3_instance_models/       ← train instance-level classifier / regressor
├── 4_frame_models/          ← train frame-level classifier / regressor
└── 5_analysis/              ← post-hoc analysis of trained runs
```

Each chapter has its own `README.md` with run order and gotchas.

## Quick start

1. Clone the repo.
2. Create a venv and `pip install -r requirements.txt`.
3. Read `BLUEPRINT.md` once end-to-end.
4. Pick a chapter and read its `README.md`.
5. Open the matching notebook in that chapter and run cells.

## Status

Work in progress. See `BLUEPRINT.md` section 7 for the execution order and current focus.

## Datasets

- **ROG 1.1** — Slovenian spoken corpus with sentiment and dialogue-act annotations. [CLARIN handle](http://hdl.handle.net/11356/2062).
- **GOS 2.1** — Slovenian reference speech corpus, used here for word-aligned primary-stress annotations. [CLARIN handle](http://hdl.handle.net/11356/1863).
- **ParlaSpeech 3.0** — Croatian, Serbian, Czech, Polish parliamentary speech. Filled-pause and primary-stress layers. [Project page](https://clarinsi.github.io/parlaspeech/) · [CLARIN handle](http://hdl.handle.net/11356/1833).

## License

MIT. See `LICENSE`.
