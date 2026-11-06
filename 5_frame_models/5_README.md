# Chapter 4 — Frame models

Per-frame classification. Feed an utterance WAV, predict a label per model frame (~49 Hz — the Wav2Vec2-base CNN's real output rate, not the nominal 50). Two frame tasks share the engine:

- **Filled-pause frames** — binary FP / not-FP over a whole utterance (ParlaSpeech `utterance_frame` from `11c`).
- **Primary-stress frames** (HR/RS) — word as the instance. `41` loads the utterance WAV and the 50 Hz label sequence, then slices both in memory by the record's word bounds. No word WAVs on disk.

A regression twin — `52_train_frame_regression` — ships as a **scaffold** (dated 2026-11-02). The notebook mirrors `41` structurally and the chapter-4 regression twin content-wise. The engine (`utils_frame_train.py`) is still classification-only, so `42` stops at `load_config` with a dated port-plan error. Made for structural completeness; tested when the first continuous frame-level target lands. The port plan lives at the bottom of `52_train_frame_regression.py` under *What to wire in the engine*.

Engine parity with the chapter-4 twins: run-mode tiers, GPU guard, stage timer, attention-mask handling, GPU flush between phases, inference spot-check. The frame-specific pieces are the per-frame head, token-CE with `ignore_index=-100`, and per-record label alignment to the model's real CNN output length.

## Run

```bash
mamba activate ssp-cuda
cd 5_frame_models
jupyter lab 51_train_frame_classification.ipynb
```

Edit the top `Config`:

- `cfg.target` — `parlaspeech_fp_frames` or `parlaspeech_primary_stress_frames`.
- `cfg.langs` — language filter (`()` = all supported for the target).

Run cells top-to-bottom. First cell is the same GPU guard pattern as chapter 4 — type `y` to arm the reserved GPU.

## Output

`runs/<task>/<timestamp>/` carries per-frame predictions including `prob_pos` (softmax positive-class probability) for downstream QC thresholding, plus standard frame metrics.

## What's missing

- **The shared-engine refactor** — chapter 5 is still standalone (the whole engine lives inline in `41`). The chapter-4 pattern (`utils_frame_train.py` + `5_config.json` + py runner) is planned. Full runs should wait for that refactor.
- **Frame regression engine wiring** — the `42` scaffold is in place, but `utils_frame_train.py` still raises when `task_type='regression'`. Chapter 3's `utils_instance_train.py` is the port template; the step-by-step plan is in `52_train_frame_regression.py` under *What to wire in the engine*.
