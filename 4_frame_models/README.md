# Chapter 4 — Frame models

Per-frame classification. Feed an utterance WAV, predict a label per model frame (~49 Hz — the Wav2Vec2-base CNN's real output rate, not the nominal 50). Two frame tasks share the engine:

- **Filled-pause frames** — binary FP / not-FP over a whole utterance (ParlaSpeech `utterance_frame` from `11c`).
- **Primary-stress frames** (HR/RS) — word as the instance. `41` loads the utterance WAV and the 50 Hz label sequence, then slices both in memory by the record's word bounds. No word WAVs on disk.

Engine parity with the chapter-3 twins: run-mode tiers, GPU guard, stage timer, attention-mask handling, GPU flush between phases, inference spot-check. The frame-specific pieces are the per-frame head, token-CE with `ignore_index=-100`, and per-record label alignment to the model's real CNN output length.

## Run

```bash
mamba activate ssp-cuda
cd 4_frame_models
jupyter lab 41_train_frame_classification.ipynb
```

Edit the top `Config`:

- `cfg.target` — `parlaspeech_fp_frames` or `parlaspeech_primary_stress_frames`.
- `cfg.langs` — language filter (`()` = all supported for the target).

Run cells top-to-bottom. First cell is the same GPU guard pattern as chapter 3 — type `y` to arm the reserved GPU.

## Output

`runs/<task>/<timestamp>/` carries per-frame predictions including `prob_pos` (softmax positive-class probability) for downstream QC thresholding, plus standard frame metrics.

## What's missing

- **The phase-E lift** — chapter 4 is still standalone. The chapter-3 pattern (`utils_frame_train.py` + `config.json` + py runners) is planned. Full runs should wait for that lift.
- **`42_train_frame_regression.ipynb`** — completeness twin for scalar-per-frame regression. No annotated continuous target exists yet; the twin will be code for future work.
