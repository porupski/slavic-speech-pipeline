# Chapter 3 — Instance models

Train a Wav2Vec2 instance classifier or regressor on whole-utterance audio. One target switch picks the task: classification (gender, filled-pause presence/count, ParlaSpeech-HR benchmark tasks) or regression (`speaker_age`, `sentiment_logit`).

The engine lives in `utils_instance_train.py`, every knob lives in `3_config.json`, and the notebooks are tutorial twins that import only what each cell needs.

## Run

```bash
mamba activate ssp-cuda            # ssp on CPU works for test/demo, slowly
cd 3_instance_models
```

**Notebooks — interactive, demo-scale:**

- `31_train_instance_classification.ipynb`
- `32_train_instance_regression.ipynb`

Open one, run cells top-to-bottom. The first cell is a GPU guard — type `y` to arm the reserved GPU (`shared.reserved_gpu` in `3_config.json`), anything else runs CPU.

**Py runners — full / unattended:**

```bash
python run_31_classification.py --mode full --use_gpu
python run_32_regression.py     --mode demo --use_gpu --target hr_bench_v3_age
```

Same engine, no prompts. Made for `tmux`. This is the right tool for full-corpus runs — notebooks hit kernel memory limits under full load.

## Knobs

Everything user-facing is in `3_config.json`:

- `run_mode` — `test` (plumbing only), `demo` (capped, ~1–2 h, tangible), `full` (caps off).
- `classification.target` / `regression.target` — name from the `TARGETS` registry in `utils_instance_train.py`. Call `available_targets()` to list.
- `shared.*` — model, batch size, LR, epochs, runs/models dirs, reserved GPU.
- `modes.*` — per-mode overrides on top of `shared`.

You should never need to edit the `.py` files — change values in `3_config.json`.

## Which encoder for which task?

Rule of thumb — pick w2v-BERT when the label depends on *what phoneme was
spoken at that moment*; pick wav2vec2-base / XLS-R when it depends on
*voice-timbre or acoustic envelope*.

| Task                         | Content-driven? | Preferred encoder          |
|------------------------------|-----------------|----------------------------|
| Gender                       | no              | anything, wav2vec2-base    |
| Speaker ID (within-speaker)  | no              | wav2vec2-base / XLS-R      |
| Age (regression)             | partly          | either                     |
| Filled-pause detection       | yes (phonetic shape) | w2v-BERT              |
| Primary stress (frame-level) | yes (syllable / prominence) | w2v-BERT       |
| Sentiment                    | mix             | either                     |

For a bigger model like `xls-r-300m`, bump the learning rate and the epoch
budget — the demo mode's 10 epochs at `lr 1e-5` under-trains it. On the
ParlaSpeech-HR benchmark v3 speaker-ID task, `wav2vec2-base` lands at 0.988
macro-F1 (TEST) while the same demo budget on `xls-r-300m` is still at 0.43.

## Output

Each run writes:

- `runs/<task>/<timestamp>/` — predictions JSONL (with full provenance and `pred_raw`), metrics, per-epoch logs, confusion matrix (classification) or scatter plot (regression).
- `models/<task>/<timestamp>/` — the trained model.

In `test` mode both mirror under `runs/test/` and `models/test/` so test outputs never collide with real ones.

## `legacy/`

Holds the frozen standalone `31`/`32` notebooks from before the shared-engine refactor — the whole engine inline, kept for reference. **Never edited.**
