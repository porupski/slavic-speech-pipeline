# Chapter 6 · Inference

Run pre-trained speech encoders over a folder of audio, emit an inference JSONL
plus (optionally) per-file TextGrids for visual QA in Praat.

## Variants

Chapter 6 mirrors chapter 3's task/head split. Four variants are planned; only
frame classification is implemented so far:

| #  | file                                       | status              |
|----|--------------------------------------------|---------------------|
| 61 | `61_instance_classification_inference`     | future              |
| 62 | `62_instance_regression_inference`         | future              |
| 63 | `63_frame_classification_inference`        | **implemented**     |
| 64 | `64_frame_regression_inference`            | future              |

The engine + config pattern is the same as chapter 3: knobs live in
`config.json`, the core lives in `utils_frame_infer.py` (61/62/64 will add
`utils_instance_infer.py`), and each variant has a paired notebook + headless
runner.

---

## 63 · frame classification inference

For models with per-frame classifier heads
(`Wav2Vec2BertForAudioFrameClassification`,
`Wav2Vec2ForAudioFrameClassification`, etc.). Default model is
[`classla/wav2vecbert2-filledPause`](https://huggingface.co/classla/wav2vecbert2-filledPause) —
the Slavic filled-pause detector.

### Run it

Point `audio_dir` in `config.json` at a folder of `.wav` / `.flac` / etc.
(recursive), then:

```bash
# Notebook (interactive; prompts for GPU or CPU)
jupyter lab 63_frame_classification_inference.ipynb

# Or headless runner
python run_63_frame_classification_inference.py -r my_run -m demo --use_gpu
```

Outputs land under `runs/{run_name}/`:

- `inference.jsonl`   — one JSON line per input file
- `textgrids/*.TextGrid` — two tiers per file (raw / postproc events)
- `examples.png` — 3 randomly-sampled waveform + FP-interval overlays
- `run_summary.txt`   — file / event counts, elapsed time

### Model selection

Default is `classla/wav2vecbert2-filledPause`, downloaded lazily into the
project-local `stock_models/` folder (via `HF_HOME` set inside
`utils_frame_infer.py`). Repeat runs use the cache — no re-download.

To use a different model, set `model_name` in `config.json`, or pass
`--model_name` on the runner. Any HF repo id or local checkpoint dir works,
provided it implements `AutoModelForAudioFrameClassification`. **Chapter-3
utterance-level checkpoints do NOT work here** — the head is different.

### Chunking

Long audio is split into non-overlapping fixed-length chunks
(`chunk_length_s`, default 30 s — matches how the FP model was trained).
Event timestamps are already global (chunk-offset applied). The per-file
`chunks` array in the JSONL records the boundaries so you can trace which
chunk covered which timestamp. Events straddling a boundary get split; for
filled pauses (typically <300 ms) this is rare.

Set `chunk_length_s: 0` to disable chunking (files must be pre-segmented).

### JSONL line schema

One line per input file. Order of keys is fixed; the optional `frames` array
(raw per-frame class ids) is appended at the very end when
`keep_frame_labels: true`, so JSONL previews stay readable.

```json
{
  "audio_path": "data/inference_input/foo.wav",
  "duration_s": 128.4,
  "model": "classla/wav2vecbert2-filledPause",
  "model_revision": "5e75061",
  "frame_ms": 20.0,
  "id2label": {"0": "no_fp", "1": "fp"},
  "chunks": [
    {"start_s": 0.0,  "end_s": 30.0},
    {"start_s": 30.0, "end_s": 60.0},
    {"start_s": 60.0, "end_s": 90.0},
    {"start_s": 90.0, "end_s": 120.0},
    {"start_s": 120.0, "end_s": 128.4}
  ],
  "raw_events":      [{"start_s": 12.34, "end_s": 12.66, "label": "fp", "mean_prob": 0.87}],
  "postproc_events": [{"start_s": 12.34, "end_s": 12.66, "label": "fp", "mean_prob": 0.87}],
  "postproc_applied": {
    "drop_short":     true,
    "short_cutoff_s": 0.08,
    "drop_initial":   true,
    "drop_final":     true
  },
  "frames": [0, 0, 0, 1, 1, ...]
}
```

Both `raw_events` and `postproc_events` are always present, so you can re-run
postproc offline against `raw_events` without touching the GPU.

### Postprocessing

Applied on top of `raw_events`, per the FP-BERT model card. Each flag is
independently toggleable in `config.json`:

- `drop_initial` — drop events starting at `0.0` s (often segmentation artefacts)
- `drop_final`   — drop events ending at the file end (same reason)
- `drop_short`   — drop events shorter than `short_cutoff_s` (default 0.08 s)

### Resume

Rerunning with the same `run_name` **skips input files already covered** in
`inference.jsonl`. Delete the JSONL (or use a fresh `run_name`) to force a
full re-run.

### Confidence

Per event we record `mean_prob` — the mean softmax probability of the winning
class over the event's frames. Useful for downstream thresholding.

### GPU

Same import-order contract as chapter 3: `CUDA_VISIBLE_DEVICES` must be set
**before** `utils_frame_infer` is imported. The notebook has an interactive
guard; the runner uses `--use_gpu`.

### Sanity check

For the default FP model, the HF page publishes ROG-Art dev F1 = 0.943 with
`drop_short + drop_initial + drop_final`. If you ever want a regression test,
point the runner at ROG-Art dev and check that your event list matches
theirs — pipeline correctness verifier.

---

## Future work

- **61 / 62 · instance-level inference.** Most of the plumbing here lifts
  directly; the frames→events step becomes a single-value emission (one label
  or one float per file). Landing point: `utils_instance_infer.py`.
- **64 · frame regression inference.** Rare, but useful if we ever train a
  per-frame continuous marker (e.g. instantaneous pitch or a VAD-adjacent
  score).
- **Metadata JSONL passthrough.** Optional companion JSONL keyed by
  `audio_path` that gets merged into each inference line under a `metadata`
  field. Would make chained pipelines (chapter-3 speaker-ID → chapter-6 FP)
  trivially composable — pass the ch3 output JSONL as the metadata source and
  every FP line gets the speaker id it was recorded from. Deferred; for now
  join the two JSONLs offline on the `audio_path` field (identical relative
  path in both, and both include `duration_s` as a sanity check).
