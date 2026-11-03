# Chapter 6 · Inference

Run pre-trained speech encoders over a folder of audio, emit an inference
JSONL plus (optionally) per-file TextGrids for visual QA in Praat.

Chapter 6 mirrors chapter 3 and chapter 4 on both axes: **shape** (frame vs
instance) and **head** (classification vs regression). The four quadrants
each have one entry point.

| #  | File                                              | Shape    | Head           | Status                        |
|----|---------------------------------------------------|----------|----------------|-------------------------------|
| 61 | `61_instance_classification_inference-WIP`        | instance | classification | SCAFFOLD (2026-11-02)         |
| 62 | `62_instance_regression_inference-WIP`            | instance | regression     | SCAFFOLD (2026-11-02)         |
| 63 | `63_frame_classification_inference`               | frame    | classification | **live** — FP + primary stress |
| 64 | `64_frame_regression_inference-WIP`               | frame    | regression     | SCAFFOLD (2026-11-02)         |

Each live quadrant has a notebook (`.ipynb`), its jupytext-paired `.py`
source of truth, and a CLI twin (`run_*.py`) for unattended runs in tmux.

## Tasks

Per-task knobs (which trained model to load, which postproc rules apply,
what task identifier to stamp into the JSONL) live in `6_tasks.json`.
Entries are grouped by quadrant:

```jsonc
{
  "frame_classification": {
    "fp":              { "model_name": "classla/wav2vecbert2-filledPause", ... },
    "primary_stress":  { "model_name": "auto", ... }
  },
  "frame_regression":        { "_readme": "scaffold" },
  "instance_classification": { "_readme": "scaffold" },
  "instance_regression":     { "_readme": "scaffold" }
}
```

Each entry point reads only its own quadrant. Pick a task via the `TASK`
constant at the top of the notebook, or `--task` on the CLI runner. Add a
new task by appending an entry under its quadrant.

Keys starting with `_` are documentation; the engine strips them before
layering.

## Model-name resolution

`model_name` in a task entry can be one of three forms:

1. `"auto"` — the engine reads `models/_best_models/<task_name>.json`, a
   trace file emitted by the training pipeline. Trace file schema:
   ```json
   { "model_path": "runs/<run>/_model/best_model",
     "metrics":    { "f1_positive": 0.9004 } }
   ```
   `metrics` is optional and informational only.
2. A project-relative directory that exists — resolved to its absolute
   path (so Hugging Face's `from_pretrained` treats it as a local
   checkpoint).
3. Anything else — passed through to `from_pretrained` as-is (HF repo
   id or absolute path).

`"auto"` is the preferred form for in-repo checkpoints — the chapter 4
training pipeline will emit the trace file so inference finds the current
best run without any config edit.

## Common config

Common knobs — the ones shared across every quadrant — live in
`6_config.json`: audio directory, runs directory, batch size, chunk length,
reserved GPU, and the three run-mode caps (`test`, `demo`, `full`).

## Running 63 (the one live entry point)

```bash
# Notebook (interactive; prompts for GPU or CPU)
jupyter lab 63_frame_classification_inference.ipynb
# Edit TASK at the top: "fp" or "primary_stress".

# Or headless runner
python run_63_frame_classification_inference.py --task fp              -m demo -r fp_demo --use_gpu
python run_63_frame_classification_inference.py --task primary_stress  -m demo -r ps_demo --use_gpu
```

Outputs land under `runs/{run_name}/`:

- `inference.jsonl` — one JSON line per input file
- `textgrids/*.TextGrid` — two tiers per file (raw / postproc events)
- `examples.png` — three randomly-sampled waveform + event overlays
- `run_summary.txt` — file / event counts, elapsed time

### JSONL line schema

One line per input file. Order of keys is fixed; the optional `frames`
array (raw per-frame class ids) is appended at the very end when
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
    {"start_s": 0.0,   "end_s": 30.0},
    {"start_s": 30.0,  "end_s": 60.0}
  ],
  "raw_events":      [{"start_s": 12.34, "end_s": 12.66, "label": "fp", "mean_prob": 0.87}],
  "postproc_events": [{"start_s": 12.34, "end_s": 12.66, "label": "fp", "mean_prob": 0.87}],
  "postproc_applied": { "drop_short": true, "drop_initial": true, "drop_final": true, "short_cutoff_s": 0.08 },
  "frames": [0, 0, 0, 1, 1, ...]
}
```

Both `raw_events` and `postproc_events` are always present, so you can
re-run postproc offline against `raw_events` without touching the GPU.

### Chunking

Long audio is split into non-overlapping fixed-length chunks
(`chunk_length_s`, default 30 s — matches how the FP model was trained).
Event timestamps are already global (chunk-offset applied). The per-file
`chunks` array records the boundaries so you can trace which chunk
covered which timestamp. Events straddling a boundary get split; for
filled pauses (typically < 300 ms) this is rare. Set `chunk_length_s: 0`
to disable chunking (files must be pre-segmented).

### Resume

Rerunning with the same `run_name` **skips input files already covered**
in `inference.jsonl`. Delete the JSONL (or use a fresh `run_name`) to
force a full re-run.

### Sanity check for FP

For the default FP model, the HF page publishes ROG-Art dev F1 = 0.943
with `drop_short + drop_initial + drop_final`. Point the runner at
ROG-Art dev and check your event list matches theirs — pipeline
correctness verifier.

## Files

| File | Purpose |
|---|---|
| `63_frame_classification_inference.py` / `.ipynb` | One entry point for every frame-classification task. Picks a task via a `TASK` constant at the top. Live tasks: `fp`, `primary_stress`. Uses `utils_frame_infer.py`. |
| `run_63_frame_classification_inference.py` | Headless CLI twin of 63. Required `--task` flag picks the task. |
| `61_instance_classification_inference-WIP.py` / `.ipynb` | SCAFFOLD (2026-11-02). Entry point for the instance_classification quadrant. Candidate first tasks: `gender`, `FP_presence`, `speaker_id`. Needs `utils_instance_infer.py` ported. |
| `run_61_instance_classification_inference-WIP.py` | SCAFFOLD CLI twin of 61. |
| `62_instance_regression_inference-WIP.py` / `.ipynb` | SCAFFOLD (2026-11-02). Entry point for the instance_regression quadrant. Candidate first tasks: `age`, `sentiment`. Needs `utils_instance_infer.py` ported. |
| `run_62_instance_regression_inference-WIP.py` | SCAFFOLD CLI twin of 62. |
| `64_frame_regression_inference-WIP.py` / `.ipynb` | SCAFFOLD (2026-11-02). Entry point for the frame_regression quadrant. Needs a new `utils_frame_regression_infer.py` module (sibling of the live `utils_frame_infer.py`). |
| `run_64_frame_regression_inference-WIP.py` | SCAFFOLD CLI twin of 64. |
| `utils_frame_infer.py` | The frame-classification engine. Reads `6_config.json` + `6_tasks.json/frame_classification/<task>` + run-mode overrides. Handles chunking, batched forward pass, frames → events + postproc, JSONL + TextGrid writes, waveform plots. `QUADRANT = "frame_classification"`. |
| `utils_instance_infer.py` | SCAFFOLD (2026-11-02). Will serve both instance quadrants once wired. Port template: chapter 3's `utils_instance_train.py` predict path. |
| `6_config.json` | Common knobs (audio dir, runs dir, batch size, chunk length, reserved GPU, run-mode caps). Shared across every task. |
| `6_tasks.json` | Task registry. Entries grouped by quadrant. Each entry carries model, postproc, task identifier, output flags. |
| `vowels_by_language.json` | Grapheme-level vowel sets per language. Used by the future vowel-nucleus accuracy metric. HR, RS, SI include `r` (syllabic liquid). |
| `6_README.md` | This file. |
