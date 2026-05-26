# Chapter 1 — Data preparation

This chapter has one job: take any source dataset (in any format) and emit a **canonical JSONL** that every downstream chapter knows how to consume.

If chapter 1 is right, the rest of the pipeline is dataset-agnostic. If chapter 1 is wrong, we patch forever.

---

## 1. What "instance" means in this pipeline

An **instance** is the trained-on unit. It is the thing the model gets as input and produces an output for.

That unit can be either:

- **An utterance** — a whole sentence / segment / audio clip. Used for *audio-instance* tasks where the model emits one label per clip (e.g. "this sentence's sentiment is positive").
- **A sub-utterance segment** — a slice of a larger audio file marked by `start_t` / `end_t`. Used for things like "this 200ms window contains a filled pause."
- **A whole audio file with frame-level labels inside it** — used for *audio-frame* tasks (chapter 4). The model gets the whole clip and emits a sequence of labels, one per audio frame (typically 50 Hz for Wav2Vec2). The "instance" here is the whole clip; the labels are a list aligned to frames.

The model and trainer don't care which kind it is. They get a path to a WAV and a label. The trainer's `task_type` (`"classification"` / `"regression"`) tells it how to read the label; the head shape (single value vs. sequence) is chosen by the chapter (3 = instance, 4 = frame).

**Rule:** one JSONL line = one instance, whichever kind.

---

## 2. The canonical JSONL contract

One JSONL file per dataset. One JSON object per line. UTF-8.

### Required keys (always present, on every line)

| Key | Type | Meaning |
|---|---|---|
| `instance_id` | str | Globally unique. Convention: `{dataset}_{file_id}_{speaker}_{start}_{end}`. |
| `dataset` | str | Source dataset name. `"ROG"`, `"GOS"`, `"ParlaSpeech-HR"`, etc. |
| `audio_path` | str | **Project-relative** path to the WAV that the trainer will load. Typically under `data/cut_audio/<dataset>/...`. |
| `split` | str | One of `"train" | "dev" | "test"`. Always assigned. |
| `labels` | dict | Always present. May be empty `{}` if the dataset has no labels yet. |
| `metadata` | dict | Always present. May be empty `{}`. Free-form bag for anything else. |

### Optional keys (present when the dataset has them)

| Key | Type | When |
|---|---|---|
| `file_id` | str | Identifier of the source audio file before cutting. Useful for grouping. |
| `start_t` | float | Start time in seconds, in the *source* audio. Omitted when the WAV at `audio_path` is already pre-cut and represents the whole instance. |
| `end_t` | float | End time in seconds, in the *source* audio. Same rule as `start_t`. |
| `speaker` | str | Speaker ID. Omit if unknown. |
| `text` | str | Orthographic transcription if available. |
| `frame_rate_hz` | int | **Only for frame-level entries.** Tells the trainer the rate at which the label sequence is sampled (e.g. `50`). |

### `labels` dict — instance-level vs frame-level

**Instance-level** (chapter 3): each value is a scalar.

```json
"labels": {
  "sentiment": "neutralPositive",
  "sentiment_score": 0.3,
  "dialogue_act_function": "question",
  "filled_pause_present": 1
}
```

**Frame-level** (chapter 4): each value is a list of scalars, length matches the audio length at `frame_rate_hz`.

```json
"labels": {
  "primary_stress": [0, 0, 0, 1, 1, 0, 0, ...]
}
"frame_rate_hz": 50
```

A single instance MAY mix scalar and sequence labels in the same dict (e.g. one sentence-level sentiment plus per-frame stress). The trainer reads only the `label_key` from its config, so other keys are ignored.

### `metadata` dict — free-form

Anything else useful that isn't a label. The trainer ignores this; sniff scripts and error-analysis use it.

```json
"metadata": {
  "speaker_gender": "f",
  "speaker_age": "32",
  "source_file": "ROG-Art-S01-V01.wav",
  "source_format": "EXB"
}
```

---

## 3. Worked examples — one line per dataset

### ROG (instance, sentiment + dialogue act)

```json
{
  "instance_id": "ROG_Art-S01-V01_SPK0_2.190_4.580",
  "dataset": "ROG",
  "file_id": "Art-S01-V01",
  "audio_path": "data/cut_audio/ROG/Art-S01-V01_SPK0_2.190_4.580.wav",
  "start_t": 2.190,
  "end_t": 4.580,
  "speaker": "SPK0",
  "split": "train",
  "text": "Tudi jaz mislim, da je to v redu.",
  "labels": {
    "sentiment": "neutralPositive",
    "sentiment_score": 0.3,
    "dialogue_act_function": "statement"
  },
  "metadata": {
    "speaker_gender": "f",
    "source_file": "Art-S01-V01.wav",
    "source_format": "EXB"
  }
}
```

### GOS-stress (frame, primary stress)

```json
{
  "instance_id": "GOS_Artur-N-G0001-P600001_0.000_3.240",
  "dataset": "GOS",
  "file_id": "Artur-N-G0001-P600001",
  "audio_path": "data/cut_audio/GOS/Artur-N-G0001-P600001.wav",
  "split": "train",
  "text": "Pa potem sva šla domov.",
  "frame_rate_hz": 50,
  "labels": {
    "primary_stress": [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, ...]
  },
  "metadata": {
    "source_format": "TextGrid"
  }
}
```

### ParlaSpeech-HR (instance, filled pause)

```json
{
  "instance_id": "ParlaSpeech-HR_session-2019-12-19-12-30_38_42.110_44.870",
  "dataset": "ParlaSpeech-HR",
  "file_id": "session-2019-12-19-12-30_38",
  "audio_path": "data/cut_audio/ParlaSpeech-HR/session-2019-12-19-12-30_38_42.110_44.870.wav",
  "start_t": 42.110,
  "end_t": 44.870,
  "speaker": "Plenkovic_Andrej",
  "split": "train",
  "text": "Hvala lijepa eee gospodine predsjedniče.",
  "labels": {
    "filled_pause_present": 1
  },
  "metadata": {
    "source_format": "JSONL"
  }
}
```

---

## 4. Audio file conventions

- Format: **WAV, 16 kHz, mono, 16-bit PCM**.
- Each instance has its own pre-cut WAV file referenced by `audio_path`. The trainer **never** re-cuts on the fly.
- `audio_splitter.py` does the cutting once, given a JSONL with `start_t` / `end_t` populated. Output is the same JSONL with `audio_path` rewritten to point at the cut clip.
- Paths in JSONL are **project-relative** (start with `data/...`). The trainer resolves them against `PROJECT_ROOT` at load time.
- For long whole-file inputs (frame chapter), no cutting at training time; cutting happens inside the trainer's dataloader if needed (sliding window).

---

## 5. Splits

`split` is always one of `"train"`, `"dev"`, `"test"`. No `"unassigned"`.

If the source dataset ships with predefined splits, `prep_<dataset>.py` uses them.

If not, `utils_dataprep.assign_splits()` assigns them **deterministically** by hashing `file_id` (so the same file always lands in the same split across re-runs, and so all instances from one file go to the same split — no leak). Default ratio: 80 / 10 / 10.

**No instance from the same `file_id` may appear in two splits.** Splitter enforces this.

---

## 6. Project-relative paths everywhere

No absolute paths in committed code. Each script resolves `PROJECT_ROOT`:

```python
# In any .py
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # one up from chapter folder

# In any .ipynb
from pathlib import Path
import os
# Notebook starts inside its chapter folder
PROJECT_ROOT = Path.cwd().parent
```

All paths in configs are strings relative to `PROJECT_ROOT`. The data loader joins them:

```python
audio_full_path = PROJECT_ROOT / instance["audio_path"]
```

---

## 7. `test_mode` — every runnable script supports it

Every `prep_*.py`, `train_*.py`, `inference_*.py`, and `sniff_*.py` has a `test_mode: bool = False` field in its Config dataclass.

When `test_mode = True`:

- Processes at most **N = 10** instances (configurable as `test_n`).
- Training runs **1 epoch**, **batch size 2**, **no gradient accumulation**.
- Skips heavy downloads if cached partial data exists.
- Writes outputs into a `test_runs/` or `test_processed_jsonl/` directory so it can't pollute real outputs.
- Prints `🧪 TEST MODE` at the top so it's obvious in logs.

Same code path otherwise — `test_mode` is a thin clamp, not a separate branch.

---

## 8. Folder structure produced by chapter 1

```
data/
├── raw/                              ← downloaded archives, untouched
│   ├── ROG/
│   │   ├── ROG.zip
│   │   └── ROG-Art.wav.zip
│   ├── GOS/
│   └── ParlaSpeech-HR/
│
├── unpacked/                         ← extracted source files
│   ├── ROG/
│   │   ├── ROG-Art/...
│   │   └── wav/...
│   └── ...
│
├── cut_audio/                        ← per-instance WAVs (16k mono)
│   ├── ROG/
│   ├── GOS/
│   └── ParlaSpeech-HR/
│
└── processed_jsonl/                  ← the canonical JSONL files
    ├── rog_instance.jsonl
    ├── gos_frame.jsonl
    └── parlaspeech_hr_pause.jsonl
```

`data/` is gitignored.

---

## 9. Scripts in this chapter

| Script | Job |
|---|---|
| `download_data.ipynb` / `.py` | Curl + unpack from CLARIN. Idempotent. `--dataset` flag selects which. |
| `utils_dataprep.py` | JSONL I/O, schema validation, cleaner, splitter, project-root resolver, EXB / TextGrid parsing helpers. |
| `prep_ROG.ipynb` / `.py` | EXB → canonical JSONL (instance-level, sentiment + dialogue act). |
| `prep_GOS.ipynb` / `.py` | TextGrid → canonical JSONL (frame-level, primary stress). |
| `prep_ParlaSpeech.ipynb` / `.py` | JSONL → canonical JSONL (placeholder for v1). |
| `audio_splitter.py` | Given a canonical JSONL with `start_t` / `end_t`, cut source WAVs into per-instance clips and rewrite `audio_path`. |

**Develop notebooks first.** Once a notebook is stable, distill the body into the matching `.py` and verify it runs identically end-to-end. Both import from the same `utils_dataprep.py`.

---

## 10. Schema validation

`utils_dataprep.validate_instance(obj)` checks every line against the contract above. It is called:

- At the end of every `prep_*.py` before writing.
- At the start of `sniff_dataset.py` and every trainer's data loader.

Failure modes are loud: it prints the offending `instance_id` and the first failing rule, then raises. No silent skips. Use `--strict` to crash on first error; default behavior reports all errors and exits non-zero.

---

## 11. Data sources & citations

### ROG 1.1 — Slovenian spoken corpus (with annotations)

- Handle: `http://hdl.handle.net/11356/2062`
- Files used: `ROG.zip` (annotations, EXB + TEI + TRS) and `ROG-Art.wav.zip` (audio, 44.1 kHz 16-bit mono).
- Annotations include: lemmas, MSDs, UD, prosodic units, disfluencies, dialogue acts, sentiment.
- Citation: Verdonik, Darinka; et al., 2026, *Training corpus of spoken Slovenian ROG 1.1*, CLARIN.SI.

### GOS 2.1 — Slovenian reference speech corpus

- Handle: `http://hdl.handle.net/11356/1863`
- Audio is under a restricted licence, separate handle: `http://hdl.handle.net/11356/1973`.
- Primary-stress TextGrids are a **separate, manually annotated layer** on top of GOS audio (not part of the CLARIN release). These come from the user's own annotation work.
- Citation: Verdonik, Darinka; et al., 2023, *Spoken corpus Gos 2.1 (transcriptions)*, CLARIN.SI.

### ParlaSpeech 3.0 — multilingual parliamentary speech

- Handle: `http://hdl.handle.net/11356/1833`
- Project page: <https://clarinsi.github.io/parlaspeech/>
- Languages: HR, RS, CZ, PL. Total download is large (~120 GB+). The downloader supports per-language selection.
- Enrichment layers used in this pipeline: **filled pauses** (all 4 langs), **primary stress** (HR, RS).
- Citation: see CLARIN handle.

---

## 12. Run order for this chapter

```bash
cd 1_data_prep

# 1. Download (idempotent, skips if already present)
python download_data.py --dataset ROG

# 2. Convert source format to canonical JSONL (still needs audio cut)
python prep_ROG.py
# → data/processed_jsonl/rog_instance.raw.jsonl  (audio_path points at source)

# 3. Cut per-instance WAVs and rewrite audio_path
python audio_splitter.py --jsonl data/processed_jsonl/rog_instance.raw.jsonl \
                        --out   data/processed_jsonl/rog_instance.jsonl \
                        --out-audio data/cut_audio/ROG/

# 4. Verify
python ../2_data_analysis/sniff_dataset.py --jsonl data/processed_jsonl/rog_instance.jsonl
```

For test runs, set `test_mode = True` in each script's Config; everything writes to `test_*` mirrors so real outputs are safe.

---

## 13. Gotchas

- **ROG WAVs are 44.1 kHz 16-bit mono**, not 16 kHz. `audio_splitter.py` resamples to 16 kHz on the way out.
- **EXB files have nested speaker tiers**; the parser must thread sentiment / dialogue-act annotations back to the right speaker's timestamps. The current `1i0_Extract_info_from_EXB.py` handles this; we port its logic into `utils_dataprep.py`.
- **GOS audio is on a separate, restricted CLARIN handle** — downloader needs to handle the auth flow or document a manual step.
- **ParlaSpeech is huge.** Default download is HR only; full pull is opt-in.
- **TextGrid label-to-frame alignment** for GOS stress needs the source WAV's duration and a chosen `frame_rate_hz`. `utils_dataprep.textgrid_to_frame_labels()` handles this when chapter 4 starts.
