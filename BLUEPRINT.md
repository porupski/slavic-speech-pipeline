# Speech ML Pipeline — Project Blueprint

A clean, modular, plug-and-play pipeline for fine-tuning speech models on audio instance and audio frame tasks (classification + regression).

The goal: scripts so clean, organized, and easy to follow that someone can clone the repo and run an experiment within an hour.

---

## 1. Guiding principles

These are non-negotiable. Every design choice flows from them.

1. **Lego, not monolith.** Each stage (data prep, training, inference, analysis) is a self-contained module. Outputs of one stage are the inputs of the next via a single, agreed-on data format.
2. **One job per script.** No script does two things. If it feels like two things, split it.
3. **Py + ipynb pairs sharing a utils.py.** The `.py` is the lean production runner. The `.ipynb` is the same logic broken into cells, with markdown commentary, made for learning, debugging, and adapting. Both import from the same `utils_*.py` module — zero code duplication.
4. **Task type is a parameter, not a fork.** Classification and regression share ~90% of the code. They are one pipeline with `TASK_TYPE = "classification" | "regression"` as a config flag. No `_classification.py` + `_regression.py` duplication.
5. **Dataset is a parameter, not a fork.** Every dataset has its own `data_prep/` script that converts its native format into the same canonical JSONL. After that, the rest of the pipeline doesn't know or care which dataset it came from.
6. **Configs live at the top of each script in a single dataclass.** No buried magic constants, no scattered hardcoded paths. One block to edit, top of the file.
7. **More markdown, not less.** Each chapter folder gets its own README.md. Aggressive documentation now is cheaper than archaeology later.
8. **QoL only where it earns its keep.** Per-epoch checkpoint logging — yes. Resume training, hyperparam search, distributed training — not yet.

---

## 2. Canonical data format

The single agreed-on format that every dataset producer outputs and every consumer reads. This is the contract that makes the lego work.

**One JSONL line = one instance.**

```json
{
  "instance_id": "ROG-dialog-0021_SPK0_2.190_2.200",
  "dataset": "ROG",
  "file_id": "ROG-dialog-0021",
  "audio_path": "/abs/path/to/instance.wav",
  "start_t": 2.190,
  "end_t": 2.200,
  "speaker": "SPK0",
  "split": "train",
  "text": "Tudi.",
  "labels": {
    "sentiment": "neutralPositive",
    "dialogue_act_function": "question"
  },
  "metadata": {
    "speaker_gender": "f",
    "speaker_age": "32",
    "source_file": "ROG-dialog-0021.wav"
  }
}
```

**Rules.**

- `instance_id` is globally unique across all datasets. `{dataset}_{file_id}_{speaker}_{start}_{end}` works.
- `audio_path` is an absolute path to a pre-cut WAV (16 kHz mono). The splitter does this once; downstream never re-cuts.
- `split` is `"train" | "dev" | "test"`. Already assigned by the data prep stage.
- `labels` is a dict. Multiple labels per instance are fine — the trainer picks which key to use.
- `metadata` is a free-form dict for anything else useful (gender, age, source, etc.). The trainer ignores it but analysis scripts can use it.

**For frame-level tasks** (chapter 4), labels become a list aligned to a fixed frame rate:

```json
{
  "instance_id": "...",
  "audio_path": "...",
  "frame_rate_hz": 50,
  "labels": {
    "primary_stress": [0, 0, 0, 1, 1, 0, 0, ...]
  }
}
```

Same JSONL, same loaders, just the label shape differs.

---

## 3. Repository layout

```
speech-ml-pipeline/
├── README.md                          ← entry point, links to chapters
├── BLUEPRINT.md                       ← this file
├── requirements.txt
├── .gitignore
│
├── 1_data_prep/
│   ├── README.md                      ← what the canonical format is, how to add a new dataset
│   ├── utils_dataprep.py              ← shared: JSONL I/O, audio cutting, split assignment
│   ├── prep_ROG.py                    ← EXB → canonical JSONL
│   ├── prep_ROG.ipynb
│   ├── prep_GOS.py                    ← TextGrid → canonical JSONL (primary stress)
│   ├── prep_GOS.ipynb
│   ├── prep_ParlaSpeech.py            ← (future)
│   └── audio_splitter.py              ← cuts source WAVs into per-instance clips
│
├── 2_data_analysis/
│   ├── README.md
│   ├── utils_analysis.py              ← shared plotting + stats
│   ├── sniff_dataset.py               ← descriptive stats for any canonical JSONL
│   ├── sniff_dataset.ipynb
│   ├── label_distributions.py
│   └── audio_duration_stats.py
│
├── 3_instance_models/
│   ├── README.md                      ← classifier + regressor explained side-by-side
│   ├── utils_instance.py              ← shared model code, training loop, metrics dispatch
│   ├── train_instance.py              ← single script, TASK_TYPE switches behavior
│   ├── train_instance.ipynb
│   ├── inference_instance.py
│   └── inference_instance.ipynb
│
├── 4_frame_models/                    ← chapter 4, slovenian primary stress
│   ├── README.md
│   ├── utils_frame.py
│   ├── train_frame.py
│   ├── train_frame.ipynb
│   ├── inference_frame.py
│   └── inference_frame.ipynb
│
├── 5_analysis/                        ← post-hoc, after a model is trained
│   ├── README.md
│   ├── utils_results.py
│   ├── find_best_epoch.py             ← (replaces 3i0_Find_best_epoch_performer)
│   ├── error_analysis.py
│   ├── confusion_matrix_viewer.ipynb
│   └── compare_runs.py
│
└── data/                              ← gitignored, local only
    ├── raw/
    ├── processed_jsonl/
    └── cut_audio/
```

**Naming convention inside each chapter:**

- `utils_<chapter>.py` — the shared boilerplate
- `<verb>_<noun>.py` + `<verb>_<noun>.ipynb` — runner + notebook pair
- `README.md` — chapter overview, run order, gotchas

---

## 4. Chapter-by-chapter battle plan

### Chapter 1 — Data prep

**Goal:** produce a canonical JSONL per dataset.

**Order of work:**

1. Lock down the canonical JSONL spec (section 2 above). Write it into `1_data_prep/README.md`.
2. Port the EXB extractor (`1i0_Extract_info_from_EXB.py`) into `prep_ROG.py`, but emit canonical JSONL instead of the current bespoke format. The existing logic for timestamp matching, sentiment overlap, and validation is reusable as-is — wrap it.
3. Port the instance-level audio splitter (`10i1_Split_Wavs_by_json.py`) into `audio_splitter.py`. Make it dataset-agnostic: input is a canonical JSONL, output is one WAV per instance + an updated JSONL with `audio_path` populated.
4. Write `prep_GOS.py`: TextGrid → canonical JSONL with frame-level primary stress labels. (Schema in section 2.)
5. Write the cleaning logic (`4i1_Filter_out_data.py`) into `utils_dataprep.py` as a function. Don't make it a separate stage — make it a step inside each `prep_*.py`.

**Out of scope for now:** `prep_ParlaSpeech.py` (listed but not built).

**Deliverable check:** running `python prep_ROG.py` produces `data/processed_jsonl/rog_instance.jsonl` ready for training. Running `python prep_GOS.py` produces `data/processed_jsonl/gos_frame.jsonl`.

---

### Chapter 2 — Data analysis / sniff

**Goal:** look at any canonical JSONL and immediately understand what's in it. No training, no models — just stats and plots.

**Scripts:**

- `sniff_dataset.py` — counts instances, label distributions (per split), audio duration histogram, speaker counts, missing-value report. Reads canonical JSONL, prints + saves a markdown report.
- `audio_duration_stats.py` — duration percentiles, useful for picking `max_length` for the trainer.
- `label_distributions.py` — pretty plots of class balance, with imbalance warnings.

**Why this comes before training:** because every time you load a new dataset you want to see what's in it before you waste 4 hours training on garbage. This chapter is your "is the data prep stage working" check.

**Deliverable check:** `python sniff_dataset.py --jsonl rog_instance.jsonl` prints a clean report and saves a few PNGs.

---

### Chapter 3 — Audio instance models (classifier + regressor)

**Goal:** one trainer that does both classification and regression for instance-level tasks. Wav2Vec2-based.

**Single config block at the top of `train_instance.py`:**

```python
@dataclass
class Config:
    # Data
    jsonl_path: str = "data/processed_jsonl/gos_instance.jsonl"  # or rog_instance.jsonl
    label_key: str = "primary_stress_present"                    # or "sentiment", etc.
    task_type: str = "classification"                            # or "regression"

    # Model
    model_name: str = "facebook/wav2vec2-xls-r-300m"
    freeze_feature_encoder: bool = True

    # Training
    batch_size: int = 8
    grad_accum: int = 2
    learning_rate: float = 1e-5
    num_epochs: int = 20
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "runs/"
    use_cuda: bool = True
    cuda_device: str = "0"
```

**Architecture:**

- `utils_instance.py` holds: dataset loader (reads canonical JSONL), preprocessing, the `EpochCheckpointCallback`, the model factory (returns a classification head or regression head depending on `task_type`), and a metrics dispatcher (`macro_f1`, `accuracy`, `spearman` for classification; `mse`, `mae`, `spearman` for regression).
- `train_instance.py` is small — config dataclass, then `main()` that calls into `utils_instance` for everything.
- `train_instance.ipynb` walks through the same steps with markdown explanations between cells. Imports the same `utils_instance.py`.
- Two-phase training (TRAIN→DEV for development, then TRAIN+DEV→TEST for final eval) preserved from your current scripts — it's a good pattern, keep it.

**Recommended starting target: GOS primary-stress at the instance level** (does this word/segment contain primary stress, yes/no — binary classification). It's clean, you have ground truth, and it makes a strong first sanity check before touching ROG sentiment.

**Then add ROG as a second target** by changing only `jsonl_path` and `label_key`.

**Deliverable check:** `python train_instance.py` runs end-to-end, saves per-epoch logs + confusion matrix, and writes a best-model directory.

---

### Chapter 4 — Audio frame models (Slovenian primary stress)

**Goal:** frame-level classification (and optionally regression) — predict a label for every 20ms frame of audio.

**What changes from chapter 3:**

- Model head outputs `(seq_len, num_labels)` instead of `(num_labels,)`.
- Labels are sequences aligned to the model's output frame rate (typically 50 Hz for Wav2Vec2).
- Metrics now include frame-level F1, IoU/boundary metrics, plus optional event-level metrics (start/end of stress region).
- The sliding-window audio splitter (`0i0_split_audio_60s.py`) becomes relevant for handling long files at inference time.

**What stays the same:**

- Canonical JSONL format (just with sequence labels — see section 2).
- The `EpochCheckpointCallback` pattern.
- Two-phase TRAIN→DEV / TRAIN+DEV→TEST training.
- The config-dataclass-at-top convention.

**Reference:** the parlastress repo (Croatian primary stress) — pull in its model head and frame-alignment logic when we hit this chapter.

**Order of work:**

1. Get instance-level GOS working in chapter 3 first. Don't start chapter 4 until chapter 3 is solid.
2. Then re-prep the GOS data with frame-level labels (`prep_GOS.py` already designed for this in section 2).
3. Adapt the chapter 3 trainer into `train_frame.py`. Most of `utils_instance.py` is reusable — `utils_frame.py` adds the sequence-head model and the frame-level metrics.

**Deliverable check:** `python train_frame.py` trains on GOS Slovenian primary stress and produces frame-level F1 comparable to the Croatian parlastress results.

---

### Chapter 5 — Post-hoc analysis

**Goal:** after a model has trained, understand what it did.

**Scripts:**

- `find_best_epoch.py` — the cleaned-up version of `3i0_Find_best_epoch_performer.py`. Works for both classification and regression run directories.
- `error_analysis.py` — for a given run + epoch, pull the predictions JSON, find the worst N misclassifications, and dump audio paths + transcripts + gold/pred labels into a CSV for manual inspection.
- `compare_runs.py` — load multiple run directories and produce side-by-side metric comparisons.
- `confusion_matrix_viewer.ipynb` — interactive notebook for poking at confusion matrices across epochs.

**Why a separate chapter:** these scripts run *after* training and don't need the model loaded. They operate purely on the JSON output of chapter 3/4 runs. Keeping them separate keeps the training chapters lean.

---

## 5. The py + ipynb pattern, concretely

Yes, notebooks can import from `.py` files in the same folder. Standard pattern:

```python
# In train_instance.ipynb, cell 1:
from utils_instance import (
    load_canonical_jsonl,
    build_model,
    EpochCheckpointCallback,
    compute_metrics,
)
```

For the import to work, either (a) the notebook is started from the chapter folder, or (b) the notebook adds the folder to `sys.path`. The notebook can include this defensively:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))  # or path to chapter folder
```

**The pair convention:**

- `train_instance.py` — top of file: config dataclass + `main()`. Bottom: `if __name__ == "__main__": main()`. Total length: \~100-150 lines. Easy to read top-to-bottom in one sitting.
- `train_instance.ipynb` — markdown cell explaining what the chapter does, then cells that walk through the exact same steps `main()` does, but with checkpoints between them ("OK, the dataset loaded — let's look at one example" etc.). Imports the same utils so logic is never duplicated.

**Rule:** if you find yourself writing logic in the notebook that isn't in the `.py`, that logic belongs in `utils_*.py`. The notebook is for orchestration and exposition, not implementation.

---

## 6. What we are deliberately *not* doing yet

To stay focused and not let scope creep eat the project:

- No ASR. Tempting, but it's a different paradigm (seq2seq/CTC, WER) and adds a third code path. Sentiment/stress classification and regression cover the instance + frame matrix already.
- No multi-task heads. One label per training run.
- No hyperparam search. Manually pick configs, log them, compare with `compare_runs.py`.
- No distributed training. Single-GPU is enough for now.
- No model serving / API. Inference scripts produce JSON; that's the deliverable.
- No `prep_ParlaSpeech.py` implementation. Listed as a placeholder.
- No real-time / streaming inference.

---

## 7. Execution order — what to actually build first

Strict order. Don't skip ahead.

**Phase A — foundation (chapter 1 + 2)**

1. Write `1_data_prep/README.md` with the canonical JSONL spec.
2. Write `utils_dataprep.py` (JSONL I/O, splitter, cleaner).
3. Write `prep_GOS.py` — primary-stress, instance-level. This is your sanity-check dataset, so it goes first.
4. Write `audio_splitter.py`. Run it against GOS output to cut WAVs.
5. Write `sniff_dataset.py` (chapter 2). Point it at the GOS JSONL. Confirm everything looks sane.

**Phase B — first model (chapter 3)**

6. Write `utils_instance.py` and `train_instance.py`. Run on GOS primary-stress as binary classification. This is the moment of truth: pipeline works end-to-end on a clean task.
7. Write `inference_instance.py`. Confirm it loads a saved model and runs.
8. Write `find_best_epoch.py` (chapter 5) — small, useful, gives the training loop closure.

**Phase C — port ROG into the pipeline**

9. Write `prep_ROG.py` emitting canonical JSONL.
10. Run `sniff_dataset.py` on the ROG output. Look at label balance.
11. Run `train_instance.py` with ROG sentiment as target — change two config lines.
12. Run `train_instance.py` again with `task_type="regression"` — change one config line.

**Phase D — frame models (chapter 4)**

13. Re-prep GOS with frame-level labels.
14. Write `utils_frame.py` and `train_frame.py`. Reuse as much from chapter 3 as possible.
15. Train Slovenian primary stress frame classifier. Compare to parlastress Croatian results.

**Stop here.** This is the project's defined endpoint.

---

## 8. Future plans (out of scope for v1)

Park these in a `FUTURE.md` once the v1 is done:

- ParlaSpeech data prep.
- Multi-task learning (sentiment + dialogue act jointly).
- Frame-level regression (e.g. continuous stress strength).
- Cross-lingual transfer experiments.
- ASR baseline for comparison.
- Hyperparameter sweep tooling.
- An inference web demo.

---

## 9. Naming + small conventions

- Folders: lowercase with underscores. Numeric prefix shows order (`1_data_prep`).
- Scripts: `verb_noun.py`. `train_instance.py`, not `instance_train.py` or `tr_inst.py`.
- Utils: `utils_<chapter>.py`. One per chapter.
- Configs: always at the top of the runnable script, in a `@dataclass` named `Config`.
- Run output folders: `runs/{dataset}_{task}_{timestamp}_{hp_summary}/` — same as your current pattern, which is good.
- Emoji in print statements: fine, keeps logs readable. Don't go overboard.
- Comments: more is better than less for the first version. Strip later if it gets noisy.

---

## 10. Sanity checklist before declaring v1 done

- [ ] A new person can clone the repo and run `python train_instance.py` (with the data already downloaded) without editing anything other than the config block.
- [ ] Each chapter folder has a README explaining what's in it and how to run it.
- [ ] No code is duplicated between classification and regression.
- [ ] No code is duplicated between `.py` and `.ipynb` — both import the same utils.
- [ ] Adding a new dataset means writing one `prep_*.py`. Nothing else changes.
- [ ] Adding a new target label means changing two lines of config. Nothing else changes.
- [ ] The frame model works on Slovenian primary stress with reasonable F1.
