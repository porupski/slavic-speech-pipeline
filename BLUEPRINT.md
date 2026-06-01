# Speech ML Pipeline — Project Blueprint

A modular, plug-and-play pipeline for fine-tuning Wav2Vec2 models on Slavic speech corpora. Instance- and frame-level classification (regression deferred).

**Mission for v1:** end-to-end ROG-Art filled-pause frame model. Clone the repo → run the notebooks in order → working classifier.

---

## 1. Guiding principles

1. **Lego, not monolith.** Each stage is a self-contained notebook connected via canonical JSONL.
2. **One job per notebook.** No notebook does two things (the dual-output prep is one job: extract from EXB).
3. **Notebook-first, py runners last.** Notebooks are the production artifact for v1. Final ceremonial step before shipping: refactor into `main.py` + `utils_<chapter>.py`. Not before.
4. **Task type is a parameter.** Classification + regression share the pipeline; one config flag switches. (Regression deferred for chapter 4.)
5. **Dataset is a parameter.** Each corpus has one prep notebook; downstream is dataset-agnostic.
6. **Config dataclass at the top of every notebook.** No buried constants.
7. **Aggressive markdown.** Each chapter has a README. Each notebook has section headers.
8. **QoL only where it earns its keep.** Per-epoch logging — yes. Resume training, hyperparam search, distributed — no.

---

## 2. Canonical data format

One JSONL line = one record. Same schema across instance and frame flavors; only `labels` differs in shape.

**Instance record:**
```json
{
  "instance_id": "ROG-Art_Rog-Art-J-G3003_2.190_3.450",
  "dataset": "ROG-Art",
  "file_id": "Rog-Art-J-Gvecg-P500001",
  "audio_path": "data/cut_audio/ROG-Art/<file>.wav",
  "split": "train",
  "speaker": "Artur-J-G3003",
  "start_t": 2.190, "end_t": 3.450,
  "text": "eee dober dan",
  "labels": {
    "sentiment": "neutralPositive",
    "filled_pause_present": 1
  },
  "metadata": {"source_file": "...", "subcorpus": "Artur-J"}
}
```

**Frame record** (one per source WAV, full duration):
```json
{
  "instance_id": "ROG-Art_Rog-Art-J-Gvecg-P500001_0.000_2613.000",
  "dataset": "ROG-Art",
  "file_id": "Rog-Art-J-Gvecg-P500001",
  "audio_path": "data/cut_audio/ROG-Art-Full/<file>.wav",
  "split": "train",
  "start_t": 0.0, "end_t": 2613.000,
  "frame_rate_hz": 50,
  "labels": {"filled_pause": [0, 0, 1, 1, 0, ...]},
  "metadata": {"n_frames": {"filled_pause": 130650}}
}
```

**Rules.**
- `instance_id` globally unique. `{dataset}_{file_id}_{speaker?}_{start}_{end}` works.
- `audio_path` is **project-relative** (e.g. `data/cut_audio/...`), not absolute. `utils_dataprep.from_project_relative` resolves it.
- `audio_path` points to a **16 kHz mono WAV.** Instance: per-segment cut. Frame: normalized full-file copy.
- `split` ∈ `{train, dev, test}`. Assigned by prep, grouped by `file_id` so no recording leaks.
- `labels` is a dict; trainer picks one key. Multiple labels per record encouraged.
- `metadata` is free-form; trainer ignores it, analysis scripts can use it.

**Frame-rate constraint:** `frame_rate_hz == 50` (Wav2Vec2-base ~native rate). Chapter 4 resamples labels per-record to the model's actual output frame count internally.

---

## 3. Data layout on disk

```
data/
├── raw/<CORPUS>/<CORPUS>.zip            # downloaded archives, gitignored
├── unpacked/<CORPUS>/.../EXB|WAV/       # post-unzip (nesting varies by corpus)
├── cut_audio/
│   ├── <CORPUS>/                        # per-instance cuts (chapter 3 input)
│   └── <CORPUS>-Full/                   # normalized full-file 16 kHz mono (chapter 4 input)
├── processed_jsonl/
│   ├── <corpus>_instance.raw.jsonl      # prep output — points at cuts that may not exist yet
│   ├── <corpus>_instance.jsonl          # post-audio_splitter (paths verified)
│   └── <corpus>_frame.raw.jsonl         # prep output — points at full-file WAVs (no splitter step)
└── reports/                             # sniff_dataset outputs
```

**`.raw.jsonl` vs `.jsonl`:** prep notebooks emit `.raw.jsonl`. After `audio_splitter` runs and confirms all instance cuts exist, it writes the `.jsonl` (no `.raw`) version. Frame JSONLs skip this — the normalize-and-copy step is inside the prep notebook itself.

---

## 4. Repository layout

```
slavic-speech-pipeline/
├── README.md
├── BLUEPRINT.md                          ← this file, the activity log
├── requirements.txt
├── .gitignore
│
├── 1_data_prep/
│   ├── README.md
│   ├── utils_dataprep.py                 ← the ONLY utils file (for now)
│   ├── 10_download_data.ipynb            ← chapter 1, step 0
│   ├── 11a_prep_ROG.ipynb                ← chapter 1, step 1, variant a (ROG-Art)
│   ├── 11b_prep_ROG_dia.ipynb            ← step 1, variant b (ROG-Dialog)
│   ├── 11c_prep_ParlaSpeech.ipynb        ← step 1, variant c (future)
│   └── 12_audio_splitter.ipynb           ← step 2
│
├── 2_data_analysis/
│   ├── README.md
│   └── 20_sniff_dataset.ipynb
│
├── 3_instance_models/
│   ├── README.md
│   └── 30_train_instance.ipynb
│
├── 4_frame_models/
│   ├── README.md
│   └── 40_train_frame.ipynb
│
├── 5_analysis/                           ← stub; see chapter 5 below
│   └── README.md
│
└── data/                                 ← gitignored
```

**Naming.** `<chapter><step><variant?>_<verb_noun>.ipynb`. No separator between chapter and step (`30`, not `3_0` or `3.0`). Variant letter `a/b/c` for parallel "same step, different dataset" notebooks. Folders: `<n>_<name>/`.

---

## 5. Notebook conventions

Every notebook in this repo follows the same shape. New chapters mirror existing ones.

1. **Title + overview** (markdown). What this notebook does, inputs, outputs.
2. **Setup** — find `PROJECT_ROOT` via `utils_dataprep`. Add `1_data_prep/` to `sys.path`.
3. **Imports.**
4. **Config dataclass** at the top, named `Config`. `test_mode: bool` field, defaults vary by chapter.
5. **The actual work**, broken into numbered sections with markdown commentary between.

**Standalone for now.** The only project import is `utils_dataprep`. No cross-notebook imports yet; some logic is duplicated across notebooks. That's intentional — we factor only when patterns have stabilized across ≥2 notebooks.

**Test mode** mirrors outputs to `runs/test/`, `models/test/`, or prefixes with `test_` (for prep). Real and test runs never collide.

**Configs.** One dataclass, all knobs. No magic constants buried in cells.

**Literal UTF-8 emojis** if a notebook is built programmatically. Surrogate-pair escapes (`\uXXXX`) break nbformat round-trip.

**Per-epoch logs** for training notebooks: `runs/<run>/<phase>/epoch_logs/<n>/{epoch_summary.json, predictions.json, *.png}`. Best model only saved in phase 2 to `models/<run>/best_model/`.

---

## 6. Chapter-by-chapter plan

### Chapter 1 — Data prep

**Goal:** every dataset → canonical JSONL. EXB-based corpora (ROG-Art, ROG-Dialog) emit **both** instance and frame JSONLs from one EXB pass.

**Notebooks:**
- `10_download_data.ipynb` — fetch corpora, unzip into `data/unpacked/`.
- `11a_prep_ROG.ipynb` — ROG-Art EXB → `rog_instance.raw.jsonl` + `rog_frame.raw.jsonl`. Also normalizes WAVs to 16 kHz mono into `data/cut_audio/ROG-Art-Full/` for the frame model.
- `11b_prep_ROG_dia.ipynb` — same dual-output pattern for ROG-Dialog.
- `11c_prep_ParlaSpeech.ipynb` — placeholder, not yet built.
- `12_audio_splitter.ipynb` — cuts source WAVs into per-instance clips for the instance JSONLs. **Currently ROG-Dialog only**; generalizing to ROG-Art is a follow-up.

**Dual-output prep pattern.** For corpora with both segment-level and continuous annotations, one notebook emits two JSONLs from a single parse pass:
- `<corpus>_instance.raw.jsonl` — one record per segment (e.g. colloq), with whatever scalar labels apply.
- `<corpus>_frame.raw.jsonl` — one record per source WAV, with a 50 Hz label sequence covering the whole file.

Both share the same canonical schema and the same `file_id`-grouped splits (same seed) so no leakage and the splits agree across flavors.

---

### Chapter 2 — Sniff dataset

**Goal:** point at any canonical JSONL → print stats, save a markdown report + plots.

**Notebook:** `20_sniff_dataset.ipynb`. Works for both instance and frame JSONLs.

**Why before training:** every time a new dataset lands, you want to see what's in it before training on it. This is the data-prep sanity check.

---

### Chapter 3 — Instance models

**Goal:** one trainer, both classification and regression for instance-level labels.

**Notebook:** `30_train_instance.ipynb`. Wav2Vec2 + classification head OR regression head, switched by `cfg.task_type`. Two-phase TRAIN→DEV, TRAIN+DEV→TEST.

**v1 targets:** ROG-Art sentiment, ROG-Dialog sentiment, ROG-Art filled_pause_present (binary instance-level sanity check against the frame model).

---

### Chapter 4 — Frame models

**Goal:** frame-level classification on filled pauses (ROG-Art, ROG-Dialog).

**Notebook:** `40_train_frame.ipynb`. Custom `Wav2Vec2ForFrameClassification` head outputs `(B, T, num_labels)`. Token-CE with `ignore_index=-100`. Per-record label alignment via the model's actual CNN output length (works across Wav2Vec2 variants).

**v1 constraints:** `frame_rate_hz == 50` only. `task_type == "classification"` only — frame regression deferred.

**Reference:** parlastress (Croatian filled-pause / primary-stress repo) for the model head and metric patterns.

---

### Chapter 5 — Post-hoc analysis (stub)

**Goal:** operate on saved run directories from chapters 3/4. No model loading.

**Planned notebooks (not yet built):**
- `50_find_best_epoch.ipynb` — scan a run directory, report best epoch by configurable metric. Works for instance and frame runs.
- `51_error_analysis.ipynb` — pull `predictions.json`, surface worst-N records with audio paths + gold/pred for manual inspection.
- `52_compare_runs.ipynb` — side-by-side metrics across multiple run directories.
- `53_frame_event_metrics.ipynb` — boundary / IoU / event-level metrics for frame runs (deferred from chapter 4 training loop, as planned).

**Why a separate chapter:** these don't touch GPUs or models. Keeping them out of chapter 3/4 keeps training notebooks lean.

---

## 7. Execution order

Strict. Don't skip ahead.

**Phase A — foundation**
1. Download data (`10_download_data.ipynb`).
2. Prep ROG-Art (`11a_prep_ROG.ipynb`) → both instance and frame JSONLs.
3. Sniff (`20_sniff_dataset.ipynb`) on the frame JSONL to confirm signal.

**Phase B — frame model end-to-end (the headline v1 deliverable)**
4. Train frame (`40_train_frame.ipynb`) on ROG-Art filled pause. End-to-end pipeline check.
5. `50_find_best_epoch.ipynb` (chapter 5) — closes the training loop.

**Phase C — second corpus**
6. Prep ROG-Dialog (`11b_prep_ROG_dia.ipynb`) — dual output.
7. Generalize `12_audio_splitter.ipynb` to ROG-Art.
8. Re-train chapter 4 on combined or per-corpus frame data.

**Phase D — instance models**
9. Train instance (`30_train_instance.ipynb`) on ROG-Art filled_pause_present (sanity vs frame).
10. Train instance on ROG sentiment.

**Phase E — final polish**
11. Chapter 5 remaining notebooks.
12. Refactor stable notebooks → `main.py` + `utils_<chapter>.py`. Ceremonial step before shipping.

---

## 8. Not doing yet (FUTURE.md material)

ASR. Multi-task heads. Hyperparam search. Distributed training. Model serving. Real-time inference. ParlaSpeech prep. Frame-level regression. Cross-lingual transfer. GOS primary stress.

---

## 9. Sanity checklist for v1

- [ ] A new person clones the repo, runs phase A + B notebooks in order, only edits the top Config cell, gets a working frame model.
- [ ] Every chapter folder has a README explaining run order.
- [ ] Adding a new corpus = one new `11<x>_prep_<CORPUS>.ipynb`.
- [ ] Adding a new target label = change `label_key` (and `label_order` if classification) in chapter 3/4 Config.
- [ ] Chapter 4 trains on ROG-Art filled pause with reasonable frame-F1 (compare to parlastress Croatian).
- [ ] Notebooks all run cleanly top-to-bottom in test mode.
- [ ] Final ceremonial pass: refactor to `main.py` + `utils_<chapter>.py`.

---

## 10. Activity log

Mark items as ✅ when done and working end-to-end. When something gets checked, ask Claude to update this section.

**Done**
- ✅ `utils_dataprep.py` — JSONL I/O, validation, splits, instance ID
- ✅ `10_download_data.ipynb`
- ✅ `11b_prep_ROG_dia.ipynb` — ROG-Dialog instance JSONL (frame output pending re-pass)
- ✅ `12_audio_splitter.ipynb` — ROG-Dialog only
- ✅ `20_sniff_dataset.ipynb`
- ✅ `30_train_instance.ipynb`
- ✅ `40_train_frame.ipynb` — chapter 4 trainer, verified end-to-end on synthetic fixture

**In progress**
- 🔄 `11a_prep_ROG.ipynb` — rebuilt with dual output (instance + frame), verified on synthetic fixture, needs run on real ROG-Art

**Next up**
- [ ] Run `11a_prep_ROG.ipynb` on real ROG-Art → run `40_train_frame.ipynb` against the resulting JSONL end-to-end
- [ ] `11b_prep_ROG_dia.ipynb` — add dual-output (frame JSONL for filled pauses)
- [ ] `12_audio_splitter.ipynb` — generalize to ROG-Art
- [ ] `50_find_best_epoch.ipynb`

---

## 11. Notes for the conversation log

Stuff worth remembering between sessions but not worth surfacing in the main flow:

- Wav2Vec2-base CNN stride = 320 samples @ 16 kHz, *nominal* 50 Hz. Actual output is ~49 Hz (kernel-vs-stride boundary loss in first conv). Chapter 4 handles this with per-record alignment via the model's `_get_feat_extract_output_lengths` (replicated as pure config arithmetic).
- Transformers 5.x requires `label2id` keys to be **strings** when passing to `from_pretrained`. Internal dicts can stay int-keyed; stringify only at the boundary.
- Transformers 5.x renamed `no_cuda=` → `use_cpu=` in `TrainingArguments`.
- Use `soundfile.read` not `torchaudio.load` — recent torchaudio requires `torchcodec`.
- HF Trainer requires `accelerate>=1.1.0` in transformers 5.x.
- ROG.zip unpacks into a triple-nested `ROG/ROG/ROG/ROG-Art/` tree. ROG-Art.wav.zip unpacks into `ROG-Art.wav/ROG/ROG-Art/WAV/`.
- ROG-Art source WAVs are 44.1 kHz mono. Resampled once in `11a_prep_ROG.ipynb` to 16 kHz mono into `data/cut_audio/ROG-Art-Full/`.
- ROG `vocalDisfluency` tier values seen: `filledPause`, `silentPause`, `lengthening`. v1 targets `filledPause` only; the prep notebook is configurable to take others.
- Frame label union across speakers for multi-speaker dialog files (frame[t]=1 if any speaker has a matching event at time t). Worth revisiting once we see real ROG-Dialog frame distributions.
