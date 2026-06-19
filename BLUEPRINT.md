# Speech ML Pipeline — Project Blueprint

A modular, plug-and-play pipeline for fine-tuning Wav2Vec2 models on Slavic speech corpora. Instance- and frame-level classification **and regression**, switched by a config flag.

**Mission for v1:** a clean **instance → frame ladder** on Slavic speech, secured one rung at a time, culminating in a **primary-stress frame model** (feed one word, the model marks which frames carry the primary stress). Clone the repo → run the notebooks in order → a working model at each rung.

> **Active path:** the working corpus is **ParlaSpeech** (HR first — it carries the `filled_pauses`, `words`, and stress/align tiers we need; RS/PL/CZ slot in via the same prep), and the north star is **primary stress**. On top of the full corpora, two **ParlaSpeech-HR benchmark** bundles (v1 and v3) provide pre-built, speaker-disjoint splits for gender / speaker-id / power-status / age (and v3: orientation). The benchmark archives are pre-publication — not yet on CLARIN — so they ship out-of-band into `data/benchmarking/`. ROG-Art/ROG-Dialog remain supported corpora and their prep notebooks still work, but the live path is the ParlaSpeech ladder below.

### The ladder (secure the instance pipeline before the frame pipeline)

1. **gender** — utterance instance classification. *Proof of work / pipeline demo* (audio gender is near-trivial; if the model can't learn it, the wiring is broken). ✅ **secured**
2. **filled-pause presence** — utterance instance classification (also `filled_pause_count`). The first real target. ✅ pipeline ready
3. **FP type** — per-FP-event instance classification (vowel / vowel+nasal / nasal / other / NA). *Needs the annotator deliverable.* `cut=True` (event clips).
4. **sentiment / age** — utterance instance **regression** (`sentiment_logit`; `speaker_age`). ✅ **secured** (age demo run)
5. **filled-pause frames** — frame classification: per-frame FP / not-FP over the whole utterance. ✅ pipeline ready (`41`, demo-verified)
6. **primary stress frames** — frame classification, **word as the instance**. **HR/RS only** (`primary_stress` + `words_align`, via 11c `word_frame`). ← north star
7. **frame regression** — same machinery, scalar-per-frame target. **No data yet** — `42` to be built as a completeness twin of `41`.

---

## 1. Guiding principles

1. **Modular, not monolith.** Each stage is a self-contained notebook connected via canonical JSONL.
2. **One job per notebook.**
3. **Notebook-first, py runners last.** Standalone notebooks are the production artifact until a chapter is secured. Then the **next-stage lift** turns each secured chapter into four artifacts: a frozen **legacy notebook** (`legacy/`, never edited again), a shared **`utils_<chapter>.py`**, a **light tutorial notebook** (per-cell imports from utils, prose-guided), and a **py runner** — all driven by one **`config.json`**. Chapter 3 has been lifted; chapter 4 is next. (Genuinely shared cross-prep helpers were the early exception — see §5.)
4. **Task type is a parameter.** Classification and regression share the pipeline; one config flag switches. Instance regression is live (sentiment, age); frame regression is the only deferred flavor.
5. **Dataset is a parameter.** Each corpus has one prep notebook; downstream is dataset-agnostic.
6. **Config dataclass at the top of every notebook.** No buried constants.
7. **Aggressive markdown.** Each chapter has a README. Each notebook uses a real heading hierarchy (grouped `#` sections, nested `##` subsections) — not a flat wall of identical headers.
8. **QoL only where it earns its keep.** Per-epoch logging, persistent audio index, stage-timing harness — yes. Resume training, hyperparam search, distributed — no.

**Three run tiers** (training notebooks): a single `RUN_MODE` knob (`"test"` | `"demo"` | `"full"`) with a `MODES` dict + `apply_mode`/`cap_split` applying each tier's overrides — **test** (tiny model, a couple dozen records — proves plumbing, no real result), **demo** (real model, capped data, ~1–2 h, tangible number), **full** (caps off, whole corpus). `cap_split` caps train/dev/test identically (the old `DEMO_*` pattern left TEST uncapped — fixed). Base `num_epochs=3`; test/demo override to 1/2. The run-mode cell is byte-identical across 31/32/41.

---

## 2. Canonical data format

One JSONL line = one record. Same schema across instance and frame flavors; only `labels` differs in shape.

**One file per instance-shape, not per task.** Files are split by **(unit × cut × scalar-vs-frame)** — i.e. by *what one record/instance is and what audio it points at* — **not** by which label you train on. Multiple label keys living in one file is encouraged; the trainer picks one via `label_key`. Same label + same instances ⇒ same file (redundant to split); different instances (different audio/cuts) ⇒ different file (necessary). This is the prep-side mirror of the `TARGETS` registry in chapter 3.

So gender, filled-pause presence/count, sentiment, and age all live in **one** `utterance_instance` file (same whole-utterance audio, same splits). FP-type (event clips) and primary stress (word instances) get their own handling because the instances differ.

**Instance record:**
```json
{
  "instance_id": "ParlaMint-HR_2019-06-20-0.u55735_953-975",
  "dataset": "ParlaSpeech-HR",
  "file_id": "<youtube-id>",
  "audio_path": "data/cut_audio/ParlaSpeech-HR/<youtube-id>/<file>.wav",
  "split": "train",
  "speaker": "JandrokovicGordan",
  "text": "To odmah da raščistimo.",
  "labels": {
    "speaker_gender": "M",
    "speaker_age": 54,
    "filled_pause_present": 1,
    "filled_pause_count": 2,
    "sentiment_logit": 3.428,
    "sentiment_6": "Neutral Positive"
  },
  "metadata": {"source_audio": "<youtube-id>/<file>.flac", "audio_length": 2.04, "words": [...], "speaker_info": {...}}
}
```

**Frame record** (one per source WAV, full duration):
```json
{
  "instance_id": "...",
  "dataset": "ParlaSpeech-HR",
  "file_id": "<youtube-id>",
  "audio_path": "data/cut_audio/ParlaSpeech-HR/<youtube-id>/<file>.wav",
  "split": "train",
  "frame_rate_hz": 50,
  "labels": {"filled_pause": [0, 0, 1, 1, 0, ...]},
  "metadata": {"audio_length": 2.04, "words": [...]}
}
```

**Rules.**
- `instance_id` globally unique. For ParlaSpeech the source corpus `id` is already unique and is used directly; the `{dataset}_{file_id}_{speaker?}_{start}_{end}` form (via `make_instance_id`) is the fallback for corpora without one.
- `audio_path` is **project-relative**, points to a **16 kHz mono WAV**. Instance whole-utterance: per-utterance WAV. Sliced recipes: per-segment cut. Frame: normalized full-file copy.
- `label` values may be `null` when a tier is unavailable for that record (e.g. FP inference failed, gender `"-"`). The trainer drops `null` per-task, so one missing tier never costs the others.
- `split` ∈ `{train, dev, test}`, assigned by prep. **Grouped by `speaker` for ParlaSpeech** (no speaker leaks across splits — essential for gender), by `file_id` for the EXB corpora.
- `labels` is a dict; trainer picks one key. `metadata` is free-form; keep `words` (needed for word-level recipes), keep full `speaker_info`. The `_align` tiers (`words_align`, `chars_align`) are bulky and HR/RS-only — kept commented in prep, **enable on demand** (primary stress needs `words_align`).

**No `.raw.jsonl` step.** Prep decides the cuts, writes timestamps (rounded to 3 dp), and cuts/converts audio inline via `utils_audio_splitter` (see §5). One JSONL per recipe, paths already pointing at real WAVs.

**Frame-rate constraint:** `frame_rate_hz == 50`. Chapter 4 resamples labels per-record to the model's actual output frame count internally.

---

## 3. Data layout on disk

```
data/
├── raw/<CORPUS>/                       # downloaded archives, gitignored
├── unpacked/<CORPUS>/.../              # post-unpack (nesting varies wildly by corpus)
├── cut_audio/
│   ├── ParlaSpeech-<LANG>/<hash>/*.wav # per-utterance 16 kHz mono (clean layout)
│   └── <CORPUS>-Full/                  # normalized full-file 16 kHz mono (frame input)
├── benchmarking/
│   ├── ParlaSpeech-HR-benchmark-v1/
│   │   ├── ParlaSpeech-HR-benchmark-v1.jsonl         # source benchmark JSONL
│   │   └── audio/<hash>/<hash>_<start>-<end>.wav     # pre-cut 16 kHz mono WAVs
│   └── ParlaSpeech-HR-benchmark-v3/
│       ├── ParlaSpeech-HR-benchmark-v3.jsonl
│       └── audio/<hash>/<hash>_<start>-<end>.wav
├── processed_jsonl/
│   ├── parlaspeech_<lang>_utterance_instance.jsonl   # one recipe = one file
│   ├── parlaspeech_<lang>_utterance_frame.jsonl
│   ├── parlaspeech_<lang>_word_frame.jsonl
│   ├── parlaspeech_<lang>_audio_index.json           # persistent {basename → path} scan
│   ├── parlaspeech_hr_bench_v1_<task>.jsonl          # 11e output, one file per benchmark task
│   └── parlaspeech_hr_bench_v3_<task>.jsonl          # 11d output, one file per benchmark task
└── reports/                            # sniff_dataset outputs
```

**Benchmark layout.** The two ParlaSpeech-HR benchmark bundles drop in pre-built under `data/benchmarking/`: one JSONL with per-task splits + an `audio/` tree of 16 kHz mono WAVs already keyed by the YouTube hash. The benchmark archives are not on CLARIN yet — they ship out-of-band. `11d`/`11e` parse them into one canonical pipeline JSONL per task (the benchmark's splits are per-task, so the same utterance can be gender/train and age/dev — no shared top-level `split` is possible).

**Audio-source resolution.** Unpacked ParlaSpeech audio is nested under `*.part*/` trees that differ by corpus (HR/RS: `partX/{hash}/`; CZ: `partX/audio/psp/Y/M/D/`) and the nesting is **not** reproducible from the JSONL `audio` field. `utils_audio_splitter` solves this with one recursive scan keyed by **basename** (the filename `{hash}_{start}-{end}.flac` is unique per corpus), cached to `..._audio_index.json` so re-runs skip the scan. After prep, downstream only ever touches the clean `cut_audio/` WAVs.

---

## 4. Repository layout

```
slavic-speech-pipeline/
├── README.md
├── BLUEPRINT.md                          ← this file, the activity log
│
├── 0_env_setup/                          ← CPU and CUDA env install scripts + pinned requirements
│   ├── ENV_SETUP.md
│   ├── requirements_cpu.txt
│   ├── requirements_cuda.txt
│   ├── setup_env_cpu.sh
│   └── setup_env_cuda.sh
│
├── 1_data_prep/
│   ├── README.md
│   ├── utils_dataprep.py                          ← JSONL I/O, schema, splits, ids, audio helpers
│   ├── utils_audio_splitter.py                    ← shared cutter/converter (resolver-driven, multicore)
│   ├── 10_download_data.ipynb
│   ├── 11a_prep_ROG-art.ipynb                     ← ROG-Art (EXB, dual output)
│   ├── 11b_prep_ROG-dia.ipynb                     ← ROG-Dialog
│   ├── 11c_prep_parlaspeech.ipynb                 ← ParlaSpeech (recipe registry; built)
│   ├── 11d_prep_parlaspeech_benchmark_v3.ipynb    ← ParlaSpeech-HR benchmark v3 → one JSONL per task
│   └── 11e_prep_parlaspeech_benchmark_v1.ipynb    ← ParlaSpeech-HR benchmark v1 → one JSONL per task
│
├── 2_data_analysis/   └── 20_sniff_dataset.ipynb   (semi-stub; full overhaul pending — see §6)
├── 3_instance_models/                             ← next-stage lift complete
│   ├── legacy/                                    ← frozen standalone twins (pre-lift)
│   │   ├── 31_train_instance_classification.ipynb
│   │   └── 32_train_instance_regression.ipynb
│   ├── utils_instance_train.py                    ← shared engine + TARGETS registry
│   ├── config.json                                ← all user-facing knobs
│   ├── 31_train_instance_classification.ipynb     ← light tutorial twin (gender, FP presence/count, hr_bench_v1/v3)
│   ├── 32_train_instance_regression.ipynb         ← light tutorial twin (sentiment_logit, age, hr_bench_v3 age/orientation)
│   ├── run_31_classification.py                   ← py runner
│   └── run_32_regression.py                       ← py runner
├── 4_frame_models/
│   ├── 41_train_frame_classification.ipynb        ← FP frames / primary-stress frames
│   └── 42_train_frame_regression.ipynb            ← planned, completeness twin (no data yet)
├── 5_tg_minter/
│   ├── 51_tg_minter.ipynb                         ← ParlaSpeech v3 TextGrids → annotation-ready bundles
│   └── 52_annotation_stats.ipynb                  ← read annotated TGs back, surface stats
└── data/                                          ← gitignored
```

> `30_train_instance.ipynb` was **split into the twins `31`/`32`** — one clean pile per task. They share a byte-identical run engine (see §5); only task-specific cells differ. The chapter-3 next-stage lift extracted that engine into `utils_instance_train.py`; the standalone pre-lift twins are frozen in `legacy/`.

**Three utils files now.** `utils_dataprep.py`, `utils_audio_splitter.py`, and `utils_instance_train.py`. The splitter is genuinely shared by every prep notebook (one cutter, many corpora), so it lived as a module from the start — the early sanctioned exception. `utils_instance_train.py` is the chapter-3 lift; **next up is chapter 4** — `utils_frame_train.py` once 41/42 are secured.

**Naming.** `<chapter><step><variant?>_<verb_noun>.ipynb`. No separator between chapter and step (`31`, not `3_1`). Variant letter `a/b/c` for "same step, different dataset". Folders `<n>_<name>/`.

---

## 5. Notebook & module conventions

Every notebook follows the same shape: title/overview → setup (`PROJECT_ROOT` via `utils_dataprep`) → imports → `Config` dataclass → work sections grouped under a real heading hierarchy.

**Standalone-ish.** Project imports are limited to `utils_dataprep` and `utils_audio_splitter`. No cross-*notebook* imports. Logic duplicated across notebooks is factored only after it stabilizes across ≥2 of them.

**Twin notebooks.** `31_train_instance_classification` and `32_train_instance_regression` are deliberate **twins**: post-lift they import the same `utils_instance_train.py` engine — `run_phase`, the two-phase TRAIN→DEV / TRAIN+DEV→TEST loop, run-directory setup, the run-mode cell, and the stage-timing harness — and only task-specific cells differ (label handling, metrics, per-epoch artifacts, model factory). `41` still carries its own inline engine (chapter 4 has not been lifted yet); its shared cells stay byte-identical with the pre-lift twins frozen in `legacy/` until `utils_frame_train.py` lands.

**Stage-timing harness.** Both training notebooks carry an identical tiny timer (`STAGE_TIMES` + `mark()` + `print_stage_breakdown`, stdlib-only) stamping `literal start → data prep → model prep → end phase 1 → end phase 2 → end script`, plus a rough ETA after the demo-cap counts that recalibrates from phase 1's real rate. Partial-run safe. Also lifts to utils.

**Recipe registry (prep).** Each prep notebook defines a `RECIPES` dict — one entry per instance-shape it can emit — with `unit` (`utterance`/`word`/`event`), `level` (`instance`/`frame`), `cut` (bool), `requires` (corpus-specific tiers that gate the recipe), and `out` (path). `cut=True` triggers `utils_audio_splitter` to slice; `cut=False` whole-file converts. This mirrors the `TARGETS` registry in chapter 3 (prep has recipes, the trainer has targets).

**The splitter (`utils_audio_splitter`).** Dumb by design: prep writes the cuts, the splitter executes them. Corpus knowledge lives in a **resolver** (`record → SourceRef | None`): `make_stem_scan_resolver` (session-WAV corpora like ROG), `make_record_path_resolver` (direct path), `make_flac_index_resolver` (ParlaSpeech basename index + persistent cache). `cut_dataset(..., num_workers=0)` = sequential + LRU cache (best for big shared sources); `num_workers>0` = **process pool** (`parallel_backend="process"`, the default) for true multi-core on many small independent FLACs, with a `"thread"` fallback for fork-unfriendly platforms. Workers receive only `(instance_id, audio_path, SourceRef)` — the big resolver index stays in the parent.

**Test mode** (`RUN_MODE="test"`) mirrors outputs under `*/test/` or `data/test_processed_jsonl/`. Real and test runs never collide.

**Literal UTF-8 emojis** when a notebook is built programmatically (`ensure_ascii=False`); surrogate escapes break nbformat round-trip.

---

## 6. Chapter-by-chapter plan

### Chapter 0 — Env setup
**Goal:** one-shot install of either the CPU or the CUDA Python environment used by every downstream notebook.

- `setup_env_cpu.sh` — creates the `ssp` mamba env (CPU / dev / data prep).
- `setup_env_cuda.sh` — creates the `ssp-cuda` mamba env (GPU training; CUDA 12.4).
- `requirements_cpu.txt` / `requirements_cuda.txt` — pinned dependencies per env.
- `ENV_SETUP.md` — prerequisites (miniforge / micromamba), activation, GPU verification.

### Chapter 1 — Data prep
**Goal:** every dataset → canonical JSONL(s), one file per instance-shape.

- `10_download_data.ipynb` — fetch + unpack.
- `11a_prep_ROG-art.ipynb` / `11b_prep_ROG-dia.ipynb` — EXB corpora, dual output.
- `11c_prep_parlaspeech.ipynb` — recipe registry; emits `utterance_instance` (gender, age, filled_pause_present, filled_pause_count, sentiment_logit, sentiment_6), `utterance_frame` (50 Hz FP sequence), and `word_frame` (primary stress, HR/RS — carries the utterance `audio_path` + `start_t`/`end_t` word bounds; the trainer slices in memory, no word WAVs on disk). Converts FLAC→WAV via the splitter (whole-file, multicore process pool, cached index). Per-language loop with stage timer. Still stubbed: `event_instance` (FP-type, needs annotator).
- `11d_prep_parlaspeech_benchmark_v3.ipynb` — parse the pre-built ParlaSpeech-HR benchmark v3 (under `data/benchmarking/`) into one JSONL per task (`gender` / `speaker_id` / `power_status` / `age` / `orientation`). Splits come from the benchmark's per-task `benchmark` key (no `assign_splits`); audio is already 16 kHz mono — `audio_path` points straight at it. v3's `age` and `orientation` are regression; the other three are classification.
- `11e_prep_parlaspeech_benchmark_v1.ipynb` — sibling to `11d` for the ParlaSpeech-HR benchmark v1. Same per-task-file layout (`gender` / `speaker_id` / `power_status` / `age`), all four classification (v1's age is a `young`/`old` group, not a continuous value). Labels are normalized to v3's casing where they overlap so a single chapter-3 target family spans both benchmarks.

### Chapter 2 — Dataset analysis (semi-stub, overhaul pending)
`20_sniff_dataset.ipynb` — current state: point at any canonical JSONL → stats + markdown report. Works for instance and frame. Kept as a semi-stub until the planned overhaul: full label distributions, audio-duration histograms, speaker/split summaries. The replacement chapter will not carry the "sniff" framing.

### Chapter 3 — Instance models (twins, lifted)
Wav2Vec2-base + a task head, two-phase (TRAIN→DEV, then TRAIN+DEV→TEST). Engine in `utils_instance_train.py`, knobs in `config.json`; the notebooks are tutorial twins.

- `31_train_instance_classification.ipynb` — `AutoModelForAudioClassification`. Targets: ParlaSpeech `speaker_gender`, `filled_pause_present` / `_count`; ROG sentiment / FP; ParlaSpeech-HR benchmark v1 (`hr_bench_v1_gender` / `_speaker_id` / `_power_status` / `_age`) and v3 (`hr_bench_v3_gender` / `_speaker_id` / `_power_status`). Confusion matrices (stacked counts + row-norm %).
- `32_train_instance_regression.ipynb` — `Wav2Vec2ForRegression` (masked mean-pool + `Linear(1)`). Targets: `sentiment_logit`, `speaker_age`, plus ParlaSpeech-HR benchmark v3 (`hr_bench_v3_age`, `hr_bench_v3_orientation`). Scatter + distribution plots. **Target normalization** (`cfg.normalize="zscore"`, fit TRAIN-only, inverted before metrics + saved predictions) — regression-only.

**Py runners** (`run_31_classification.py` / `run_32_regression.py`) drive the same engine without prompts — `--mode`, `--target`, `--use_gpu`. This is the tool for full-corpus runs.

**Shared engine details:** input-gated GPU guard for notebooks / `--use_gpu` flag for runners (**reserved GPU from `shared.reserved_gpu`** in `config.json`, never touch another), configurable CPU workers, `attention_mask` returned by the feature extractor + passed by the collator (so pooling excludes padding), GPU flush between phases (`del`/`gc`/`empty_cache`), stage timer + ETA, and an inference spot-check (5 random TEST examples, real units / class + hit-marker). Predictions carry full provenance (`file_id`, `start_t`/`end_t` from `metadata.audio_*`) and a populated `pred_raw`. The `TARGETS` registry supports optional per-target constraint overrides (`TARGET_CONSTRAINT_FIELDS`, currently `max_duration_s`) so a preset can self-declare benchmark caps.

### Chapter 4 — Frame models
`41_train_frame_classification.ipynb` — built to engine parity with the chapter-3 twins (run-mode cell, timer, attention-mask, GPU flush, spot-check). Custom frame head → `(B, T, num_labels)`, token-CE with `ignore_index=-100`, per-record label alignment to the model's actual CNN output length. `frame_rate_hz == 50` only. **Task-keyed `TARGETS`** (`parlaspeech_fp_frames`, `parlaspeech_primary_stress_frames`) with `jsonl_template` + supported langs; `cfg.langs` is an orthogonal language knob (`()` = all available). `save_predictions_json` stores per-frame `prob_pos` (softmax positive-class probability) for downstream QC thresholding. Reference: parlastress (Croatian FP/stress). Two v1 frame tasks, in order:

- **FP frames (rung 5)** — binary per-frame **FP / not-FP** over the whole utterance. ParlaSpeech (`utterance_frame` from 11c). Demo-verified in `41`; full run waits on the py runners.
- **Primary-stress frames (rung 6, HR/RS) — the doosey.** The **word is the instance**: 11c's `word_frame` recipe carries the utterance `audio_path` + `start_t`/`end_t` word bounds, and `41` slices audio + the 50 Hz label sequence **in memory** at preprocess time — no word WAVs on disk.

Frame regression (rung 7): `42_train_frame_regression.ipynb` will be built as the completeness twin of `41` (same engine, regression head), with a prominent note that **no annotated data exists yet** for a continuous per-frame target — code for future work, not runnable on real data.

### Chapter 5 — TG minter
**Goal:** turn ParlaSpeech v3 utterances into TextGrids an annotator can edit, then read the annotated TGs back into the canonical JSONL space.

- `51_tg_minter.ipynb` — slices ParlaSpeech v3 records into annotation-ready TextGrid bundles (audio + TG, one per utterance), seeded with the existing tiers.
- `52_annotation_stats.ipynb` — reads annotated TGs back and surfaces stats for downstream consumers.

---

## 7. Execution order

Follow the ladder; secure each rung before the next, and the whole **instance** pipeline before the **frame** pipeline.

**Stage A — foundation** ✅
1. Download + unpack (`10`).
2. Prep ParlaSpeech-HR (`11c`) → `utterance_instance` + `utterance_frame` (+ `word_frame` for HR/RS).
3. Sniff (`20`) the instance JSONL.

**Stage B — instance pipeline (rungs 1–4)** ✅ *secured*
4. `31` gender (demo) → filled-pause presence/count; `32` sentiment / age (regression).
5. (rung 3 FP-type slots in once the annotator deliverable lands.)

**Stage C — frame pipeline (rungs 5–6)** ✅ *engine secured (demo)*
6. `41` FP frames (ParlaSpeech) — built, demo-verified.
7. `41` primary-stress frames (HR/RS) — the north star; 11c `word_frame` recipe confirmed on real HR.

**Stage D — benchmarks** ← *active*
8. ParlaSpeech-HR benchmark v1 / v3 prep (`11d`/`11e`) → `31`/`32` targets `hr_bench_v1_*` and `hr_bench_v3_*`. Numbers come back here.

**Stage E — breadth**
9. Other languages (`cfg.langs`); ROG corpora; combined runs. **Full training runs** happen here — via the py runners, not notebooks (kernel memory limits under full load).

**Next stage — abstraction & polish** (chapter 3 ✅ done, chapter 4 next)
10. Per secured chapter: final demo run → freeze standalone notebooks into `legacy/` → lift engine to `utils_<chapter>.py` + `config.json` → light tutorial notebooks (per-cell imports) → py runners (`--mode`, `--config`, `--use_gpu` overriding the interactive GPU guard). Chapter 3 (`utils_instance_train.py`) is complete; chapter 4 (`utils_frame_train.py`) is next, with `41` lifted and `42` built as its completeness twin.
11. Chapter 5 (TG minter) follow-ups as the annotator deliverable matures; chapter 2 overhaul (distributions/durations/speaker summaries) lands alongside.

---

## 8. Not doing yet (FUTURE.md material)

ASR. Multi-task heads. Hyperparam search. Distributed training. Model serving. Real-time inference. **Frame-level regression on real data** (rung 7 — `42` gets built as a completeness twin, but no annotated continuous frame target exists yet). Cross-lingual transfer. **GOS** (the GOS-specific stress target is dropped — primary stress comes from ParlaSpeech HR/RS instead).

---

## 9. Sanity checklist for v1

- [ ] Clone → run phase A + B notebooks in order, only edit the top `Config`, get a working instance model (start with gender). ✅
- [ ] Every chapter folder has a README.
- [ ] Adding a corpus = one new `11<x>_prep_<CORPUS>.ipynb` + (if needed) a resolver.
- [ ] Adding a target = change `target` (the `TARGETS` preset) in chapter 3/4.
- [ ] Frame model trains primary stress (HR/RS) with reasonable frame-F1 (compare to parlastress).
- [ ] Notebooks run cleanly top-to-bottom in test mode.
- [ ] Final ceremonial pass: refactor to `main.py` + `utils_<chapter>.py`.

---

## 10. Activity log

Mark items ✅ when done and working **end-to-end on real data**. Ask Claude to update this when something gets checked.

**Done (real data)**
- ✅ `utils_dataprep.py` — JSONL I/O, validation, splits, instance id
- ✅ `utils_audio_splitter.py` — resolver-driven cutter/converter (stem-scan, record-path, FLAC basename-index + cache); **process-pool multicore** (`parallel_backend="process"`, thread fallback); confirmed via 11c on real HR and RS
- ✅ `10_download_data.ipynb`
- ✅ `11b_prep_ROG-dia.ipynb` — ROG-Dialog instance JSONL (frame output pending re-pass)
- ✅ `20_sniff_dataset.ipynb`
- ✅ `11c_prep_parlaspeech.ipynb` — recipe registry, `utterance_instance` + `utterance_frame` + **`word_frame`** (utterance path + word bounds, in-memory slicing downstream), speaker-grouped splits, multicore convert + stage timer; **confirmed on real HR and RS** (HR first)
- ✅ `11d_prep_parlaspeech_benchmark_v3.ipynb` — ParlaSpeech-HR benchmark v3 → one JSONL per task (gender/speaker_id/power_status/age/orientation); splits baked in by the benchmark, audio already 16 kHz mono
- ✅ `11e_prep_parlaspeech_benchmark_v1.ipynb` — ParlaSpeech-HR benchmark v1 → one JSONL per task (all classification; v1's age is young/old), label casing normalized to v3
- ✅ `31_train_instance_classification.ipynb` — gender on real ParlaSpeech-HR; attention-mask pooling, GPU flush, timer/ETA, spot-check, populated `pred_raw`; lifted to engine import
- ✅ `32_train_instance_regression.ipynb` — age on real data; train-only z-score normalization (real-unit metrics), attention-mask pooling fix, GPU flush, timer/ETA, spot-check; lifted to engine import
- ✅ `41_train_frame_classification.ipynb` — frame head, per-record label alignment, task-keyed `TARGETS` + `cfg.langs`, `prob_pos` in saved predictions; engine parity with the twins; demo-verified
- ✅ **Run-mode refactor (31/32/41)** — `RUN_MODE`/`MODES`/`apply_mode`/`cap_split` replaces `test_mode` + `DEMO_*`; caps now apply to train/dev/test identically (fixed TEST-leak in demo runs); run-mode cell byte-identical across all three (md5-confirmed)
- ✅ **Chapter-3 lift complete** — `utils_instance_train.py` (engine + `TARGETS` + `TARGET_CONSTRAINT_FIELDS`), `config.json` (all user-facing knobs), light tutorial notebooks `31`/`32`, py runners `run_31_classification.py` / `run_32_regression.py` (`--mode`, `--target`, `--use_gpu`); pre-lift standalone twins frozen in `3_instance_models/legacy/`
- ✅ `0_env_setup/` — CPU and CUDA env install scripts + pinned requirements + `ENV_SETUP.md`
- ✅ `5_tg_minter/` — `51_tg_minter.ipynb` (ParlaSpeech v3 → annotation-ready TextGrids) and `52_annotation_stats.ipynb` (read annotated TGs back)

**Verified on fixture, pending real-data run**
- 🧪 `11a_prep_ROG-art.ipynb` — dual output, fixture-verified, needs run on real ROG-Art

**Next up**
- [ ] Chapter 4 lift: extract the `41` engine into `utils_frame_train.py` + `config.json` + `run_41_frame_classification.py`; freeze pre-lift `41` into `4_frame_models/legacy/`; build `42_train_frame_regression.ipynb` as the completeness twin (no annotated continuous frame target exists yet — code for future work)
- [ ] Chapter 2 overhaul — full label distributions, audio-duration histograms, speaker/split summaries; drop the "sniff" framing
- [ ] Public release of the ParlaSpeech-HR benchmark archives (currently dropped into `data/benchmarking/` out-of-band, pre-publication)
- [ ] Pending decision: small "demo ParlaSpeech" dataset in the `10` download registry as the default for demo runs

---

## 11. Notes for the conversation log

- ParlaSpeech audio nesting differs by corpus and is **not** derivable from the JSONL `audio` field (HR/RS: `partX/{hash}/`; CZ: `partX/audio/psp/Y/M/D/`). Resolve by **basename** (unique per corpus) via one recursive scan, cached to `..._audio_index.json`. HR is ~1.4M files — the cache matters there; RS ~290k is fine either way.
- ParlaSpeech v3 JSONL tiers (`sentiment`, `words`, `filled_pauses`, `_align`) can be absent or `null` per language; prep treats them all as optional and writes `null` labels rather than crashing. `filled_pauses == null` means FP inference failed (≠ `[]` = none detected).
- ParlaSpeech splits are grouped by **`Speaker_ID`**, not file — gender especially leaks badly otherwise.
- **Primary stress is the north star** (rung 6, HR/RS — `primary_stress` + `words_align`, emitted by 11c's `word_frame` recipe). Word as instance via **in-memory slicing**: `41` loads the utterance WAV + 50 Hz label sequence and slices both by the record's `start_t`/`end_t` word bounds; no word WAVs on disk.
- **Full training runs are deferred to the py runners** — notebooks hit kernel memory limits under full load. Notebooks own test/demo tiers; runners own full runs (tmux, unattended, `--use_gpu` instead of the interactive guard).
- **Splitter parallelism:** `num_workers>0` uses a **process pool** by default — `librosa.resample` is CPU-bound and mostly GIL-held, so threads barely scaled (~1 core, ~1 h on RS); processes give true multicore (target ≤15 min). Thread backend kept for fork-unfriendly platforms. Process backend relies on Linux `fork` (works in Jupyter, no `__main__` guard).
- **Regression target normalization** (`32`): targets are z-scored on **TRAIN-only** mean/std, applied to the training labels, and **inverted before** metrics + saved predictions, so MSE/MAE/scatter read in real units (years, logits). Stats snapshot to `label_normalization.json`. Classification is **not** normalized — `normalize` is a regression-only knob. Catalyst: raw `speaker_age` (magnitudes ~20–90) gave mae≈37 / spearman=nan; normalization + the pooling fix resolved it.
- **Attention-mask pooling** (both twins): `wav2vec2-base` ships `return_attention_mask=False`, so the collator emitted no mask and the model mean-pooled **over padding** → near-constant predictions (the regression collapse; latent in classification too). Fix: `feature_extractor.return_attention_mask = True` at load **and** `return_attention_mask=True` in the collator's `pad`.
- **GPU hygiene:** `run_phase` frees the model + empties the CUDA cache between phases (`del trainer, model; gc.collect(); torch.cuda.empty_cache()`). Orphaned notebook kernels otherwise pile up holding VRAM on GPU 2 — kill stale PIDs by hand (`nvidia-smi` → `kill`) and restart kernels between runs.
- Wav2Vec2-base CNN stride = 320 samples @ 16 kHz, *nominal* 50 Hz, actual ~49 Hz. Chapter 4 handles per-record alignment via the model's feat-extract output length.
- Transformers 5.x: `label2id` keys must be **strings** at the `from_pretrained` boundary; `no_cuda=` → `use_cpu=`. HF Trainer needs `accelerate>=1.1.0`.
- Use `soundfile.read`, not `torchaudio.load` (recent torchaudio wants `torchcodec`).
- Wav2Vec2-base, not XLS-R 300M (XLS-R crashed the CPU box).
- ROG.zip triple-nests as `ROG/ROG/ROG/ROG-Art/`; ROG-Art.wav.zip → `ROG-Art.wav/ROG/ROG-Art/WAV/`. ROG source WAVs are 44.1 kHz mono → resampled to 16 kHz. ROG `vocalDisfluency` tier values: `filledPause`, `silentPause`, `lengthening` (v1 uses `filledPause`).
- Frame label union across speakers for multi-speaker dialog files (frame[t]=1 if any speaker has a matching event). Revisit on real ROG-Dialog frame distributions.