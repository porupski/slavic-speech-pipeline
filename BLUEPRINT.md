# Speech ML Pipeline — Project Blueprint

A modular, plug-and-play pipeline for fine-tuning Wav2Vec2 models on Slavic speech corpora. Instance- and frame-level classification **and regression**, switched by a config flag.

**Mission for v1:** a clean **instance → frame ladder** on Slavic speech, secured one rung at a time, culminating in a **primary-stress frame model** (feed one word, the model marks which frames carry the primary stress). Clone the repo → run the notebooks in order → a working model at each rung.

> **Active path:** the working corpus is **ParlaSpeech** (RS first — it carries the `filled_pauses`, `words`, and stress/align tiers we need), and the north star is **primary stress**. ROG-Art/ROG-Dialog remain supported corpora and their prep notebooks still work, but the live path is the ParlaSpeech ladder below.

### The ladder (secure the instance pipeline before the frame pipeline)

1. **gender** — utterance instance classification. *Proof of work / pipeline demo* (audio gender is near-trivial; if the model can't learn it, the wiring is broken). ✅ **secured**
2. **filled-pause presence** — utterance instance classification (also `filled_pause_count`). The first real target. ✅ pipeline ready
3. **FP type** — per-FP-event instance classification (vowel / vowel+nasal / nasal / other / NA). *Needs the annotator deliverable.* `cut=True` (event clips).
4. **sentiment / age** — utterance instance **regression** (`sentiment_logit`; `speaker_age`). ✅ **secured** (age demo run)
5. **filled-pause frames** — frame classification: per-frame FP / not-FP over the whole utterance. ← **next chapter**
6. **primary stress frames** — frame classification, **word as the instance**. **HR/RS only** (needs `primary_stress` + `words_align`). ← north star
7. **frame regression** — same machinery, scalar-per-frame target. **Deferred** — no suitable continuous frame target identified yet (only FPs available).

---

## 1. Guiding principles

1. **Lego, not monolith.** Each stage is a self-contained notebook connected via canonical JSONL.
2. **One job per notebook.**
3. **Notebook-first, py runners last.** Notebooks are the production artifact for v1. Final ceremonial step before shipping: refactor into `main.py` + `utils_<chapter>.py`. Not before. (Genuinely shared cross-prep helpers are the exception — see §5.)
4. **Task type is a parameter.** Classification and regression share the pipeline; one config flag switches. Instance regression is live (sentiment, age); frame regression is the only deferred flavor.
5. **Dataset is a parameter.** Each corpus has one prep notebook; downstream is dataset-agnostic.
6. **Config dataclass at the top of every notebook.** No buried constants.
7. **Aggressive markdown.** Each chapter has a README. Each notebook uses a real heading hierarchy (grouped `#` sections, nested `##` subsections) — not a flat wall of identical headers.
8. **QoL only where it earns its keep.** Per-epoch logging, persistent audio index, stage-timing harness — yes. Resume training, hyperparam search, distributed — no.

**Three run tiers** (training notebooks): **test mode** (tiny model, a couple dozen records — proves plumbing, no real result), **demo run** (real model, capped data via `DEMO_*`, ~1–2 h, tangible number), **full run** (caps off, whole corpus).

---

## 2. Canonical data format

One JSONL line = one record. Same schema across instance and frame flavors; only `labels` differs in shape.

**One file per instance-shape, not per task.** Files are split by **(unit × cut × scalar-vs-frame)** — i.e. by *what one record/instance is and what audio it points at* — **not** by which label you train on. Multiple label keys living in one file is encouraged; the trainer picks one via `label_key`. Same label + same instances ⇒ same file (redundant to split); different instances (different audio/cuts) ⇒ different file (necessary). This is the prep-side mirror of the `TARGETS` registry in chapter 3.

So gender, filled-pause presence/count, sentiment, and age all live in **one** `utterance_instance` file (same whole-utterance audio, same splits). FP-type (event clips) and primary stress (word instances) get their own handling because the instances differ.

**Instance record:**
```json
{
  "instance_id": "ParlaMint-RS_2019-06-20-0.u55735_953-975",
  "dataset": "ParlaSpeech-RS",
  "file_id": "ouafo6IB8GY",
  "audio_path": "data/cut_audio/ParlaSpeech-RS/ouafo6IB8GY/<file>.wav",
  "split": "train",
  "speaker": "ArsicVeroljub",
  "text": "To odmah da raščistimo.",
  "labels": {
    "speaker_gender": "M",
    "speaker_age": 54,
    "filled_pause_present": 1,
    "filled_pause_count": 2,
    "sentiment_logit": 3.428,
    "sentiment_6": "Neutral Positive"
  },
  "metadata": {"source_audio": "ouafo6IB8GY/<file>.flac", "audio_length": 2.04, "words": [...], "speaker_info": {...}}
}
```

**Frame record** (one per source WAV, full duration):
```json
{
  "instance_id": "...",
  "dataset": "ParlaSpeech-RS",
  "file_id": "ouafo6IB8GY",
  "audio_path": "data/cut_audio/ParlaSpeech-RS/<hash>/<file>.wav",
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
├── processed_jsonl/
│   ├── parlaspeech_<lang>_utterance_instance.jsonl   # one recipe = one file
│   ├── parlaspeech_<lang>_utterance_frame.jsonl
│   └── parlaspeech_<lang>_audio_index.json           # persistent {basename → path} scan
└── reports/                            # sniff_dataset outputs
```

**Audio-source resolution.** Unpacked ParlaSpeech audio is nested under `*.part*/` trees that differ by corpus (HR/RS: `partX/{hash}/`; CZ: `partX/audio/psp/Y/M/D/`) and the nesting is **not** reproducible from the JSONL `audio` field. `utils_audio_splitter` solves this with one recursive scan keyed by **basename** (the filename `{hash}_{start}-{end}.flac` is unique per corpus), cached to `..._audio_index.json` so re-runs skip the scan. After prep, downstream only ever touches the clean `cut_audio/` WAVs.

---

## 4. Repository layout

```
slavic-speech-pipeline/
├── README.md
├── BLUEPRINT.md                          ← this file, the activity log
├── requirements.txt
│
├── 1_data_prep/
│   ├── README.md
│   ├── utils_dataprep.py                 ← JSONL I/O, schema, splits, ids, audio helpers
│   ├── utils_audio_splitter.py           ← shared cutter/converter (resolver-driven, multicore)
│   ├── 10_download_data.ipynb
│   ├── 11a_prep_ROG.ipynb                ← ROG-Art (EXB, dual output)
│   ├── 11b_prep_ROG_dia.ipynb            ← ROG-Dialog
│   ├── 11c_prep_ParlaSpeech.ipynb        ← ParlaSpeech (recipe registry; built)
│   └── 12_audio_splitter.ipynb           ← RETIRED for new preps; standalone re-cut tool only
│
├── 2_data_analysis/   └── 20_sniff_dataset.ipynb
├── 3_instance_models/
│   ├── 31_train_classification.ipynb     ← gender, FP presence/count (twin of 32)
│   └── 32_train_regression.ipynb         ← sentiment_logit, age (twin of 31)
├── 4_frame_models/    └── 40_train_frame.ipynb
├── 5_analysis/        └── README.md  (stub; see §6)
└── data/                                 ← gitignored
```

> `30_train_instance.ipynb` was **split into the twins `31`/`32`** — one clean pile per task. They share a byte-identical run engine (see §5); only task-specific cells differ.

**Two utils files now.** `utils_dataprep.py` and `utils_audio_splitter.py`. The splitter is genuinely shared by every prep notebook (one cutter, many corpora), so it lives as a module from the start — this is the sanctioned exception to "no utils until phase E." No *further* utils files until phase E (the chapter-3 engine is the next planned lift → `utils_instance_train.py`).

**Naming.** `<chapter><step><variant?>_<verb_noun>.ipynb`. No separator between chapter and step (`31`, not `3_1`). Variant letter `a/b/c` for "same step, different dataset". Folders `<n>_<name>/`.

---

## 5. Notebook & module conventions

Every notebook follows the same shape: title/overview → setup (`PROJECT_ROOT` via `utils_dataprep`) → imports → `Config` dataclass → work sections grouped under a real heading hierarchy.

**Standalone-ish.** Project imports are limited to `utils_dataprep` and `utils_audio_splitter`. No cross-*notebook* imports. Logic duplicated across notebooks is factored only after it stabilizes across ≥2 of them.

**Twin notebooks (chapter 3).** `31_train_classification` and `32_train_regression` are deliberate **twins**: the run engine — `run_phase`, the two-phase TRAIN→DEV / TRAIN+DEV→TEST loop, run-directory setup, and the stage-timing harness — is **byte-identical** across the two; only the task-specific cells differ (label handling, metrics, per-epoch artifacts, model factory). Keep them identical. They lift to `utils_instance_train.py` in phase E.

**Stage-timing harness.** Both training notebooks carry an identical tiny timer (`STAGE_TIMES` + `mark()` + `print_stage_breakdown`, stdlib-only) stamping `literal start → data prep → model prep → end phase 1 → end phase 2 → end script`, plus a rough ETA after the demo-cap counts that recalibrates from phase 1's real rate. Partial-run safe. Also lifts to utils.

**Recipe registry (prep).** Each prep notebook defines a `RECIPES` dict — one entry per instance-shape it can emit — with `unit` (`utterance`/`word`/`event`), `level` (`instance`/`frame`), `cut` (bool), `requires` (corpus-specific tiers that gate the recipe), and `out` (path). `cut=True` triggers `utils_audio_splitter` to slice; `cut=False` whole-file converts. This mirrors the `TARGETS` registry in chapter 3 (prep has recipes, the trainer has targets).

**The splitter (`utils_audio_splitter`).** Dumb by design: prep writes the cuts, the splitter executes them. Corpus knowledge lives in a **resolver** (`record → SourceRef | None`): `make_stem_scan_resolver` (session-WAV corpora like ROG), `make_record_path_resolver` (direct path), `make_flac_index_resolver` (ParlaSpeech basename index + persistent cache). `cut_dataset(..., num_workers=0)` = sequential + LRU cache (best for big shared sources); `num_workers>0` = **process pool** (`parallel_backend="process"`, the default) for true multi-core on many small independent FLACs, with a `"thread"` fallback for fork-unfriendly platforms. Workers receive only `(instance_id, audio_path, SourceRef)` — the big resolver index stays in the parent.

**Test mode** mirrors outputs under `*/test/` or `data/test_processed_jsonl/`. Real and test runs never collide.

**Literal UTF-8 emojis** when a notebook is built programmatically (`ensure_ascii=False`); surrogate escapes break nbformat round-trip.

---

## 6. Chapter-by-chapter plan

### Chapter 1 — Data prep
**Goal:** every dataset → canonical JSONL(s), one file per instance-shape.

- `10_download_data.ipynb` — fetch + unpack.
- `11a_prep_ROG.ipynb` / `11b_prep_ROG_dia.ipynb` — EXB corpora, dual output.
- `11c_prep_ParlaSpeech.ipynb` — recipe registry; emits `utterance_instance` (gender, age, filled_pause_present, filled_pause_count, sentiment_logit, sentiment_6) + `utterance_frame` (50 Hz FP sequence). Converts FLAC→WAV via the splitter (whole-file, multicore process pool, cached index). Per-language loop with stage timer. Future recipes stubbed: `event_instance` (FP-type, needs annotator) and `word_frame` (primary stress, HR/RS — needs `words_align` un-commented).
- `12_audio_splitter.ipynb` — retired for new preps (cutting moved into prep via the module); kept as a standalone re-cut tool.

### Chapter 2 — Sniff dataset
`20_sniff_dataset.ipynb` — point at any canonical JSONL → stats + markdown report. Works for instance and frame.

### Chapter 3 — Instance models (twins)
Wav2Vec2-base + a task head, two-phase (TRAIN→DEV, then TRAIN+DEV→TEST). Identical engine across the two notebooks; per-task cells differ.

- `31_train_classification.ipynb` — `AutoModelForAudioClassification`. Targets: ParlaSpeech `speaker_gender` (demo), `filled_pause_present` / `_count`; plus ROG sentiment / FP. Confusion matrices (stacked counts + row-norm %).
- `32_train_regression.ipynb` — `Wav2Vec2ForRegression` (masked mean-pool + `Linear(1)`). Targets: `sentiment_logit`, `speaker_age`. Scatter + distribution plots. **Target normalization** (`cfg.normalize="zscore"`, fit TRAIN-only, inverted before metrics + saved predictions) — regression-only.

**Shared engine details:** input-gated GPU guard (**GPU 2 reserved**, never touch another), configurable CPU workers, `attention_mask` returned by the feature extractor + passed by the collator (so pooling excludes padding), GPU flush between phases (`del`/`gc`/`empty_cache`), stage timer + ETA, and an inference spot-check (5 random TEST examples, real units / class + hit-marker). Predictions carry full provenance (`file_id`, `start_t`/`end_t` from `metadata.audio_*`) and a populated `pred_raw`.

### Chapter 4 — Frame models ← **next**
`40_train_frame.ipynb` — custom frame head → `(B, T, num_labels)`, token-CE with `ignore_index=-100`, per-record label alignment to the model's actual CNN output length. `frame_rate_hz == 50` only. Reference: parlastress (Croatian FP/stress). Two v1 frame tasks, in order:

- **FP frames (rung 5)** — binary per-frame **FP / not-FP** over the whole utterance. ParlaSpeech-RS (`utterance_frame` already emitted by 11c), then ROG. The straightforward first frame target; reuses chapter-3's engine spirit (two-phase, timer, spot-check) adapted to a per-frame head + metrics.
- **Primary-stress frames (rung 6, HR/RS) — the doosey.** The **word is the instance**, and we frame over the word to mark which frames carry the primary stress. **Design lean: don't cut word WAVs to disk.** Load the per-utterance WAV once and slice the loaded array **and** the 50 Hz label sequence by each word's `words_align` time bounds at dataset-build time — the word becomes an instance without writing millions of tiny word files (`cut=False`, in-memory slicing rather than `utils_audio_splitter`). Evaluate this against on-disk word cuts before committing. Requires un-commenting the `words_align` (+ `primary_stress`) tiers in 11c, HR/RS only.

Frame regression (rung 7) stays deferred — no continuous per-frame target available yet beyond FPs.

### Chapter 5 — Post-hoc analysis (stub)
`50_find_best_epoch.ipynb`, `51_error_analysis.ipynb`, `52_compare_runs.ipynb`, `53_frame_event_metrics.ipynb`. Operate on saved run dirs; no GPUs/models.

---

## 7. Execution order

Follow the ladder; secure each rung before the next, and the whole **instance** pipeline before the **frame** pipeline.

**Phase A — foundation** ✅
1. Download + unpack (`10`).
2. Prep ParlaSpeech-RS (`11c`) → `utterance_instance` + `utterance_frame`.
3. Sniff (`20`) the instance JSONL.

**Phase B — instance pipeline (rungs 1–4)** ✅ *secured*
4. `31` gender (demo) → filled-pause presence/count; `32` sentiment / age (regression).
5. (rung 3 FP-type slots in once the annotator deliverable lands.)

**Phase C — frame pipeline (rungs 5–6)** ← **active next**
6. `40` FP frames (ParlaSpeech-RS), then `50_find_best_epoch`.
7. `40` primary-stress frames (HR/RS) — the north star (un-comment `words_align`/`primary_stress` in 11c first).

**Phase D — breadth**
8. Other languages (change `cfg.lang`); ROG corpora; combined runs.

**Phase E — final polish**
9. Remaining chapter-5 notebooks.
10. Refactor stable notebooks → `main.py` + `utils_<chapter>.py` (first lift: chapter-3 engine → `utils_instance_train.py`).

---

## 8. Not doing yet (FUTURE.md material)

ASR. Multi-task heads. Hyperparam search. Distributed training. Model serving. Real-time inference. **Frame-level regression** (rung 7, deferred — no suitable continuous frame target identified; only FPs available). Cross-lingual transfer. **GOS** (the GOS-specific stress target is dropped — primary stress comes from ParlaSpeech HR/RS instead).

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
- ✅ `utils_audio_splitter.py` — resolver-driven cutter/converter (stem-scan, record-path, FLAC basename-index + cache); **process-pool multicore** (`parallel_backend="process"`, thread fallback); confirmed via 11c on real RS
- ✅ `10_download_data.ipynb`
- ✅ `11b_prep_ROG_dia.ipynb` — ROG-Dialog instance JSONL (frame output pending re-pass)
- ✅ `12_audio_splitter.ipynb` — ROG-Dialog only (now retired for new preps)
- ✅ `20_sniff_dataset.ipynb`
- ✅ `11c_prep_ParlaSpeech.ipynb` — recipe registry, `utterance_instance` + `utterance_frame`, speaker-grouped splits, multicore convert + stage timer; **confirmed on real ParlaSpeech-RS**
- ✅ `31_train_classification.ipynb` — gender on real ParlaSpeech-HR; attention-mask pooling, GPU flush, timer/ETA, spot-check, populated `pred_raw`
- ✅ `32_train_regression.ipynb` — age on real data; train-only z-score normalization (real-unit metrics), attention-mask pooling fix, GPU flush, timer/ETA, spot-check
- ✅ `30_train_instance.ipynb` — **split into `31`/`32`** (historical: ran ROG-Dialog sentiment end-to-end)

**Verified on fixture, pending real-data run**
- 🧪 `11a_prep_ROG.ipynb` — dual output, fixture-verified, needs run on real ROG-Art
- 🧪 `40_train_frame.ipynb` — fixture-verified, but **predates the new chapter-4 task plan + engine parity** (timer, attention-mask, flush, twin-style spot-check); needs a rework pass

**Next up**
- [ ] Chapter 4: **FP frames** on ParlaSpeech-RS (`utterance_frame`) — bring `40` up to engine parity with 31/32, then run
- [ ] Chapter 4: **primary-stress frames** (HR/RS) — un-comment `words_align`/`primary_stress` in 11c; decide in-memory word-slicing vs on-disk word cuts
- [ ] `50_find_best_epoch.ipynb`
- [ ] Phase E lift: chapter-3 engine → `utils_instance_train.py`

---

## 11. Notes for the conversation log

- ParlaSpeech audio nesting differs by corpus and is **not** derivable from the JSONL `audio` field (HR/RS: `partX/{hash}/`; CZ: `partX/audio/psp/Y/M/D/`). Resolve by **basename** (unique per corpus) via one recursive scan, cached to `..._audio_index.json`. HR is ~1.4M files — the cache matters there; RS ~290k is fine either way.
- ParlaSpeech v3 JSONL tiers (`sentiment`, `words`, `filled_pauses`, `_align`) can be absent or `null` per language; prep treats them all as optional and writes `null` labels rather than crashing. `filled_pauses == null` means FP inference failed (≠ `[]` = none detected).
- ParlaSpeech splits are grouped by **`Speaker_ID`**, not file — gender especially leaks badly otherwise.
- **Primary stress is the north star** (rung 6, HR/RS — needs `primary_stress` + `words_align`). Not deprecated; only GOS-specific stress is dropped. Don't write stress code until the FP-frame rung is secured. Likely approach: **word as instance via in-memory slicing** — load the utterance WAV + 50 Hz label sequence, slice both by `words_align` word bounds; no word WAVs on disk (`cut=False`).
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