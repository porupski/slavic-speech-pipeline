# %% [markdown]
# # Slavic Speech Pipeline · Colab demo
#
# This notebook is a compact demonstration of the
# [slavic-speech-pipeline](https://github.com/porupski/slavic-speech-pipeline).
# The notebook fine-tunes a speech encoder on Croatian parliamentary speech.
# The default task is to predict speaker gender, while a one-line change
# switches the task to speaker identity. The notebook then evaluates the
# model.
#
# A complete run takes 45 to 60 minutes on a free Colab T4 GPU.
#
# ### About the models
#
# `facebook/wav2vec2-base` is a self-supervised speech encoder from Meta,
# pretrained on 960 hours of English audiobooks. The encoder maps raw audio
# to vectors that show the acoustic structure of the audio.
#
# Fine-tuning adapts these pretrained vectors to a new task. This notebook
# uses fine-tuning to classify utterances by speaker gender. Fine-tuning
# needs only a small amount of task-labelled data.

# %% [markdown]
# ---
# ## 0 · Install the dependencies
#
# Colab has torch and transformers preinstalled. Some versions can be too old
# or incomplete. The cell below installs the specific versions the notebook
# needs.

# %%
# %pip install -q "transformers>=4.40" "datasets>=2.18" "accelerate>=0.27" "soundfile" "librosa" "seaborn" "scikit-learn"

# %% [markdown]
# ---
# ## 1 · Settings
#
# All settings that a user can change are in the cell below.
#
# **`TASK`** selects the classification target. The demo supports two options:
#
# - `"gender"` — binary M and F. Trains fast and gets high accuracy.
# - `"speaker_id"` — 50-way speaker identification. Harder and slower to train.
#
# **`BASE_MODEL`** is the encoder to fine-tune from. The notebook uses this
# encoder when `PRETRAINED_MODEL` is `None`. The default is
# `facebook/wav2vec2-base`. The default is small and fits the free Colab GPU.
# Other wav2vec2-family models also work but need more VRAM. Examples are
# `wav2vec2-large` and `wav2vec2-xls-r-300m` for Slavic language coverage.
#
# **`PRETRAINED_MODEL`** is an optional override. Set this to a HuggingFace
# repo id of a model already fine-tuned for the task. The notebook then skips
# training and only evaluates.

# %%
# ── task ────────────────────────────────────────────────────────────────
TASK = "speaker_id"                # "gender" | "speaker_id"

# ── model ───────────────────────────────────────────────────────────────
BASE_MODEL       = "facebook/wav2vec2-base"
PRETRAINED_MODEL = None            # e.g. "your-user/wav2vec2-hr-gender" → skips training

# ── data ────────────────────────────────────────────────────────────────
# ParlaSpeech-HR-benchmark-v3 on the HuggingFace Hub.
#   https://huggingface.co/datasets/porupski/ParlaSpeech-HR-benchmark_v3
DATASET_HF_ID = "porupski/ParlaSpeech-HR-benchmark_v3"

# ── fine-tuning knobs ──────────────────────────────────────────────────
# NOTE: gender converges in 2 epochs. speaker_id (50-class) usually needs
# 5+ epochs (and often a higher LR, e.g. 5e-5). At 2 epochs and LR=1e-5, the
# classifier head does not warm up and macro-F1 stays near random.
NUM_EPOCHS    = 2
BATCH_SIZE    = 8   # 8 ≈ 6.5 GB VRAM, 16 ≈ 9.1 GB (T4 has 15 GB)
LEARNING_RATE = 1e-5

MAX_TRAIN     = 10000              # cap on train instances for speed on T4
MAX_EVAL      = 3000               # cap on dev and test instances

SEED = 1234

# %% [markdown]
# ---
# ## 2 · Imports and device check
#
# The free Colab tier provides a T4 GPU with about 15 GB of VRAM. This is
# enough VRAM for `wav2vec2-base`. Enable the GPU from the menu if the GPU
# is off:
# **Runtime → Change runtime type → Hardware accelerator: T4 GPU**.
# Re-run this cell after you enable the GPU.
#
# This cell also flushes RAM and VRAM at the start, preventing stale state
# from a prior failed run.
#
# The cell selects the mixed-precision dtype from the GPU compute capability:
#
# - **bf16** on Ampere or newer (compute capability 8 or higher).
# - **fp16** on Turing and Volta (T4, V100).
#
# The Turing and Volta GPUs do not have native bf16 tensor cores, so
# `bf16=True` on these GPUs causes a silent fallback to fp32.

# %%
import gc
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm.auto import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)


def flush_memory(note: str = ""):
    """Free Python garbage and the CUDA cache. Use between heavy phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if note:
        print(f"🧹 memory flushed ({note})")


# Flush any state left behind by a prior run of this cell.
flush_memory("startup")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEVICE}")
if DEVICE == "cuda":
    print(f"gpu    = {torch.cuda.get_device_name(0)}")
    _cc = torch.cuda.get_device_capability(0)[0]
    USE_BF16 = _cc >= 8
    USE_FP16 = not USE_BF16
    print(f"mixed precision = {'bf16' if USE_BF16 else 'fp16'}  (compute cap {_cc})")
else:
    USE_BF16 = USE_FP16 = False
    print("⚠️  No GPU detected. Training will be very slow.")
    print("    Runtime → Change runtime type → T4 GPU, then re-run.")

# %% [markdown]
# ---
# ## 3 · Download the ParlaSpeech-HR-benchmark-v3 dataset
#
# The dataset is a HuggingFace Parquet dataset with a single `train` shard
# holding all 22 000 utterances. The full download size is about 8 GB with
# the audio inlined.
#
# Task-level splits (`train`, `dev`, `test`) are stored in per-task columns
# named `benchmark_{TASK}_split`. This layout permits the same utterance to
# be in the `gender` train split and the `age` test split at the same time.
#
# The first download takes about 15 minutes on the Colab network. The cache
# is retained for the rest of the session. The filter and grouping steps
# run locally after the download. Local processing is faster than a
# row-by-row streaming decode.
#
# The code removes the full dataset from RAM after grouping. This reclaims
# several GB of memory before the preprocessing step.

# %%
LABEL_COL = f"benchmark_{TASK}_label"
SPLIT_COL = f"benchmark_{TASK}_split"
LABEL_ORDER_FIXED = {"gender": ["M", "F"], "speaker_id": None}[TASK]

print(f"downloading {DATASET_HF_ID} (default config, train shard) …")
ds = load_dataset(DATASET_HF_ID, "default", split="train")
print(f"total rows: {len(ds)}")

# Filter to the rows for the selected task. This step touches scalar columns
# only, so it does not decode the audio column.
task_ds = ds.filter(
    lambda r: r[SPLIT_COL] in ("train", "dev", "test")
              and r[LABEL_COL] not in (None, ""),
    desc=f"filter to task={TASK}",
)
task_ds = task_ds.shuffle(seed=SEED)
print(f"task rows (all splits): {len(task_ds)}")

# Group rows into train, dev, and test buckets. Apply the per-split cap.
# The audio decode happens here, once for each row that survives the cap.
CAPS = {"train": MAX_TRAIN, "dev": MAX_EVAL, "test": MAX_EVAL}
buckets = {"train": [], "dev": [], "test": []}
for rec in tqdm(task_ds, desc="bucketing", total=len(task_ds)):
    split = rec[SPLIT_COL]
    if len(buckets[split]) >= CAPS[split]:
        continue
    audio = rec["audio"]
    buckets[split].append({
        "waveform":    np.asarray(audio["array"], dtype=np.float32),
        "sr":          int(audio["sampling_rate"]),
        "label":       str(rec[LABEL_COL]),
        "instance_id": rec.get("instance_id", ""),
    })
    if all(len(buckets[s]) >= CAPS[s] for s in buckets):
        break

train_recs = buckets["train"]
dev_recs   = buckets["dev"]
test_recs  = buckets["test"]
print(f"train = {len(train_recs)}   dev = {len(dev_recs)}   test = {len(test_recs)}")

# The full HF dataset is no longer needed. The waveforms are in the buckets.
# Remove the dataset to reclaim RAM before the preprocess spike.
del ds, task_ds, buckets
flush_memory("post-bucketing")

# %% [markdown]
# ---
# ## 4 · Build the label to integer map
#
# HuggingFace classifiers need integer class ids, not strings, so the next
# cell builds a two-way label-to-integer map. For `gender`, the order is
# fixed: `M`, `F`. For `speaker_id`, the code finds the 50 speaker names
# from the training split.

# %%
if LABEL_ORDER_FIXED is not None:
    LABEL_ORDER = LABEL_ORDER_FIXED
else:
    LABEL_ORDER = sorted({r["label"] for r in train_recs})

label2id = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
id2label = {i: lbl for lbl, i in label2id.items()}
print(f"{len(LABEL_ORDER)} classes")
if len(LABEL_ORDER) <= 10:
    print(f"  {LABEL_ORDER}")
else:
    print(f"  first 5: {LABEL_ORDER[:5]}   last 5: {LABEL_ORDER[-5:]}")

# %% [markdown]
# ---
# ## 5 · Load the encoder and the feature extractor
#
# `AutoFeatureExtractor` converts the waveform to a tensor.
# `AutoModelForAudioClassification` adds a small linear classifier head on
# top of the encoder. If `PRETRAINED_MODEL` is set, the notebook goes to
# inference only. If `PRETRAINED_MODEL` is `None`, the notebook does
# fine-tuning below.

# %%
MODEL_NAME = PRETRAINED_MODEL if PRETRAINED_MODEL else BASE_MODEL
FINE_TUNE  = PRETRAINED_MODEL is None

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
feature_extractor.return_attention_mask = True   # padding-safe pooling

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_ORDER),
    label2id={str(k): int(v) for k, v in label2id.items()},
    id2label={int(k): str(v) for k, v in id2label.items()},
    ignore_mismatched_sizes=True,
).to(DEVICE)

print(f"model : {MODEL_NAME}")
print(f"mode  : {'fine-tune from base' if FINE_TUNE else 'inference-only (skipping training)'}")

# %% [markdown]
# ---
# ## 6 · Convert waveforms to model inputs
#
# The download step provided decoded 16 kHz mono waveforms. In this step,
# the feature extractor normalises the waveforms. The output is
# `input_values`, one float array for each clip. The next cell adds padding
# at batch time. Batch-time padding prevents short clips from wasting memory.
#
# **Chunked processing.** A single call with `MAX_TRAIN=10 000` clips causes
# a RAM spike above the Colab limit of about 12 GB, and the kernel then
# stops without a message. To prevent the spike, the code does the
# following:
#
# 1. Splits the records into small chunks.
# 2. Runs the feature extractor on each chunk.
# 3. Removes the raw waveform buffers for those records.
# 4. Concatenates the small Arrow tables at the end of the loop.
#
# The Arrow concatenation is a metadata operation. The concatenation does
# not copy the data.
#
# The `test_recs` list keeps its waveforms. The audio-playback cell at the
# end of the notebook needs the waveforms.

# %%
def preprocess_to_dataset(recs, chunk_size: int = 200, free_source: bool = False):
    """Chunked feature extraction. Peak RAM is bounded to chunk_size clips.
    If free_source is True, the function releases each waveform after use."""
    if not recs:
        return None
    sub_ds = []
    n_chunks = (len(recs) + chunk_size - 1) // chunk_size
    for i in tqdm(range(n_chunks), desc="preprocess", unit="chunk"):
        chunk = recs[i * chunk_size : (i + 1) * chunk_size]
        audio_arrays = [r["waveform"] for r in chunk]
        inputs = feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors=None, padding=False,
        )
        sub_ds.append(Dataset.from_dict({
            "input_values": inputs["input_values"],
            "labels":       [label2id[r["label"]] for r in chunk],
        }))
        if free_source:
            for r in chunk:
                r["waveform"] = None
        del audio_arrays, inputs
    gc.collect()
    return concatenate_datasets(sub_ds)


print("preprocessing train …" if FINE_TUNE else "skipping train preprocessing")
train_ds = preprocess_to_dataset(train_recs, free_source=True) if FINE_TUNE else None
print("preprocessing test …")
# free_source=False keeps the test_recs waveforms for the audio-playback cell.
test_ds = preprocess_to_dataset(test_recs, free_source=False)

# The dev_recs list is not used again. Release its waveforms too.
for r in dev_recs:
    r["waveform"] = None
flush_memory("post-preprocess")

# %% [markdown]
# ---
# ## 7 · Batch-time padding
#
# Utterances have different lengths. The collator pads each batch to the
# length of the longest clip in that batch. The collator also builds an
# attention mask, telling the model which frames are real audio and which
# frames are padding.

# %%
class DataCollator:
    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]
        batch = feature_extractor.pad(
            {"input_values": input_values},
            padding=True, return_attention_mask=True, return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


collator = DataCollator()

# %% [markdown]
# ---
# ## 8 · Fine-tune the model
#
# This cell runs only when the notebook starts from a base encoder.
#
# The CNN feature encoder is the expensive front end of the model. The code
# freezes this front end. Only the transformer layers and the classifier
# head receive updates during training.
#
# Mixed precision uses:
#
# - bf16 on Ampere or newer GPUs.
# - fp16 on Turing and Volta GPUs (T4, V100).
#
# A typical run on the Colab T4 takes about 30 minutes for 10 000 clips
# over 2 epochs. After training, the code releases the Trainer and the
# optimiser states, freeing VRAM before the evaluation step.

# %%
if FINE_TUNE:
    if hasattr(getattr(model, "wav2vec2", None), "freeze_feature_encoder"):
        model.wav2vec2.freeze_feature_encoder()
        print("🔒 CNN feature encoder frozen")

    args = TrainingArguments(
        output_dir="./training_tmp",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.10,
        logging_steps=max(1, (len(train_ds) // BATCH_SIZE * NUM_EPOCHS) // 20),
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        bf16=USE_BF16,
        fp16=USE_FP16,
        seed=SEED,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, data_collator=collator,
    )
    trainer.train()
    print("✅ fine-tuning done")

    # The Trainer and the optimizer states hold VRAM until garbage collection.
    # Release them before evaluation to free VRAM for the forward pass.
    del trainer, args
    flush_memory("post-training")
else:
    print(f"skipping training — loaded pretrained model {MODEL_NAME}")

# %% [markdown]
# ---
# ## 9 · Evaluate the model on the test split
#
# The test split is held out from training. For the `gender` task, the test
# split is speaker-disjoint from the train split. This means the model did
# not hear these speakers during training.
#
# A high score on this test shows that the model learned the difference
# between male and female voices, not that it memorised individual
# speakers.

# %%
from torch.utils.data import DataLoader

model.eval()
preds_all, labels_all = [], []
dl = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collator)
with torch.no_grad():
    for batch in dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(
            input_values=batch["input_values"],
            attention_mask=batch.get("attention_mask"),
        ).logits
        preds_all.extend(logits.argmax(-1).cpu().numpy().tolist())
        labels_all.extend(batch["labels"].cpu().numpy().tolist())

acc = accuracy_score(labels_all, preds_all)
f1  = f1_score(labels_all, preds_all, average="macro")
print(f"test accuracy  = {acc:.3f}")
print(f"test macro F1  = {f1:.3f}")

del dl
flush_memory("post-evaluation")

# %% [markdown]
# ---
# ## 10 · Confusion matrix
#
# The rows of the matrix show the true label. The columns show the model
# prediction. Cells on the diagonal are correct predictions. Cells off the
# diagonal are errors.
#
# For the `gender` task, a good result shows an almost-clean diagonal. For
# the `speaker_id` task with 50 classes, a good result shows a bright
# diagonal and dim off-diagonal cells.

# %%
cm = confusion_matrix(labels_all, preds_all, labels=list(range(len(LABEL_ORDER))))
figsize = (6, 5) if len(LABEL_ORDER) <= 10 else (12, 10)
plt.figure(figsize=figsize)
annot = True if len(LABEL_ORDER) <= 10 else False
sns.heatmap(cm, annot=annot, fmt="d",
            xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
            cmap="Blues", cbar=False)
plt.xlabel("predicted"); plt.ylabel("true")
plt.title(f"{TASK} · test-set confusion  (acc={acc:.2f}, macro-F1={f1:.2f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 11 · Listen to sample predictions
#
# This cell shows six random test clips. Four clips are correct predictions
# and two clips are errors. Each panel shows:
#
# - The true label.
# - The model prediction.
# - An audio player.
#
# Use the audio player to verify that the model behaviour matches the
# metric scores.

# %%
from IPython.display import Audio, display

rng = random.Random(SEED + 5)

correct_indices   = [i for i in range(len(test_recs)) if id2label[int(preds_all[i])] == test_recs[i]["label"]]
incorrect_indices = [i for i in range(len(test_recs)) if id2label[int(preds_all[i])] != test_recs[i]["label"]]
picks_true  = rng.sample(correct_indices,   min(4, len(correct_indices)))
picks_false = rng.sample(incorrect_indices, min(2, len(incorrect_indices)))
picks = picks_true + picks_false

for i in picks:
    rec  = test_recs[i]
    true = rec["label"]
    pred = id2label[int(preds_all[i])]
    mark = "✅" if true == pred else "❌"
    print(f"{mark}  true: {true:<25}  pred: {pred}   ({rec['instance_id']})")
    display(Audio(rec["waveform"], rate=rec["sr"]))

# %% [markdown]
# ---
# ## What's next
#
# The full pipeline provides:
#
# - Dataset preparation for ParlaSpeech HR, PL, RS, CZ and ROG.
# - Speaker-disjoint, per-class stratified split construction for train,
#   dev, and test.
# - Instance-level training for classification and regression. Single-phase
#   and two-phase modes are supported. Label normalization is available for
#   continuous targets.
# - Per-frame training for tasks such as filled-pause detection.
# - Inference over a folder of audio. Inference supports resume and Praat
#   TextGrid export.
#
# Full repository:
# [slavic-speech-pipeline](https://github.com/porupski/slavic-speech-pipeline)
