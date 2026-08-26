# Supported base models — chapter 4 (frame classification)

Encoder handles the frame trainer accepts in `config.json` under `model_name`.
`build_model` dispatches on `hf_config.model_type` and picks the matching head:

| `model_type`     | Head class in this chapter          | Input key         |
|------------------|-------------------------------------|-------------------|
| `wav2vec2`       | `Wav2Vec2ForFrameCLS`               | `input_values`    |
| `wav2vec2-bert`  | `Wav2Vec2BertForFrameCLS`           | `input_features`  |

Both heads share the same per-frame classifier tail (`dropout → Linear →
CrossEntropyLoss(ignore_index=-100)`).

| Model handle | Family | Params | Notes |
|---|---|---|---|
| `facebook/wav2vec2-base` | wav2vec2 | ~95 M | Small, English pretraining. Baseline that still trains fast on a T4. |
| `facebook/wav2vec2-xls-r-300m` | wav2vec2 | ~300 M | Multilingual pretraining includes Slavic. Better zero-shot on Slavic content. |
| `facebook/w2v-bert-2.0` | wav2vec2-bert | ~600 M | Meta wav2vec-BERT 2.0. Best on content-heavy tasks (primary stress belongs here). Input is 80-dim log-mel filterbanks. |
| `classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier` | wav2vec2-bert | ~600 M | HR/RS-finetuned frame classifier from CLASSLA. Loads body + classifier head with `ignore_mismatched_sizes=True`. Great as a warm start for Slovenian. |

## Label alignment

Frame labels (source rate = 50 Hz) are resampled per-record to the model's
actual output frame count during preprocess:

- **wav2vec2**: `input_values` at 16 kHz → CNN stride formula
  `(in - kernel) // stride + 1` per conv layer, using the model's own
  `conv_kernel` / `conv_stride` from config. Output is ~50 Hz.
- **wav2vec2-bert**: `input_features` from the SeamlessM4T extractor
  (already at 50 Hz frame rate). Only the optional **adapter** downsamples;
  we replicate `add_adapter`, `num_adapter_layers`, `adapter_kernel_size`,
  `adapter_stride` from the config. Output is ~25 Hz when the adapter is on,
  ~50 Hz when it is off.

## Adding a new model family

1. Add the `model_type` branch to `detect_model_family` and
   `compute_output_length` in `utils_frame_train.py`.
2. Add a matching frame-classification head (`XxxForFrameCLS`) subclassing
   the family's `PreTrainedModel`. Reuse `_frame_cls_forward` for the head +
   loss so behavior stays uniform across families.
3. Extend `build_model`'s dispatch.
4. Add a row to this table with the observed VRAM figure and any
   task-specific notes.
