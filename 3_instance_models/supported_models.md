# Supported base models

This table lists the encoder handles that the chapter-3 trainers accept
in `config.json` under `model_name`. Each row names the input format,
the head compatibility, an approximate VRAM figure, and the tasks where
the model is a good fit.

| Model handle | Input type | Head compatibility | VRAM at BS=8 | Best-fit tasks | Notes |
|---|---|---|---|---|---|
| `facebook/wav2vec2-base` | 16 kHz raw waveform | Wav2Vec2 sequence-classification and regression heads | tbd | Speaker ID, gender, prosodic targets | Small model (~95 M params). Pretrained on 960 hours of English audio. Strong on speaker tasks in spite of the language mismatch. |
| `facebook/wav2vec2-xls-r-300m` | 16 kHz raw waveform | Wav2Vec2 sequence-classification and regression heads | tbd | Slavic-language content tasks, speaker ID | Larger model (~300 M params). Multilingual pretraining includes Slavic languages. |
| `facebook/w2v-bert-2.0` | Log-Mel filterbank features (80-dim, extracted by `SeamlessM4TFeatureExtractor`) | Wav2Vec2Bert sequence-classification head | tbd | Content-heavy tasks, transcription-adjacent targets | Meta wav2vec-BERT 2.0. Best on content tasks. Underperforms on speaker ID in prior runs. |

## VRAM notes

The VRAM figures are from single-GPU runs at batch size 8 with the
default `max_duration_s = 15.0`. Values change with batch size, audio
length, and mixed-precision settings. Fill in the table after a run.

## Adding a new model

1. Add the handle to `config.json` under `model_name`.
2. Confirm that the model exposes a sequence-classification head, or
   that the trainer can wrap the model with one.
3. Run the `test` mode first to check that the input pipeline matches
   the expected format for the model.
4. Add a row to this table with the observed VRAM figure and any
   task-specific notes.
