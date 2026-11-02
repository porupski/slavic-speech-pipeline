"""utils_instance_infer — chapter 6 instance-level inference engine.

**⚠️  SCAFFOLD, 2026-11-02. Not yet wired end to end.**

Serves the ``instance_classification`` and ``instance_regression`` quadrants
of chapter 6. The entry-point notebooks are:
  * 61_instance_classification_inference-WIP.* (scaffold)
  * 62_instance_regression_inference-WIP.* (scaffold)

The port template is chapter 3's ``utils_instance_train.py`` — specifically
its model-build + predict path. The training path is NOT needed here; only
the inference side (load checkpoint, mean-pool over masked frames, emit one
label or one float per record).

Port plan (dated 2026-11-02). To make 61 and 62 train end to end, follow the
same shape as ``utils_frame_infer.py``:

 1. Add Config fields: ``batch_size``, ``background_class_id`` (classification
    only), ``postproc_*`` (if any), ``normalize`` (regression only — port the
    chapter-3 z-score normalizer).
 2. Implement ``load_model(cfg, device)`` with
    ``AutoModelForAudioClassification`` for classification tasks, and the
    chapter-3 ``Wav2Vec2ForRegression`` family for regression tasks. Dispatch
    on ``cfg.task_type``.
 3. Implement ``run_inference(cfg, run_name)``: walk cfg.audio_dir, read each
    WAV, run the model, emit one JSONL line per file. For regression, decode
    through the normalizer so JSONL values read in natural units.
 4. Register at least one task per quadrant in ``tasks.json``:
      - instance_classification: ``gender``, ``FP_presence``, ``speaker_id``
      - instance_regression:     ``age``, ``sentiment``
 5. Fill in each scaffold's TASK selector and remove its ``-WIP`` suffix.
"""

from __future__ import annotations

from dataclasses import dataclass


QUADRANTS = ("instance_classification", "instance_regression")


@dataclass
class Config:
    # -- Run mode ------------------------------------------------------------
    run_mode: str = "full"
    cap_files: int | None = None

    # -- Model ---------------------------------------------------------------
    model_name: str = "auto"
    task_name: str = ""
    task_type: str = "classification"  # "classification" | "regression"

    # -- Input / output ------------------------------------------------------
    audio_dir: str = "data/inference_input"
    runs_dir: str = "runs"

    # -- Hardware ------------------------------------------------------------
    use_cuda: bool = True
    reserved_gpu: str = "2"


def load_config(path, run_mode: str, task: str, quadrant: str):
    """SCAFFOLD stub, 2026-11-02. The real loader will mirror
    ``utils_frame_infer.load_config``: read config.json shared block, apply
    the task entry from tasks.json/<quadrant>/<task>, overlay the run_mode."""
    if quadrant not in QUADRANTS:
        raise ValueError(
            f"quadrant={quadrant!r} not served by utils_instance_infer; "
            f"expected one of {QUADRANTS}"
        )
    raise NotImplementedError(
        "utils_instance_infer is a scaffold (2026-11-02). "
        "Port the predict path from chapter 3's utils_instance_train.py. "
        f"Target quadrant: {quadrant!r}. Target task: {task!r}. "
        "See the port plan in this module's docstring and in "
        "61_instance_classification_inference-WIP.py / "
        "62_instance_regression_inference-WIP.py."
    )
