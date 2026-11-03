# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: ssp
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 64 · Frame regression inference (chapter 6)
#
# **⚠️  SCAFFOLD — 2026-11-02. Not yet wired end to end.**
#
# Entry point for the `frame_regression` quadrant: one float per model
# frame. Rare today — the training twin (`42_train_frame_regression-WIP`)
# in chapter 4 is also a scaffold. Candidate wire-ups once a trained
# per-frame regressor exists: per-frame prominence, per-frame log-pitch,
# per-frame energy derivative.
#
# Status today:
#   * Engine: `utils_frame_infer.py` serves the sibling `frame_classification`
#     quadrant only (hard-coded via its `QUADRANT` constant). This scaffold
#     needs a `utils_frame_regression_infer.py` module — not yet written.
#   * Task registry: `6_tasks.json/frame_regression` is empty.
#
# Port plan (dated 2026-11-02):
#   1. Create `utils_frame_regression_infer.py` by copying
#      `utils_frame_infer.py` and swapping the model class
#      (`AutoModelForAudioFrameClassification` → the matching regression
#      head), the loss, and the frames→events collapse (regression needs
#      a per-frame numeric emission, not an event collapse). Set its
#      `QUADRANT = "frame_regression"`.
#   2. Register a task under `6_tasks.json/frame_regression`.
#   3. Point this notebook at `utils_frame_regression_infer`.
#   4. Rename this file / its `.ipynb` / its runner to drop the `-WIP`
#      suffix.

# %% [markdown]
# # Setup
#
# Running the next cell today raises cleanly with the port plan.

# %%
raise NotImplementedError(
    "64_frame_regression_inference is a scaffold (2026-11-02). "
    "Engine needed: utils_frame_regression_infer.py (not yet written). "
    "See the port plan at the top of this notebook and in 6_README.md."
)
