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
# # 62 · Instance regression inference (chapter 6)
#
# **⚠️  SCAFFOLD — 2026-11-02. Not yet wired end to end.**
#
# Entry point for the `instance_regression` quadrant: one float per input
# audio file. First candidates for wire-up: `age`, `sentiment`.
#
# Status today:
#   * Engine stub: `utils_instance_infer.py` — raises at `load_config`.
#   * Task registry: `tasks.json/instance_regression` is empty.
#
# When ready to wire this quadrant, follow the port plan at the top of
# `utils_instance_infer.py`. Then populate `tasks.json` with a task entry
# (`model_name`, `task_name`, `task_type: "regression"`, `normalize`,
# `label_scale`), and rename this notebook / its `.ipynb` / its runner to
# drop the `-WIP` suffix.

# %% [markdown]
# # Setup
#
# Pick a task (empty until tasks.json/instance_regression is populated)
# and a run mode, then run the next cell. It will raise cleanly with the
# port plan.

# %%
TASK     = ""        # populate once a task is registered under tasks.json/instance_regression
RUN_MODE = "demo"    # "test" | "demo" | "full"

import sys
from pathlib import Path
HERE = Path.cwd()
if HERE.name != "6_inference":
    candidate = HERE / "6_inference"
    if candidate.exists():
        HERE = candidate
sys.path.insert(0, str(HERE))
CONFIG_PATH = HERE / "config.json"

import utils_instance_infer as uii
cfg = uii.load_config(CONFIG_PATH, run_mode=RUN_MODE, task=TASK,
                      quadrant="instance_regression")
