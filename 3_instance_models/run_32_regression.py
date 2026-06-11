#!/usr/bin/env python3
"""run_32_regression — instance regression runner (chapter 3).

The command-line twin of 32_train_instance_regression.ipynb — same engine
(utils_instance_train.py), same config.json, no interactive prompts. This is
the tool for FULL runs: launch it in tmux and walk away.

Usage:
  python run_32_regression.py                       # run_mode from config.json, CPU
  python run_32_regression.py --mode demo --use_gpu # demo run on the reserved GPU
  python run_32_regression.py -m full --use_gpu -t parlaspeech_rs_sentiment

Device policy: CPU by default (safe, prompt-free). --use_gpu arms the reserved
GPU from config.json (shared.reserved_gpu) — the explicit, non-interactive
replacement for the notebook's input-gated GPU guard. We NEVER touch another GPU.
"""

import argparse
import json
import os
import sys
from pathlib import Path

TASK_TYPE = "regression"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # utils_instance_train.py lives next to this runner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", default=str(HERE / "config.json"),
                   help="path to config.json")
    p.add_argument("--mode", "-m", choices=["test", "demo", "full"], default=None,
                   help="override run_mode from config.json")
    p.add_argument("--target", "-t", default=None,
                   help="override the regression target preset from config.json")
    p.add_argument("--use_gpu", action="store_true",
                   help="arm the reserved GPU (no prompt); default is CPU")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Environment — MUST precede the utils import (which imports torch) ────
    # CUDA_VISIBLE_DEVICES only takes effect if set before torch's first CUDA
    # call; MPLBACKEND=Agg keeps matplotlib headless for tmux/ssh runs.
    os.environ.setdefault("MPLBACKEND", "Agg")
    reserved_gpu = json.loads(Path(args.config).read_text())["shared"].get("reserved_gpu", "2")
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = reserved_gpu
        print(f"🚀 GPU mode — CUDA_VISIBLE_DEVICES={reserved_gpu}  "
              f"(physical GPU {reserved_gpu} → cuda:0 in-process)")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"🖥️  CPU mode (pass --use_gpu to arm physical GPU {reserved_gpu})")

    import utils_instance_train as uit
    from utils_instance_train import (
        load_config, validate_config, resolve_device, print_config_summary,
        resolve_target, load_and_cap_splits, print_rough_eta,
        fit_normalizer,
        make_run_dirs, load_feature_extractor, run_phase,
        print_recalibrated_eta, print_run_summary, plot_test_scatter, spot_check,
        mark, print_stage_breakdown,
    )
    mark("literal start")
    print(f"PROJECT_ROOT = {uit.PROJECT_ROOT}")
    print(f"HF_HOME      = {os.environ['HF_HOME']}")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg, raw_config = load_config(args.config, task_type=TASK_TYPE, run_mode=args.mode)
    if args.target is not None:
        cfg.target = args.target
        resolve_target(cfg)
        if cfg.task_type != TASK_TYPE:
            raise SystemExit(f"--target {args.target!r} is a {cfg.task_type} preset; "
                             f"this runner is {TASK_TYPE}-only.")
    validate_config(cfg)
    device = resolve_device(cfg, use_cuda=args.use_gpu)
    if cfg.run_mode == "full" and device == "cpu":
        print("⚠️  full run on CPU — this will take a very long time. "
              "Did you forget --use_gpu?")
    print("✅ config valid\n")
    print_config_summary(cfg, device)

    # ── Data ──────────────────────────────────────────────────────────────────
    mark("data prep")
    train_records, dev_records, test_records = load_and_cap_splits(cfg)
    print_rough_eta(len(train_records), len(dev_records), cfg)

    # ── Target normalization (fit on TRAIN only) ─────────────────────────────
    normalizer = fit_normalizer(cfg, train_records, dev_records, test_records)
    label2id, id2label = None, None   # classification-only; defined for twin-identical engine calls

    # ── Run ───────────────────────────────────────────────────────────────────
    run_dir, model_dir, run_name = make_run_dirs(cfg, train_records, normalizer=normalizer)
    feature_extractor = load_feature_extractor(cfg)

    mark("model prep")
    phase1_results, phase1_best = run_phase(
        phase_name="phase1_dev",
        train_records=train_records, eval_records=dev_records,
        eval_split_name="DEV", save_best_model=False,
        cfg=cfg, run_dir=run_dir, model_dir=model_dir,
        feature_extractor=feature_extractor,
        label2id=label2id, id2label=id2label, device=device,
        normalizer=normalizer,
    )
    mark("end phase 1")

    print_recalibrated_eta(len(train_records), len(dev_records), cfg)
    phase2_results, phase2_best = run_phase(
        phase_name="phase2_test",
        train_records=train_records + dev_records, eval_records=test_records,
        eval_split_name="TEST", save_best_model=True,
        cfg=cfg, run_dir=run_dir, model_dir=model_dir,
        feature_extractor=feature_extractor,
        label2id=label2id, id2label=id2label, device=device,
        normalizer=normalizer,
    )
    mark("end phase 2")

    # ── Report ────────────────────────────────────────────────────────────────
    print_run_summary(cfg, run_name, run_dir, model_dir, phase1_best, phase2_best,
                      normalizer=normalizer)
    plot_test_scatter(run_dir, phase2_best, show=False)
    spot_check(run_dir, phase2_best, cfg.task_type)

    mark("end script")
    print_stage_breakdown()


if __name__ == "__main__":
    main()
