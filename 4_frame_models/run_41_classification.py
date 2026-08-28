#!/usr/bin/env python3
"""run_41_classification — frame classification runner (chapter 4).

Command-line twin of 41_train_frame_classification.ipynb — same engine
(utils_frame_train.py), same config.json, no interactive prompts. This is the
tool for FULL runs: launch it in tmux and walk away.

Usage:
  python run_41_classification.py                             # config.json's run_mode, CPU
  python run_41_classification.py --mode demo --use_gpu
  python run_41_classification.py -m full --use_gpu -t si_primary_stress_frames
  python run_41_classification.py -m full --use_gpu --model_name facebook/w2v-bert-2.0

Device policy: CPU by default (safe, prompt-free). --use_gpu arms the reserved
GPU from config.json (shared.reserved_gpu). We NEVER touch another GPU.
"""

import argparse
import json
import os
import sys
from pathlib import Path

TASK_TYPE = "classification"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


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
                   help="override the classification target preset")
    p.add_argument("--model_name", default=None,
                   help="override the model_name (e.g. facebook/w2v-bert-2.0)")
    p.add_argument("--use_gpu", action="store_true",
                   help="arm the reserved GPU (no prompt); default is CPU")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Environment — MUST precede the utils import ─────────────────────────
    os.environ.setdefault("MPLBACKEND", "Agg")
    reserved_gpu = json.loads(Path(args.config).read_text())["shared"].get("reserved_gpu", "2")
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = reserved_gpu
        print(f"🚀 GPU mode — CUDA_VISIBLE_DEVICES={reserved_gpu}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"🖥️  CPU mode (pass --use_gpu to arm physical GPU {reserved_gpu})")

    import utils_frame_train as uft
    from utils_frame_train import (
        load_config, validate_config, resolve_device, print_config_summary,
        resolve_target, load_and_cap_splits, print_rough_eta,
        validate_frame_rate, drop_long_records, build_label_maps,
        make_run_dirs, load_feature_extractor, run_phase,
        print_recalibrated_eta, print_run_summary, spot_check,
        plot_test_confusion, plot_test_example_predictions,
        mark, print_stage_breakdown, print_project_info,
    )
    mark("literal start")
    print_project_info()

    # ── Config ────────────────────────────────────────────────────────────
    cfg, raw_config = load_config(args.config, task_type=TASK_TYPE, run_mode=args.mode)
    if args.target is not None:
        cfg.target = args.target
        resolve_target(cfg)
        if cfg.task_type != TASK_TYPE:
            raise SystemExit(f"--target {args.target!r} is a {cfg.task_type} preset; "
                             f"this runner is {TASK_TYPE}-only.")
    if args.model_name is not None:
        cfg.model_name = args.model_name
    validate_config(cfg)
    device = resolve_device(cfg, use_cuda=args.use_gpu)
    if cfg.run_mode == "full" and device == "cpu":
        print("⚠️  full run on CPU — this will take a very long time. "
              "Did you forget --use_gpu?")
    print("✅ config valid\n")
    print_config_summary(cfg, device)

    # ── Data ──────────────────────────────────────────────────────────────
    mark("data prep")
    train_records, dev_records, test_records = load_and_cap_splits(cfg)
    for _split, _recs in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
        validate_frame_rate(_recs, cfg.required_frame_rate_hz)
    if cfg.enable_max_audio_seconds:
        train_records, n_tr = drop_long_records(train_records, cfg.max_audio_seconds)
        dev_records,   n_dv = drop_long_records(dev_records,   cfg.max_audio_seconds)
        test_records,  n_te = drop_long_records(test_records,  cfg.max_audio_seconds)
        print(f"dropped >{cfg.max_audio_seconds}s: train={n_tr} dev={n_dv} test={n_te}")
    print_rough_eta(len(train_records), len(dev_records), cfg)

    label2id, id2label, num_labels = build_label_maps(cfg, train_records, dev_records, test_records)

    # ── Run ───────────────────────────────────────────────────────────────
    run_dir, model_dir, run_name = make_run_dirs(cfg, train_records)
    feature_extractor = load_feature_extractor(cfg)

    mark("model prep")
    phase1_results, phase1_best = run_phase(
        phase_name="phase1_dev",
        train_records=train_records, eval_records=dev_records,
        eval_split_name="DEV", save_best_model=False,
        cfg=cfg, run_dir=run_dir, model_dir=model_dir,
        feature_extractor=feature_extractor,
        label2id=label2id, id2label=id2label, device=device,
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
    )
    mark("end phase 2")

    print_run_summary(cfg, run_name, run_dir, model_dir, phase1_best, phase2_best)
    spot_check(run_dir, phase2_best)
    plot_test_confusion(run_dir, phase2_best, label2id, cfg.label_order, show=False)
    plot_test_example_predictions(run_dir, phase2_best, id2label,
                                  n_per_tier=cfg.n_examples_to_plot, show=False)

    mark("end script")
    print_stage_breakdown()


if __name__ == "__main__":
    main()
