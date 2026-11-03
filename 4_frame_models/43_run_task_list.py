#!/usr/bin/env python3
"""43_run_task_list — chapter-4 sequential task-list runner.

Walks the ``TASKS`` list below one by one. Each task is a full
phase-1 (DEV) → phase-2 (TEST) frame-classification run through the same
engine as ``run_41_classification``.
aggregates the best DEV + best TEST metrics into a CSV, plots a bar chart
comparing macro-F1 and positive-class F1 across tasks, and lets you
fire-and-forget overnight. A crash in one cell does not kill the queue — the
error is written into that task's CSV row and the next cell runs.

Every cell writes its own ``<list_root>/<label>/`` tree with:
- per-epoch predictions.json / epoch_summary.json / example_predictions.png
- top-level confusion_matrix_test.png + example_predictions_test.png
- effective config snapshot
- best_model/ (weights only for the best phase-2 epoch)

Aggregate summary lands at ``<list_root>/task_list_summary.csv`` +
``<list_root>/task_list_summary.png``. ``tasks.json`` snapshots the exact TASKS
list for provenance.

Usage:
  python 43_compare_models.py --use_gpu
  python 43_compare_models.py --use_gpu --dry_run          # print tasks, exit
  python 43_compare_models.py --use_gpu --limit 3          # first 3 tasks only
  python 43_compare_models.py --use_gpu --list_dir grid_lr_sweep

Base config is `4_config.json` under the chosen ``--mode`` (default: full);
per-task overrides layer on top.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TASK_TYPE = "classification"


# ══════════════════════════════════════════════════════════════════════════
# TASK LIST — edit this list to change the queue
# ══════════════════════════════════════════════════════════════════════════
# Each cell is a dict of Config field overrides. Every key must name a real
# Config field in utils_frame_train.py (typos are a hard error). The `label`
# is not a Config field; it names the cell's sub-directory and CSV row.
#
# Baseline (facebook/w2v-bert-2.0) got macro-F1 ≈ 0.94 / F1+ ≈ 0.90 at
# lr=1e-5, bs=64, 15 epochs. The sweep below varies LR around that anchor
# for the vanilla body and probes the CLASSLA HR/RS-finetuned head at
# finetune-scale LRs.

TASKS: list[dict] = [
    # ── facebook/w2v-bert-2.0 (from-scratch head) ────────────────────────
    {"label": "bert_vanilla_lr1e5", "model_name": "facebook/w2v-bert-2.0",
     "learning_rate": 1e-5, "num_epochs": 20, "batch_size": 64},
    {"label": "bert_vanilla_lr3e5", "model_name": "facebook/w2v-bert-2.0",
     "learning_rate": 3e-5, "num_epochs": 20, "batch_size": 64},
    {"label": "bert_vanilla_lr5e5", "model_name": "facebook/w2v-bert-2.0",
     "learning_rate": 5e-5, "num_epochs": 20, "batch_size": 64},
    {"label": "bert_vanilla_lr1e5_bs32", "model_name": "facebook/w2v-bert-2.0",
     "learning_rate": 1e-5, "num_epochs": 20, "batch_size": 32},

    # ── classla/Wav2Vec2BertPrimaryStress... (already HR/RS-finetuned) ───
    # Fine-tune scale LRs — the head is already trained on primary stress.
    {"label": "classla_lr1e6", "model_name": "classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier",
     "learning_rate": 1e-6, "num_epochs": 15, "batch_size": 64},
    {"label": "classla_lr3e6", "model_name": "classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier",
     "learning_rate": 3e-6, "num_epochs": 15, "batch_size": 64},
    {"label": "classla_lr1e5", "model_name": "classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier",
     "learning_rate": 1e-5, "num_epochs": 20, "batch_size": 64},
    # {"label": "classla_lr3e5", "model_name": "classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier",
    #  "learning_rate": 3e-5, "num_epochs": 15, "batch_size": 64},
    {"label": "classla_lr1e5_bs32", "model_name": "classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier",
     "learning_rate": 1e-5, "num_epochs": 20, "batch_size": 32},

    # ── longer runs, best LR per family ──────────────────────────────────
    {"label": "bert_vanilla_lr1e5_30ep", "model_name": "facebook/w2v-bert-2.0",
     "learning_rate": 1e-5, "num_epochs": 30, "batch_size": 64},
    {"label": "classla_lr1e5_25ep", "model_name": "classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier",
     "learning_rate": 1e-5, "num_epochs": 25, "batch_size": 64},
]


CSV_FIELDS = [
    "task_idx", "label", "model_name", "learning_rate", "num_epochs", "batch_size",
    "wall_time_s",
    "phase1_best_epoch", "phase1_macro_f1", "phase1_f1_pos", "phase1_acc", "phase1_loss",
    "phase2_best_epoch", "phase2_macro_f1", "phase2_f1_pos", "phase2_acc", "phase2_loss",
    "run_dir", "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", default=str(HERE / "4_config.json"),
                   help="path to 4_config.json (base layer)")
    p.add_argument("--mode", "-m", choices=["test", "demo", "full"], default="full",
                   help="base run_mode; cell overrides layer on top")
    p.add_argument("--use_gpu", action="store_true",
                   help="arm the reserved GPU (no prompt); default is CPU")
    p.add_argument("--dry_run", action="store_true",
                   help="print the planned tasks + exit")
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N tasks (for smoke tests)")
    p.add_argument("--list_dir", default=None,
                   help="task-list sub-directory name under runs/ (default: <timestamp>_task_list)")
    return p.parse_args()


def print_task_plan(tasks: list[dict]) -> None:
    w = max(len(c["label"]) for c in tasks) if tasks else 5
    print(f"planned {len(tasks)} tasks:")
    for i, c in enumerate(tasks, 1):
        print(f"  {i:>2}. {c['label']:<{w}}  model={c['model_name']}  "
              f"lr={c['learning_rate']}  epochs={c['num_epochs']}  bs={c['batch_size']}")


def run_one_task(task_idx: int, task: dict, args, list_root: "Path") -> dict:
    """Run one task. Returns the CSV row dict (partial on failure)."""
    # Imports live inside so a failing cell can't kill later ones on import-time
    # errors (also so the GPU guard in main() runs before torch is imported).
    from utils_frame_train import (
        load_config, validate_config, resolve_device, print_config_summary,
        load_and_cap_splits, validate_frame_rate, drop_long_records,
        build_label_maps, load_feature_extractor, run_phase,
        plot_test_confusion, plot_test_example_predictions, release_gpu,
        PROJECT_ROOT,
    )

    row = {k: "" for k in CSV_FIELDS}
    row.update({
        "task_idx":       task_idx,
        "label":          task["label"],
        "model_name":     task.get("model_name", ""),
        "learning_rate":  task.get("learning_rate", ""),
        "num_epochs":     task.get("num_epochs", ""),
        "batch_size":     task.get("batch_size", ""),
    })
    t0 = time.time()
    try:
        # Base config (4_config.json + mode overrides). Cell overrides layer on top.
        cfg, _ = load_config(args.config, task_type=TASK_TYPE, run_mode=args.mode)
        overrides = {k: v for k, v in cell.items() if k != "label"}
        from dataclasses import fields as _fields
        valid = {f.name for f in _fields(cfg)}
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(f"cell {task['label']!r}: unknown Config fields {sorted(unknown)}")
        for k, v in overrides.items():
            setattr(cfg, k, v)
        validate_config(cfg)
        device = resolve_device(cfg, use_cuda=args.use_gpu)
        print_config_summary(cfg, device)

        # Data
        train_records, dev_records, test_records = load_and_cap_splits(cfg)
        for _s, _r in [("train", train_records), ("dev", dev_records), ("test", test_records)]:
            validate_frame_rate(_r, cfg.required_frame_rate_hz)
        if cfg.enable_max_audio_seconds:
            train_records, _ = drop_long_records(train_records, cfg.max_audio_seconds)
            dev_records,   _ = drop_long_records(dev_records,   cfg.max_audio_seconds)
            test_records,  _ = drop_long_records(test_records,  cfg.max_audio_seconds)
        label2id, id2label, num_labels = build_label_maps(
            cfg, train_records, dev_records, test_records)

        # List-scoped run/model dirs (all tasks sit under one list_root)
        run_dir   = list_root / task["label"]
        model_dir = list_root / task["label"] / "_model"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2, default=str))
        (run_dir / "task.json").write_text(json.dumps(task, indent=2, default=str))

        feature_extractor = load_feature_extractor(cfg)

        _, phase1_best = run_phase(
            phase_name="phase1_dev",
            train_records=train_records, eval_records=dev_records,
            eval_split_name="DEV", save_best_model=False,
            cfg=cfg, run_dir=run_dir, model_dir=model_dir,
            feature_extractor=feature_extractor,
            label2id=label2id, id2label=id2label, device=device,
        )
        _, phase2_best = run_phase(
            phase_name="phase2_test",
            train_records=train_records + dev_records, eval_records=test_records,
            eval_split_name="TEST", save_best_model=True,
            cfg=cfg, run_dir=run_dir, model_dir=model_dir,
            feature_extractor=feature_extractor,
            label2id=label2id, id2label=id2label, device=device,
        )

        # Top-level summary plots (also written per task)
        plot_test_confusion(run_dir, phase2_best, label2id, cfg.label_order, show=False)
        plot_test_example_predictions(run_dir, phase2_best, id2label,
                                      n_per_tier=cfg.n_examples_to_plot, show=False)

        row.update({
            "phase1_best_epoch": phase1_best.get("epoch"),
            "phase1_macro_f1":   phase1_best.get("eval_frame_macro_f1"),
            "phase1_f1_pos":     phase1_best.get("eval_frame_f1_positive"),
            "phase1_acc":        phase1_best.get("eval_frame_accuracy"),
            "phase1_loss":       phase1_best.get("eval_loss"),
            "phase2_best_epoch": phase2_best.get("epoch"),
            "phase2_macro_f1":   phase2_best.get("eval_frame_macro_f1"),
            "phase2_f1_pos":     phase2_best.get("eval_frame_f1_positive"),
            "phase2_acc":        phase2_best.get("eval_frame_accuracy"),
            "phase2_loss":       phase2_best.get("eval_loss"),
            "run_dir":           str(run_dir.relative_to(PROJECT_ROOT)),
        })
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        print(f"\n💥 TASK {task.get('label', '?')} FAILED:\n{traceback.format_exc()}",
              file=sys.stderr)
        try:
            release_gpu(verbose=False)
        except Exception:
            pass
    finally:
        row["wall_time_s"] = round(time.time() - t0, 1)
    return row


def make_comparison_plot(csv_path: Path, out_path: Path) -> None:
    """Horizontal bar chart of macro-F1 + positive-class F1 per task, sorted
    ascending by macro-F1. Skips failed tasks."""
    import numpy as np
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            try:
                r["_macro"] = float(r["phase2_macro_f1"])
                r["_fpos"]  = float(r["phase2_f1_pos"])
            except (TypeError, ValueError):
                continue
            rows.append(r)
    if not rows:
        print("no successful tasks to plot"); return
    rows.sort(key=lambda r: r["_macro"])
    labels = [r["label"] for r in rows]
    macro = [r["_macro"] for r in rows]
    fpos  = [r["_fpos"]  for r in rows]

    y = np.arange(len(rows))
    h = 0.4
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(rows))))
    ax.barh(y - h/2, macro, height=h, label="macro F1",   color="#4c78a8")
    ax.barh(y + h/2, fpos,  height=h, label="F1 positive", color="#f58518")
    for i, (m, p) in enumerate(zip(macro, fpos)):
        ax.text(m + 0.005, i - h/2, f"{m:.3f}", va="center", fontsize=8)
        ax.text(p + 0.005, i + h/2, f"{p:.3f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("score (best TEST epoch)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Task list — chapter-4 frame classification")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    args = parse_args()
    tasks = TASKS[: args.limit] if args.limit else TASKS
    if args.dry_run:
        print_task_plan(tasks); return

    # ── Environment — MUST precede the utils import ───────────────────────
    os.environ.setdefault("MPLBACKEND", "Agg")
    reserved_gpu = json.loads(Path(args.config).read_text())["shared"].get("reserved_gpu", "2")
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = reserved_gpu
        print(f"🚀 GPU mode — CUDA_VISIBLE_DEVICES={reserved_gpu}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"🖥️  CPU mode (pass --use_gpu to arm physical GPU {reserved_gpu})")

    import utils_frame_train as uft
    from utils_frame_train import print_project_info
    print_project_info()

    # Task-list root — toolkit-wide naming contract: {YYYY-MM-DD}_{HHMMSS}_grid
    # Timestamp first → folders sort chronologically. Colons dropped for portability.
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    list_name = args.list_dir or f"{ts}_task_list"
    list_root = uft.PROJECT_ROOT / "runs" / list_name
    list_root.mkdir(parents=True, exist_ok=True)
    print(f"task-list root : {list_root.relative_to(uft.PROJECT_ROOT)}")

    (list_root / "tasks.json").write_text(json.dumps(tasks, indent=2))
    (list_root / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    csv_path = list_root / "task_list_summary.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    print_task_plan(tasks)
    list_t0 = time.time()
    for task_idx, cell in enumerate(tasks, 1):
        header = f" TASK {task_idx}/{len(tasks)}: {task['label']} "
        print(f"\n{'='*70}\n{header:=^70}\n{'='*70}")
        row = run_one_task(task_idx, task, args, list_root)
        # Append + flush after every cell so tail -f works and a mid-sweep
        # crash still leaves partial results on disk.
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        sys.stdout.flush(); sys.stderr.flush()
        elapsed = time.time() - list_t0
        remaining = len(tasks) - task_idx
        eta = elapsed / task_idx * remaining if task_idx else 0.0
        print(f"\n📊 cell done in {row['wall_time_s']}s  |  "
              f"elapsed {int(elapsed//60)}m  |  eta {int(eta//60)}m for {remaining} left")

    # Aggregate plot
    plot_path = list_root / "task_list_summary.png"
    make_comparison_plot(csv_path, plot_path)

    total = int(time.time() - list_t0)
    print(f"\n{'='*70}")
    print(f"TASK LIST DONE in {total//60}m {total%60}s  ({len(tasks)} tasks)")
    print(f"summary CSV : {csv_path.relative_to(uft.PROJECT_ROOT)}")
    print(f"summary PNG : {plot_path.relative_to(uft.PROJECT_ROOT)}")
    print(f"per-task    : {list_root.relative_to(uft.PROJECT_ROOT)}/<cell_label>/")


if __name__ == "__main__":
    main()
