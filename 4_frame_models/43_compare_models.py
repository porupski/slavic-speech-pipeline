#!/usr/bin/env python3
"""43_compare_models — chapter-4 model + hyperparameter sweep.

Runs every cell in ``CELLS`` below through the same engine as
``run_41_classification`` (one full phase1_dev → phase2_test per cell),
aggregates the best DEV + best TEST metrics into a CSV, plots a bar chart
comparing macro-F1 and positive-class F1 across cells, and lets you
fire-and-forget overnight. A crash in one cell does not kill the sweep — the
error is written into that cell's CSV row and the next cell runs.

Every cell writes its own ``<grid_root>/<label>/`` tree with:
- per-epoch predictions.json / epoch_summary.json / example_predictions.png
- top-level confusion_matrix_test.png + example_predictions_test.png
- effective config snapshot
- best_model/ (weights only for the best phase-2 epoch)

Aggregate summary lands at ``<grid_root>/grid_summary.csv`` +
``<grid_root>/grid_summary.png``. ``cells.json`` snapshots the exact CELLS
list for provenance.

Usage:
  python 43_compare_models.py --use_gpu
  python 43_compare_models.py --use_gpu --dry_run          # print cells, exit
  python 43_compare_models.py --use_gpu --limit 3          # first 3 cells only
  python 43_compare_models.py --use_gpu --grid_dir grid_lr_sweep

Base config is `config.json` under the chosen ``--mode`` (default: full);
per-cell overrides layer on top.
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
# GRID — edit this list to change the sweep
# ══════════════════════════════════════════════════════════════════════════
# Each cell is a dict of Config field overrides. Every key must name a real
# Config field in utils_frame_train.py (typos are a hard error). The `label`
# is not a Config field; it names the cell's sub-directory and CSV row.
#
# Baseline (facebook/w2v-bert-2.0) got macro-F1 ≈ 0.94 / F1+ ≈ 0.90 at
# lr=1e-5, bs=64, 15 epochs. The sweep below varies LR around that anchor
# for the vanilla body and probes the CLASSLA HR/RS-finetuned head at
# finetune-scale LRs.

CELLS: list[dict] = [
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
     "learning_rate": 1e-5, "num_epochs": 15, "batch_size": 64},
    {"label": "classla_lr3e5", "model_name": "classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier",
     "learning_rate": 3e-5, "num_epochs": 15, "batch_size": 64},

    # ── longer runs, best LR per family ──────────────────────────────────
    {"label": "bert_vanilla_lr1e5_30ep", "model_name": "facebook/w2v-bert-2.0",
     "learning_rate": 1e-5, "num_epochs": 30, "batch_size": 64},
    {"label": "classla_lr1e5_25ep", "model_name": "classla/Wav2Vec2BertPrimaryStressAudioFrameClassifier",
     "learning_rate": 1e-5, "num_epochs": 25, "batch_size": 64},
]


CSV_FIELDS = [
    "cell_idx", "label", "model_name", "learning_rate", "num_epochs", "batch_size",
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
    p.add_argument("--config", "-c", default=str(HERE / "config.json"),
                   help="path to config.json (base layer)")
    p.add_argument("--mode", "-m", choices=["test", "demo", "full"], default="full",
                   help="base run_mode; cell overrides layer on top")
    p.add_argument("--use_gpu", action="store_true",
                   help="arm the reserved GPU (no prompt); default is CPU")
    p.add_argument("--dry_run", action="store_true",
                   help="print the planned cells + exit")
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N cells (for smoke tests)")
    p.add_argument("--grid_dir", default=None,
                   help="grid sub-directory name under runs/ (default: grid_<timestamp>)")
    return p.parse_args()


def print_cell_plan(cells: list[dict]) -> None:
    w = max(len(c["label"]) for c in cells) if cells else 5
    print(f"planned {len(cells)} cells:")
    for i, c in enumerate(cells, 1):
        print(f"  {i:>2}. {c['label']:<{w}}  model={c['model_name']}  "
              f"lr={c['learning_rate']}  epochs={c['num_epochs']}  bs={c['batch_size']}")


def run_one_cell(cell_idx: int, cell: dict, args, grid_root: "Path") -> dict:
    """Run one grid cell. Returns the CSV row dict (partial on failure)."""
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
        "cell_idx":       cell_idx,
        "label":          cell["label"],
        "model_name":     cell.get("model_name", ""),
        "learning_rate":  cell.get("learning_rate", ""),
        "num_epochs":     cell.get("num_epochs", ""),
        "batch_size":     cell.get("batch_size", ""),
    })
    t0 = time.time()
    try:
        # Base config (config.json + mode overrides). Cell overrides layer on top.
        cfg, _ = load_config(args.config, task_type=TASK_TYPE, run_mode=args.mode)
        overrides = {k: v for k, v in cell.items() if k != "label"}
        from dataclasses import fields as _fields
        valid = {f.name for f in _fields(cfg)}
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(f"cell {cell['label']!r}: unknown Config fields {sorted(unknown)}")
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

        # Grid-scoped run/model dirs (all cells sit under one grid_root)
        run_dir   = grid_root / cell["label"]
        model_dir = grid_root / cell["label"] / "_model"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(asdict(cfg), indent=2, default=str))
        (run_dir / "cell.json").write_text(json.dumps(cell, indent=2, default=str))

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

        # Top-level summary plots (also written per cell)
        plot_test_confusion(run_dir, phase2_best, label2id, cfg.label_order, show=False)
        plot_test_example_predictions(run_dir, phase2_best, id2label,
                                      n_examples=cfg.n_examples_to_plot, show=False)

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
        print(f"\n💥 CELL {cell.get('label', '?')} FAILED:\n{traceback.format_exc()}",
              file=sys.stderr)
        try:
            release_gpu(verbose=False)
        except Exception:
            pass
    finally:
        row["wall_time_s"] = round(time.time() - t0, 1)
    return row


def make_comparison_plot(csv_path: Path, out_path: Path) -> None:
    """Horizontal bar chart of macro-F1 + positive-class F1 per cell, sorted
    ascending by macro-F1. Skips failed cells."""
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
        print("no successful cells to plot"); return
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
    ax.set_title("Grid comparison — chapter-4 frame classification")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    args = parse_args()
    cells = CELLS[: args.limit] if args.limit else CELLS
    if args.dry_run:
        print_cell_plan(cells); return

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

    # Grid root
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    grid_name = args.grid_dir or f"grid_{ts}"
    grid_root = uft.PROJECT_ROOT / "runs" / grid_name
    grid_root.mkdir(parents=True, exist_ok=True)
    print(f"grid root : {grid_root.relative_to(uft.PROJECT_ROOT)}")

    (grid_root / "cells.json").write_text(json.dumps(cells, indent=2))
    (grid_root / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    csv_path = grid_root / "grid_summary.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    print_cell_plan(cells)
    grid_t0 = time.time()
    for cell_idx, cell in enumerate(cells, 1):
        header = f" CELL {cell_idx}/{len(cells)}: {cell['label']} "
        print(f"\n{'='*70}\n{header:=^70}\n{'='*70}")
        row = run_one_cell(cell_idx, cell, args, grid_root)
        # Append + flush after every cell so tail -f works and a mid-sweep
        # crash still leaves partial results on disk.
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        sys.stdout.flush(); sys.stderr.flush()
        elapsed = time.time() - grid_t0
        remaining = len(cells) - cell_idx
        eta = elapsed / cell_idx * remaining if cell_idx else 0.0
        print(f"\n📊 cell done in {row['wall_time_s']}s  |  "
              f"elapsed {int(elapsed//60)}m  |  eta {int(eta//60)}m for {remaining} left")

    # Aggregate plot
    plot_path = grid_root / "grid_summary.png"
    make_comparison_plot(csv_path, plot_path)

    total = int(time.time() - grid_t0)
    print(f"\n{'='*70}")
    print(f"GRID DONE in {total//60}m {total%60}s  ({len(cells)} cells)")
    print(f"summary CSV : {csv_path.relative_to(uft.PROJECT_ROOT)}")
    print(f"summary PNG : {plot_path.relative_to(uft.PROJECT_ROOT)}")
    print(f"per-cell    : {grid_root.relative_to(uft.PROJECT_ROOT)}/<cell_label>/")


if __name__ == "__main__":
    main()
