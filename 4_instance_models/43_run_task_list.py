#!/usr/bin/env python3
"""43_run_task_list — chapter-4 sequential task-list runner.

Walks the ``TASKS`` list below one by one. Each task is a full
phase-1 (DEV) → phase-2 (TEST) instance-classification/regression run through
the same engine as ``run_41_classification`` / ``run_42_regression``. A crash
in one task does not kill the queue — the error lands in that task's CSV row
and the next task runs. All stdout + stderr is teed into each run folder's
``run.log``, so tmux it and walk away.

Every task writes its own ``<list_root>/<label>/`` tree with:
- per-epoch predictions.json / epoch_summary.json
- top-level confusion_matrix_test.png (classification) or scatter_test.png (regression)
- effective config snapshot
- ``best_model/`` (weights for the best phase-2 epoch)

Aggregate summary lands at ``<list_root>/task_list_summary.csv`` +
``<list_root>/task_list_summary.png``. ``tasks.json`` snapshots the exact
TASKS list for provenance.

Usage:
  python 43_run_task_list.py --use_gpu
  python 43_run_task_list.py --use_gpu --dry_run              # print tasks, exit
  python 43_run_task_list.py --use_gpu --limit 2              # first 2 only
  python 43_run_task_list.py --use_gpu --list_dir my_run_set

  python 43_run_task_list.py --use_gpu 2>&1 | tee ../runs/43_task_list_$(date +%Y-%m-%d_%H%M%S).log

Base config is ``4_config.json`` under the chosen ``--mode`` (default: demo);
per-task overrides layer on top.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, fields as _dc_fields
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ══════════════════════════════════════════════════════════════════════════
# TASKS — edit this list to change the queue
# ══════════════════════════════════════════════════════════════════════════
# Each task is a dict of Config field overrides plus a `label` (sub-dir name,
# CSV row key) and an optional `task_type` ("classification" or "regression").
# task_type selects which config block to load; the remaining keys must match
# real Config fields in utils_instance_train.py (typos are a hard error).

TASKS: list[dict] = [
    # Pick a `target` from the TARGETS registry (see available_targets() in
    # utils_instance_train.py) — this resolves jsonl_path, label_key and
    # label_order in one step. Overriding `label_key` alone is wrong: the
    # jsonl_path stays at whatever the config block pointed at and the split
    # comes back empty.
    #
    # Four-way comparison on ParlaSpeech-HR benchmark v3 (demo = 10 epochs).
    # w2v2-base vs xls-r-300m, gender vs speaker-ID. Bigger model gets the
    # higher LR it needs. One age regression closes the queue.
    {"label": "gender_w2v2-base_lr1e5",
     "model_name":    "facebook/wav2vec2-base",
     "target":        "hr_bench_v3_gender",
     "task_type":     "classification",
     "num_epochs": 10,
     "learning_rate": 1e-5},
    {"label": "gender_xls-r-300m_lr3e5",
     "model_name":    "facebook/wav2vec2-xls-r-300m",
     "target":        "hr_bench_v3_gender",
     "task_type":     "classification",
     "num_epochs": 10,
     "learning_rate": 3e-5},
    {"label": "speakerid_w2v2-base_lr1e5",
     "model_name":    "facebook/wav2vec2-base",
     "target":        "hr_bench_v3_speaker_id",
     "task_type":     "classification",
     "num_epochs": 10,     
     "learning_rate": 1e-5},
    {"label": "speakerid_xls-r-300m_lr3e5",
     "model_name":    "facebook/wav2vec2-xls-r-300m",
     "target":        "hr_bench_v3_speaker_id",
     "task_type":     "classification",
     "num_epochs": 10,
     "learning_rate": 3e-5},
    {"label": "age_w2v2-base_lr1e5",
     "model_name":    "facebook/wav2vec2-base",
     "target":        "hr_bench_v3_age",
     "task_type":     "regression",
     "num_epochs": 20,
     "learning_rate": 1e-5},
    {"label": "age_xls-r-300m_lr3e5",
     "model_name":    "facebook/wav2vec2-xls-r-300m",
     "target":        "hr_bench_v3_age",
     "task_type":     "regression",
     "num_epochs": 20,
     "learning_rate": 3e-5},
    {"label": "orientation_w2v2-base_lr1e5",
     "model_name":    "facebook/wav2vec2-base",
     "target":        "hr_bench_v3_orientation",
     "task_type":     "regression",
     "num_epochs": 20,
     "learning_rate": 1e-5},
    {"label": "orientation_xls-r-300m_lr3e5",
     "model_name":    "facebook/wav2vec2-xls-r-300m",
     "target":        "hr_bench_v3_orientation",
     "task_type":     "regression",
     "num_epochs": 20,
     "learning_rate": 3e-5},
]


CSV_FIELDS = [
    "task_idx", "label", "task_type", "model_name", "label_key",
    "learning_rate", "num_epochs", "batch_size", "wall_time_s",
    "phase1_best_epoch", "phase1_macro_f1", "phase1_acc", "phase1_spearman", "phase1_loss",
    "phase2_best_epoch", "phase2_macro_f1", "phase2_acc", "phase2_spearman", "phase2_loss",
    "run_dir", "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", default=str(HERE / "4_config.json"),
                   help="path to the chapter-4 4_config.json (base layer)")
    p.add_argument("--mode", "-m", choices=["test", "demo", "full"], default="demo",
                   help="base run_mode; per-task overrides layer on top")
    p.add_argument("--use_gpu", action="store_true",
                   help="arm the reserved GPU (no prompt); default is CPU")
    p.add_argument("--dry_run", action="store_true",
                   help="print the planned tasks + exit")
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N tasks (for smoke tests)")
    p.add_argument("--list_dir", default=None,
                   help="task-list sub-directory under runs/ (default: <timestamp>_task_list)")
    return p.parse_args()


def print_task_plan(tasks: list[dict]) -> None:
    w = max(len(t["label"]) for t in tasks) if tasks else 5
    print(f"planned {len(tasks)} tasks:")
    for i, t in enumerate(tasks, 1):
        extras = " ".join(f"{k}={v}" for k, v in t.items()
                          if k not in {"label", "model_name"})
        print(f"  {i:>2}. {t['label']:<{w}}  model={t.get('model_name', '-')}  {extras}")


def run_one_task(task_idx: int, task: dict, args, list_root: "Path") -> dict:
    """Run one task. Returns the CSV row dict (partial on failure)."""
    # Imports live inside so a failing task can't kill the rest of the queue
    # on an import-time error (also so the GPU guard in main() runs before
    # torch is imported).
    from utils_instance_train import (
        load_config, validate_config, resolve_device, print_config_summary,
        load_and_cap_splits, build_label_maps, load_feature_extractor,
        run_phase, plot_test_confusion, plot_test_scatter, spot_check,
        release_gpu, fit_normalizer, resolve_target, TARGETS, PROJECT_ROOT,
    )

    row = {k: "" for k in CSV_FIELDS}
    row.update({
        "task_idx":      task_idx,
        "label":         task["label"],
        "task_type":     task.get("task_type", "classification"),
        "model_name":    task.get("model_name", ""),
        "label_key":     task.get("label_key", ""),
        "learning_rate": task.get("learning_rate", ""),
    })
    t0 = time.time()
    task_type = task.get("task_type", "classification")

    try:
        # Base config (4_config.json + mode overrides). Per-task overrides layer on top.
        cfg, _ = load_config(args.config, task_type=task_type, run_mode=args.mode)
        overrides = {k: v for k, v in task.items() if k not in {"label", "task_type"}}
        valid = {f.name for f in _dc_fields(cfg)}
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(f"task {task['label']!r}: unknown Config fields {sorted(unknown)}")
        for k, v in overrides.items():
            setattr(cfg, k, v)
        # If the task overrode `target`, resolve it again so jsonl_path, label_key
        # and label_order follow — otherwise the cfg still points at the previous
        # task's JSONL and splits come back empty.
        if "target" in overrides:
            resolve_target(cfg, TARGETS)
            if cfg.task_type != task_type:
                raise ValueError(
                    f"task {task['label']!r}: target {cfg.target!r} is a "
                    f"{cfg.task_type} preset, but task_type is {task_type!r}"
                )
            row["label_key"] = cfg.label_key
        validate_config(cfg)
        device = resolve_device(cfg, use_cuda=args.use_gpu)

        # Mirror the final effective knobs into the CSV row (post-override).
        row["num_epochs"] = getattr(cfg, "num_epochs", "")
        row["batch_size"] = getattr(cfg, "batch_size", "")

        print_config_summary(cfg, device)

        # Data
        train_records, dev_records, test_records = load_and_cap_splits(cfg)
        label2id, id2label, num_labels = build_label_maps(
            cfg, train_records, dev_records, test_records)
        normalizer = (fit_normalizer(cfg, train_records, dev_records, test_records)
                      if task_type == "regression" else None)

        # List-scoped run/model dirs (all tasks sit under one list_root).
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
            normalizer=normalizer,
        )
        _, phase2_best = run_phase(
            phase_name="phase2_test",
            train_records=train_records + dev_records, eval_records=test_records,
            eval_split_name="TEST", save_best_model=True,
            cfg=cfg, run_dir=run_dir, model_dir=model_dir,
            feature_extractor=feature_extractor,
            label2id=label2id, id2label=id2label, device=device,
            normalizer=normalizer,
        )

        # Top-level summary artefacts
        if task_type == "classification":
            plot_test_confusion(run_dir, phase2_best, label2id, cfg.label_order, show=False)
        else:
            plot_test_scatter(run_dir, phase2_best, show=False)
        spot_check(run_dir, phase2_best, task_type)

        row.update({
            "phase1_best_epoch": phase1_best.get("epoch"),
            "phase1_macro_f1":   phase1_best.get("eval_macro_f1"),
            "phase1_acc":        phase1_best.get("eval_accuracy"),
            "phase1_spearman":   phase1_best.get("eval_spearman"),
            "phase1_loss":       phase1_best.get("eval_loss"),
            "phase2_best_epoch": phase2_best.get("epoch"),
            "phase2_macro_f1":   phase2_best.get("eval_macro_f1"),
            "phase2_acc":        phase2_best.get("eval_accuracy"),
            "phase2_spearman":   phase2_best.get("eval_spearman"),
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
    """Horizontal bar chart of macro-F1 + accuracy per task, sorted ascending
    by macro-F1. Skips failed tasks."""
    import numpy as np
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            try:
                r["_macro"] = float(r["phase2_macro_f1"])
                r["_acc"]   = float(r["phase2_acc"])
            except (TypeError, ValueError):
                continue
            rows.append(r)
    if not rows:
        print("no successful tasks to plot"); return
    rows.sort(key=lambda r: r["_macro"])
    labels = [r["label"] for r in rows]
    macro  = [r["_macro"] for r in rows]
    acc    = [r["_acc"]   for r in rows]

    y = np.arange(len(rows))
    h = 0.4
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(rows))))
    ax.barh(y - h/2, macro, height=h, label="macro F1", color="#4c78a8")
    ax.barh(y + h/2, acc,   height=h, label="accuracy", color="#f58518")
    for i, (m, a) in enumerate(zip(macro, acc)):
        ax.text(m + 0.005, i - h/2, f"{m:.3f}", va="center", fontsize=8)
        ax.text(a + 0.005, i + h/2, f"{a:.3f}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("score (best TEST epoch)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Task list — chapter-4 instance classification")
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

    import utils_instance_train as uit
    from utils_instance_train import print_project_info
    print_project_info()

    # List root — toolkit-wide naming contract: {YYYY-MM-DD}_{HHMMSS}_task_list
    # Timestamp first → folders sort chronologically. Colons dropped for portability.
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    list_name = args.list_dir or f"{ts}_task_list"
    list_root = uit.PROJECT_ROOT / "runs" / list_name
    list_root.mkdir(parents=True, exist_ok=True)
    print(f"task-list root : {list_root.relative_to(uit.PROJECT_ROOT)}")

    (list_root / "tasks.json").write_text(json.dumps(tasks, indent=2))
    (list_root / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    csv_path = list_root / "task_list_summary.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    print_task_plan(tasks)
    list_t0 = time.time()
    for task_idx, task in enumerate(tasks, 1):
        header = f" TASK {task_idx}/{len(tasks)}: {task['label']} "
        print(f"\n{'='*70}\n{header:=^70}\n{'='*70}")
        row = run_one_task(task_idx, task, args, list_root)
        # Append + flush after every task so tail -f works and a mid-list crash
        # still leaves partial results on disk.
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        sys.stdout.flush(); sys.stderr.flush()
        elapsed = time.time() - list_t0
        remaining = len(tasks) - task_idx
        eta = elapsed / task_idx * remaining if task_idx else 0.0
        print(f"\n📊 task done in {row['wall_time_s']}s  |  "
              f"elapsed {int(elapsed//60)}m  |  eta {int(eta//60)}m for {remaining} left")

    plot_path = list_root / "task_list_summary.png"
    make_comparison_plot(csv_path, plot_path)

    total = int(time.time() - list_t0)
    print(f"\n{'='*70}")
    print(f"TASK LIST DONE in {total//60}m {total%60}s  ({len(tasks)} tasks)")
    print(f"summary CSV : {csv_path.relative_to(uit.PROJECT_ROOT)}")
    print(f"summary PNG : {plot_path.relative_to(uit.PROJECT_ROOT)}")
    print(f"per-task    : {list_root.relative_to(uit.PROJECT_ROOT)}/<task_label>/")


if __name__ == "__main__":
    main()
