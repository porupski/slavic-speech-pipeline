#!/usr/bin/env python3
"""45_analyze_task_list — unified summary for mixed classification + regression queues.

``43_run_task_list.py``'s baked-in summary PNG keys only on ``phase2_macro_f1``
and ``phase2_acc``, which are empty for regression cells — so a mixed queue
(e.g. Phase 4.3's phase0 classification + Stage A/B/C regression) produces a
bar chart covering only a subset of cells, or no chart at all if every cell is
regression. This analyzer reads one or more task-list run roots and emits:

  task_list_unified.csv  — one row per task across all input dirs, with a
                           ``best_metric`` column that holds macro-F1 for
                           classification cells and Spearman for regression
                           cells.
  task_list_unified.png  — side-by-side ranked bar charts (classification | regression),
                           sorted ascending by the primary metric.
  task_list_unified.md   — top-N table per task_type, plus failure list.

Usage:
  python 45_analyze_task_list.py ../runs/2026-11-21_143634_task_list
  python 45_analyze_task_list.py ../runs/2026-11-21_143634_task_list \\
                                  ../runs/2026-11-22_011427_task_list \\
                                  --out_dir ../runs/p43_combined
  python 45_analyze_task_list.py --top 10 ../runs/2026-11-21_143634_task_list

If ``--out_dir`` is not set:
  - one input dir  → files land in that dir
  - multiple inputs → files land in the first dir with a ``_plus<N>`` suffix.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


CLASSIFICATION_METRIC = "phase2_macro_f1"
CLASSIFICATION_SECONDARY = "phase2_acc"
REGRESSION_METRIC = "phase2_spearman"
REGRESSION_SECONDARY = "phase2_loss"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dirs", nargs="+",
                   help="one or more runs/<task_list>/ directories")
    p.add_argument("--out_dir", default=None,
                   help="output directory (default: first run dir, or <first>_plus<N>)")
    p.add_argument("--top", type=int, default=5,
                   help="top-N per task_type in the markdown report")
    p.add_argument("--baseline", type=float, default=0.278,
                   help="horizontal reference line on the classification panel "
                        "(default: ROG 3-cls majority-neutral)")
    p.add_argument("--ceiling", type=float, default=0.469,
                   help="horizontal reference line on the classification panel "
                        "(default: XLM-R-parlasent × R_full iso ceiling)")
    return p.parse_args()


def load_rows(run_dirs: list[Path]) -> list[dict]:
    """Walk each run_dir's task_list_summary.csv and attach a ``run_name`` for
    provenance. Returns one dict per task row across all inputs."""
    out: list[dict] = []
    for rd in run_dirs:
        csv_path = rd / "task_list_summary.csv"
        if not csv_path.exists():
            print(f"warn: {csv_path} not found — skipping", file=sys.stderr)
            continue
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                r["run_name"] = rd.name
                out.append(r)
    return out


def _f(x) -> float | None:
    """Parse a CSV cell to float, returning None for empty / 'nan' / bad values."""
    if x in (None, "", "nan", "NaN"):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:   # NaN
        return None
    return v


def build_unified(rows: list[dict]) -> list[dict]:
    """Attach ``best_metric`` and ``best_secondary`` per row — Spearman for
    regression, macro-F1 for classification. Failed rows get None."""
    for r in rows:
        if r.get("error"):
            r["best_metric"] = None
            r["best_secondary"] = None
            r["metric_name"] = ""
            continue
        if r.get("task_type") == "regression":
            r["best_metric"] = _f(r.get(REGRESSION_METRIC))
            r["best_secondary"] = _f(r.get(REGRESSION_SECONDARY))
            r["metric_name"] = "spearman"
        else:
            r["best_metric"] = _f(r.get(CLASSIFICATION_METRIC))
            r["best_secondary"] = _f(r.get(CLASSIFICATION_SECONDARY))
            r["metric_name"] = "macro_f1"
    return rows


def write_unified_csv(rows: list[dict], out_path: Path) -> None:
    cols = [
        "run_name", "task_idx", "label", "task_type", "model_name", "label_key",
        "learning_rate", "num_epochs", "batch_size", "wall_time_s",
        "metric_name", "best_metric", "best_secondary",
        "phase2_macro_f1", "phase2_acc", "phase2_spearman", "phase2_loss",
        "phase1_macro_f1", "phase1_acc", "phase1_spearman", "phase1_loss",
        "run_dir", "error",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_path}")


def _plot_panel(ax, rows: list[dict], metric_label: str, secondary_label: str,
                baseline: float | None, ceiling: float | None,
                title: str) -> None:
    """Horizontal bar chart on one axis. `rows` ordered ascending by best_metric."""
    import numpy as np
    labels = [r["label"] for r in rows]
    primary = [r["best_metric"] for r in rows]
    secondary = [r["best_secondary"] if r["best_secondary"] is not None else 0 for r in rows]

    y = np.arange(len(rows))
    h = 0.4
    ax.barh(y - h / 2, primary, height=h, label=metric_label, color="#4c78a8")
    ax.barh(y + h / 2, secondary, height=h, label=secondary_label, color="#f58518")
    for i, (m, s) in enumerate(zip(primary, secondary)):
        if m is not None:
            ax.text(m + 0.005, i - h / 2, f"{m:.3f}", va="center", fontsize=8)
        if s is not None:
            ax.text(s + 0.005, i + h / 2, f"{s:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    if baseline is not None:
        ax.axvline(baseline, color="gray", linestyle=":", linewidth=1,
                   label=f"baseline {baseline:.3f}")
    if ceiling is not None:
        ax.axvline(ceiling, color="green", linestyle=":", linewidth=1,
                   label=f"ceiling {ceiling:.3f}")
    ax.set_xlabel("score")
    ax.set_xlim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)


def plot_unified(rows: list[dict], out_path: Path,
                 baseline: float | None, ceiling: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = [r for r in rows if r["best_metric"] is not None]
    cls_rows = sorted([r for r in good if r["task_type"] == "classification"],
                      key=lambda r: r["best_metric"])
    regr_rows = sorted([r for r in good if r["task_type"] == "regression"],
                       key=lambda r: r["best_metric"])
    n_cls, n_regr = len(cls_rows), len(regr_rows)

    if not cls_rows and not regr_rows:
        print("no successful rows to plot")
        return

    # Panel widths/heights proportional to row counts; blank side if empty.
    panels = (1 if cls_rows else 0) + (1 if regr_rows else 0)
    fig_w = 14 if panels == 2 else 10
    fig_h = max(4, 0.55 * max(n_cls, n_regr, 1))
    fig, axes = plt.subplots(1, panels, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    panel_idx = 0
    if cls_rows:
        _plot_panel(axes[panel_idx], cls_rows,
                    metric_label="macro F1", secondary_label="accuracy",
                    baseline=baseline, ceiling=ceiling,
                    title=f"Classification (n={n_cls})")
        panel_idx += 1
    if regr_rows:
        _plot_panel(axes[panel_idx], regr_rows,
                    metric_label="Spearman ρ", secondary_label="loss",
                    baseline=None, ceiling=None,
                    title=f"Regression (n={n_regr})")

    fig.suptitle(f"Task list — unified summary ({n_cls + n_regr} cells, "
                 f"best phase-2 epoch)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _fmt_row(r: dict) -> str:
    bm = f"{r['best_metric']:.3f}" if r['best_metric'] is not None else "—"
    bs = f"{r['best_secondary']:.3f}" if r['best_secondary'] is not None else "—"
    wt = r.get("wall_time_s", "")
    try:
        wt = f"{float(wt):.0f}s"
    except (TypeError, ValueError):
        wt = str(wt)
    return (f"| {r['label']} | {r.get('model_name', '')} | {r.get('target', '')} | "
            f"{bm} | {bs} | {wt} |")


def write_markdown(rows: list[dict], out_path: Path, top: int) -> None:
    cls_rows = sorted([r for r in rows if r["task_type"] == "classification"
                       and r["best_metric"] is not None],
                      key=lambda r: r["best_metric"], reverse=True)
    regr_rows = sorted([r for r in rows if r["task_type"] == "regression"
                        and r["best_metric"] is not None],
                       key=lambda r: r["best_metric"], reverse=True)
    failed = [r for r in rows if r.get("error")]

    lines = ["# Task list — unified summary", ""]
    lines.append(f"Total tasks: {len(rows)}   "
                 f"(classification {sum(1 for r in rows if r['task_type']=='classification')}, "
                 f"regression {sum(1 for r in rows if r['task_type']=='regression')}, "
                 f"failed {len(failed)})")
    lines.append("")

    if cls_rows:
        lines += [f"## Classification — top {min(top, len(cls_rows))} (macro F1)", "",
                  "| label | model | target | macro_f1 | accuracy | wall |",
                  "|---|---|---|---:|---:|---:|"]
        for r in cls_rows[:top]:
            lines.append(_fmt_row(r))
        lines.append("")
    if regr_rows:
        lines += [f"## Regression — top {min(top, len(regr_rows))} (Spearman ρ)", "",
                  "| label | model | target | spearman | loss | wall |",
                  "|---|---|---|---:|---:|---:|"]
        for r in regr_rows[:top]:
            lines.append(_fmt_row(r))
        lines.append("")
    if failed:
        lines += ["## Failed tasks", "",
                  "| label | model | target | error |",
                  "|---|---|---|---|"]
        for r in failed:
            lines.append(f"| {r['label']} | {r.get('model_name','')} | "
                         f"{r.get('target','')} | {r.get('error','')[:120]} |")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def default_out_dir(run_dirs: list[Path]) -> Path:
    if len(run_dirs) == 1:
        return run_dirs[0]
    return run_dirs[0].with_name(f"{run_dirs[0].name}_plus{len(run_dirs) - 1}")


def main() -> None:
    args = parse_args()
    run_dirs = [Path(d).resolve() for d in args.run_dirs]
    missing = [d for d in run_dirs if not d.exists()]
    if missing:
        print(f"ERROR: missing run dir(s): {missing}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir(run_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(run_dirs)
    if not rows:
        print("no rows loaded", file=sys.stderr); sys.exit(1)
    rows = build_unified(rows)

    write_unified_csv(rows, out_dir / "task_list_unified.csv")
    plot_unified(rows, out_dir / "task_list_unified.png",
                 baseline=args.baseline, ceiling=args.ceiling)
    write_markdown(rows, out_dir / "task_list_unified.md", args.top)

    # Console summary
    print("\n── console summary ──")
    good = [r for r in rows if r["best_metric"] is not None]
    for tt in ("classification", "regression"):
        subset = sorted([r for r in good if r["task_type"] == tt],
                        key=lambda r: r["best_metric"], reverse=True)
        if not subset:
            continue
        print(f"\n{tt} — top 5:")
        for r in subset[:5]:
            print(f"  {r['best_metric']:.3f}  {r['label']}")


if __name__ == "__main__":
    main()
