#!/usr/bin/env python3
"""run_63_frame_classification_inference — frame-classification inference runner (chapter 6).

The command-line twin of 63_frame_classification_inference.ipynb — same engine
(utils_frame_infer.py), same config.json, no interactive prompts. This is the
tool for FULL inference runs: launch it in tmux and walk away.

Usage:
  python run_63_frame_classification_inference.py --run_name my_run
  python run_63_frame_classification_inference.py -m demo -r fp_demo --use_gpu
  python run_63_frame_classification_inference.py -m full -r fp_full --use_gpu --audio_dir data/inference_input/parliamentary

Device policy: CPU by default (safe, prompt-free). --use_gpu arms the reserved
GPU from config.json (shared.reserved_gpu) — the explicit, non-interactive
replacement for the notebook's input-gated GPU guard. We NEVER touch another GPU.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # utils_frame_infer.py lives next to this runner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", default=str(HERE / "config.json"),
                   help="path to config.json")
    p.add_argument("--mode", "-m", choices=["test", "demo", "full"], default="full",
                   help="run_mode; caps cap_files")
    p.add_argument("--run_name", "-r", default=None,
                   help="name of the output run folder under runs/. "
                        "Omit → auto-generate from cfg (Ch3-style).")
    p.add_argument("--audio_dir", default=None,
                   help="override cfg.audio_dir (relative to project root or absolute)")
    p.add_argument("--model_name", default=None,
                   help="override cfg.model_name (HF repo id or local checkpoint dir)")
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

    import utils_frame_infer as ufi
    from utils_frame_infer import (
        load_config, print_config_summary, resolve_device,
        run_inference, sample_plot_examples, N_PLOT_EXAMPLES,
        print_project_info,
    )
    print_project_info()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config, run_mode=args.mode)
    cfg.use_cuda = args.use_gpu
    if args.audio_dir is not None:
        cfg.audio_dir = args.audio_dir
    if args.model_name is not None:
        cfg.model_name = args.model_name

    device = resolve_device(cfg)
    if cfg.run_mode == "full" and device == "cpu":
        print("⚠️  full inference run on CPU — this will be slow. "
              "Did you forget --use_gpu?")
    print_config_summary(cfg, device)

    # ── Run ───────────────────────────────────────────────────────────────────
    result = run_inference(cfg, run_name=args.run_name)

    # ── Sample plots ──────────────────────────────────────────────────────────
    sample_plot_examples(result["out_jsonl"], result["run_dir"],
                         n=N_PLOT_EXAMPLES, show=False)


if __name__ == "__main__":
    main()
