#!/usr/bin/env python3
"""run_62_instance_regression_inference — SCAFFOLD, 2026-11-02.

CLI twin of 62_instance_regression_inference-WIP.ipynb. Not yet wired.
The runner shape matches run_63_frame_classification_inference.py; it will
dispatch into utils_instance_infer.load_config once the engine is ported.

See the port plan at the top of utils_instance_infer.py.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", default=str(HERE / "config.json"))
    p.add_argument("--task", "-t", required=True,
                   help="task key under tasks.json/instance_regression")
    p.add_argument("--mode", "-m", choices=["test", "demo", "full"], default="full")
    p.add_argument("--run_name", "-r", default=None)
    p.add_argument("--audio_dir", default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--use_gpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import utils_instance_infer as uii
    uii.load_config(args.config, run_mode=args.mode, task=args.task,
                    quadrant="instance_regression")


if __name__ == "__main__":
    main()
