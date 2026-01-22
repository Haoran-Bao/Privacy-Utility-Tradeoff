import argparse
import os
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train.runner import run_experiment
from src.utils.config import apply_overrides, load_config


def _apply_special_overrides(cfg: Dict, overrides: List[str]) -> None:
    for item in overrides:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        if key == "eps":
            cfg.setdefault("dp", {})
            cfg["dp"]["privacy_mode"] = "target_epsilon"
            cfg["dp"]["target_epsilon"] = float(raw)
            cfg.pop("eps", None)


def run_with_config(cfg: Dict) -> str:
    return run_experiment(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DP experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("overrides", nargs="*", help="Override key=value pairs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    _apply_special_overrides(cfg, args.overrides)

    cfg["project_root"] = PROJECT_ROOT

    run_dir = run_with_config(cfg)
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
