import argparse
import copy
import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train.runner import run_experiment
from src.utils.config import deep_merge, load_config


def _prepare_cfg(base_cfg, method_cfg):
    if base_cfg:
        return deep_merge(base_cfg, method_cfg)
    return method_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sweep over epsilons/seeds/methods")
    parser.add_argument("--config", required=True, help="Path to sweep YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        sweep_cfg = yaml.safe_load(f) or {}
    base_cfg_path = sweep_cfg.get("base_config")
    if base_cfg_path and not os.path.isabs(base_cfg_path):
        base_cfg_path = os.path.join(os.path.dirname(args.config), base_cfg_path)
    base_cfg = load_config(base_cfg_path) if base_cfg_path else {}

    methods = sweep_cfg.get("methods", [])
    seeds = sweep_cfg.get("seeds", [])
    epsilons = sweep_cfg.get("epsilons", [])
    method_cfgs = sweep_cfg.get("method_configs", {})
    dp_sam_noise = sweep_cfg.get("dp_sam_noise_multipliers", [])
    dp_sam_privacy_mode = sweep_cfg.get("dp_sam_privacy_mode", "fixed_noise")

    for method in methods:
        method_cfg_path = method_cfgs.get(method)
        if method_cfg_path and not os.path.isabs(method_cfg_path):
            method_cfg_path = os.path.join(os.path.dirname(args.config), method_cfg_path)
        method_cfg = load_config(method_cfg_path) if method_cfg_path else {}
        base = copy.deepcopy(base_cfg)
        cfg_base = _prepare_cfg(base, method_cfg)

        for eps_idx, eps in enumerate(epsilons):
            for seed in seeds:
                cfg = copy.deepcopy(cfg_base)
                cfg["method"] = method
                cfg["seed"] = seed
                cfg["project_root"] = PROJECT_ROOT

                cfg.setdefault("dp", {})
                if method == "dp_sgd":
                    cfg["dp"]["privacy_mode"] = "target_epsilon"
                    cfg["dp"]["target_epsilon"] = float(eps)
                else:
                    if dp_sam_privacy_mode == "target_epsilon":
                        cfg["dp"]["privacy_mode"] = "target_epsilon"
                        cfg["dp"]["target_epsilon"] = float(eps)
                    else:
                        cfg["dp"]["privacy_mode"] = "fixed_noise"
                        if dp_sam_noise and eps_idx < len(dp_sam_noise):
                            cfg["dp"]["noise_multiplier"] = float(dp_sam_noise[eps_idx])

                run_experiment(cfg)


if __name__ == "__main__":
    main()
