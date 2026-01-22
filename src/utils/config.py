import os
from typing import Any, Dict, Iterable

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.pop("base_config", None)
    if base_path:
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)
        base_cfg = load_config(base_path)
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def _set_by_path(cfg: Dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        _set_by_path(cfg, key, value)
    return cfg
