from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_method_cfg(spec: str) -> dict:
    """'path.yaml' or 'path.yaml:key' (selects one entry of a multi-method file)."""
    path, _, key = spec.partition(":")
    cfg = load_yaml(PROJECT_ROOT / path)
    return copy.deepcopy(cfg[key] if key else cfg)


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def resolve_market_cfg(cfg: dict, market: str) -> dict:
    cfg = copy.deepcopy(cfg)
    overrides = cfg.pop("market_overrides", {}) or {}
    return deep_update(cfg, overrides.get(market, {}))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
