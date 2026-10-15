"""Method name -> class. Register new methods (e.g. methods/proposed) here."""
from __future__ import annotations

from regime_alloc.methods.base import Method
from regime_alloc.methods.classic.methods import CRP, MOM, MVO


def _stable():
    from regime_alloc.methods.stable.method import Stable
    return Stable


REGISTRY = {
    "crp": lambda: CRP,
    "mvo": lambda: MVO,
    "mom": lambda: MOM,
    "stable": _stable,
}


def build_method(cfg: dict, market: str, **kw) -> Method:
    return REGISTRY[cfg["method"]]()(cfg, market, **kw)


def method_label(cfg: dict) -> str:
    """Directory/table name; ablated STABLE variants get a suffix."""
    name = cfg.get("label", cfg["method"])
    ab = cfg.get("ablation", {})
    off = [k.replace("use_", "") for k, v in ab.items() if v is False]
    return name + ("-no_" + "_".join(off) if off else "")
