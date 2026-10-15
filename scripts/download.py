"""Download raw prices: python scripts/download.py [--market all|sp500|csi300|eurostoxx|kospi200]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regime_alloc.data.download import download_macro, download_market  # noqa: E402
from regime_alloc.utils import PROJECT_ROOT, load_yaml  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--market", default="all")
args = p.parse_args()
cfg = load_yaml(PROJECT_ROOT / "configs/data/universes.yaml")
download_macro(cfg)
for m in (cfg["markets"] if args.market == "all" else [args.market]):
    download_market(m, cfg)
