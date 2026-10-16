"""Train and/or evaluate one method on one market.

python scripts/run_method.py --market sp500 --method configs/methods/stable.yaml \
    --split configs/splits/paper_main.yaml --exp paper_table3_4 [--skip-train] [--epochs N]
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regime_alloc.data.dataset import build_market_data, load_split  # noqa: E402
from regime_alloc.eval.backtest import run_backtest  # noqa: E402
from regime_alloc.methods.registry import build_method, method_label  # noqa: E402
from regime_alloc.utils import (PROJECT_ROOT, deep_update, load_method_cfg,  # noqa: E402
                                resolve_market_cfg)

p = argparse.ArgumentParser()
p.add_argument("--market", required=True)
p.add_argument("--method", required=True, help="cfg path or path:key")
p.add_argument("--split", default="configs/splits/paper_main.yaml")
p.add_argument("--exp", default="adhoc")
p.add_argument("--skip-train", action="store_true")
p.add_argument("--set", default="{}", help='JSON overrides, e.g. {"train":{"epochs":1}}')
args = p.parse_args()

cfg = resolve_market_cfg(load_method_cfg(args.method), args.market)
deep_update(cfg, json.loads(args.set))
split = load_split(args.split)
feat = cfg.get("features", {})
kal = feat.get("kalman", {})
data = build_market_data(args.market, split, nu=feat.get("nu", 120), seg_len=feat.get("seg_len", 20),
                         kalman_delta=kal.get("delta", 1e-4), kalman_r_window=kal.get("r_window", 120))
label = method_label(cfg)
out = PROJECT_ROOT / "results" / args.exp / args.market / label
out.mkdir(parents=True, exist_ok=True)
with open(out / "config.json", "w") as f:
    json.dump({"method": cfg, "split": split, "tickers": data.tickers, "dropped": data.dropped}, f, indent=2)
print(f"[{args.market}/{label}] S={data.S} dropped={len(data.dropped)} "
      f"train_taus={len(data.train_taus)} rebalances={len(data.rebalance_taus)}")

method = build_method(cfg, args.market)
if args.skip_train:
    method.load(out, data)
else:
    method.fit(data, out)

t_eval = time.time()
res = run_backtest(data, method)
rf = out / "resources.json"
resources = json.load(open(rf)) if rf.exists() else {}
resources.update({"eval_time_s": round(time.time() - t_eval, 1), "host_cpu": platform.processor() or platform.machine(),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")})
with open(rf, "w") as f:
    json.dump(resources, f, indent=2)
res["daily"].to_csv(out / "daily_returns.csv")
res["weights"].to_csv(out / "weights.csv")
metrics = {"portfolio": res["portfolio"], "forecast": res.get("forecast")}
with open(out / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[{args.market}/{label}] " + json.dumps(metrics))
