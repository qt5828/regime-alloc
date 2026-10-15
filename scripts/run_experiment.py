"""Run an experiment config (markets × methods), one subprocess per
(market, method) spread over GPUs, then write summary tables.

python scripts/run_experiment.py configs/experiments/paper_table3_4.yaml \
    [--gpus 0,1,2,3] [--set '{"train":{"epochs":1}}'] [--summary-only]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from regime_alloc.methods.registry import method_label  # noqa: E402
from regime_alloc.utils import load_method_cfg, load_yaml  # noqa: E402

# Paper Table 3 / Table 4 (for side-by-side comparison)
PAPER = {
    "sp500":     {"stable": (1.85, 7.82, 13.43, 3.51, 5.62), "crp": (0.82, 8.89, 14.44), "mvo": (1.18, 9.00, 21.18), "mom": (0.03, 11.87, 17.00)},
    "csi300":    {"stable": (-0.41, 8.85, 17.17, 3.89, 6.09), "crp": (-0.70, 10.96, 18.21), "mvo": (-0.66, 13.18, 25.93), "mom": (-0.47, 15.57, 28.77)},
    "eurostoxx": {"stable": (2.92, 3.84, 10.88, 2.49, 4.78), "crp": (1.31, 5.40, 12.96), "mvo": (0.48, 8.35, 16.75), "mom": (1.42, 5.41, 12.71)},
    "kospi200":  {"stable": (1.61, 8.34, 17.82, 8.15, 8.67), "crp": (0.76, 8.72, 26.55), "mvo": (0.45, 11.84, 29.49), "mom": (0.33, 13.49, 23.34)},
}


def summarize(exp: dict) -> str:
    res_dir = ROOT / "results" / exp["name"]
    lines = ["| Market | Method | ASR | RMDD % | AVol % | MSE ×1e-4 | DTW ×1e-3 | paper (ASR / RMDD / AVol / MSE / DTW) |",
             "|---|---|---|---|---|---|---|---|"]
    for m in exp["markets"]:
        for spec in exp["methods"]:
            label = method_label(load_method_cfg(spec))
            f = res_dir / m / label / "metrics.json"
            if not f.exists():
                lines.append(f"| {m} | {label} | – | – | – | – | – | missing |")
                continue
            r = json.load(open(f))
            pm, fm = r["portfolio"], r.get("forecast") or {}
            mse = f"{fm['MSE']*1e4:.2f}" if fm else "–"
            dtw = f"{fm['DTW']*1e3:.2f}" if fm else "–"
            paper = PAPER.get(m, {}).get(label)
            ps = " / ".join(f"{v:.2f}" for v in paper) if paper else ""
            lines.append(f"| {m} | {label} | {pm['ASR']:.2f} | {pm['RMDD']:.2f} | {pm['AVol']:.2f} | {mse} | {dtw} | {ps} |")
    table = "\n".join(lines)
    (res_dir / "summary.md").write_text(table + "\n")
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp")
    p.add_argument("--gpus", default="0,1,2,3,4,5,7")   # GPU 6 fails CUDA init on this host
    p.add_argument("--set", default="{}")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    exp = load_yaml(ROOT / args.exp)

    if not args.summary_only:
        from regime_alloc.data.dataset import build_market_data, load_split
        split = load_split(exp["split"])
        for m in exp["markets"]:          # build caches once, before parallel jobs
            d = build_market_data(m, split)
            print(f"[data] {m}: S={d.S}, dropped={d.dropped}")
        gpus = args.gpus.split(",")
        jobs = [(m, spec) for spec in exp["methods"] for m in exp["markets"]]
        logs = ROOT / "results" / exp["name"] / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        running, gi = [], 0
        for m, spec in jobs:
            label = method_label(load_method_cfg(spec))
            env = dict(os.environ, PYTHONUNBUFFERED="1")   # live logs in results/<exp>/logs
            if load_method_cfg(spec)["method"] == "stable":     # only STABLE needs a GPU
                env["CUDA_VISIBLE_DEVICES"] = gpus[gi % len(gpus)]
                gi += 1
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            cmd = [sys.executable, str(ROOT / "scripts/run_method.py"), "--market", m,
                   "--method", spec, "--split", exp["split"], "--exp", exp["name"], "--set", args.set]
            if args.skip_train:
                cmd.append("--skip-train")
            log = open(logs / f"{m}_{label}.log", "w")
            running.append((m, label, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)))
        while any(pr.poll() is None for *_, pr in running):
            time.sleep(10)
        for m, label, pr in running:
            print(f"{m:<10} {label:<12} exit={pr.returncode}")
    print(summarize(exp))


if __name__ == "__main__":
    main()
