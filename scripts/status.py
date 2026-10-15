"""Progress monitor for runs under results/.

python scripts/status.py                    # all experiments, one snapshot
python scripts/status.py paper_table3_4     # one experiment
python scripts/status.py --watch 60         # refresh every 60 s (Ctrl-C to stop)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STALL_SEC = 600   # no train-log update for this long -> "stalled?"


def _fmt_td(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def run_status(run_dir: Path) -> dict:
    cfg_f, log_f, met_f = run_dir / "config.json", run_dir / "train_log.jsonl", run_dir / "metrics.json"
    st = {"run": f"{run_dir.parent.name}/{run_dir.name}", "state": "pending", "progress": "",
          "eta": "", "last": "", "result": ""}
    total = None
    if cfg_f.exists():
        total = json.load(open(cfg_f))["method"].get("train", {}).get("epochs")
    if log_f.exists() and log_f.stat().st_size:
        recs = [json.loads(line) for line in open(log_f)]
        r = recs[-1]
        ep = r["epoch"]
        st["progress"] = f"{ep}/{total}" if total else str(ep)
        st["last"] = f"loss {r['loss']:.3f} mse {r['mse']:.3f} gate {r['gate_mean']:.2f}"
        age = time.time() - log_f.stat().st_mtime
        if total and ep < total:
            per_ep = r["elapsed_s"] / ep
            st["eta"] = _fmt_td(per_ep * (total - ep))
            st["state"] = "training" if age < STALL_SEC else f"stalled? ({_fmt_td(age)} idle)"
        else:
            st["state"] = "evaluating"
    elif cfg_f.exists():
        st["state"] = "running"
    if met_f.exists():
        m = json.load(open(met_f))
        p, f = m["portfolio"], m.get("forecast") or {}
        st["state"] = "done"
        st["result"] = f"ASR {p['ASR']:.2f} RMDD {p['RMDD']:.2f} AVol {p['AVol']:.2f}" + (
            f" | MSE {f['MSE']*1e4:.2f}e-4 DTW {f['DTW']*1e3:.2f}e-3" if f else "")
    return st


def experiment_status(exp_dir: Path) -> list[dict]:
    runs = sorted(p.parent for p in exp_dir.glob("*/*/config.json"))
    rows = [run_status(r) for r in runs]
    # failed jobs: log ends with a traceback and no metrics.json
    for row in rows:
        log = exp_dir / "logs" / (row["run"].replace("/", "_") + ".log")
        if row["state"] != "done" and log.exists() and "Traceback" in log.read_text()[-5000:]:
            row["state"] = f"FAILED (see {log.relative_to(ROOT)})"
    return rows


def print_table(exp_dir: Path):
    rows = experiment_status(exp_dir)
    n_done = sum(r["state"] == "done" for r in rows)
    print(f"\n== {exp_dir.name}  ({n_done}/{len(rows)} done)  {datetime.now():%Y-%m-%d %H:%M:%S}")
    w = {"run": 26, "state": 12, "progress": 9, "eta": 9, "last": 38}
    print("  ".join(k.ljust(v) for k, v in w.items()) + "result")
    for r in rows:
        print("  ".join(str(r[k]).ljust(v) for k, v in w.items()) + r["result"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp", nargs="?")
    p.add_argument("--watch", type=int, default=0, help="refresh interval in seconds")
    a = p.parse_args()
    res = ROOT / "results"
    exps = [res / a.exp] if a.exp else sorted(d for d in res.iterdir() if d.is_dir())
    while True:
        if a.watch:
            os.system("clear")
        for e in exps:
            print_table(e)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
