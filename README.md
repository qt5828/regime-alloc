# regime-alloc

A research codebase for regime-aware portfolio allocation. It includes a reproduction of **STABLE** (Soun et al., ICLR 2026), which is the baseline, and leaves room for a new proposed method. All methods share the same data, splits, backtester and metrics.

## Setup
```bash
/usr/local/bin/python3.11 -m venv .venv
.venv/bin/pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt     # (without the torch line)
```
On this host, GPU 6 fails CUDA init. `run_experiment.py` uses `0,1,2,3,4,5,7` by default.

## Run
```bash
.venv/bin/python scripts/download.py                       # yfinance, 2012-01-01 ~ 2026-10-15 → data/raw/
.venv/bin/python scripts/run_experiment.py configs/experiments/paper_table3_4.yaml
# single job:  scripts/run_method.py --market sp500 --method configs/methods/stable.yaml --exp X [--set '{"train":{"epochs":1}}']
.venv/bin/python -m pytest
```
Outputs go to `results/<exp>/<market>/<method>/`: `metrics.json`, `daily_returns.csv`, `weights.csv`, `model.pt`, `train_log.jsonl`. The run also writes `results/<exp>/summary.md`, which compares your numbers with paper Tables 3 and 4.

## Layout
```
configs/
  data/universes.yaml        market → GICS sector → tickers, index proxy, macro tickers
  splits/*.yaml              train/test windows (paper_main = paper Table 2)
  methods/*.yaml             hyperparameters, each tagged [paper] / [ref:X] / [assume]
  experiments/*.yaml         markets × methods × split
regime_alloc/
  data/        download (yfinance + parquet cache), universe filter, MarketData bundle (dataset.py)
  features/    macro m_τ, Kalman β_τ (stock embedding)
  diffusion/   noise schedule, DDIM sampler
  nn/          conditional 1-D UNet
  portfolio/   Black–Litterman + max-Sharpe MVO
  eval/        backtest (shared by all methods), metrics (ASR/RMDD/AVol/MSE/DTW)
  methods/     base.py (interface) · registry.py · stable/ · classic/ (CRP, MVO, MOM) · proposed/ (new method goes here)
scripts/       download, run_method, run_experiment
tests/         BL / DDIM / Kalman / metric / model sanity tests
```
**To add a method:** subclass `methods/base.Method` (`fit`, `load`, `allocate`, and optionally `forecast`), then register it in `methods/registry.py` and add a config under `configs/methods/`. A method sees only `MarketData` and the rebalance index τ, and must not read rows after τ.

**Leakage guards:** the calendar stops at `test_end`. Normalisation and Kalman filtering are causal. Return scale is fit on the train window only. Training targets never cross `train_end`. On non-US markets, US-quoted macro series are lagged by one day.

## Paper → code
| Paper | Code |
|---|---|
| m_τ (Table 1): index, DXY, 10y−3m, VIX, gold; ν-rolling z-score + log-diff | `features/macro.py` (d_m = 10) |
| β_τ^(s) Kalman embedding (Das & Ghoshal 2010) | `features/kalman_beta.py` |
| c_τ^(s) = [β, normalised close, daily log returns] | `methods/stable/method.py::StableFeatures` |
| CDG + MLG: ε̂ = ε̂_n + z(ε̂_n^(s) − ε̂_n), z = g_π(h_f) ∈ [0, z_max], Eq. 2 loss | `methods/stable/model.py` |
| DDIM (§3.3, A.2) | `diffusion/ddim.py` |
| Views (Eq. 3–4), BL posterior (Eq. 6–7), w* (Eq. 8), μ_prior = Σ_prior w_eq (A.3) | `portfolio/black_litterman.py` |
| Ablations (A.7) | `ablation.use_kfe / use_mlg / use_bl` in `configs/methods/stable.yaml` |

## Values not stated in the paper
| Item | Choice | Source |
|---|---|---|
| Data source | yfinance adjusted close | user decision |
| CSI300 / KOSPI200 index | 510300.SS / 069500.KS ETFs (yfinance has no 000300.SS or ^KS200 history) | assume |
| 3-month rate | ^IRX (13-week T-bill) | assume |
| Universe | hand-picked from GICS top caps as of 2025-03; dropped if not listed for the whole window → S&P 55, CSI 55, EURO 42 (paper 37), KOSPI 39 (paper 44) | assume |
| Noise schedule | linear 1e-4→0.02 rescaled by 1000/T | Improved-DDPM |
| UNet | lucidrains `Unet1D` without attention, dim 64, mults (1,2,4) | ref |
| Optimizer | Adam lr 2e-4, batch 128, grad-clip 1 | DDPM |
| EMA | 0.995 | Diffusion-TS |
| Kalman Q, R | Q = δ/(1−δ)I with δ = 1e-4; R estimated adaptively from a 120-day innovation window | Chan / Das & Ghoshal |
| Return history in c_τ | last ℓ = 20 days; returns scaled by per-stock train std | assume |
| Prior covariance | ν-day daily cov × ℓ (matched to the ℓ-day view horizon) | assume |
| Ω when k ≤ S | Σ_view is singular → pseudo-inverse | assume |
| Backtest | rebalance every ℓ days, buy-and-hold in between, no costs | assume |
| DTW | \|·\| cost, normalised by ℓ | assume |
| MVO / MOM baselines | long-only max-Sharpe on 120 days / top-third equal weight on 120-day momentum | Markowitz / Jegadeesh–Titman |

## Status
- Done: data pipeline, STABLE, CRP/MVO/MOM, an end-to-end smoke run (1 epoch) and the unit tests.
- Not yet: deep baselines (DeepTrader, MetaTrader, AlphaMix, Diffusion-TS, AEC-GAN, KoVAE) and the Appendix A.4–A.6 experiments.
