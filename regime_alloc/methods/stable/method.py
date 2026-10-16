"""STABLE method: training (Eq. 2), DDIM sampling of k paths, BL–MVO allocation."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from regime_alloc.data.dataset import MarketData
from regime_alloc.diffusion.ddim import ddim_sample
from regime_alloc.diffusion.schedule import DiffusionSchedule
from regime_alloc.methods.base import Method
from regime_alloc.methods.stable.model import StableNet
from regime_alloc.portfolio.black_litterman import bl_mvo_weights
from regime_alloc.utils import set_seed


class StableFeatures:
    """Builds (m_τ, c_τ^(s), r_0) tensors from MarketData.
    c_τ^(s) = [β_τ^(s), normalised price level, last H daily log returns].
    Returns are divided by the per-stock train-window std (ret_scale)."""

    def __init__(self, data: MarketData, hist_len: int, use_kfe: bool, device):
        self.L, self.H = data.seg_len, hist_len
        self.scale = torch.tensor(data.ret_scale, dtype=torch.float32, device=device)
        beta = data.beta if use_kfe else np.zeros_like(data.beta)   # ablation A.7: β = 0
        self.macro = torch.tensor(data.macro, dtype=torch.float32, device=device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=device)
        self.pxz = torch.tensor(data.px_z, dtype=torch.float32, device=device)
        self.ret = torch.tensor(data.logret, dtype=torch.float32, device=device) / self.scale
        self.macro_dim = data.macro.shape[1]
        self.stock_dim = data.beta.shape[2] + 1 + hist_len
        self.device = device

    def cond(self, tau: torch.Tensor, s: torch.Tensor):
        m = self.macro[tau]
        offs = torch.arange(-self.H + 1, 1, device=self.device)
        hist = self.ret[tau[:, None] + offs[None], s[:, None]]          # (B, H)
        c = torch.cat([self.beta[tau, s], self.pxz[tau, s][:, None], hist], dim=1)
        return m, c

    def target(self, tau: torch.Tensor, s: torch.Tensor):
        offs = torch.arange(1, self.L + 1, device=self.device)
        return self.ret[tau[:, None] + offs[None], s[:, None]]          # (B, ℓ)

    def valid_pairs(self, taus: np.ndarray, S: int) -> torch.Tensor:
        tau = torch.tensor(np.repeat(taus, S), device=self.device)
        s = torch.tensor(np.tile(np.arange(S), len(taus)), device=self.device)
        ok = []
        for i in range(0, len(tau), 65536):
            t, ss = tau[i:i + 65536], s[i:i + 65536]
            m, c = self.cond(t, ss)
            y = self.target(t, ss)
            ok.append(torch.isfinite(m).all(1) & torch.isfinite(c).all(1) & torch.isfinite(y).all(1))
        ok = torch.cat(ok)
        return torch.stack([tau[ok], s[ok]], dim=1)


class Stable(Method):
    name = "stable"

    def __init__(self, cfg: dict, market: str, device: str | None = None):
        super().__init__(cfg, market)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sched = DiffusionSchedule(cfg["diffusion"]["timesteps"],
                                       cfg["diffusion"]["beta_schedule"]).to(self.device)
        self.net = self.ema = self.feats = None
        self._path_cache: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- setup
    def _setup(self, data: MarketData):
        ab = self.cfg["ablation"]
        self.feats = StableFeatures(data, self.cfg["features"]["hist_len"], ab["use_kfe"], self.device)
        mc = self.cfg["model"]
        self.net = StableNet(self.feats.macro_dim, self.feats.stock_dim, mc["embed_dim"],
                             mc["unet_dim"], mc["unet_dim_mults"], mc["z_max"],
                             use_mlg=ab["use_mlg"]).to(self.device)
        self.ema = copy.deepcopy(self.net).eval().requires_grad_(False)

    # ---------------------------------------------------------------- train
    def fit(self, data: MarketData, out_dir: Path) -> None:
        tc = self.cfg["train"]
        set_seed(tc["seed"])
        self._setup(data)
        pairs = self.feats.valid_pairs(data.train_taus, data.S)
        opt = torch.optim.Adam(self.net.parameters(), lr=tc["lr"])
        g = torch.Generator(device="cpu").manual_seed(tc["seed"])
        out_dir.mkdir(parents=True, exist_ok=True)
        log = open(out_dir / "train_log.jsonl", "w")
        print(f"[stable/{self.market}] {len(pairs)} training pairs, "
              f"{sum(p.numel() for p in self.net.parameters())/1e6:.2f}M params, device={self.device}")
        t0 = time.time()
        for epoch in range(tc["epochs"]):
            perm = torch.randperm(len(pairs), generator=g).to(self.device)
            acc = torch.zeros(3, device=self.device)   # loss, mse, gate (no per-step sync)
            nb = 0
            for i in range(0, len(perm), tc["batch_size"]):
                idx = pairs[perm[i:i + tc["batch_size"]]]
                tau, s = idx[:, 0], idx[:, 1]
                m, c = self.feats.cond(tau, s)
                r0 = self.feats.target(tau, s)
                n = torch.randint(0, self.sched.T, (len(r0),), device=self.device)
                eps = torch.randn_like(r0)
                r_n = self.sched.q_sample(r0, n, eps)
                eps_hat, z = self.net(r_n, n, m, c)
                mse = ((eps - eps_hat) ** 2).sum(1).mean()      # ‖ε − ε̂‖²₂
                loss = mse + tc["l2_weight"] * self.net.l2_penalty()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), tc["grad_clip"])
                opt.step()
                with torch.no_grad():
                    for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                        pe.lerp_(p, 1 - tc["ema_decay"])
                acc += torch.stack([loss.detach(), mse.detach(), z.detach().mean()])
                nb += 1
            a = (acc / nb).tolist()
            rec = {"epoch": epoch + 1, "loss": a[0], "mse": a[1], "gate_mean": a[2],
                   "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[stable/{self.market}] " + json.dumps(rec))
        log.close()
        res = {"device": str(self.device),
               "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
               "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated(self.device) / 2**20, 1)
               if self.device.type == "cuda" else None,
               "params": sum(p.numel() for p in self.net.parameters()),
               "train_pairs": len(pairs), "epochs": tc["epochs"], "batch_size": tc["batch_size"],
               "train_time_s": round(time.time() - t0, 1),
               "sec_per_epoch": round((time.time() - t0) / tc["epochs"], 1)}
        with open(out_dir / "resources.json", "w") as f:
            json.dump(res, f, indent=2)
        torch.save({"net": self.net.state_dict(), "ema": self.ema.state_dict()},
                   out_dir / "model.pt")

    def load(self, out_dir: Path, data: MarketData | None = None) -> None:
        set_seed(self.cfg["train"]["seed"])
        self._setup(data)
        st = torch.load(out_dir / "model.pt", map_location=self.device)
        self.net.load_state_dict(st["net"])
        self.ema.load_state_dict(st["ema"])

    # ---------------------------------------------------------------- infer
    @torch.no_grad()
    def forecast(self, data: MarketData, tau: int) -> np.ndarray:
        if tau in self._path_cache:
            return self._path_cache[tau]
        dc = self.cfg["diffusion"]
        k, S, L = dc["n_paths"], data.S, data.seg_len
        t = torch.full((S,), tau, device=self.device)
        s = torch.arange(S, device=self.device)
        m, c = self.feats.cond(t, s)
        assert torch.isfinite(m).all() and torch.isfinite(c).all(), f"NaN condition at τ={tau}"
        h_m_pad, h_f = self.ema.conditions(m, c)
        h_m_pad, h_f = h_m_pad.repeat_interleave(k, 0), h_f.repeat_interleave(k, 0)
        gen = torch.Generator(device=self.device).manual_seed(self.cfg["train"]["seed"] * 100003 + tau)
        r0 = ddim_sample(lambda r, n: self.ema.eps(r, n, h_m_pad, h_f)[0], self.sched,
                         (S * k, L), dc["ddim_steps"], dc["eta"], self.device, gen)
        paths = (r0.view(S, k, L) * self.feats.scale[:, None, None]).cpu().numpy()
        self._path_cache[tau] = paths
        return paths

    def allocate(self, data: MarketData, tau: int) -> np.ndarray:
        nu = self.cfg["features"]["nu"]
        window = data.simret[tau - nu + 1: tau + 1]
        w, info = bl_mvo_weights(window, self.forecast(data, tau),
                                 self.cfg["bl"]["view_precision"], self.cfg["ablation"]["use_bl"])
        self.last_info = info
        return w
