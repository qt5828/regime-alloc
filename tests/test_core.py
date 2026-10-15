import numpy as np
import torch

from regime_alloc.diffusion.ddim import ddim_sample
from regime_alloc.diffusion.schedule import DiffusionSchedule
from regime_alloc.eval.metrics import asr, avol, dtw_batch, rmdd
from regime_alloc.features.kalman_beta import kalman_beta
from regime_alloc.methods.stable.model import StableNet
from regime_alloc.portfolio.black_litterman import bl_mvo_weights, view_moments


def test_bl_reduces_to_equal_weight_without_view_certainty():
    # paper A.4: Ω -> 0  =>  posterior -> prior  =>  w* -> w_eq
    rng = np.random.default_rng(0)
    window = rng.normal(0, 0.01, (120, 5))
    paths = rng.normal(0, 1.0, (5, 50, 20))      # huge view variance vs 1% daily prior => tiny Ω
    w, _ = bl_mvo_weights(window, paths)
    np.testing.assert_allclose(w, np.full(5, 0.2), atol=2e-2)


def test_bl_handles_singular_view_covariance():
    rng = np.random.default_rng(1)
    S, k = 30, 10                                # k < S => Σ_view rank-deficient
    window = rng.normal(0, 0.01, (120, S))
    paths = rng.normal(0.001, 0.01, (S, k, 20))
    _, sv = view_moments(paths)
    assert np.linalg.matrix_rank(sv) < S
    w, _ = bl_mvo_weights(window, paths)
    assert np.isfinite(w).all() and abs(w.sum() - 1) < 1e-9


def test_ddim_eta0_recovers_x0_with_oracle_eps():
    sched = DiffusionSchedule(200)
    x0 = torch.randn(4, 20)
    # oracle ε for a fixed target x0: ε = (r_n - sqrt(ᾱ) x0) / sqrt(1-ᾱ)
    def eps_fn(r, n):
        ab = sched.alpha_bar[n][:, None]
        return (r - ab.sqrt() * x0) / (1 - ab).sqrt()
    out = ddim_sample(eps_fn, sched, (4, 20), n_steps=30, eta=0.0)
    torch.testing.assert_close(out, x0, atol=1e-4, rtol=1e-4)


def test_kalman_recovers_constant_beta():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 3))
    true = np.array([0.5, -1.0, 2.0])
    y = X @ true + rng.normal(0, 0.1, 3000)
    b = kalman_beta(y, X)
    np.testing.assert_allclose(b[-1], true, atol=0.1)
    assert np.isnan(b[0]).all()                  # causal warm-up


def test_metrics_hand_values():
    d = np.array([0.1, -0.5, 0.2])
    # nav: 1, 1.1, 0.55, 0.66 -> max drawdown from 1.1 to 0.55 = 50%
    assert abs(rmdd(d) - 50.0) < 1e-9
    assert abs(avol(d) - d.std(ddof=1) * np.sqrt(252) * 100) < 1e-9
    assert abs(asr(d) - d.mean() / d.std(ddof=1) * np.sqrt(252)) < 1e-9
    a = np.array([[0.0, 1.0, 2.0]]); b = np.array([[0.0, 0.0, 1.0, 2.0][:3]])
    assert dtw_batch(a, a)[0] == 0.0
    assert dtw_batch(a, b)[0] >= 0.0


def test_stable_net_shapes_and_gate_range():
    net = StableNet(macro_dim=10, stock_dim=31, z_max=2.0)
    eps, z = net(torch.randn(8, 20), torch.randint(0, 200, (8,)), torch.randn(8, 10), torch.randn(8, 31))
    assert eps.shape == (8, 20) and z.shape == (8, 1)
    assert (z >= 0).all() and (z <= 2.0).all()
