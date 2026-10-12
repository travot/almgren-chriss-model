import numpy as np
import pandas as pd
import pytest

from optexec import calibration as cal
from optexec import data
from optexec import volatility as vol


def simulate_garch(n, omega, alpha, beta, seed=0):
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    x = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    z = rng.standard_normal(n)
    for i in range(n):
        x[i] = np.sqrt(h[i]) * z[i]
        if i < n - 1:
            h[i + 1] = omega + alpha * x[i] ** 2 + beta * h[i]
    return x, h


def test_garch_filter_matches_loop():
    x, h = simulate_garch(2000, 2e-7, 0.08, 0.9)
    assert np.allclose(vol.garch_filter(x, 2e-7, 0.08, 0.9, h0=h[0]), h, rtol=1e-10)


def test_garch_parameters_are_recovered():
    x, _ = simulate_garch(30_000, 2e-7, 0.08, 0.90, seed=5)
    fit = vol.fit_garch11(x)
    assert fit.alpha == pytest.approx(0.08, abs=0.02)
    assert fit.beta == pytest.approx(0.90, abs=0.03)
    assert fit.persistence == pytest.approx(0.98, abs=0.01)
    assert fit.unconditional_variance == pytest.approx(2e-7 / 0.02, rel=0.3)


def test_ewma_recursion():
    r = np.array([0.0, 1.0, 0.0, 2.0])
    h = vol.ewma_variance(r, lam=0.5)
    assert h[1] == pytest.approx(0.5 * h[0] + 0.5 * 0.0)
    assert h[2] == pytest.approx(0.5 * h[1] + 0.5 * 1.0)
    assert h[3] == pytest.approx(0.5 * h[2] + 0.5 * 0.0)


def test_qlike_is_minimised_at_the_truth():
    rv = np.full(100, 4.0)
    assert vol.qlike(rv, rv) == pytest.approx(0.0)
    assert vol.qlike(rv, rv * 0.5) > 0 and vol.qlike(rv, rv * 2) > 0


def test_garch_horizon_forecast_mean_reverts():
    fit = vol.GarchFit(omega=1e-7, alpha=0.05, beta=0.9, loglik=0.0, success=True)
    unc = fit.unconditional_variance
    assert vol.garch_horizon_variance(fit, unc, 10) == pytest.approx(10 * unc)
    assert vol.garch_horizon_variance(fit, 5 * unc, 10) > 10 * unc


def test_forecast_evaluation_ranks_dynamic_models_first():
    market, _ = data.synthetic_market(n_days=40, seed=2)
    r = cal.log_returns(market["close"])
    scores = vol.score_forecasts(vol.evaluate_horizon_forecasts(r, horizon=60, train_frac=0.6))
    assert scores.loc["garch", "QLIKE"] < scores.loc["constant", "QLIKE"]
    assert scores.loc["ewma", "QLIKE"] < scores.loc["constant", "QLIKE"]
