"""Volatility forecasting: EWMA and GARCH(1,1), with out-of-sample evaluation.

GARCH(1,1) is implemented from scratch (Gaussian quasi-maximum likelihood) so every step is
visible::

    sigma2_t = omega + alpha * r_{t-1}**2 + beta * sigma2_{t-1}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter


def _linear_recursion(drive: np.ndarray, coef: float, h0: float) -> np.ndarray:
    """Solve ``h_t = drive_{t-1} + coef * h_{t-1}`` with ``h_0 = h0`` (vectorised with a linear filter)."""
    h = np.empty(len(drive) + 1)
    h[0] = h0
    h[1:] = lfilter([1.0], [1.0, -coef], drive, zi=[coef * h0])[0]
    return h


def ewma_variance(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics-style EWMA: ``h_t`` is the variance forecast for time ``t`` given returns up to ``t-1``."""
    r = np.asarray(returns, float)
    h0 = np.mean(r[: min(len(r), 100)] ** 2)
    return _linear_recursion((1.0 - lam) * r[:-1] ** 2, lam, h0)


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    loglik: float
    success: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)


def garch_filter(returns: np.ndarray, omega: float, alpha: float, beta: float, h0: float | None = None) -> np.ndarray:
    """Conditional variances ``h_t`` (forecast of ``r_t`` given information up to ``t-1``)."""
    r = np.asarray(returns, float)
    start = np.var(r) if h0 is None else h0
    return _linear_recursion(omega + alpha * r[:-1] ** 2, beta, start)


def _neg_loglik(theta: np.ndarray, r: np.ndarray, scale: float) -> float:
    omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e12
    h = garch_filter(r, omega, alpha, beta)
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e12
    return 0.5 * float(np.sum(np.log(h) + r**2 / h))


def fit_garch11(returns: np.ndarray) -> GarchFit:
    """Quasi-maximum-likelihood estimate of (omega, alpha, beta).

    Returns are rescaled internally so the optimiser is well conditioned; the parameters returned are
    for the original scale.
    """
    r = np.asarray(returns, float)
    r = r - r.mean()
    scale = np.std(r)
    z = r / scale
    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.80), (0.03, 0.95)]:
        theta0 = np.array([max(1e-6, 1.0 - a0 - b0), a0, b0])
        res = minimize(
            _neg_loglik, theta0, args=(z, scale), method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000},
        )
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta = best.x
    n = len(z)
    loglik = -best.fun - 0.5 * n * np.log(2 * np.pi) - n * np.log(scale)
    return GarchFit(
        omega=float(omega * scale**2), alpha=float(alpha), beta=float(beta),
        loglik=float(loglik), success=bool(best.success),
    )


def garch_horizon_variance(fit: GarchFit, h_next: float, horizon: int) -> float:
    """Forecast of the variance accumulated over the next ``horizon`` steps.

    ``h_next`` is the one-step-ahead conditional variance; the k-step forecast mean-reverts to the
    unconditional variance at rate ``alpha + beta``.
    """
    pers, uncond = fit.persistence, fit.unconditional_variance
    k = np.arange(horizon)
    return float(np.sum(uncond + pers**k * (h_next - uncond)))


def qlike(realized: np.ndarray, forecast: np.ndarray) -> float:
    """Mean QLIKE loss ``RV/F - log(RV/F) - 1`` (robust to noise in the variance proxy)."""
    ratio = np.asarray(realized, float) / np.asarray(forecast, float)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def evaluate_horizon_forecasts(
    returns: pd.Series, horizon: int = 60, train_frac: float = 0.6, ewma_lambda: float = 0.94
) -> pd.DataFrame:
    """Out-of-sample comparison of forecasts of the variance over the next ``horizon`` bars.

    GARCH is fitted on the first ``train_frac`` of the sample only. At the start of each
    non-overlapping test window, three forecasts of the window's variance are made:
    constant (training variance), EWMA and GARCH(1,1); they are scored against the realised
    window variance with QLIKE and with the mean squared error of volatility.
    """
    r = returns.to_numpy()
    n_train = int(len(r) * train_frac)
    fit = fit_garch11(r[:n_train])
    h_garch = garch_filter(r, fit.omega, fit.alpha, fit.beta, h0=fit.unconditional_variance)
    h_ewma = ewma_variance(r, ewma_lambda)
    const = np.var(r[:n_train])
    rows = []
    for s in range(n_train, len(r) - horizon, horizon):
        realized = float(np.sum(r[s : s + horizon] ** 2))
        rows.append(
            {
                "start": returns.index[s],
                "realized": realized,
                "constant": const * horizon,
                "ewma": h_ewma[s] * horizon,
                "garch": garch_horizon_variance(fit, h_garch[s], horizon),
            }
        )
    out = pd.DataFrame(rows).set_index("start")
    out.attrs["garch_fit"] = fit
    return out


def score_forecasts(table: pd.DataFrame) -> pd.DataFrame:
    """QLIKE and vol-RMSE per forecasting method (lower is better)."""
    rv = table["realized"].to_numpy()
    res = {}
    for m in [c for c in table.columns if c != "realized"]:
        f = table[m].to_numpy()
        res[m] = {
            "QLIKE": qlike(rv, f),
            "vol_RMSE": float(np.sqrt(np.mean((np.sqrt(rv) - np.sqrt(f)) ** 2))),
        }
    return pd.DataFrame(res).T
