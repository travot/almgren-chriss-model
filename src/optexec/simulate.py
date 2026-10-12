"""Monte Carlo simulation of the implementation shortfall of an execution schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ac import ACParams


@dataclass(frozen=True)
class ShortfallStats:
    mean: float
    std: float
    var95: float
    cvar95: float
    n_paths: int

    @property
    def stderr_of_mean(self) -> float:
        return self.std / np.sqrt(self.n_paths)


def standardized_shocks(rng: np.random.Generator, shape: tuple[int, int], df: float | None) -> np.ndarray:
    """Unit-variance shocks: Gaussian if ``df`` is None, else Student-t with ``df`` degrees (> 2)."""
    if df is None:
        return rng.standard_normal(shape)
    if df <= 2:
        raise ValueError("df must be > 2 for a finite variance")
    return rng.standard_t(df, size=shape) / np.sqrt(df / (df - 2.0))


def simulate_shortfall(
    p: ACParams,
    n: np.ndarray,
    n_paths: int = 100_000,
    impact: Callable[[np.ndarray], np.ndarray] | None = None,
    sigmas: np.ndarray | None = None,
    df: float | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """Simulate ``n_paths`` shortfalls (price units * shares) for the schedule ``n``.

    The execution price of slice ``k`` is ``S_{k-1} - eps - h(n_k / tau)`` where
    ``S_k = S_{k-1} + sigma_k sqrt(tau) xi_k - gamma n_k``. By default ``h`` is the linear
    impact of the model, so the sample mean and variance converge to the closed forms.
    """
    n = np.asarray(n, dtype=float)
    if n.shape != (p.N,):
        raise ValueError(f"n must have length N={p.N}")
    rng = np.random.default_rng(seed)
    h = impact if impact is not None else (lambda v: p.eta * v)
    s = np.full(p.N, p.sigma) if sigmas is None else np.asarray(sigmas, dtype=float)

    xi = standardized_shocks(rng, (n_paths, p.N), df)
    noise = xi * (s * np.sqrt(p.tau))  # (M, N)
    cum_noise_before = np.cumsum(noise, axis=1) - noise  # shocks up to interval k-1
    cum_trades_before = np.cumsum(n) - n  # shares sold before interval k
    s_before = p.S0 + cum_noise_before - p.gamma * cum_trades_before  # S_{k-1}
    exec_price = s_before - p.eps - h(n / p.tau)
    return p.X * p.S0 - exec_price @ n


def summarize(shortfall: np.ndarray, level: float = 0.95) -> ShortfallStats:
    """Mean, standard deviation, VaR and CVaR (expected shortfall) at ``level``."""
    shortfall = np.asarray(shortfall, dtype=float)
    var = float(np.quantile(shortfall, level))
    tail = shortfall[shortfall >= var]
    return ShortfallStats(
        mean=float(shortfall.mean()),
        std=float(shortfall.std(ddof=1)),
        var95=var,
        cvar95=float(tail.mean()),
        n_paths=int(shortfall.size),
    )
