"""Discrete-time Almgren-Chriss optimal execution.

Reference: R. Almgren and N. Chriss, "Optimal execution of portfolio transactions",
Journal of Risk 3 (2000).

Setting
-------
A trader must sell ``X`` shares by time ``T``, split in ``N`` intervals of length
``tau = T / N``. ``n_k`` is the number of shares sold in interval ``k`` and
``x_k = X - n_1 - ... - n_k`` the shares still held after interval ``k``
(``x_0 = X`` and ``x_N = 0``).

Price dynamics (arithmetic random walk with linear permanent impact)::

    S_k = S_{k-1} + sigma * sqrt(tau) * xi_k - gamma * n_k

The price actually obtained in interval ``k`` includes a temporary impact::

    S~_k = S_{k-1} - eps - eta * n_k / tau

The implementation shortfall is ``X * S_0 - sum_k n_k * S~_k``. Its mean and variance are::

    E = gamma * X**2 / 2 + eps * X + (eta_tilde / tau) * sum_k n_k**2,   eta_tilde = eta - gamma * tau / 2
    V = tau * sum_k sigma_k**2 * x_k**2

The optimal trajectory minimises ``E + lam * V`` (``lam`` is the risk aversion).
See ``docs/theory.md`` for the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class ACParams:
    """Parameters of the Almgren-Chriss model (all in consistent units).

    Attributes
    ----------
    X : shares to sell (> 0)
    T : horizon, in time units (for example minutes)
    N : number of trading intervals
    sigma : price volatility per sqrt(time unit), in price units
    gamma : permanent impact, price units per share traded
    eta : temporary impact, price units per (share / time unit)
    eps : fixed cost per share (half spread), price units
    S0 : arrival price
    """

    X: float
    T: float
    N: int
    sigma: float
    gamma: float
    eta: float
    eps: float = 0.0
    S0: float = 1.0

    def __post_init__(self) -> None:
        if self.X <= 0 or self.T <= 0 or self.N < 1:
            raise ValueError("X and T must be positive and N >= 1")
        if self.sigma < 0 or self.gamma < 0 or self.eps < 0:
            raise ValueError("sigma, gamma and eps must be non-negative")
        if self.eta_tilde <= 0:
            raise ValueError(
                f"eta_tilde = eta - gamma*tau/2 = {self.eta_tilde:.3g} must be positive: "
                "use a smaller interval tau or a larger temporary impact"
            )

    @property
    def tau(self) -> float:
        return self.T / self.N

    @property
    def eta_tilde(self) -> float:
        return self.eta - 0.5 * self.gamma * self.tau

    @property
    def times(self) -> np.ndarray:
        """Grid t_0 = 0, ..., t_N = T."""
        return np.linspace(0.0, self.T, self.N + 1)


def kappa(p: ACParams, lam: float) -> float:
    """Decay rate of the optimal trajectory, solution of cosh(kappa tau) = 1 + (kappa_tilde tau)^2 / 2."""
    if lam <= 0:
        return 0.0
    kappa_tilde_sq = lam * p.sigma**2 / p.eta_tilde
    return float(np.arccosh(1.0 + 0.5 * kappa_tilde_sq * p.tau**2) / p.tau)


def lambda_for_kappa(p: ACParams, kappa_T: float) -> float:
    """Risk aversion that produces ``kappa * T = kappa_T``.

    ``kappa * T`` is a dimensionless measure of urgency: 0 is TWAP, around 1 to 2 is a moderate
    front-loading and 5 or more is aggressive. Using it avoids having to reason about the raw
    units of ``lam``.
    """
    if kappa_T <= 0:
        return 0.0
    k = kappa_T / p.T
    kappa_tilde_sq = 2.0 * (np.cosh(k * p.tau) - 1.0) / p.tau**2
    return float(kappa_tilde_sq * p.eta_tilde / p.sigma**2)


def _sinh_ratio(a: np.ndarray, b: float) -> np.ndarray:
    """sinh(a) / sinh(b) computed without overflow (a, b >= 0)."""
    a = np.asarray(a, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = np.exp(a - b) * (1.0 - np.exp(-2.0 * a)) / (1.0 - np.exp(-2.0 * b))
    return np.where(a == 0.0, 0.0, ratio)


def optimal_holdings(p: ACParams, lam: float) -> np.ndarray:
    """Optimal holdings x_0, ..., x_N (length N + 1) for risk aversion ``lam``.

    ``lam = 0`` gives the linear (TWAP) trajectory, larger ``lam`` front-loads the sale.
    """
    k = kappa(p, lam)
    t = p.times
    if k * p.T < 1e-9:  # risk-neutral limit (and avoids 0/0)
        return p.X * (1.0 - t / p.T)
    return p.X * _sinh_ratio(k * (p.T - t), k * p.T)


def trades_from_holdings(x: np.ndarray) -> np.ndarray:
    """Shares sold in each interval: n_k = x_{k-1} - x_k."""
    return -np.diff(x)


def optimal_trades(p: ACParams, lam: float) -> np.ndarray:
    return trades_from_holdings(optimal_holdings(p, lam))


def twap_trades(p: ACParams) -> np.ndarray:
    """Equal slices: the risk-neutral solution."""
    return np.full(p.N, p.X / p.N)


def immediate_trades(p: ACParams) -> np.ndarray:
    """Everything in the first interval: zero price risk, maximal temporary impact."""
    n = np.zeros(p.N)
    n[0] = p.X
    return n


def holdings_from_trades(p: ACParams, n: np.ndarray) -> np.ndarray:
    return np.concatenate([[p.X], p.X - np.cumsum(n)])


def expected_cost(p: ACParams, n: np.ndarray) -> float:
    """Expected implementation shortfall in price units * shares (linear impact)."""
    n = np.asarray(n, dtype=float)
    return float(
        0.5 * p.gamma * p.X**2 + p.eps * p.X + (p.eta_tilde / p.tau) * np.sum(n**2)
    )


def variance_cost(p: ACParams, n: np.ndarray, sigmas: np.ndarray | None = None) -> float:
    """Variance of the shortfall: tau * sum_k sigma_k^2 x_k^2.

    ``sigmas`` optionally gives a volatility per interval (length N); by default the
    constant ``p.sigma`` is used.
    """
    x = holdings_from_trades(p, n)[1:]  # x_1 .. x_N
    s = np.full(p.N, p.sigma) if sigmas is None else np.asarray(sigmas, dtype=float)
    return float(p.tau * np.sum(s**2 * x**2))


def utility(p: ACParams, n: np.ndarray, lam: float, sigmas: np.ndarray | None = None) -> float:
    """Mean-variance objective E + lam * V."""
    return expected_cost(p, n) + lam * variance_cost(p, n, sigmas)


def expected_cost_closed_form(p: ACParams, lam: float) -> float:
    """Closed-form expected cost of the optimal trajectory (Almgren-Chriss, eq. 20)."""
    k = kappa(p, lam)
    if k * p.T < 1e-9:
        return expected_cost(p, twap_trades(p))
    kt, tau, T = k * p.T, p.tau, p.T
    return float(
        0.5 * p.gamma * p.X**2
        + p.eps * p.X
        + p.eta_tilde
        * p.X**2
        * np.tanh(0.5 * k * tau)
        * (tau * np.sinh(2 * kt) + 2 * T * np.sinh(k * tau))
        / (2 * tau**2 * np.sinh(kt) ** 2)
    )


def variance_closed_form(p: ACParams, lam: float) -> float:
    """Closed-form variance of the optimal trajectory (Almgren-Chriss, eq. 21)."""
    k = kappa(p, lam)
    if k * p.T < 1e-9:
        return variance_cost(p, twap_trades(p))
    kt, tau, T = k * p.T, p.tau, p.T
    return float(
        0.5
        * p.sigma**2
        * p.X**2
        * (tau * np.sinh(kt) * np.cosh(k * (T - tau)) - T * np.sinh(k * tau))
        / (np.sinh(kt) ** 2 * np.sinh(k * tau))
    )


def efficient_frontier(p: ACParams, lams: np.ndarray) -> dict[str, np.ndarray]:
    """Expected cost and standard deviation of the optimal strategy for each ``lam``."""
    lams = np.asarray(lams, dtype=float)
    e, v = np.empty_like(lams), np.empty_like(lams)
    for i, lam in enumerate(lams):
        n = optimal_trades(p, lam)
        e[i], v[i] = expected_cost(p, n), variance_cost(p, n)
    return {"lam": lams, "expected_cost": e, "variance": v, "std": np.sqrt(v)}


def optimal_holdings_time_varying(p: ACParams, lam: float, sigmas: np.ndarray) -> np.ndarray:
    """Optimal holdings when volatility changes from interval to interval.

    The first-order conditions of ``E + lam * V`` form the tridiagonal system::

        -x_{j-1} + (2 + lam * tau^2 * sigma_j^2 / eta_tilde) * x_j - x_{j+1} = 0,   j = 1..N-1

    with ``x_0 = X`` and ``x_N = 0``. ``sigmas[j-1]`` is the volatility of the interval ending at ``t_j``.
    With constant ``sigmas`` it reproduces :func:`optimal_holdings`.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.shape != (p.N,):
        raise ValueError(f"sigmas must have length N={p.N}")
    m = p.N - 1  # unknowns x_1 .. x_{N-1}
    if m == 0:
        return np.array([p.X, 0.0])
    diag = 2.0 + lam * p.tau**2 * sigmas[:m] ** 2 / p.eta_tilde
    ab = np.zeros((3, m))
    ab[0, 1:] = -1.0
    ab[1, :] = diag
    ab[2, :-1] = -1.0
    rhs = np.zeros(m)
    rhs[0] = p.X  # contribution of x_0 = X
    x_inner = solve_banded((1, 1), ab, rhs)
    return np.concatenate([[p.X], x_inner, [0.0]])


def optimal_trades_time_varying(p: ACParams, lam: float, sigmas: np.ndarray) -> np.ndarray:
    return trades_from_holdings(optimal_holdings_time_varying(p, lam, sigmas))
