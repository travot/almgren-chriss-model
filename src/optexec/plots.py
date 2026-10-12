"""Figures. Each function saves a PNG and returns the path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import ac
from .calibration import PowerFit, OLSResult

plt.rcParams.update(
    {
        "figure.dpi": 130, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "legend.frameon": False,
    }
)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_trajectories(p: ac.ACParams, kappa_Ts: list[float], path) -> Path:
    """Optimal holdings for several levels of urgency (kappa * T = 0 is TWAP)."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    t = p.times
    for kT in kappa_Ts:
        lam = ac.lambda_for_kappa(p, kT)
        x = ac.optimal_holdings(p, lam) / p.X
        ax.plot(t, x, label="TWAP ($\\kappa T=0$)" if kT == 0 else f"$\\kappa T={kT:g}$")
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("fraction of the order still held")
    ax.set_title("Optimal liquidation trajectories")
    ax.legend()
    return _save(fig, path)


def plot_frontier(p: ac.ACParams, path, mc_points: dict[str, tuple[float, float]] | None = None) -> Path:
    """Efficient frontier: expected cost versus standard deviation (both per unit of notional, in bps)."""
    lams = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 80)]) * ac.lambda_for_kappa(p, 1.0)
    f = ac.efficient_frontier(p, lams)
    notional = p.X * p.S0
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(1e4 * f["std"] / notional, 1e4 * f["expected_cost"] / notional, color="C0", label="Almgren-Chriss frontier")
    twap, imm = ac.twap_trades(p), ac.immediate_trades(p)
    for n, name, c in [(twap, "TWAP", "C1"), (imm, "Immediate", "C3")]:
        ax.scatter(1e4 * np.sqrt(ac.variance_cost(p, n)) / notional, 1e4 * ac.expected_cost(p, n) / notional, color=c, zorder=3, label=name)
    if mc_points:
        for name, (m, s) in mc_points.items():
            ax.scatter(1e4 * s / notional, 1e4 * m / notional, marker="x", color="k", zorder=4)
        ax.scatter([], [], marker="x", color="k", label="Monte Carlo")
    ax.set_xlabel("std of shortfall (bps)")
    ax.set_ylabel("expected shortfall (bps)")
    ax.set_title("Efficient frontier")
    ax.legend()
    return _save(fig, path)


def plot_time_varying(minutes: np.ndarray, sigmas: np.ndarray, n_const: np.ndarray, n_tv: np.ndarray, path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True)
    axes[0].plot(minutes, sigmas, color="C2")
    axes[0].set_ylabel("volatility per $\\sqrt{\\mathrm{min}}$")
    axes[0].set_title("Intraday volatility and the schedules it implies")
    axes[1].step(minutes, n_const / n_const.sum(), where="mid", label="constant volatility", color="C0")
    axes[1].step(minutes, n_tv / n_tv.sum(), where="mid", label="intraday volatility", color="C1")
    axes[1].set_xlabel("minutes after start")
    axes[1].set_ylabel("share of order per slice")
    axes[1].legend()
    return _save(fig, path)


def plot_impact_fit(fit: PowerFit, lin: OLSResult, path, title: str = "Price response to signed order flow") -> Path:
    """Binned mean signed return against |Q| on log-log axes, with power-law and linear fits."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    b = fit.bins[fit.bins["impact"] > 0]
    ax.scatter(b["q"], b["impact"], s=22, color="C0", label="binned data")
    qq = np.geomspace(b["q"].min(), b["q"].max(), 100)
    ax.plot(qq, fit.predict(qq), color="C3", label=f"power law, $\\delta={fit.delta:.2f}\\pm{fit.delta_se:.2f}$")
    ax.plot(qq, lin.slope * qq, color="C2", ls="--", label="linear (Kyle $\\lambda$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|signed flow Q| per bar (base units)")
    ax.set_ylabel("mean signed return per bar")
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_vol_profile(profile: pd.Series, path, truth: np.ndarray | None = None, bucket_minutes: int = 30) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    hours = profile.index.to_numpy() / 60.0
    ax.step(hours, profile.to_numpy(), where="post", label="estimated", color="C0")
    if truth is not None:
        ax.plot(np.arange(len(truth)) / 60.0, truth, color="C3", alpha=0.7, label="true (synthetic)")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("hour of day (UTC)")
    ax.set_ylabel("relative variance")
    ax.set_title("Intraday variance profile")
    ax.legend()
    return _save(fig, path)


def plot_vol_scores(scores: pd.DataFrame, path) -> Path:
    """QLIKE of each volatility forecast (lower is better)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(scores.index, scores["QLIKE"], color=["C7", "C1", "C0"][: len(scores)])
    ax.set_ylabel("QLIKE (lower is better)")
    ax.set_title("Out-of-sample 60-minute variance forecasts")
    return _save(fig, path)


def plot_mc_check(p: ac.ACParams, kappa_Ts: list[float], mc: dict[float, tuple[float, float, float]], path) -> Path:
    """Closed form against Monte Carlo: expected cost and standard deviation (bps)."""
    notional = p.X * p.S0
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    th_e = [1e4 * ac.expected_cost(p, ac.optimal_trades(p, ac.lambda_for_kappa(p, k))) / notional for k in kappa_Ts]
    th_s = [1e4 * np.sqrt(ac.variance_cost(p, ac.optimal_trades(p, ac.lambda_for_kappa(p, k)))) / notional for k in kappa_Ts]
    mc_e = [1e4 * mc[k][0] / notional for k in kappa_Ts]
    mc_se = [1e4 * 1.96 * mc[k][2] / notional for k in kappa_Ts]
    mc_s = [1e4 * mc[k][1] / notional for k in kappa_Ts]
    axes[0].plot(kappa_Ts, th_e, "-", label="theory")
    axes[0].errorbar(kappa_Ts, mc_e, yerr=mc_se, fmt="o", ms=4, label="Monte Carlo (95% CI)")
    axes[0].set_title("expected shortfall (bps)")
    axes[1].plot(kappa_Ts, th_s, "-", label="theory")
    axes[1].plot(kappa_Ts, mc_s, "o", ms=4, label="Monte Carlo")
    axes[1].set_title("std of shortfall (bps)")
    for a in axes:
        a.set_xlabel("urgency $\\kappa T$")
    axes[0].legend()
    return _save(fig, path)
