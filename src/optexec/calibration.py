"""Estimation of volatility and market impact from bar data.

Important caveat on impact
--------------------------
Public bar data contains the *aggregate* order flow of the whole market, not the effect of
one trader's own orders. The relationship between signed flow and returns estimated here
mixes genuine price impact with information (informed traders cause both flow and price
moves). The estimates are therefore only an indication of the *shape* (linear or concave)
and *order of magnitude* of impact, not an identified causal impact for an execution desk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ac import ACParams


# --------------------------------------------------------------------------------------
# Returns, volatility and intraday seasonality
# --------------------------------------------------------------------------------------


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()


def realized_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling realised volatility per bar (square root of the mean of squared returns)."""
    return np.sqrt((returns**2).rolling(window).mean())


def intraday_variance_profile(returns: pd.Series, bucket_minutes: int = 30) -> pd.Series:
    """Relative variance by time of day: mean squared return per bucket, normalised to mean 1.

    Index is the minute of the day at which each bucket starts (UTC).
    """
    minute = returns.index.hour * 60 + returns.index.minute
    bucket = (minute // bucket_minutes) * bucket_minutes
    var = (returns**2).groupby(bucket).mean()
    return var / var.mean()


def sigma_schedule(
    profile: pd.Series,
    sigma_bar: float,
    start_minute_of_day: int,
    n_intervals: int,
    interval_minutes: float,
    bucket_minutes: int = 30,
) -> np.ndarray:
    """Volatility per interval (price-return units per sqrt(minute)) implied by the profile.

    ``sigma_bar`` is the average per-minute volatility. Interval ``k`` is assigned the profile
    value of the bucket containing its midpoint.
    """
    mids = start_minute_of_day + (np.arange(n_intervals) + 0.5) * interval_minutes
    buckets = ((mids % (24 * 60)) // bucket_minutes * bucket_minutes).astype(int)
    rel = profile.reindex(buckets).to_numpy()
    rel = np.where(np.isnan(rel), 1.0, rel)
    return sigma_bar * np.sqrt(rel)


# --------------------------------------------------------------------------------------
# Impact
# --------------------------------------------------------------------------------------


def signed_imbalance(bars: pd.DataFrame) -> pd.Series:
    """Signed taker flow per bar in base units: buys minus sells = 2 * taker_buy - volume."""
    return 2.0 * bars["taker_buy_base"] - bars["volume"]


@dataclass(frozen=True)
class OLSResult:
    slope: float
    stderr: float
    r2: float
    n: int

    @property
    def tstat(self) -> float:
        return self.slope / self.stderr if self.stderr > 0 else float("nan")


def ols_through_origin(x: np.ndarray, y: np.ndarray) -> OLSResult:
    """Least squares of y on x without intercept (heteroskedasticity-robust standard error)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxx = np.sum(x * x)
    slope = float(np.sum(x * y) / sxx)
    resid = y - slope * x
    # White (HC0) variance of the slope
    se = float(np.sqrt(np.sum((x * resid) ** 2)) / sxx)
    r2 = float(1.0 - np.sum(resid**2) / np.sum(y**2))
    return OLSResult(slope, se, r2, int(x.size))


def fit_linear_impact(q: np.ndarray, r: np.ndarray) -> OLSResult:
    """Linear impact r = lambda * Q (Kyle's lambda), no intercept."""
    return ols_through_origin(q, r)


@dataclass(frozen=True)
class PowerFit:
    """Fit of E[sign(Q) r | |Q|] = c * |Q|**delta on binned data."""

    delta: float
    delta_se: float
    c: float
    r2_loglog: float
    bins: pd.DataFrame  # columns: q (mean |Q| per bin), impact (mean signed return), count

    def predict(self, q: np.ndarray) -> np.ndarray:
        return self.c * np.abs(q) ** self.delta


def fit_power_impact(q: np.ndarray, r: np.ndarray, n_bins: int = 25, trim: float = 0.0) -> PowerFit:
    """Estimate the impact exponent by log-log regression on quantile bins of ``|Q|``.

    Bins make the regression robust to the noise in individual returns. Bins whose mean signed
    return is not positive are dropped (log undefined); the number of kept bins is in ``bins``.
    """
    q, r = np.asarray(q, float), np.asarray(r, float)
    keep = q != 0
    q, r = q[keep], r[keep]
    aq, signed_r = np.abs(q), np.sign(q) * r
    if trim > 0:
        lo, hi = np.quantile(aq, [trim, 1 - trim])
        m = (aq >= lo) & (aq <= hi)
        aq, signed_r = aq[m], signed_r[m]
    edges = np.unique(np.quantile(aq, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.searchsorted(edges, aq, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"q": aq, "impact": signed_r, "bin": which})
    bins = df.groupby("bin").agg(q=("q", "mean"), impact=("impact", "mean"), count=("q", "size"))
    pos = bins[bins["impact"] > 0]
    if len(pos) < 4:
        raise ValueError("not enough bins with positive mean impact to fit a power law")
    lx, ly = np.log(pos["q"].to_numpy()), np.log(pos["impact"].to_numpy())
    design = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    resid = ly - design @ coef
    dof = max(len(lx) - 2, 1)
    cov = (resid @ resid / dof) * np.linalg.inv(design.T @ design)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    return PowerFit(
        delta=float(coef[1]),
        delta_se=float(np.sqrt(cov[1, 1])),
        c=float(np.exp(coef[0])),
        r2_loglog=float(1.0 - resid @ resid / ss_tot),
        bins=bins,
    )


@dataclass(frozen=True)
class ReversionFit:
    """Contemporaneous impact and the part that reverts in the next bar."""

    lam_total: OLSResult  # r_t on Q_t
    next_bar: OLSResult  # r_{t+1} on Q_t (negative slope = reversion)

    @property
    def lam_temporary(self) -> float:
        return max(-self.next_bar.slope, 0.0)

    @property
    def lam_permanent(self) -> float:
        return max(self.lam_total.slope - self.lam_temporary, 0.0)


def fit_reversion(q: pd.Series, r: pd.Series) -> ReversionFit:
    """Split contemporaneous impact into a permanent and a temporary (reverting) part."""
    df = pd.concat({"q": q, "r": r}, axis=1).dropna()
    df["r_next"] = df["r"].shift(-1)
    df = df.dropna()
    return ReversionFit(
        lam_total=ols_through_origin(df["q"].to_numpy(), df["r"].to_numpy()),
        next_bar=ols_through_origin(df["q"].to_numpy(), df["r_next"].to_numpy()),
    )


def ac_params_from_calibration(
    *,
    price: float,
    sigma_ret_per_min: float,
    lam_permanent: float,
    lam_temporary: float,
    shares: float,
    horizon_min: float,
    n_intervals: int,
    half_spread_frac: float = 0.0,
    bar_minutes: float = 1.0,
) -> ACParams:
    """Convert per-bar calibration results into Almgren-Chriss parameters.

    Units: time in minutes, quantities in base units (for example BTC), prices in quote units.

    * ``sigma``: ``price * sigma_ret_per_min`` (price units per sqrt(minute));
    * ``gamma``: ``price * lam_permanent`` (a bar's net flow ``Q`` moves the price by
      ``price * lam * Q``);
    * ``eta``: temporary price drop for trading ``n`` units over a bar of ``bar_minutes``
      is ``price * lam_temporary * n``, i.e. ``eta = price * lam_temporary * bar_minutes``
      per (unit / minute);
    * ``eps``: half spread in price units.
    """
    return ACParams(
        X=shares,
        T=horizon_min,
        N=n_intervals,
        sigma=price * sigma_ret_per_min,
        gamma=price * lam_permanent,
        eta=price * lam_temporary * bar_minutes,
        eps=price * half_spread_frac,
        S0=price,
    )
