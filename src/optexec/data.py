"""Market data: loaders for Binance public files and a synthetic market with known parameters.

Binance public data (https://data.binance.vision) is free and needs no account. Two file types
are used here:

* klines (OHLCV bars) with columns
  ``open_time, open, high, low, close, volume, close_time, quote_volume, trades,
  taker_buy_base, taker_buy_quote, ignore``;
* aggTrades with columns
  ``agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker[, is_best_match]``.

Timestamps are in milliseconds, except for spot files from 2025 onwards which are in
microseconds; :func:`to_datetime_utc` detects the unit automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
AGGTRADE_COLUMNS = [
    "agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker", "is_best_match",
]


def to_datetime_utc(ts: pd.Series) -> pd.DatetimeIndex:
    """Convert epoch timestamps in s, ms, us or ns to a UTC DatetimeIndex (unit inferred)."""
    ts = pd.to_numeric(ts)
    magnitude = ts.abs().max()
    if magnitude >= 1e17:
        unit = "ns"
    elif magnitude >= 1e14:
        unit = "us"
    elif magnitude >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.DatetimeIndex(pd.to_datetime(ts, unit=unit, utc=True))


def _read_csv_any(path: str | Path, names: list[str]) -> pd.DataFrame:
    """Read a (possibly zipped) Binance CSV, with or without a header row."""
    raw = pd.read_csv(path, header=None, compression="infer")
    first = str(raw.iloc[0, 0])
    if not first.lstrip("-").replace(".", "", 1).isdigit():  # header present
        raw = raw.iloc[1:].reset_index(drop=True)
    raw = raw.iloc[:, : len(names)]
    raw.columns = names[: raw.shape[1]]
    return raw


def read_klines(paths: str | Path | list[str | Path]) -> pd.DataFrame:
    """Load one or several kline files into a UTC-indexed DataFrame of floats."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = [_read_csv_any(p, KLINE_COLUMNS) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    index = to_datetime_utc(df["open_time"])
    out = df[["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]].astype(float)
    out.index = index
    out.index.name = "time"
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def read_aggtrades(path: str | Path) -> pd.DataFrame:
    """Load an aggTrades file. ``signed_qty`` is positive for buyer-initiated trades."""
    df = _read_csv_any(path, AGGTRADE_COLUMNS)
    out = pd.DataFrame(
        {
            "price": df["price"].astype(float),
            "qty": df["quantity"].astype(float),
            "buyer_maker": df["is_buyer_maker"].astype(str).str.lower().eq("true"),
        }
    )
    out.index = to_datetime_utc(df["transact_time"])
    out.index.name = "time"
    # the buyer is the maker -> the aggressor sold
    out["signed_qty"] = np.where(out["buyer_maker"], -out["qty"], out["qty"])
    return out.sort_index()


def trades_to_bars(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Aggregate trades into kline-like bars with signed flow."""
    g = trades.resample(freq)
    bars = pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "volume": g["qty"].sum(),
            "trades": g["qty"].count().astype(float),
        }
    )
    buy = trades["qty"].where(~trades["buyer_maker"], 0.0).resample(freq).sum()
    bars["taker_buy_base"] = buy
    bars["quote_volume"] = (trades["price"] * trades["qty"]).resample(freq).sum()
    return bars.dropna(subset=["close"])


def clean_bars(df: pd.DataFrame, freq: str = "1min", max_gap: int = 5) -> pd.DataFrame:
    """Reindex to a regular grid and forward-fill short gaps (up to ``max_gap`` bars).

    Bars with no trade have zero volume. Longer gaps are left as NaN and dropped.
    """
    full = df.reindex(pd.date_range(df.index[0], df.index[-1], freq=freq, tz="UTC"))
    full["close"] = full["close"].ffill(limit=max_gap)
    for c in ("volume", "quote_volume", "trades", "taker_buy_base"):
        if c in full:
            full[c] = full[c].fillna(0.0)
    for c in ("open", "high", "low"):
        if c in full:
            full[c] = full[c].fillna(full["close"])
    return full.dropna(subset=["close"])


# --------------------------------------------------------------------------------------
# Synthetic market with known parameters (used to validate the estimators)
# --------------------------------------------------------------------------------------


def synthetic_market(
    n_days: int = 60,
    seed: int = 0,
    base_price: float = 30_000.0,
    sigma_ret: float = 6e-4,
    delta: float = 0.5,
    impact_coeff: float = 8e-4,
    reversion_share: float = 0.5,
    base_volume: float = 40.0,
    intraday_amplitude: float = 0.6,
    vol_persistence: float = 0.995,
    vol_of_vol: float = 0.06,
) -> tuple[pd.DataFrame, dict]:
    """Simulate a 1-minute market where the truth is known.

    * Volatility = base * intraday U-shape * slowly varying stochastic factor.
    * Signed order flow ``Q`` is heavy-tailed (Student-t) and scaled by ``base_volume``.
    * The 1-minute return contains ``impact_coeff * sign(Q) * |Q / base_volume|**delta``;
      a share ``reversion_share`` of it is temporary and reverts in the next minute.

    Returns the bars (columns open/high/low/close/volume/taker_buy_base/quote_volume/trades)
    and a dictionary with the true parameters.
    """
    rng = np.random.default_rng(seed)
    n = n_days * 24 * 60
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    minute_of_day = (idx.hour * 60 + idx.minute).to_numpy()

    # intraday profile of volatility and volume: peak around 14:00 UTC
    phase = 2 * np.pi * (minute_of_day - 14 * 60) / (24 * 60)
    profile = 1.0 + intraday_amplitude * np.cos(phase)
    profile = profile / np.sqrt(np.mean(profile**2))  # mean square 1 -> sigma_ret is the RMS

    # slowly varying log-volatility factor (AR(1))
    log_f = np.zeros(n)
    shocks = rng.standard_normal(n) * vol_of_vol
    for t in range(1, n):
        log_f[t] = vol_persistence * log_f[t - 1] + shocks[t]
    factor = np.exp(log_f - 0.5 * np.var(log_f))
    sigma_t = sigma_ret * profile * factor

    # signed order flow, in coins
    scale = base_volume * np.sqrt(profile)
    q = rng.standard_t(4, size=n) / np.sqrt(2.0) * 0.5 * scale
    vol = np.abs(q) * 2.0 + scale * rng.lognormal(0.0, 0.5, size=n)
    q = np.clip(q, -0.95 * vol, 0.95 * vol)
    taker_buy = 0.5 * (vol + q)

    # returns: permanent part + temporary part that reverts next minute
    impact_ret = impact_coeff * np.sign(q) * np.abs(q / base_volume) ** delta
    temp = reversion_share * impact_ret
    ret = sigma_t * rng.standard_normal(n) + impact_ret
    ret[1:] -= temp[:-1]
    close = base_price * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[base_price], close[:-1]])
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close),
            "low": np.minimum(open_, close),
            "close": close,
            "volume": vol,
            "taker_buy_base": taker_buy,
            "quote_volume": vol * close,
            "trades": np.maximum(1.0, np.round(vol / 0.05)),
        },
        index=idx,
    )
    truth = {
        "sigma_ret": sigma_ret,
        "delta": delta,
        "impact_coeff": impact_coeff,
        "reversion_share": reversion_share,
        "base_volume": base_volume,
        "intraday_amplitude": intraday_amplitude,
        "profile": profile,
        "sigma_t": sigma_t,
    }
    return bars, truth
