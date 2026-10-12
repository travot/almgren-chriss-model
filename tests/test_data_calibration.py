import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from optexec import calibration as cal
from optexec import data


# ---- loaders -----------------------------------------------------------------------------------


def test_timestamp_units_are_detected():
    base = pd.Timestamp("2025-03-01", tz="UTC")
    for unit, factor in [("s", 1), ("ms", 1_000), ("us", 1_000_000)]:
        ts = pd.Series([int(base.timestamp() * factor), int((base + pd.Timedelta(minutes=1)).timestamp() * factor)])
        assert data.to_datetime_utc(ts)[0] == base, unit


def _kline_csv(header: bool) -> str:
    rows = ["1735689600000000,100,101,99,100.5,10,1735689659999999,1005,50,6,603,0",
            "1735689660000000,100.5,102,100,101,12,1735689719999999,1212,60,7,707,0"]
    head = ["open_time,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base,taker_buy_quote,ignore"]
    return "\n".join((head if header else []) + rows) + "\n"


@pytest.mark.parametrize("header", [False, True])
def test_read_klines_with_and_without_header_and_zip(tmp_path, header):
    csv = tmp_path / "BTCUSDT-1m-2025-01.csv"
    csv.write_text(_kline_csv(header))
    zpath = tmp_path / "BTCUSDT-1m-2025-01.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(csv, csv.name)
    for path in (csv, zpath):
        df = data.read_klines(path)
        assert list(df.index) == [pd.Timestamp("2025-01-01 00:00", tz="UTC"), pd.Timestamp("2025-01-01 00:01", tz="UTC")]
        assert df["close"].tolist() == [100.5, 101.0]
        assert df["taker_buy_base"].tolist() == [6.0, 7.0]


def test_read_aggtrades_signs_flow(tmp_path):
    csv = tmp_path / "agg.csv"
    csv.write_text("1,100,2,1,1,1735689600000000,True,True\n2,101,3,2,2,1735689601000000,False,True\n")
    df = data.read_aggtrades(csv)
    # buyer is maker -> aggressor sold (negative)
    assert df["signed_qty"].tolist() == [-2.0, 3.0]
    bars = data.trades_to_bars(df, "1min")
    assert bars["volume"].iloc[0] == 5.0
    assert bars["taker_buy_base"].iloc[0] == 3.0
    assert (2 * bars["taker_buy_base"] - bars["volume"]).iloc[0] == pytest.approx(1.0)


def test_clean_bars_fills_short_gaps_only():
    idx = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:01", "2025-01-01 00:04", "2025-01-01 01:00"], utc=True)
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0], "volume": 1.0}, index=idx)
    out = data.clean_bars(df, max_gap=5)
    assert out.loc["2025-01-01 00:03", "close"] == 2.0  # filled
    assert "2025-01-01 00:30" not in out.index.strftime("%Y-%m-%d %H:%M")  # long gap dropped
    assert out.loc["2025-01-01 00:02", "volume"] == 0.0


# ---- synthetic market and estimators ---------------------------------------------------------------


@pytest.fixture(scope="module")
def market():
    return data.synthetic_market(n_days=45, seed=3)


def test_synthetic_market_is_valid(market):
    bars, truth = market
    assert bars["close"].gt(0).all()
    assert (bars["taker_buy_base"] <= bars["volume"]).all() and (bars["taker_buy_base"] >= 0).all()
    assert truth["delta"] == 0.5


def test_power_law_exponent_is_recovered(market):
    bars, truth = market
    r = cal.log_returns(bars["close"])
    q = cal.signed_imbalance(bars).loc[r.index]
    fit = cal.fit_power_impact(q.to_numpy(), r.to_numpy())
    assert fit.delta == pytest.approx(truth["delta"], abs=0.08)
    assert fit.r2_loglog > 0.95
    lin = cal.fit_linear_impact(q.to_numpy(), r.to_numpy())
    assert lin.tstat > 10 and lin.r2 < 0.5  # a linear model explains much less


def test_reversion_share_is_recovered(market):
    bars, truth = market
    r = cal.log_returns(bars["close"])
    q = cal.signed_imbalance(bars).loc[r.index]
    rev = cal.fit_reversion(q, r)
    assert rev.lam_temporary / rev.lam_total.slope == pytest.approx(truth["reversion_share"], abs=0.06)
    assert rev.lam_permanent + rev.lam_temporary == pytest.approx(rev.lam_total.slope, rel=1e-6)


def test_ols_through_origin_recovers_slope():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    y = 2.5 * x + rng.standard_normal(5000) * 0.1
    res = cal.ols_through_origin(x, y)
    assert res.slope == pytest.approx(2.5, abs=0.02)
    assert res.r2 > 0.99


def test_intraday_profile_correlates_with_truth(market):
    bars, truth = market
    r = cal.log_returns(bars["close"])
    prof = cal.intraday_variance_profile(r)
    assert prof.mean() == pytest.approx(1.0)
    true_var = pd.Series(truth["profile"] ** 2).groupby(np.arange(len(truth["profile"])) // 30 * 30 % 1440).mean()
    true_var = (true_var / true_var.mean()).reindex(prof.index)
    assert np.corrcoef(prof, true_var)[0, 1] > 0.9


def test_sigma_schedule_maps_buckets():
    prof = pd.Series({0: 4.0, 30: 1.0}, dtype=float)
    sig = cal.sigma_schedule(prof, 1.0, start_minute_of_day=0, n_intervals=4, interval_minutes=15, bucket_minutes=30)
    assert sig.tolist() == [2.0, 2.0, 1.0, 1.0]


def test_ac_params_conversion_units():
    p = cal.ac_params_from_calibration(
        price=100.0, sigma_ret_per_min=1e-3, lam_permanent=1e-5, lam_temporary=4e-5,
        shares=10.0, horizon_min=30, n_intervals=30,
    )
    assert p.sigma == pytest.approx(0.1)
    assert p.gamma == pytest.approx(1e-3)
    assert p.eta == pytest.approx(4e-3)
