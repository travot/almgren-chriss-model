"""Full study on real Binance 1-minute klines.

    python scripts/download_binance.py --symbol BTCUSDT --interval 1m --start 2025-01 --end 2025-06
    python scripts/run_real_data.py --symbol BTCUSDT

Writes figures and ``summary.md`` to ``results/``. The summary reports numbers and mechanical
readouts only: interpreting them (and deciding which limits of the model are actually visible in
the data) is left to the reader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optexec import ac, calibration as cal, data, plots, simulate, volatility as vol  # noqa: E402


def md_table(df: pd.DataFrame, fmt: str = "{:.4g}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [fmt.format(v) if isinstance(v, (int, float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--horizon", type=int, default=60, help="order horizon in minutes (one slice per minute)")
    ap.add_argument("--participation", type=float, default=0.10, help="order size as a fraction of median volume over the horizon")
    ap.add_argument("--kappa-T", type=float, default=2.0, help="urgency of the Almgren-Chriss schedules")
    ap.add_argument("--half-spread-bps", type=float, default=0.0)
    args = ap.parse_args()

    out = Path(args.out)
    fig = out / "figures"
    files = sorted(Path(args.raw_dir).glob(f"{args.symbol}-1m-*.zip")) + sorted(Path(args.raw_dir).glob(f"{args.symbol}-1m-*.csv"))
    if not files:
        sys.exit(f"No {args.symbol}-1m-*.zip files in {args.raw_dir}. Run scripts/download_binance.py first.")

    bars = data.clean_bars(data.read_klines(files))
    r = cal.log_returns(bars["close"])
    q = cal.signed_imbalance(bars).loc[r.index]
    lines = [f"# Real-data results: {args.symbol}\n",
             f"Period: {bars.index[0]:%Y-%m-%d} to {bars.index[-1]:%Y-%m-%d} ({len(bars):,} one-minute bars, {len(files)} files).\n"]

    # ---- impact
    lin = cal.fit_linear_impact(q.to_numpy(), r.to_numpy())
    pw = cal.fit_power_impact(q.to_numpy(), r.to_numpy())
    rev = cal.fit_reversion(q, r)
    plots.plot_impact_fit(pw, lin, fig / "impact_fit.png", title=f"{args.symbol}: price response to signed flow")
    z = (1.0 - pw.delta) / pw.delta_se if pw.delta_se > 0 else float("nan")
    impact_table = pd.DataFrame(
        {"value": [lin.slope, lin.tstat, lin.r2, pw.delta, pw.delta_se, pw.r2_loglog, rev.lam_total.slope,
                   rev.next_bar.slope, rev.lam_temporary, rev.lam_permanent]},
        index=pd.Index(["linear lambda (return per unit)", "linear t-stat", "linear R2", "power-law delta", "delta std error",
                        "power-law log-log R2", "contemporaneous impact", "next-bar coefficient (negative = reversion)",
                        "temporary part", "permanent part"], name="quantity"),
    )
    lines += [
        "\n## Market impact (aggregate order flow, 1-minute bars)\n",
        md_table(impact_table),
        f"\nReadout: delta = {pw.delta:.3f} ± {pw.delta_se:.3f}; the distance from 1 (linear) is {z:.1f} standard errors. "
        "Caveat: aggregate flow mixes price impact with information, see `optexec.calibration`.\n",
        "\n![Impact](figures/impact_fit.png)\n",
    ]

    # ---- volatility
    profile = cal.intraday_variance_profile(r)
    plots.plot_vol_profile(profile, fig / "vol_profile.png")
    table = vol.evaluate_horizon_forecasts(r, horizon=60, train_frac=0.6)
    scores = vol.score_forecasts(table)
    plots.plot_vol_scores(scores, fig / "vol_scores.png")
    gf = table.attrs["garch_fit"]
    lines += [
        "\n## Volatility\n",
        f"Ratio of the highest to the lowest intraday variance bucket: {profile.max() / profile.min():.2f}.\n",
        "\nOut-of-sample forecasts of the 60-minute variance (GARCH fitted on the first 60% of the sample):\n",
        md_table(scores.rename_axis("method")),
        f"\nGARCH(1,1): alpha = {gf.alpha:.3f}, beta = {gf.beta:.3f}, persistence = {gf.persistence:.4f}.\n",
        "\n![Profile](figures/vol_profile.png)\n![Scores](figures/vol_scores.png)\n",
    ]

    # ---- Almgren-Chriss with the calibrated parameters
    horizon = args.horizon
    shares = args.participation * float(bars["volume"].median()) * horizon
    lam_perm, lam_temp = rev.lam_permanent, rev.lam_temporary
    notes = []
    tau = 1.0  # one slice per minute
    if lam_temp <= 0:
        lam_temp = 0.5 * rev.lam_total.slope
        notes.append("No reversion was detected, so the temporary impact is **assumed** to be half of the contemporaneous impact.")
    floor = 0.55 * lam_perm * tau
    if lam_temp <= floor:
        notes.append(f"The estimated temporary impact ({lam_temp:.3g}) violates the model's condition eta > gamma*tau/2; it is **floored** at {floor:.3g}. "
                     "This is itself a finding about the model on this data.")
        lam_temp = floor
    sigma_bar = float(r.std())
    p = ac.ACParams(X=shares, T=float(horizon), N=horizon, sigma=sigma_bar, gamma=lam_perm, eta=lam_temp,
                    eps=args.half_spread_bps * 1e-4, S0=1.0)  # prices relative to the arrival price
    plots.plot_trajectories(p, [0, 1, 2, 4, 8], fig / "trajectories.png")
    notional = p.X * p.S0
    rows = {}
    for label, kT in [("TWAP", 0.0), (f"Almgren-Chriss (kappa T = {args.kappa_T:g})", args.kappa_T)]:
        n = ac.optimal_trades(p, ac.lambda_for_kappa(p, kT))
        sf = simulate.summarize(simulate.simulate_shortfall(p, n, n_paths=100_000, seed=0, df=5))
        rows[label] = {"mean_bps": 1e4 * sf.mean / notional, "std_bps": 1e4 * sf.std / notional, "cvar95_bps": 1e4 * sf.cvar95 / notional}
    lam_ac = ac.lambda_for_kappa(p, args.kappa_T)
    long_p = ac.ACParams(X=shares, T=480.0, N=480, sigma=sigma_bar, gamma=lam_perm, eta=lam_temp, eps=p.eps, S0=1.0)
    lam_long = ac.lambda_for_kappa(long_p, args.kappa_T)
    fixed = 0.5 * long_p.gamma * long_p.X**2 + long_p.eps * long_p.X
    tv_rows = {}
    for label, start in [("starts 02:00 UTC", 120), ("starts 06:00 UTC", 360), ("starts 12:00 UTC", 720)]:
        sig = cal.sigma_schedule(profile, sigma_bar, start, long_p.N, long_p.tau, 30)
        u_c = ac.utility(long_p, ac.optimal_trades(long_p, lam_long), lam_long, sig) - fixed
        u_t = ac.utility(long_p, ac.optimal_trades_time_varying(long_p, lam_long, sig), lam_long, sig) - fixed
        tv_rows[label] = {"improvement_pct": 100 * (1 - u_t / u_c)}
    lines += [
        "\n## Almgren-Chriss with the calibrated parameters\n",
        f"Order: {shares:.3g} units ({args.participation:.0%} of the median volume over {horizon} minutes), one slice per minute. "
        f"Parameters relative to price: permanent {lam_perm:.3g}, temporary {lam_temp:.3g}, sigma per minute {sigma_bar:.3g}. "
        "Shortfall by Monte Carlo with Student-t(5) shocks, in bps of notional:\n",
    ]
    lines += [f"- {n}" for n in notes]
    lines += [md_table(pd.DataFrame(rows).T.rename_axis("strategy"), "{:.2f}"),
              "\nGain from following the intraday volatility profile on an 8-hour order (controllable part of the objective):\n",
              md_table(pd.DataFrame(tv_rows).T.rename_axis("window"), "{:.2f}"),
              "\n![Trajectories](figures/trajectories.png)\n"]

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Done. See {out / 'summary.md'}")


if __name__ == "__main__":
    main()
