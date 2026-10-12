"""End-to-end smoke test: write fake files in the exact Binance format and run the real-data script."""

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from optexec import data

ROOT = Path(__file__).resolve().parents[1]


def test_real_data_pipeline_runs_on_binance_format_files(tmp_path):
    bars, _ = data.synthetic_market(n_days=20, seed=12)
    raw = tmp_path / "raw"
    raw.mkdir()
    for (y, m), g in bars.groupby([bars.index.year, bars.index.month]):
        ts = (g.index.astype("int64") // 1000).to_numpy()  # microseconds, as in 2025 spot files
        df = pd.DataFrame({"a": ts, "o": g.open, "h": g.high, "l": g.low, "c": g.close, "v": g.volume, "ct": ts + 59_999_999,
                           "qv": g.quote_volume, "n": g.trades, "tb": g.taker_buy_base, "tq": g.taker_buy_base * g.close, "ig": 0})
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False)
        with zipfile.ZipFile(raw / f"BTCUSDT-1m-{y}-{m:02d}.zip", "w") as z:
            z.writestr(f"BTCUSDT-1m-{y}-{m:02d}.csv", buf.getvalue())
    out = tmp_path / "out"
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_real_data.py"), "--raw-dir", str(raw), "--out", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, res.stderr
    summary = (out / "summary.md").read_text()
    assert "power-law delta" in summary and "Almgren-Chriss with the calibrated parameters" in summary
    assert (out / "figures" / "impact_fit.png").exists()
