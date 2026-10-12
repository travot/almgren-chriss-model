"""Download Binance public klines (free, no account) into data/raw.

    python scripts/download_binance.py --symbol BTCUSDT --interval 1m --start 2025-01 --end 2025-06

Files come from https://data.binance.vision (monthly archives). Each archive is verified against
the SHA-256 checksum published next to it. Existing files are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://data.binance.vision/data/spot/monthly/klines"


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            time.sleep(2 * (attempt + 1))
        except urllib.error.URLError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", required=True, help="first month, YYYY-MM")
    ap.add_argument("--end", required=True, help="last month, YYYY-MM")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = 0
    for y, m in months(args.start, args.end):
        name = f"{args.symbol}-{args.interval}-{y}-{m:02d}.zip"
        target = out / name
        if target.exists():
            print(f"skip   {name} (already present)")
            continue
        url = f"{BASE}/{args.symbol}/{args.interval}/{name}"
        try:
            blob = fetch(url)
            expected = fetch(url + ".CHECKSUM").decode().split()[0]
        except urllib.error.HTTPError as e:
            print(f"missing {name} (HTTP {e.code}); the month may not be published yet")
            status = 1
            continue
        if hashlib.sha256(blob).hexdigest() != expected:
            print(f"BAD CHECKSUM {name}; file not saved")
            status = 1
            continue
        target.write_bytes(blob)
        print(f"saved  {name} ({len(blob) / 1e6:.1f} MB)")
    return status


if __name__ == "__main__":
    sys.exit(main())
