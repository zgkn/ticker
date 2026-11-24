"""Download daily history for the backtest universe, many tickers in one run.

Writes backtest/data/<TICKER>_daily.csv for each ticker and backtest/data/manifest.csv
(first date, last date, rows, status). Separate from the dashboard and from the pattern
study: nothing under docs/ or study/ is touched.

Prices are split/dividend adjusted (auto_adjust=True), same as the study data.
A ticker that fails is recorded in the manifest and the run carries on; the run only
fails if every ticker failed.

Usage:
    python scripts/backtest_fetch.py                      # universe file, start 2014-01-01
    python scripts/backtest_fetch.py --start 2014-01-01 --tickers "AAPL,MSFT"
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_lib  # noqa: E402

UNIVERSE = Path("backtest/universe.txt")
OUT = Path("backtest/data")
PAUSE_SECONDS = 1.5  # be gentle with Yahoo; 42 tickers


def read_universe(path: Path) -> list[str]:
    lines = [ln.split("#")[0].strip() for ln in path.read_text().splitlines()]
    return data_lib.parse_symbols([ln for ln in lines if ln])


def fetch_since(symbol: str, start: str) -> pd.DataFrame:
    import yfinance as yf

    def go() -> pd.DataFrame:
        df = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError(f"no price data returned for {symbol}")
        return df

    df = data_lib._retry(go, f"prices {symbol}")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"
    return df


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--tickers", default="", help="comma list; empty = the universe file")
    ap.add_argument("--universe", default=str(UNIVERSE), help="universe file (must sit under backtest*/)")
    ap.add_argument("--out", default=str(OUT), help="output folder (must sit under backtest*/)")
    args = ap.parse_args(argv[1:])

    try:
        start = date.fromisoformat(args.start).isoformat()
    except ValueError:
        raise SystemExit(f"bad --start {args.start!r}, expected YYYY-MM-DD")

    for label, val in (("--universe", args.universe), ("--out", args.out)):
        pth = Path(val)
        if pth.is_absolute() or ".." in pth.parts or not pth.parts[0].startswith("backtest"):
            raise SystemExit(f"{label} must be a relative path under backtest*/, got {val!r}")
    universe, out = Path(args.universe), Path(args.out)

    symbols = data_lib.parse_symbols([args.tickers]) if args.tickers.strip() else read_universe(universe)
    if not symbols:
        raise SystemExit("no tickers to fetch")

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, sym in enumerate(symbols, 1):
        try:
            df = fetch_since(sym, start)
            df.to_csv(out / f"{sym}_daily.csv", float_format="%.4f")
            rows.append(dict(ticker=sym, status="ok", first=df.index[0].date(), last=df.index[-1].date(),
                             rows=len(df), zero_volume_days=int((df["volume"] == 0).sum())))
            print(f"[{n}/{len(symbols)}] {sym}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
        except Exception as exc:  # noqa: BLE001 - record and carry on
            rows.append(dict(ticker=sym, status=f"FAILED: {exc}", first="", last="", rows=0, zero_volume_days=0))
            print(f"[{n}/{len(symbols)}] {sym}: FAILED ({exc})")
        time.sleep(PAUSE_SECONDS)

    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"done: {ok} ok, {len(rows) - ok} failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
