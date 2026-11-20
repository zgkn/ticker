"""Download long daily history + earnings dates for one ticker, for the pattern study.

Writes study/data/<TICKER>_daily.csv and study/data/<TICKER>_earnings.csv.
Separate from the dashboard: nothing under docs/ is touched.

Prices are split/dividend adjusted (data_lib.fetch_prices, auto_adjust=True).
Earnings dates come from Yahoo via yfinance and can be missing; the price file is
still written when they are.

Usage: python scripts/study_fetch.py NVDA 6y
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_lib  # noqa: E402


def main(argv: list[str]) -> int:
    symbol = data_lib.normalize_symbol(argv[1] if len(argv) > 1 else "NVDA")
    period = argv[2] if len(argv) > 2 else "6y"
    if period not in {"3y", "5y", "6y", "7y", "10y", "max"}:
        raise SystemExit(f"unsupported period {period!r}")

    out = Path("study/data")
    out.mkdir(parents=True, exist_ok=True)

    prices = data_lib.fetch_prices(symbol, period=period)
    prices.index.name = "date"
    prices.to_csv(out / f"{symbol}_daily.csv", float_format="%.4f")
    print(f"{symbol}: {len(prices)} daily bars, {prices.index[0].date()} to {prices.index[-1].date()}")

    try:
        import yfinance as yf

        ed = data_lib._retry(lambda: yf.Ticker(symbol).get_earnings_dates(limit=60), f"earnings dates {symbol}")
        if ed is None or ed.empty:
            raise RuntimeError("no earnings dates returned")
        dates = pd.DatetimeIndex(ed.index).tz_localize(None).normalize().unique().sort_values()
        pd.DataFrame({"earnings_date": dates.strftime("%Y-%m-%d")}).to_csv(out / f"{symbol}_earnings.csv", index=False)
        print(f"{symbol}: {len(dates)} earnings dates, {dates[0].date()} to {dates[-1].date()}")
    except Exception as exc:  # noqa: BLE001 - earnings are optional context
        print(f"warning: earnings dates unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
