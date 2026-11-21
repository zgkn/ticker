"""Scan the screener universe for chart patterns that completed in the last N trading days.

Uses the project's coded detectors (scripts/chartpatterns.py) on 3y of daily bars.
Educational use only: per the pattern library, no pattern is "Workable" yet, so this
builds a watchlist, not a buy list.

Usage: python scripts/pattern_scan.py --lookback 10 --min-price 10 --max-price 100
Writes study/scan/pattern_scan.json and study/scan/pattern_scan.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chartpatterns  # noqa: E402
import screener  # noqa: E402

log = logging.getLogger("pattern_scan")
MIN_BARS = 260
MIN_AVG_VOLUME = 500_000


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=float, default=10)
    ap.add_argument("--min-price", type=float, default=10)
    ap.add_argument("--max-price", type=float, default=100)
    ap.add_argument("--bullish-only", default="true")
    a = ap.parse_args(argv[1:])
    lookback, bull_only = int(a.lookback), str(a.bullish_only).lower() == "true"

    symbols = screener.fetch_universe()
    if not symbols:
        raise SystemExit("empty universe")
    prices = screener.download_prices(symbols, period="3y")
    log.info("downloaded %d of %d tickers", len(prices), len(symbols))

    rows, scanned = [], 0
    for sym, df in prices.items():
        if len(df) < MIN_BARS:
            continue
        last = float(df.close.iloc[-1])
        if not a.min_price <= last <= a.max_price or df.volume.tail(50).mean() < MIN_AVG_VOLUME:
            continue
        scanned += 1
        try:
            found = chartpatterns.detect_all(df)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: detector failed (%s)", sym, exc)
            continue
        n = len(df)
        for x in found:
            ago = n - 1 - x["done"]
            if ago > lookback or (bull_only and x["direction"] != "bull"):
                continue
            lvl = x.get("level") or x.get("floor")
            sma200 = float(df.close.tail(200).mean())
            rows.append(dict(
                symbol=sym, pattern=x["pattern"], direction=x["direction"], date=x["date"], bars_ago=ago,
                entry_close=round(float(df.close.iloc[x["done"]]), 2), last_close=round(last, 2),
                move_since=round(last / float(df.close.iloc[x["done"]]) - 1, 4),
                level=round(float(lvl), 2) if lvl else None,
                pct_off_52w_high=round(last / float(df.close.tail(252).max()) - 1, 4),
                above_200d=bool(last > sma200), avg_volume=int(df.volume.tail(50).mean()),
            ))
    rows.sort(key=lambda r: (r["bars_ago"], r["symbol"]))
    out = Path("study/scan")
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                as_of=str(max(d.index[-1] for d in prices.values()).date()),
                params=dict(lookback=lookback, min_price=a.min_price, max_price=a.max_price, bullish_only=bull_only),
                funnel=dict(universe=len(symbols), downloaded=len(prices), scanned=scanned, detections=len(rows)))
    (out / "pattern_scan.json").write_text(json.dumps(dict(meta, rows=rows), indent=1))
    pd.DataFrame(rows).to_csv(out / "pattern_scan.csv", index=False)
    # recent bars for every detected symbol, so charts can be drawn without re-downloading
    keep = {}
    for sym in sorted({r["symbol"] for r in rows}):
        t = prices[sym].tail(160)
        keep[sym] = dict(d=[str(i.date()) for i in t.index], c=[round(float(v), 3) for v in t.close],
                         v=[int(v) for v in t.volume])
    (out / "pattern_scan_prices.json").write_text(json.dumps(keep, separators=(",", ":")))
    print(json.dumps(meta["funnel"]))
    for r in rows[:40]:
        print(r["symbol"], r["pattern"], r["date"], r["entry_close"], r["last_close"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
