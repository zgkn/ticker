"""Two-stage screener: scan a universe for CAN SLIM candidates under a price cap.

    python scripts/screener.py --max-price 100 --min-price 10

Stage 1 (price only, one bulk download for the whole universe) keeps stocks that
  - trade between --min-price and --max-price,
  - average at least MIN_AVG_VOLUME shares a day (50 sessions),
  - sit within --max-off-high percent of their 52-week high (default 25; 100 = no limit),
  - close above their 200-day average, and
  - have beaten SPY on the weighted 12-month return (the L proxy).
The best MAX_STAGE2 by outperformance go on to stage 2.

Stage 2 scores all seven CAN SLIM letters (fetching fundamentals, which is slow
and rate-limited, only for these survivors). The shortlist is every stock whose
C and A letters both PASS, ranked by letters passed, at most --top rows.

Output: docs/data/screener.json (the table, parameters and funnel counts) and
docs/data/screen/<SYMBOL>.json (full chart + scorecard for each shortlisted
stock; the folder is rebuilt on every run).

Universe: S&P 500 and S&P MidCap 400 constituents (Wikipedia tables) plus any
symbols in universe.txt.

CAVEATS:
  - Prices are dividend/split adjusted, so the price filter can differ slightly
    from the quoted price. The cap applies to the latest adjusted close.
  - Stage 1 is a cheap approximation of N, S and L; a stock that fails it is
    never scored on the earnings letters, so a strong grower far below its
    highs (a recovery play) cannot appear here by design.
  - Wikipedia's tables and Yahoo's bulk downloads can change or throttle. A
    failure to fetch a chunk or a ticker is logged and that ticker is skipped.
  - Same limits as the dashboard: L and I are proxies, fundamentals come from
    yfinance and are not cross-checked.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import build_site  # noqa: E402
import data_lib  # noqa: E402
import indicators as ind  # noqa: E402
import market  # noqa: E402

log = logging.getLogger("screener")

DEFAULT_MIN_PRICE = 10.0
DEFAULT_MAX_PRICE = 100.0
MIN_AVG_VOLUME = 500_000
MAX_OFF_HIGH = 0.25
MIN_BARS = 253           # weighted 12-month performance needs > 252 bars
MAX_STAGE2 = 60
DEFAULT_TOP = 20
CHUNK = 100

ROOT = os.path.join(os.path.dirname(__file__), "..")
UNIVERSE_FILE = os.path.join(ROOT, "universe.txt")
SCREENER_JSON = os.path.join(build_site.DATA_DIR, "screener.json")
SCREEN_DIR = os.path.join(build_site.DATA_DIR, "screen")

WIKI = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}
UA = "Mozilla/5.0 (compatible; ticker-dashboard)"
SYMBOL_COLS = ("Symbol", "Ticker symbol", "Ticker")


# --- universe ----------------------------------------------------------------

def parse_wikipedia_symbols(html: str) -> list[str]:
    """Constituent tickers from the first table that has a symbol column. BRK.B -> BRK-B (Yahoo style)."""
    for table in pd.read_html(io.StringIO(html)):
        col = next((c for c in SYMBOL_COLS if c in table.columns), None)
        if col is None:
            continue
        out = []
        for raw in table[col].dropna().astype(str):
            sym = raw.strip().upper().replace(".", "-")
            if data_lib.SYMBOL_RE.match(sym):
                out.append(sym)
        if out:
            return list(dict.fromkeys(out))
    return []


def read_universe_file(path: str = UNIVERSE_FILE) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return data_lib.parse_symbols([ln.split("#")[0].strip() for ln in fh if ln.split("#")[0].strip()])


def fetch_universe(path: str = UNIVERSE_FILE) -> list[str]:
    import requests

    symbols: list[str] = []
    for name, url in WIKI.items():
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            got = parse_wikipedia_symbols(r.text)
            log.info("universe: %s -> %d symbols", name, len(got))
            symbols += got
        except Exception as exc:  # noqa: BLE001 - one list failing should not stop the screen
            log.warning("universe: %s unavailable (%s)", name, exc)
    symbols += read_universe_file(path)
    return [s for s in dict.fromkeys(symbols) if s not in (data_lib.BENCHMARK, data_lib.SECOND_INDEX)]


# --- bulk prices -------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame | None:
    df = df.rename(columns=str.lower)
    if not {"open", "high", "low", "close", "volume"} <= set(df.columns):
        return None
    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    if df.empty:
        return None
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def split_download(raw: pd.DataFrame | None, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Split yfinance's bulk frame (columns = ticker x field) into one clean OHLCV frame per ticker."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        have = set(raw.columns.get_level_values(0))
        for s in symbols:
            if s in have:
                df = _clean(raw[s])
                if df is not None:
                    out[s] = df
    elif len(symbols) == 1:                      # yfinance flattens the columns for a single ticker
        df = _clean(raw)
        if df is not None:
            out[symbols[0]] = df
    return out


def download_prices(symbols: list[str], period: str = data_lib.HISTORY_PERIOD, chunk: int = CHUNK) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        try:
            raw = data_lib._retry(
                lambda: yf.download(part, period=period, interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False),
                f"bulk prices {i // chunk + 1}",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("bulk prices chunk %d failed (%s); skipping %d tickers", i // chunk + 1, exc, len(part))
            continue
        got = split_download(raw, part)
        log.info("bulk prices: chunk %d/%d -> %d of %d tickers", i // chunk + 1, -(-len(symbols) // chunk), len(got), len(part))
        out.update(got)
        time.sleep(1.0)
    return out


# --- stage 1 -----------------------------------------------------------------

def stage1(prices: dict[str, pd.DataFrame], spy: pd.DataFrame, min_price: float, max_price: float,
           min_volume: float = MIN_AVG_VOLUME, max_off_high: float = MAX_OFF_HIGH) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Cheap price/trend filter. Returns (survivors best-first, rejection counts by reason)."""
    spy_perf = ind.weighted_performance(spy["close"])
    rejected = {"too_little_history": 0, "price_out_of_range": 0, "low_volume": 0, "too_far_below_high": 0, "below_200d": 0, "lagging_spy": 0}
    keep: list[dict[str, Any]] = []
    for sym, df in prices.items():
        if len(df) < MIN_BARS or spy_perf is None:
            rejected["too_little_history"] += 1
            continue
        close = float(df["close"].iloc[-1])
        if not (min_price <= close <= max_price):
            rejected["price_out_of_range"] += 1
            continue
        avg_vol = float(df["volume"].tail(50).mean())
        if avg_vol < min_volume:
            rejected["low_volume"] += 1
            continue
        off = close / float(df["high"].tail(252).max()) - 1.0
        if off < -max_off_high:
            rejected["too_far_below_high"] += 1
            continue
        sma200 = ind.sma(df["close"], 200).iloc[-1]
        if pd.isna(sma200) or close <= sma200:
            rejected["below_200d"] += 1
            continue
        perf = ind.weighted_performance(df["close"])
        if perf is None or perf - spy_perf <= 0:
            rejected["lagging_spy"] += 1
            continue
        keep.append({"symbol": sym, "close": close, "avg_volume": avg_vol, "pct_off_high": off, "outperf": perf - spy_perf})
    keep.sort(key=lambda r: -r["outperf"])
    return keep, rejected


# --- stage 2 + shortlist -----------------------------------------------------

def stage2(survivors: list[dict[str, Any]], prices: dict[str, pd.DataFrame], spy: pd.DataFrame, qqq: pd.DataFrame | None,
           fundamentals_fn: Callable[[str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Score every CAN SLIM letter for the survivors. Returns (payloads, symbols that failed)."""
    payloads, failed = [], []
    for row in survivors:
        sym = row["symbol"]
        try:
            f = fundamentals_fn(sym)
            p = build_site.build_payload(sym, prices[sym], spy, qqq, f)
            p["screen"] = {"outperf": row["outperf"], "avg_volume": row["avg_volume"]}
            payloads.append(p)
            t = p["canslim"]["tally"]
            log.info("%s: %d/%d pass (C %s, A %s)", sym, t["pass"], t["scored"], p["canslim"]["letters"]["C"]["status"], p["canslim"]["letters"]["A"]["status"])
        except Exception:  # noqa: BLE001 - one ticker must not stop the screen
            log.exception("%s: scoring failed", sym)
            failed.append(sym)
    return payloads, failed


def shortlist(payloads: list[dict[str, Any]], top: int = DEFAULT_TOP) -> list[dict[str, Any]]:
    """C and A must both PASS; rank by letters passed, then by outperformance vs SPY."""
    ok = [p for p in payloads if p["canslim"]["letters"]["C"]["status"] == "PASS" and p["canslim"]["letters"]["A"]["status"] == "PASS"]
    ok.sort(key=lambda p: (-p["canslim"]["tally"]["pass"], -p["screen"]["outperf"]))
    return ok[:top]


def screener_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = build_site.summary_row(payload)
    row["outperf"] = payload["screen"]["outperf"]
    row["avg_volume"] = payload["screen"]["avg_volume"]
    return row


# --- orchestration -----------------------------------------------------------

def run(min_price: float, max_price: float, top: int = DEFAULT_TOP, max_stage2: int = MAX_STAGE2,
        out_json: str | None = None, out_dir: str | None = None, max_off_high: float = MAX_OFF_HIGH) -> int:
    out_json, out_dir = out_json or SCREENER_JSON, out_dir or SCREEN_DIR
    universe = fetch_universe()
    if not universe:
        log.error("empty universe: Wikipedia tables unavailable and universe.txt is empty")
        return 2
    log.info("universe: %d symbols; price %.0f-%.0f", len(universe), min_price, max_price)

    spy = data_lib.fetch_prices(data_lib.BENCHMARK)
    try:
        qqq = data_lib.fetch_prices(data_lib.SECOND_INDEX)
    except Exception as exc:  # noqa: BLE001
        log.warning("QQQ unavailable (%s)", exc)
        qqq = None

    prices = download_prices(universe)
    survivors, rejected = stage1(prices, spy, min_price, max_price, max_off_high=max_off_high)
    log.info("stage 1: %d of %d downloaded passed; rejected %s", len(survivors), len(prices), rejected)
    if not prices:
        log.error("no price data downloaded (Yahoo unavailable or throttled)")
        return 1

    scored, failed = stage2(survivors[:max_stage2], prices, spy, qqq, lambda s: data_lib.fetch_fundamentals(s, with_news=False))
    picks = shortlist(scored, top)
    mkt = market.evaluate({"SPY": spy, **({"QQQ": qqq} if qqq is not None else {})})
    log.info("stage 2: scored %d (%d failed); shortlist %d with C and A passing", len(scored), len(failed), len(picks))

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for p in picks:
        build_site.write_ticker(p, out_dir)
    build_site.write_json(out_json, {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": spy.index[-1].strftime("%Y-%m-%d"),
        "params": {"min_price": min_price, "max_price": max_price, "min_avg_volume": MIN_AVG_VOLUME, "max_off_high": max_off_high, "top": top, "max_stage2": max_stage2},
        "market": {"status": mkt["status"], "headline": mkt["headline"]},
        "funnel": {
            "universe": len(universe), "downloaded": len(prices), "stage1": len(survivors), "scored": len(scored),
            "scoring_failed": len(failed), "shortlist": len(picks), "stage1_rejections": rejected,
            "capped_at": max_stage2 if len(survivors) > max_stage2 else None,
        },
        "tickers": [screener_row(p) for p in picks],
    })
    return 1 if failed and not scored else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE)
    ap.add_argument("--max-price", type=float, default=DEFAULT_MAX_PRICE)
    ap.add_argument("--max-off-high", type=float, default=MAX_OFF_HIGH * 100,
                    help="most a stock may sit below its 52-week high, in percent (default 25; 100 = no limit)")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP)
    ap.add_argument("--max-stage2", type=int, default=MAX_STAGE2)
    a = ap.parse_args(argv)
    if not (0 <= a.min_price < a.max_price):
        ap.error("need 0 <= --min-price < --max-price")
    if not (0 <= a.max_off_high <= 100):
        ap.error("--max-off-high must be 0-100 (percent)")
    if not (1 <= a.top <= 100 and 1 <= a.max_stage2 <= 300):
        ap.error("--top must be 1-100 and --max-stage2 1-300")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run(a.min_price, a.max_price, a.top, a.max_stage2, max_off_high=a.max_off_high / 100.0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
