"""CLI entry point: fetch data for tickers and rebuild docs/data/*.json.

    python scripts/fetch_and_build.py NVDA AAPL      # explicit tickers (space or comma separated)
    python scripts/fetch_and_build.py                # falls back to watchlist.txt

One bad ticker never stops the others: failures are logged, the rest are
still written, and the exit code is 1 if any ticker failed so the workflow run
still shows red after committing what did succeed.
"""
from __future__ import annotations

import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import build_site  # noqa: E402
import data_lib  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_and_build")

WATCHLIST = os.path.join(os.path.dirname(__file__), "..", "watchlist.txt")


def read_watchlist(path: str = WATCHLIST) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [ln.split("#")[0].strip() for ln in fh if ln.split("#")[0].strip()]


def parse_tickers(args: list[str]) -> list[str]:
    """Split on commas/whitespace, validate, upper-case, de-duplicate (order kept)."""
    raw = [t for a in args for t in re.split(r"[,\s]+", a) if t]
    return list(dict.fromkeys(data_lib.normalize_symbol(t) for t in raw))


def main(argv: list[str]) -> int:
    tickers = parse_tickers(argv) or parse_tickers(read_watchlist())
    if not tickers:
        log.error("no tickers given and watchlist.txt is empty")
        return 2

    log.info("fetching benchmarks %s, %s", data_lib.BENCHMARK, data_lib.SECOND_INDEX)
    spy = data_lib.fetch_prices(data_lib.BENCHMARK)
    try:
        qqq = data_lib.fetch_prices(data_lib.SECOND_INDEX)
    except Exception as exc:  # noqa: BLE001 - M falls back to SPY alone
        log.warning("QQQ unavailable (%s); market direction uses SPY only", exc)
        qqq = None

    rows, failed = [], []
    for sym in tickers:
        try:
            log.info("%s: fetching", sym)
            prices = data_lib.fetch_prices(sym)
            fundamentals = data_lib.fetch_fundamentals(sym)
            payload = build_site.build_payload(sym, prices, spy, qqq, fundamentals)
            build_site.write_ticker(payload)
            rows.append(build_site.summary_row(payload))
            t = payload["canslim"]["tally"]
            log.info("%s: %d/%d pass", sym, t["pass"], t["scored"])
        except Exception:  # noqa: BLE001 - isolate per-ticker failures
            log.exception("%s: failed", sym)
            failed.append(sym)

    if rows:
        build_site.update_index(rows)
    if failed:
        log.error("failed tickers: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
