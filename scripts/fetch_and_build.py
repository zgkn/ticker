"""CLI entry point: fetch data for tickers and rebuild docs/data/*.json.

    python scripts/fetch_and_build.py NVDA AAPL      # explicit tickers (space or comma separated)
    python scripts/fetch_and_build.py                # falls back to watchlist.txt + candidates.txt

One bad ticker never stops the others: failures are logged, the rest are
still written, and the exit code is 1 if any ticker failed so the workflow run
still shows red after committing what did succeed.
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import build_site  # noqa: E402
import data_lib  # noqa: E402
import lists  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_and_build")

parse_tickers = data_lib.parse_symbols


def main(argv: list[str]) -> int:
    membership = lists.read_lists()
    tickers = parse_tickers(argv) or list(membership)
    if not tickers:
        log.error("no tickers given and watchlist.txt / candidates.txt are empty")
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

    if rows or os.path.exists(os.path.join(build_site.DATA_DIR, "index.json")):
        build_site.update_index(rows, lists=membership)
    if failed:
        log.error("failed tickers: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
