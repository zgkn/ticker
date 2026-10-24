"""Market-data access (yfinance) for the dashboard.

Everything network-facing lives here so the rest of the pipeline (indicators,
CAN SLIM, JSON build) stays pure and testable with synthetic data.

CAVEATS:
  - yfinance scrapes Yahoo Finance; it is unofficial and can break or
    rate-limit. Every call is retried, and fundamentals fields that come back
    empty are left as None / [] -- callers must treat missing data as "n/a",
    never as zero.
  - Prices are split/dividend adjusted (auto_adjust=True), so historical
    closes will not match the prints of the day.
  - Quarterly fundamentals from yfinance usually cover only the last ~4-5
    quarters, which is the minimum needed for a year-over-year comparison.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable

import pandas as pd

log = logging.getLogger("data_lib")

# Yahoo symbols: letters/digits plus . - ^ = (BRK-B, ^GSPC, ES=F). Symbols end
# up in file names and workflow inputs, so reject anything else up front.
SYMBOL_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,15}$")

BENCHMARK = "SPY"
SECOND_INDEX = "QQQ"
HISTORY_PERIOD = "3y"  # 200-day average + 12-month RS need > 1y of warm-up before the 2y shown


def normalize_symbol(raw: str) -> str:
    sym = raw.strip().upper()
    if not SYMBOL_RE.match(sym):
        raise ValueError(f"invalid ticker symbol: {raw!r}")
    return sym


def _retry(fn: Callable[[], Any], what: str, attempts: int = 4) -> Any:
    delay = 2.0
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - yfinance raises many unrelated types
            if i == attempts:
                raise
            log.warning("%s failed (%s); retry %d/%d in %.0fs", what, exc, i, attempts - 1, delay)
            time.sleep(delay)
            delay *= 2


def fetch_prices(symbol: str, period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """Daily OHLCV with a tz-naive DatetimeIndex and lowercase columns."""
    import yfinance as yf

    def go() -> pd.DataFrame:
        df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError(f"no price data returned for {symbol}")
        return df

    df = _retry(go, f"prices {symbol}")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _row(df: pd.DataFrame | None, labels: tuple[str, ...]) -> list[tuple[str, float]]:
    """First matching income-statement row as ascending [(iso_date, value)], NaNs dropped."""
    if df is None or df.empty:
        return []
    for label in labels:
        if label in df.index:
            s = pd.to_numeric(df.loc[label], errors="coerce").dropna()
            pairs = [(pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)) for d, v in s.items()]
            return sorted(pairs)
    return []


def _earnings_dates_eps(t: Any) -> list[tuple[str, float]]:
    """Reported EPS history (more quarters than the income statement, usually)."""
    ed = t.get_earnings_dates(limit=16)
    if ed is None or ed.empty or "Reported EPS" not in ed.columns:
        return []
    s = pd.to_numeric(ed["Reported EPS"], errors="coerce").dropna()
    return sorted((pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)) for d, v in s.items())


def _safe(fn: Callable[[], Any], what: str, default: Any) -> Any:
    try:
        return _retry(fn, what, attempts=3)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s unavailable: %s", what, exc)
        return default


def fetch_fundamentals(symbol: str) -> dict[str, Any]:
    """Fundamentals for CAN SLIM. Missing fields are None / empty lists, never fabricated.

    quarterly_eps / annual_eps / quarterly_revenue: ascending [(YYYY-MM-DD, value)].
    """
    import yfinance as yf

    t = yf.Ticker(symbol)
    info = _safe(lambda: t.info, f"info {symbol}", {}) or {}
    q_stmt = _safe(lambda: t.quarterly_income_stmt, f"quarterly income {symbol}", None)
    a_stmt = _safe(lambda: t.income_stmt, f"annual income {symbol}", None)

    eps_labels = ("Diluted EPS", "Basic EPS")
    rev_labels = ("Total Revenue", "Operating Revenue")
    q_eps = _row(q_stmt, eps_labels)
    eps_source = "income statement (GAAP diluted EPS)"
    if len(q_eps) < 5:
        alt = _safe(lambda: _earnings_dates_eps(t), f"earnings dates {symbol}", [])
        if len(alt) > len(q_eps):
            q_eps, eps_source = alt, "reported EPS (earnings dates)"

    return {
        "symbol": symbol,
        "name": info.get("longName") or info.get("shortName") or symbol,
        "quarterly_eps": q_eps,
        "quarterly_eps_source": eps_source,
        "quarterly_revenue": _row(q_stmt, rev_labels),
        "annual_eps": _row(a_stmt, eps_labels),
        "roe": info.get("returnOnEquity"),
        "float_shares": info.get("floatShares"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "inst_pct": info.get("heldPercentInstitutions"),
    }
