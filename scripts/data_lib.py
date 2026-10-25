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
  - Yahoo's news endpoint returned an empty list from GitHub's runners (seen
    live for AAPL), so headlines also come from Google News RSS, an unofficial
    free feed. Either source can be empty or change shape; news is context only
    and an empty result never fails a run.
"""
from __future__ import annotations

import logging
import re
import time
from email.utils import parsedate_to_datetime
from urllib.parse import quote
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


def parse_symbols(args: list[str]) -> list[str]:
    """Split on commas/whitespace, validate, upper-case, de-duplicate (order kept)."""
    raw = [t for a in args for t in re.split(r"[,\s]+", a) if t]
    return list(dict.fromkeys(normalize_symbol(t) for t in raw))


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


def _http_url(u: Any) -> str:
    """Only http(s) links are kept: the text comes from a third party and ends up in an href."""
    return u if isinstance(u, str) and re.match(r"^https?://", u, re.I) else ""


def parse_news(raw: list[dict] | None) -> list[dict[str, str]]:
    """Normalise yfinance's news payload to [{title, summary, date, source, url}], newest first.

    yfinance has shipped two shapes: the current one nests everything under
    item["content"] (title, summary, pubDate, provider.displayName,
    canonicalUrl.url); the older one is flat (title, publisher, link,
    providerPublishTime as epoch seconds). Items without a title are dropped.
    """
    out = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        c = item.get("content") if isinstance(item.get("content"), dict) else item
        title = (c.get("title") or "").strip()
        if not title:
            continue
        when = c.get("pubDate") or c.get("displayTime") or item.get("providerPublishTime")
        try:
            date = (pd.Timestamp(when, unit="s") if isinstance(when, (int, float)) else pd.Timestamp(when)).strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001 - unparseable/missing date: keep the headline undated
            date = ""
        provider = c.get("provider") if isinstance(c.get("provider"), dict) else {}
        url = ""
        for link in (c.get("canonicalUrl"), c.get("clickThroughUrl")):
            if isinstance(link, dict) and _http_url(link.get("url")):
                url = link["url"]
                break
        url = url or _http_url(c.get("link")) or _http_url(item.get("link"))
        out.append({
            "title": title,
            "summary": (c.get("summary") or "").strip(),
            "date": date,
            "source": provider.get("displayName") or c.get("publisher") or item.get("publisher") or "",
            "url": url,
        })
    return sorted(out, key=lambda n: n["date"], reverse=True)


NEWS_TABS = ("news", "press releases")
GNEWS_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
GNEWS_UA = "Mozilla/5.0 (compatible; ticker-dashboard)"
GNEWS_WINDOW = "45d"
GNEWS_PER_QUERY = 15
GNEWS_TIMEOUT = 10
# Google answers 503 to GitHub's runners (seen live). One failed request switches the
# feed off for the rest of the process instead of retrying for every ticker.
_gnews_down = False
_CORP_SUFFIX = re.compile(r"[,.\s]+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|n\.?v|s\.?a|ag|llc|class [a-z]|common stock)\.?$", re.I)
INNOVATION_QUERY = 'launch OR unveils OR introduces OR "new product" OR partnership OR patent'


def short_company_name(name: str, symbol: str) -> str:
    """'Apple Inc.' -> 'Apple' (for a news query); falls back to the symbol."""
    n = (name or "").strip()
    while True:
        m = _CORP_SUFFIX.search(n)
        if not m:
            break
        n = n[:m.start()].strip()
    return n.strip(" ,.") or symbol


def parse_google_rss(xml_bytes: bytes) -> list[dict[str, str]]:
    """Google News RSS -> [{title, summary, date, source, url}], newest first.

    Parsed with lxml with entity resolution and network access off (the feed
    is third-party). Google appends " - Publisher" to titles; it is moved to
    `source`. Non-http(s) links are dropped.
    """
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    root = etree.fromstring(xml_bytes, parser)
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3].rstrip()
        if not title:
            continue
        try:
            date = parsedate_to_datetime(item.findtext("pubDate") or "").strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001 - missing/garbled date: keep undated
            date = ""
        out.append({"title": title, "summary": "", "date": date, "source": source, "url": _http_url((item.findtext("link") or "").strip())})
    return sorted(out, key=lambda n: n["date"], reverse=True)


def fetch_google_news(symbol: str, name: str) -> list[dict[str, str]]:
    """Recent headlines from Google News RSS: one general query and one launch/innovation query.

    Single attempt per query, no retries: if Google refuses (503/429 from cloud IPs),
    the feed is skipped for the rest of the run.
    """
    global _gnews_down
    import requests

    if _gnews_down:
        return []
    company = short_company_name(name, symbol)
    base = f'"{company}" {symbol}'
    queries = {"general": f"{base} when:{GNEWS_WINDOW}", "launches": f"{base} ({INNOVATION_QUERY}) when:{GNEWS_WINDOW}"}
    merged: dict[str, dict[str, str]] = {}
    for label, q in queries.items():
        try:
            r = requests.get(GNEWS_URL.format(q=quote(q)), headers={"User-Agent": GNEWS_UA}, timeout=GNEWS_TIMEOUT)
            r.raise_for_status()
            items = parse_google_rss(r.content)[:GNEWS_PER_QUERY]
        except Exception as exc:  # noqa: BLE001
            _gnews_down = True
            log.warning("%s: google news unavailable (%s); skipping it for the rest of this run", symbol, exc)
            break
        log.info("%s: google news[%s] parsed=%d newest=%s", symbol, label, len(items), items[0]["date"] if items else "-")
        for n in items:
            merged.setdefault(n["title"].casefold(), n)
    return list(merged.values())


def fetch_news(symbol: str, name: str = "", count: int = 20) -> list[dict[str, str]]:
    """Headlines for symbol from Yahoo ("news" + "press releases") and Google News, merged, newest first.

    Each Yahoo tab is fetched on a fresh Ticker because yfinance caches the
    first tab's result on the instance. Counts are logged so an empty result
    can be told apart from a format change. Never raises: news is context only.
    """
    import yfinance as yf

    merged: dict[str, dict[str, str]] = {}
    for tab in NEWS_TABS:
        raw = _safe(lambda: yf.Ticker(symbol).get_news(count=count, tab=tab), f"news[{tab}] {symbol}", []) or []
        parsed = parse_news(raw)
        log.info("%s: yahoo news[%s] raw=%d parsed=%d newest=%s", symbol, tab, len(raw), len(parsed), parsed[0]["date"] if parsed else "-")
        if raw and not parsed:
            first = raw[0] if isinstance(raw[0], dict) else {}
            content = first.get("content") if isinstance(first.get("content"), dict) else {}
            log.warning("%s: yahoo news[%s] returned items we could not parse; item keys=%s content keys=%s",
                        symbol, tab, sorted(first), sorted(content))
        for n in parsed:
            merged.setdefault(n["title"].casefold(), n)
    try:
        for n in fetch_google_news(symbol, name):
            merged.setdefault(n["title"].casefold(), n)
    except Exception as exc:  # noqa: BLE001 - e.g. lxml missing or malformed feed
        log.warning("%s: google news failed: %s", symbol, exc)
    return sorted(merged.values(), key=lambda n: n["date"], reverse=True)


def fetch_fundamentals(symbol: str, with_news: bool = True) -> dict[str, Any]:
    """Fundamentals for CAN SLIM. Missing fields are None / empty lists, never fabricated.

    quarterly_eps / annual_eps / quarterly_revenue: ascending [(YYYY-MM-DD, value)].
    with_news=False skips the headline fetch (the screener scores many tickers and
    news is context only).
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

    name = info.get("longName") or info.get("shortName") or symbol
    news = fetch_news(symbol, name) if with_news else None

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
        "news": news,
    }
