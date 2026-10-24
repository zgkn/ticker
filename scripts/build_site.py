"""Turns prices + fundamentals into the JSON the static page renders.

docs/data/<SYMBOL>.json   one file per ticker: candles, indicator series, CAN SLIM scorecard
docs/data/index.json      one summary row per ticker, for the Overview tab

All the maths lives in Python (indicators.py / canslim.py / market.py); the
page only draws what is in these files. docs/index.html is static boilerplate.

CAVEAT: only the last DISPLAY_BARS sessions are shipped for charting, but the
indicators are computed on the full ~3 years fetched so the 200-day average and
12-month relative strength are already warmed up on the first bar shown.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

import canslim
import indicators as ind
import market

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
DATA_DIR = os.path.join(DOCS_DIR, "data")
DISPLAY_BARS = 504  # ~2 years of trading days

PRICE_COLS = ("open", "high", "low", "close", "volume")
# Indicator columns drawn on the chart; the rest (high52, updown_vol50, ...) feed the scorecard only.
CHART_SERIES = ("sma10", "sma21", "sma50", "sma150", "sma200", "vol_sma50", "rsi14", "macd", "macd_signal", "macd_hist", "rs_line")


def _list(s: pd.Series, nd: int = 4) -> list[float | None]:
    return [None if pd.isna(x) else round(float(x), nd) for x in s]


def build_payload(
    symbol: str,
    prices: pd.DataFrame,
    spy: pd.DataFrame,
    qqq: pd.DataFrame | None,
    fundamentals: dict[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Pure function: DataFrames in, JSON-ready dict out."""
    px = prices.join(ind.compute_all(prices, spy["close"]))
    mkt = market.evaluate({"SPY": spy, **({"QQQ": qqq} if qqq is not None else {})})
    card = canslim.evaluate(px, spy["close"], fundamentals, mkt)

    view = px.tail(DISPLAY_BARS)
    rs = view["rs_line"]
    first_rs = rs.dropna().iloc[0] if rs.notna().any() else None
    if first_rs:
        view = view.assign(rs_line=rs / first_rs * 100.0)  # rebased to 100 at the first bar shown

    last = px.iloc[-1]
    close, high52, low52 = float(last["close"]), float(last["high52"]), float(last["low52"])
    atr14 = None if pd.isna(last["atr14"]) else float(last["atr14"])
    series = {c: _list(view[c], 2 if c != "volume" else 0) for c in PRICE_COLS}
    series.update({name: _list(view[name], 4) for name in CHART_SERIES})

    return {
        "symbol": symbol,
        "name": fundamentals.get("name") or symbol,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "as_of": px.index[-1].strftime("%Y-%m-%d"),
        "benchmark": "SPY",
        "dates": [d.strftime("%Y-%m-%d") for d in view.index],
        "series": series,
        "levels": {
            "close": close,
            "high52": high52,
            "low52": low52,
            "pct_off_high": close / high52 - 1.0,
            "sma50": None if pd.isna(last["sma50"]) else float(last["sma50"]),
            "sma200": None if pd.isna(last["sma200"]) else float(last["sma200"]),
            "atr14": atr14,
            "atr_pct": None if atr14 is None else atr14 / close,
            "rsi14": None if pd.isna(last["rsi14"]) else float(last["rsi14"]),
        },
        "canslim": card,
    }


def summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    c = payload["canslim"]
    return {
        "symbol": payload["symbol"],
        "name": payload["name"],
        "as_of": payload["as_of"],
        "close": payload["levels"]["close"],
        "pct_off_high": payload["levels"]["pct_off_high"],
        "rsi14": payload["levels"]["rsi14"],
        "tally": c["tally"],
        "letters": {k: v["status"] for k, v in c["letters"].items()},
    }


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


def write_ticker(payload: dict[str, Any], data_dir: str | None = None) -> None:
    write_json(os.path.join(data_dir or DATA_DIR, f"{payload['symbol']}.json"), payload)


def update_index(rows: list[dict[str, Any]], data_dir: str | None = None) -> dict[str, Any]:
    """Merge rows into data/index.json by symbol (tickers accumulate across runs)."""
    path = os.path.join(data_dir or DATA_DIR, "index.json")
    existing: dict[str, dict] = {}
    if os.path.exists(path):
        with open(path) as fh:
            existing = {r["symbol"]: r for r in json.load(fh).get("tickers", [])}
    existing.update({r["symbol"]: r for r in rows})
    index = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers": sorted(existing.values(), key=lambda r: r["symbol"]),
    }
    write_json(path, index)
    return index
