"""CAN SLIM "M": market direction, judged on the major indexes (SPY, QQQ).

An index is scored on three things O'Neil watches:
  - trend: close above its 50- and 200-day averages
  - distribution days in the last 25 sessions: a close down >= 0.2% on higher
    volume than the prior session. Five or more is a sign of institutional selling.
  - follow-through day: after a low, a >= 1.25% gain on higher volume on day
    4-7 of a rally attempt, which O'Neil treats as confirmation of a new uptrend.

CAVEATS:
  - Distribution days are not aged out by a subsequent 5% rally, as O'Neil
    does. This can overcount after a strong bounce.
  - SPY / QQQ stand in for the S&P 500 / Nasdaq Composite, and their volume is
    the ETF's own, not the index's.
  - The follow-through scan is a simplified rule, reported for context only
    and not used in the score.
"""
from __future__ import annotations

import pandas as pd

import indicators as ind

DIST_WINDOW = 25
DIST_DROP = -0.002
DIST_WARN = 5
FTD_LOOKBACK = 60
FTD_GAIN = 0.0125

PASS, PARTIAL, FAIL, NA = "PASS", "PARTIAL", "FAIL", "N/A"
_RANK = {FAIL: 0, PARTIAL: 1, PASS: 2}


def distribution_days(df: pd.DataFrame, window: int = DIST_WINDOW) -> int:
    change = df["close"].pct_change()
    heavier = df["volume"] > df["volume"].shift(1)
    return int(((change <= DIST_DROP) & heavier).tail(window).sum())


def follow_through_day(df: pd.DataFrame, lookback: int = FTD_LOOKBACK) -> str | None:
    """Date of the most recent follow-through day within lookback sessions, or None."""
    d = df.tail(lookback).copy()
    if len(d) < 10:
        return None
    d["chg"] = d["close"].pct_change()
    d["heavier"] = d["volume"] > d["volume"].shift(1)
    found = None
    # Try every close as a rally-attempt low; day 1 is the first up close after it.
    closes = d["close"].to_numpy()
    for low_i in range(1, len(d) - 4):
        # a low: lowest close of the surrounding window so far
        if closes[low_i] > closes[: low_i + 1].min():
            continue
        day1 = next((i for i in range(low_i + 1, len(d)) if closes[i] > closes[i - 1]), None)
        if day1 is None:
            continue
        for k in range(day1 + 3, min(day1 + 7, len(d))):  # day 4..7 of the attempt
            if closes[day1:k + 1].min() < closes[low_i]:
                break  # low undercut: attempt failed
            if d["chg"].iloc[k] >= FTD_GAIN and d["heavier"].iloc[k]:
                found = d.index[k].strftime("%Y-%m-%d")
                break
    return found


def _score_index(name: str, df: pd.DataFrame) -> dict:
    close = df["close"]
    s50, s200 = ind.sma(close, 50), ind.sma(close, 200)
    last = float(close.iloc[-1])
    above50 = bool(pd.notna(s50.iloc[-1]) and last > s50.iloc[-1])
    above200 = bool(pd.notna(s200.iloc[-1]) and last > s200.iloc[-1])
    dist = distribution_days(df)
    ftd = follow_through_day(df)

    if above50 and above200 and dist < DIST_WARN:
        status = PASS
    elif not above50 and (not above200 or dist >= DIST_WARN):
        status = FAIL
    else:
        status = PARTIAL

    return {
        "name": name,
        "status": status,
        "close": last,
        "above50": above50,
        "above200": above200,
        "dist_days": dist,
        "ftd": ftd,
    }


def evaluate(indexes: dict[str, pd.DataFrame]) -> dict:
    """Score each index; the overall M status is the most conservative of them."""
    scored = [_score_index(n, df) for n, df in indexes.items() if df is not None and len(df) >= 60]
    if not scored:
        return {"status": NA, "headline": "Index data unavailable", "metrics": [], "notes": []}

    status = min((s["status"] for s in scored), key=_RANK.__getitem__)
    metrics, notes = [], []
    for s in scored:
        n = s["name"]
        metrics.append({"k": f"{n} vs 50d / 200d", "v": f"{'above' if s['above50'] else 'below'} / {'above' if s['above200'] else 'below'}"})
        metrics.append({"k": f"{n} distribution days (25d)", "v": str(s["dist_days"])})
        metrics.append({"k": f"{n} last follow-through day", "v": s["ftd"] or "none in last 60 sessions"})
        if s["dist_days"] >= DIST_WARN:
            notes.append(f"{n} has {s['dist_days']} distribution days in 25 sessions: heavy institutional selling.")
    headline = {
        PASS: "Major indexes in confirmed uptrends",
        PARTIAL: "Market direction mixed",
        FAIL: "Market under pressure: new buys carry higher risk",
    }[status]
    return {"status": status, "headline": headline, "metrics": metrics, "notes": notes}
