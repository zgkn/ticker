"""Rule-based "chart watch": what stands out on the latest bars, and the levels to watch.

Each observation is a pattern a swing trader would check by eye, encoded as a
rule so it is applied the same way to every ticker on every run:

  trend structure   - the seven-point "trend template" (Stage 2 uptrend), extension from the
                      50-day average, moving-average crosses
  setup             - new 52-week highs (and whether volume confirms), proximity to the high,
                      volume drying up in a tight range
  volume            - heavy-volume days, accumulation vs distribution
  momentum          - RSI extremes and divergences, recent MACD crosses
  leadership        - relative-strength line at highs or fading
  risk              - volatility, gaps

Output: {"observations": [{tag, title, detail}], "levels": [{side, label, price, pct, note}]}
where tag is bullish / bearish / caution / info.

CAVEATS:
  - Descriptive, not predictive and not advice: a rule firing says what the chart
    looks like, not what the price will do. The page repeats this.
  - Every threshold below is a common rule of thumb (O'Neil, Minervini), not a tuned
    or back-tested value. They live in the T dict so they are easy to change.
  - Fixed look-back windows assume daily bars. Missing data (NaN) skips a rule rather
    than guessing.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

T = {
    "extended_pct": 0.15,        # close this far above the 50-day = extended
    "near_high_pct": 0.03,       # within 3% of the 52-week high = breakout watch
    "breakout_volume": 1.4,      # volume vs 50-day average that confirms a breakout
    "heavy_volume": 1.5,         # "heavy" day
    "heavy_move": 0.02,          # +/-2% on a heavy-volume day
    "dry_volume": 0.7,           # last 5 days' volume vs average
    "tight_range": 0.04,         # last 5 days' high-low range / close
    "rsi_over": 70,
    "rsi_under": 30,
    "div_rsi_gap": 3.0,          # RSI points lower/higher for a divergence
    "atr_high": 0.04,            # ATR as % of price
    "gap_pct": 0.03,
    "dist_days_warn": 5,         # distribution days in 25 sessions
    "updown_acc": 1.5,
    "updown_dist": 0.8,
    "stop_low": 0.07,            # O'Neil's 7-8% loss-cutting rule of thumb
    "stop_high": 0.08,
}


def _v(s: pd.Series, i: int = -1) -> float | None:
    try:
        x = s.iloc[i]
    except IndexError:
        return None
    return None if pd.isna(x) else float(x)


def _money(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.2f}"


def _vol(x: float) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.1f}{suffix}"
    return f"{x:.0f}"


def _obs(tag: str, title: str, detail: str) -> dict[str, str]:
    return {"tag": tag, "title": title, "detail": detail}


def _trend(px: pd.DataFrame, out: list) -> None:
    c = _v(px["close"])
    s50, s150, s200 = _v(px["sma50"]), _v(px["sma150"]), _v(px["sma200"])
    s200_ago = _v(px["sma200"], -22)
    lo, hi = _v(px["low52"]), _v(px["high52"])
    if None in (c, s50, s150, s200, s200_ago, lo, hi):
        return
    checks = [
        ("price above the 150- and 200-day averages", c > s150 and c > s200),
        ("150-day above 200-day", s150 > s200),
        ("200-day average rising vs a month ago", s200 > s200_ago),
        ("50-day above the 150- and 200-day", s50 > s150 and s50 > s200),
        ("price above the 50-day", c > s50),
        ("price at least 30% above the 52-week low", c >= 1.3 * lo),
        ("price within 25% of the 52-week high", c >= 0.75 * hi),
    ]
    met = sum(ok for _, ok in checks)
    missing = [name for name, ok in checks if not ok]
    why = "Leaders in a Stage 2 uptrend typically meet all seven; stocks failing several are rarely where the strongest moves start."
    if met == 7:
        out.append(_obs("bullish", "Strong uptrend structure (7/7 trend-template checks)", why))
    elif met >= 5:
        out.append(_obs("info", f"Uptrend mostly intact ({met}/7 trend-template checks)", f"Missing: {'; '.join(missing)}. {why}"))
    else:
        out.append(_obs("bearish", f"Weak trend structure ({met}/7 trend-template checks)", f"Missing: {'; '.join(missing)}. {why}"))


def _extension_and_crosses(px: pd.DataFrame, out: list) -> None:
    c, s50 = _v(px["close"]), _v(px["sma50"])
    if c is None or s50 is None:
        return
    ext = c / s50 - 1.0
    if ext > T["extended_pct"]:
        out.append(_obs("caution", f"Extended: {ext * 100:.0f}% above the 50-day average",
                        "Stocks stretched this far above their average often pull back or consolidate; buying here means a wider gap to the nearest support."))
    elif c < s50:
        out.append(_obs("caution", "Trading below the 50-day average",
                        "Leaders usually hold the 50-day. A close below it, especially on heavy volume, is a common warning sign."))
    s50s, s200s = px["sma50"], px["sma200"]
    for i in range(-1, -11, -1):
        a, b, a0, b0 = _v(s50s, i), _v(s200s, i), _v(s50s, i - 1), _v(s200s, i - 1)
        if None in (a, b, a0, b0):
            break
        if a0 <= b0 and a > b:
            out.append(_obs("bullish", "Golden cross: 50-day moved above the 200-day (last 10 sessions)", "A slow, lagging signal: it confirms a trend change that has already started."))
            break
        if a0 >= b0 and a < b:
            out.append(_obs("bearish", "Death cross: 50-day moved below the 200-day (last 10 sessions)", "A slow, lagging signal: it confirms weakness that has already happened."))
            break


def _highs_and_volume(px: pd.DataFrame, out: list) -> None:
    c, hi = _v(px["close"]), _v(px["high52"])
    vol, vol50 = _v(px["volume"]), _v(px["vol_sma50"])
    if c is None or hi is None:
        return
    prior = px["close"].iloc[-253:-1]
    vr = vol / vol50 if vol is not None and vol50 else None
    if len(prior) >= 200 and c >= prior.max():
        if vr is not None and vr >= T["breakout_volume"]:
            out.append(_obs("bullish", f"New 52-week closing high on strong volume ({vr:.1f}x average)",
                            "Breakouts backed by heavy volume suggest institutional buying; this is the confirmation O'Neil looks for."))
        elif vr is not None:
            out.append(_obs("caution", f"New 52-week closing high on light volume ({vr:.1f}x average)",
                            f"Unconfirmed: breakouts usually need volume of at least {T['breakout_volume']}x the 50-day average. Watch for follow-through or failure."))
    else:
        off = c / hi - 1.0
        if off >= -T["near_high_pct"]:
            out.append(_obs("info", f"Within {-off * 100:.1f}% of the 52-week high: breakout watch",
                            f"A close above {_money(hi)} on volume of {T['breakout_volume']}x average or more would be a confirmed breakout; a rejection here is also common."))
        elif off <= -0.20:
            out.append(_obs("info", f"{-off * 100:.0f}% below the 52-week high",
                            "A long way from the highs: overhead supply from holders who bought higher can cap rallies."))

    # volume dry-up in a tight range
    if vol50:
        last5 = px.tail(5)
        rng = (last5["high"].max() - last5["low"].min()) / c
        if last5["volume"].mean() / vol50 < T["dry_volume"] and rng < T["tight_range"]:
            out.append(_obs("info", "Volume drying up in a tight range (last 5 sessions)",
                            "Quiet, narrow trading is typical of a base tightening before the next move; the direction is not signalled."))

    # heavy-volume days in the last 10 sessions
    if vol50:
        chg = px["close"].pct_change()
        recent = px.tail(10)
        for date in reversed(list(recent.index)):
            r, v = chg.loc[date], px.loc[date, "volume"]
            if pd.isna(r) or v < T["heavy_volume"] * vol50:
                continue
            if r <= -T["heavy_move"]:
                out.append(_obs("caution", f"Heavy-volume selloff on {date:%Y-%m-%d} ({r * 100:.1f}% on {v / vol50:.1f}x volume)",
                                "Large down days on high volume are how institutions leave; watch whether the stock recovers the day's range."))
                break
            if r >= T["heavy_move"]:
                out.append(_obs("bullish", f"Heavy-volume advance on {date:%Y-%m-%d} (+{r * 100:.1f}% on {v / vol50:.1f}x volume)",
                                "Sharp gains on rising volume are a sign of institutional buying."))
                break

    ud = _v(px["updown_vol50"])
    if ud is not None:
        if ud >= T["updown_acc"]:
            out.append(_obs("bullish", f"Accumulation: up-day volume is {ud:.1f}x down-day volume (50 sessions)", "More shares change hands on advances than declines."))
        elif ud <= T["updown_dist"]:
            out.append(_obs("bearish", f"Distribution: up-day volume is only {ud:.1f}x down-day volume (50 sessions)", "More shares change hands on declines than advances."))

    change = px["close"].pct_change()
    heavier = px["volume"] > px["volume"].shift(1)
    dist = int(((change <= -0.002) & heavier).tail(25).sum())
    if dist >= T["dist_days_warn"]:
        out.append(_obs("caution", f"{dist} distribution days in the last 25 sessions",
                        "Down days on higher volume than the day before: a cluster of them shows persistent selling."))


def _momentum(px: pd.DataFrame, out: list) -> None:
    rsi = _v(px["rsi14"])
    if rsi is not None:
        if rsi >= T["rsi_over"]:
            out.append(_obs("caution", f"RSI {rsi:.0f}: overbought zone",
                            "In strong uptrends RSI can stay above 70 for weeks, so it is not a sell signal by itself; it does mean new buys are chasing strength."))
        elif rsi <= T["rsi_under"]:
            out.append(_obs("caution", f"RSI {rsi:.0f}: oversold zone", "Oversold can persist in a downtrend; look for the price to stop making lower lows first."))

    # divergence: price makes a new extreme but RSI does not
    close, r = px["close"], px["rsi14"]
    if len(px) > 80 and r.tail(80).notna().all():
        recent, earlier = slice(-20, None), slice(-60, -20)
        i1, i0 = close.iloc[recent].idxmax(), close.iloc[earlier].idxmax()
        if close[i1] > close[i0] and r[i1] < r[i0] - T["div_rsi_gap"]:
            out.append(_obs("bearish", "Bearish RSI divergence",
                            f"Price made a higher high ({i0:%b %d} to {i1:%b %d}) while RSI made a lower high: momentum is not confirming the move."))
        j1, j0 = close.iloc[recent].idxmin(), close.iloc[earlier].idxmin()
        if close[j1] < close[j0] and r[j1] > r[j0] + T["div_rsi_gap"]:
            out.append(_obs("bullish", "Bullish RSI divergence",
                            f"Price made a lower low ({j0:%b %d} to {j1:%b %d}) while RSI made a higher low: selling momentum is easing."))

    m, sg = px["macd"], px["macd_signal"]
    diff = (m - sg).dropna()
    if len(diff) >= 7:
        tail = diff.tail(6)
        if tail.iloc[0] <= 0 < tail.iloc[-1] or tail.iloc[0] < 0 <= tail.iloc[-1]:
            zero = "above" if _v(m) > 0 else "below"
            out.append(_obs("bullish", f"MACD crossed above its signal line (last 5 sessions, {zero} zero)",
                            "Crosses above the zero line carry more weight than those below it." if zero == "below" else "A cross while MACD is above zero fits an established uptrend."))
        elif tail.iloc[0] >= 0 > tail.iloc[-1] or tail.iloc[0] > 0 >= tail.iloc[-1]:
            zero = "above" if _v(m) > 0 else "below"
            out.append(_obs("bearish" if zero == "below" else "caution", f"MACD crossed below its signal line (last 5 sessions, {zero} zero)",
                            "Momentum is turning down; confirm with price breaking support."))


def _leadership_and_risk(px: pd.DataFrame, out: list) -> None:
    rs = px["rs_line"].dropna()
    c, hi = _v(px["close"]), _v(px["high52"])
    if len(rs) >= 252:
        now, peak = rs.iloc[-1], rs.tail(252).max()
        avg50 = rs.tail(50).mean()
        if now >= 0.98 * peak:
            out.append(_obs("bullish", "Relative-strength line at/near a 52-week high",
                            "The stock is outperforming the market (SPY) more than at any point in the past year; O'Neil looks for RS highs ahead of or alongside price highs."))
        elif now < 0.97 * avg50 and c is not None and hi is not None and c / hi - 1.0 >= -0.10:
            out.append(_obs("caution", "Relative strength is fading while price is near its high",
                            "The stock is lagging the market over the last couple of months even though the chart looks strong; leadership may be rotating."))

    atr = _v(px["atr14"])
    if atr is not None and c:
        pct = atr / c
        if pct >= T["atr_high"]:
            out.append(_obs("caution", f"High volatility: ATR is {pct * 100:.1f}% of price",
                            "Daily swings are wide, so a normal stop distance risks more per share; smaller position sizes are common practice."))

    prev_close = px["close"].shift(1)
    gap = (px["open"] / prev_close - 1.0).tail(5)
    for date, g in reversed(list(gap.items())):
        if pd.notna(g) and abs(g) >= T["gap_pct"]:
            out.append(_obs("info", f"Gap {'up' if g > 0 else 'down'} of {abs(g) * 100:.1f}% on {date:%Y-%m-%d}",
                            "Watch whether the gap holds: gaps that fill within days often signal a failed move."))
            break


def _levels(px: pd.DataFrame) -> list[dict[str, Any]]:
    c = _v(px["close"])
    if c is None:
        return []
    vol50 = _v(px["vol_sma50"])
    levels: list[dict[str, Any]] = []

    def add(side: str, label: str, price: float | None, note: str) -> None:
        if price is not None:
            levels.append({"side": side, "label": label, "price": round(price, 2), "pct": price / c - 1.0, "note": note})

    hi = _v(px["high52"])
    if hi is not None and hi > c:
        trig = f" on volume above {_vol(T['breakout_volume'] * vol50)} shares ({T['breakout_volume']}x the 50-day average)" if vol50 else ""
        add("up", "52-week high", hi, f"A close above it{trig} is the classic breakout confirmation.")
    elif hi is not None:
        add("ref", "52-week high (now)", hi, "At the high: a close back below it on heavy volume would flag a failed breakout.")
    swing = float(px["high"].tail(20).max())
    if hi is None or swing < 0.97 * hi:
        if swing > c:
            add("up", "20-day high", swing, "Short-term resistance from the recent swing high.")

    for name, key, note in (("21-day average", "sma21", "First support in strong uptrends; leaders often bounce here."),
                            ("50-day average", "sma50", "Key support for leaders; a close below on heavy volume is a warning."),
                            ("200-day average", "sma200", "Long-term trend line; losing it often marks the end of a Stage 2 uptrend.")):
        v = _v(px[key])
        if v is None:
            continue
        if v < c:
            add("down", name, v, note)
        else:
            add("up", name + " (resistance)", v, "Price is below it: the average now acts as resistance until reclaimed.")

    atr = _v(px["atr14"])
    add("ref", f"{int(T['stop_low'] * 100)}-{int(T['stop_high'] * 100)}% below today's close", c * (1 - T["stop_low"]),
        f"O'Neil's rule of thumb is to cut losses 7-8% below the purchase price. Shown from today's close ({_money(c * (1 - T['stop_high']))} to {_money(c * (1 - T['stop_low']))}) for scale only.")
    if atr is not None:
        add("ref", "2 x ATR below today's close", c - 2 * atr, f"ATR(14) is {_money(atr)}; two ATRs is a common volatility-based distance for judging noise vs a real move.")
    levels.sort(key=lambda lv: -lv["price"])
    return levels


def analyze(px: pd.DataFrame) -> dict[str, Any]:
    """px: prices joined with indicators (indicators.compute_all), full history."""
    out: list[dict[str, str]] = []
    for step in (_trend, _extension_and_crosses, _highs_and_volume, _momentum, _leadership_and_risk):
        step(px, out)
    return {"observations": out, "levels": _levels(px)}
