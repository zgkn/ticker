"""William O'Neil's CAN SLIM, scored per letter from yfinance data.

Each letter returns {status, headline, metrics, notes}; status is PASS /
PARTIAL / FAIL / N/A. N/A means the data was not available (yfinance
fundamentals are patchy) and the letter is left out of the tally -- it is never
guessed. This is an information screen, not a buy signal.

CAVEATS (the page repeats the first two):
  - L (leader) is a PROXY. IBD ranks a stock against every other stock; here it
    is the stock's weighted 12-month return versus the SPY's, with no
    universe percentile.
  - I (institutions) is a PROXY. yfinance gives the percent of shares held by
    institutions but no reliable quarter-over-quarter change or fund quality.
  - EPS comes from yfinance's income statement (GAAP diluted) or, when that has
    too few quarters, reported EPS. O'Neil uses "as reported" earnings, so
    figures can differ from IBD's.
  - Thresholds are O'Neil's rules of thumb, kept in THRESHOLDS so they are easy to change.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

import indicators as ind

PASS, PARTIAL, FAIL, NA = "PASS", "PARTIAL", "FAIL", "N/A"

THRESHOLDS = {
    "eps_growth_pass": 0.25,       # C: quarterly EPS growth YoY
    "eps_growth_partial": 0.10,
    "annual_cagr_pass": 0.25,      # A: 3-year annual EPS growth (CAGR)
    "roe_pass": 0.17,
    "near_high_pass": 0.05,        # N: within 5% of the 52-week high
    "near_high_partial": 0.15,
    "updown_pass": 1.2,            # S: up/down volume ratio (50 days)
    "updown_partial": 1.0,
    "rs_outperf_pass": 0.15,       # L: weighted 12m return over the benchmark's, in return points
    "inst_low": 0.10,              # I: institutional ownership band
    "inst_mid_low": 0.30,
    "inst_high": 0.90,
}

LETTERS = {
    "C": "Current quarterly earnings",
    "A": "Annual earnings",
    "N": "New highs",
    "S": "Supply & demand",
    "L": "Leader or laggard",
    "I": "Institutional sponsorship",
    "M": "Market direction",
}


# --- formatting helpers -----------------------------------------------------

def _pct(x: float | None, signed: bool = True) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def _big(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.2f}{suffix}"
    return f"{x:.0f}"


def _grade(oks: list[bool | None]) -> str:
    """All available checks true -> PASS; some -> PARTIAL; none -> FAIL; nothing available -> N/A."""
    avail = [o for o in oks if o is not None]
    if not avail:
        return NA
    if all(avail):
        return PASS
    return PARTIAL if any(avail) else FAIL


def _result(status: str, headline: str, metrics: list[tuple[str, str]], notes: list[str]) -> dict[str, Any]:
    return {"status": status, "headline": headline, "metrics": [{"k": k, "v": v} for k, v in metrics], "notes": notes}


def _na(why: str) -> dict[str, Any]:
    return _result(NA, why, [], [])


# --- earnings helpers -------------------------------------------------------

def _yoy(series: list[tuple[str, float]], idx: int) -> tuple[float | None, float | None, str]:
    """Growth of series[idx] versus the entry ~1 year earlier.

    Returns (growth, prior_value, kind) where kind is 'ok', 'turnaround'
    (prior <= 0 < current), 'negative' (both <= 0) or 'missing'.
    """
    cur_date, cur = series[idx]
    cd = date.fromisoformat(cur_date)
    prior = None
    for d, v in series[:idx]:
        if abs((cd - date.fromisoformat(d)).days - 365) <= 45:
            prior = v
    if prior is None:
        return None, None, "missing"
    if prior > 0:
        return (cur - prior) / prior, prior, "ok"
    return None, prior, "turnaround" if cur > 0 else "negative"


# --- the letters ------------------------------------------------------------

def letter_c(f: dict) -> dict[str, Any]:
    eps = f.get("quarterly_eps") or []
    if len(eps) < 5:
        return _na("Need 5 quarters of EPS for a year-over-year comparison")
    growth, prior, kind = _yoy(eps, len(eps) - 1)
    if kind == "missing":
        return _na("No EPS found for the same quarter a year earlier")
    th = THRESHOLDS
    metrics = [("Latest quarter", eps[-1][0]), ("EPS (latest / year ago)", f"{eps[-1][1]:.2f} / {prior:.2f}")]
    notes = [f"EPS source: {f.get('quarterly_eps_source', 'yfinance')}."]

    if kind == "turnaround":
        status, headline = PARTIAL, "Turnaround: EPS positive after a loss/zero a year ago"
        metrics.append(("EPS growth YoY", "n/a (prior <= 0)"))
    elif kind == "negative":
        status, headline = FAIL, "Losses in both quarters"
        metrics.append(("EPS growth YoY", "n/a (loss)"))
    else:
        metrics.append(("EPS growth YoY", _pct(growth)))
        status = PASS if growth >= th["eps_growth_pass"] else PARTIAL if growth >= th["eps_growth_partial"] else FAIL
        headline = f"EPS {_pct(growth)} YoY (O'Neil: >= {th['eps_growth_pass'] * 100:.0f}%)"
        if len(eps) >= 6:
            prev_growth, _, prev_kind = _yoy(eps, len(eps) - 2)
            if prev_kind == "ok":
                metrics.append(("Prior quarter growth", _pct(prev_growth)))
                notes.append("Growth is accelerating." if growth > prev_growth else "Growth is decelerating.")

    rev = f.get("quarterly_revenue") or []
    if len(rev) >= 5:
        rg, _, rk = _yoy(rev, len(rev) - 1)
        if rk == "ok":
            metrics.append(("Revenue growth YoY", _pct(rg)))
            if rg < th["eps_growth_pass"] <= (growth or 0):
                notes.append("Revenue growth is below the EPS growth: earnings may be margin- or buyback-driven.")
    return _result(status, headline, metrics, notes)


def letter_a(f: dict) -> dict[str, Any]:
    th = THRESHOLDS
    eps = (f.get("annual_eps") or [])[-4:]
    cagr = None
    metrics: list[tuple[str, str]] = []
    notes: list[str] = []

    if len(eps) >= 2:
        yoys = [(b[1] - a[1]) / a[1] if a[1] > 0 else None for a, b in zip(eps, eps[1:])]
        metrics.append(("Annual EPS", " → ".join(f"{v:.2f}" for _, v in eps)))
        metrics.append(("YoY growth", " / ".join(_pct(g) if g is not None else "n/a" for g in yoys)))
        if eps[0][1] > 0 and eps[-1][1] > 0:
            cagr = (eps[-1][1] / eps[0][1]) ** (1.0 / (len(eps) - 1)) - 1.0
            metrics.append((f"EPS CAGR ({len(eps) - 1}y)", _pct(cagr)))
        if len(eps) < 4:
            notes.append(f"Only {len(eps) - 1} year(s) of growth available; O'Neil looks at 3.")
        if any(g is not None and g < 0 for g in yoys):
            notes.append("EPS fell in at least one year: O'Neil wants steady annual growth.")
    else:
        notes.append("Annual EPS history unavailable.")

    roe = f.get("roe")
    if roe is not None:
        metrics.append(("Return on equity", _pct(roe, signed=False)))

    eps_ok = None if cagr is None else cagr >= th["annual_cagr_pass"]
    roe_ok = None if roe is None else roe >= th["roe_pass"]
    status = _grade([eps_ok, roe_ok])
    if status == NA:
        return _na("Annual EPS and ROE unavailable")
    headline = {PASS: "Strong, consistent annual growth", PARTIAL: "Mixed annual growth / returns", FAIL: "Annual growth below O'Neil's bar"}[status]
    return _result(status, headline, metrics, notes)


def letter_n(px: pd.DataFrame) -> dict[str, Any]:
    th = THRESHOLDS
    last = float(px["close"].iloc[-1])
    high52 = float(px["high52"].iloc[-1])
    off = last / high52 - 1.0
    status = PASS if -off <= th["near_high_pass"] else PARTIAL if -off <= th["near_high_partial"] else FAIL
    s50, s200 = px["sma50"].iloc[-1], px["sma200"].iloc[-1]
    metrics = [("Close", f"{last:.2f}"), ("52-week high", f"{high52:.2f}"), ("Off high", _pct(off))]
    notes = []
    if pd.notna(s50):
        metrics.append(("vs 50-day avg", _pct(last / s50 - 1.0)))
    if pd.notna(s200):
        metrics.append(("vs 200-day avg", _pct(last / s200 - 1.0)))
        if last < s200:
            notes.append("Trading below the 200-day average: not in a Stage 2 uptrend.")
    if pd.notna(s50) and last / s50 - 1.0 > 0.15:
        notes.append("More than 15% above the 50-day average: extended; breakouts this far along carry more risk.")
    headline = f"{_pct(off)} from the 52-week high"
    if status == PASS:
        headline += " (at or near highs)"
    return _result(status, headline, metrics, notes)


def letter_s(px: pd.DataFrame, f: dict) -> dict[str, Any]:
    th = THRESHOLDS
    ud = px["updown_vol50"].iloc[-1]
    vol, vol50 = float(px["volume"].iloc[-1]), px["vol_sma50"].iloc[-1]
    flt, out = f.get("float_shares"), f.get("shares_outstanding")
    metrics: list[tuple[str, str]] = []
    notes: list[str] = []

    if pd.notna(vol50) and vol50 > 0:
        metrics.append(("Volume vs 50d average", f"{vol / vol50:.2f}x"))
    metrics.append(("Float", _big(flt)))
    if out:
        metrics.append(("Shares outstanding", _big(out)))
    if flt:
        notes.append("Small float (< 50M): supply is tight, moves can be sharp." if flt < 5e7 else "Large float: supply is ample, expect slower, steadier moves.")

    if pd.isna(ud):
        return _result(NA, "Volume history too short or empty", metrics, notes)
    metrics.insert(0, ("Up/down volume (50d)", f"{ud:.2f}"))
    status = PASS if ud >= th["updown_pass"] else PARTIAL if ud >= th["updown_partial"] else FAIL
    headline = {PASS: "Volume favours buyers (accumulation)", PARTIAL: "Volume roughly balanced", FAIL: "Volume favours sellers (distribution)"}[status]
    return _result(status, headline, metrics, notes)


def letter_l(px: pd.DataFrame, bench_close: pd.Series) -> dict[str, Any]:
    th = THRESHOLDS
    mine, theirs = ind.weighted_performance(px["close"]), ind.weighted_performance(bench_close)
    if mine is None or theirs is None:
        return _na("Need more than a year of price history")
    outperf = mine - theirs
    rs = px["rs_line"].dropna()
    metrics = [("Weighted 12m return", _pct(mine)), ("SPY weighted 12m return", _pct(theirs)), ("Outperformance", _pct(outperf))]
    notes = ["Proxy for IBD's RS Rating: relative to SPY only, not ranked against all stocks."]
    if len(rs) >= 252:
        at_high = rs.iloc[-1] >= rs.tail(252).max() * 0.98
        metrics.append(("RS line", "at/near 52-week high" if at_high else "below its 52-week high"))
        if at_high:
            notes.append("RS line at a high: leading the market.")
    status = PASS if outperf >= th["rs_outperf_pass"] else PARTIAL if outperf > 0 else FAIL
    headline = {PASS: "Market leader (strong outperformance)", PARTIAL: "Modest outperformance", FAIL: "Laggard versus the market"}[status]
    return _result(status, headline, metrics, notes)


def letter_i(f: dict) -> dict[str, Any]:
    th = THRESHOLDS
    pct = f.get("inst_pct")
    if pct is None:
        return _na("Institutional ownership unavailable")
    metrics = [("Held by institutions", _pct(pct, signed=False))]
    notes = ["Proxy: no quarter-over-quarter change or fund-quality data from yfinance."]
    if th["inst_mid_low"] <= pct <= th["inst_high"]:
        status, headline = PASS, "Meaningful institutional sponsorship"
    elif pct < th["inst_low"]:
        status, headline = FAIL, "Little institutional ownership"
    else:
        status = PARTIAL
        headline = "Light institutional ownership" if pct < th["inst_mid_low"] else "Very heavily owned (crowded)"
    return _result(status, headline, metrics, notes)


# --- assembly ---------------------------------------------------------------

def evaluate(px: pd.DataFrame, bench_close: pd.Series, fundamentals: dict, market: dict) -> dict[str, Any]:
    """px: prices joined with indicator columns (see indicators.compute_all)."""
    letters = {
        "C": letter_c(fundamentals),
        "A": letter_a(fundamentals),
        "N": letter_n(px),
        "S": letter_s(px, fundamentals),
        "L": letter_l(px, bench_close),
        "I": letter_i(fundamentals),
        "M": market,
    }
    for key, res in letters.items():
        res["letter"], res["label"] = key, LETTERS[key]

    count = {s: sum(1 for r in letters.values() if r["status"] == s) for s in (PASS, PARTIAL, FAIL, NA)}
    scored = count[PASS] + count[PARTIAL] + count[FAIL]
    weak = [k for k, r in letters.items() if r["status"] in (FAIL, PARTIAL)]
    summary = f"{count[PASS]} of {scored} scored letters pass ({count[PARTIAL]} partial, {count[FAIL]} fail"
    summary += f"; {count[NA]} not available)." if count[NA] else ")."
    if weak:
        summary += " Needs attention: " + ", ".join(f"{k} ({letters[k]['status'].lower()})" for k in weak) + "."
    return {
        "letters": letters,
        "tally": {"pass": count[PASS], "partial": count[PARTIAL], "fail": count[FAIL], "na": count[NA], "scored": scored},
        "summary": summary,
    }
