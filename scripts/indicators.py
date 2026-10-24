"""Technical indicators: pure functions on pandas Series/DataFrames.

Prices are a DataFrame with a DatetimeIndex and lowercase columns
open/high/low/close/volume. Every indicator returns a Series aligned to the
input index (NaN while warming up).

To add an indicator: write the function, then add one line to INDICATORS.
compute_all() runs the registry and build_site.py ships whatever it returns
to the page, so a new entry shows up in data/<SYMBOL>.json automatically (the
page still has to be told how to draw it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing: seeded with the simple mean of the first n values, then
    avg = (prev * (n - 1) + x) / n. (pandas ewm seeds from the first value, which
    differs from the textbook figures for the first few dozen bars.)"""
    out = np.full(len(s), np.nan)
    vals = s.to_numpy(dtype=float)
    valid = np.flatnonzero(~np.isnan(vals))
    if len(valid) < n:
        return pd.Series(out, index=s.index)
    start = valid[0] + n - 1
    out[start] = vals[valid[0]:start + 1].mean()
    for i in range(start + 1, len(vals)):
        out[i] = (out[i - 1] * (n - 1) + vals[i]) / n
    return pd.Series(out, index=s.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder(gain.where(delta.notna()), n)
    avg_loss = wilder(loss.where(delta.notna()), n)
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses at all in the window -> RSI is 100 (rs is inf/NaN there).
    out = out.where(avg_loss != 0, 100.0).where(avg_gain.notna())
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's average true range."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return wilder(tr, n)


def rolling_high(df: pd.DataFrame, n: int = 252) -> pd.Series:
    return df["high"].rolling(n, min_periods=1).max()


def rolling_low(df: pd.DataFrame, n: int = 252) -> pd.Series:
    return df["low"].rolling(n, min_periods=1).min()


def updown_volume_ratio(df: pd.DataFrame, n: int = 50) -> pd.Series:
    """Volume on up-close days / volume on down-close days over the last n sessions.

    > 1 means more volume is trading on advances than declines (accumulation).
    """
    change = df["close"].diff()
    up = df["volume"].where(change > 0, 0.0).rolling(n, min_periods=n).sum()
    down = df["volume"].where(change < 0, 0.0).rolling(n, min_periods=n).sum()
    return up / down.replace(0.0, np.nan)


# IBD-style relative-strength weighting: cumulative return over the last
# 3, 6, 9 and 12 months (in trading days), weighted 40/20/20/20.
RS_WEIGHTS = ((63, 0.4), (126, 0.2), (189, 0.2), (252, 0.2))


def weighted_performance(close: pd.Series) -> float | None:
    """Weighted 12-month performance of the latest bar, or None if history is too short."""
    if len(close) <= RS_WEIGHTS[-1][0]:
        return None
    last = float(close.iloc[-1])
    total = 0.0
    for days, weight in RS_WEIGHTS:
        base = float(close.iloc[-1 - days])
        if base <= 0:
            return None
        total += weight * (last / base - 1.0)
    return total


def rs_line(close: pd.Series, bench_close: pd.Series) -> pd.Series:
    """Price relative to the benchmark (ratio). A rising line = outperforming."""
    bench = bench_close.reindex(close.index).ffill()
    return close / bench


# name -> function(prices, bench_close) -> Series. Keep names stable: the page
# looks them up by key.
INDICATORS = {
    "sma10": lambda df, b: sma(df["close"], 10),
    "sma21": lambda df, b: sma(df["close"], 21),
    "sma50": lambda df, b: sma(df["close"], 50),
    "sma150": lambda df, b: sma(df["close"], 150),
    "sma200": lambda df, b: sma(df["close"], 200),
    "vol_sma50": lambda df, b: sma(df["volume"], 50),
    "rsi14": lambda df, b: rsi(df["close"], 14),
    "macd": lambda df, b: macd(df["close"])["macd"],
    "macd_signal": lambda df, b: macd(df["close"])["macd_signal"],
    "macd_hist": lambda df, b: macd(df["close"])["macd_hist"],
    "atr14": lambda df, b: atr(df, 14),
    "high52": lambda df, b: rolling_high(df, 252),
    "low52": lambda df, b: rolling_low(df, 252),
    "updown_vol50": lambda df, b: updown_volume_ratio(df, 50),
    "rs_line": lambda df, b: rs_line(df["close"], b),
}


def compute_all(df: pd.DataFrame, bench_close: pd.Series) -> pd.DataFrame:
    """Return df's index with one column per registered indicator."""
    return pd.DataFrame({name: fn(df, bench_close) for name, fn in INDICATORS.items()}, index=df.index)
