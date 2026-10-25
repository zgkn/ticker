"""Synthetic market data for tests and for previewing the page offline."""
import numpy as np
import pandas as pd


def make_prices(n=756, drift=0.0006, vol=0.015, seed=1, start_price=100.0, end="2026-10-23"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    ret = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = (rng.lognormal(15, 0.35, n)).round()
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def make_fundamentals(strong=True):
    q = [("2025-06-30", 1.00), ("2025-09-30", 1.05), ("2025-12-31", 1.20), ("2026-03-31", 1.30), ("2026-06-30", 1.50 if strong else 0.95)]
    return {
        "symbol": "TEST",
        "name": "Test Corp",
        "quarterly_eps": q,
        "quarterly_eps_source": "income statement (GAAP diluted EPS)",
        "quarterly_revenue": [("2025-06-30", 100.0), ("2025-09-30", 105.0), ("2025-12-31", 112.0), ("2026-03-31", 118.0), ("2026-06-30", 140.0 if strong else 99.0)],
        "annual_eps": [("2022-12-31", 1.0), ("2023-12-31", 1.4), ("2024-12-31", 2.0), ("2025-12-31", 2.9)] if strong else [("2022-12-31", 3.0), ("2023-12-31", 2.5), ("2024-12-31", 2.4), ("2025-12-31", 2.2)],
        "roe": 0.25 if strong else 0.05,
        "float_shares": 2.0e9,
        "shares_outstanding": 2.1e9,
        "inst_pct": 0.62 if strong else 0.05,
    }


def make_px(close_path, volume=None, high_pad=0.005, low_pad=0.005, bench=None):
    """Prices from a close path, joined with every indicator (what chart_notes/canslim consume)."""
    import indicators as ind

    n = len(close_path)
    idx = pd.bdate_range(end="2026-10-23", periods=n)
    close = pd.Series(np.asarray(close_path, dtype=float), index=idx)
    df = pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]), "high": close * (1 + high_pad), "low": close * (1 - low_pad), "close": close,
                       "volume": np.full(n, 1e6) if volume is None else np.asarray(volume, dtype=float)}, index=idx)
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df.join(ind.compute_all(df, close if bench is None else bench))
