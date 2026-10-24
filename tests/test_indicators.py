import numpy as np
import pandas as pd
import pytest

import indicators as ind
from synth import make_prices


def test_sma_known_values():
    s = pd.Series([1.0, 2, 3, 4, 5])
    out = ind.sma(s, 3)
    assert out.isna().tolist() == [True, True, False, False, False]
    assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_rsi_matches_wilder_recursion():
    close = make_prices(n=120, seed=7)["close"]
    n = 14
    d = close.diff().to_numpy()
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = gain[1:n + 1].mean(), loss[1:n + 1].mean()
    expected = {n: 100 - 100 / (1 + ag / al)}
    for i in range(n + 1, len(close)):
        ag, al = (ag * (n - 1) + gain[i]) / n, (al * (n - 1) + loss[i]) / n
        expected[i] = 100 - 100 / (1 + ag / al)
    got = ind.rsi(close, n)
    for i, v in expected.items():
        assert got.iloc[i] == pytest.approx(v, abs=1e-6)


def test_rsi_all_gains_is_100_and_all_losses_is_0():
    up = pd.Series(np.arange(1.0, 60.0))
    assert ind.rsi(up).iloc[-1] == 100.0
    assert ind.rsi(up[::-1].reset_index(drop=True)).iloc[-1] == pytest.approx(0.0)


def test_macd_hist_is_line_minus_signal():
    m = ind.macd(make_prices(n=200)["close"])
    valid = m.dropna()
    assert (valid["macd_hist"] - (valid["macd"] - valid["macd_signal"])).abs().max() < 1e-12


def test_atr_constant_range():
    n = 40
    df = pd.DataFrame({"high": [11.0] * n, "low": [9.0] * n, "close": [10.0] * n})
    assert ind.atr(df).iloc[-1] == pytest.approx(2.0)


def test_updown_volume_ratio():
    close = pd.Series([10.0, 11, 10, 11, 10, 11])
    df = pd.DataFrame({"close": close, "volume": [100.0, 300, 100, 300, 100, 300]})
    # changes: +,-,+,-,+ -> up volume 900, down volume 200 (n=5 window)
    assert ind.updown_volume_ratio(df, 5).iloc[-1] == pytest.approx(900 / 200)


def test_weighted_performance():
    close = pd.Series(np.full(300, 100.0))
    close.iloc[-1] = 110.0
    # every lookback sees +10%, weights sum to 1
    assert ind.weighted_performance(close) == pytest.approx(0.10)
    assert ind.weighted_performance(close.head(200)) is None


def test_compute_all_columns_and_alignment():
    px, bench = make_prices(seed=1), make_prices(seed=2)
    out = ind.compute_all(px, bench["close"])
    assert set(out.columns) == set(ind.INDICATORS)
    assert out.index.equals(px.index)
    assert out["sma200"].isna().sum() == 199
    assert out["rs_line"].notna().all()
