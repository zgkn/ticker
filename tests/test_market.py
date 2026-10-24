import numpy as np
import pandas as pd

import market
from synth import make_prices


def _df(closes, vols):
    idx = pd.bdate_range(end="2026-10-23", periods=len(closes))
    return pd.DataFrame({"close": closes, "volume": vols}, index=idx)


def test_distribution_days_counts_down_days_on_higher_volume():
    closes = [100.0] * 30
    vols = [1000.0] * 30
    # two qualifying days: -0.5% on higher volume
    closes[28], vols[28] = 99.5, 1500.0
    closes[29], vols[29] = 99.0, 1600.0
    # a down day on LOWER volume does not count
    closes[27], vols[27] = 99.9, 900.0
    assert market.distribution_days(_df(closes, vols)) == 2


def test_uptrend_passes_downtrend_fails():
    up = make_prices(n=300, drift=0.002, vol=0.004, seed=3)
    down = make_prices(n=300, drift=-0.002, vol=0.004, seed=4)
    assert market.evaluate({"SPY": up})["status"] == market.PASS
    assert market.evaluate({"SPY": down})["status"] == market.FAIL


def test_overall_is_most_conservative_index():
    up = make_prices(n=300, drift=0.002, vol=0.004, seed=3)
    down = make_prices(n=300, drift=-0.002, vol=0.004, seed=4)
    assert market.evaluate({"SPY": up, "QQQ": down})["status"] == market.FAIL


def test_no_data_is_na():
    assert market.evaluate({"SPY": make_prices(n=20)})["status"] == market.NA


def test_follow_through_day_detected():
    # decline, bottom, then a rally attempt with a +2% heavy-volume day on day 4
    closes = list(np.linspace(120, 100, 40)) + [101, 102, 101.5, 104.1, 104.5, 105]
    vols = [1000.0] * 40 + [900, 950, 900, 1500, 1000, 1000]
    assert market.follow_through_day(_df(closes, vols)) is not None
    flat = _df([100.0] * 50, [1000.0] * 50)
    assert market.follow_through_day(flat) is None
