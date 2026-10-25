import json

import numpy as np
import pytest

import chart_notes as cn
from synth import make_px, make_prices


def titles(res):
    return [o["title"] for o in res["observations"]]


def find(res, text):
    return [o for o in res["observations"] if text in o["title"]]


def test_steady_uptrend_is_seven_of_seven_and_extended_or_not():
    px = make_px(np.linspace(50, 100, 400))
    res = cn.analyze(px)
    assert find(res, "7/7")[0]["tag"] == "bullish"


def test_downtrend_is_weak_and_below_50d():
    px = make_px(np.linspace(100, 50, 400))
    res = cn.analyze(px)
    weak = find(res, "Weak trend structure")
    assert weak and weak[0]["tag"] == "bearish" and "Missing:" in weak[0]["detail"]
    assert find(res, "below the 50-day")


def test_extended_above_50d():
    path = np.concatenate([np.linspace(50, 60, 350), np.linspace(60, 90, 50)])
    res = cn.analyze(make_px(path))
    assert find(res, "Extended")


def test_new_high_needs_volume_to_be_confirmed():
    path = np.concatenate([np.linspace(50, 100, 399), [101.0]])
    light = cn.analyze(make_px(path))
    assert find(light, "light volume")[0]["tag"] == "caution"
    vol = np.full(400, 1e6)
    vol[-1] = 2.5e6
    strong = cn.analyze(make_px(path, vol))
    assert find(strong, "strong volume")[0]["tag"] == "bullish"


def test_heavy_volume_selloff_is_flagged_with_its_date():
    path = np.concatenate([np.linspace(50, 100, 397), [100.0, 100.0, 96.0]])
    vol = np.full(400, 1e6)
    vol[-1] = 3e6
    res = cn.analyze(make_px(path, vol))
    hit = find(res, "Heavy-volume selloff")
    assert hit and hit[0]["tag"] == "caution" and "2026-10-23" in hit[0]["title"]


def test_rsi_overbought_on_a_steep_run():
    res = cn.analyze(make_px(np.concatenate([np.full(300, 50.0), np.linspace(50, 80, 100)])))
    assert find(res, "overbought")


def _walk(start, steps):
    """Price path from a list of per-bar fractional changes."""
    out, p = [], start
    for st in steps:
        p *= 1 + st
        out.append(p)
    return out


def test_bearish_rsi_divergence_when_price_makes_a_higher_high_on_weaker_momentum():
    base = [0.003, -0.003] * 140                    # noisy base
    rally = [0.03] * 10                             # sharp run: very high RSI at the first peak
    pullback = [-0.01] * 15
    grind = [0.015, -0.005] * 20                    # slower climb to a higher high, RSI lower
    path = _walk(50.0, base + rally + pullback + grind)
    res = cn.analyze(make_px(path))
    div = find(res, "Bearish RSI divergence")
    assert div and div[0]["tag"] == "bearish"


def test_no_divergence_on_a_plain_uptrend():
    assert not find(cn.analyze(make_px(np.linspace(50, 100, 400))), "divergence")


def test_levels_have_breakout_trigger_stop_reference_and_are_sorted():
    px = make_px(np.concatenate([np.linspace(50, 100, 380), np.linspace(100, 99.9, 20)]), volume=np.full(400, 2e6))
    levels = cn.analyze(px)["levels"]
    by = {lv["label"]: lv for lv in levels}
    assert "52-week high" in by and by["52-week high"]["side"] == "up"
    assert "2.8M" in by["52-week high"]["note"]            # 1.4 x 2.0M average volume
    assert "50-day average" in by and by["50-day average"]["side"] == "down"
    assert any(lv["label"].startswith("7-8%") for lv in levels)
    prices = [lv["price"] for lv in levels]
    assert prices == sorted(prices, reverse=True)
    assert by["50-day average"]["pct"] < 0 < by["52-week high"]["pct"]


def test_short_history_does_not_crash_and_is_strict_json():
    for n in (5, 30, 60, 120):
        res = cn.analyze(make_px(np.linspace(50, 60, n)))
        json.dumps(res, allow_nan=False)


def test_build_payload_includes_chart_notes():
    import build_site
    from synth import make_fundamentals

    p = build_site.build_payload("T", make_prices(seed=1), make_prices(seed=2), None, make_fundamentals(), generated_at="t")
    json.dumps(p, allow_nan=False)
    assert set(p["chart_notes"]) == {"observations", "levels"}
    assert all(o["tag"] in {"bullish", "bearish", "caution", "info"} for o in p["chart_notes"]["observations"])
