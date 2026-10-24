import numpy as np
import pandas as pd
import pytest

import canslim as cs
import indicators as ind
from synth import make_fundamentals, make_prices

PASS, PARTIAL, FAIL, NA = cs.PASS, cs.PARTIAL, cs.FAIL, cs.NA


def eps(*vals):
    dates = ["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"][: len(vals)]
    return list(zip(dates, vals))


def test_c_pass_partial_fail_na():
    assert cs.letter_c({"quarterly_eps": eps(1.0, 1, 1, 1, 1.5)})["status"] == PASS          # +50%
    assert cs.letter_c({"quarterly_eps": eps(1.0, 1, 1, 1, 1.15)})["status"] == PARTIAL      # +15%
    assert cs.letter_c({"quarterly_eps": eps(1.0, 1, 1, 1, 1.02)})["status"] == FAIL         # +2%
    assert cs.letter_c({"quarterly_eps": eps(1.0, 1, 1)})["status"] == NA                    # < 5 quarters
    assert cs.letter_c({})["status"] == NA


def test_c_turnaround_and_loss():
    assert cs.letter_c({"quarterly_eps": eps(-0.2, 1, 1, 1, 0.3)})["status"] == PARTIAL
    assert cs.letter_c({"quarterly_eps": eps(-0.2, 1, 1, 1, -0.1)})["status"] == FAIL


def test_c_acceleration_note_needs_six_quarters():
    res = cs.letter_c({"quarterly_eps": eps(1.0, 1.0, 1, 1, 1.5, 1.8)})
    assert any("accelerating" in n for n in res["notes"])


def test_c_ignores_prior_quarter_that_is_not_a_year_back():
    # 5 entries but the "year ago" slot is missing (dates skip): must not compare against the wrong quarter
    f = {"quarterly_eps": [("2026-03-31", 1.0), ("2026-06-30", 1.0), ("2026-09-30", 1.0), ("2026-10-31", 1.0), ("2026-12-31", 1.1)]}
    assert cs.letter_c(f)["status"] == NA


def test_a_pass_fail_and_partial():
    assert cs.letter_a(make_fundamentals(True))["status"] == PASS
    assert cs.letter_a(make_fundamentals(False))["status"] == FAIL
    f = make_fundamentals(True) | {"roe": 0.05}
    assert cs.letter_a(f)["status"] == PARTIAL
    assert cs.letter_a({})["status"] == NA


def test_a_uses_roe_alone_when_eps_missing():
    assert cs.letter_a({"roe": 0.30})["status"] == PASS


def _px(close_path, volume=None):
    n = len(close_path)
    idx = pd.bdate_range(end="2026-10-23", periods=n)
    close = pd.Series(close_path, index=idx, dtype=float)
    df = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
                       "volume": volume if volume is not None else np.full(n, 1e6)}, index=idx)
    return df.join(ind.compute_all(df, close))


def test_n_levels():
    near = _px(np.linspace(50, 100, 300))
    assert cs.letter_n(near)["status"] == PASS
    mid = _px(np.concatenate([np.linspace(50, 100, 250), np.linspace(100, 90, 50)]))
    assert cs.letter_n(mid)["status"] == PARTIAL
    far = _px(np.concatenate([np.linspace(50, 100, 250), np.linspace(100, 60, 50)]))
    res = cs.letter_n(far)
    assert res["status"] == FAIL
    assert any("200-day" in n for n in res["notes"])


def test_s_accumulation_vs_distribution():
    n = 300
    close = np.cumsum(np.tile([1.0, -1.0], n // 2)) + 100
    up_heavy = np.tile([3e6, 1e6], n // 2)      # up days heavy
    down_heavy = np.tile([1e6, 3e6], n // 2)
    f = {"float_shares": 2e9, "shares_outstanding": 2.1e9}
    assert cs.letter_s(_px(close, up_heavy), f)["status"] == PASS
    assert cs.letter_s(_px(close, down_heavy), f)["status"] == FAIL


def test_l_leader_and_laggard():
    n = 400
    bench = pd.Series(np.linspace(100, 110, n), index=pd.bdate_range(end="2026-10-23", periods=n))
    leader = _px(np.linspace(100, 200, n))
    laggard = _px(np.linspace(100, 90, n))
    leader = leader.assign(rs_line=ind.rs_line(leader["close"], bench))
    laggard = laggard.assign(rs_line=ind.rs_line(laggard["close"], bench))
    assert cs.letter_l(leader, bench)["status"] == PASS
    assert cs.letter_l(laggard, bench)["status"] == FAIL
    assert cs.letter_l(leader.head(100), bench.head(100))["status"] == NA


def test_i_bands():
    assert cs.letter_i({"inst_pct": 0.6})["status"] == PASS
    assert cs.letter_i({"inst_pct": 0.2})["status"] == PARTIAL
    assert cs.letter_i({"inst_pct": 0.95})["status"] == PARTIAL
    assert cs.letter_i({"inst_pct": 0.02})["status"] == FAIL
    assert cs.letter_i({})["status"] == NA


def test_evaluate_tally_and_summary():
    px_raw = make_prices(seed=1, drift=0.001)
    bench = make_prices(seed=2, drift=0.0003)
    px = px_raw.join(ind.compute_all(px_raw, bench["close"]))
    market = {"status": PASS, "headline": "x", "metrics": [], "notes": []}
    res = cs.evaluate(px, bench["close"], make_fundamentals(True), market)
    t = res["tally"]
    assert set(res["letters"]) == set("CANSLIM")
    assert t["pass"] + t["partial"] + t["fail"] + t["na"] == 7
    assert t["scored"] == 7 - t["na"]
    assert f"{t['pass']} of {t['scored']}" in res["summary"]
    assert all(r["label"] for r in res["letters"].values())


def test_evaluate_with_no_fundamentals_marks_na_not_zero():
    px_raw, bench = make_prices(seed=1), make_prices(seed=2)
    px = px_raw.join(ind.compute_all(px_raw, bench["close"]))
    market = {"status": NA, "headline": "x", "metrics": [], "notes": []}
    res = cs.evaluate(px, bench["close"], {}, market)
    assert res["letters"]["C"]["status"] == NA and res["letters"]["A"]["status"] == NA
    assert res["tally"]["na"] >= 4  # C, A, I, M
