import json
import os

import numpy as np
import pandas as pd
import pytest

import build_site
import data_lib
import screener
from synth import make_fundamentals, make_ohlcv, make_prices

N = 400


def line(a, b, n=N):
    return np.linspace(a, b, n)


@pytest.fixture
def spy():
    return make_ohlcv(line(100, 140))            # +40% over the window


def test_parse_wikipedia_symbols_picks_the_symbol_table_and_yahoo_style():
    html = """<table><tr><th>Index</th></tr><tr><td>x</td></tr></table>
    <table><tr><th>Symbol</th><th>Security</th></tr>
      <tr><td>AAA</td><td>A</td></tr><tr><td>BRK.B</td><td>B</td></tr><tr><td>BF.B</td><td>C</td></tr>
      <tr><td>AAA</td><td>dup</td></tr><tr><td>bad sym!</td><td>D</td></tr></table>"""
    assert screener.parse_wikipedia_symbols(html) == ["AAA", "BRK-B", "BF-B"]
    assert screener.parse_wikipedia_symbols("<table><tr><th>X</th></tr><tr><td>1</td></tr></table>") == []


def test_split_download_multiindex_flat_and_missing():
    a, b = make_ohlcv(line(10, 20, 50)), make_ohlcv(line(30, 40, 50))
    raw = pd.concat({"AAA": a.rename(columns=str.title), "BBB": b.rename(columns=str.title)}, axis=1)
    out = screener.split_download(raw, ["AAA", "BBB", "ZZZ"])
    assert sorted(out) == ["AAA", "BBB"] and list(out["AAA"].columns) == ["open", "high", "low", "close", "volume"]
    assert screener.split_download(a.rename(columns=str.title), ["AAA"])["AAA"]["close"].iloc[-1] == pytest.approx(20)
    assert screener.split_download(None, ["AAA"]) == {} and screener.split_download(pd.DataFrame(), ["AAA"]) == {}


def test_stage1_applies_each_filter_and_ranks_survivors(spy):
    thin = np.full(N, 1e5)
    prices = {
        "GOOD": make_ohlcv(line(40, 80)),                                   # in range, steady uptrend, beats SPY (+100% vs +40%)
        "BEST": make_ohlcv(line(30, 90)),                                   # beats SPY by even more
        "PRICEY": make_ohlcv(line(150, 300)),
        "CHEAP": make_ohlcv(line(2, 6)),
        "THIN": make_ohlcv(line(40, 80), volume=thin),
        "FAR": make_ohlcv(np.concatenate([line(40, 100, 300), line(100, 60, 100)])),      # 40% off its high, still in range
        "DOWN": make_ohlcv(line(90, 50)),
        "LAG": make_ohlcv(line(50, 60)),                                    # steady but slower than SPY
        "SHORT": make_ohlcv(line(40, 80, 100)),
    }
    keep, rej = screener.stage1(prices, spy, 10, 100)
    assert [k["symbol"] for k in keep] == ["BEST", "GOOD"]                  # best-first by outperformance
    assert rej["price_out_of_range"] == 2 and rej["low_volume"] == 1 and rej["too_little_history"] == 1
    assert rej["too_far_below_high"] >= 2 and rej["lagging_spy"] == 1
    assert sum(rej.values()) + len(keep) == len(prices)
    assert all(10 <= k["close"] <= 100 for k in keep) and keep[0]["outperf"] > keep[1]["outperf"] > 0


def test_price_cap_is_a_parameter(spy):
    prices = {"GOOD": make_ohlcv(line(40, 80))}
    assert screener.stage1(prices, spy, 10, 100)[0]
    assert not screener.stage1(prices, spy, 10, 60)[0]          # last close 80 > cap 60
    assert not screener.stage1(prices, spy, 90, 200)[0]         # last close 80 < min 90


def _payloads(spy_frame, strong):
    prices = {s: make_prices(seed=i + 5, drift=0.001) for i, s in enumerate(strong)}
    fund = lambda sym: make_fundamentals(strong[sym]) | {"symbol": sym, "name": sym}
    rows = [{"symbol": s, "outperf": 0.5 - 0.1 * i, "avg_volume": 1e6} for i, s in enumerate(strong)]
    return screener.stage2(rows, prices, spy_frame, None, fund)


def test_shortlist_requires_c_and_a_and_ranks_by_pass_count():
    spy_frame = make_prices(seed=2)
    payloads, failed = _payloads(spy_frame, {"S1": True, "W1": False, "S2": True, "W2": False})
    assert not failed and len(payloads) == 4
    picks = screener.shortlist(payloads, top=10)
    assert {p["symbol"] for p in picks} == {"S1", "S2"}                      # weak-earnings names never make it
    assert all(p["canslim"]["letters"][k]["status"] == "PASS" for p in picks for k in "CA")
    passes = [p["canslim"]["tally"]["pass"] for p in picks]
    assert passes == sorted(passes, reverse=True)
    assert len(screener.shortlist(payloads, top=1)) == 1


def test_stage2_isolates_a_failing_ticker():
    spy_frame = make_prices(seed=2)
    prices = {"OK": make_prices(seed=5), "BAD": make_prices(seed=6)}

    def fund(sym):
        if sym == "BAD":
            raise RuntimeError("yahoo said no")
        return make_fundamentals(True) | {"symbol": sym, "name": sym}

    rows = [{"symbol": s, "outperf": 0.1, "avg_volume": 1e6} for s in prices]
    payloads, failed = screener.stage2(rows, prices, spy_frame, None, fund)
    assert [p["symbol"] for p in payloads] == ["OK"] and failed == ["BAD"]


def test_run_end_to_end_writes_table_and_per_ticker_files(tmp_path, monkeypatch, spy):
    out_json, out_dir = str(tmp_path / "screener.json"), str(tmp_path / "screen")
    os.makedirs(out_dir)
    open(os.path.join(out_dir, "STALE.json"), "w").write("{}")               # leftovers from a previous run must go
    universe = {"AAA": make_ohlcv(line(25, 80, 756)), "BBB": make_ohlcv(line(90, 50, 756)), "CCC": make_ohlcv(line(200, 400, 756))}
    spy_p = make_prices(seed=2, drift=0.0003)
    monkeypatch.setattr(screener, "fetch_universe", lambda *a, **k: list(universe))
    monkeypatch.setattr(data_lib, "fetch_prices", lambda sym, period=data_lib.HISTORY_PERIOD: spy_p)
    monkeypatch.setattr(screener, "download_prices", lambda syms, **k: {s: universe[s] for s in syms})
    monkeypatch.setattr(data_lib, "fetch_fundamentals", lambda sym, with_news=True: make_fundamentals(True) | {"symbol": sym, "name": sym})
    assert screener.run(10, 100, top=5, out_json=out_json, out_dir=out_dir) == 0

    res = json.load(open(out_json))
    json.dumps(res, allow_nan=False)
    assert res["params"]["max_price"] == 100 and res["funnel"]["universe"] == 3 and res["funnel"]["downloaded"] == 3
    assert res["funnel"]["stage1"] == res["funnel"]["scored"] == 1             # only AAA: BBB is falling, CCC is over the cap
    assert res["funnel"]["stage1_rejections"]["price_out_of_range"] >= 1       # CCC is far above the cap
    syms = [t["symbol"] for t in res["tickers"]]
    assert syms == ["AAA"]
    assert sorted(f[:-5] for f in os.listdir(out_dir)) == sorted(syms)         # one file per shortlisted ticker, stale one gone
    assert all(t["letters"]["C"] == "PASS" and t["letters"]["A"] == "PASS" for t in res["tickers"])
    assert set(res["market"]) == {"status", "headline"}


def test_run_fails_cleanly_with_no_universe_or_no_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(screener, "fetch_universe", lambda *a, **k: [])
    assert screener.run(10, 100, out_json=str(tmp_path / "s.json"), out_dir=str(tmp_path / "d")) == 2
    monkeypatch.setattr(screener, "fetch_universe", lambda *a, **k: ["AAA"])
    monkeypatch.setattr(data_lib, "fetch_prices", lambda sym, period=data_lib.HISTORY_PERIOD: make_prices(seed=2))
    monkeypatch.setattr(screener, "download_prices", lambda syms, **k: {})
    assert screener.run(10, 100, out_json=str(tmp_path / "s.json"), out_dir=str(tmp_path / "d")) == 1
    assert not (tmp_path / "s.json").exists()                                  # nothing published on failure


def test_cli_argument_validation():
    for bad in (["--min-price", "50", "--max-price", "10"], ["--top", "0"], ["--max-stage2", "1000"]):
        with pytest.raises(SystemExit):
            screener.main(bad)
