import json

import pytest

import build_site
import data_lib
import fetch_and_build
from synth import make_fundamentals, make_prices


def payload(symbol="TEST", seed=1):
    spy, qqq = make_prices(seed=2), make_prices(seed=3)
    return build_site.build_payload(symbol, make_prices(seed=seed), spy, qqq, make_fundamentals(True), generated_at="2026-10-24T00:00:00+00:00")


def test_payload_is_strict_json_and_aligned():
    p = payload()
    json.dumps(p, allow_nan=False)
    n = len(p["dates"])
    assert n == build_site.DISPLAY_BARS
    for name, arr in p["series"].items():
        assert len(arr) == n, name
    assert p["series"]["sma200"][0] is not None       # warmed up on the first bar shown
    assert p["series"]["rs_line"][0] == pytest.approx(100.0)
    assert p["as_of"] == p["dates"][-1]
    assert set(p["canslim"]["letters"]) == set("CANSLIM")


def test_payload_without_qqq_still_builds():
    p = build_site.build_payload("X", make_prices(seed=1), make_prices(seed=2), None, make_fundamentals(), generated_at="t")
    assert p["canslim"]["letters"]["M"]["status"] in ("PASS", "PARTIAL", "FAIL")


def test_index_merges_by_symbol(tmp_path):
    d = str(tmp_path)
    build_site.update_index([build_site.summary_row(payload("AAA"))], d)
    idx = build_site.update_index([build_site.summary_row(payload("BBB")), build_site.summary_row(payload("AAA", seed=5))], d)
    assert [t["symbol"] for t in idx["tickers"]] == ["AAA", "BBB"]
    build_site.write_ticker(payload("AAA"), d)
    assert json.load(open(tmp_path / "AAA.json"))["symbol"] == "AAA"


def test_symbol_validation():
    assert fetch_and_build.parse_tickers(["nvda, aapl", "BRK-B nvda"]) == ["NVDA", "AAPL", "BRK-B"]
    for bad in ('A"; rm -rf /', "../etc/passwd", "A B/C", "x" * 40):
        if " " in bad:
            continue
        with pytest.raises(ValueError):
            data_lib.normalize_symbol(bad)
    with pytest.raises(ValueError):
        fetch_and_build.parse_tickers(["../x"])


def test_cli_isolates_a_failing_ticker(tmp_path, monkeypatch):
    spy, qqq = make_prices(seed=2), make_prices(seed=3)

    def fake_prices(sym, period=data_lib.HISTORY_PERIOD):
        if sym == "BAD":
            raise RuntimeError("no price data returned for BAD")
        return {"SPY": spy, "QQQ": qqq}.get(sym, make_prices(seed=9))

    monkeypatch.setattr(data_lib, "fetch_prices", fake_prices)
    monkeypatch.setattr(data_lib, "fetch_fundamentals", lambda sym: make_fundamentals(True) | {"symbol": sym, "name": sym})
    monkeypatch.setattr(build_site, "DATA_DIR", str(tmp_path))

    assert fetch_and_build.main(["GOOD", "BAD", "good"]) == 1          # failure is reported...
    assert (tmp_path / "GOOD.json").exists() and not (tmp_path / "BAD.json").exists()   # ...but GOOD is still written
    idx = json.load(open(tmp_path / "index.json"))
    assert [t["symbol"] for t in idx["tickers"]] == ["GOOD"]

    assert fetch_and_build.main(["GOOD"]) == 0


def test_cli_rejects_bad_symbol_before_any_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network must not be touched")
    monkeypatch.setattr(data_lib, "fetch_prices", boom)
    with pytest.raises(ValueError):
        fetch_and_build.main(["NVDA;rm"])
