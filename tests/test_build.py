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


def test_parse_news_handles_current_and_legacy_yfinance_shapes():
    raw = [
        {"id": "1", "content": {"title": " New thing launched ", "summary": "s", "pubDate": "2026-10-20T14:30:00Z",
                                 "provider": {"displayName": "Reuters"}, "canonicalUrl": {"url": "https://x.test/1"}}},
        {"title": "Legacy item", "publisher": "Old Wire", "link": "https://x.test/2", "providerPublishTime": 1792000000},
        {"content": {"title": "Bad link", "pubDate": "2026-10-21T00:00:00Z", "canonicalUrl": {"url": "javascript:alert(1)"}}},
        {"content": {"title": ""}},
        "garbage",
    ]
    out = data_lib.parse_news(raw)
    assert [n["title"] for n in out] == ["Bad link", "New thing launched", "Legacy item"]
    by_title = {n["title"]: n for n in out}
    assert by_title["New thing launched"] == {"title": "New thing launched", "summary": "s", "date": "2026-10-20", "source": "Reuters", "url": "https://x.test/1"}
    assert by_title["Legacy item"]["source"] == "Old Wire" and by_title["Legacy item"]["url"] == "https://x.test/2"
    assert by_title["Legacy item"]["date"].startswith("20")
    assert by_title["Bad link"]["url"] == ""          # non-http(s) links are dropped
    assert data_lib.parse_news(None) == [] and data_lib.parse_news([]) == []


def test_fetch_news_merges_tabs_dedupes_and_uses_a_fresh_ticker_per_tab(monkeypatch):
    import yfinance

    made = []

    class FakeTicker:
        def __init__(self, sym):
            self.tabs = []
            made.append(self)

        def get_news(self, count=10, tab="news"):
            self.tabs.append(tab)
            if tab == "news":
                return [{"content": {"title": "Shared headline", "pubDate": "2026-10-20T00:00:00Z"}},
                        {"content": {"title": "Only in news", "pubDate": "2026-10-22T00:00:00Z"}}]
            return [{"content": {"title": "shared headline", "pubDate": "2026-10-20T00:00:00Z"}},     # same title, other case
                    {"content": {"title": "Acme unveils new thing", "pubDate": "2026-10-23T00:00:00Z"}}]

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    out = data_lib.fetch_news("ACME")
    assert [n["title"] for n in out] == ["Acme unveils new thing", "Only in news", "Shared headline"]
    assert len(made) == 2 and [t.tabs for t in made] == [["news"], ["press releases"]]


def test_fetch_news_survives_a_failing_tab(monkeypatch):
    import yfinance

    class FlakyTicker:
        def __init__(self, sym):
            pass

        def get_news(self, count=10, tab="news"):
            if tab == "news":
                raise RuntimeError("boom")
            return [{"content": {"title": "Press release", "pubDate": "2026-10-23T00:00:00Z"}}]

    monkeypatch.setattr(yfinance, "Ticker", FlakyTicker)
    monkeypatch.setattr(data_lib.time, "sleep", lambda s: None)   # skip retry back-off
    assert [n["title"] for n in data_lib.fetch_news("ACME")] == ["Press release"]
