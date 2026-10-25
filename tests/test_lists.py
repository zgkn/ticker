import json

import pytest

import build_site
import fetch_and_build
import lists
from synth import make_fundamentals, make_prices


@pytest.fixture
def files(tmp_path):
    w, c = tmp_path / "watchlist.txt", tmp_path / "candidates.txt"
    w.write_text("# watched\nNVDA\naapl  # my core holding\n\nMSFT\n")
    c.write_text("# maybe\nPLTR\nNU\n")
    return str(w), str(c)


def test_read_lists_order_dedupe_and_watch_wins(files):
    w, c = files
    open(c, "a").write("NVDA\n")                      # on both lists -> counts as watch
    assert lists.read_lists(w, c) == {"NVDA": "watch", "AAPL": "watch", "MSFT": "watch", "PLTR": "consider", "NU": "consider"}
    assert list(lists.read_lists(w, c)) == ["NVDA", "AAPL", "MSFT", "PLTR", "NU"]


def test_read_lists_missing_files_are_empty(tmp_path):
    assert lists.read_lists(str(tmp_path / "no.txt"), str(tmp_path / "nope.txt")) == {}


def test_add_moves_between_lists_and_keeps_comments(files):
    w, c = files
    out = lists.apply("watchlist", ["PLTR", "AAPL"], w, c)
    assert out == {"added": ["PLTR"], "removed_from_other": ["PLTR"]}      # AAPL was already on the watchlist
    assert lists.read_lists(w, c)["PLTR"] == "watch"
    assert open(c).read() == "# maybe\nNU\n"
    assert open(w).read().startswith("# watched\nNVDA\naapl  # my core holding\n")      # comments untouched
    lists.apply("candidates", ["MSFT"], w, c)
    assert lists.read_lists(w, c)["MSFT"] == "consider" and "MSFT" not in open(w).read()


def test_add_to_a_file_without_trailing_newline(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("AAA")
    lists.add_symbols(str(p), ["BBB"])
    assert p.read_text() == "AAA\nBBB\n"


def test_remove_deletes_from_both_lists_and_published_data(files, tmp_path):
    w, c = files
    d = tmp_path / "data"
    for sym in ("NU", "AAPL"):
        build_site.write_ticker(build_site.build_payload(sym, make_prices(seed=1), make_prices(seed=2), None, make_fundamentals(), generated_at="t"), str(d))
    build_site.update_index([build_site.summary_row(json.load(open(d / f"{s}.json"))) for s in ("NU", "AAPL")], str(d))
    out = lists.apply("remove", ["NU", "ZZZ"], w, c, str(d))
    assert out["removed"] == ["NU"] and out["data_deleted"] == ["NU"]
    assert not (d / "NU.json").exists() and (d / "AAPL.json").exists()
    assert [t["symbol"] for t in json.load(open(d / "index.json"))["tickers"]] == ["AAPL"]
    assert "NU" not in lists.read_lists(w, c)


def test_unknown_action_and_bad_symbols_are_rejected(files):
    with pytest.raises(ValueError):
        lists.apply("delete-everything", ["AAPL"], *files)
    with pytest.raises(ValueError):
        lists.main(["watchlist", "../etc/passwd"])


def test_update_index_tags_every_row_with_its_list(tmp_path):
    d = str(tmp_path)
    row = lambda s: build_site.summary_row(build_site.build_payload(s, make_prices(seed=1), make_prices(seed=2), None, make_fundamentals(), generated_at="t"))
    build_site.update_index([row("AAA"), row("BBB"), row("CCC")], d)
    idx = build_site.update_index([row("AAA")], d, lists={"AAA": "watch", "BBB": "consider"})
    assert {t["symbol"]: t["list"] for t in idx["tickers"]} == {"AAA": "watch", "BBB": "consider", "CCC": "other"}


def test_cli_without_tickers_refreshes_both_lists(files, monkeypatch, tmp_path):
    import data_lib

    w, c = files
    monkeypatch.setattr(lists, "WATCHLIST", w)
    monkeypatch.setattr(lists, "CANDIDATES", c)
    monkeypatch.setattr(build_site, "DATA_DIR", str(tmp_path))
    spy = make_prices(seed=2)
    seen = []
    monkeypatch.setattr(data_lib, "fetch_prices", lambda sym, period=data_lib.HISTORY_PERIOD: seen.append(sym) or (spy if sym in ("SPY", "QQQ") else make_prices(seed=9)))
    monkeypatch.setattr(data_lib, "fetch_fundamentals", lambda sym: make_fundamentals(True) | {"symbol": sym, "name": sym})
    assert fetch_and_build.main([]) == 0
    assert [s for s in seen if s not in ("SPY", "QQQ")] == ["NVDA", "AAPL", "MSFT", "PLTR", "NU"]
    idx = {t["symbol"]: t["list"] for t in json.load(open(tmp_path / "index.json"))["tickers"]}
    assert idx == {"NVDA": "watch", "AAPL": "watch", "MSFT": "watch", "PLTR": "consider", "NU": "consider"}
