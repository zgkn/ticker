"""Build a throw-away copy of the site filled with SYNTHETIC data, for previewing the page offline.

    python tests/make_demo_site.py /path/to/out_dir

Never point this at docs/: the data is random and must not be published as real.
"""
import os
import shutil
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import build_site  # noqa: E402
from synth import make_fundamentals, make_prices  # noqa: E402


def main(out: str) -> None:
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    shutil.copy(os.path.join(HERE, "..", "docs", "index.html"), os.path.join(out, "index.html"))
    data_dir = os.path.join(out, "data")
    spy, qqq = make_prices(seed=2, drift=0.0005, vol=0.009), make_prices(seed=3, drift=0.0007, vol=0.012)
    specs = [("DEMOUP", 11, 0.0012, True), ("DEMODN", 12, -0.0008, False), ("DEMOCHOP", 13, 0.0001, True)]
    rows = []
    for sym, seed, drift, strong in specs:
        news = [
            {"title": f"{sym} unveils new Rocket X platform", "date": "2026-10-21", "source": "Demo Wire", "url": "https://example.com/rocket", "summary": ""},
            {"title": "Analysts trim price target <img src=x onerror=alert(1)>", "date": "2026-10-22", "source": "Demo Blog", "url": "javascript:alert(1)", "summary": ""},
            {"title": f"{sym} launches cloud service for small teams", "date": "2026-10-05", "source": "Demo Wire", "url": "https://example.com/cloud", "summary": ""},
        ]
        f = make_fundamentals(strong) | {"symbol": sym, "name": f"{sym.title()} (synthetic)", "news": news}
        p = build_site.build_payload(sym, make_prices(seed=seed, drift=drift), spy, qqq, f)
        build_site.write_ticker(p, data_dir)
        rows.append(build_site.summary_row(p))
    build_site.update_index(rows, data_dir, lists={"DEMOUP": "watch", "DEMODN": "consider"})   # DEMOCHOP stays "other"

    # Screener demo: two SYNTHETIC "found" stocks that are not on any list (they live under data/screen/)
    import screener
    picks = []
    for sym, seed in (("SCRNA", 31), ("SCRNB", 32)):
        f = make_fundamentals(True) | {"symbol": sym, "name": f"{sym.title()} (synthetic)"}
        p = build_site.build_payload(sym, make_prices(seed=seed, drift=0.0012), spy, qqq, f)
        p["screen"] = {"outperf": 0.4 - 0.1 * (seed - 31), "avg_volume": 1.2e6}
        build_site.write_ticker(p, os.path.join(data_dir, "screen"))
        picks.append(p)
    build_site.write_json(os.path.join(data_dir, "screener.json"), {
        "generated_at": "2026-10-24T08:17:00+00:00", "as_of": "2026-10-23",
        "params": {"min_price": 10, "max_price": 100, "min_avg_volume": 500000, "max_off_high": 0.25, "top": 20},
        "market": {"status": "PARTIAL", "headline": "Market direction mixed"},
        "funnel": {"universe": 903, "downloaded": 887, "stage1": 74, "scored": 60, "scoring_failed": 2, "shortlist": 2,
                   "stage1_rejections": {"price_out_of_range": 410, "low_volume": 38, "too_far_below_high": 255, "below_200d": 90, "lagging_spy": 20, "too_little_history": 0}, "capped_at": 60},
        "tickers": [screener.screener_row(p) for p in picks],
    })
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
