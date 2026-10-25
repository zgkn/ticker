"""Print a markdown scorecard table from docs/data/index.json (for $GITHUB_STEP_SUMMARY)."""
import json
import os
import sys

PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "index.json")
GLYPH = {"PASS": "✅", "PARTIAL": "🟡", "FAIL": "❌", "N/A": "➖"}


SCREENER = os.path.join(os.path.dirname(PATH), "screener.json")


def screener_summary() -> None:
    if not os.path.exists(SCREENER):
        print("No screener.json was produced.")
        return
    with open(SCREENER) as fh:
        sc = json.load(fh)
    f, p = sc["funnel"], sc["params"]
    print(f"## Screener: ${p['min_price']:.0f}-${p['max_price']:.0f}\n")
    print(f"Scanned {f['downloaded']} -> {f['stage1']} passed price/trend filters -> {f['scored']} scored -> **{f['shortlist']}** with C and A passing. "
          f"Market direction: **{sc['market']['status']}**.\n")
    if sc["tickers"]:
        print("| Ticker | Close | Off 52w high | C | A | N | S | L | I | M | Pass |")
        print("|---|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---:|")
        for t in sc["tickers"]:
            letters = " | ".join(GLYPH.get(t["letters"].get(k, "N/A"), "➖") for k in "CANSLIM")
            print(f"| **{t['symbol']}** | {t['close']:.2f} | {t['pct_off_high'] * 100:+.1f}% | {letters} | {t['tally']['pass']}/{t['tally']['scored']} |")
    print("\n_Information only, not investment advice._")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "screener":
        return screener_summary()
    if not os.path.exists(PATH):
        print("No data/index.json was produced.")
        return
    with open(PATH) as fh:
        tickers = json.load(fh).get("tickers", [])
    print("## CAN SLIM scorecard\n")
    print("| Ticker | Close | Off 52w high | RSI | C | A | N | S | L | I | M | Pass |")
    print("|---|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---:|")
    for t in tickers:
        letters = " | ".join(GLYPH.get(t["letters"].get(k, "N/A"), "➖") for k in "CANSLIM")
        rsi = "n/a" if t["rsi14"] is None else f"{t['rsi14']:.0f}"
        print(f"| **{t['symbol']}** | {t['close']:.2f} | {t['pct_off_high'] * 100:+.1f}% | {rsi} | {letters} | {t['tally']['pass']}/{t['tally']['scored']} |")
    print("\n_Information only, not investment advice. L and I are proxies._")


if __name__ == "__main__":
    sys.exit(main())
