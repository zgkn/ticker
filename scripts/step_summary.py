"""Print a markdown scorecard table from docs/data/index.json (for $GITHUB_STEP_SUMMARY)."""
import json
import os
import sys

PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "index.json")
GLYPH = {"PASS": "✅", "PARTIAL": "🟡", "FAIL": "❌", "N/A": "➖"}


def main() -> None:
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
