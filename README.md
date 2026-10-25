# Ticker: swing-trading research dashboard

Daily candlesticks, indicators and a William O'Neil **CAN SLIM** scorecard for tickers you choose,
published as a static GitHub Pages site. Information for swing trading only, not investment advice.

## How it works

1. The **Analyze** workflow (Actions tab → *Run workflow*, or the daily schedule) downloads prices and
   fundamentals from Yahoo Finance via `yfinance`.
2. Python computes every indicator and the CAN SLIM scorecard and writes `docs/data/<TICKER>.json`
   plus `docs/data/index.json`, then commits them.
3. `docs/index.html` (static, no dependencies) draws the candlestick chart, scorecard and overview from that JSON.

## One-time setup

- Settings → Pages → *Deploy from a branch* → branch `main`, folder `/docs`.
- Run **Analyze** with e.g. `NVDA AAPL`. Leave the tickers empty to refresh everything on both lists.
- Note: on a public repo the site and the tickers you analyse are public.

## Lists

- `watchlist.txt`: tickers you are actively watching. `candidates.txt`: tickers under consideration for the watchlist.
- Both are refreshed every weekday after the US close and shown in separate sections of the **Overview** tab
  (a ticker on both counts as watchlist). Tickers analysed once but on neither list appear under **Other**
  and are not refreshed; their "As of" date shows how old they are.
- Move tickers without editing files: Actions → **Analyze** → Run workflow → enter tickers → **Add to**:
  `watchlist`, `candidates`, or `remove` (takes them off both lists and deletes their published data; nothing is fetched).
  Adding to one list takes the ticker off the other.
- Or edit the two text files directly (one symbol per line, `#` for comments).

## Pages

- **Chart**: candles, SMA 10/21/50/150/200 (toggle in the legend), volume with 50-day average, RSI(14),
  MACD(12,26,9), relative-strength line vs SPY. Drag to pan, scroll/pinch to zoom.
- **Chart watch** (under the chart): a rule-based reading of the latest bars, with what stands out
  (trend-template checks, extension from the 50-day, breakouts and whether volume confirms, heavy-volume days,
  RSI/MACD, RS-line leadership, volatility, gaps) and a table of levels to watch (breakout trigger, moving-average
  support, loss-cutting references). Rules and thresholds are in `scripts/chart_notes.py`. Descriptive, not predictive.
- **CAN SLIM**: one card per letter with PASS / PARTIAL / FAIL / N/A, the numbers behind it, and a tally.
- **Overview**: every analysed ticker in one sortable table.

## CAN SLIM mapping (thresholds in `THRESHOLDS`, `scripts/canslim.py`)

| | Check |
|---|---|
| C | Latest-quarter EPS growth YoY ≥ 25% (≥ 10% partial); accelerating/decelerating noted |
| A | 3-year EPS CAGR ≥ 25% and ROE ≥ 17% |
| N | Within 5% of the 52-week high (within 15% partial) |
| S | Up/down volume (50d) ≥ 1.2; float and volume-vs-average shown |
| L | Weighted 12-month return vs SPY (**proxy** for IBD's RS Rating) |
| I | % held by institutions, 30–90% (**proxy**; no ownership trend from yfinance) |
| M | SPY and QQQ vs 50/200-day averages and distribution days (worst of the two) |

Data that Yahoo does not provide is shown as **N/A** and excluded from the tally, never guessed.

## Add an indicator

Write a function in `scripts/indicators.py`, add one line to `INDICATORS`, and (to draw it) add the key to
`CHART_SERIES` in `scripts/build_site.py` and a drawing branch in `docs/index.html`.

## Local use

```bash
pip install -r requirements-dev.txt
python -m pytest                                   # tests use synthetic data, no network
python scripts/fetch_and_build.py NVDA AAPL        # needs access to Yahoo Finance
python scripts/lists.py candidates "PLTR UBER"      # edit the lists (watchlist | candidates | remove)
python -m http.server -d docs 8000                 # then open http://localhost:8000
python tests/make_demo_site.py /tmp/demo           # preview the page with SYNTHETIC data, offline
```
