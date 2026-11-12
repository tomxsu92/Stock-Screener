# Stock Potential Screener

A free stock screener that runs entirely on GitHub:

- **GitHub Actions** pulls one year of prices and fundamentals for every ticker in
  `scripts/tickers.txt` (via Yahoo Finance / yfinance) each weekday after the market closes.
- It scores every stock 0–100 on **trend, momentum, growth, value and analyst upside**, and writes a
  rule-based **entry zone, stop loss, two targets, hold time and exit rules** for each one.
- **GitHub Pages** serves `index.html`, which lets you search, filter and sort the results.

> Educational tool only. Not financial advice.

## Set up (one time)

1. Push these files to a **public** repo (e.g. `stock-screener`).
2. **Turn on Pages:** repo → Settings → Pages → *Build and deployment* → Source: **Deploy from a branch**,
   Branch: **main**, folder **/ (root)** → Save.
   Your site will be at `https://<your-username>.github.io/stock-screener/`.
3. **Allow the Action to save data:** Settings → Actions → General → *Workflow permissions* →
   **Read and write permissions** → Save.
4. **Get real data now:** Actions tab → **Update stock data** → **Run workflow**. It takes a few
   minutes. After that it runs automatically every weekday at 2:30 PM Pacific.

Until step 4 runs, the site shows clearly labeled **demo data** (fake companies).

## Change which stocks are scanned

Edit `scripts/tickers.txt`. Put one ticker per line, and use Yahoo's symbols (e.g. `BRK-B`). Commit the change,
then run the workflow again or wait for the next scheduled run.

## How it works

| Piece | File |
|---|---|
| Web page (search, filters, detail panel) | `index.html` |
| Data builder + scoring + trade plan | `scripts/fetch_data.py` |
| Daily schedule | `.github/workflows/update-data.yml` |
| Output the page reads | `data/stocks.json` |

**Score** = 25% Trend + 25% Momentum + 20% Growth + 15% Value + 15% Analyst upside.
Every part except Trend is ranked against the other stocks in your list.
Change the weights at the top of `fetch_data.py`.

**Signals:** Buy zone, Uptrend, Wait (overbought), Watch (mixed), Avoid (downtrend).

**Plan:** the entry zone is the area between the 20- and 50-day averages (in an uptrend), or the point where price
reclaims the 200-day average. The stop sits 2× ATR below the entry zone. Target 1 is the analyst average target
(or 2× risk), and target 2 is 3× risk.

Run locally:

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # real data
python scripts/fetch_data.py --demo   # fake data, no internet
python -m http.server                 # then open http://localhost:8000
```
