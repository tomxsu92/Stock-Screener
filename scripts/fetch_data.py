#!/usr/bin/env python3
"""
Stock Potential Screener - data builder.

Downloads one year of daily prices + fundamentals for every ticker in
scripts/tickers.txt (via yfinance), computes technical indicators, gives each
stock a 0-100 "potential score", and writes a rule-based entry / stop / target /
hold plan. Output: data/stocks.json, which the web page reads.

Usage:
    python scripts/fetch_data.py            # real data (needs internet + yfinance)
    python scripts/fetch_data.py --demo     # fake demo data, no internet needed

Educational tool only - not financial advice.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TICKERS_FILE = ROOT / "scripts" / "tickers.txt"
OUT_FILE = ROOT / "data" / "stocks.json"

WEIGHTS = {"trend": 0.25, "momentum": 0.25, "growth": 0.20, "value": 0.15, "upside": 0.15}
NEUTRAL = 40.0  # score used when a metric is missing (slightly below average)


# --------------------------------------------------------------------------- helpers
def num(x):
    """Return a clean float or None (handles NaN, inf, strings, None)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def rnd(x, d=2):
    x = num(x)
    return None if x is None else round(x, d)


def load_tickers() -> list[str]:
    out = []
    for line in TICKERS_FILE.read_text().splitlines():
        t = line.split("#")[0].strip().upper()
        if t and t not in out:
            out.append(t)
    return out


def pct_rank(values: list, higher_is_better=True) -> list:
    """Percentile rank 0-100 for each value; None stays None."""
    s = pd.Series([num(v) for v in values], dtype="float64")
    r = s.rank(pct=True, ascending=higher_is_better) * 100
    return [None if pd.isna(v) else float(v) for v in r]


# --------------------------------------------------------------------------- indicators
def rsi(close: pd.Series, n=14) -> float | None:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    val = 100 - 100 / (1 + rs)
    return num(val.iloc[-1]) if len(val) else None


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n=14) -> float | None:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return num(tr.rolling(n).mean().iloc[-1])


def ret(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return num((close.iloc[-1] / close.iloc[-1 - days] - 1) * 100)


def technicals(h: pd.DataFrame) -> dict | None:
    h = h.dropna(subset=["Close"])
    if len(h) < 60:
        return None
    c, hi, lo, v = h["Close"], h["High"], h["Low"], h["Volume"]
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean() if len(c) >= 200 else c.rolling(len(c)).mean()
    price = num(c.iloc[-1])
    high52, low52 = num(hi.max()), num(lo.min())
    vol20 = num(v.tail(20).mean())
    vol60 = num(v.tail(60).mean())
    spark = c.tail(120).iloc[::2].round(2).tolist()
    return {
        "price": price,
        "change_1d": ret(c, 1),
        "ret_1m": ret(c, 21),
        "ret_3m": ret(c, 63),
        "ret_6m": ret(c, 126),
        "ret_1y": ret(c, len(c) - 1),
        "sma20": num(sma20.iloc[-1]),
        "sma50": num(sma50.iloc[-1]),
        "sma200": num(sma200.iloc[-1]),
        "sma50_slope": num((sma50.iloc[-1] / sma50.iloc[-21] - 1) * 100) if len(c) > 70 else None,
        "rsi": rsi(c),
        "atr": atr(hi, lo, c),
        "high52": high52,
        "low52": low52,
        "off_high": num((price / high52 - 1) * 100) if high52 else None,
        "vol_ratio": num(vol20 / vol60) if vol20 and vol60 else None,
        "spark": [num(x) for x in spark],
    }


# --------------------------------------------------------------------------- scoring
def trend_score(t: dict) -> float:
    p, s50, s200 = t["price"], t["sma50"], t["sma200"]
    pts = 0.0
    if p and s50 and p > s50:
        pts += 30
    if s50 and s200 and s50 > s200:
        pts += 30
    if p and s200 and p > s200:
        pts += 25
    if (t.get("sma50_slope") or 0) > 0:
        pts += 15
    return pts


def score_universe(rows: list[dict]) -> None:
    mom_raw = []
    for r in rows:
        parts = [(r.get("ret_6m"), 0.5), (r.get("ret_3m"), 0.3), (r.get("ret_1m"), 0.2)]
        have = [(v, w) for v, w in parts if v is not None]
        mom_raw.append(sum(v * w for v, w in have) / sum(w for _, w in have) if have else None)

    rev = pct_rank([r.get("revenue_growth") for r in rows])
    eps = pct_rank([r.get("earnings_growth") for r in rows])
    mom = pct_rank(mom_raw)
    upside = pct_rank([r.get("upside") for r in rows])

    # Value: lower PEG is better; fall back to lower forward P/E. Ignore negatives.
    peg = [r.get("peg") if (r.get("peg") or 0) > 0 else None for r in rows]
    fpe = [r.get("forward_pe") if (r.get("forward_pe") or 0) > 0 else None for r in rows]
    peg_r = pct_rank(peg, higher_is_better=False)
    fpe_r = pct_rank(fpe, higher_is_better=False)

    for i, r in enumerate(rows):
        g = [x for x in (rev[i], eps[i]) if x is not None]
        sub = {
            "trend": trend_score(r),
            "momentum": mom[i] if mom[i] is not None else NEUTRAL,
            "growth": sum(g) / len(g) if g else NEUTRAL,
            "value": peg_r[i] if peg_r[i] is not None else (fpe_r[i] if fpe_r[i] is not None else NEUTRAL),
            "upside": upside[i] if upside[i] is not None else NEUTRAL,
        }
        r["scores"] = {k: round(v) for k, v in sub.items()}
        r["score"] = round(sum(sub[k] * w for k, w in WEIGHTS.items()))
        r.update(trade_plan(r))


# --------------------------------------------------------------------------- trade plan
def trade_plan(r: dict) -> dict:
    p, s20, s50, s200 = r["price"], r["sma20"], r["sma50"], r["sma200"]
    a = r.get("atr") or p * 0.02
    rsi_v = r.get("rsi") or 50
    up_trend = s50 > s200 and p > s200
    down_trend = p < s200 and s50 < s200

    if down_trend:
        signal, why = "Avoid", "Downtrend: price and 50-day average are below the 200-day average."
    elif rsi_v > 70 or p > s20 * 1.08:
        signal, why = "Wait", "Extended / overbought. Wait for a pullback toward the entry zone."
    elif up_trend and min(s20, s50) * 0.98 <= p <= max(s20, s50) * 1.01:
        signal, why = "Buy zone", "Uptrend and price has pulled back near its 20/50-day averages."
    elif up_trend and p > s50:
        signal, why = "Uptrend", "Healthy uptrend. Better risk/reward on a dip to the entry zone."
    else:
        signal, why = "Watch", "Mixed trend. Wait for price to hold above the 200-day average."

    if up_trend:
        lo, hi = sorted([s20, s50])
        entry_note = "Pullback to 20-50 day averages"
    else:
        lo, hi = s200, s200 * 1.03
        entry_note = "Reclaim and hold the 200-day average"
    if hi / lo - 1 < 0.02:
        hi = lo * 1.02
    mid = (lo + hi) / 2

    stop = max(lo - 2 * a, mid * 0.75)  # never risk more than 25%
    risk = mid - stop
    analyst = r.get("target_price")
    if analyst and analyst > mid + risk:
        t1, t1_src = analyst, "Analyst average target"
    else:
        t1, t1_src = mid + 2 * risk, "2x risk"
    t2 = max(mid + 3 * risk, t1 * 1.1)

    s = r["scores"]
    if s["trend"] >= 70 and s["growth"] >= 60:
        hold = "6-12 months"
        hold_note = "Trend + growth. Hold while it stays above the 50-day average; review after each earnings report."
    elif s["value"] >= 60 and s["trend"] < 70:
        hold = "12+ months"
        hold_note = "Value-driven idea. Needs time for the market to re-rate it; review quarterly."
    elif s["momentum"] >= 70:
        hold = "1-3 months"
        hold_note = "Momentum trade. Take profits into strength; exit fast if momentum fades."
    else:
        hold = "3-6 months"
        hold_note = "Medium-term position. Re-check monthly."

    exits = [
        f"Stop loss: exit if it closes below ${stop:,.2f}.",
        f"Take ~1/3 profit at target 1 (${t1:,.2f}), move the stop up to your entry price.",
        f"Let the rest run to target 2 (${t2:,.2f}) with a trailing stop of 2x ATR (${2 * a:,.2f}).",
        "Trend exit: two daily closes below the 50-day average.",
    ]
    return {
        "signal": signal,
        "signal_note": why,
        "entry_low": rnd(lo),
        "entry_high": rnd(hi),
        "entry_note": entry_note,
        "stop": rnd(stop),
        "target1": rnd(t1),
        "target1_src": t1_src,
        "target2": rnd(t2),
        "risk_pct": rnd(risk / mid * 100, 1),
        "reward_risk": rnd((t1 - mid) / risk, 1) if risk > 0 else None,
        "hold": hold,
        "hold_note": hold_note,
        "exit_rules": exits,
    }


# --------------------------------------------------------------------------- real data
def fundamentals(info: dict, price: float) -> dict:
    target = num(info.get("targetMeanPrice"))
    peg = num(info.get("trailingPegRatio")) or num(info.get("pegRatio"))
    return {
        "name": info.get("shortName") or info.get("longName"),
        "sector": info.get("sector") or "Other",
        "industry": info.get("industry"),
        "market_cap": num(info.get("marketCap")),
        "forward_pe": num(info.get("forwardPE")),
        "trailing_pe": num(info.get("trailingPE")),
        "peg": peg,
        "revenue_growth": rnd((num(info.get("revenueGrowth")) or 0) * 100, 1) if info.get("revenueGrowth") is not None else None,
        "earnings_growth": rnd((num(info.get("earningsGrowth")) or 0) * 100, 1) if info.get("earningsGrowth") is not None else None,
        "profit_margin": rnd((num(info.get("profitMargins")) or 0) * 100, 1) if info.get("profitMargins") is not None else None,
        "target_price": target,
        "upside": rnd((target / price - 1) * 100, 1) if target and price else None,
        "analyst_rating": info.get("recommendationKey"),
        "analyst_count": num(info.get("numberOfAnalystOpinions")),
    }


def fetch_real(tickers: list[str]) -> list[dict]:
    import yfinance as yf

    print(f"Downloading price history for {len(tickers)} tickers...")
    hist = yf.download(tickers, period="1y", interval="1d", group_by="ticker",
                       auto_adjust=True, threads=True, progress=False)
    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            h = hist[t] if isinstance(hist.columns, pd.MultiIndex) else hist
            tech = technicals(h)
            if not tech:
                print(f"  [{i}/{len(tickers)}] {t}: not enough history, skipped")
                continue
            info = {}
            for attempt in range(3):
                try:
                    info = yf.Ticker(t).info or {}
                    break
                except Exception as e:  # rate limits etc.
                    print(f"  {t}: info retry {attempt + 1} ({e})")
                    time.sleep(2 * (attempt + 1))
            row = {"ticker": t, **tech, **fundamentals(info, tech["price"])}
            row["name"] = row["name"] or t
            rows.append(row)
            print(f"  [{i}/{len(tickers)}] {t} ok")
            time.sleep(0.3)
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {t}: error {e}")
    return rows


# --------------------------------------------------------------------------- demo data
def fetch_demo(n=40) -> list[dict]:
    rng = np.random.default_rng(7)
    sectors = ["Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
               "Industrials", "Energy", "Communication Services"]
    rows = []
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252)
    for i in range(n):
        drift = rng.normal(0.0004, 0.0012)
        vol = rng.uniform(0.012, 0.035)
        start = rng.uniform(20, 400)
        close = start * np.exp(np.cumsum(rng.normal(drift, vol, len(dates))))
        high = close * (1 + rng.uniform(0, vol, len(dates)))
        low = close * (1 - rng.uniform(0, vol, len(dates)))
        volume = rng.integers(1_000_000, 30_000_000, len(dates)).astype(float)
        h = pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume}, index=dates)
        tech = technicals(h)
        price = tech["price"]
        target = price * rng.uniform(0.85, 1.4) if rng.random() > 0.1 else None
        info = {
            "shortName": f"Demo Company {i + 1:02d}",
            "sector": sectors[i % len(sectors)],
            "industry": "Demo industry",
            "marketCap": rng.uniform(2e9, 2e12),
            "forwardPE": rng.uniform(8, 70),
            "trailingPE": rng.uniform(10, 90),
            "trailingPegRatio": rng.uniform(0.5, 4) if rng.random() > 0.2 else None,
            "revenueGrowth": rng.normal(0.1, 0.15),
            "earningsGrowth": rng.normal(0.12, 0.3) if rng.random() > 0.15 else None,
            "profitMargins": rng.normal(0.15, 0.1),
            "targetMeanPrice": target,
            "recommendationKey": rng.choice(["strong_buy", "buy", "hold", "underperform"]),
            "numberOfAnalystOpinions": int(rng.integers(3, 50)),
        }
        rows.append({"ticker": f"DEMO{i + 1:02d}", **tech, **fundamentals(info, price)})
    return rows


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="generate fake demo data (no internet)")
    args = ap.parse_args()

    rows = fetch_demo() if args.demo else fetch_real(load_tickers())
    if not rows:
        sys.exit("No data fetched - keeping the previous data/stocks.json")

    score_universe(rows)
    rows.sort(key=lambda r: r["score"], reverse=True)

    for r in rows:  # round for a smaller file
        for k, v in list(r.items()):
            if isinstance(v, float):
                r[k] = round(v, 2)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "demo": bool(args.demo),
        "count": len(rows),
        "weights": WEIGHTS,
        "stocks": rows,
    }
    OUT_FILE.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {len(rows)} stocks to {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
