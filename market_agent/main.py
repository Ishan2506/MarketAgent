"""Entry point: python -m market_agent.main [--universe NIFTY500] [--force] [--demo]"""
import argparse
import logging
import shutil
import sys
import time

import numpy as np
import pandas as pd

from . import config, data, report, tracker
from .indicators import build_panel
from .model import train_and_predict
from .scoring import apply_scores
from .universe import load_universe

log = logging.getLogger("market_agent")


def _demo_prices(n=60, days=800, seed=7):
    """Synthetic random-walk prices so the pipeline can be tested offline."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)

    def walk(start, vol):
        r = rng.normal(0.0004, vol, days) + 0.1 * np.r_[0, rng.normal(0, vol, days - 1)]
        c = start * np.exp(np.cumsum(r))
        o = c * (1 + rng.normal(0, vol / 3, days))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, days)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, days)))
        v = rng.lognormal(13, 0.5, days)
        return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)

    prices = {f"DEMO{i:02d}.NS": walk(rng.uniform(50, 3000), rng.uniform(0.01, 0.03)) for i in range(n)}
    uni = pd.DataFrame({"symbol": [t[:-3] for t in prices], "company": [f"Demo Co {i}" for i in range(n)],
                        "industry": "Demo", "exchange": "NSE", "ticker": list(prices)})
    return uni, prices, walk(20000, 0.009), walk(14, 0.04), walk(70000, 0.009)


def run(universe_name, force=False, demo=False):
    t0 = time.time()
    horizon = config.HORIZON_DAYS

    if demo:
        uni, prices, nifty, vix, sensex = _demo_prices()
        fundamentals = pd.DataFrame(index=uni["ticker"])
        delivery = pd.DataFrame(columns=["deliv_per", "deliv_avg5", "deliv_ratio"])
        reports_dir = config.REPORTS_DIR / "demo"
    else:
        uni = load_universe(universe_name)
        log.info("Universe %s: %d companies", universe_name, len(uni))
        idx = data.download_prices([config.MARKET_INDEX, config.VIX_INDEX, config.SENSEX_INDEX])
        if config.MARKET_INDEX not in idx:
            log.error("Could not download NIFTY data - Yahoo Finance unreachable?")
            return 1
        nifty, vix, sensex = idx[config.MARKET_INDEX], idx.get(config.VIX_INDEX), idx.get(config.SENSEX_INDEX)
        reports_dir = config.REPORTS_DIR

    as_of = nifty.index.max()
    as_of_d = as_of.date()
    out_path = reports_dir / f"{as_of:%Y}" / f"{as_of:%m}" / f"Market_Prediction_{as_of:%Y-%m-%d}.xlsx"
    if out_path.exists() and not force:
        log.info("Report for %s already exists (market holiday?). Use --force to rebuild.", as_of_d)
        return 0

    if not demo:
        prices = data.download_prices(uni["ticker"].tolist())
        if len(prices) < 10:
            log.error("Too few stocks downloaded (%d); aborting", len(prices))
            return 1
        fundamentals = data.load_fundamentals(list(prices))
        delivery = data.load_delivery(as_of_d)

    log.info("Computing indicators for %d stocks", len(prices))
    panel = build_panel(prices, nifty, vix, horizon)

    log.info("Training model (%d rows)", len(panel))
    today, metrics, importance = train_and_predict(panel, horizon, as_of)

    today = today.merge(uni, on="ticker", how="left")
    today = today.merge(fundamentals.rename_axis("ticker").reset_index(),
                        on="ticker", how="left", suffixes=("", "_f"))
    if "company_name" in today:
        today["company"] = today["company"].fillna(today["company_name"])
    if "yf_industry" in today:
        today["industry"] = today["industry"].fillna(today["yf_industry"])
    today = today.merge(delivery, left_on="symbol", right_index=True, how="left")

    # Drop illiquid / penny stocks that are hard to trade
    liquid = (today["turnover_cr"] >= config.MIN_AVG_TURNOVER_CR) & (today["close"] >= config.MIN_PRICE)
    log.info("Excluding %d illiquid/penny stocks", int((~liquid).sum()))
    today = today[liquid | demo].copy()

    today = apply_scores(today, horizon, config.ML_WEIGHT, as_of)

    nc, vc = nifty["Close"], (vix["Close"] if vix is not None and len(vix) else None)
    market = {
        "NIFTY 50 close": f"{nc.iloc[-1]:,.2f} ({nc.pct_change().iloc[-1]*100:+.2f}% today)",
        "NIFTY 50 trend": "Above 50-DMA (bullish)" if nc.iloc[-1] > nc.rolling(50).mean().iloc[-1] else "Below 50-DMA (bearish)",
        "NIFTY 5-day / 20-day change": f"{nc.pct_change(5).iloc[-1]*100:+.2f}% / {nc.pct_change(20).iloc[-1]*100:+.2f}%",
    }
    if sensex is not None and len(sensex):
        sc = sensex["Close"]
        market["SENSEX close"] = f"{sc.iloc[-1]:,.2f} ({sc.pct_change().iloc[-1]*100:+.2f}% today)"
    if vc is not None:
        market["India VIX"] = f"{vc.iloc[-1]:.2f} ({'high fear' if vc.iloc[-1] > 20 else 'calm' if vc.iloc[-1] < 14 else 'normal'})"
    market["Stocks predicted UP vs DOWN"] = (
        f"{today['prediction'].str.contains('UP').sum()} vs {today['prediction'].str.contains('DOWN').sum()}")

    # Track record
    closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()})
    history = tracker.load() if not demo else pd.DataFrame(columns=tracker.COLS)
    history = tracker.evaluate(history, closes, horizon)
    by_signal, by_date = tracker.summary(history)
    detail = history[history["correct"].notna()].sort_values("date", ascending=False).head(3000).copy()
    if not detail.empty:
        detail["correct"] = np.where(detail["correct"].astype(bool), "YES", "NO")
        detail["date"] = pd.to_datetime(detail["date"]).dt.date
        detail = detail.rename(columns={"date": "Prediction Date", "symbol": "Symbol", "close": "Price Then",
                                        "prob_up": "Prob. Up", "prediction": "Prediction",
                                        "expected_ret": "Predicted Move", "actual_ret": "Actual Return",
                                        "max_ret_in_window": "Max Return in Window", "correct": "Correct?"}) \
            .drop(columns=["ticker"])
    history = tracker.append(history, today, as_of)

    report.write_report(out_path, as_of_d, today, market, metrics, importance, by_signal, by_date,
                        detail, "DEMO (synthetic data)" if demo else universe_name, horizon)
    shutil.copyfile(out_path, reports_dir / "Latest_Market_Prediction.xlsx")
    if not demo:
        tracker.save(history)
    log.info("Report written: %s (%.1f min)", out_path, (time.time() - t0) / 60)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="NSE/BSE short-term price direction agent")
    ap.add_argument("--universe", default=config.UNIVERSE,
                    help="NIFTY50 | NIFTY100 | NIFTY200 | NIFTY500 | NSE_ALL")
    ap.add_argument("--force", action="store_true", help="rebuild even if today's report exists")
    ap.add_argument("--demo", action="store_true", help="run offline on synthetic data (for testing)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    return run(args.universe.upper(), force=args.force, demo=args.demo)


if __name__ == "__main__":
    sys.exit(main())
