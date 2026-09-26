"""Downloads prices (Yahoo Finance), fundamentals (Yahoo Finance) and
delivery data (NSE bhavcopy)."""
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from . import config

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _split_download(raw, tickers):
    out = {}
    if raw is None or raw.empty:
        return out
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({tickers[0]: raw}, axis=1)
    level0 = set(raw.columns.get_level_values(0))
    for t in tickers:
        if t not in level0:
            continue
        df = raw[t]
        if not set(OHLCV).issubset(df.columns):
            continue
        df = df[OHLCV].dropna(subset=["Close"])
        if len(df) > 0:
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            out[t] = df
    return out


def download_prices(tickers, years=None, chunk=80, retries=3):
    """Return {ticker: OHLCV DataFrame} of daily split/dividend adjusted prices."""
    period = f"{years or config.HISTORY_YEARS}y"
    result = {}
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        for attempt in range(retries):
            try:
                raw = yf.download(batch, period=period, interval="1d", group_by="ticker",
                                  auto_adjust=True, threads=True, progress=False)
                got = _split_download(raw, batch)
                result.update(got)
                break
            except Exception as e:  # noqa: BLE001
                log.warning("Download batch %d failed (attempt %d): %s", i // chunk, attempt + 1, e)
                time.sleep(5 * (attempt + 1))
        log.info("Prices: %d/%d tickers downloaded", len(result), len(tickers))

    # Stocks that failed on NSE: try the BSE listing instead
    missing = [t for t in tickers if t not in result and t.endswith(".NS")]
    if missing:
        alt = [t[:-3] + ".BO" for t in missing]
        try:
            raw = yf.download(alt, period=period, interval="1d", group_by="ticker",
                              auto_adjust=True, threads=True, progress=False)
            for bo, df in _split_download(raw, alt).items():
                result[bo[:-3] + ".NS"] = df.assign(_via_bse=True)
            log.info("Recovered %d tickers via BSE", sum(1 for t in missing if t in result))
        except Exception as e:  # noqa: BLE001
            log.warning("BSE fallback failed: %s", e)
    return result


FUNDAMENTAL_FIELDS = {
    "longName": "company_name",
    "sector": "sector",
    "industry": "yf_industry",
    "marketCap": "market_cap",
    "trailingPE": "pe",
    "forwardPE": "forward_pe",
    "priceToBook": "pb",
    "returnOnEquity": "roe",
    "returnOnAssets": "roa",
    "debtToEquity": "debt_to_equity",
    "earningsGrowth": "earnings_growth",
    "revenueGrowth": "revenue_growth",
    "profitMargins": "profit_margin",
    "operatingMargins": "operating_margin",
    "dividendYield": "dividend_yield",
    "beta": "beta",
    "currentRatio": "current_ratio",
    "heldPercentInsiders": "promoter_holding",
    "heldPercentInstitutions": "institutional_holding",
    "targetMeanPrice": "analyst_target",
    "recommendationMean": "analyst_rating",
    "recommendationKey": "analyst_view",
    "numberOfAnalystOpinions": "analyst_count",
    "earningsTimestamp": "earnings_ts",
}


def _fetch_info(ticker):
    for attempt in range(2):
        try:
            info = yf.Ticker(ticker).info or {}
            row = {v: info.get(k) for k, v in FUNDAMENTAL_FIELDS.items()}
            row["ticker"] = ticker
            return row
        except Exception:  # noqa: BLE001
            time.sleep(2 + attempt * 3)
    return {"ticker": ticker}


def load_fundamentals(tickers, max_age_days=None, workers=8, budget_sec=1500):
    """Fundamentals change slowly, so they are cached in data/fundamentals.csv
    and only refreshed when older than ``max_age_days``."""
    max_age_days = config.FUNDAMENTALS_MAX_AGE_DAYS if max_age_days is None else max_age_days
    cache_path = config.DATA_DIR / "fundamentals.csv"
    cache = pd.DataFrame(columns=["ticker", "fetched_on"])
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    cache = cache.drop_duplicates("ticker", keep="last").set_index("ticker")

    if not config.FETCH_FUNDAMENTALS:
        return cache.reindex(tickers)

    today = pd.Timestamp.today().normalize()
    fetched = pd.to_datetime(cache.get("fetched_on"), errors="coerce")
    fresh = set(cache.index[(today - fetched).dt.days <= max_age_days]) if len(cache) else set()
    todo = [t for t in tickers if t not in fresh]
    log.info("Fundamentals: %d cached, %d to fetch", len(tickers) - len(todo), len(todo))

    rows, start = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_info, t): t for t in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            if len(row) > 1:
                row["fetched_on"] = today.date().isoformat()
                rows.append(row)
            if n % 100 == 0:
                log.info("Fundamentals: %d/%d fetched", n, len(todo))
            if time.time() - start > budget_sec:
                log.warning("Fundamentals time budget hit; remaining use cache")
                for f in futures:
                    f.cancel()
                break

    if rows:
        new = pd.DataFrame(rows).set_index("ticker")
        cache = pd.concat([cache[~cache.index.isin(new.index)], new])
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache.reset_index().to_csv(cache_path, index=False)
    return cache.reindex(tickers)


def _bhavcopy(session, d):
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv"
    r = session.get(url, timeout=20)
    if r.status_code != 200 or len(r.content) < 1000:
        return None
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda s: s.str.strip() if pd.api.types.is_string_dtype(s) else s)
    df = df[df["SERIES"] == "EQ"]
    return pd.DataFrame({
        "symbol": df["SYMBOL"],
        "date": pd.Timestamp(d),
        "deliv_per": pd.to_numeric(df["DELIV_PER"], errors="coerce"),
    })


def load_delivery(as_of, days=6, max_lookback=14):
    """Delivery % (share of traded quantity actually taken into demat) from
    NSE bhavcopies. High delivery on up-days signals genuine buying."""
    session = requests.Session()
    session.headers.update(config.HTTP_HEADERS)
    frames, d = [], as_of
    for _ in range(max_lookback):
        if len(frames) >= days:
            break
        if d.weekday() < 5:
            try:
                f = _bhavcopy(session, d)
                if f is not None:
                    frames.append(f)
            except Exception as e:  # noqa: BLE001
                log.debug("bhavcopy %s: %s", d, e)
        d -= timedelta(days=1)
    if not frames:
        log.warning("NSE delivery data unavailable today; skipping that parameter")
        return pd.DataFrame(columns=["deliv_per", "deliv_avg5", "deliv_ratio"])
    all_ = pd.concat(frames).sort_values("date")
    latest = all_.groupby("symbol").last()["deliv_per"]
    prior = all_[all_["date"] < all_["date"].max()].groupby("symbol")["deliv_per"].mean()
    out = pd.DataFrame({"deliv_per": latest, "deliv_avg5": prior})
    out["deliv_ratio"] = out["deliv_per"] / out["deliv_avg5"].replace(0, np.nan)
    log.info("Delivery data loaded for %d symbols (%d sessions)", len(out), len(frames))
    return out

