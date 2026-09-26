"""Builds the list of NSE / BSE companies to analyse."""
import io
import logging

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

INDEX_URLS = {
    "NIFTY50": "ind_nifty50list.csv",
    "NIFTY100": "ind_nifty100list.csv",
    "NIFTY200": "ind_nifty200list.csv",
    "NIFTY500": "ind_nifty500list.csv",
}
INDEX_HOSTS = [
    "https://archives.nseindia.com/content/indices/",
    "https://niftyindices.com/IndexConstituent/",
]
NSE_ALL_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

# Used only when NSE's own constituent files cannot be downloaded.
FALLBACK_SYMBOLS = """
RELIANCE TCS HDFCBANK ICICIBANK INFY BHARTIARTL SBIN ITC LT HINDUNILVR
KOTAKBANK AXISBANK BAJFINANCE HCLTECH MARUTI SUNPHARMA M&M TITAN ULTRACEMCO
NTPC ONGC POWERGRID TATAMOTORS TATASTEEL ADANIENT ADANIPORTS ASIANPAINT
BAJAJFINSV WIPRO NESTLEIND JSWSTEEL COALINDIA GRASIM TECHM HINDALCO
INDUSINDBK CIPLA DRREDDY BRITANNIA EICHERMOT HEROMOTOCO APOLLOHOSP
DIVISLAB BAJAJ-AUTO SBILIFE HDFCLIFE TATACONSUM BPCL SHRIRAMFIN TRENT BEL
HAL IOC GAIL VEDL DLF SIEMENS PIDILITIND DABUR GODREJCP HAVELLS AMBUJACEM
BANKBARODA PNB CANBK INDIGO TVSMOTOR CHOLAFIN JIOFIN ZOMATO LTIM ABB
ADANIGREEN ADANIPOWER TATAPOWER RECLTD PFC IRFC ICICIPRULI ICICIGI
SBICARD NAUKRI DMART MARICO COLPAL BERGEPAINT SHREECEM TORNTPHARM LUPIN
AUROPHARMA ZYDUSLIFE BOSCHLTD MOTHERSON UNITDSPR MCDOWELL-N PAGEIND
BHARATFORG CUMMINSIND POLYCAB DIXON PERSISTENT COFORGE MPHASIS LICI
IRCTC BHEL SAIL NMDC HINDPETRO INDHOTEL TATACOMM TATAELXSI VOLTAS
""".split()


def _session():
    s = requests.Session()
    s.headers.update(config.HTTP_HEADERS)
    try:  # NSE sets anti-bot cookies on the home page
        s.get("https://www.nseindia.com", timeout=10)
    except requests.RequestException:
        pass
    return s


def _read_csv(session, url):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    return df


def _index_constituents(name):
    session = _session()
    for host in INDEX_HOSTS:
        url = host + INDEX_URLS[name]
        try:
            df = _read_csv(session, url)
            out = pd.DataFrame({
                "symbol": df["Symbol"].str.strip(),
                "company": df["Company Name"].str.strip(),
                "industry": df.get("Industry", pd.Series(index=df.index, dtype=str)),
            })
            log.info("Loaded %d constituents of %s from %s", len(out), name, url)
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("Could not load %s: %s", url, e)
    return None


def _nse_all():
    try:
        df = _read_csv(_session(), NSE_ALL_URL)
        df = df[df["SERIES"].str.strip() == "EQ"]
        log.info("Loaded %d NSE equity symbols", len(df))
        return pd.DataFrame({
            "symbol": df["SYMBOL"].str.strip(),
            "company": df["NAME OF COMPANY"].str.strip(),
            "industry": None,
        })
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load NSE equity list: %s", e)
        return None


def _read_list(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line.upper())
    return out


def load_universe(name=None):
    """Return DataFrame[symbol, company, industry, exchange, ticker]."""
    name = (name or config.UNIVERSE).upper()
    df = _nse_all() if name == "NSE_ALL" else _index_constituents(name if name in INDEX_URLS else "NIFTY500")
    if df is None or df.empty:
        log.warning("Falling back to built-in list of %d large caps", len(FALLBACK_SYMBOLS))
        df = pd.DataFrame({"symbol": FALLBACK_SYMBOLS, "company": None, "industry": None})
    df["exchange"] = "NSE"
    df["ticker"] = df["symbol"] + ".NS"

    extra = []
    for sym in _read_list(config.CONFIG_DIR / "watchlist.txt"):
        extra.append({"symbol": sym, "company": None, "industry": None, "exchange": "NSE", "ticker": sym + ".NS"})
    for sym in _read_list(config.CONFIG_DIR / "bse_symbols.txt"):
        tkr = sym if sym.endswith(".BO") else sym + ".BO"
        extra.append({"symbol": tkr[:-3], "company": None, "industry": None, "exchange": "BSE", "ticker": tkr})
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    df = df.drop_duplicates("ticker").reset_index(drop=True)
    return df
