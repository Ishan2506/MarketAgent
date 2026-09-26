"""Central settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"


def _env(name, default, cast=str):
    val = os.environ.get(name)
    return cast(val) if val not in (None, "") else default


# Which companies to analyse: NIFTY50 | NIFTY100 | NIFTY200 | NIFTY500 | NSE_ALL
UNIVERSE = _env("UNIVERSE", "NIFTY500")

# Prediction horizon in trading days (7 trading days ~= 10 calendar days)
HORIZON_DAYS = _env("HORIZON_DAYS", 7, int)

# Years of daily history used to train the model
HISTORY_YEARS = _env("HISTORY_YEARS", 3, int)

# Fundamentals (P/E, ROE, ...) are refreshed from Yahoo when older than this
FUNDAMENTALS_MAX_AGE_DAYS = _env("FUNDAMENTALS_MAX_AGE_DAYS", 7, int)
FETCH_FUNDAMENTALS = _env("FETCH_FUNDAMENTALS", "1") == "1"

# Skip stocks that are too illiquid to trade reliably
MIN_AVG_TURNOVER_CR = _env("MIN_AVG_TURNOVER_CR", 1.0, float)  # 20-day avg, Rs crore
MIN_PRICE = _env("MIN_PRICE", 10.0, float)

# Blend of machine-learning probability and rule-based score
ML_WEIGHT = _env("ML_WEIGHT", 0.65, float)

# How many rows in the Top Bullish / Top Bearish sheets
TOP_N = _env("TOP_N", 30, int)

MARKET_INDEX = "^NSEI"   # NIFTY 50
SENSEX_INDEX = "^BSESN"  # BSE SENSEX
VIX_INDEX = "^INDIAVIX"  # India VIX

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
