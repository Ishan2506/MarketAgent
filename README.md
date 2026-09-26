# MarketAgent: NSE/BSE 1-2 week price direction predictor

A GitHub Actions agent that runs **every Tuesday at 8:00 AM IST** (using Monday's closing prices). It checks
every company in the NIFTY 500 (or all NSE stocks) and predicts whether each
share price will be **higher or lower in 7 trading days (~10 calendar days)**.
It then commits a formatted **Excel report** to this repository.

> ⚠️ This is a statistical estimate for learning and research. It is **not** investment
> advice. Short-term prices are noisy, and even a good model is wrong 40-45% of the time.
> Always use a stop loss.

## Where to find the Excel

| File | What it is |
|---|---|
| `reports/Latest_Market_Prediction.xlsx` | Always the newest report |
| `reports/YYYY/MM/Market_Prediction_YYYY-MM-DD.xlsx` | Weekly archive (dated by the last trading day) |
| `data/prediction_history.csv` | Every past prediction and how it turned out |
| `data/fundamentals.csv` | Cached fundamentals (refreshed weekly) |

Each run also attaches the Excel to the workflow run (Actions tab → run → *Artifacts*).

### Sheets in the report
- **Summary**: market mood (NIFTY, SENSEX, India VIX), signal counts, and the model's accuracy on recent unseen data.
- **Top Bullish / Top Bearish**: the 30 strongest UP and DOWN calls.
- **All Predictions**: every stock with its prediction, probability, expected move, target, likely range, stop loss, event alerts and the main reasons.
- **Technicals / Fundamentals**: the raw values of every parameter.
- **Model Importance**: which parameters mattered most to the model.
- **Track Record / Past Calls Checked**: past predictions checked against real prices once the 7 days have passed.
- **Parameter Guide**: what each parameter means, and which readings are bullish or bearish.

## Parameters analysed (all from trusted sources)

| Group | Parameters | Source |
|---|---|---|
| Trend | 20/50/200 DMA, golden/death cross, EMA 9/21, Supertrend (10,3), ADX & ±DI | Yahoo Finance prices |
| Momentum | RSI 14, MACD (12,26,9), Stochastic, Williams %R, CCI, 1d-120d returns, relative strength vs NIFTY | calculated |
| Volatility | Bollinger Bands %B / width / squeeze, ATR, 20-day volatility | calculated |
| Price action | 20-day breakout/breakdown, distance from 52-week high/low, gaps, close position in range | calculated |
| Volume | Volume surge, OBV, MFI, **delivery %** | Yahoo Finance, **NSE bhavcopy** |
| Fundamentals | P/E, forward P/E, P/B, ROE, ROA, debt/equity, earnings & revenue growth, margins, dividend yield, beta, promoter & institutional holding, analyst target & rating | Yahoo Finance |
| Events | Upcoming quarterly results date (flagged, because results can move the price sharply) | Yahoo Finance |
| Market | NIFTY 50 trend & RSI, India VIX level and change, SENSEX | Yahoo Finance |

## How a prediction is made

1. **Machine-learning model** (Gradient Boosting). Every night it retrains on about
   3 years of history for all stocks and learns which combinations of the
   technical and market parameters were followed by a rise over the next 7
   trading days. It reports its own accuracy on the most recent months of data,
   which it did not see during training.
2. **Rule-based score** (-100 to +100). It uses classic signals that traders
   read on Zerodha Kite or TradingView (for example MACD crossover, RSI
   oversold, Supertrend BUY, a breakout on high delivery), plus fundamentals.
   Each rule adds or subtracts points, and the top reasons are written into the
   Excel.
3. **Final probability** = 65% ML + 35% rules. The label is set from it:

| Prob. Up | Prediction |
|---|---|
| ≥ 62% | STRONG UP |
| ≥ 55% | UP |
| 45-55% | NEUTRAL |
| ≤ 45% | DOWN |
| ≤ 38% | STRONG DOWN |

Illiquid stocks (average turnover below ₹1 crore) and penny stocks (price below ₹10) are skipped.

## Setup (one time)

1. Push this code to GitHub. Scheduled workflows only run from the
   **default branch**, so merge this into your default branch (or make this
   branch the default).
2. Go to **Settings → Actions → General → Workflow permissions** and select
   **Read and write permissions**. The agent needs this to commit the Excel.
3. Optional: under **Settings → Secrets and variables → Actions → Variables**, add
   `UNIVERSE` = `NIFTY50`, `NIFTY100`, `NIFTY200`, `NIFTY500` (default) or `NSE_ALL`.
4. To run it immediately: **Actions → Weekly Market Prediction Agent → Run workflow**.

Extra stocks can be added in `config/watchlist.txt` (NSE) and
`config/bse_symbols.txt` (BSE-only companies, by scrip code).

Other settings (environment variables, see `market_agent/config.py`):
`HORIZON_DAYS` (default 7), `HISTORY_YEARS` (3), `ML_WEIGHT` (0.65),
`MIN_AVG_TURNOVER_CR` (1.0), `FETCH_FUNDAMENTALS` (1).

## Run locally

```bash
pip install -r requirements.txt
python -m market_agent.main --universe NIFTY100   # real data
python -m market_agent.main --demo                # offline test on synthetic data
python -m pytest tests
```
