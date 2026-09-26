"""Technical indicators (the same ones shown on Zerodha Kite / TradingView charts)
turned into numeric features for every stock and every day."""
import numpy as np
import pandas as pd


def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def _true_range(h, l, c):
    pc = c.shift()
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _adx(h, l, c, n=14):
    up, down = h.diff(), -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _true_range(h, l, c).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), plus_di, minus_di


def _supertrend(h, l, c, n=10, mult=3.0):
    atr = _true_range(h, l, c).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    hl2 = (h + l) / 2
    upper = (hl2 + mult * atr).to_numpy(copy=True)
    lower = (hl2 - mult * atr).to_numpy(copy=True)
    close = c.to_numpy()
    direction = np.ones(len(c))
    for i in range(1, len(c)):
        if np.isnan(upper[i - 1]):
            continue
        if close[i - 1] <= upper[i - 1]:
            upper[i] = min(upper[i], upper[i - 1])
        if close[i - 1] >= lower[i - 1]:
            lower[i] = max(lower[i], lower[i - 1])
        if close[i] > upper[i - 1]:
            direction[i] = 1
        elif close[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return pd.Series(direction, index=c.index)


def stock_features(df):
    """df: OHLCV for one stock. Returns a DataFrame of features indexed by date."""
    o, h, l, c, v = (df[k].astype(float) for k in ["Open", "High", "Low", "Close", "Volume"])
    v = v.replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    ret = c.pct_change()

    for n in (1, 5, 10, 20, 60, 120):
        f[f"ret_{n}d"] = c.pct_change(n)

    f["rsi_14"] = _rsi(c, 14)
    f["rsi_14_chg5"] = f["rsi_14"].diff(5)

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd_pct"] = macd / c * 100
    f["macd_hist_pct"] = (macd - signal) / c * 100
    f["macd_hist_chg3"] = f["macd_hist_pct"].diff(3)
    f["macd_cross"] = np.sign(macd - signal).diff().fillna(0) / 2  # +1 bullish cross, -1 bearish

    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n, min_periods=n).mean() - 1
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    f["sma50_vs_200"] = sma50 / sma200 - 1
    f["ema9_vs_21"] = c.ewm(span=9, adjust=False).mean() / c.ewm(span=21, adjust=False).mean() - 1

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd)
    f["bb_width"] = 4 * sd / mid
    f["bb_width_rank"] = f["bb_width"].rolling(120, min_periods=60).rank(pct=True)  # squeeze

    ll14, hh14 = l.rolling(14).min(), h.rolling(14).max()
    rng14 = (hh14 - ll14).replace(0, np.nan)
    f["stoch_k"] = 100 * (c - ll14) / rng14
    f["stoch_d"] = f["stoch_k"].rolling(3).mean()
    f["williams_r"] = -100 * (hh14 - c) / rng14

    tp = (h + l + c) / 3
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    f["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * mad)

    adx, pdi, mdi = _adx(h, l, c)
    f["adx_14"] = adx
    f["di_diff"] = pdi - mdi

    atr = _true_range(h, l, c).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["atr_pct"] = atr / c
    f["volatility_20"] = ret.rolling(20).std()

    avg_v20 = v.rolling(20, min_periods=10).mean()  # tolerate zero-volume data gaps
    f["vol_surge"] = v / avg_v20
    f["vol_ratio_5_20"] = v.rolling(5, min_periods=3).mean() / avg_v20
    f["turnover_cr"] = (c * v).rolling(20, min_periods=10).mean() / 1e7

    obv = (np.sign(c.diff()).fillna(0) * v.fillna(0)).cumsum()
    f["obv_slope_10"] = (obv - obv.shift(10)) / (v.rolling(10, min_periods=5).sum())

    raw_mf = tp * v
    pos = raw_mf.where(tp > tp.shift(), 0).rolling(14, min_periods=7).sum()
    neg = raw_mf.where(tp < tp.shift(), 0).rolling(14, min_periods=7).sum()
    f["mfi_14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))

    f["dist_52w_high"] = c / h.rolling(252, min_periods=120).max() - 1
    f["dist_52w_low"] = c / l.rolling(252, min_periods=120).min() - 1
    f["breakout_20d"] = (c >= h.shift().rolling(20).max()).astype(float)
    f["breakdown_20d"] = (c <= l.shift().rolling(20).min()).astype(float)

    f["supertrend_dir"] = _supertrend(h, l, c)
    f["gap_pct"] = o / c.shift() - 1
    f["close_in_range"] = (c - l) / (h - l).replace(0, np.nan)
    f["up_days_10"] = (ret > 0).rolling(10).mean()
    f["close"] = c
    f["atr"] = atr
    return f


def market_features(nifty, vix=None):
    c = nifty["Close"]
    m = pd.DataFrame(index=nifty.index)
    m["nifty_ret_5d"] = c.pct_change(5)
    m["nifty_ret_20d"] = c.pct_change(20)
    m["nifty_dist_sma50"] = c / c.rolling(50).mean() - 1
    m["nifty_rsi"] = _rsi(c)
    if vix is not None and len(vix):
        vc = vix["Close"].reindex(m.index).ffill()
        m["vix"] = vc
        m["vix_chg_5d"] = vc.pct_change(5)
    else:
        m["vix"] = np.nan
        m["vix_chg_5d"] = np.nan
    return m


# Features the ML model learns from (everything that exists historically,
# so the model never "peeks" at the future).
MODEL_FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
    "rsi_14", "rsi_14_chg5", "macd_pct", "macd_hist_pct", "macd_hist_chg3", "macd_cross",
    "dist_sma20", "dist_sma50", "dist_sma200", "sma50_vs_200", "ema9_vs_21",
    "bb_pctb", "bb_width", "bb_width_rank", "stoch_k", "stoch_d", "williams_r", "cci_20",
    "adx_14", "di_diff", "atr_pct", "volatility_20", "vol_surge", "vol_ratio_5_20",
    "obv_slope_10", "mfi_14", "dist_52w_high", "dist_52w_low", "breakout_20d",
    "breakdown_20d", "supertrend_dir", "gap_pct", "close_in_range", "up_days_10",
    "rs_vs_nifty_20d", "rs_vs_nifty_5d",
    "nifty_ret_5d", "nifty_ret_20d", "nifty_dist_sma50", "nifty_rsi", "vix", "vix_chg_5d",
]


def build_panel(prices, nifty, vix, horizon):
    """Stack features for all stocks into one long table (date, ticker, features, target)."""
    mkt = market_features(nifty, vix)
    frames = []
    for t, df in prices.items():
        if len(df) < 130:
            continue
        f = stock_features(df)
        f = f.join(mkt, how="left")
        f[mkt.columns] = f[mkt.columns].ffill()
        f["rs_vs_nifty_20d"] = f["ret_20d"] - f["nifty_ret_20d"]
        f["rs_vs_nifty_5d"] = f["ret_5d"] - f["nifty_ret_5d"]
        fwd = f["close"].shift(-horizon) / f["close"] - 1
        f["fwd_ret"] = fwd
        f["fwd_max_ret"] = f["close"][::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1) / f["close"] - 1
        f["ticker"] = t
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    return panel.reset_index()
