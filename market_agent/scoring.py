"""Rule-based scoring. Each rule is a widely-used trading signal; the stock
gets points for bullish readings and loses points for bearish ones. The
reasons are written into the Excel so every prediction is explainable."""
import numpy as np
import pandas as pd


def _v(row, key):
    val = row.get(key)
    try:
        return None if val is None or pd.isna(val) else float(val)
    except (TypeError, ValueError):
        return None


def score_row(row, horizon_days):
    tech, fund, reasons = 0.0, 0.0, []

    def add(points, why, bucket="tech"):
        nonlocal tech, fund
        if bucket == "tech":
            tech += points
        else:
            fund += points
        reasons.append(f"{'+' if points > 0 else ''}{points:g} {why}")

    c = _v(row, "close")
    # --- Trend ---
    d50, d200, s50_200 = _v(row, "dist_sma50"), _v(row, "dist_sma200"), _v(row, "sma50_vs_200")
    if d50 is not None:
        add(6 if d50 > 0 else -6, "price above 50-DMA" if d50 > 0 else "price below 50-DMA")
    if d200 is not None:
        add(5 if d200 > 0 else -5, "price above 200-DMA" if d200 > 0 else "price below 200-DMA")
    if s50_200 is not None:
        add(4 if s50_200 > 0 else -4, "50-DMA above 200-DMA (golden cross zone)" if s50_200 > 0
            else "50-DMA below 200-DMA (death cross zone)")
    st = _v(row, "supertrend_dir")
    if st is not None:
        add(6 if st > 0 else -6, "Supertrend BUY" if st > 0 else "Supertrend SELL")
    adx, di = _v(row, "adx_14"), _v(row, "di_diff")
    if adx is not None and di is not None and adx > 25:
        add(6 if di > 0 else -6, f"strong trend ADX {adx:.0f} ({'+DI' if di > 0 else '-DI'} leading)")

    # --- Momentum ---
    rsi = _v(row, "rsi_14")
    if rsi is not None:
        if rsi < 30:
            add(7, f"RSI {rsi:.0f} oversold (bounce likely)")
        elif rsi > 75:
            add(-7, f"RSI {rsi:.0f} overbought (pullback risk)")
        elif 50 <= rsi <= 68:
            add(4, f"RSI {rsi:.0f} in bullish zone")
        elif 32 <= rsi < 45:
            add(-3, f"RSI {rsi:.0f} weak")
    mh, mhc, mx = _v(row, "macd_hist_pct"), _v(row, "macd_hist_chg3"), _v(row, "macd_cross")
    if mx:
        add(8 * mx, "MACD bullish crossover" if mx > 0 else "MACD bearish crossover")
    elif mh is not None and mhc is not None:
        if mh > 0 and mhc > 0:
            add(5, "MACD histogram positive & rising")
        elif mh < 0 and mhc < 0:
            add(-5, "MACD histogram negative & falling")
    sk, sd = _v(row, "stoch_k"), _v(row, "stoch_d")
    if sk is not None and sd is not None:
        if sk < 20 and sk > sd:
            add(4, "Stochastic turning up from oversold")
        elif sk > 80 and sk < sd:
            add(-4, "Stochastic turning down from overbought")
    mfi = _v(row, "mfi_14")
    if mfi is not None:
        if mfi < 20:
            add(4, f"MFI {mfi:.0f} oversold")
        elif mfi > 80:
            add(-4, f"MFI {mfi:.0f} overbought")
    rs = _v(row, "rs_vs_nifty_20d")
    if rs is not None and abs(rs) > 0.03:
        add(4 if rs > 0 else -4, f"{'out' if rs > 0 else 'under'}performing NIFTY by {rs*100:+.1f}% (20d)")

    # --- Volatility / price action ---
    pb = _v(row, "bb_pctb")
    if pb is not None:
        if pb < 0:
            add(4, "closed below lower Bollinger band")
        elif pb > 1:
            add(-3, "closed above upper Bollinger band (stretched)")
    squeeze = _v(row, "bb_width_rank")
    if squeeze is not None and squeeze < 0.1 and _v(row, "breakout_20d"):
        add(8, "breakout from Bollinger squeeze")
    elif _v(row, "breakout_20d"):
        add(5, "20-day high breakout")
    if _v(row, "breakdown_20d"):
        add(-5, "20-day low breakdown")
    d52 = _v(row, "dist_52w_high")
    if d52 is not None and d52 > -0.03:
        add(4, "near 52-week high (strength)")

    # --- Volume / delivery ---
    vs, r1 = _v(row, "vol_surge"), _v(row, "ret_1d")
    if vs is not None and r1 is not None and vs > 1.8:
        add(6 if r1 > 0 else -6, f"volume {vs:.1f}x average on {'up' if r1 > 0 else 'down'} day")
    obv = _v(row, "obv_slope_10")
    if obv is not None and abs(obv) > 0.3:
        add(3 if obv > 0 else -3, "OBV rising (accumulation)" if obv > 0 else "OBV falling (distribution)")
    dp, dr = _v(row, "deliv_per"), _v(row, "deliv_ratio")
    if dp is not None and dr is not None and r1 is not None and dr > 1.3 and dp > 40:
        add(6 if r1 > 0 else -6, f"high delivery {dp:.0f}% ({dr:.1f}x avg) on {'up' if r1 > 0 else 'down'} day")

    # --- Fundamentals (smaller weight: they move prices slowly) ---
    roe, de = _v(row, "roe"), _v(row, "debt_to_equity")
    if roe is not None:
        if roe > 0.18:
            add(3, f"ROE {roe*100:.0f}% strong", "fund")
        elif roe < 0.05:
            add(-3, f"ROE {roe*100:.0f}% weak", "fund")
    if de is not None:
        if de < 50:
            add(2, "low debt (D/E < 0.5)", "fund")
        elif de > 200:
            add(-3, f"high debt (D/E {de/100:.1f})", "fund")
    eg, rg = _v(row, "earnings_growth"), _v(row, "revenue_growth")
    if eg is not None and abs(eg) > 0.1:
        add(3 if eg > 0 else -3, f"earnings growth {eg*100:+.0f}% YoY", "fund")
    if rg is not None and abs(rg) > 0.1:
        add(2 if rg > 0 else -2, f"revenue growth {rg*100:+.0f}% YoY", "fund")
    pm = _v(row, "profit_margin")
    if pm is not None and pm < 0:
        add(-3, "loss-making (negative margin)", "fund")
    pe = _v(row, "pe")
    if pe is not None and pe > 100:
        add(-2, f"very expensive (P/E {pe:.0f})", "fund")
    tgt = _v(row, "analyst_target")
    if tgt and c:
        up = tgt / c - 1
        if up > 0.15:
            add(3, f"analyst target {up*100:+.0f}% away", "fund")
        elif up < -0.05:
            add(-3, f"trading above analyst target ({up*100:+.0f}%)", "fund")
    rating = _v(row, "analyst_rating")
    if rating is not None:
        if rating <= 2.0:
            add(2, "analysts rate BUY", "fund")
        elif rating >= 3.5:
            add(-2, "analysts rate SELL", "fund")

    # --- Market mood ---
    n20, vix = _v(row, "nifty_dist_sma50"), _v(row, "vix")
    if n20 is not None:
        add(3 if n20 > 0 else -3, "NIFTY above its 50-DMA" if n20 > 0 else "NIFTY below its 50-DMA")
    if vix is not None and vix > 20:
        add(-3, f"India VIX high ({vix:.1f}) - nervous market")

    total = float(np.clip(tech + fund, -100, 100))
    reasons.sort(key=lambda r: -abs(float(r.split()[0])))
    return tech, fund, total, "; ".join(reasons[:8])


def earnings_flag(ts, as_of, horizon_days):
    try:
        if ts is None or pd.isna(ts):
            return ""
        d = pd.Timestamp(int(ts), unit="s").normalize()
        days = (d - pd.Timestamp(as_of)).days
        if 0 <= days <= horizon_days * 1.5:
            return f"Results on {d.date()} - expect big move"
        return ""
    except (TypeError, ValueError, OverflowError):
        return ""


def apply_scores(df, horizon, ml_weight, as_of):
    out = df.apply(lambda r: score_row(r, horizon), axis=1, result_type="expand")
    df["technical_score"], df["fundamental_score"], df["rule_score"], df["reasons"] = \
        out[0], out[1], out[2], out[3]
    rule_prob = 0.5 + df["rule_score"] / 200.0
    df["prob_up"] = ml_weight * df["ml_prob_up"] + (1 - ml_weight) * rule_prob

    def label(p):
        if p >= 0.62:
            return "STRONG UP"
        if p >= 0.55:
            return "UP"
        if p <= 0.38:
            return "STRONG DOWN"
        if p <= 0.45:
            return "DOWN"
        return "NEUTRAL"

    df["prediction"] = df["prob_up"].map(label)
    df["confidence"] = (df["prob_up"] - 0.5).abs() * 2
    move = df["atr"] * np.sqrt(horizon)
    df["expected_ret"] = df["ml_expected_ret"]
    df["target_price"] = df["close"] * (1 + df["expected_ret"])
    df["likely_low"] = df["close"] - move
    df["likely_high"] = df["close"] + move
    df["stop_loss"] = np.where(df["prob_up"] >= 0.5, df["close"] - 1.5 * df["atr"], df["close"] + 1.5 * df["atr"])
    df["event_alert"] = [earnings_flag(ts, as_of, horizon) for ts in df.get("earnings_ts", [None] * len(df))]
    return df
