"""Keeps a history of every prediction and checks it once the horizon has
passed, so you can see how accurate the agent really is."""
import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)
HISTORY = config.DATA_DIR / "prediction_history.csv"
COLS = ["date", "ticker", "symbol", "close", "prob_up", "prediction", "expected_ret",
        "actual_ret", "max_ret_in_window", "correct"]


def load():
    if HISTORY.exists():
        h = pd.read_csv(HISTORY, parse_dates=["date"])
        return h.reindex(columns=COLS)
    return pd.DataFrame(columns=COLS)


def evaluate(history, closes, horizon):
    """closes: DataFrame (date x ticker) of close prices."""
    if history.empty:
        return history
    idx = closes.index
    todo = history["actual_ret"].isna()
    for i in history.index[todo]:
        d, t = history.at[i, "date"], history.at[i, "ticker"]
        if t not in closes.columns or d not in idx:
            continue
        pos = idx.get_loc(d)
        if pos + horizon >= len(idx):
            continue
        path = closes[t].iloc[pos + 1: pos + horizon + 1]
        base = history.at[i, "close"]
        if path.isna().all() or not base:
            continue
        history.at[i, "actual_ret"] = path.dropna().iloc[-1] / base - 1
        history.at[i, "max_ret_in_window"] = path.max() / base - 1
    pred = history["prediction"].astype(str)
    up, down = pred.str.contains("UP"), pred.str.contains("DOWN")
    done = history["actual_ret"].notna()
    history["correct"] = np.where(done & up, history["actual_ret"] > 0,
                                  np.where(done & down, history["actual_ret"] < 0, np.nan))
    history.loc[~(done & (up | down)), "correct"] = np.nan
    return history


def append(history, today_df, as_of):
    new = today_df.assign(date=pd.Timestamp(as_of))[
        ["date", "ticker", "symbol", "close", "prob_up", "prediction", "expected_ret"]]
    history = history[history["date"] != pd.Timestamp(as_of)]
    return pd.concat([history, new.reindex(columns=COLS)], ignore_index=True)


def save(history):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = history.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    for c in ["close", "prob_up", "expected_ret", "actual_ret", "max_ret_in_window"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    out.to_csv(HISTORY, index=False)


def summary(history):
    done = history[history["correct"].notna()].copy()
    if done.empty:
        return pd.DataFrame(), pd.DataFrame()
    done["correct"] = done["correct"].astype(bool)
    by_signal = done.groupby("prediction").agg(
        predictions=("correct", "size"), hit_rate=("correct", "mean"),
        avg_actual_return=("actual_ret", "mean")).reset_index()
    by_date = done.groupby("date").agg(
        predictions=("correct", "size"), hit_rate=("correct", "mean")).reset_index() \
        .sort_values("date", ascending=False)
    return by_signal, by_date
