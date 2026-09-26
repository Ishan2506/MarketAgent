"""Machine-learning model: learns from ~3 years of history which indicator
combinations were followed by a price rise over the next HORIZON days."""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score

from .indicators import MODEL_FEATURES

log = logging.getLogger(__name__)


def _classifier():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=42)


def _regressor():
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=300, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, random_state=42)


def _sample_every(df, step):
    """Using every day creates heavily overlapping targets; sampling every
    few days trains faster with almost no information loss."""
    days = np.sort(df["date"].unique())
    keep = set(days[::step])
    return df[df["date"].isin(keep)]


def train_and_predict(panel, horizon, predict_date):
    labeled = panel[panel["fwd_ret"].notna() & (panel["date"] < predict_date)].copy()
    labeled = labeled.dropna(subset=["ret_60d", "rsi_14"])
    labeled["y"] = (labeled["fwd_ret"] > 0).astype(int)
    train_rows = _sample_every(labeled, 2)

    # ---- Walk-forward validation: train on older data, test on the most
    # recent ~20% of dates (with a gap so targets don't overlap).
    days = np.sort(train_rows["date"].unique())
    cut = days[int(len(days) * 0.8)]
    gap_end = days[min(len(days) - 1, int(len(days) * 0.8) + horizon)]
    tr = train_rows[train_rows["date"] < cut]
    te = train_rows[train_rows["date"] > gap_end]
    metrics = {}
    importance = pd.DataFrame(columns=["feature", "importance"])
    if len(tr) > 1000 and len(te) > 200 and te["y"].nunique() == 2:
        clf = _classifier().fit(tr[MODEL_FEATURES], tr["y"])
        p = clf.predict_proba(te[MODEL_FEATURES])[:, 1]
        hi, lo = p >= 0.6, p <= 0.4
        metrics = {
            "Validation period": f"{pd.Timestamp(te['date'].min()).date()} to {pd.Timestamp(te['date'].max()).date()}",
            "Validation rows": len(te),
            "Accuracy (all)": accuracy_score(te["y"], p >= 0.5),
            "ROC-AUC (0.5 = coin flip)": f"{roc_auc_score(te['y'], p):.3f}",
            "Base rate (stocks that rose)": te["y"].mean(),
            "Hit rate when prob >= 60% (UP calls)": te["y"][hi].mean() if hi.any() else np.nan,
            "Number of UP calls >= 60%": int(hi.sum()),
            "Hit rate when prob <= 40% (DOWN calls)": 1 - te["y"][lo].mean() if lo.any() else np.nan,
            "Number of DOWN calls <= 40%": int(lo.sum()),
            "Avg return of top-decile picks": te["fwd_ret"][p >= np.quantile(p, 0.9)].mean(),
            "Avg return of bottom-decile picks": te["fwd_ret"][p <= np.quantile(p, 0.1)].mean(),
        }
        sample = te.sample(min(len(te), 15000), random_state=0)
        pi = permutation_importance(clf, sample[MODEL_FEATURES], sample["y"], scoring="roc_auc",
                                    n_repeats=3, random_state=0, n_jobs=-1)
        importance = pd.DataFrame({"feature": MODEL_FEATURES, "importance": pi.importances_mean}) \
            .sort_values("importance", ascending=False)
        log.info("Validation: acc=%.3f auc=%.3f", metrics["Accuracy (all)"], float(metrics["ROC-AUC (0.5 = coin flip)"]))

    # ---- Final model on all history
    clf = _classifier().fit(train_rows[MODEL_FEATURES], train_rows["y"])
    reg = _regressor().fit(train_rows[MODEL_FEATURES], train_rows["fwd_ret"].clip(-0.3, 0.3))

    today = panel[panel["date"] == predict_date].copy()
    today["ml_prob_up"] = clf.predict_proba(today[MODEL_FEATURES])[:, 1]
    today["ml_expected_ret"] = reg.predict(today[MODEL_FEATURES])
    return today, metrics, importance
