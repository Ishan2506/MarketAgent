import numpy as np
import pandas as pd

from market_agent import main, tracker


def test_demo_run_creates_report():
    assert main.main(["--demo", "--force"]) == 0
    x = pd.ExcelFile(main.config.REPORTS_DIR / "demo" / "Latest_Market_Prediction.xlsx")
    assert {"Summary", "Top Bullish", "Top Bearish", "All Predictions", "Parameter Guide"} <= set(x.sheet_names)
    preds = pd.read_excel(x, "All Predictions")
    assert preds["Prob. Up"].between(0, 1).all()


def test_tracker_marks_correct_calls():
    idx = pd.bdate_range("2026-01-01", periods=20)
    closes = pd.DataFrame({"A.NS": np.linspace(100, 119, 20), "B.NS": np.linspace(100, 81, 20)}, index=idx)
    hist = pd.DataFrame({
        "date": [idx[0], idx[0], idx[15]], "ticker": ["A.NS", "B.NS", "A.NS"], "symbol": ["A", "B", "A"],
        "close": [100.0, 100.0, 115.0], "prob_up": [0.7, 0.7, 0.7], "prediction": ["UP", "UP", "UP"],
        "expected_ret": [0.02] * 3}).reindex(columns=tracker.COLS)
    out = tracker.evaluate(hist, closes, horizon=7)
    assert out.loc[0, "correct"] == 1 and out.loc[1, "correct"] == 0
    assert np.isnan(out.loc[2, "actual_ret"])  # horizon not reached yet
    assert abs(out.loc[0, "actual_ret"] - 0.07) < 1e-9
