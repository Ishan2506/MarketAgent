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


def test_zero_volume_day_and_missing_last_candle_keep_stock():
    from market_agent.indicators import build_panel
    from market_agent.model import train_and_predict
    uni, prices, nifty, vix, _ = main._demo_prices(n=40)
    t0, t1 = list(prices)[:2]
    prices[t0].iloc[-5, prices[t0].columns.get_loc("Volume")] = 0   # data gap
    prices[t1] = prices[t1].iloc[:-1]                              # no candle on the last day
    panel = build_panel(prices, nifty, vix, horizon=7)
    today, _, _ = train_and_predict(panel, 7, nifty.index.max())
    assert set(prices) == set(today["ticker"])
    assert today.set_index("ticker")["turnover_cr"].notna().all()


def test_email_contains_summary_and_attachment(monkeypatch):
    from unittest import mock

    from market_agent import notify
    report = main.config.ROOT / "reports" / "Latest_Market_Prediction.xlsx"
    monkeypatch.setenv("MAIL_TO", "me@example.com")
    monkeypatch.setenv("SMTP_USERNAME", "bot@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    with mock.patch("smtplib.SMTP_SSL") as smtp:
        assert notify.send(report) == 0
    server = smtp.return_value.__enter__.return_value
    server.login.assert_called_once_with("bot@example.com", "secret")
    msg = server.send_message.call_args[0][0]
    assert msg["To"] == "me@example.com" and msg["Subject"].startswith("Market Prediction Report")
    parts = list(msg.iter_attachments())
    assert parts and parts[0].get_filename().endswith(".xlsx")
    html_body = msg.get_body(("html",)).get_content()
    assert "Top 10 likely to go UP" in html_body


def test_email_skipped_without_secrets(monkeypatch):
    from market_agent import notify
    for k in ("MAIL_TO", "SMTP_USERNAME", "SMTP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    assert notify.send() == 2
