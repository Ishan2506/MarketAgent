"""Emails the latest Excel report. Settings come from environment variables
(set as GitHub Actions secrets so they never appear in the public repo):

  MAIL_TO        recipient(s), comma separated
  SMTP_USERNAME  sender login, e.g. yourname@gmail.com
  SMTP_PASSWORD  for Gmail: a 16-character App Password, not your normal password
  SMTP_SERVER    default smtp.gmail.com
  SMTP_PORT      default 465 (SSL); 587 uses STARTTLS
  MAIL_FROM      default SMTP_USERNAME
  REPORT_URL     optional link to the report on GitHub
"""
import html
import logging
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger("market_agent.notify")
XLSX_MIME = ("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _fmt(v, pct=False):
    if isinstance(v, (int, float)) and not pd.isna(v):
        return f"{v * 100:.1f}%" if pct else f"{v:,.2f}"
    return "" if pd.isna(v) else html.escape(str(v))


def _table(df, cols, pct_cols=()):
    df = df[[c for c in cols if c in df.columns]]
    head = "".join(f"<th style='background:#1F3864;color:#fff;padding:4px 8px;text-align:left'>{c}</th>"
                   for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = "".join(f"<td style='padding:4px 8px;border-bottom:1px solid #ddd'>{_fmt(r[c], c in pct_cols)}</td>"
                        for c in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table style='border-collapse:collapse;font-size:13px'><tr>{head}</tr>{''.join(rows)}</table>"


def build_message(report_path, sender, recipients, report_url=None):
    x = pd.ExcelFile(report_path)
    summary = pd.read_excel(x, "Summary").dropna(subset=["Item"])
    info = dict(zip(summary["Item"], summary["Value"]))
    as_of = info.get("Data as of (market close)", "")
    bull = pd.read_excel(x, "Top Bullish").head(10)
    bear = pd.read_excel(x, "Top Bearish").head(10)

    keys = ["Stocks analysed", "NIFTY 50 close", "NIFTY 50 trend", "India VIX", "Stocks predicted UP vs DOWN",
            "Accuracy (all)", "Prediction horizon"]
    mood = "".join(f"<tr><td style='padding:3px 10px 3px 0'><b>{k}</b></td>"
                   f"<td>{_fmt(info[k], k == 'Accuracy (all)')}</td></tr>" for k in keys if k in info)
    cols = ["Symbol", "Company", "Close (Rs)", "Prediction", "Prob. Up", "Target (Rs)", "Stop Loss (Rs)"]
    pct = {"Prob. Up"}
    link = f"<p><a href='{html.escape(report_url)}'>Open the report on GitHub</a></p>" if report_url else ""
    body = f"""<html><body style='font-family:Arial,sans-serif'>
<h2 style='color:#1F3864'>Weekly NSE/BSE Prediction - data as of {html.escape(str(as_of))}</h2>
<table style='font-size:14px'>{mood}</table>
<h3 style='color:#00873C'>Top 10 likely to go UP</h3>{_table(bull, cols, pct) if 'Symbol' in bull else '<p>None</p>'}
<h3 style='color:#C00000'>Top 10 likely to go DOWN</h3>{_table(bear, cols, pct) if 'Symbol' in bear else '<p>None</p>'}
<p>The full Excel with all stocks, technicals, fundamentals and track record is attached.</p>{link}
<p style='color:#777;font-size:12px'>Statistical estimate for research only - not investment advice.
Always use a stop loss.</p></body></html>"""

    msg = EmailMessage()
    msg["Subject"] = f"Market Prediction Report - {as_of}"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(f"Weekly market prediction report (data as of {as_of}) is attached."
                    + (f"\n{report_url}" if report_url else ""))
    msg.add_alternative(body, subtype="html")
    msg.add_attachment(Path(report_path).read_bytes(), maintype=XLSX_MIME[0], subtype=XLSX_MIME[1],
                       filename=f"Market_Prediction_{as_of}.xlsx")
    return msg


def send(report_path=None):
    report_path = Path(report_path or config.REPORTS_DIR / "Latest_Market_Prediction.xlsx")
    recipients = [r.strip() for r in os.environ.get("MAIL_TO", "").split(",") if r.strip()]
    user, password = os.environ.get("SMTP_USERNAME"), os.environ.get("SMTP_PASSWORD")
    if not (recipients and user and password):
        log.warning("Email skipped: set the MAIL_TO, SMTP_USERNAME and SMTP_PASSWORD secrets")
        return 2
    if not report_path.exists():
        log.error("Report not found: %s", report_path)
        return 1
    server = os.environ.get("SMTP_SERVER") or "smtp.gmail.com"
    port = int(os.environ.get("SMTP_PORT") or 465)
    sender = os.environ.get("MAIL_FROM") or user
    msg = build_message(report_path, sender, recipients, os.environ.get("REPORT_URL"))
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(server, port, context=ctx, timeout=60) as s:
            s.login(user, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(server, port, timeout=60) as s:
            s.starttls(context=ctx)
            s.login(user, password)
            s.send_message(msg)
    log.info("Report emailed to %d recipient(s)", len(recipients))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(send(sys.argv[1] if len(sys.argv) > 1 else None))
