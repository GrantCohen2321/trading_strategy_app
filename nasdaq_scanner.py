"""
nasdaq_scanner.py
-----------------
Daily scanner for breakout, parabolic, and episodic‑pivot setups across
all NASDAQ tickers. Sends results via email.

Author: <Your Name>
Date: 2025-06-21
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient
import smtplib
from email.mime.text import MIMEText

# ----------------------------------------------------------------------
# Load secrets from .env (DO NOT COMMIT REAL KEYS)
# ----------------------------------------------------------------------
load_dotenv()  # looks for .env in current working directory

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT")

if not all([POLYGON_API_KEY, SMTP_USER, SMTP_PASS, ALERT_RECIPIENT]):
    raise EnvironmentError("Missing one or more required environment variables.")

client = RESTClient(POLYGON_API_KEY)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def get_nasdaq_tickers() -> List[str]:
    """Return a list of NASDAQ common stock tickers."""
    tickers = []
    for resp in client.list_tickers(market="stocks", exchange="NASDAQ"):
        if resp.ticker.isalpha() and resp.type == "CS":
            tickers.append(resp.ticker)
    return tickers


def fetch_daily_bars(ticker: str, lookback: int = 120) -> pd.DataFrame:
    """Fetch the last `lookback` daily bars for a ticker."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback * 2)  # buffer for weekends/holidays
    bars = client.get_aggs(ticker, 1, "day", start_date, end_date, limit=lookback)
    df = pd.DataFrame([b.__dict__ for b in bars])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("date")


# ----------------------------------------------------------------------
# Pattern detectors
# ----------------------------------------------------------------------

def is_breakout(df: pd.DataFrame, window: int = 52) -> bool:
    """True if today's high > rolling max high of prior `window` days."""
    if len(df) < window + 1:
        return False
    prev_max = df["high"].iloc[-window - 1 : -1].max()
    return df["high"].iloc[-1] > prev_max


def is_parabolic(df: pd.DataFrame, atr_mult: float = 3.0, ma_period: int = 20) -> bool:
    """True if price > MA + atr_mult * ATR (simplified)."""
    if len(df) < ma_period + 1:
        return False
    ma = df["close"].rolling(ma_period).mean().iloc[-1]
    atr = (df["high"] - df["low"]).rolling(ma_period).mean().iloc[-1]
    return df["close"].iloc[-1] > ma + atr_mult * atr


def is_episodic_pivot(df: pd.DataFrame, volume_mult: float = 3.0) -> bool:
    """
    True if today's volume > volume_mult * avg volume of last 50 days
    AND today's close > prior day's high.
    """
    if len(df) < 51:
        return False
    avg_vol = df["volume"].iloc[-51:-1].mean()
    return (
        df["volume"].iloc[-1] > volume_mult * avg_vol
        and df["close"].iloc[-1] > df["high"].iloc[-2]
    )


# ----------------------------------------------------------------------
# Scanner core
# ----------------------------------------------------------------------

def scan_all() -> Dict[str, List[str]]:
    tickers = get_nasdaq_tickers()
    results = {"breakout": [], "parabolic": [], "episodic_pivot": []}

    for tkr in tickers:
        try:
            df = fetch_daily_bars(tkr)
            if df.empty:
                continue
            if is_breakout(df):
                results["breakout"].append(tkr)
            if is_parabolic(df):
                results["parabolic"].append(tkr)
            if is_episodic_pivot(df):
                results["episodic_pivot"].append(tkr)
        except Exception as e:
            print(f"Error scanning {tkr}: {e}")

    return results


def format_email(res: Dict[str, List[str]]) -> str:
    html = "<h2>Daily NASDAQ Scan Results</h2>"
    for key, lst in res.items():
        html += f"<h3>{key.title()} ({len(lst)})</h3>"
        html += ", ".join(lst) if lst else "None"
    return html


def send_email(body: str):
    msg = MIMEText(body, "html")
    msg["Subject"] = f"NASDAQ Scan {datetime.utcnow().date()}"
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_RECIPIENT

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    scan_results = scan_all()
    email_body = format_email(scan_results)
    send_email(email_body)
    print("Scan complete and email sent.")


if __name__ == "__main__":
    main()
