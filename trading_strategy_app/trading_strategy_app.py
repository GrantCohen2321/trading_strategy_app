matplotlib.use('Agg')
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import concurrent.futures
from nasdaq_scanner import is_episodic_pivot

# # 1. Get NASDAQ Tickers via Polygon
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

def get_nasdaq_tickers() -> list:
    """Fetch all active NASDAQ tickers (~5,000) using Polygon’s REST API."""
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not set in environment variables")

    tickers = []
    url = (
        "https://api.polygon.io/v3/reference/tickers?"
        "market=stocks&exchange=XNAS&active=true&limit=1000&apiKey=" + POLYGON_API_KEY
    )

    while url:
        resp = requests.get(url, timeout=30).json()
        tickers.extend([item["ticker"] for item in resp.get("results", [])])
        url = resp.get("next_url")
        if url:
            # The next_url from Polygon doesn’t include the API key
            url += f"&apiKey={POLYGON_API_KEY}"
    return tickers

# 2. Configuration
LOOKBACK_PERIOD = {'1m': 21, '3m': 63, '6m': 126}
MOVING_AVERAGES = [10, 20, 50]
end = datetime.today()
start = end - timedelta(days=365)
CSV_FILE = "backtest_results.csv"
FULL_LOG_FILE = "scan_log.csv"
PERFORMANCE_FILE = "performance_log.csv"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

EMAIL_SENDER = "GCohenRE@GrantCohenRealty.com"
EMAIL_RECEIVER = "GCohenRE@GrantCohenRealty.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_PASSWORD = "exlgrgoohbqiapvh"

# 3. Stock Data Functions
def fetch_data(ticker):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        data['Ticker'] = ticker
        return data
    except:
        return pd.DataFrame()

def calculate_returns(data):
    for label, days in LOOKBACK_PERIOD.items():
        data[f'{label}_return'] = data['Close'].pct_change(periods=days)
    return data

def add_moving_averages(data):
    for ma in MOVING_AVERAGES:
        data[f'SMA_{ma}'] = data['Close'].rolling(window=ma).mean()
    return data

# 4. Logs
results_log = []
debug_log = []

def log_result(strategy, ticker, ret, entry_date=None):
    results_log.append({"Strategy": strategy, "Ticker": ticker, "Return": round(ret * 100, 2), "EntryDate": entry_date})

def log_debug(ticker, note):
    debug_log.append({"Ticker": ticker, "Note": note})

def save_chart(data, ticker, entry_date, strategy):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close')
    for ma in MOVING_AVERAGES:
        plt.plot(data[f'SMA_{ma}'], label=f'SMA {ma}')
    if entry_date in data.index:
        plt.axvline(entry_date, color='g', linestyle='--', label='Entry')
    plt.title(f'{ticker} - {strategy} Setup')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/{ticker}_{strategy}.png")
    plt.close()

# 5. Strategy Backtesters
def backtest_ep(data):
    matched = False
    for i in range(1, len(data) - 10):
        prev = data.iloc[i - 1]
        today = data.iloc[i]
        gap = ((today['Open'] - prev['Close']) / prev['Close']).item()
        avg_volume = data['Volume'].iloc[i - 20:i].mean().item()
        today_volume = today['Volume'].item()
        if gap > 0.05 and today_volume > avg_volume:
            matched = True
            entry = today['Open']
            stop = prev['Low'].item()
            target = entry + 2 * (entry - stop)
            entry_date = today.name
            for j in range(i + 1, min(i + 6, len(data))):
                low = data['Low'].iloc[j].item()
                high = data['High'].iloc[j].item()
                if low <= stop:
                    log_result("EP", data['Ticker'].iloc[0], (stop - entry) / entry, entry_date)
                    save_chart(data, data['Ticker'].iloc[0], entry_date, "EP")
                    return
                elif high >= target:
                    log_result("EP", data['Ticker'].iloc[0], (target - entry) / entry, entry_date)
                    save_chart(data, data['Ticker'].iloc[0], entry_date, "EP")
                    return
    if not matched:
        log_debug(data['Ticker'].iloc[0], "No EP setup")

def backtest_breakouts(data):
    matched = False
    for i in range(60, len(data) - 10):
        recent_high = data['Close'].iloc[i - 60:i].max()
        entry = data['Close'].iloc[i]
        if entry >= recent_high * 0.98:
            matched = True
            stop = data['Low'].iloc[i - 5:i].min()
            target = entry + 2 * (entry - stop)
            entry_date = data.index[i]
            for j in range(i + 1, min(i + 6, len(data))):
                low = data['Low'].iloc[j].item()
                high = data['High'].iloc[j].item()
                if low <= stop:
                    log_result("Breakout", data['Ticker'].iloc[0], (stop - entry) / entry, entry_date)
                    save_chart(data, data['Ticker'].iloc[0], entry_date, "Breakout")
                    return
                elif high >= target:
                    log_result("Breakout", data['Ticker'].iloc[0], (target - entry) / entry, entry_date)
                    save_chart(data, data['Ticker'].iloc[0], entry_date, "Breakout")
                    return
    if not matched:
        log_debug(data['Ticker'].iloc[0], "No Breakout setup")

def backtest_parabolic(data):
    matched = False
    for i in range(5, len(data) - 10):
        recent = data['Close'].iloc[i - 5:i]
        gain = float((recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0])
        entry = data['Open'].iloc[i]
        entry_date = data.index[i]
        if gain > 0.5:
            matched = True
            stop = data['High'].iloc[i - 1]
            target = entry - 2 * (stop - entry)
        elif gain < -0.3:
            matched = True
            stop = data['Low'].iloc[i - 1]
            target = entry + 2 * (entry - stop)
        else:
            continue
        for j in range(i + 1, min(i + 6, len(data))):
            high = data['High'].iloc[j].item()
            low = data['Low'].iloc[j].item()
            if gain > 0.5 and high >= stop:
                log_result("Parabolic", data['Ticker'].iloc[0], (stop - entry) / entry, entry_date)
                save_chart(data, data['Ticker'].iloc[0], entry_date, "Parabolic")
                return
            elif gain > 0.5 and low <= target:
                log_result("Parabolic", data['Ticker'].iloc[0], (target - entry) / entry, entry_date)
                save_chart(data, data['Ticker'].iloc[0], entry_date, "Parabolic")
                return
            elif gain < -0.3 and low <= stop:
                log_result("Parabolic", data['Ticker'].iloc[0], (stop - entry) / entry, entry_date)
                save_chart(data, data['Ticker'].iloc[0], entry_date, "Parabolic")
                return
            elif gain < -0.3 and high >= target:
                log_result("Parabolic", data['Ticker'].iloc[0], (target - entry) / entry, entry_date)
                save_chart(data, data['Ticker'].iloc[0], entry_date, "Parabolic")
                return
    if not matched:
        log_debug(data['Ticker'].iloc[0], "No Parabolic setup")

# 6. Email Notification
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        required_vars = [SMTP_SERVER, SMTP_PORT, EMAIL_SENDER, EMAIL_RECEIVER, SMTP_PASSWORD]
    if any(v is None or v == "" for v in required_vars):
        print("Email skipped: SMTP environment variables are not fully set.")
        return

        import smtplib
    
            with smtpllib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_SENDER, SMTP_PASSWORD)
                server.send_message(msg)
                print("Email sent!")
    except Exception as e:
        print(f"Email failed: {e}")

# 7. Worker
def process_ticker(ticker):
    df = fetch_data(ticker)
    if df.empty:
        log_debug(ticker, "No data fetched")
        return
    df = calculate_returns(df)
    df = add_moving_averages(df)
    backtest_ep(df)
    backtest_breakouts(df)
    backtest_parabolic(df)

# 8. Run full scan
def run_scan():
    tickers = get_nasdaq_tickers()
    print(f"Scanning {len(tickers)} tickers...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_ticker, tickers)

    df_results = pd.DataFrame(results_log)
    df_debug = pd.DataFrame(debug_log)

    if not df_debug.empty:
        df_debug.to_csv(FULL_LOG_FILE, index=False)

    if not df_results.empty:
        df_results['Return'] = pd.to_numeric(df_results['Return'], errors='coerce')
        df_results.dropna(subset=['Return'], inplace=True)
        df_results.to_csv(CSV_FILE, index=False)
        summary = df_results.groupby("Strategy").agg({"Return": ['count', 'mean', 'min', 'max']}).round(2)
        print("\nBacktest Summary:\n", summary)
        send_email("Daily Backtest Summary", summary.to_string())

        if os.path.exists(PERFORMANCE_FILE):
            df_perf = pd.read_csv(PERFORMANCE_FILE)
        else:
            df_perf = pd.DataFrame()
        summary['Date'] = pd.to_datetime(datetime.today()).date()
        df_new = summary.reset_index()
        df_perf = pd.concat([df_perf, df_new], ignore_index=True)
        df_perf.to_csv(PERFORMANCE_FILE, index=False)
    else:
        print("No results to report.")
        send_email("Backtest Scan Completed", "No valid setups were found.\n\nSee scan_log.csv for full ticker notes.")

# 9. Main
if __name__ == '__main__':
    run_scan()
