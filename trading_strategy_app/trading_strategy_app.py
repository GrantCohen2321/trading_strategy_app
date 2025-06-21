import matplotlib
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
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv

# Initialize environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Get S&P 500 Tickers
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = []
    for row in table.find_all('tr')[1:]:
        symbol = row.find_all('td')[0].text.strip()
        if '.' not in symbol:
            tickers.append(symbol)
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


import requests# Async stock data fetcher
async def fetch_data_async(session, ticker: str, start: str, end: str):
    """Fetch price data asynchronously using yfinance in a thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: yf.download(ticker, start=start, end=end, progress=False),

# Fetch all NASDAQ tickers via Polygon

def get_nasdaq_tickers() -> list:
    """Return a list of active NASDAQ tickers using the Polygon API."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        logging.error("POLYGON_API_KEY not found in environment variables.")
        return []

    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "exchange": "XNAS",
        "active": "true",
        "limit": 1000,
        "apiKey": api_key,
    }
    tickers = []
    while True:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            logging.error("Polygon API error: %s", resp.text)
            break
        data = resp.json()
        tickers.extend([item["ticker"] for item in data.get("results", [])])
        next_url = data.get("next_url")
        if not next_url:
            break
        # Extract cursor parameter for next page
        cursor = next_url.split("cursor=")[-1]
        params["cursor"] = cursor
    return tickers
    )

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
        
        
        # Qullamaggie Breakout / High RVOL Setup

def backtest_qullamaggie(df: pd.DataFrame):
    """Return True/False if the latest bar meets Qullamaggie-style breakout criteria.

    Criteria (simplified):
    1. Close within 2% of 52-week high
    2. Relative Volume (today vol ÷ 50-day avg) >= 2
    3. Gap-up >= 4% over yesterday close
    4. Close above 10-EMA and 50-SMA
    """
    if df.shape[0] < 252:
        return False  # Not enough data for 52-week high

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # 52-week high proximity
    high_52w = df['High'].rolling(window=252).max().iloc[-1]
    cond1 = latest['Close'] >= 0.98 * high_52w

    # Relative volume
    avg_vol_50 = df['Volume'].rolling(window=50).mean().iloc[-1]
    rvol = latest['Volume'] / avg_vol_50 if avg_vol_50 else 0
    cond2 = rvol >= 2

    # Gap-up percentage
    gap_pct = (latest['Open'] - prev['Close']) / prev['Close']
    cond3 = gap_pct >= 0.04

    # Moving averages
    ema10 = df['Close'].ewm(span=10).mean().iloc[-1]
    sma50 = df['Close'].rolling(window=50).mean().iloc[-1]
    cond4 = latest['Close'] > ema10 and latest['Close'] > sma50

    matched = cond1 and cond2 and cond3 and cond4

    if matched:
        log_result("Qullamaggie", latest['Ticker'], (latest['Close'] - prev['Close']) / prev['Close'], latest.name.date(), "Qullamaggie")
        save_chart(df, latest['Ticker'], latest.name.date(), "Qullamaggie")
    else:
        log_debug(latest['Ticker'], "No Qullamaggie setup")

    return matched

log_debug(data['Ticker'].iloc[0], "No Parabolic setup")
# Risk Management Helpers

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range (ATR) using Wilder's method."""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean().iloc[-1]
    return atr


def atr_stop(entry_price: float, atr: float, multiplier: float = 1.5) -> float:
    """Stop loss price based on ATR multiplier below entry."""
    return entry_price - multiplier * atr


def trailing_stop(current_high: float, atr: float, multiplier: float = 1.0) -> float:
    """Trailing stop that moves up as price makes new highs."""
    return current_high - multiplier * atr


def position_size(account_risk: float, entry_price: float, stop_price: float) -> int:
    """
    Position sizing based on fixed dollar risk.
    account_risk: dollars willing to risk on trade
    entry_price: trade entry price
    stop_price: stop loss price
    """
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share == 0:
        return 0
    shares = account_risk // risk_per_share
    return int(shares)


# 6. Email Notification
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
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
       backtest_qullamaggie(df)
# 7. Historical Backtesting

def historical_backtest(tickers, start_date: str, end_date: str, account_risk: float = 100):
    """Very simple walk‑forward backtest:
    1. Loop through each day in the date range
    2. Apply Qullamaggie scanner on each day's data
    3. Simulate entry at close, exit after 5 trading days or stop
    4. Record R multiple
    """
        # Auto-fetch NASDAQ tickers if requested
    if tickers is None or (isinstance(tickers, str) and tickers.upper() == "ALL"):
        tickers = get_nasdaq_tickers()

results = []
    for ticker in tickers:
        df = fetch_data(ticker, start_date, end_date)
        if df.empty or len(df) < 252:
            continue
        df = add_moving_averages(df)
        for i in range(252, len(df) - 5):
            window = df.iloc[: i + 1].copy()
            if backtest_qullamaggie(window):
                entry_idx = window.index[-1]
                entry_price = window['Close'].iloc[-1]
                atr = calculate_atr(window)
                stop_price = atr_stop(entry_price, atr)

                # Exit after 5 bars (placeholder exit logic)
                exit_idx = i + 5
                exit_price = df['Close'].iloc[exit_idx]
                gain = (exit_price - entry_price) / entry_price
                r_multiple = gain / ((entry_price - stop_price) / entry_price)

                results.append({
                    'Ticker': ticker,
                    'EntryDate': entry_idx.date(),
                    'ExitDate': df.index[exit_idx].date(),
                    'Return': gain,
                    'R': r_multiple,
                })

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv('backtest_results.csv', index=False)
        print(f"Saved {len(df_results)} trades to backtest_results.csv")
    else:
        print("No trades generated in backtest.")


# 8. Run full scan
def run_scan():
    tickers = get_sp500_tickers()
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
