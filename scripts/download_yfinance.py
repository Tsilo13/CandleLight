import yfinance as yf
import pandas as pd
import os

def fetch_and_save_yfinance_data(symbol, interval="5m", period="60d", output_file="data/intraday_5min.csv"):
    """
    Fetches intraday data from Yahoo Finance and saves it to a CSV file.
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL').
        interval (str): Data interval ('1m', '5m', etc.).
        period (str): Data period ('1d', '5d', '60d', '1y', etc.).
        output_file (str): File path to save the data.
    """
    print(f"Fetching {interval} data for {symbol} over the past {period}...")
    try:
        # Fetch data
        data = yf.download(tickers=symbol, interval=interval, period=period)
        if data.empty:
            print(f"No data found for {symbol} at {interval} interval.")
            return
        
        # Reset index for clean CSV format
        data.reset_index(inplace=True)
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error fetching data: {e}")

# Example Usage
if __name__ == "__main__":
    SYMBOL = "SPY"  # Replace with your desired stock symbol
    OUTPUT_FILE = "data/SPY_5min_60d.csv"
    
    fetch_and_save_yfinance_data(symbol=SYMBOL, interval="5m", period="1mo", output_file=OUTPUT_FILE)
