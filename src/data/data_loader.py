import os
import pandas as pd
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries

# Load environment variables from the .env file
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def load_data(filepath):
    """
    Load OHLC data from a CSV file and tokenize it.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Tokenized data as a Pandas DataFrame.
    """
    try:
        data = pd.read_csv(filepath)
        print("Data loaded from file.")
        return preprocess_data(data)  # Tokenize the loaded data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def fetch_ohlc_data(symbol, interval='60min', outputsize='full'):
    """
    Fetch OHLC data using Alpha Vantage API and tokenize it.
    Args:
        symbol (str): Stock ticker symbol.
        interval (str): Time interval (e.g., '1min', '5min', '15min', '60min').
        outputsize (str): 'compact' (latest 100 data points) or 'full' (full history).
    Returns:
        pd.DataFrame: Tokenized OHLC data as a Pandas DataFrame.
    """
    if not API_KEY:
        raise ValueError("API key not found. Please set ALPHA_VANTAGE_API_KEY in the .env file.")
    
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    try:
        data, _ = ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Rename columns for consistency
        print("Data fetched from Alpha Vantage.")
        return preprocess_data(data)  # Tokenize the fetched data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def preprocess_data(data):
    """
    Process OHLC data into tokenized format.
    Args:
        data (pd.DataFrame): OHLC data.
    Returns:
        pd.DataFrame: Processed data with relative body, upper wick, and lower wick.
    """
    # Ensure the necessary columns are present
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not set(required_columns).issubset(data.columns):
        raise ValueError(f"Data must contain columns: {required_columns}")

    # Convert columns to numeric, forcing errors to NaN
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with invalid data (e.g., non-numeric values)
    data = data.dropna(subset=required_columns)

    # Compute tokenized features
    data['Relative Body'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'])
    data['Upper Wick'] = (data['High'] - data[['Open', 'Close']].max(axis=1)) / (data['High'] - data['Low'])
    data['Lower Wick'] = (data[['Open', 'Close']].min(axis=1) - data['Low']) / (data['High'] - data['Low'])

    # Return only the tokenized columns
    return data[['Relative Body', 'Upper Wick', 'Lower Wick']]


# Example usage (for testing purposes only)
if __name__ == "__main__":
    # Example 1: Load and tokenize data from file
    filepath = 'data/SPY_5min_60d.csv'
    tokenized_from_file = load_data(filepath)
    if tokenized_from_file is not None:
        print("\nTokenized Data from CSV:")
        print(tokenized_from_file.head())
    
    # Example 2: Fetch and tokenize data from Alpha Vantage
    symbol = 'SPY'
    tokenized_from_api = fetch_ohlc_data(symbol, interval='5min')
    if tokenized_from_api is not None:
        print("\nTokenized Data from Alpha Vantage:")
        print(tokenized_from_api.head())
