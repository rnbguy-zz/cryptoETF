import os
import sys
from datetime import datetime, timedelta
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Binance API URL for Klines data
klines_url = 'https://api.binance.com/api/v3/klines'

# Directory to save outputs
output_directory = './outputs/'
os.makedirs(output_directory, exist_ok=True)

# Set the folder path and the time range
folder_path = "./outputs"
days_to_check = 90  # Number of past days to check

# Get today's date
today = datetime.now()

# Generate a list of all dates in the past `days_to_check` days, excluding today
expected_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days_to_check + 1)]

# Get the list of files in the folder
try:
    files = os.listdir(folder_path)
except FileNotFoundError:
    print(f"Error: Folder '{folder_path}' does not exist.")
    sys.exit(1)

# Extract dates from filenames
existing_dates = []
for file in files:
    if file.startswith("output_") and file.endswith(".xlsx"):
        try:
            date_part = file.split("_")[1].split(".")[0]
            # Ensure the date format is valid
            datetime.strptime(date_part, "%Y-%m-%d")
            existing_dates.append(date_part)
        except ValueError:
            print(f"Error: Invalid date format in file '{file}'.")
            sys.exit(1)

# Find missing dates
missing_dates = [date for date in expected_dates if date not in existing_dates]

# Set date_input to the furthest missing date
if missing_dates:
    date_input = min(missing_dates)
else:
    date_input = (today - timedelta(days=days_to_check)).strftime("%Y-%m-%d")

print(f"Auto-detected date_input as {date_input}")

input_date = datetime.strptime(date_input, "%Y-%m-%d")
days_ago = (datetime.now() - input_date).days + 3
daysback_start = days_ago

daysback_end = 0

# Function to fetch historical data for a specific coin
def fetch_historical_data(symbol):
    try:
        end_time = int((datetime.utcnow() - timedelta(days=daysback_end)).timestamp() * 1000)
        start_time = int((datetime.utcnow() - timedelta(days=daysback_start)).timestamp() * 1000)

        params = {'symbol': symbol, 'interval': '1d', 'startTime': start_time, 'endTime': end_time}
        response = requests.get(klines_url, params=params)
        data = response.json()

        if not data or 'code' in data:
            print(f"Error fetching data for {symbol}: {data}")
            return None

        ohlc_data = {
            'Date': [datetime.utcfromtimestamp(item[0] / 1000).strftime('%Y-%m-%d') for item in data],
            'Open': [float(item[1]) for item in data],
            'Close': [float(item[4]) for item in data]
        }
        df = pd.DataFrame(ohlc_data)
        df['Date'] = pd.to_datetime(df['Date'])

        df['Slope'] = None
        for i in range(3, len(df)):
            open_price_3_days_prior = df.iloc[i - 3]['Open']
            close_price = df.iloc[i]['Close']
            df.at[i, 'Slope'] = ((close_price - open_price_3_days_prior) / open_price_3_days_prior) * 100

        df = df.dropna(subset=['Slope'])
        return df
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None

# Fetch all USDT trading pairs from Binance
def get_all_symbols():
    print("Fetching all symbols...")
    latest_prices_url = 'https://api.binance.com/api/v3/ticker/bookTicker'
    response = requests.get(latest_prices_url)
    data = response.json()
    return [item['symbol'] for item in data if item['symbol'].endswith("USDT")]

# Main function to download data and create daily Excel files
def create_daily_excels():
    symbols = get_all_symbols()
    print(f"Found {len(symbols)} symbols.")

    daily_data = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_historical_data, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if df is not None:
                    for _, row in df.iterrows():
                        target_date = row['Date'].strftime('%Y-%m-%d')
                        if target_date not in daily_data:
                            daily_data[target_date] = []
                        daily_data[target_date].append({'symbol': symbol, 'Slope': row['Slope']})
            except Exception as e:
                print(f"Error with symbol {symbol}: {e}")

    for target_date, data in daily_data.items():
        output_filename = os.path.join(output_directory, f'output_{target_date}.xlsx')
        daily_df = pd.DataFrame(data)
        daily_df.to_excel(output_filename, index=False)
        print(f"Saved data for {target_date} to {output_filename}")

    print("Finished creating daily Excel files.")

if __name__ == "__main__":
    create_daily_excels()

