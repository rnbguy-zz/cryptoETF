import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

'''
How to use:
1. Wait for Buy day (green circle). Don't buy yet.
2. Wait for the next day where we exceed the green dot (yellow dot)
3. Run oversoldanalysis, with previous day=green circle date. current day=next day we exceed green (yellow circle)
4. Buy on yellow circle, any of the small drops, it may dip a tiny bit, try by smallest Drop in the list of small. 

Note:
- Doesn't ned to be 1k increase
'''

# Binance API URL for Klines data
klines_url = 'https://api.binance.com/api/v3/klines'

# AEDT timezone setup for printing dates (not for data handling)
aedt = pytz.timezone('Australia/Sydney')
uptotoday=0 #if you want data all the way to now litrally. Still testing 1.
daysago=1 # yesterday default.

# Function to calculate slope from open 2 days ago to close yesterday in UTC
def calculate_slope(df, yesterdays_date, waitperiod=(daysago+1)):  # Using UTC dates now
    try:
        # Extract the relevant comparison dates
        open_date = (yesterdays_date - timedelta(days=waitperiod)).date()  # 2 days ago in UTC
        yesterday_date_only = yesterdays_date.date()  # Yesterday in UTC

        # Convert the index to `date` objects
        df['IndexDate'] = df.index  # Simply assign the index to a new column
        df['IndexDate'] = pd.to_datetime(df['IndexDate']).dt.date  # Convert to date objects

        # Check if both dates exist in the 'IndexDate' column
        if open_date not in df['IndexDate'].values or yesterday_date_only not in df['IndexDate'].values:
            print(f"Missing data for {open_date} or {yesterday_date_only}")
            print(df)
            return None

        # Get the open price 2 days ago in UTC
        open_price = df.loc[df['IndexDate'] == open_date, 'Open'].values[0]
        # Get the close price of yesterday in UTC
        close_price = df.loc[df['IndexDate'] == yesterday_date_only, 'Close'].values[0]

        # Calculate the slope (percentage increase or decrease)
        slope = ((close_price - open_price) / open_price) * 100
        return slope
    except Exception as e:
        print(f"Error calculating slope: {e}")
        return None


def get_historical_data_today(symbol, lookback_days=(3+daysago)):
    # Fetch daily candles up to yesterday
    end_time = int(datetime.utcnow().timestamp() * 1000)  # Current UTC time in milliseconds
    start_time = int((datetime.utcnow() - timedelta(days=lookback_days)).timestamp() * 1000)  # Lookback days ago in milliseconds
    breakpoint()
    params = {
        'symbol': symbol,
        'interval': '1d',  # Daily candles for past days
        'startTime': start_time,
        'endTime': end_time  # Include data up to current time
    }

    response = requests.get(klines_url, params=params)
    data = response.json()

    if not data or 'code' in data:
        print(f"Error fetching daily data for {symbol}: {data}")
        return None

    # Parse the daily data into a DataFrame
    ohlc_data = {
        'Date': [datetime.utcfromtimestamp(item[0] / 1000).date() for item in data],  # UTC date
        'Open': [float(item[1]) for item in data],  # Open price
        'Close': [float(item[4]) for item in data]  # Close price
    }

    df_daily = pd.DataFrame(ohlc_data)
    df_daily.set_index('Date', inplace=True)

    # Fetch intraday data for today to get the ongoing candle
    start_time_today = int(datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    end_time_today = int(datetime.utcnow().timestamp() * 1000)  # Current time

    params_intraday = {
        'symbol': symbol,
        'interval': '1m',  # 1-minute candles for today
        'startTime': start_time_today,
        'endTime': end_time_today
    }

    response_today = requests.get(klines_url, params=params_intraday)
    data_today = response_today.json()

    if not data_today or 'code' in data_today:
        print(f"Error fetching intraday data for {symbol}: {data_today}")
        return df_daily  # Return just daily data if intraday data is not available

    # Today's "Open" is the first intraday candle's open, "Close" is the last candle's close
    today_open = float(data_today[0][1])
    today_close = float(data_today[-1][4])

    # Create a row for today with combined open and close prices
    today_date = datetime.utcnow().date()
    df_today = pd.DataFrame({'Open': [today_open], 'Close': [today_close]}, index=[today_date])

    # Combine daily and today's intraday data
    df_combined = pd.concat([df_daily, df_today])

    return df_combined

# Function to get historical data from Binance's Klines endpoint
def get_historical_data(symbol, interval='1d', lookback_days=(3+daysago)):  # Fetch 3 days of data in UTC
    end_time = int(datetime.utcnow().timestamp() * 1000)  # Current UTC time
    start_time = int((datetime.utcnow() - timedelta(days=lookback_days)).timestamp() * 1000)  # 3 days ago

    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time
    }

    response = requests.get(klines_url, params=params)
    data = response.json()

    if not data or 'code' in data:
        print(f"Error fetching data for {symbol}: {data}")
        return None

    # Convert the response into a DataFrame
    ohlc_data = {
        'Date': [datetime.utcfromtimestamp(item[0] / 1000).date() for item in data],  # Keep UTC dates
        'Open': [float(item[1]) for item in data],
        'Close': [float(item[4]) for item in data]
    }

    df = pd.DataFrame(ohlc_data)
    df.set_index('Date', inplace=True)

    return df


# Function to process each symbol
def process_symbol(symbol):
    try:
        # Get historical data for the symbol in UTC
        if uptotoday==0:
            df = get_historical_data(symbol, lookback_days=(daysago+4))
        else:
            df = get_historical_data_today(symbol, lookback_days=wait_period + daysago)

        if df is not None:
            # Calculate the slope for the given symbol
            slope = calculate_slope(df, yesterdays_date, wait_period)

            if slope is not None:
                print(f"Slope for {symbol}: {slope:.5f}%")
                return {'symbol': symbol, 'Slope': slope}

    except Exception as e:
        print(f"Error processing symbol {symbol}: {e}")

    return None


# Get the latest prices from Binance for USDT pairs
latest_prices_url = 'https://api.binance.com/api/v3/ticker/bookTicker'
latest_prices_response = requests.get(latest_prices_url)
latest_prices_data = latest_prices_response.json()

# Extract symbols ending with "USDT" (but override for now with 'BTCUSDT' for testing)
cryptos = [item['symbol'] for item in latest_prices_data if item['symbol'].endswith("USDT")]

# Define yesterday's date and the wait period for calculations in UTC
yesterdays_date = datetime.utcnow() - timedelta(days=daysago)  # Use UTC time for consistent data handling
wait_period = 3  # Set wait period to 2 days ago
columns = ['symbol', 'Slope']

# Placeholder for result data
results = []

# Process symbols concurrently using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(process_symbol, symbol): symbol for symbol in cryptos}

    for future in as_completed(futures):
        result = future.result()
        if result:
            results.append(result)

# Save results to an Excel file in AEDT (for display purposes)
if uptotoday==0:
    output_date_aedt = datetime.now(aedt) - timedelta(days=daysago)  # AEDT for file naming
else:
    output_date_aedt = datetime.now(aedt)

if results:
    output_df = pd.DataFrame(results)
    output_filename = f'./outputs/output_{output_date_aedt.strftime("%Y-%m-%d")}.xlsx'
    output_df.to_excel(output_filename, index=False)
    print(f"Results saved to {output_filename}")

    slope_avg = output_df['Slope'].mean()
    print(f"Average Slope: {slope_avg}")
else:
    print("No results to save.")

print('Im finished, next step: run timeslope_plotter.py.')