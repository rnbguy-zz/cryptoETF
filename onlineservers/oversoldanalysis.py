import time
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import pandas_ta as ta
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

#modify this to go back in time.
today_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
#today_date = '2025-10-24'

# Binance API Credentials
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'

recipient_email = "mina.moussa@hotmail.com"
sender_email = "minamoussa903@gmail.com"  # Replace with your email
sender_password = "thjj eryc yzym dylb"  # Replace with your email password

#Yellow circle days: ['2024-10-12', '2024-10-27', '2024-11-16', '2024-12-11', '2024-12-22', '2025-01-02', '2025-01-11', '2025-01-22', '2025-01-30']

# Define final and check dates
final_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
with open("yellowcircle.txt", "r") as file:
    check_dates = [line.strip() for line in file.readlines()]

# Get all USDT trading pairs
symbols = []
if not symbols:
    latest_prices_url = 'https://api.binance.com/api/v3/ticker/bookTicker'
    latest_prices_response = requests.get(latest_prices_url)
    latest_prices_data = latest_prices_response.json()
    symbols = [item['symbol'] for item in latest_prices_data if item['symbol'].endswith("USDT")]

# Initialize Binance API client
client = Client(API_KEY, API_SECRET)


def fetch_historical_data(symbol, interval, start_date, end_date):
    """Fetch historical price data for a given symbol within a time range."""
    try:
        klines = client.get_historical_klines(symbol, interval, start_date, end_date)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()  # Return empty dataframe on failure


def calculate_rsi(df):
    """Compute RSI from historical price data."""
    try:
        df['RSI'] = ta.rsi(df['close'], length=14)
        return df[['RSI']].dropna()
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return None  # Return None if RSI can't be calculated


def download_all_data(symbols, check_dates, final_date):
    """Download historical data for all symbols covering the entire period."""
    historical_data = {}

    # Determine the full range of required historical data
    earliest_date = min(datetime.strptime(date, "%Y-%m-%d") for date in check_dates)
    latest_date = datetime.strptime(final_date, "%Y-%m-%d")

    # Include buffer for RSI calculation
    start_date = earliest_date - timedelta(days=7)
    end_date = latest_date

    # Fetch historical data for all symbols
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_historical_data, symbol, Client.KLINE_INTERVAL_1HOUR,
                                   start_date.strftime("%Y-%m-%d %H:%M:%S"),
                                   end_date.strftime("%Y-%m-%d %H:%M:%S")): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                historical_data[symbol] = future.result()
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

    return historical_data


def compute_rsi_for_dates(historical_data, symbols, check_dates, final_date):
    """Compute RSI values for all check dates and the final date."""
    rsi_values = {symbol: {} for symbol in symbols}

    for symbol in symbols:
        df = historical_data.get(symbol, pd.DataFrame())
        if df.empty:
            continue

        # Compute RSI once and extract values for required dates
        df_rsi = calculate_rsi(df)
        if df_rsi is None or df_rsi.empty:
            continue

        for date in check_dates + [final_date]:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            nearest_rsi = df_rsi[df_rsi.index <= date_obj].iloc[-1] if not df_rsi[
                df_rsi.index <= date_obj].empty else None
            if nearest_rsi is not None:
                rsi_values[symbol][date] = nearest_rsi['RSI']

    return rsi_values


def detect_rsi_drops(rsi_values, check_dates, final_date):
    """Identify significant RSI drops between check dates and the final date."""
    sudden_drops = {}

    for symbol, rsis in rsi_values.items():
        rsi_dates = sorted(rsis.keys())
        for i in range(1, len(rsi_dates)):
            previous_rsi = rsis[rsi_dates[i - 1]]
            current_rsi = rsis[rsi_dates[i]]

            if previous_rsi - current_rsi > 10:
                sudden_drops[symbol] = {
                    'Previous Date': rsi_dates[i - 1],
                    'Current Date': rsi_dates[i],
                    'Previous RSI': previous_rsi,
                    'Current RSI': current_rsi,
                    'Drop': previous_rsi - current_rsi
                }

    return sudden_drops


def kmeans_cluster_rsi_drops(rsi_drops, n_clusters):
    """Apply K-means clustering to categorize RSI drops."""
    drop_values = np.array([[data['Drop']] for data in rsi_drops.values()])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(drop_values)

    clustered_drops = {i: [] for i in range(n_clusters)}
    for i, (symbol, data) in enumerate(rsi_drops.items()):
        clustered_drops[clusters[i]].append({symbol: data})

    return clustered_drops


def label_clusters(clustered_drops):
    """Label clusters based on RSI drop magnitude."""
    cluster_averages = {i: np.mean([entry[list(entry.keys())[0]]['Drop'] for entry in cluster_data])
                        for i, cluster_data in clustered_drops.items()}
    sorted_clusters = sorted(cluster_averages.items(), key=lambda x: x[1])
    labels = ['VerySmall', 'Small', 'Medium', 'Large']

    return {labels[i]: clustered_drops[cluster_id] for i, (cluster_id, _) in enumerate(sorted_clusters)}


# Execution workflow
historical_data = download_all_data(symbols, check_dates, final_date)
rsi_values = compute_rsi_for_dates(historical_data, symbols, check_dates, final_date)
rsi_drops = detect_rsi_drops(rsi_values, check_dates, final_date)

# Clustering
if rsi_drops:
    num_clusters = min(len(rsi_drops), 4)
    clustered_drops = kmeans_cluster_rsi_drops(rsi_drops, num_clusters)
    labeled_clusters = label_clusters(clustered_drops)

    # Save output
    filename = f"./oversold_analysis/{final_date}_oversold.txt"
    with open(filename, "w") as file:
        for label, cluster_data in labeled_clusters.items():
            if label == 'Small':
                file.write(f"\n{label} Drop Cluster:\n")
                cluster_list = [{**entry[list(entry.keys())[0]], 'Symbol': list(entry.keys())[0]} for entry in
                                cluster_data]
                cluster_df = pd.DataFrame(cluster_list)
                if not cluster_df.empty:
                    file.write(cluster_df.sort_values(by='Drop', ascending=True).to_string(index=False))

    print(cluster_df.sort_values(by='Drop', ascending=True).to_string(index=False))
    print("Labeled Clusters saved successfully.")

else:
    print("No sudden RSI drops detected.")

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta

# Get today's date in the required format
filename_pattern = f"{final_date}_oversold.txt"  # Adjust the filename pattern as needed

# Function to send an email with the file as an attachment
def send_email_with_attachment(file_path, recipient_email, sender_email, sender_password):
    try:
        subject = "Oversold symbols to buy."
        body = "Please find the attached file containing the oversold symbols.\n\n"

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach the file
        if os.path.exists(file_path):
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={os.path.basename(file_path)}'
            )
            msg.attach(part)
        else:
            # Add a note in the email body if the file is missing
            msg.attach(MIMEText("No file found to attach.", 'plain'))

        # Send the email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print("Email sent successfully.")

    except Exception as e:
        print(f"Error sending email: {e}")


# Set the directory and file path
directory = r"./oversold_analysis/"  # Adjust the directory if needed
file_path = os.path.join(directory, filename_pattern)
send_email_with_attachment(file_path, recipient_email, sender_email, sender_password)
