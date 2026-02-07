# List of possible symbols: ['NEIROUSDT', 'ETHUSDT', 'IDUSDT', 'AXSUSDT', 'DOGEUSDT', 'GMTUSDT', 'IOUSDT', 'ALICEUSDT', 'CHZUSDT']
symbol = 'GNOUSDT'
start_date = '2024-11-16'  # Same name as output
end_date = '2024-11-28'

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from binance.client import Client
from datetime import datetime, timedelta, timezone
import os

import matplotlib
import os

# ? Detect X11 Forwarding and Set Backend
if "DISPLAY" in os.environ and os.environ["DISPLAY"]:
    print(f"? X11 forwarding detected. DISPLAY = {os.environ['DISPLAY']}")
    try:
        matplotlib.use('TkAgg')  # Use Tk-based X11 backend
    except ImportError:
        print("?? TkAgg not found. Falling back to Agg.")
        matplotlib.use('Agg')  # Fallback for non-GUI environments
else:
    print("?? X11 not detected. Using non-GUI backend (Agg).")
    matplotlib.use('Agg')  # Non-GUI fallback



# ? Initialize the Binance client with your API key and secret
api_key = 'your_api_key_here'
api_secret = 'your_api_secret_here'
client = Client(api_key, api_secret)


def fetch_5m_data(symbol, start_date_str, end_date_str):
    """Fetches 5-minute interval historical candlestick data from Binance."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    start_time = int(start_date.replace(hour=0, minute=0, second=0).timestamp() * 1000)
    end_time = int(end_date.replace(hour=23, minute=59, second=59).timestamp() * 1000)

    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_5MINUTE, start_time, end_time)

    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
    df = pd.DataFrame(klines, columns=columns)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
    df['close'] = df['close'].astype(float)

    return df


def plot_5m_data(symbol, start_date_str, end_date_str):
    """Plots the fetched 5-minute candlestick data and opens it in an X11 window."""
    df = fetch_5m_data(symbol, start_date_str, end_date_str)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    df.set_index('timestamp', inplace=True)
    all_minutes = pd.date_range(start=start_date, end=end_date + timedelta(days=1) - timedelta(minutes=5),
                                freq='5T', tz='UTC')
    df = df.reindex(all_minutes)

    min_price = df['close'].min()
    max_price = df['close'].max()
    min_time = df['close'].idxmin()
    max_time = df['close'].idxmax()

    opening_price = df['close'].iloc[0]
    perc_decrease = ((min_price - opening_price) / opening_price) * 100
    perc_increase = ((max_price - opening_price) / opening_price) * 100

    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['close'], label=f'{symbol} Closing Price')

    plt.scatter(min_time, min_price, color='red', zorder=5, label="Lowest Point")
    plt.text(min_time, min_price, f'{perc_decrease:.2f}%\n(Low)\n{min_price}', horizontalalignment='right', color='red')
    print('Symbol:', symbol)
    print("Target price:", (min_price * 1.1))
    print("Buy price on dip days:", (min_price * 0.94))

    plt.scatter(max_time, max_price, color='green', zorder=5, label="Highest Point")
    plt.text(max_time, max_price, f'{perc_increase:.2f}%\n(High)', horizontalalignment='left', color='green')

    plt.gca().xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%d-%b %H:%M', tz=timezone.utc))  
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=6))

    plt.title(f'{symbol} Price from {start_date_str} to {end_date_str} UTC')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()

    plt.tight_layout()

    def on_click(event):
        """Handles click events to calculate percentage changes between two points."""
        if on_click.points is None:
            on_click.points = []
        on_click.points.append((event.xdata, event.ydata))

        if len(on_click.points) == 2:
            price_1 = on_click.points[0][1]
            price_2 = on_click.points[1][1]
            percentage_change = ((price_2 - price_1) / price_1) * 100

            plt.plot([on_click.points[0][0], on_click.points[1][0]],
                     [on_click.points[0][1], on_click.points[1][1]], color='blue', linestyle='--', linewidth=2)

            plt.text((on_click.points[0][0] + on_click.points[1][0]) / 2,
                     (on_click.points[0][1] + on_click.points[1][1]) / 2,
                     f'{percentage_change:.2f}%', fontsize=12, color='blue')

            on_click.points = []
            plt.draw()

    on_click.points = []
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    plt.show()
    print('? Plot displayed successfully!')


# ? Run the function with X11 support
plot_5m_data(symbol, start_date, end_date)
