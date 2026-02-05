symbol = 'CTKUSDT'
start_date = '2025-10-25'  # date you got buy now email.
from datetime import date
end_date = date.today().strftime("%Y-%m-%d")

import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime, timedelta, timezone

# Initialize the Binance client with your API key and secret
api_key = 'your_api_key_here'
api_secret = 'your_api_secret_here'
client = Client(api_key, api_secret)


def fetch_5m_data(symbol, start_date_str, end_date_str):
    """Fetch 5-minute interval historical candlestick data from Binance (UTC)."""
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(end_date_str,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    start_time = int(start_dt.replace(hour=0, minute=0, second=0).timestamp() * 1000)
    end_time   = int(end_dt.replace(  hour=23, minute=59, second=59).timestamp() * 1000)

    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_5MINUTE, start_time, end_time)

    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ]
    df = pd.DataFrame(klines, columns=columns)

    # Parse/convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def plot_5m_data(symbol, start_date_str, end_date_str):
    """Plot 5m data and print all results including target check."""
    df = fetch_5m_data(symbol, start_date_str, end_date_str)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    df.set_index('timestamp', inplace=True)
    all_minutes = pd.date_range(
        start=start_date,
        end=end_date + timedelta(days=1) - timedelta(minutes=5),
        freq='5T', tz='UTC'
    )
    df = df.reindex(all_minutes)

    # === Determine correct day open and target ===
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    day_slice = df.loc[df.index.date == start_date.date()]
    if not day_slice.empty:
        opening_price = float(day_slice['open'].iloc[0])
    else:
        opening_price = float(df['open'].iloc[0])

    target_price = opening_price * 1.07  # 7% target

    # === NEW SECTION: check if highest price since open exceeded target ===
    since_open = df.loc[df.index.date >= start_date.date()]
    if not since_open.empty:
        max_high = float(since_open['high'].max())
        if max_high >= target_price:
            cleared_pct = (max_high / target_price - 1) * 100
            print(f"Highest since open: {max_high:.6f} ✅ Target HIT (cleared by {cleared_pct:.2f}%).")
        else:
            needed_pct = (target_price / max_high - 1) * 100 if max_high > 0 else float('inf')
            print(f"Highest since open: {max_high:.6f} ⏳ Target NOT hit (need +{needed_pct:.2f}% more).")
    # === END NEW SECTION ===

    # === Live price and gain calculation ===
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
    except Exception as e:
        # Fallback: use last known close from dataframe if live call fails
        current_price = float(df['close'].dropna().iloc[-1])
        print(f"(Live price fetch failed: {e}. Using last known close instead.)")

    if current_price >= target_price:
        above_pct = (current_price / target_price - 1) * 100
        print(f"Symbol: {symbol}")
        print(f"Open price on {start_date_str}: {opening_price:.6f}")
        print(f"Target sell price (7%): {target_price:.6f}")
        print(f"Current price: {current_price:.6f}")
        print(f"Already ABOVE target by {above_pct:.2f}% (Δ = {current_price - target_price:.6f}).")
    else:
        gain_needed_pct = (target_price / current_price - 1) * 100
        print(f"Symbol: {symbol}")
        print(f"Open price on {start_date_str}: {opening_price:.6f}")
        print(f"Target sell price (7%): {target_price:.6f}")
        print(f"Current price: {current_price:.6f}")
        print(f"Gain needed to hit target: {gain_needed_pct:.2f}% (Δ = {target_price - current_price:.6f}).")

    # === Plotting ===
    min_price = df['close'].min()
    max_price = df['close'].max()
    min_time = df['close'].idxmin()
    max_time = df['close'].idxmax()

    perc_decrease = ((min_price - opening_price) / opening_price) * 100
    perc_increase = ((max_price - opening_price) / opening_price) * 100

    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['close'], label=f'{symbol} Closing Price')

    plt.scatter(min_time, min_price, color='red', zorder=5, label="Lowest Point")
    plt.text(min_time, min_price, f'{perc_decrease:.2f}%\n(Low)\n{min_price:.6f}',
             horizontalalignment='right', color='red')

    plt.scatter(max_time, max_price, color='green', zorder=5, label="Highest Point")
    plt.text(max_time, max_price, f'{perc_increase:.2f}%\n(High)', horizontalalignment='left', color='green')

    plt.gca().xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%d-%b %H:%M', tz=timezone.utc)
    )
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=6))

    plt.axhline(target_price, linestyle='--', linewidth=1, color='orange', label='Target (7%)')
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
                     [on_click.points[0][1], on_click.points[1][1]],
                     color='blue', linestyle='--', linewidth=2)

            plt.text((on_click.points[0][0] + on_click.points[1][0]) / 2,
                     (on_click.points[0][1] + on_click.points[1][1]) / 2,
                     f'{percentage_change:.2f}%', fontsize=12, color='blue')

            on_click.points = []
            plt.draw()

    on_click.points = []
    plt.show()


# Run
plot_5m_data(symbol, start_date, end_date)
