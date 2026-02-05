#!/usr/bin/env python3
"""
bullish.py (Binance)

Enter a date (YYYY-MM-DD). Script checks the *previous day* BTCUSDT 1D candle
from Binance and labels the market as:

BULLISH = close > open
BEARISH = close < open
NEUTRAL = close == open

Requires:
  pip install requests
"""

import sys
from datetime import datetime, timedelta, timezone, date
import requests


BINANCE_BASE = "https://api.binance.com"
SYMBOL = "BTCUSDT"
INTERVAL = "1d"


def parse_date(s: str) -> date:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Use YYYY-MM-DD (e.g., 2026-01-30)")


def utc_day_bounds_ms(d: date) -> tuple[int, int]:
    """Return (start_ms, end_ms) for UTC day d."""
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def fetch_kline(symbol: str, interval: str, day: date) -> tuple[float, float, float, float]:
    """
    Fetch the daily kline for `day` (UTC) from Binance.
    Returns (open, high, low, close).
    """
    start_ms, end_ms = utc_day_bounds_ms(day)

    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1,
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if not data:
        raise RuntimeError(f"No kline returned for {symbol} on {day} (UTC).")

    k = data[0]
    # [ open_time, open, high, low, close, volume, close_time, ... ]
    o = float(k[1])
    h = float(k[2])
    l = float(k[3])
    c = float(k[4])
    return o, h, l, c


def mood_from_open_close(o: float, c: float) -> str:
    if c > o:
        return "BULLISH"
    if c < o:
        return "BEARISH"
    return "NEUTRAL"


def main() -> int:
    try:
        input_str = input("Enter a date (YYYY-MM-DD): ").strip()
        target = parse_date(input_str)

        prev_day = target - timedelta(days=1)
        o, h, l, c = fetch_kline(SYMBOL, INTERVAL, prev_day)

        mood = mood_from_open_close(o, c)
        pct = ((c - o) / o) * 100 if o else 0.0

        print("\nResult (UTC day candle)")
        print("----------------------")
        print(f"Input date:         {target.isoformat()}")
        print(f"Previous day:       {prev_day.isoformat()}")
        print(f"Proxy:              {SYMBOL} @ Binance ({INTERVAL})")
        print(f"Open:               {o:.2f}")
        print(f"High:               {h:.2f}")
        print(f"Low:                {l:.2f}")
        print(f"Close:              {c:.2f}")
        print(f"Change (O->C):      {pct:+.2f}%")
        print(f"Market mood:        {mood}")

        return 0

    except requests.HTTPError as e:
        print(f"\nERROR: HTTP error from Binance: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

