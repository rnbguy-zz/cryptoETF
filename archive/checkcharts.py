#run this script
#paste list here: https://chatgpt.com/c/68a0135f-b880-8329-bf3f-c6611a72bb61
# on mobile paste into entry strategy for 15% move chat the 1 chart from binance
# update aws config script to monitor this
# run chart draw to check it's dipped 10% and find the sell price. Write down sell price in phone notes because you'll forget.

#!/usr/bin/env python3
import os
import re
import glob
from datetime import datetime, timedelta, timezone
from dateutil.parser import isoparse
import ccxt
import math
import time
from typing import List, Tuple, Set, Optional

# ==== CONFIG ====
THRESHOLD  = 7.0           # percent gain threshold (e.g., 7.0 for +7%)
# You can force a file by setting env var OVERSOLD_FILE=/full/path/to/2025-08-15_oversold.txt
# =================

FILENAME_RE = re.compile(r'(?P<date>\d{4}-\d{2}-\d{2})_oversold\.txt$')

LINE_RE = re.compile(
    r"""
    ^\s*
    (?P<prev>\d{4}-\d{2}-\d{2})       # Previous Date
    \s+
    (?P<curr>\d{4}-\d{2}-\d{2})       # Current Date
    \s+
    (?P<prev_rsi>[-+]?\d+(?:\.\d+)?)  # Previous RSI
    \s+
    (?P<curr_rsi>[-+]?\d+(?:\.\d+)?)  # Current RSI
    \s+
    (?P<drop>[-+]?\d+(?:\.\d+)?)      # Drop
    \s+
    (?P<symbol>[A-Z0-9]+USDT)         # Symbol (ends with USDT, may have digits)
    \s*$
    """,
    re.VERBOSE
)

def get_current_price(exch, pair):
    """Return the current spot price for the pair using ticker data, with a safe fallback."""
    t = exch.fetch_ticker(pair)
    for key in ("last", "close", "bid", "ask"):
        v = t.get(key)
        if v is not None and v > 0:
            return float(v)
    # Fallback: last 1m candle close
    rows = fetch_ohlcv_with_retries(exch, pair, "1m", limit=1)
    if rows:
        return float(rows[-1][4])  # close
    raise RuntimeError(f"No current price found for {pair}")


def find_input_file() -> Tuple[str, str]:
    """
    Returns (filepath, date_in_filename).
    Priority:
      1) OVERSOLD_FILE env var
      2) Latest by date among *_oversold.txt in CWD
    """
    forced = os.environ.get("OVERSOLD_FILE")
    if forced:
        base = os.path.basename(forced)
        m = FILENAME_RE.search(base)
        if not m:
            raise RuntimeError(f"OVERSOLD_FILE does not match pattern YYYY-MM-DD_oversold.txt: {forced}")
        if not os.path.isfile(forced):
            raise RuntimeError(f"OVERSOLD_FILE not found: {forced}")
        return forced, m.group("date")

    candidates = []
    for path in glob.glob("oversoldfilestocheck/*_oversold.txt"):
        m = FILENAME_RE.search(os.path.basename(path))
        if m:
            try:
                d = datetime.strptime(m.group("date"), "%Y-%m-%d").date()
                candidates.append((d, path))
            except ValueError:
                pass

    if not candidates:
        raise RuntimeError("No *_oversold.txt files found in current directory.")

    # pick latest date
    candidates.sort(key=lambda x: x[0])
    date_obj, filepath = candidates[-1]
    return filepath, date_obj.strftime("%Y-%m-%d")

def parse_symbols_from_file(filepath: str, target_date: str) -> List[str]:
    """
    Read file and return symbols whose Current Date equals target_date.
    """
    symbols: Set[str] = set()
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            curr_date = m.group("curr")
            if curr_date == target_date:
                sym = m.group("symbol").strip().upper()
                symbols.add(sym)
    return sorted(symbols)

def normalize(sym: str) -> str:
    """Convert 'BTCUSDT' -> 'BTC/USDT' (ccxt format)."""
    s = sym.strip().upper()
    if "/" not in s and s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    return s

def denormalize(pair: str) -> str:
    """Convert 'BTC/USDT' -> 'BTCUSDT'."""
    return pair.replace("/", "")

def day_bounds_utc(day_str: str) -> Tuple[int, int]:
    """Return UTC start/end ms for the calendar date (e.g., '2025-08-10')."""
    dt = isoparse(day_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    end   = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)

def fetch_ohlcv_with_retries(exch, symbol, timeframe, since=None, limit=1000, max_tries=3):
    for i in range(max_tries):
        try:
            return exch.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except ccxt.NetworkError:
            if i == max_tries - 1:
                raise
            time.sleep(0.8 * (2 ** i))
        except Exception:
            raise

def fetch_ohlcv_from_date_to_now(exch, symbol, start_ms, timeframe="1d", limit=1000):
    """
    Paginate OHLCV from start_ms -> now for the given timeframe.
    Returns a list of rows [ts, o, h, l, c, v].
    """
    all_rows = []
    since = start_ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    while True:
        batch = fetch_ohlcv_with_retries(exch, symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        # Deduplicate overlap
        if all_rows and batch[0][0] <= all_rows[-1][0]:
            batch = [row for row in batch if row[0] > all_rows[-1][0]]
        all_rows.extend(batch)
        # Stop if we've gone far enough (daily candles aligned to 00:00 UTC on Binance)
        if len(batch) < limit or batch[-1][0] >= now_ms - 24*60*60*1000:
            break
        since = batch[-1][0] + 1
    return all_rows

def find_open_on_date(rows, start_ms, end_ms) -> Optional[float]:
    """From daily rows, find the candle covering [start_ms, end_ms)."""
    for ts, o, h, l, c, v in rows:
        if ts == start_ms:
            return float(o)
    for ts, o, h, l, c, v in rows:
        if start_ms <= ts < end_ms:
            return float(o)
    return None

def max_high_from_date(rows, start_ms) -> Optional[float]:
    """Return the maximum high among rows with ts >= start_ms, or None if none."""
    highs = [float(h) for ts, o, h, l, c, v in rows if ts >= start_ms]
    return max(highs) if highs else None

def main():
    # 1) Locate and parse the oversold file
    filepath, date_from_filename = find_input_file()
    symbols_from_file = parse_symbols_from_file(filepath, date_from_filename)

    if not symbols_from_file:
        print(f"[INFO] No symbols in {os.path.basename(filepath)} with Current Date == {date_from_filename}. Nothing to do.")
        return

    print(f"[INFO] Using file: {os.path.basename(filepath)}")
    print(f"[INFO] DATE_UTC set to: {date_from_filename}")
    print(f"[INFO] {len(symbols_from_file)} symbols found for that date:")
    print("       " + ", ".join(symbols_from_file))

    DATE_UTC = date_from_filename  # auto-set from filename
    start_ms, end_ms = day_bounds_utc(DATE_UTC)

    # 2) Proceed with the exceed check using parsed symbols
    exch = ccxt.binance({"enableRateLimit": True})
    exch.load_markets()
    markets = exch.markets

    did_not_exceed = []  # (pair, open, max_high, gain%)
    exceeded       = []  # (pair, open, max_high, gain%)

    for raw_sym in symbols_from_file:
        pair = normalize(raw_sym)

        if pair not in markets:
            print(f"[WARN] Skipping (not found on Binance spot): {pair}")
            continue

        try:
            rows = fetch_ohlcv_from_date_to_now(exch, pair, start_ms, timeframe="1d", limit=1000)
            if not rows:
                print(f"[WARN] No candles for {pair} from {DATE_UTC} onward.")
                continue

            open_px = find_open_on_date(rows, start_ms, end_ms)
            if open_px is None or open_px <= 0 or math.isnan(open_px):
                print(f"[WARN] Could not find open for {pair} on {DATE_UTC} (maybe not listed/trading that day).")
                continue

            max_h = max_high_from_date(rows, start_ms)
            if max_h is None:
                print(f"[WARN] No highs found for {pair} from {DATE_UTC} onward.")
                continue

            gain_pct = (max_h - open_px) / open_px * 100.0

            if gain_pct < THRESHOLD:
                did_not_exceed.append((pair, open_px, max_h, gain_pct))
            else:
                exceeded.append((pair, open_px, max_h, gain_pct))

        except ccxt.BadSymbol:
            print(f"[ERROR] Bad symbol on Binance: {pair}")
        except Exception as e:
            print(f"[ERROR] {pair}: {e}")

    print(f"\n=== Ever-exceeded check from {DATE_UTC} (UTC) up to today (UTC) ===")
    print(f"Threshold: +{THRESHOLD:.2f}% from that day's open\n")

    if did_not_exceed:
        print("Still have NOT exceeded threshold since that open:")
        for sym, o, h, g in sorted(did_not_exceed, key=lambda x: x[0]):
            print(f"  {sym:>14} | open={o:.8f} max_high={h:.8f} max_gain={g:.3f}%")
    else:
        print("Still have NOT exceeded threshold: (none)")

    if exceeded:
        print("\nHave exceeded (or met) threshold at some point since that open:")
        for sym, o, h, g in sorted(exceeded, key=lambda x: x[0]):
            print(f"  {sym:>14} | open={o:.8f} max_high={h:.8f} max_gain={g:.3f}%")

    # --- Extra: print simple Python lists for downstream use ---
    did_not_list  = [denormalize(s) for s, *_ in did_not_exceed]
    exceeded_list = [denormalize(s) for s, *_ in exceeded]

    print("\n# Python lists for reuse")
    print("DID_NOT_EXCEED = [")
    for s in sorted(did_not_list):
        print(f'    "{s}",')
    print("]")

    print("\nEXCEEDED = [")
    for s in sorted(exceeded_list):
        print(f'    "{s}",')
    print("]")

    # --- New: among the ones that have NOT exceeded the threshold, find those now 10% below the open ---
    fell_10_below = []  # (pair, open_px, current_px, pct_change_vs_open)

    for pair, open_px, _max_h, _gain in did_not_exceed:
        try:
            curr_px = get_current_price(exch, pair)
            pct_change = (curr_px - open_px) / open_px * 100.0
            if curr_px <= open_px * 0.90:  # 10% or more below the open
                fell_10_below.append((pair, open_px, curr_px, pct_change))
        except Exception as e:
            print(f"[WARN] Could not fetch current price for {pair}: {e}")

    if fell_10_below:
        print("\nNow ≥10% BELOW the open (from filename date):")
        for sym, o, cp, pct in sorted(fell_10_below, key=lambda x: x[0]):
            print(f"  {sym:>14} | open={o:.8f} current={cp:.8f} change={pct:.3f}%")
    else:
        print("\nNow ≥10% BELOW the open: (none)")

    # Export as a simple Python list too
    fell_10_list = [denormalize(s) for s, *_ in fell_10_below]
    print("\nFELL_10_BELOW = [")
    for s in sorted(fell_10_list):
        print(f'    "{s}",')
    print("]")


if __name__ == "__main__":
    main()
