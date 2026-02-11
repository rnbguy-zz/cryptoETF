#!/usr/bin/env python3
"""
Rebuild pipeline:
1) Recreate ./outputs/output_YYYY-MM-DD.xlsx (symbol, Slope) for each day in range
2) Compute yellow circle days using the same logic style as your timeslope_plotter
3) For each day, create oversold_analysis/YYYY-MM-DD_oversold.txt based on
   check_dates derived from yellow circles (as-of that day), using optimized kline caching.

Optimization:
- Downloads DAILY klines once per symbol for the whole range (+3 days buffer), then splits/uses.
- Downloads HOURLY klines for RSI only for the needed window per symbol and caches per-day.
  (So subsequent days reuse the cache rather than redownloading.)
"""

import os
import sys
import json
import time
import math
import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.cluster import KMeans

BINANCE = "https://api.binance.com"
KLINES_URL = f"{BINANCE}/api/v3/klines"
BOOKTICKER_URL = f"{BINANCE}/api/v3/ticker/bookTicker"
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "tE06bWu6VfzgIB1wlyZfZzaZwPe0F6RyVQrp0Fh7B8fvTzNyhxe8UZSrJV3y0Iu0").strip()


def iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def parse_iso(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def daterange(start: date, end: date) -> List[date]:
    # inclusive
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur += timedelta(days=1)
    return out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)



def ensure_cached_range_bulk_1d_for_symbol(
    cache_dir: Path, symbol: str, start_day: date, end_day: date
):
    """
    Fetch 1d klines ONCE for start_day..end_day, split into per-day cache.
    IMPORTANT: do NOT write empty cache files, so temporary fetch issues can be retried.
    """
    # If every day exists already, skip
    all_present = True
    for d in daterange(start_day, end_day):
        if read_cached_day(cache_dir, "1d", symbol, d) is None:
            all_present = False
            break
    if all_present:
        return

    start_ms = int(datetime(start_day.year, start_day.month, start_day.day).timestamp() * 1000)
    end_ms = int((datetime(end_day.year, end_day.month, end_day.day) + timedelta(days=1)).timestamp() * 1000) - 1

    klines = fetch_klines_interval(symbol, "1d", start_ms, end_ms, limit=1000)
    buckets = split_klines_by_day(klines)

    # Write cache only for days that actually have rows.
    # If a day is missing (rows == []), leave it uncached so a later run can retry.
    for d in daterange(start_day, end_day):
        k = iso(d)
        rows = buckets.get(k, [])
        if rows:
            write_cached_day(cache_dir, "1d", symbol, d, rows)


def build_daily_data_from_symbol_1d_cache(
    cache_dir: Path, symbol: str, days: List[date]
) -> Dict[str, dict]:
    """
    For each day d, compute slope using Open(d-3) and Close(d).
    Returns mapping: day_str -> {"symbol":..., "Slope":...}
    """
    out = {}
    for d in days:
        d0 = d - timedelta(days=3)
        k0 = read_cached_day(cache_dir, "1d", symbol, d0) or []
        k1 = read_cached_day(cache_dir, "1d", symbol, d) or []
        if not k0 or not k1:
            continue

        try:
            open0 = float(k0[0][1])
            close1 = float(k1[0][4])
            if open0 == 0:
                continue
            slope = ((close1 - open0) / open0) * 100.0
            out[iso(d)] = {"symbol": symbol, "Slope": float(slope)}
        except Exception:
            continue

    return out


def rebuild_outputs_bulk_1d(
    symbols: List[str],
    days: List[date],
    cache_dir: Path,
    outputs_dir: Path,
    max_workers: int = 12,
):
    """
    Fast & output-safe:
    - Bulk-download 1d candles per symbol for [min(days)-3 .. max(days)]
    - Compute slopes locally for all days
    - Write one excel per day
    """
    ensure_dir(outputs_dir)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    start_day = min(days) - timedelta(days=3)
    end_day = max(days)

    # 1) Warm 1d cache per symbol ONCE
    print(f"[outputs] warming 1d cache per symbol for {iso(start_day)}..{iso(end_day)}")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(ensure_cached_range_bulk_1d_for_symbol, cache_dir, sym, start_day, end_day): sym
                for sym in symbols}
        for fut in as_completed(futs):
            _ = fut.result()

    # 2) Compute slopes and aggregate by day
    daily_data: Dict[str, List[dict]] = {iso(d): [] for d in days}

    print("[outputs] computing slopes from cached 1d candles...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(build_daily_data_from_symbol_1d_cache, cache_dir, sym, days): sym
                for sym in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            per_day = fut.result() or {}
            for day_str, row in per_day.items():
                daily_data[day_str].append(row)

    # 3) Write excel files (only if missing)
    for d in days:
        day_str = iso(d)
        out_file = outputs_dir / f"output_{day_str}.xlsx"
        if out_file.exists():
            continue
        rows = daily_data.get(day_str, [])
        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(out_file, index=False)
            print(f"[outputs] wrote {out_file} rows={len(df)}")
        else:
            print(f"[outputs] no rows for {day_str} (file not created)")


def safe_get(url: str, params: dict, retries: int = 5, backoff: float = 1.2) -> Optional[list]:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                # rate limit
                time.sleep(backoff ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "code" in data:
                # Binance error
                time.sleep(backoff ** attempt)
                continue
            return data
        except Exception:
            time.sleep(backoff ** attempt)
    return None

def get_all_usdt_symbols() -> List[str]:
    data = safe_get(BOOKTICKER_URL, params={})
    if not data:
        raise RuntimeError("Failed to fetch bookTicker symbols from Binance.")
    return [x["symbol"] for x in data if x.get("symbol", "").endswith("USDT")]

# ---------- Kline caching (split per day) ----------

def day_cache_path(cache_dir: Path, interval: str, symbol: str, d: date) -> Path:
    # e.g. cache/1h/BTCUSDT/2026-02-01.json
    return cache_dir / interval / symbol / f"{iso(d)}.json"

def read_cached_day(cache_dir: Path, interval: str, symbol: str, d: date) -> Optional[list]:
    p = day_cache_path(cache_dir, interval, symbol, d)
    if not p.exists():
        return None
    try:
        txt = p.read_text().strip()
        if not txt:
            return None
        return json.loads(txt)
    except Exception:
        return None

def write_cached_day(cache_dir: Path, interval: str, symbol: str, d: date, rows: list):
    p = day_cache_path(cache_dir, interval, symbol, d)
    ensure_dir(p.parent)
    p.write_text(json.dumps(rows))

def split_klines_by_day(klines: list) -> Dict[str, list]:
    """
    klines rows are Binance arrays:
    [
      openTime, open, high, low, close, volume, closeTime, quoteAssetVolume,
      trades, takerBuyBase, takerBuyQuote, ignore
    ]
    We'll bucket by UTC date from openTime.
    """
    buckets: Dict[str, list] = {}
    for row in klines:
        ot = int(row[0])
        d = datetime.utcfromtimestamp(ot / 1000).date()
        k = iso(d)
        buckets.setdefault(k, []).append(row)
    return buckets

def fetch_klines_interval(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> list:
    """
    Paginate Binance klines from start_ms..end_ms inclusive-ish.
    """
    out = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit
        }
        data = safe_get(KLINES_URL, params=params)
        if not data:
            break
        out.extend(data)
        if len(data) < limit:
            break
        # advance to next candle after last openTime
        last_ot = int(data[-1][0])
        cur = last_ot + 1
        # gentle pacing
        time.sleep(0.02)
    return out

def ensure_cached_range(
    cache_dir: Path,
    interval: str,
    symbol: str,
    start_day: date,
    end_day: date
):
    """
    Ensure per-day cache files exist for start_day..end_day.
    Only fetches missing days; downloads in larger window chunks, then splits.
    """
    missing = []
    for d in daterange(start_day, end_day):
        if read_cached_day(cache_dir, interval, symbol, d) is None:
            missing.append(d)
    if not missing:
        return

    # fetch one continuous window covering missing range (min..max)
    fetch_start = min(missing)
    fetch_end = max(missing)

    start_ms = int(datetime(fetch_start.year, fetch_start.month, fetch_start.day).timestamp() * 1000)
    # end_ms is end of fetch_end day
    end_ms = int((datetime(fetch_end.year, fetch_end.month, fetch_end.day) + timedelta(days=1)).timestamp() * 1000) - 1

    klines = fetch_klines_interval(symbol, interval, start_ms, end_ms)
    buckets = split_klines_by_day(klines)

    # write only missing days we have data for
    for d in missing:
        k = iso(d)
        rows = buckets.get(k, [])
        if rows:
            write_cached_day(cache_dir, interval, symbol, d, rows)
        else:
            # write empty to avoid refetch loops
            write_cached_day(cache_dir, interval, symbol, d, [])

# ---------- Step 1: rebuild outputs (daily slope files) ----------

def compute_slope_for_day(symbol: str, d: date, cache_dir: Path) -> Optional[float]:
    """
    Slope = (Close[d] - Open[d-3]) / Open[d-3] * 100
    Uses 1d klines cache.
    """
    # need day-3 and day
    d0 = d - timedelta(days=3)
    ensure_cached_range(cache_dir, "1d", symbol, d0, d)

    k0 = read_cached_day(cache_dir, "1d", symbol, d0) or []
    k1 = read_cached_day(cache_dir, "1d", symbol, d) or []
    if not k0 or not k1:
        return None

    # each day for 1d interval usually one candle, take first
    row0 = k0[0]
    row1 = k1[0]
    try:
        open0 = float(row0[1])
        close1 = float(row1[4])
        if open0 == 0:
            return None
        return ((close1 - open0) / open0) * 100.0
    except Exception:
        return None

def rebuild_outputs(
    symbols: List[str],
    days: List[date],
    cache_dir: Path,
    outputs_dir: Path,
    max_workers: int = 12,
):
    ensure_dir(outputs_dir)

    # We do per-day building, but to reduce downloads we rely on cached 1d per symbol/day.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    for d in days:
        out_file = outputs_dir / f"output_{iso(d)}.xlsx"
        if out_file.exists():
            continue

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(compute_slope_for_day, sym, d, cache_dir): sym for sym in symbols}
            for fut in as_completed(futs):
                sym = futs[fut]
                val = fut.result()
                if val is not None and not math.isnan(val):
                    results.append({"symbol": sym, "Slope": float(val)})

        if results:
            df = pd.DataFrame(results)
            df.to_excel(out_file, index=False)
            print(f"[outputs] wrote {out_file} rows={len(df)}")
        else:
            print(f"[outputs] no data for {iso(d)} (file not created)")

# ---------- Step 2: compute yellow circle days (as close as your logic) ----------

def filter_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    q1 = df[column].quantile(0.05)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]

def custom_pct_change(current, previous) -> float:
    if previous == 0 or pd.isna(previous):
        return 0.0
    return (abs(current - previous) / abs(previous)) * 100.0

def check_slope_below_previous_two(series: pd.Series) -> bool:
    if len(series) < 3:
        return False
    return series.iloc[-1] < series.iloc[-2] and series.iloc[-1] < series.iloc[-3]

def calculate_slope_after_drop(previous_slope: float, threshold_pct: float) -> float:
    return previous_slope - (previous_slope * (threshold_pct / 100.0))

def compute_yellow_circle_days_from_outputs(outputs_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reads all output_YYYY-MM-DD.xlsx, computes avg slope series, then applies your same logic:
    - Previous Slope
    - Slope Change (%)
    - Window 3/5 checks + thresholds
    - green dot condition: Green Line Above 20 (as in your code)
    - yellow circle: first day after a green dot where blue line exceeds that green dot’s blue level
    Returns df_results and yellow_circle_days (YYYY-MM-DD strings).
    """
    files = sorted([p for p in outputs_dir.glob("output_*.xlsx")])
    if not files:
        return pd.DataFrame(), []

    data = {"Date": [], "Average Slope": []}
    for p in files:
        date_str = p.stem.split("_", 1)[1]
        try:
            df = pd.read_excel(p)
            if "Slope" not in df.columns:
                continue
            df_f = filter_outliers(df, "Slope")
            avg = df_f["Slope"].mean()
            if not pd.isna(avg):
                data["Date"].append(date_str)
                data["Average Slope"].append(float(avg))
        except Exception:
            continue

    df_results = pd.DataFrame(data)
    if df_results.empty:
        return df_results, []

    df_results["Date"] = pd.to_datetime(df_results["Date"])
    df_results = df_results.sort_values("Date").reset_index(drop=True)

    df_results["Previous Slope"] = df_results["Average Slope"].shift(1)
    df_results["Slope Change (%)"] = df_results.apply(
        lambda r: custom_pct_change(r["Average Slope"], r["Previous Slope"]), axis=1
    )

    df_results["Slope Below Previous Two (Window 3)"] = df_results["Average Slope"].rolling(3).apply(
        check_slope_below_previous_two, raw=False
    )
    df_results["Slope Below Previous Two (Window 5)"] = df_results["Average Slope"].rolling(5).apply(
        check_slope_below_previous_two, raw=False
    )

    df_results["High Change (Window 3)"] = df_results["Slope Change (%)"] > (
        df_results["Slope Change (%)"].rolling(3).mean() + df_results["Slope Change (%)"].rolling(3).std()
    )
    df_results["High Change (Window 5)"] = df_results["Slope Change (%)"] > (
        df_results["Slope Change (%)"].rolling(5).mean() + df_results["Slope Change (%)"].rolling(5).std()
    )

    # Build "new slopes" like your green/red logic (we only need the green condition)
    new_slopes_window_3 = [np.nan]
    new_slopes_window_5 = [np.nan]
    for i in range(1, len(df_results)):
        prev = df_results["Previous Slope"].iloc[i]
        if pd.isna(prev):
            new_slopes_window_3.append(np.nan)
            new_slopes_window_5.append(np.nan)
            continue
        w3 = (df_results["Slope Change (%)"].rolling(3).mean().iloc[i] +
              df_results["Slope Change (%)"].rolling(3).std().iloc[i])
        w5 = (df_results["Slope Change (%)"].rolling(5).mean().iloc[i] +
              df_results["Slope Change (%)"].rolling(5).std().iloc[i])
        new_slopes_window_3.append(calculate_slope_after_drop(prev, w3))
        new_slopes_window_5.append(calculate_slope_after_drop(prev, w5))

    summary_df = pd.DataFrame({
        "Date": df_results["Date"],
        "Current Average Slope (Blue Line)": df_results["Average Slope"],
        "New Slope after Window 3 (Green Line)": new_slopes_window_3,
        "New Slope after Window 5 (Red Line)": new_slopes_window_5,
    })

    summary_df["Green Line Above 20"] = (
        (summary_df["New Slope after Window 3 (Green Line)"] >
         summary_df["Current Average Slope (Blue Line)"] + 10) &
        (summary_df["New Slope after Window 3 (Green Line)"] > 10)
    )

    # Now implement your “yellow circle” detector state machine:
    green_dot_days = []
    yellow_circle_days = []
    last_green_slope = None
    last_green_date = None
    pending_yellow_check = False

    for i in range(len(summary_df)):
        current_date = summary_df["Date"].iloc[i]
        current_slope = summary_df["Current Average Slope (Blue Line)"].iloc[i]

        if bool(summary_df["Green Line Above 20"].iloc[i]):
            # remove consecutive greens
            if last_green_date is None or (current_date - last_green_date).days > 1:
                green_dot_days.append(current_date.strftime("%Y-%m-%d"))
                last_green_slope = current_slope
                last_green_date = current_date
                pending_yellow_check = True
            else:
                # if green removed but pending check and slope rises above last green slope -> yellow
                if pending_yellow_check and last_green_slope is not None and current_slope > last_green_slope:
                    yellow_circle_days.append(current_date.strftime("%Y-%m-%d"))
                    pending_yellow_check = False
        else:
            if pending_yellow_check and last_green_slope is not None and current_slope > last_green_slope:
                yellow_circle_days.append(current_date.strftime("%Y-%m-%d"))
                pending_yellow_check = False

    # Enforce >=4 day spacing like your filter
    if yellow_circle_days:
        dt_list = [parse_iso(x) for x in yellow_circle_days]
        dt_list.sort()
        filtered = []
        prev = None
        for d in dt_list:
            if prev is None or (d - prev) >= timedelta(days=4):
                filtered.append(d)
            prev = d
        yellow_circle_days = [iso(d) for d in filtered]

    return df_results, yellow_circle_days

def check_dates_for_day(yellow_circle_days: List[str], day_str: str, n: int = 3) -> List[str]:
    """
    Mimic your writer:
    - If 'day' is itself a yellow circle day: take n entries BEFORE that day
    - Else: take last n yellow circle days (before that day)
    """
    # only consider yellows <= day
    y = [d for d in yellow_circle_days if d <= day_str]
    if not y:
        return []

    if day_str in y:
        idx = y.index(day_str)
        return y[max(0, idx - n):idx]
    else:
        # last n prior
        prior = [d for d in y if d < day_str]
        return prior[-n:] if prior else y[-n:]

# ---------- Step 3: oversold analysis (same output style, optimized caching) ----------

def hourly_to_rsi_series(hourly_klines: list) -> pd.DataFrame:
    if not hourly_klines:
        return pd.DataFrame()

    df = pd.DataFrame(hourly_klines, columns=[
        "open_time","open","high","low","close","volume","close_time","qav","trades","tbb","tbq","ignore"
    ])

    # IMPORTANT: match original script -> tz-naive timestamps
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")  # <-- no utc=True
    df["close"] = df["close"].astype(float)
    df = df.set_index("open_time")

    df["RSI"] = ta.rsi(df["close"], length=14)
    return df[["RSI"]].dropna()



def rsi_at_or_before(df_rsi: pd.DataFrame, day_str: str) -> Optional[float]:
    if df_rsi.empty:
        return None

    # IMPORTANT: match original script -> naive midnight
    date_obj = datetime.strptime(day_str, "%Y-%m-%d")  # naive
    sub = df_rsi[df_rsi.index <= date_obj]
    if sub.empty:
        return None
    return float(sub.iloc[-1]["RSI"])


def detect_rsi_drops(rsi_values: Dict[str, Dict[str, float]]) -> Dict[str, dict]:
    sudden = {}
    for sym, rsis in rsi_values.items():
        if not rsis:
            continue
        rsi_dates = sorted(rsis.keys())
        for i in range(1, len(rsi_dates)):
            prev_d = rsi_dates[i-1]
            cur_d  = rsi_dates[i]
            prev_r = rsis[prev_d]
            cur_r  = rsis[cur_d]
            if prev_r - cur_r > 10:
                sudden[sym] = {
                    "Previous Date": prev_d,
                    "Current Date": cur_d,
                    "Previous RSI": prev_r,
                    "Current RSI": cur_r,
                    "Drop": prev_r - cur_r
                }
    return sudden

def kmeans_cluster_rsi_drops(rsi_drops: Dict[str, dict], n_clusters: int) -> Dict[int, list]:
    drop_values = np.array([[data["Drop"]] for data in rsi_drops.values()])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(drop_values)
    grouped = {i: [] for i in range(n_clusters)}
    for i, (sym, data) in enumerate(rsi_drops.items()):
        grouped[int(clusters[i])].append({sym: data})
    return grouped

def label_clusters(clustered: Dict[int, list]) -> Dict[str, list]:
    cluster_averages = {
        cid: np.mean([entry[list(entry.keys())[0]]["Drop"] for entry in items])
        for cid, items in clustered.items()
        if items
    }
    sorted_clusters = sorted(cluster_averages.items(), key=lambda x: x[1])
    labels = ["VerySmall", "Small", "Medium", "Large"]
    labeled = {}
    for i, (cid, _) in enumerate(sorted_clusters):
        if i < len(labels):
            labeled[labels[i]] = clustered[cid]
    return labeled

def build_oversold_for_day(
    symbols: List[str],
    cache_dir: Path,
    oversold_dir: Path,
    day: date,
    check_dates: List[str],
    max_workers: int = 12,
):
    """
    Create oversold_analysis/{day}_oversold.txt using check_dates + [day]
    RSI is computed from cached hourly klines. Downloads only missing hourly days.
    """
    ensure_dir(oversold_dir)
    final_date = iso(day)

    if not check_dates:
        # No check dates => cannot compute "between check dates"; still create empty file if you want
        out = oversold_dir / f"{final_date}_oversold.txt"
        out.write_text("")
        print(f"[oversold] {final_date}: no check_dates; wrote empty {out.name}")
        return

    # Determine earliest hourly day needed: min(check_dates) - 7 days buffer
    earliest = min(parse_iso(x) for x in check_dates)
    start_day = earliest - timedelta(days=7)
    end_day = day  # inclusive

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Compute RSI values at each date for each symbol
    rsi_values: Dict[str, Dict[str, float]] = {sym: {} for sym in symbols}

    def process_symbol(sym: str):
        # ensure hourly cache exists
        ensure_cached_range(cache_dir, "1h", sym, start_day, end_day)

        # stitch hourly klines for [start_day..end_day]
        rows = []
        for d0 in daterange(start_day, end_day):
            rows.extend(read_cached_day(cache_dir, "1h", sym, d0) or [])
        df_rsi = hourly_to_rsi_series(rows)
        if df_rsi.empty:
            return sym, {}

        vals = {}
        for ds in check_dates + [final_date]:
            v = rsi_at_or_before(df_rsi, ds)
            if v is not None:
                vals[ds] = v
        return sym, vals

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_symbol, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            sym, vals = fut.result()
            if vals:
                rsi_values[sym] = vals

    rsi_drops = detect_rsi_drops(rsi_values)

    out_path = oversold_dir / f"{final_date}_oversold.txt"

    if not rsi_drops:
        out_path.write_text("")
        print(f"[oversold] {final_date}: no drops; wrote empty {out_path.name}")
        return

    num_clusters = min(len(rsi_drops), 4)
    clustered = kmeans_cluster_rsi_drops(rsi_drops, num_clusters)
    labeled = label_clusters(clustered)

    # Match your output: only write the Small cluster, sorted ascending by Drop
    with open(out_path, "w") as f:
        for label, cluster_data in labeled.items():
            if label == "Small":
                f.write(f"\n{label} Drop Cluster:\n")
                cluster_list = [
                    {**entry[list(entry.keys())[0]], "Symbol": list(entry.keys())[0]}
                    for entry in cluster_data
                ]
                cluster_df = pd.DataFrame(cluster_list)
                if not cluster_df.empty:
                    f.write(cluster_df.sort_values(by="Drop", ascending=True).to_string(index=False))

    print(f"[oversold] wrote {out_path}")

# ---------- Main orchestration ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-back", type=int, default=90, help="How many days back from yesterday to rebuild")
    ap.add_argument("--start-date", type=str, default="", help="YYYY-MM-DD (inclusive). Overrides --days-back if set.")
    ap.add_argument("--end-date", type=str, default="", help="YYYY-MM-DD (inclusive). Defaults to yesterday UTC.")
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--cache-dir", type=str, default="./cache")
    ap.add_argument("--outputs-dir", type=str, default="./outputs")
    ap.add_argument("--oversold-dir", type=str, default="./oversold_analysis")
    ap.add_argument("--yellow-dir", type=str, default="./yellowcircle_history", help="Write per-day yellowcircle txt here")
    ap.add_argument("--symbols-limit", type=int, default=0, help="For testing: limit symbols to first N")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    outputs_dir = Path(args.outputs_dir)
    oversold_dir = Path(args.oversold_dir)
    yellow_dir = Path(args.yellow_dir)
    ensure_dir(cache_dir)
    ensure_dir(outputs_dir)
    ensure_dir(oversold_dir)
    ensure_dir(yellow_dir)

    # Date range: default = last N days ending yesterday (UTC)
    utc_today = datetime.utcnow().date()
    default_end = utc_today - timedelta(days=1)

    if args.end_date:
        end = parse_iso(args.end_date)
    else:
        end = default_end

    if args.start_date:
        start = parse_iso(args.start_date)
    else:
        start = end - timedelta(days=args.days_back - 1)

    days = daterange(start, end)
    if not days:
        print("No days to process.")
        return

    print(f"[range] {iso(start)} .. {iso(end)}  (days={len(days)})")

    # Symbols list
    symbols = get_all_usdt_symbols()
    if args.symbols_limit and args.symbols_limit > 0:
        symbols = symbols[:args.symbols_limit]
    print(f"[symbols] {len(symbols)}")

    # 1) Rebuild missing output_*.xlsx (slope)
    print("\n=== Step 1: rebuild outputs ===")
    #rebuild_outputs(symbols, days, cache_dir, outputs_dir, max_workers=args.max_workers)
    rebuild_outputs_bulk_1d(symbols, days, cache_dir, outputs_dir, max_workers=args.max_workers)

    # 2) Compute yellow circle days from outputs
    print("\n=== Step 2: compute yellow circles ===")
    df_results, yellow_circle_days = compute_yellow_circle_days_from_outputs(outputs_dir)
    print(f"[yellow] found {len(yellow_circle_days)} yellow days in outputs history")

    # Save a “master” list for reference
    (yellow_dir / "yellow_circle_days_all.txt").write_text("\n".join(yellow_circle_days))

    # 3) For each day, derive check_dates as-of that day and build oversold file
    print("\n=== Step 3: rebuild oversold files ===")
    for d in days:
        d_str = iso(d)
        check_dates = check_dates_for_day(yellow_circle_days, d_str, n=3)

        # also write the per-day yellowcircle input file (so you can inspect/re-run)
        (yellow_dir / f"yellowcircle_{d_str}.txt").write_text("\n".join(check_dates))

        # oversold output
        out_path = oversold_dir / f"{d_str}_oversold.txt"
        if out_path.exists() and out_path.stat().st_size > 0:
            # keep existing non-empty
            continue

        build_oversold_for_day(
            symbols=symbols,
            cache_dir=cache_dir,
            oversold_dir=oversold_dir,
            day=d,
            check_dates=check_dates,
            max_workers=args.max_workers,
        )

    # Also write a conventional yellowcircle.txt for the last processed day (handy compatibility)
    last_day = iso(days[-1])
    last_check = check_dates_for_day(yellow_circle_days, last_day, n=3)
    Path("yellowcircle.txt").write_text("\n".join(last_check))
    print(f"\n[done] wrote yellowcircle.txt for {last_day}: {last_check}")

if __name__ == "__main__":
    main()

