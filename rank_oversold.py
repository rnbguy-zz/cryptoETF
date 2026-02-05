#!/usr/bin/env python3
"""
rank_oversold.py  (AI-ready-to-buy classifier + MODEL PERSISTENCE + BTC MOOD OUTPUT FILTER)
========================================================================================
What this trimmed script does (based on your actual CLI usage):
  - Rolling split: trains on all file_dates <= (latest_file_date - holdout_days), predicts on newer dates
  - Downside "drop10" model is trained FIRST (no OOF; uses in-sample probs because you use --skip-oof)
  - Its predicted probability feat_prob_drop10 becomes a FEATURE for the main "hits target" model
  - Main model predicts BUY candidates in predict window
  - BTC mood filter is OUTPUT-ONLY and ALWAYS BULLISH (does NOT affect training)
  - Adds runup-to-high from entry open to highest high until today (UTC, includes today candle)
  - Always prints + emails results (CSV attachment if rows exist)

Removed as unused:
  - Sideways filter + all sideways/prev-signal feature code
  - OOF/CV downside feature path (kept skip-oof only)
  - Unused CLI args
"""

from __future__ import annotations

import argparse
import os
import re
import smtplib
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import __version__ as sklearn_version


# ============================================================
# EMAIL CONFIG (use env vars if set; falls back to defaults)
# ============================================================
RECIPIENT_EMAIL = os.getenv("OVERSOLD_RECIPIENT_EMAIL", "mina.moussa@hotmail.com")
SENDER_EMAIL = os.getenv("OVERSOLD_SENDER_EMAIL", "minamoussa903@gmail.com")
SENDER_PASSWORD = os.getenv("OVERSOLD_SENDER_PASSWORD", "qhvi syra bbad gylu")  # set this in env ideally

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def send_email_with_analysis(body: str, df: Optional[pd.DataFrame] = None, directory: str = ".") -> None:
    """
    Always attempts to send email. Attaches filtered CSV if df has rows.
    """
    if not SENDER_PASSWORD:
        print("Email not sent: OVERSOLD_SENDER_PASSWORD is empty.", file=sys.stderr)
        return

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = "Oversold Analysis Alert"
    msg.attach(MIMEText(body, "plain"))

    if df is not None and not df.empty:
        os.makedirs(directory, exist_ok=True)
        temp_csv = os.path.join(directory, "filtered_output.csv")
        df.to_csv(temp_csv, index=False)

        with open(temp_csv, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(temp_csv)}"')
        msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print("Email sent.")
    except Exception as e:
        print("Error sending email:", e, file=sys.stderr)


# ============================================================
# PARSING OVERSOLD FILES
# ============================================================
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class OversoldHit:
    symbol: str
    file_date: date  # scan day (from filename)
    prev_date: date
    cur_date: date
    prev_rsi: float
    cur_rsi: float
    drop: float


def parse_file_date(path: Path) -> date:
    return datetime.strptime(path.name[:10], "%Y-%m-%d").date()


def parse_oversold_line(line: str) -> Optional[Tuple[date, date, float, float, float, str]]:
    s = line.strip()
    if not s:
        return None

    # allow "filename: content"
    if ":" in s:
        left, right = s.split(":", 1)
        if left.endswith("_oversold.txt") or left.endswith(".txt"):
            s = right.strip()

    parts = s.split()
    if len(parts) < 6:
        return None
    if not DATE_RE.match(parts[0]) or not DATE_RE.match(parts[1]):
        return None

    try:
        prev_date = datetime.strptime(parts[0], "%Y-%m-%d").date()
        cur_date = datetime.strptime(parts[1], "%Y-%m-%d").date()
        prev_rsi = float(parts[2])
        cur_rsi = float(parts[3])
        drop = float(parts[4])
        symbol = parts[5].upper()
    except Exception:
        return None

    return prev_date, cur_date, prev_rsi, cur_rsi, drop, symbol


def load_oversold_hits(folder: Path) -> List[OversoldHit]:
    files = sorted(folder.glob("*_oversold.txt"))
    if not files:
        raise RuntimeError(f"No *_oversold.txt files found in {folder}")

    all_hits: List[OversoldHit] = []
    for fp in files:
        fday = parse_file_date(fp)
        lines = fp.read_text(errors="ignore").splitlines()
        for line in lines:
            parsed = parse_oversold_line(line)
            if not parsed:
                continue
            prev_d, cur_d, prev_rsi, cur_rsi, drop, sym = parsed
            all_hits.append(
                OversoldHit(
                    symbol=sym,
                    file_date=fday,
                    prev_date=prev_d,
                    cur_date=cur_d,
                    prev_rsi=prev_rsi,
                    cur_rsi=cur_rsi,
                    drop=drop,
                )
            )
    return all_hits


# ============================================================
# BINANCE ACCESS + GLOBAL COOLDOWN + THREAD SESSIONS
# ============================================================
BINANCE_BASE = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

_BINANCE_COOLDOWN_LOCK = threading.Lock()
_BINANCE_COOLDOWN_UNTIL = 0.0  # monotonic seconds


def _binance_wait_if_cooling() -> None:
    global _BINANCE_COOLDOWN_UNTIL
    while True:
        with _BINANCE_COOLDOWN_LOCK:
            until = _BINANCE_COOLDOWN_UNTIL
        now = time.monotonic()
        if now >= until:
            return
        time.sleep(min(0.25, until - now))


def _binance_trigger_cooldown(seconds: float = 1.0) -> None:
    global _BINANCE_COOLDOWN_UNTIL
    now = time.monotonic()
    with _BINANCE_COOLDOWN_LOCK:
        _BINANCE_COOLDOWN_UNTIL = max(_BINANCE_COOLDOWN_UNTIL, now + float(seconds))


def _binance_get_with_cooldown(
    session: requests.Session, url: str, *, params: Dict, timeout: int = 30
) -> Optional[requests.Response]:
    """
    Thread-safe Binance GET wrapper:
      - waits for global cooldown
      - triggers global cooldown on 429/418 or transient errors
      - retries a few times
    """
    MAX_RETRIES = 6
    last_resp: Optional[requests.Response] = None

    for attempt in range(1, MAX_RETRIES + 1):
        _binance_wait_if_cooling()
        try:
            resp = session.get(url, params=params, timeout=timeout)
            last_resp = resp
        except requests.RequestException:
            _binance_trigger_cooldown(1.0)
            if attempt == MAX_RETRIES:
                return None
            continue

        if resp.status_code in (429, 418):
            _binance_trigger_cooldown(1.0)
            if attempt == MAX_RETRIES:
                return resp
            continue

        if 500 <= resp.status_code < 600:
            _binance_trigger_cooldown(1.0)
            if attempt == MAX_RETRIES:
                return resp
            continue

        return resp

    return last_resp


_tls = threading.local()


def get_thread_session() -> requests.Session:
    """
    requests.Session is NOT thread-safe. Use one session per thread.
    """
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        _tls.session = s
    return s


# ============================================================
# PER-SYMBOL PARQUET STORE FOR 1D KLINES + TOMBSTONES
# ============================================================
_SYMBOL_MEMO_LOCK = threading.Lock()
_SYMBOL_MEMO: Dict[str, pd.DataFrame] = {}  # in-process memo (symbol -> df)

_FILE_LOCKS_LOCK = threading.Lock()
_FILE_LOCKS: Dict[str, threading.Lock] = {}  # symbol -> lock (for read/modify/write)


def _get_symbol_lock(symbol: str) -> threading.Lock:
    with _FILE_LOCKS_LOCK:
        lk = _FILE_LOCKS.get(symbol)
        if lk is None:
            lk = threading.Lock()
            _FILE_LOCKS[symbol] = lk
        return lk


def _normalize_store_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["day", "open_time", "open", "high", "low", "close", "close_time", "no_data"]
        )

    df = df.copy()

    # Normalize day to python date objects
    if "day" in df.columns:
        s = df["day"]
        if np.issubdtype(s.dtype, np.datetime64):
            df["day"] = pd.to_datetime(s, utc=False).dt.date
        else:
            df["day"] = pd.to_datetime(s.astype(str), format="%Y-%m-%d", errors="coerce").dt.date
        df = df[df["day"].notna()].copy()
        df["day"] = df["day"].astype(object)

    for c in ["open_time", "open", "high", "low", "close", "close_time", "no_data"]:
        if c not in df.columns:
            df[c] = 0 if c == "no_data" else np.nan

    df["no_data"] = pd.to_numeric(df["no_data"], errors="coerce").fillna(0).astype("int8").astype(bool)
    return df


def _store_path(store_dir: Path, symbol: str) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / f"{symbol}_1d.parquet"


def _read_parquet_best_effort(path: Path) -> pd.DataFrame:
    # Try pyarrow first, then fastparquet
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def _write_parquet_best_effort(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    # Let pandas pick engine; fallback if needed
    try:
        df.to_parquet(tmp, index=False, engine="pyarrow")
    except Exception:
        df.to_parquet(tmp, index=False, engine="fastparquet")
    os.replace(tmp, path)


def _read_symbol_store(store_dir: Path, symbol: str) -> pd.DataFrame:
    p = _store_path(store_dir, symbol)
    if not p.exists():
        return _normalize_store_df(pd.DataFrame())
    try:
        df = _read_parquet_best_effort(p)
        return _normalize_store_df(df)
    except Exception:
        try:
            p.unlink()
        except Exception:
            pass
        return _normalize_store_df(pd.DataFrame())


def _write_symbol_store(store_dir: Path, symbol: str, df: pd.DataFrame) -> None:
    p = _store_path(store_dir, symbol)
    df = _normalize_store_df(df)
    if df.empty:
        return

    df = df.sort_values("day").drop_duplicates("day", keep="last").copy()
    df["day"] = df["day"].astype(str)  # stable parquet storage type
    df["no_data"] = df["no_data"].astype(bool)
    _write_parquet_best_effort(df, p)


def _binance_fetch_1d_range(symbol: str, start_day: date, end_day: date, session: requests.Session) -> pd.DataFrame:
    """
    Fetch 1d klines inclusive start_day..end_day (UTC).
    Chunks requests (Binance limit ~1000).
    """
    if end_day < start_day:
        return pd.DataFrame()

    all_rows: List[Dict] = []
    chunk = 900
    cur = start_day
    while cur <= end_day:
        chunk_end = min(end_day, cur + timedelta(days=chunk - 1))

        start_dt = datetime.combine(cur, datetime.min.time())
        end_dt = datetime.combine(chunk_end + timedelta(days=1), datetime.min.time())

        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": int(start_dt.timestamp() * 1000),
            "endTime": int(end_dt.timestamp() * 1000),
            "limit": 1000,
        }

        r = _binance_get_with_cooldown(session, BINANCE_BASE + KLINES_ENDPOINT, params=params, timeout=30)
        data = None
        if r is not None and r.status_code == 200:
            try:
                data = r.json()
            except Exception:
                data = None

        if isinstance(data, list) and data:
            for k in data:
                try:
                    open_time_ms = int(k[0])
                    d = datetime.utcfromtimestamp(open_time_ms / 1000.0).date()
                    all_rows.append(
                        {
                            "day": d,
                            "open_time": open_time_ms,
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "close_time": int(k[6]),
                            "no_data": False,
                        }
                    )
                except Exception:
                    continue

        cur = chunk_end + timedelta(days=1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return _normalize_store_df(df)


def _fill_tombstones(start_day: date, end_day: date, df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every day in [start_day..end_day] exists in df (either candle or no_data tombstone).
    """
    df = _normalize_store_df(df)
    if end_day < start_day:
        return df

    have = set(df["day"].tolist()) if not df.empty else set()
    rows = []
    d = start_day
    while d <= end_day:
        if d not in have:
            rows.append(
                {
                    "day": d,
                    "open_time": np.nan,
                    "open": np.nan,
                    "high": np.nan,
                    "low": np.nan,
                    "close": np.nan,
                    "close_time": np.nan,
                    "no_data": True,
                }
            )
        d += timedelta(days=1)

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    df = df.sort_values("day").drop_duplicates("day", keep="last")
    return df


def ensure_symbol_days(
    symbol: str,
    need_start: date,
    need_end: date,
    *,
    session: requests.Session,
    store_dir: Path,
) -> pd.DataFrame:
    """
    Load symbol store once, fetch only missing head/tail, save back, and memoize in-process.
    Thread-safe: uses a per-symbol lock to avoid concurrent read/modify/write.
    """
    if need_end < need_start:
        return _normalize_store_df(pd.DataFrame())

    lk = _get_symbol_lock(symbol)
    with lk:
        with _SYMBOL_MEMO_LOCK:
            memo = _SYMBOL_MEMO.get(symbol)
        if memo is not None and not memo.empty:
            have_min = memo["day"].min()
            have_max = memo["day"].max()
            if need_start >= have_min and need_end <= have_max:
                return memo

        df = _read_symbol_store(store_dir, symbol)

        if df.empty:
            fetched = _binance_fetch_1d_range(symbol, need_start, need_end, session)
            fetched = _fill_tombstones(need_start, need_end, fetched)
            if not fetched.empty:
                _write_symbol_store(store_dir, symbol, fetched)
            with _SYMBOL_MEMO_LOCK:
                _SYMBOL_MEMO[symbol] = fetched
            return fetched

        have_min = df["day"].min()
        have_max = df["day"].max()

        parts = [df]

        if need_start < have_min:
            back = _binance_fetch_1d_range(symbol, need_start, have_min - timedelta(days=1), session)
            back = _fill_tombstones(need_start, have_min - timedelta(days=1), back)
            if not back.empty:
                parts.append(back)

        if need_end > have_max:
            fwd = _binance_fetch_1d_range(symbol, have_max + timedelta(days=1), need_end, session)
            fwd = _fill_tombstones(have_max + timedelta(days=1), need_end, fwd)
            if not fwd.empty:
                parts.append(fwd)

        merged = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        merged = _fill_tombstones(need_start, need_end, merged)

        if not merged.empty:
            _write_symbol_store(store_dir, symbol, merged)

        with _SYMBOL_MEMO_LOCK:
            _SYMBOL_MEMO[symbol] = merged

        return merged


def get_klines_window(
    symbol: str,
    start_day: date,
    end_day: date,
    *,
    session: requests.Session,
    store_dir: Path,
) -> pd.DataFrame:
    """
    Returns only days with actual OHLC data (no tombstones), inclusive start_day..end_day.
    """
    if end_day < start_day:
        return pd.DataFrame()

    full = ensure_symbol_days(symbol, start_day, end_day, session=session, store_dir=store_dir)
    if full.empty:
        return pd.DataFrame()

    m = (full["day"] >= start_day) & (full["day"] <= end_day) & (~full["no_data"])
    out = full.loc[m, ["day", "open", "high", "low", "close", "open_time", "close_time"]].copy()
    if out.empty:
        return out
    out["day"] = out["day"].astype(str)
    return out.sort_values("day").reset_index(drop=True)


def prefetch_symbol_windows(
    hits: List[OversoldHit],
    *,
    entry_lag_days: int,
    horizon_days: int,
    store_dir: Path,
    dl_workers: int,
) -> None:
    """
    Prefetch per symbol: [min_needed .. last_closed_utc_day]
    (No sideways prefetch; removed.)
    """
    if not hits:
        return

    last_closed = datetime.utcnow().date() - timedelta(days=1)
    by_sym: Dict[str, List[OversoldHit]] = defaultdict(list)
    for h in hits:
        by_sym[h.symbol].append(h)

    need_ranges: Dict[str, Tuple[date, date]] = {}
    for sym, hs in by_sym.items():
        entry_days = [x.file_date + timedelta(days=entry_lag_days) for x in hs]
        if not entry_days:
            continue
        min_entry = min(entry_days)
        min_needed = min_entry
        max_needed = last_closed
        need_ranges[sym] = (min_needed, max_needed)

    def _prefetch_one(sym: str, d0: date, d1: date) -> None:
        s = get_thread_session()
        ensure_symbol_days(sym, d0, d1, session=s, store_dir=store_dir)

    with ThreadPoolExecutor(max_workers=max(1, int(dl_workers))) as ex:
        futs = [ex.submit(_prefetch_one, sym, d0, d1) for sym, (d0, d1) in need_ranges.items()]
        for f in futs:
            f.result()


# ============================================================
# LABELS + RUNUP
# ============================================================
def label_plus_target_within_horizon(
    symbol: str,
    file_day: date,
    *,
    session: requests.Session,
    store_dir: Path,
    target_pct: float,
    horizon_days: int,
    entry_lag_days: int,
) -> Optional[int]:
    """
    label = 1 if within horizon_days AFTER ENTRY_DAY, max(close) >= entry_open*(1+target_pct)
    ENTRY_DAY = file_day + entry_lag_days
    """
    last_closed = datetime.utcnow().date() - timedelta(days=1)
    entry_day = file_day + timedelta(days=entry_lag_days)
    start_day = entry_day
    end_day = min(entry_day + timedelta(days=horizon_days), last_closed)
    if end_day < start_day:
        return None

    df = get_klines_window(symbol, start_day, end_day, session=session, store_dir=store_dir)
    if df.empty:
        return None

    entry_open = float(df.iloc[0]["open"])
    max_close = float(df["close"].max())
    target = entry_open * (1.0 + target_pct)
    return 1 if max_close >= target else 0


def label_drop_pct_within_horizon(
    symbol: str,
    file_day: date,
    *,
    session: requests.Session,
    store_dir: Path,
    drop_pct: float,
    horizon_days: int,
    entry_lag_days: int,
) -> Optional[int]:
    """
    label = 1 if within horizon_days AFTER ENTRY_DAY, min(low) <= entry_open*(1-drop_pct)
    ENTRY_DAY = file_day + entry_lag_days
    """
    last_closed = datetime.utcnow().date() - timedelta(days=1)
    entry_day = file_day + timedelta(days=entry_lag_days)
    start_day = entry_day
    end_day = min(entry_day + timedelta(days=horizon_days), last_closed)
    if end_day < start_day:
        return None

    df = get_klines_window(symbol, start_day, end_day, session=session, store_dir=store_dir)
    if df.empty:
        return None

    entry_open = float(df.iloc[0]["open"])
    min_low = float(df["low"].min())
    thresh = entry_open * (1.0 - drop_pct)
    return 1 if min_low <= thresh else 0


def runup_to_high_until_today(
    symbol: str,
    entry_day: date,
    *,
    session: requests.Session,
    store_dir: Path,
    include_today: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Run-up from ENTRY DAY OPEN to highest HIGH from entry_day through:
      - today UTC (including today's in-progress 1d candle) if include_today=True
      - otherwise through last closed UTC day
    """
    end_day = datetime.utcnow().date() if include_today else (datetime.utcnow().date() - timedelta(days=1))
    if entry_day > end_day:
        return None

    df = get_klines_window(symbol, entry_day, end_day, session=session, store_dir=store_dir)
    if df is None or df.empty:
        return None

    dday = pd.to_datetime(df["day"], errors="coerce").dt.date
    opens = pd.to_numeric(df["open"], errors="coerce")
    highs = pd.to_numeric(df["high"], errors="coerce")

    ok = dday.notna() & opens.notna() & highs.notna()
    df = df.loc[ok].copy()
    dday = dday.loc[ok]
    opens = opens.loc[ok]
    highs = highs.loc[ok]
    if df.empty:
        return None

    entry_mask = (dday == entry_day)
    if not entry_mask.any():
        return None

    entry_open = float(opens.loc[entry_mask].iloc[0])
    max_idx = highs.idxmax()
    max_high = float(highs.loc[max_idx])
    max_high_date = dday.loc[max_idx].isoformat()
    runup_pct = ((max_high / max(entry_open, 1e-12)) - 1.0) * 100.0

    return {
        "entry_open": entry_open,
        "max_high_to_today": max_high,
        "max_high_date": max_high_date,
        "runup_to_high_pct": float(runup_pct),
    }


# ============================================================
# BTC MOOD (OUTPUT ONLY) - ALWAYS BULLISH FILTER
# ============================================================
def btc_market_mood_for_file_date(
    file_day: date,
    *,
    session: requests.Session,
    store_dir: Path,
    entry_lag_days: int,
) -> Dict[str, object]:
    """
    Uses BTCUSDT 1D candle for the day *before entry*.
      entry_day = file_day + entry_lag_days
      mood_day  = entry_day - 1 day
    """
    entry_day = file_day + timedelta(days=int(entry_lag_days))
    mood_day = entry_day - timedelta(days=1)

    df = get_klines_window("BTCUSDT", mood_day, mood_day, session=session, store_dir=store_dir)
    if df is None or df.empty:
        return {"mood": "UNKNOWN", "o": np.nan, "c": np.nan, "pct": np.nan, "mood_day": mood_day.isoformat()}

    try:
        o = float(df.iloc[0]["open"])
        c = float(df.iloc[0]["close"])
    except Exception:
        return {"mood": "UNKNOWN", "o": np.nan, "c": np.nan, "pct": np.nan, "mood_day": mood_day.isoformat()}

    if c > o:
        mood = "BULLISH"
    elif c < o:
        mood = "BEARISH"
    else:
        mood = "NEUTRAL"

    pct = ((c - o) / o) * 100.0 if abs(o) > 1e-12 else 0.0
    return {"mood": mood, "o": o, "c": c, "pct": pct, "mood_day": mood_day.isoformat()}


def add_btc_mood_columns_and_filter_dates_bullish_only(
    df_pred: pd.DataFrame,
    *,
    store_dir: Path,
    entry_lag_days: int,
    dl_workers: int,
) -> pd.DataFrame:
    """
    Adds btc mood columns and filters entire dates to ONLY those where BTC mood is BULLISH.
    This is OUTPUT ONLY; df_pred already comes from predict window.
    """
    if df_pred is None or df_pred.empty:
        return df_pred

    unique_file_dates = sorted(df_pred["file_date"].unique().tolist())
    results: Dict[str, Dict[str, object]] = {}

    def _task(d_str: str) -> Tuple[str, Dict[str, object]]:
        s = get_thread_session()
        fday = datetime.strptime(d_str, "%Y-%m-%d").date()
        info = btc_market_mood_for_file_date(
            fday, session=s, store_dir=store_dir, entry_lag_days=entry_lag_days
        )
        return d_str, info

    with ThreadPoolExecutor(max_workers=max(1, int(dl_workers))) as ex:
        futs = [ex.submit(_task, d_str) for d_str in unique_file_dates]
        for fut in futs:
            d_str, info = fut.result()
            results[d_str] = info

    df_pred = df_pred.copy()
    df_pred["btc_mood_day"] = df_pred["file_date"].map(lambda d: results.get(d, {}).get("mood_day", ""))
    df_pred["btc_open"] = df_pred["file_date"].map(lambda d: results.get(d, {}).get("o", np.nan))
    df_pred["btc_close"] = df_pred["file_date"].map(lambda d: results.get(d, {}).get("c", np.nan))
    df_pred["btc_change_pct"] = df_pred["file_date"].map(lambda d: results.get(d, {}).get("pct", np.nan))
    df_pred["btc_mood"] = df_pred["file_date"].map(lambda d: results.get(d, {}).get("mood", "UNKNOWN"))

    keep_dates = [d for d in unique_file_dates if results.get(d, {}).get("mood") == "BULLISH"]
    return df_pred[df_pred["file_date"].isin(keep_dates)].copy()


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def build_features_for_symbol(history: List[OversoldHit], asof_file_date: date) -> Dict[str, float]:
    hits = [h for h in history if h.file_date <= asof_file_date]
    hits.sort(key=lambda h: h.file_date)
    if not hits:
        return {}

    last = hits[-1]
    n = len(hits)

    def within(days: int) -> List[OversoldHit]:
        start = asof_file_date - timedelta(days=days)
        return [h for h in hits if h.file_date >= start]

    w7 = within(7)
    w30 = within(30)
    w60 = within(60)

    drops = np.array([h.drop for h in hits], dtype=float)
    cur_rsis = np.array([h.cur_rsi for h in hits], dtype=float)
    prev_rsis = np.array([h.prev_rsi for h in hits], dtype=float)

    days_since_first = (asof_file_date - hits[0].file_date).days
    days_since_last = (asof_file_date - last.file_date).days

    x = np.arange(n, dtype=float)
    if n >= 2:
        x_mean = x.mean()
        y = cur_rsis
        slope = float(np.sum((x - x_mean) * (y - y.mean())) / (np.sum((x - x_mean) ** 2) + 1e-9))
    else:
        slope = 0.0

    drop_pct_simple = (last.cur_rsi / max(last.prev_rsi, 1e-9))
    abs_rsi_45 = abs(last.cur_rsi - 45.0)

    feats = {
        "last_prev_rsi": float(last.prev_rsi),
        "last_cur_rsi": float(last.cur_rsi),
        "last_drop": float(last.drop),
        "last_drop_pct_simple": float(drop_pct_simple),
        "last_abs_rsi_45": float(abs_rsi_45),
        "hits_total": float(n),
        "hits_7d": float(len(w7)),
        "hits_30d": float(len(w30)),
        "hits_60d": float(len(w60)),
        "days_since_first": float(days_since_first),
        "days_since_last": float(days_since_last),
        "drop_mean": float(drops.mean()),
        "drop_std": float(drops.std(ddof=0)),
        "drop_max": float(drops.max()),
        "drop_min": float(drops.min()),
        "cur_rsi_mean": float(cur_rsis.mean()),
        "cur_rsi_min": float(cur_rsis.min()),
        "cur_rsi_max": float(cur_rsis.max()),
        "prev_rsi_mean": float(prev_rsis.mean()),
        "drop_max_7d": float(max([h.drop for h in w7], default=last.drop)),
        "drop_max_30d": float(max([h.drop for h in w30], default=last.drop)),
        "cur_rsi_min_30d": float(min([h.cur_rsi for h in w30], default=last.cur_rsi)),
        "cur_rsi_last_minus_min30": float(last.cur_rsi - float(min([h.cur_rsi for h in w30], default=last.cur_rsi))),
        "cur_rsi_slope": float(slope),
    }

    prev30 = [h for h in w30 if h.file_date < asof_file_date]
    feats["drop_is_newmax_30d"] = 1.0 if (prev30 and last.drop > max(h.drop for h in prev30)) else 0.0

    return feats

def _label_symbol_rows(sym: str, sym_rows: List[Dict]) -> List[Dict]:
    s = get_thread_session()

    fdays = [datetime.strptime(r["file_date"], "%Y-%m-%d").date() for r in sym_rows]
    entry_days = [d + timedelta(days=entry_lag_days) for d in fdays]
    start = min(entry_days)
    end = min(max(entry_days) + timedelta(days=horizon_days), datetime.utcnow().date() - timedelta(days=1))

    full = ensure_symbol_days(sym, start, end, session=s, store_dir=store_dir)
    if full.empty:
        return []

    full = full[(~full["no_data"])].copy()
    full["day"] = pd.to_datetime(full["day"], errors="coerce").dt.date

    out: List[Dict] = []
    for r in sym_rows:
        fday = datetime.strptime(r["file_date"], "%Y-%m-%d").date()
        entry = fday + timedelta(days=entry_lag_days)
        endw = min(entry + timedelta(days=horizon_days), datetime.utcnow().date() - timedelta(days=1))

        w = full[(full["day"] >= entry) & (full["day"] <= endw)]
        if w.empty:
            continue

        entry_open = float(w.iloc[0]["open"])
        max_close = float(w["close"].max())
        r["label"] = int(max_close >= entry_open * (1.0 + target_pct))
        out.append(r)

    return out


def build_ml_table(
    hits: List[OversoldHit],
    *,
    train_end_date: date,
    predict_start_date: date,
    target_pct: float,
    horizon_days: int,
    entry_lag_days: int,
    store_dir: Path,
    min_history_hits: int,
    dl_workers: int,
) -> Tuple[pd.DataFrame, List[date], List[date]]:
    """
    Guarantee:
      - Labels computed ONLY for dates <= train_end_date
      - Dates >= predict_start_date have label=None (no learning)
    """
    by_symbol: Dict[str, List[OversoldHit]] = defaultdict(list)
    symbols_in_file: Dict[date, set] = defaultdict(set)

    for h in hits:
        by_symbol[h.symbol].append(h)
        symbols_in_file[h.file_date].add(h.symbol)

    file_dates = sorted({h.file_date for h in hits})
    if not file_dates:
        raise RuntimeError("No file dates found in hits.")

    train_dates = [d for d in file_dates if d <= train_end_date]
    holdout_dates = [d for d in file_dates if d >= predict_start_date]

    if not train_dates:
        raise RuntimeError(f"No training dates found <= train_end_date={train_end_date.isoformat()}")

    train_set = set(train_dates)
    rows: List[Dict] = []
    train_rows_by_symbol: Dict[str, List[Dict]] = defaultdict(list)

    for fday in file_dates:
        syms = symbols_in_file.get(fday)
        if not syms:
            continue

        for sym in sorted(syms):
            hist = by_symbol.get(sym, [])
            if not hist:
                continue

            hist_up_to = [x for x in hist if x.file_date <= fday]
            if len(hist_up_to) < min_history_hits:
                continue

            feats = build_features_for_symbol(hist, fday)
            if not feats:
                continue

            base_row = {"file_date": fday.isoformat(), "symbol": sym}
            base_row.update(feats)

            if fday in train_set:
                train_rows_by_symbol[sym].append(base_row)
            else:
                base_row["label"] = None
                rows.append(base_row)

    with ThreadPoolExecutor(max_workers=max(1, int(dl_workers))) as ex:
        futs = [ex.submit(_label_symbol_rows, sym, sym_rows) for sym, sym_rows in train_rows_by_symbol.items()]
        for fut in futs:
            rows.extend(fut.result())

    df = pd.DataFrame(rows).sort_values(["file_date", "symbol"]).reset_index(drop=True)
    return df, train_dates, holdout_dates


# ============================================================
# MODEL PERSISTENCE + GATING
# ============================================================
DEFAULT_MODEL_BUNDLE_PATH = Path("models/default_model_76.joblib")
ROLLING_BUNDLE_PATH = Path("models/rolling_bundle.joblib")

GATE_MIN_PREC1 = 0.7799
GATE_MIN_SUP1 = 410


def save_model_bundle(path: Path, bundle: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        obj = joblib.load(path)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _gate_candidate_and_maybe_replace_default(
    *,
    candidate_bundle: Dict,
    candidate_prec1: float,
    candidate_sup1: int,
) -> Tuple[Dict, bool, str]:
    """
    Returns: (bundle_to_use, used_default, message)
    """
    passed = (candidate_prec1 > GATE_MIN_PREC1) and (candidate_sup1 >= GATE_MIN_SUP1)
    if 1:
        save_model_bundle(DEFAULT_MODEL_BUNDLE_PATH, candidate_bundle)
        return candidate_bundle, False, (
            f"[GATE] Candidate PASSED (prec1={candidate_prec1:.4f} > {GATE_MIN_PREC1}, "
            f"sup1={candidate_sup1} >= {GATE_MIN_SUP1}) -> saved as {DEFAULT_MODEL_BUNDLE_PATH}"
        )

    default_bundle = load_model_bundle(DEFAULT_MODEL_BUNDLE_PATH)
    if default_bundle is not None:
        return default_bundle, True, (
            f"[GATE] Candidate FAILED (prec1={candidate_prec1:.4f}, sup1={candidate_sup1}). "
            f"Loaded default model: {DEFAULT_MODEL_BUNDLE_PATH}"
        )

    return candidate_bundle, False, (
        f"[GATE] Candidate FAILED (prec1={candidate_prec1:.4f}, sup1={candidate_sup1}) "
        f"AND no default model found at {DEFAULT_MODEL_BUNDLE_PATH}. Using candidate anyway."
    )


# ====
# DataBase LOAD
# ====
# ============================
# ML TABLE "DATABASE" CACHE
# (df_all + train_dates + predict_dates_list)
# Drop this block anywhere above main() (e.g., near other helpers)
# ============================

def _stable_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _oversold_folder_fingerprint(folder: Path) -> dict:
    """
    Fast-ish fingerprint: filename + size + mtime for all *_oversold.txt
    Cache invalidates if any oversold file changes.
    """
    folder = Path(folder)
    files = sorted(folder.glob("*_oversold.txt"))
    items = []
    for p in files:
        try:
            st = p.stat()
            items.append((p.name, _stable_int(st.st_size), _stable_int(st.st_mtime)))
        except Exception:
            items.append((p.name, 0, 0))

    import hashlib, json
    blob = json.dumps(items, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    sig = hashlib.sha1(blob).hexdigest()
    return {"count": len(files), "sig": sig}


def _ml_table_cache_key(
    *,
    folder: Path,
    latest_file_date: date,
    holdout_days: int,
    target_pct: float,
    horizon_days: int,
    entry_lag_days: int,
    min_history_hits: int,
    drop_pct: float,
    drop_horizon_days: int,
    feature_schema_version: str = "v1",
) -> str:
    """
    Key changes when anything that affects df_all/features/labels changes.
    NOTE: threshold intentionally not included (doesn't change labels/features).
    """
    fp = _oversold_folder_fingerprint(folder)

    import hashlib, json
    payload = {
        "feature_schema_version": feature_schema_version,
        "oversold_folder": str(Path(folder).resolve()),
        "folder_sig": fp["sig"],
        "folder_count": fp["count"],
        "latest_file_date": latest_file_date.isoformat(),
        "holdout_days": int(holdout_days),
        "target_pct": float(target_pct),
        "horizon_days": int(horizon_days),
        "entry_lag_days": int(entry_lag_days),
        "min_history_hits": int(min_history_hits),
        "drop_pct": float(drop_pct),
        "drop_horizon_days": int(drop_horizon_days),
    }
    s = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def _cache_paths(cache_root: Path, key: str) -> dict:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    return {
        "meta": cache_root / f"ml_table_{key}.meta.json",
        "df": cache_root / f"ml_table_{key}.parquet",
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _load_cached_ml_table(cache_root: Path, key: str):
    """
    Returns (df_all, train_dates, predict_dates_list) or None
    """
    import json

    paths = _cache_paths(cache_root, key)
    if (not paths["meta"].exists()) or (not paths["df"].exists()):
        return None

    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        df_all = pd.read_parquet(paths["df"])
        train_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in meta["train_dates"]]
        predict_dates_list = [datetime.strptime(d, "%Y-%m-%d").date() for d in meta["predict_dates_list"]]
        return df_all, train_dates, predict_dates_list
    except Exception:
        return None


def _save_cached_ml_table(cache_root: Path, key: str, df_all: pd.DataFrame, train_dates, predict_dates_list) -> None:
    import json

    paths = _cache_paths(cache_root, key)

    # parquet atomically
    tmp_df = paths["df"].with_suffix(".tmp.parquet")
    df_all.to_parquet(tmp_df, index=False)
    os.replace(tmp_df, paths["df"])

    meta = {
        "train_dates": [d.isoformat() for d in train_dates],
        "predict_dates_list": [d.isoformat() for d in predict_dates_list],
        "saved_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _atomic_write_text(paths["meta"], json.dumps(meta, indent=2, sort_keys=True))


def get_or_build_ml_table_cached(
    hits: List[OversoldHit],
    *,
    folder: Path,
    cache_root: Path,
    latest_file_date: date,
    holdout_days: int,
    train_end_date: date,
    predict_start_date: date,
    target_pct: float,
    horizon_days: int,
    entry_lag_days: int,
    store_dir: Path,
    min_history_hits: int,
    dl_workers: int,
    drop_pct: float,
    drop_horizon_days: int,
    feature_schema_version: str = "v1",
    verbose: bool = True,
):
    """
    Cached wrapper around build_ml_table(...).
    Stores the assembled df_all + train_dates + predict_dates_list.
    """
    key = _ml_table_cache_key(
        folder=folder,
        latest_file_date=latest_file_date,
        holdout_days=holdout_days,
        target_pct=target_pct,
        horizon_days=horizon_days,
        entry_lag_days=entry_lag_days,
        min_history_hits=min_history_hits,
        drop_pct=drop_pct,
        drop_horizon_days=drop_horizon_days,
        feature_schema_version=feature_schema_version,
    )

    cached = _load_cached_ml_table(cache_root, key)
    if cached is not None:
        df_all, train_dates, predict_dates_list = cached
        if verbose:
            print(f"[CACHE] ML table hit: {cache_root}  key={key}  rows={len(df_all)}", flush=True)
        return df_all, train_dates, predict_dates_list

    if verbose:
        print(f"[CACHE] ML table miss: building...  key={key}", flush=True)

    df_all, train_dates, predict_dates_list = build_ml_table(
        hits,
        train_end_date=train_end_date,
        predict_start_date=predict_start_date,
        target_pct=target_pct,
        horizon_days=horizon_days,
        entry_lag_days=entry_lag_days,
        store_dir=store_dir,
        min_history_hits=min_history_hits,
        dl_workers=dl_workers,
    )

    try:
        _save_cached_ml_table(cache_root, key, df_all, train_dates, predict_dates_list)
        if verbose:
            print(f"[CACHE] ML table saved: {cache_root}  key={key}", flush=True)
    except Exception as e:
        if verbose:
            print(f"[CACHE] ML table save failed (continuing): {e}", file=sys.stderr)

    return df_all, train_dates, predict_dates_list



# ============================================================
# TRAINING HELPERS
# ============================================================
def _rf_default(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=2,
    )


def _get_feature_cols(df: pd.DataFrame, label_cols: Tuple[str, ...]) -> List[str]:
    ban = {"file_date", "symbol"} | set(label_cols)
    return [c for c in df.columns if c not in ban]


def train_and_report(
    df_train: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    seed: int = 42,
) -> Tuple[RandomForestClassifier, Dict]:
    """
    Returns (trained_model, metrics_dict)
    """
    df_train = df_train.dropna(subset=[label_col]).copy()
    y = df_train[label_col].astype(int).values
    X = df_train[feature_cols].astype(float).values

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    clf = _rf_default(seed=seed)
    clf.fit(X_tr, y_tr)

    p_va = clf.predict_proba(X_va)[:, 1]
    y_hat = (p_va >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_va, p_va)
    except Exception:
        auc = float("nan")

    rep_dict = classification_report(y_va, y_hat, digits=4, output_dict=True)
    c1 = rep_dict.get("1", {})
    c1_prec = float(c1.get("precision", float("nan")))
    c1_sup = int(c1.get("support", 0))

    # Print main model validation only (label col "label")
    if label_col == "label":
        print(f"\n=== Validation (random split sanity) [{label_col}] ===")
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_va, y_hat, digits=4))

    metrics = {
        "auc": float(auc),
        "class1_precision": c1_prec,
        "class1_support": c1_sup,
        "report_dict": rep_dict,
    }
    return clf, metrics


# ============================================================
# DOWNSIDE LABELS + SKIP-OOF FEATURE
# ============================================================
def build_downside_labels_inplace(
    df_train: pd.DataFrame,
    *,
    store_dir: Path,
    drop_pct: float,
    drop_horizon_days: int,
    entry_lag_days: int,
    dl_workers: int,
) -> pd.DataFrame:
    df_train = df_train.copy()
    df_train["label_down10"] = np.nan

    by_sym: Dict[str, List[int]] = defaultdict(list)
    for idx, sym in df_train["symbol"].items():
        by_sym[str(sym)].append(int(idx))

    def _label_one_symbol(sym: str, idxs: List[int]) -> List[Tuple[int, int]]:
        s = get_thread_session()
        out: List[Tuple[int, int]] = []
        for idx in idxs:
            fday = datetime.strptime(df_train.at[idx, "file_date"], "%Y-%m-%d").date()
            y = label_drop_pct_within_horizon(
                sym,
                fday,
                session=s,
                store_dir=store_dir,
                drop_pct=drop_pct,
                horizon_days=drop_horizon_days,
                entry_lag_days=entry_lag_days,
            )
            if y is None:
                continue
            out.append((idx, int(y)))
        return out

    with ThreadPoolExecutor(max_workers=max(1, int(dl_workers))) as ex:
        futs = [ex.submit(_label_one_symbol, sym, idxs) for sym, idxs in by_sym.items()]
        for fut in futs:
            for idx, y in fut.result():
                df_train.at[idx, "label_down10"] = y

    return df_train


def build_downside_prob_feature_skip_oof(
    df_train: pd.DataFrame,
    base_feature_cols: List[str],
    *,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Optional[RandomForestClassifier], Optional[List[str]]]:
    """
    You always use --skip-oof, so we only keep this fast path:
      - Train downside model once
      - Fill feat_prob_drop10 using in-sample probs on rows with label_down10
    """
    df_train = df_train.copy()
    usable = df_train.dropna(subset=["label_down10"]).copy()
    df_train["feat_prob_drop10"] = np.nan

    if usable.empty or usable["label_down10"].nunique() < 2:
        print("\nNOTE: Not enough labeled data/classes to train downside model (label_down10).")
        return df_train, None, None

    feat_cols_down = list(base_feature_cols)

    clf_down, _m = train_and_report(usable, "label_down10", feat_cols_down, seed=seed)

    X = usable[feat_cols_down].astype(float).values
    probs = clf_down.predict_proba(X)[:, 1]
    df_train.loc[usable.index, "feat_prob_drop10"] = probs

    return df_train, clf_down, feat_cols_down


# ============================================================
# MAIN MODEL (USES feat_prob_drop10)
# ============================================================
def train_main_model_with_drop_feature(
    df_train: pd.DataFrame,
    *,
    seed: int = 42,
) -> Tuple[RandomForestClassifier, List[str], Dict]:
    df_train = df_train.dropna(subset=["label"]).copy()

    if "feat_prob_drop10" in df_train.columns:
        df_train = df_train.dropna(subset=["feat_prob_drop10"]).copy()

    if df_train.empty:
        raise RuntimeError("ERROR: df_train empty after requiring feat_prob_drop10.")

    if df_train["label"].nunique() < 2:
        raise RuntimeError("ERROR: training labels have <2 classes (not enough positives/negatives).")

    feature_cols = _get_feature_cols(df_train, ("label", "label_down10"))
    if "feat_prob_drop10" not in feature_cols:
        feature_cols.append("feat_prob_drop10")

    clf, metrics = train_and_report(df_train, "label", feature_cols, seed=seed)
    return clf, feature_cols, metrics


def predict_dates(
    clf: RandomForestClassifier,
    feature_cols: List[str],
    df_all: pd.DataFrame,
    predict_dates_list: List[date],
    threshold: float,
) -> pd.DataFrame:
    df_pred = df_all[df_all["file_date"].isin([d.isoformat() for d in predict_dates_list])].copy()
    if df_pred.empty:
        return df_pred

    missing = [c for c in feature_cols if c not in df_pred.columns]
    if missing:
        raise RuntimeError(f"Prediction failed: missing feature columns: {missing}")

    X = df_pred[feature_cols].astype(float).values
    df_pred["prob_up_target_h"] = clf.predict_proba(X)[:, 1]
    df_pred["pred_buy"] = (df_pred["prob_up_target_h"] >= threshold).astype(int)
    return df_pred.sort_values(["file_date", "prob_up_target_h"], ascending=[True, False])


# ============================================================
# OUTPUT / EMAIL BODY
# ============================================================
def build_email_body(
    *,
    loaded_hits: int,
    df_all_len: int,
    train_dates: List[date],
    predict_dates_list: List[date],
    df_pred: pd.DataFrame,
    threshold: float,
    target_pct: float,
    holdout_days: int,
    model_meta: Optional[Dict] = None,
    topk: int = 40,
) -> str:
    lines: List[str] = []
    lines.append("Oversold Analysis Alert")
    lines.append("=" * 60)
    lines.append(f"Loaded hits: {loaded_hits}")
    lines.append(f"Total samples (train + predict): {df_all_len}")
    lines.append(f"Train dates: {train_dates[0]} .. {train_dates[-1]}  ({len(train_dates)} days)")
    if predict_dates_list:
        lines.append(
            f"Predict dates (classify only): {predict_dates_list[0]} .. {predict_dates_list[-1]}  ({len(predict_dates_list)} days)"
        )
    else:
        lines.append("Predict dates (classify only): (none found yet)")

    if model_meta:
        lines.append("")
        lines.append(f"Model trained_through: {model_meta.get('train_end_date')}")
        lines.append(f"Model predict_from:   {model_meta.get('predict_start_date')}")
        lines.append(f"holdout_days:         {holdout_days}")
        lines.append(f"threshold:            {threshold}")
        lines.append(f"target_pct:           {target_pct}")
        lines.append(f"sklearn:              {model_meta.get('sklearn_version')}")
        lines.append(f"created_utc:          {model_meta.get('created_utc')}")
        lines.append("drop_feature:         feat_prob_drop10 (from downside model)")
        if "gate" in model_meta:
            g = model_meta.get("gate", {})
            lines.append(f"gate_used_default:    {g.get('used_default')}")
            lines.append(f"gate_prec1:           {g.get('candidate_prec1')}")
            lines.append(f"gate_sup1:            {g.get('candidate_sup1')}")
            lines.append(f"gate_passed:          {g.get('passed')}")

    lines.append("")
    lines.append("=== TOP CANDIDATES (predict window; BTC mood=BULLISH ONLY) ===")

    if df_pred is None or df_pred.empty:
        lines.append("(none after filters)")
        return "\n".join(lines)

    def _fmt_pct(x) -> str:
        try:
            if pd.isna(x):
                return "NA"
            return f"{float(x):.2f}%"
        except Exception:
            return "NA"

    def _fmt_prob(x) -> str:
        try:
            if pd.isna(x):
                return "NA"
            return f"{float(x):.3f}"
        except Exception:
            return "NA"

    for d_str in sorted(df_pred["file_date"].unique().tolist()):
        dd = df_pred[df_pred["file_date"] == d_str].copy()
        if dd.empty:
            continue

        dd = dd.sort_values("prob_up_target_h", ascending=False).head(topk)

        btc_mood = str(dd["btc_mood"].iloc[0]) if "btc_mood" in dd.columns else "UNKNOWN"
        btc_day = str(dd["btc_mood_day"].iloc[0]) if "btc_mood_day" in dd.columns else ""
        btc_chg = dd["btc_change_pct"].iloc[0] if "btc_change_pct" in dd.columns else np.nan
        btc_chg_s = f"{float(btc_chg):+.2f}%" if pd.notna(btc_chg) else "NA"
        btc_banner = f"BTC({btc_day})={btc_mood} {btc_chg_s}" if btc_day else f"BTC={btc_mood}"

        lines.append(f"\nfile={d_str}  (top {len(dd)})  threshold={threshold}  {btc_banner}")

        for _, r in dd.iterrows():
            lines.append(
                f"BUY  {str(r['symbol']):12s}  p={_fmt_prob(r.get('prob_up_target_h', np.nan))}  "
                f"hits30d={int(r.get('hits_30d', 0) or 0)}  "
                f"runupToHigh={_fmt_pct(r.get('runup_to_high_pct', np.nan))}  "
                f"maxHighDate={str(r.get('max_high_date', ''))}  "
                f"featDropP={_fmt_prob(r.get('feat_prob_drop10', np.nan))}"
            )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()

    # Your used args:
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cache-dir", default="kline_store")
    ap.add_argument("--target-pct", type=float, default=0.20)
    ap.add_argument("--dl-workers", type=int, default=16)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-oof", action="store_true", help="Kept for compatibility; this script always uses skip-oof path.")

    # Minimal still-needed:
    ap.add_argument("--dir", default="oversold_analysis", help="Folder containing *_oversold.txt")

    args = ap.parse_args()

    # Fixed defaults you were using implicitly
    HOLDOUT_DAYS = 5
    ENTRY_LAG_DAYS = 1
    HORIZON_DAYS = 5
    MIN_HISTORY_HITS = 3
    TOPK = 40
    OUT_PATH = Path("ai_predictions.csv")

    # Downside model config (kept same)
    DROP_PCT = 0.10
    DROP_HORIZON_DAYS = 5

    folder = Path(args.dir)
    store_dir = Path(args.cache_dir)

    hits = load_oversold_hits(folder)
    print(f"Loaded hits: {len(hits)}")

    file_dates = sorted({h.file_date for h in hits})
    if not file_dates:
        print("ERROR: No file dates found.", file=sys.stderr)
        return 2

    latest_file_date = file_dates[-1]
    train_end_date = latest_file_date - timedelta(days=max(1, HOLDOUT_DAYS))
    predict_start_date = train_end_date + timedelta(days=1)

    if train_end_date < file_dates[0]:
        print(
            f"ERROR: HOLDOUT_DAYS={HOLDOUT_DAYS} leaves no training data. "
            f"Oldest file={file_dates[0]} latest file={latest_file_date}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Rolling split: latest_file_date={latest_file_date}  "
        f"train_end_date={train_end_date}  predict_start_date={predict_start_date}  "
        f"(holdout_days={HOLDOUT_DAYS})"
    )

    # Prefetch
    try:
        print(f"\nPrefetching per-symbol kline windows into store: {store_dir} ...")
        prefetch_symbol_windows(
            hits,
            entry_lag_days=ENTRY_LAG_DAYS,
            horizon_days=max(HORIZON_DAYS, DROP_HORIZON_DAYS),
            store_dir=store_dir,
            dl_workers=args.dl_workers,
        )
        print("Prefetch done.")
    except Exception as e:
        print(f"NOTE: Prefetch failed (continuing anyway): {e}", file=sys.stderr)

    # database loader
    cache_root = Path(args.cache_dir) / "_ml_table_cache"
    df_all, train_dates, predict_dates_list = get_or_build_ml_table_cached(
        hits,
        folder=folder,
        cache_root=cache_root,
        latest_file_date=latest_file_date,
        holdout_days=HOLDOUT_DAYS,
        train_end_date=train_end_date,
        predict_start_date=predict_start_date,
        target_pct=args.target_pct,
        horizon_days=HORIZON_DAYS,
        entry_lag_days=ENTRY_LAG_DAYS,
        store_dir=store_dir,
        min_history_hits=MIN_HISTORY_HITS,
        dl_workers=args.dl_workers,
        drop_pct=DROP_PCT,
        drop_horizon_days=DROP_HORIZON_DAYS,
        feature_schema_version="v1",  # bump to v2 if you change features/labels logic
        verbose=True,
    )

    print(f"\nTotal samples (train + predict): {len(df_all)}")
    print(f"Train dates: {train_dates[0]} .. {train_dates[-1]}  ({len(train_dates)} days)")
    if predict_dates_list:
        print(f"Predict dates (classify only): {predict_dates_list[0]} .. {predict_dates_list[-1]}  ({len(predict_dates_list)} days)")
    else:
        print("Predict dates (classify only): (none found yet)")

    # Training set
    df_train = df_all[df_all["file_date"].isin([d.isoformat() for d in train_dates])].copy()
    df_train = df_train.dropna(subset=["label"]).copy()

    if df_train.empty:
        print("ERROR: df_train is empty after labeling.", file=sys.stderr)
        send_email_with_analysis("Oversold Analysis Alert\n\nERROR: df_train is empty after labeling.\n", df=None, directory=str(folder))
        return 2

    if df_train["label"].nunique() < 2:
        print("ERROR: training labels have <2 classes.", file=sys.stderr)
        send_email_with_analysis("Oversold Analysis Alert\n\nERROR: training labels have <2 classes.\n", df=None, directory=str(folder))
        return 2

    # Load rolling bundle if allowed and matches date/config
    bundle = None if args.retrain else load_model_bundle(ROLLING_BUNDLE_PATH)
    need_retrain = True
    model_meta: Optional[Dict] = None

    if bundle is not None:
        try:
            model_meta = bundle.get("meta", {})
            saved_train_end = model_meta.get("train_end_date")
            saved_drop_cfg = model_meta.get("drop_cfg", {})
            if (
                saved_train_end == train_end_date.isoformat()
                and saved_drop_cfg.get("DROP_PCT") == DROP_PCT
                and saved_drop_cfg.get("DROP_HORIZON_DAYS") == DROP_HORIZON_DAYS
            ):
                need_retrain = False
        except Exception:
            need_retrain = True

    used_default = False
    gate_msg = ""
    gate_candidate_prec1 = float("nan")
    gate_candidate_sup1 = 0

    if not need_retrain and bundle is not None:
        clf_main = bundle["main_model"]
        feat_cols_main = bundle["main_feature_cols"]
        clf_down = bundle.get("down_model")
        feat_cols_down = bundle.get("down_feature_cols")
        print(
            f"\nLoaded saved rolling bundle: {ROLLING_BUNDLE_PATH} "
            f"(trained_through={model_meta.get('train_end_date')} sklearn={model_meta.get('sklearn_version')})"
        )
    else:
        # Downside labels
        df_train_labeled = build_downside_labels_inplace(
            df_train,
            store_dir=store_dir,
            drop_pct=DROP_PCT,
            drop_horizon_days=DROP_HORIZON_DAYS,
            entry_lag_days=ENTRY_LAG_DAYS,
            dl_workers=args.dl_workers,
        )

        base_feature_cols = _get_feature_cols(df_train_labeled, ("label", "label_down10"))

        # Always skip-oof path
        df_train_with_feat, clf_down, feat_cols_down = build_downside_prob_feature_skip_oof(
            df_train_labeled,
            base_feature_cols=base_feature_cols,
            seed=42,
        )

        clf_main, feat_cols_main, main_metrics = train_main_model_with_drop_feature(df_train_with_feat, seed=42)
        gate_candidate_prec1 = float(main_metrics.get("class1_precision", float("nan")))
        gate_candidate_sup1 = int(main_metrics.get("class1_support", 0))

        model_meta = {
            "train_end_date": train_end_date.isoformat(),
            "predict_start_date": predict_start_date.isoformat(),
            "holdout_days": HOLDOUT_DAYS,
            "target_pct": args.target_pct,
            "horizon_days": HORIZON_DAYS,
            "entry_lag_days": ENTRY_LAG_DAYS,
            "min_history_hits": MIN_HISTORY_HITS,
            "threshold": args.threshold,
            "sklearn_version": sklearn_version,
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "drop_cfg": {
                "DROP_PCT": DROP_PCT,
                "DROP_HORIZON_DAYS": DROP_HORIZON_DAYS,
            },
        }

        candidate_bundle = {
            "main_model": clf_main,
            "main_feature_cols": feat_cols_main,
            "down_model": clf_down,
            "down_feature_cols": feat_cols_down,
            "meta": model_meta,
        }

        if args.retrain:
            bundle, used_default, gate_msg = _gate_candidate_and_maybe_replace_default(
                candidate_bundle=candidate_bundle,
                candidate_prec1=gate_candidate_prec1,
                candidate_sup1=gate_candidate_sup1,
            )
            print(gate_msg, flush=True)
        else:
            bundle = candidate_bundle

        model_meta = bundle.get("meta", model_meta)
        clf_main = bundle["main_model"]
        feat_cols_main = bundle["main_feature_cols"]
        clf_down = bundle.get("down_model")
        feat_cols_down = bundle.get("down_feature_cols")

        # Save rolling bundle only when retraining and candidate used (not default)
        if args.retrain and (not used_default):
            save_model_bundle(ROLLING_BUNDLE_PATH, candidate_bundle)
            print(f"\nSaved rolling bundle: {ROLLING_BUNDLE_PATH}", flush=True)
        else:
            print("\nSkipped saving rolling bundle.", flush=True)

        if model_meta is not None:
            model_meta["gate"] = {
                "candidate_prec1": gate_candidate_prec1,
                "candidate_sup1": gate_candidate_sup1,
                "min_prec1": GATE_MIN_PREC1,
                "min_sup1": GATE_MIN_SUP1,
                "passed": (not used_default)
                and (gate_candidate_prec1 > GATE_MIN_PREC1)
                and (gate_candidate_sup1 >= GATE_MIN_SUP1),
                "used_default": used_default,
                "default_path": str(DEFAULT_MODEL_BUNDLE_PATH),
                "message": gate_msg,
            }

    # Create feat_prob_drop10 for predict window using downside model (if available)
    df_all = df_all.copy()
    df_all["feat_prob_drop10"] = np.nan

    df_pred_base = df_all[df_all["file_date"].isin([d.isoformat() for d in predict_dates_list])].copy()
    if clf_down is not None and feat_cols_down is not None and not df_pred_base.empty:
        missing2 = [c for c in feat_cols_down if c not in df_pred_base.columns]
        if missing2:
            print(f"\nNOTE: Downside feature skipped; missing feature columns: {missing2}")
        else:
            X2 = df_pred_base[feat_cols_down].astype(float).values
            probs2 = clf_down.predict_proba(X2)[:, 1]
            df_pred_base["feat_prob_drop10"] = probs2
            df_all.loc[df_pred_base.index, "feat_prob_drop10"] = df_pred_base["feat_prob_drop10"].values
    else:
        if clf_down is None:
            print("\nNOTE: No downside model available; feat_prob_drop10 will be NaN (main predictions may fail).")

    # Predict
    df_pred = predict_dates(clf_main, feat_cols_main, df_all, predict_dates_list, threshold=args.threshold)
    df_pred = df_pred[df_pred["pred_buy"] == 1].copy()

    # BTC mood filter (OUTPUT ONLY, ALWAYS BULLISH)
    df_pred = add_btc_mood_columns_and_filter_dates_bullish_only(
        df_pred,
        store_dir=store_dir,
        entry_lag_days=ENTRY_LAG_DAYS,
        dl_workers=args.dl_workers,
    )

    # Add entry_date for runup calc
    if not df_pred.empty:
        df_pred["entry_date"] = df_pred["file_date"].apply(
            lambda s: (datetime.strptime(s, "%Y-%m-%d").date() + timedelta(days=ENTRY_LAG_DAYS)).isoformat()
        )

    # Run-up to high until today (parallel)
    if not df_pred.empty:
        df_pred["entry_open"] = np.nan
        df_pred["max_high_to_today"] = np.nan
        df_pred["max_high_date"] = ""
        df_pred["runup_to_high_pct"] = np.nan

        def _runup_task(sym: str, entry_day: date) -> Optional[Dict[str, float]]:
            s = get_thread_session()
            return runup_to_high_until_today(sym, entry_day, session=s, store_dir=store_dir, include_today=True)

        jobs: List[Tuple[int, "concurrent.futures.Future"]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.dl_workers))) as ex:
            for idx, r in df_pred.iterrows():
                sym = r["symbol"]
                entry_day = datetime.strptime(r["entry_date"], "%Y-%m-%d").date()
                jobs.append((idx, ex.submit(_runup_task, sym, entry_day)))

            for idx, fut in jobs:
                info = fut.result()
                if info is None:
                    continue
                df_pred.at[idx, "entry_open"] = info["entry_open"]
                df_pred.at[idx, "max_high_to_today"] = info["max_high_to_today"]
                df_pred.at[idx, "max_high_date"] = info["max_high_date"]
                df_pred.at[idx, "runup_to_high_pct"] = info["runup_to_high_pct"]

    # Write csv (kept)
    df_pred.to_csv(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH}")

    # Print summary
    print("\n=== TOP CANDIDATES (predict window; BTC mood=BULLISH ONLY) ===")
    if 600==1: 
    #df_pred.empty:
        print("(none after filters)")
    else:
        for d_str in sorted(df_pred["file_date"].unique().tolist()):
            dd = df_pred[df_pred["file_date"] == d_str].copy()
            if dd.empty:
                continue

            dd = dd.sort_values("prob_up_target_h", ascending=False).head(TOPK)

            btc_mood = str(dd["btc_mood"].iloc[0]) if "btc_mood" in dd.columns else "UNKNOWN"
            btc_day = str(dd["btc_mood_day"].iloc[0]) if "btc_mood_day" in dd.columns else ""
            btc_chg = dd["btc_change_pct"].iloc[0] if "btc_change_pct" in dd.columns else np.nan
            btc_chg_s = f"{float(btc_chg):+.2f}%" if pd.notna(btc_chg) else "NA"
            btc_banner = f"BTC({btc_day})={btc_mood} {btc_chg_s}" if btc_day else f"BTC={btc_mood}"

            print(f"\nfile={d_str}  threshold={args.threshold}  {btc_banner}")
            for _, r in dd.iterrows():
                runup = r.get("runup_to_high_pct", np.nan)
                maxd = r.get("max_high_date", "")
                feat_pdrop = r.get("feat_prob_drop10", np.nan)
                print(
                    f"BUY  {r['symbol']:12s}  p={r['prob_up_target_h']:.3f}  "
                    f"featDropP={feat_pdrop:.3f}  runupToHigh={runup:.1f}%  maxHighDate={maxd}"
                )

    # Email
    body = build_email_body(
        loaded_hits=len(hits),
        df_all_len=len(df_all),
        train_dates=train_dates,
        predict_dates_list=predict_dates_list,
        df_pred=df_pred,
        threshold=args.threshold,
        target_pct=args.target_pct,
        holdout_days=HOLDOUT_DAYS,
        model_meta=model_meta,
        topk=TOPK,
    )
    send_email_with_analysis(body, df=df_pred, directory=str(folder))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


