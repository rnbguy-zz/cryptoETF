#!/usr/bin/env python3
"""
rank_oversold.py  (AI-ready-to-buy classifier + SIDEWAYS FILTER + MODEL PERSISTENCE)
+ Rolling split: trains only on files older than N days (default 5), predicts on the newest N days
+ Adds "run-up to highest HIGH since entry_date until today (UTC)" to output
+ ALWAYS emails the output

CHANGE YOU ASKED FOR (IMPORTANT):
- The "drop prediction" is now a FIRST-STAGE model.
- Its predicted probability (OOF on training rows, normal preds on predict window)
  is added as a FEATURE used to train/predict the main "hits target" model.

NEW CHANGE YOU ASKED FOR (MODEL GATING):
- Default holdout-days is now 5.
- When --retrain is used, script trains a candidate model and checks MAIN model validation:
    precision(class 1) > GATE_MIN_PREC1 AND support(class 1) >= GATE_MIN_SUP1
  If TRUE: overwrite default model bundle at models/default_model_76.joblib
  Else: load and use models/default_model_76.joblib (if exists), instead of the candidate.

NEW CHANGE YOU ASKED FOR (PARALLEL DOWNLOADS + GLOBAL COOLDOWN):
- Adds --dl-workers (default 16) and runs Binance label/runup/sideways downloads in parallel threads
- If any thread hits Binance rate-limit (HTTP 429/418) or transient errors, it triggers a global
  1-second cooldown that all threads honor before making further requests.

NEW CHANGE (YOUR LATEST ASK): MUCH MORE EFFICIENT KLINE STORAGE
- Replaces per-day JSON cache with ONE file per symbol: kline_store/{SYMBOL}_1d.parquet
- Incrementally extends symbol history (head/tail) when needed and writes back (deduped by day)
- Uses tombstone rows (no_data=True) so missing days are not refetched every run
- Adds optional prefetch per symbol for the run (fast: 1 load + 0-2 fetches per symbol),
  then all label/runup/sideways logic slices in-memory dataframes.

Notes:
- --cache-dir now points to the SYMBOL STORE folder (default: kline_store)
- Requires: pandas + pyarrow (or fastparquet) for parquet support.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import smtplib
import joblib
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import __version__ as sklearn_version

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


# ============================================================
# EMAIL CONFIG
# ============================================================
recipient_email = "mina.moussa@hotmail.com"
sender_email = "minamoussa903@gmail.com"
sender_password = "qhvi syra bbad gylu"


def send_email_with_analysis(body: str, df: Optional[pd.DataFrame] = None, directory: str = ".") -> None:
    """
    Always attempts to send email. Attaches filtered CSV if df has rows.
    """
    if not sender_password:
        print("Email not sent: sender_password is empty.", file=sys.stderr)
        return

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
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
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("Email sent.")
    except Exception as e:
        print("Error sending email:", e, file=sys.stderr)


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

BINANCE_BASE = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"


# ============================================================
# BINANCE GLOBAL COOLDOWN (shared across ALL threads)
# ============================================================
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


# -----------------------------
# Thread-local requests.Session
# -----------------------------
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
# DEFAULT MODEL GATING CONFIG
# ============================================================
DEFAULT_MODEL_BUNDLE_PATH = Path("models/default_model_76.joblib")
GATE_MIN_PREC1 = 0.7799
GATE_MIN_SUP1 = 410


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


# -----------------------------
# Model persistence
# -----------------------------
def save_model_bundle(path: Path, bundle: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        obj = joblib.load(path)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


# ============================================================
# NEW: Per-symbol Parquet store for klines + tombstones
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

    # ---- 1) Make "day" a clean python date column ----
    if "day" in df.columns:
        s = df["day"]

        if np.issubdtype(s.dtype, np.datetime64):
            df["day"] = pd.to_datetime(s, utc=False).dt.date
        else:
            # Force string parsing (prevents the "Can't infer object conversion type" bug)
            df["day"] = pd.to_datetime(
                s.astype(str), format="%Y-%m-%d", errors="coerce"
            ).dt.date

        # Drop any rows we still couldn't parse
        df = df[df["day"].notna()].copy()

        # Keep as python-date objects (safe for parquet + slicing)
        df["day"] = df["day"].astype(object)

    # ---- 2) Ensure all expected columns exist ----
    for c in ["open_time", "open", "high", "low", "close", "close_time", "no_data"]:
        if c not in df.columns:
            df[c] = 0 if c == "no_data" else np.nan

    # ---- 3) Make no_data a *clean boolean column* (this fixes your crash) ----
    df["no_data"] = pd.to_numeric(df["no_data"], errors="coerce").fillna(0).astype("int8").astype(bool)
    return df


def _store_path(store_dir: Path, symbol: str) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / f"{symbol}_1d.parquet"


def _read_symbol_store(store_dir: Path, symbol: str) -> pd.DataFrame:
    p = _store_path(store_dir, symbol)
    if not p.exists():
        return _normalize_store_df(pd.DataFrame())
    try:
        df = pd.read_parquet(p, engine="fastparquet")
        return _normalize_store_df(df)
    except Exception:
        # if corrupted, delete and start fresh
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

    # IMPORTANT: avoid "Can't infer object conversion type" on parquet writes
    df = df.sort_values("day").drop_duplicates("day", keep="last").copy()
    df["day"] = df["day"].astype(str)  # <- force stable storage type
    df["no_data"] = df["no_data"].astype(bool)

    tmp = p.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)



def _binance_fetch_1d_range(symbol: str, start_day: date, end_day: date, session: requests.Session) -> pd.DataFrame:
    """
    Fetch 1d klines inclusive start_day..end_day (UTC).
    Automatically chunks requests if range > 1000 days (Binance limit).
    Returns rows with day as python date.
    """
    if end_day < start_day:
        return pd.DataFrame()

    all_rows: List[Dict] = []

    # Chunk up to 900 days to be safe (limit is 1000)
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
    df = _normalize_store_df(df)
    return df


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
        # If already memoized and covers needed range, return it
        with _SYMBOL_MEMO_LOCK:
            memo = _SYMBOL_MEMO.get(symbol)
        if memo is not None and not memo.empty:
            have_min = memo["day"].min()
            have_max = memo["day"].max()
            if need_start >= have_min and need_end <= have_max:
                return memo

        # Read from disk
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

        # Extend backwards if needed
        if need_start < have_min:
            back = _binance_fetch_1d_range(symbol, need_start, have_min - timedelta(days=1), session)
            back = _fill_tombstones(need_start, have_min - timedelta(days=1), back)
            if not back.empty:
                parts.append(back)

        # Extend forwards if needed
        if need_end > have_max:
            fwd = _binance_fetch_1d_range(symbol, have_max + timedelta(days=1), need_end, session)
            fwd = _fill_tombstones(have_max + timedelta(days=1), need_end, fwd)
            if not fwd.empty:
                parts.append(fwd)

        if len(parts) > 1:
            merged = pd.concat(parts, ignore_index=True)
        else:
            merged = parts[0]

        # Also ensure tombstones inside the requested window if there are gaps (rare but possible)
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
    Columns: day (YYYY-MM-DD string), open, high, low, close, open_time, close_time
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
    out = out.sort_values("day").reset_index(drop=True)
    return out


def prefetch_symbol_windows(
    hits: List[OversoldHit],
    *,
    entry_lag_days: int,
    sideways_days: int,
    horizon_days: int,
    store_dir: Path,
    dl_workers: int,
) -> None:
    """
    Prefetches per symbol: [min_needed .. last_closed_utc_day]
    This makes later label/runup/sideways mostly pure slicing.
    """
    if not hits:
        return

    last_closed = datetime.utcnow().date() - timedelta(days=1)
    by_sym: Dict[str, List[OversoldHit]] = defaultdict(list)
    for h in hits:
        by_sym[h.symbol].append(h)

    need_ranges: Dict[str, Tuple[date, date]] = {}
    for sym, hs in by_sym.items():
        # compute earliest entry day and pre-entry window start
        entry_days = [x.file_date + timedelta(days=entry_lag_days) for x in hs]
        if not entry_days:
            continue
        min_entry = min(entry_days)
        min_needed = min_entry - timedelta(days=max(0, int(sideways_days)))
        max_needed = last_closed  # runup uses entry..yesterday
        # also ensure horizon doesn't exceed last_closed anyway; max_needed already covers it
        need_ranges[sym] = (min_needed, max_needed)

    def _prefetch_one(sym: str, d0: date, d1: date) -> None:
        s = get_thread_session()
        # quiet: no prints here; you can add prints if you want
        ensure_symbol_days(sym, d0, d1, session=s, store_dir=store_dir)

    with ThreadPoolExecutor(max_workers=max(1, int(dl_workers))) as ex:
        futs = []
        for sym, (d0, d1) in need_ranges.items():
            futs.append(ex.submit(_prefetch_one, sym, d0, d1))
        for f in futs:
            _ = f.result()


# -----------------------------
# Labels / runup / sideways now use per-symbol store
# -----------------------------
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
    LAST_CLOSED_UTC_DAY = datetime.utcnow().date() - timedelta(days=1)
    entry_day = file_day + timedelta(days=entry_lag_days)
    start_day = entry_day
    end_day = min(entry_day + timedelta(days=horizon_days), LAST_CLOSED_UTC_DAY)
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
    LAST_CLOSED_UTC_DAY = datetime.utcnow().date() - timedelta(days=1)
    entry_day = file_day + timedelta(days=entry_lag_days)
    start_day = entry_day
    end_day = min(entry_day + timedelta(days=horizon_days), LAST_CLOSED_UTC_DAY)
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
) -> Optional[Dict[str, float]]:
    """
    Computes run-up from entry open to the highest HIGH observed from entry_day to today (UTC, last closed day).
    Returns:
      entry_open, max_high_to_today, max_high_date, runup_to_high_pct
    """
    today_utc = datetime.utcnow().date() - timedelta(days=1)
    if entry_day > today_utc:
        return None

    df = get_klines_window(symbol, entry_day, today_utc, session=session, store_dir=store_dir)
    if df.empty:
        return None

    entry_open = float(df.iloc[0]["open"])
    highs = df["high"].astype(float)
    i = int(highs.idxmax())
    max_high = float(highs.loc[i])
    max_high_date = str(df.loc[i, "day"]) if "day" in df.columns else ""
    runup_pct = ((max_high / max(entry_open, 1e-12)) - 1.0) * 100.0

    return {
        "entry_open": entry_open,
        "max_high_to_today": max_high,
        "max_high_date": max_high_date,
        "runup_to_high_pct": runup_pct,
    }


def sideways_stats_pre_entry(
    symbol: str,
    entry_day: date,
    *,
    session: requests.Session,
    store_dir: Path,
    sideways_days: int,
) -> Optional[Dict[str, float]]:
    """
    Returns stats for the N days BEFORE entry_day:
      window: [entry_day - sideways_days, entry_day - 1]
    """
    if sideways_days <= 0:
        return None

    start = entry_day - timedelta(days=sideways_days)
    end = entry_day - timedelta(days=1)

    df = get_klines_window(symbol, start, end, session=session, store_dir=store_dir)
    if df.empty or len(df) < max(2, sideways_days - 1):
        return None

    first_open = float(df.iloc[0]["open"])
    max_high = float(df["high"].max())
    min_low = float(df["low"].min())
    range_pct = (max_high - min_low) / max(first_open, 1e-12)

    closes = df["close"].astype(float).values
    rets = np.abs(np.diff(closes) / np.maximum(closes[:-1], 1e-12))
    max_abs_close_change = float(np.max(rets)) if len(rets) else 0.0

    return {
        "sideways_range_pct": float(range_pct),
        "sideways_max_abs_close_change_pct": float(max_abs_close_change),
        "sideways_days": float(sideways_days),
    }


def passes_sideways_filter(
    stats: Optional[Dict[str, float]],
    *,
    max_range_pct: float,
    max_abs_close_change_pct: float,
) -> bool:
    if stats is None:
        return False
    return (
        stats["sideways_range_pct"] <= max_range_pct
        and stats["sideways_max_abs_close_change_pct"] <= max_abs_close_change_pct
    )


# -----------------------------
# Features (history-only)
# -----------------------------
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
        "cur_rsi_last_minus_min30": float(
            last.cur_rsi - float(min([h.cur_rsi for h in w30], default=last.cur_rsi))
        ),
        "cur_rsi_slope": float(slope),
    }

    prev30 = [h for h in w30 if h.file_date < asof_file_date]
    feats["drop_is_newmax_30d"] = 1.0 if (prev30 and last.drop > max(h.drop for h in prev30)) else 0.0

    return feats


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


def build_ml_table(
    hits: List[OversoldHit],
    *,
    train_end_date: date,
    predict_start_date: date,
    target_pct: float,
    horizon_days: int,
    entry_lag_days: int,
    store_dir: Path,
    min_history_hits: int = 3,
    dl_workers: int = 16,
) -> Tuple[pd.DataFrame, List[date], List[date]]:
    """
    Key guarantee:
      - Labels are computed ONLY for dates <= train_end_date
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

    rows: List[Dict] = []
    label_jobs: List[Tuple[Dict, "concurrent.futures.Future"]] = []

    def _compute_target_label(sym: str, fday: date) -> Optional[int]:
        s = get_thread_session()
        return label_plus_target_within_horizon(
            sym,
            fday,
            session=s,
            store_dir=store_dir,
            target_pct=target_pct,
            horizon_days=horizon_days,
            entry_lag_days=entry_lag_days,
        )

    with ThreadPoolExecutor(max_workers=max(1, int(dl_workers))) as ex:
        for fday in file_dates:
            for sym in sorted(symbols_in_file[fday]):
                hist = by_symbol[sym]
                hist_up_to = [x for x in hist if x.file_date <= fday]
                if len(hist_up_to) < min_history_hits:
                    continue

                feats = build_features_for_symbol(hist, fday)
                if not feats:
                    continue

                base_row = {"file_date": fday.isoformat(), "symbol": sym}
                base_row.update(feats)

                if fday in train_dates:
                    fut = ex.submit(_compute_target_label, sym, fday)
                    label_jobs.append((base_row, fut))
                else:
                    base_row["label"] = None
                    rows.append(base_row)

        for base_row, fut in label_jobs:
            y = fut.result()
            if y is None:
                continue
            base_row["label"] = int(y)
            rows.append(base_row)

    df = pd.DataFrame(rows)
    return df, train_dates, holdout_dates


# ============================================================
# TRAINING HELPERS (generic)
# ============================================================
from sklearn.ensemble import ExtraTreesClassifier   # <-- add this import




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

    metrics_dict includes:
      - auc
      - class1_precision
      - class1_support
      - report_dict (full classification report)
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

    if label_col and label_col != "label_down10":
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
# NEW: Downside model -> OOF feature for main model
# ============================================================
def build_downside_labels_inplace(
    df_train: pd.DataFrame,
    *,
    store_dir: Path,
    drop_pct: float,
    drop_horizon_days: int,
    entry_lag_days: int,
    dl_workers: int = 16,
) -> pd.DataFrame:
    df_train = df_train.copy()
    df_train["label_down10"] = np.nan

    def _compute(sym: str, fday: date) -> Optional[int]:
        s = get_thread_session()
        return label_drop_pct_within_horizon(
            sym,
            fday,
            session=s,
            store_dir=store_dir,
            drop_pct=drop_pct,
            horizon_days=drop_horizon_days,
            entry_lag_days=entry_lag_days,
        )

    jobs: List[Tuple[int, "concurrent.futures.Future"]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(dl_workers))) as ex:
        for idx, r in df_train.iterrows():
            sym = r["symbol"]
            fday = datetime.strptime(r["file_date"], "%Y-%m-%d").date()
            jobs.append((idx, ex.submit(_compute, sym, fday)))

        for idx, fut in jobs:
            y2 = fut.result()
            if y2 is None:
                continue
            df_train.at[idx, "label_down10"] = int(y2)

    return df_train


def build_oof_downside_prob_feature(
    df_train: pd.DataFrame,
    base_feature_cols: List[str],
    *,
    seed: int = 42,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Optional[RandomForestClassifier], Optional[List[str]]]:
    df_train = df_train.copy()

    usable = df_train.dropna(subset=["label_down10"]).copy()
    if usable.empty or usable["label_down10"].nunique() < 2:
        print("\nNOTE: Not enough labeled data/classes to train downside model (label_down10).")
        df_train["feat_prob_drop10"] = np.nan
        return df_train, None, None

    feat_cols_down = list(base_feature_cols)
    df_train["feat_prob_drop10"] = np.nan

    X = usable[feat_cols_down].astype(float).values
    y = usable["label_down10"].astype(int).values

    skf = StratifiedKFold(n_splits=min(n_splits, int(np.bincount(y).min())), shuffle=True, random_state=seed)
    if skf.get_n_splits() < 2:
        print("\nNOTE: Not enough minority samples for CV; training downside model without OOF feature.")
        df_train["feat_prob_drop10"] = np.nan
        final_clf, _m = train_and_report(usable, "label_down10", feat_cols_down, seed=seed)
        probs = final_clf.predict_proba(usable[feat_cols_down].astype(float).values)[:, 1]
        df_train.loc[usable.index, "feat_prob_drop10"] = probs
        return df_train, final_clf, feat_cols_down

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        clf = _rf_default(seed=seed + fold)
        clf.fit(X[tr_idx], y[tr_idx])
        p_va = clf.predict_proba(X[va_idx])[:, 1]
        va_rows = usable.iloc[va_idx].index
        df_train.loc[va_rows, "feat_prob_drop10"] = p_va

    final_clf, _m = train_and_report(usable, "label_down10", feat_cols_down, seed=seed)
    return df_train, final_clf, feat_cols_down


# ============================================================
# MAIN model predictions (uses downside prob as a feature)
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
        raise RuntimeError("ERROR: df_train is empty after requiring feat_prob_drop10. Not enough downside labels/preds.")

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


def build_email_body(
    *,
    loaded_hits: int,
    df_all_len: int,
    train_dates: List[date],
    predict_dates_list: List[date],
    df_pred: pd.DataFrame,
    args: argparse.Namespace,
    model_meta: Optional[Dict] = None,
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
        lines.append(f"holdout_days:         {model_meta.get('holdout_days')}")
        lines.append(f"sklearn:              {model_meta.get('sklearn_version')}")
        lines.append(f"created_utc:          {model_meta.get('created_utc')}")
        lines.append(f"drop_feature:         feat_prob_drop10 (from downside model)")
        if "gate" in model_meta:
            g = model_meta.get("gate", {})
            lines.append(f"gate_used_default:    {g.get('used_default')}")
            lines.append(f"gate_prec1:           {g.get('candidate_prec1')}")
            lines.append(f"gate_sup1:            {g.get('candidate_sup1')}")
            lines.append(f"gate_passed:          {g.get('passed')}")

    lines.append("")
    lines.append("=== TOP CANDIDATES (predict window) ===")

    if df_pred.empty:
        lines.append("(none after filters)")
        return "\n".join(lines)

    unique_dates = sorted(df_pred["file_date"].unique().tolist())
    for d_str in unique_dates:
        dd = df_pred[df_pred["file_date"] == d_str].copy()
        if dd.empty:
            continue

        dd = dd.sort_values("prob_up_target_h", ascending=False).head(args.topk)
        d = datetime.strptime(d_str, "%Y-%m-%d").date()
        entry_day = d + timedelta(days=args.entry_lag_days)

        if args.sideways_only:
            lines.append(
                f"\nfile={d.isoformat()}  entry={entry_day.isoformat()}  "
                f"(top {len(dd)})  threshold={args.threshold}  "
                f"sideways={args.sideways_days}d range<={args.sideways_max_range_pct:.2f} "
                f"maxAbsClose<={args.sideways_max_abs_change_pct:.2f}"
            )
        else:
            lines.append(
                f"\nfile={d.isoformat()}  entry={entry_day.isoformat()}  (top {len(dd)})  threshold={args.threshold}"
            )

        for _, r in dd.iterrows():
            runup = r.get("runup_to_high_pct", np.nan)
            maxd = r.get("max_high_date", "")

            dropflag = int(r.get("pred_drop_10_next5d", 0))
            drop_txt = "YES" if dropflag == 1 else "NO"

            feat_pdrop = r.get("feat_prob_drop10", np.nan)

            lines.append(
                f"BUY  {r['symbol']:12s}  p={r['prob_up_target_h']:.3f}  "
                f"hits={int(r['hits_30d'])}"
                f"  runupToHigh={runup:.1f}% maxHighDate={maxd}"
                f"  featDropP={feat_pdrop:.3f}"
                f"  drop10={drop_txt}"
            )

    return "\n".join(lines)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="oversold_analysis", help="Folder containing *_oversold.txt")

    ap.add_argument("--target-pct", type=float, default=0.20, help="Target gain threshold (e.g. 0.15 = +15%)")
    ap.add_argument("--horizon-days", type=int, default=5, help="Horizon window in days after entry day")
    ap.add_argument(
        "--entry-lag-days",
        type=int,
        default=1,
        help="ENTRY_DAY = file_date + entry_lag_days (filenames 1 day behind).",
    )
    ap.add_argument("--min-history-hits", type=int, default=3, help="Min oversold occurrences before sample is used")

    # IMPORTANT: this is now a PARQUET STORE DIR (one file per symbol)
    ap.add_argument("--cache-dir", default="kline_store", help="Symbol kline store folder (parquet per symbol)")

    ap.add_argument("--out", default="ai_predictions.csv", help="Output CSV path for predictions")
    ap.add_argument("--threshold", type=float, default=0.55, help="BUY threshold on probability")
    ap.add_argument("--topk", type=int, default=40, help="Top-K predictions per day to print")

    ap.add_argument(
        "--holdout-days",
        type=int,
        default=5,
        help="Number of most recent file-days to NOT train on; predict only (rolling split).",
    )

    ap.add_argument(
        "--dl-workers",
        type=int,
        default=16,
        help="Parallel workers for Binance downloads/label/runup/sideways (network I/O bound).",
    )

    ap.add_argument(
        "--sideways-only",
        action="store_true",
        help="If set, ONLY return BUY symbols that pass sideways filter (pre-entry consolidation).",
    )
    ap.add_argument("--sideways-days", type=int, default=3, help="How many days BEFORE entry must be sideways.")
    ap.add_argument("--sideways-max-range-pct", type=float, default=0.08, help="Max overall range over sideways window.")
    ap.add_argument(
        "--sideways-max-abs-change-pct",
        type=float,
        default=0.06,
        help="Max single-day abs close-to-close change.",
    )

    ap.add_argument("--model-path", default="models/default_model_76.joblib", help="Where to save/load rolling bundle")
    ap.add_argument("--retrain", action="store_true", help="Force retrain even if a saved model exists")

    args = ap.parse_args()

    DROP_PCT = 0.10
    DROP_HORIZON_DAYS = 5
    DROP_THRESHOLD = 0.50

    folder = Path(args.dir)
    store_dir = Path(args.cache_dir)

    hits = load_oversold_hits(folder)
    print(f"Loaded hits: {len(hits)}")

    file_dates = sorted({h.file_date for h in hits})
    if not file_dates:
        print("ERROR: No file dates found.", file=sys.stderr)
        return 2

    latest_file_date = file_dates[-1]
    holdout_days = max(1, int(args.holdout_days))

    train_end_date = latest_file_date - timedelta(days=holdout_days)
    predict_start_date = train_end_date + timedelta(days=1)

    if train_end_date < file_dates[0]:
        print(
            f"ERROR: --holdout-days={holdout_days} leaves no training data. "
            f"Oldest file={file_dates[0]} latest file={latest_file_date}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Rolling split: latest_file_date={latest_file_date}  "
        f"train_end_date={train_end_date}  predict_start_date={predict_start_date}  "
        f"(holdout_days={holdout_days})"
    )

    # ---------------------------------------------------------------
    # NEW: Prefetch symbol histories once per run (huge speedup)
    # ---------------------------------------------------------------
    try:
        print(f"\nPrefetching per-symbol kline windows into store: {store_dir} ...")
        prefetch_symbol_windows(
            hits,
            entry_lag_days=args.entry_lag_days,
            sideways_days=args.sideways_days if args.sideways_only else 0,
            horizon_days=max(args.horizon_days, DROP_HORIZON_DAYS),
            store_dir=store_dir,
            dl_workers=args.dl_workers,
        )
        print("Prefetch done.")
    except Exception as e:
        print(f"NOTE: Prefetch failed (continuing anyway): {e}", file=sys.stderr)

    df_all, train_dates, predict_dates_list = build_ml_table(
        hits,
        train_end_date=train_end_date,
        predict_start_date=predict_start_date,
        target_pct=args.target_pct,
        horizon_days=args.horizon_days,
        entry_lag_days=args.entry_lag_days,
        store_dir=store_dir,
        min_history_hits=args.min_history_hits,
        dl_workers=args.dl_workers,
    )

    print(f"\nTotal samples (train + predict): {len(df_all)}")
    print(f"Train dates: {train_dates[0]} .. {train_dates[-1]}  ({len(train_dates)} days)")
    if predict_dates_list:
        print(
            f"Predict dates (classify only): {predict_dates_list[0]} .. {predict_dates_list[-1]}  ({len(predict_dates_list)} days)"
        )
    else:
        print("Predict dates (classify only): (none found yet)")

    df_train = df_all[df_all["file_date"].isin([d.isoformat() for d in train_dates])].copy()
    df_train = df_train.dropna(subset=["label"]).copy()

    if df_train.empty:
        print("ERROR: df_train is empty after labeling. Check training window and data files.", file=sys.stderr)
        body = "Oversold Analysis Alert\n\nERROR: df_train is empty after labeling.\n"
        send_email_with_analysis(body, df=None, directory=str(folder))
        return 2

    if df_train["label"].nunique() < 2:
        print("ERROR: training labels have <2 classes (not enough positives/negatives).", file=sys.stderr)
        body = "Oversold Analysis Alert\n\nERROR: training labels have <2 classes (not enough positives/negatives).\n"
        send_email_with_analysis(body, df=None, directory=str(folder))
        return 2

    bundle_path = Path(args.model_path)
    bundle = None if args.retrain else load_model_bundle(bundle_path)

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
            f"\nLoaded saved rolling bundle: {bundle_path} "
            f"(trained_through={model_meta.get('train_end_date')} sklearn={model_meta.get('sklearn_version')})"
        )
    else:
        df_train_labeled = build_downside_labels_inplace(
            df_train,
            store_dir=store_dir,
            drop_pct=DROP_PCT,
            drop_horizon_days=DROP_HORIZON_DAYS,
            entry_lag_days=args.entry_lag_days,
            dl_workers=args.dl_workers,
        )

        base_feature_cols = _get_feature_cols(df_train_labeled, ("label", "label_down10"))

        df_train_with_feat, clf_down, feat_cols_down = build_oof_downside_prob_feature(
            df_train_labeled,
            base_feature_cols=base_feature_cols,
            seed=42,
            n_splits=5,
        )

        clf_main, feat_cols_main, main_metrics = train_main_model_with_drop_feature(df_train_with_feat, seed=42)
        gate_candidate_prec1 = float(main_metrics.get("class1_precision", float("nan")))
        gate_candidate_sup1 = int(main_metrics.get("class1_support", 0))

        model_meta = {
            "train_end_date": train_end_date.isoformat(),
            "predict_start_date": predict_start_date.isoformat(),
            "holdout_days": holdout_days,
            "target_pct": args.target_pct,
            "horizon_days": args.horizon_days,
            "entry_lag_days": args.entry_lag_days,
            "min_history_hits": args.min_history_hits,
            "sklearn_version": sklearn_version,
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "drop_cfg": {
                "DROP_PCT": DROP_PCT,
                "DROP_HORIZON_DAYS": DROP_HORIZON_DAYS,
                "DROP_THRESHOLD": DROP_THRESHOLD,
            },
        }

        candidate_bundle = {
            "main_model": clf_main,
            "main_feature_cols": feat_cols_main,
            "down_model": clf_down,
            "down_feature_cols": feat_cols_down,
            "meta": model_meta,
        }
        used_default = False

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

        if args.retrain and (not used_default):
            save_model_bundle(bundle_path, candidate_bundle)
            print(f"\nSaved rolling bundle (candidate used): {bundle_path}", flush=True)
        else:
            print("\nSkipped saving rolling bundle (using default or not retraining).", flush=True)

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

    # ------------------------------------------------------------------
    # Create feat_prob_drop10 for predict window (>= predict_start_date)
    # ------------------------------------------------------------------
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
            print("\nNOTE: No downside model available; main model feature feat_prob_drop10 will be NaN (predictions may fail).")

    # ------------------------------------------------------------------
    # Predict ONLY in predict window using MAIN model
    # ------------------------------------------------------------------
    df_pred = predict_dates(clf_main, feat_cols_main, df_all, predict_dates_list, threshold=args.threshold)
    df_pred = df_pred[df_pred["pred_buy"] == 1].copy()

    if not df_pred.empty:
        df_pred["entry_date"] = df_pred["file_date"].apply(
            lambda s: (datetime.strptime(s, "%Y-%m-%d").date() + timedelta(days=args.entry_lag_days)).isoformat()
        )

    # Downside prediction columns for display
    if clf_down is not None and feat_cols_down is not None and not df_pred.empty:
        missing2 = [c for c in feat_cols_down if c not in df_pred.columns]
        if missing2:
            df_pred["prob_drop_10_next5d"] = np.nan
            df_pred["pred_drop_10_next5d"] = 0
        else:
            X2 = df_pred[feat_cols_down].astype(float).values
            df_pred["prob_drop_10_next5d"] = clf_down.predict_proba(X2)[:, 1]
            df_pred["pred_drop_10_next5d"] = (df_pred["prob_drop_10_next5d"] >= DROP_THRESHOLD).astype(int)
    else:
        if not df_pred.empty:
            df_pred["prob_drop_10_next5d"] = np.nan
            df_pred["pred_drop_10_next5d"] = 0

    # Run-up to high until today (PARALLEL)
    if not df_pred.empty:
        df_pred["entry_open"] = np.nan
        df_pred["max_high_to_today"] = np.nan
        df_pred["max_high_date"] = ""
        df_pred["runup_to_high_pct"] = np.nan

        def _runup_task(sym: str, entry_day: date) -> Optional[Dict[str, float]]:
            s = get_thread_session()
            return runup_to_high_until_today(sym, entry_day, session=s, store_dir=store_dir)

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

    # SIDEWAYS FILTER applied ONLY to BUY candidates (PARALLEL)
    if args.sideways_only and not df_pred.empty:
        def _sideways_task(sym: str, entry_day: date) -> Tuple[bool, Optional[Dict[str, float]]]:
            s = get_thread_session()
            stats = sideways_stats_pre_entry(
                sym,
                entry_day,
                session=s,
                store_dir=store_dir,
                sideways_days=args.sideways_days,
            )
            ok = passes_sideways_filter(
                stats,
                max_range_pct=args.sideways_max_range_pct,
                max_abs_close_change_pct=args.sideways_max_abs_change_pct,
            )
            return ok, stats

        df_pred["sideways_pass"] = 0
        df_pred["sideways_range_pct"] = np.nan
        df_pred["sideways_max_abs_close_change_pct"] = np.nan

        jobs2: List[Tuple[int, "concurrent.futures.Future"]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.dl_workers))) as ex:
            for idx, r in df_pred.iterrows():
                sym = r["symbol"]
                entry_day = datetime.strptime(r["entry_date"], "%Y-%m-%d").date()
                jobs2.append((idx, ex.submit(_sideways_task, sym, entry_day)))

            for idx, fut in jobs2:
                ok, stats = fut.result()
                df_pred.at[idx, "sideways_pass"] = 1 if ok else 0
                if stats is not None:
                    df_pred.at[idx, "sideways_range_pct"] = stats["sideways_range_pct"]
                    df_pred.at[idx, "sideways_max_abs_close_change_pct"] = stats["sideways_max_abs_close_change_pct"]

        df_pred = df_pred[(df_pred["pred_buy"] == 1) & (df_pred["sideways_pass"] == 1)].copy()
        df_pred = df_pred.sort_values(["file_date", "prob_up_target_h"], ascending=[True, False])

    out_path = Path(args.out)
    df_pred.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")

    print("\n=== TOP CANDIDATES (predict window) ===")
    if df_pred.empty:
        print("(none after filters)")
    else:
        date_strs = sorted(df_pred["file_date"].unique().tolist())
        for d_str in date_strs:
            dd = df_pred[df_pred["file_date"] == d_str].copy()
            if dd.empty:
                continue

            dd = dd.sort_values("prob_up_target_h", ascending=False).head(args.topk)
            d = datetime.strptime(d_str, "%Y-%m-%d").date()
            entry_day = d + timedelta(days=args.entry_lag_days)

            if args.sideways_only:
                print(
                    f"\nfile={d.isoformat()}  entry={entry_day.isoformat()}  "
                    f"(top {len(dd)})  threshold={args.threshold}  "
                    f"sideways={args.sideways_days}d range<={args.sideways_max_range_pct:.2f} "
                    f"maxAbsClose<={args.sideways_max_abs_change_pct:.2f}"
                )
            else:
                print(f"\nfile={d.isoformat()}  entry={entry_day.isoformat()}  (top {len(dd)})  threshold={args.threshold}")

            for _, r in dd.iterrows():
                runup = r.get("runup_to_high_pct", np.nan)
                maxd = r.get("max_high_date", "")

                feat_pdrop = r.get("feat_prob_drop10", np.nan)
                pdrop = r.get("prob_drop_10_next5d", np.nan)
                dropflag = int(r.get("pred_drop_10_next5d", 0))
                drop_txt = "YES" if dropflag == 1 else "NO"

                print(
                    f"BUY  {r['symbol']:12s}  p={r['prob_up_target_h']:.3f}  "
                    f"featDropP={feat_pdrop:.3f}  "
                    f"runupToHigh={runup:.1f}%  maxHighDate={maxd}  "
                    f"drop10(p)={pdrop:.3f}  drop10={drop_txt}"
                )

    body = build_email_body(
        loaded_hits=len(hits),
        df_all_len=len(df_all),
        train_dates=train_dates,
        predict_dates_list=predict_dates_list,
        df_pred=df_pred,
        args=args,
        model_meta=model_meta,
    )
    send_email_with_analysis(body, df=df_pred, directory=str(folder))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

