#!/usr/bin/env python3

'''
In prod:
Increase trees a bit (stability): n_estimators=400
Allow smaller leaves (more sensitivity): min_samples_leaf=1 (can raise recall, may drop precision)

'''

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
from itertools import product
import math
import traceback

HOLDOUT_DAYS = 5
ENTRY_LAG_DAYS = 1
DROP_PCT = 0.12 #0.1 change back before using this model
OFFLINE_CACHE_ONLY = 0

# ============================================================
# EMAIL CONFIG (ENV VARS ONLY FOR PASSWORD)
# ============================================================
RECIPIENT_EMAIL = os.getenv("OVERSOLD_RECIPIENT_EMAIL", "mina.moussa@hotmail.com")
SENDER_EMAIL = os.getenv("OVERSOLD_SENDER_EMAIL", "minamoussa903@gmail.com")
SENDER_PASSWORD = os.getenv("OVERSOLD_SENDER_PASSWORD", "qhvi syra bbad gylu")  # MUST be set in env

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

MIN_HISTORY_HITS = 3

# Parameter grid
HORIZONS = [21]
DROP_HORIZONS = [21]
TPS = [0.2]
DROPS = [0.12]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]

def _stable_pct(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


def filter_df_day_by_entry_rule(
    df_day: pd.DataFrame,
    *,
    state: dict,
    p_threshold: float = 0.61,
    p_col: str = "prob_up_target_h",
) -> pd.DataFrame:
    """
    Keep a row if:
      - p > 0.61 OR
      - previous day's p for that symbol is lower than today's p

    Uses state["last_p_by_symbol"] as memory across days/runs.
    Updates last_p_by_symbol for symbols present today (using the best p today).
    """
    if df_day is None or df_day.empty:
        return df_day

    last_p_by_symbol = state.setdefault("last_p_by_symbol", {})

    # Best p today per symbol (if multiple rows exist)
    best_today = (
        df_day.groupby("symbol", as_index=False)[p_col]
        .max()
        .rename(columns={p_col: "p_today"})
    )

    allowed_syms = set()
    for _, r in best_today.iterrows():
        sym = str(r["symbol"])
        p_today_raw = r["p_today"]
        if pd.isna(p_today_raw):
            # Unknown stays unknown: don't allow, and don't update memory for this symbol
            continue
        p_today = float(p_today_raw)
        prev_p = last_p_by_symbol.get(sym)

        if (p_today > float(p_threshold)) or (prev_p is not None and p_today > prev_p):
            allowed_syms.add(sym)

    # IMPORTANT: update memory for all symbols that appeared today (even if not allowed),
    # so tomorrow can compare.
    for _, r in best_today.iterrows():
        sym = str(r["symbol"])
        p_today_raw = r["p_today"]
        if pd.isna(p_today_raw):
            continue  # don't update memory with unknown
        last_p_by_symbol[sym] = float(p_today_raw)

    if not allowed_syms:
        return df_day.iloc[0:0].copy()

    return df_day[df_day["symbol"].astype(str).isin(allowed_syms)].copy()



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
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        _tls.session = s
    return s


# ============================================================
# PER-SYMBOL PARQUET STORE FOR 1D KLINES + TOMBSTONES
# ============================================================
_SYMBOL_MEMO_LOCK = threading.Lock()
_SYMBOL_MEMO: Dict[str, pd.DataFrame] = {}

_FILE_LOCKS_LOCK = threading.Lock()
_FILE_LOCKS: Dict[str, threading.Lock] = {}


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

    df["no_data"] = pd.to_numeric(df["no_data"], errors="coerce")
    # keep NaN as NaN for now; when writing, coerce safely:
    df["no_data"] = df["no_data"].fillna(0).astype(bool)
    return df


def _store_path(store_dir: Path, symbol: str) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / f"{symbol}_1d.parquet"


def _read_parquet_best_effort(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def _write_parquet_best_effort(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
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
    df["day"] = df["day"].astype(str)
    df["no_data"] = df["no_data"].astype(bool)
    _write_parquet_best_effort(df, p)


def _binance_fetch_1d_range(symbol: str, start_day: date, end_day: date, session: requests.Session) -> pd.DataFrame:
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
            #fetched = _fill_tombstones(need_start, need_end, fetched)
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
            #back = _fill_tombstones(need_start, have_min - timedelta(days=1), back)
            if not back.empty:
                parts.append(back)

        if need_end > have_max:
            fwd = _binance_fetch_1d_range(symbol, have_max + timedelta(days=1), need_end, session)
            #fwd = _fill_tombstones(have_max + timedelta(days=1), need_end, fwd)
            if not fwd.empty:
                parts.append(fwd)

        merged = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        #merged = _fill_tombstones(need_start, need_end, merged)

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
def label_drop_pct_within_horizon(
    symbol: str,
    file_day: date,
    *,
    session: requests.Session,
    store_dir: Path,
    drop_pct: float,
    horizon_days: int,
    entry_lag_days: int,
    offline_cache_only: bool,
) -> Optional[int]:
    """
    Label downside event:
      y=1 if within [entry_day .. entry_day+horizon_days] the LOW drops <= entry_open*(1-drop_pct)
      else y=0

    Uses:
      entry_day = file_day + entry_lag_days
      entry_open = OPEN of entry_day candle
      min_low = minimum LOW over the window

    Returns None if candles missing.
    """
    last_closed = datetime.utcnow().date() - timedelta(days=1)

    entry_day = file_day + timedelta(days=int(entry_lag_days))
    start_day = entry_day
    end_day = min(entry_day + timedelta(days=int(horizon_days)), last_closed)
    if end_day < start_day:
        return None

    df = get_klines_window_maybe_offline(
        symbol,
        start_day,
        end_day,
        session=session,
        store_dir=store_dir,
        offline_cache_only=offline_cache_only,
    )
    if df is None or df.empty:
        return None

    try:
        # entry open is first row's open (start_day == entry_day)
        entry_open = float(df.iloc[0]["open"])
        min_low = float(pd.to_numeric(df["low"], errors="coerce").min())
        if not np.isfinite(entry_open) or not np.isfinite(min_low) or abs(entry_open) < 1e-12:
            return None
    except Exception:
        return None

    thresh = entry_open * (1.0 - float(drop_pct))
    return 1 if (min_low <= thresh) else 0






def build_df_train_for_models(
    df_all: pd.DataFrame,
    train_dates: List[date],
) -> pd.DataFrame:
    """
    Convenience: build df_train with labels present.
    """
    df_train = df_all[df_all["file_date"].isin([d.isoformat() for d in train_dates])].copy()
    df_train = df_train.dropna(subset=["label"]).copy()
    return df_train


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


# ============================================================
# ML TABLE CACHE
# ============================================================
def _stable_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0



# ============================================================
# MODEL PERSISTENCE + GATING
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_BUNDLE_PATH = BASE_DIR / "models" / "default_model_76.joblib"
ROLLING_BUNDLE_PATH = BASE_DIR / "models" / "rolling_bundle.joblib"

print("CWD:", Path.cwd())
print("Rolling exists:", ROLLING_BUNDLE_PATH, ROLLING_BUNDLE_PATH.exists())
print("Default exists:", DEFAULT_MODEL_BUNDLE_PATH, DEFAULT_MODEL_BUNDLE_PATH.exists())


GATE_MIN_PREC1 = 0.7799
GATE_MIN_SUP1 = 410



# ============================================================
# TRAINING HELPERS
# ============================================================
def _rf_default(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=2,
    )

def _get_feature_cols(
    df: pd.DataFrame,
    label_cols: Tuple[str, ...],
    *,
    max_nan_frac: float = 0.0,   # 0.0 = drop if any NA
) -> List[str]:
    """
    Return numeric feature columns only.
    Also drops columns with too many NaNs (default: drop if ANY NaN).
    """
    ban = {"file_date", "symbol"} | set(label_cols)
    bad_name_substrings = ("date", "day", "time", "timestamp")

    cols: List[str] = []
    n = len(df)
    if n <= 0:
        return cols

    for c in df.columns:
        if c in ban:
            continue

        lc = str(c).lower()
        if any(s in lc for s in bad_name_substrings):
            continue

        s = df[c]

        # drop datetime dtype
        if np.issubdtype(s.dtype, np.datetime64):
            continue

        # drop objects/strings
        if s.dtype == "object":
            continue

        # drop columns with NaNs (or too many NaNs)
        nan_frac = float(pd.isna(s).sum()) / float(n)
        if nan_frac > float(max_nan_frac):
            continue

        cols.append(c)

    return cols



def train_and_report(
    df_train: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    seed: int = 42,
) -> Tuple[RandomForestClassifier, Dict]:
    df_train = df_train.dropna(subset=[label_col]).copy()
    from sklearn.model_selection import TimeSeriesSplit

    df_train = df_train.dropna(subset=[label_col]).copy()
    df_train["file_date_dt"] = pd.to_datetime(df_train["file_date"])
    df_train = df_train.sort_values("file_date_dt")

    X = df_train[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()#.fillna(0.0).values
    y = df_train[label_col].astype(int).values

    tscv = TimeSeriesSplit(n_splits=3)
    for tr_idx, va_idx in tscv.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        clf = _rf_default(seed=seed)
        clf.fit(X_tr, y_tr)
        # evaluate...

    clf = _rf_default(seed=seed)
    clf.fit(X_tr, y_tr)

    p_va = clf.predict_proba(X_va)[:, 1]
    y_hat = (p_va >= 0.61).astype(int)

    try:
        auc = roc_auc_score(y_va, p_va)
    except Exception:
        auc = float("nan")



    rep_dict = classification_report(y_va, y_hat, digits=4, output_dict=True)
    c1 = rep_dict.get("1", {})
    c1_prec = float(c1.get("precision", float("nan")))
    c1_sup = int(c1.get("support", 0))

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
# DOWNSIDE LABELS + SKIP-OOF FEATURE (WITH FALLBACK)
# ============================================================
def build_downside_labels_inplace(
    df_train: pd.DataFrame,
    *,
    store_dir: Path,
    drop_pct: float,
    drop_horizon_days: int,
    entry_lag_days: int,
    dl_workers: int,
    offline_cache_only: bool,
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
                offline_cache_only=offline_cache_only,
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
    df_train = df_train.copy()
    usable = df_train.dropna(subset=["label_down10"]).copy()
    df_train["feat_prob_drop10"] = np.nan

    # Fallback: if can't train downside model, fill 0.5
    #if usable.empty or usable["label_down10"].nunique() < 2:
    #    print("\nNOTE: Downside model not trainable (label_down10 has <2 classes). Using feat_prob_drop10=0.5 fallback.")
    #    df_train["feat_prob_drop10"] = 0.5
    #    return df_train, None, None

    feat_cols_down = list(base_feature_cols)
    clf_down, _m = train_and_report(usable, "label_down10", feat_cols_down, seed=seed)

    X = usable[feat_cols_down].astype(float).values
    probs = clf_down.predict_proba(X)[:, 1]
    df_train.loc[usable.index, "feat_prob_drop10"] = probs

    # Any unlabeled rows get 0.5 fallback (keeps main model stable)
    #df_train["feat_prob_drop10"] = df_train["feat_prob_drop10"].fillna(0.5)

    return df_train, clf_down, feat_cols_down


def train_main_model_with_drop_feature(
    df_train: pd.DataFrame,
    *,
    seed: int = 42,
) -> Tuple[RandomForestClassifier, List[str], Dict]:
    df_train = df_train.dropna(subset=["label"]).copy()

    # ensure feature exists and has no NaN
    #if "feat_prob_drop10" not in df_train.columns:
    #    df_train["feat_prob_drop10"] = 0.5
    #df_train["feat_prob_drop10"] = df_train["feat_prob_drop10"].fillna(0.5)

    if df_train.empty:
        raise RuntimeError("ERROR: df_train empty after labeling/features.")

    if df_train["label"].nunique() < 2:
        raise RuntimeError("ERROR: training labels have <2 classes (not enough positives/negatives).")

    feature_cols = _get_feature_cols(df_train, ("label", "label_down10"))
    if "feat_prob_drop10" in df_train.columns and not df_train["feat_prob_drop10"].isna().any():
        if "feat_prob_drop10" not in feature_cols:
            feature_cols.append("feat_prob_drop10")

    clf, metrics = train_and_report(df_train, "label", feature_cols, seed=seed)
    return clf, feature_cols, metrics



def get_klines_window_cached_only(
    symbol: str,
    start_day: date,
    end_day: date,
    *,
    store_dir: Path,
) -> pd.DataFrame:
    """
    Strictly reads from local parquet store. Never fetches.
    Returns empty if parquet missing or window not present.
    """
    if end_day < start_day:
        return pd.DataFrame()

    df = _read_symbol_store(store_dir, symbol)
    if df is None or df.empty:
        return pd.DataFrame()

    # normalize day to date objects
    df = df.copy()
    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    df = df[(df["day"].notna()) & (~df["no_data"])].copy()

    m = (df["day"] >= start_day) & (df["day"] <= end_day)
    out = df.loc[m, ["day", "open", "high", "low", "close", "open_time", "close_time"]].copy()
    if out.empty:
        return out

    out["day"] = out["day"].astype(str)
    return out.sort_values("day").reset_index(drop=True)

def relabel_for_combo(
    df_base: pd.DataFrame,
    *,
    target_pct: float,
    horizon_days: int,
    drop_pct: float,
    drop_horizon_days: int,
    entry_lag_days: int,
    store_dir: Path,
    dl_workers: int,
    # ---------------- NEW (optional) ----------------
    enforce_sl_not_first: bool = False,
    sl_pct: float = HARD_STOP_PCT,
    sl_same_day_counts_as_first: bool = True,
) -> pd.DataFrame:
    df = df_base.copy()

    # ---- Main label (upside) ----
    df["label"] = np.nan

    def _label_up_row(row: pd.Series, *, session: requests.Session) -> float:
        """
        Returns 0/1 or np.nan if candles missing.
        """
        try:
            fday = datetime.strptime(str(row["file_date"]), "%Y-%m-%d").date()
            sym = str(row["symbol"])
        except Exception:
            return np.nan

        entry = fday + timedelta(days=int(entry_lag_days))
        endw = entry + timedelta(days=int(horizon_days))

        w = get_klines_window(
            sym,
            entry,
            endw,
            session=session,
            store_dir=store_dir,
        )
        if w is None or w.empty:
            return np.nan

        try:
            entry_open = float(w.iloc[0]["open"])
            if not np.isfinite(entry_open) or abs(entry_open) < 1e-12:
                return np.nan
        except Exception:
            return np.nan

        # --- Original behavior (keep as-is when enforce flag is off) ---
        if not bool(enforce_sl_not_first):
            try:
                max_close = float(pd.to_numeric(w["high"], errors="coerce").max())
                if not np.isfinite(max_close):
                    return np.nan
                return float(int(max_close >= entry_open * (1.0 + float(target_pct))))
            except Exception:
                return np.nan

        # --- Enforced SL-before-TP logic (label=1 only if TP happens first) ---
        try:
            ww = w.copy()
            ww["day"] = pd.to_datetime(ww["day"], errors="coerce").dt.date
            ww = ww.sort_values("day").reset_index(drop=True)

            highs = pd.to_numeric(ww["high"], errors="coerce")
            lows = pd.to_numeric(ww["low"], errors="coerce")

            tp_px = entry_open * (1.0 + float(target_pct))
            sl_px = entry_open * (1.0 - float(drop_pct))

            tp_hit = (highs >= tp_px).fillna(False).values
            sl_hit = (lows <= sl_px).fillna(False).values

            if not bool(tp_hit.any()):
                return 0.0

            first_tp_idx = int(np.argmax(tp_hit))  # valid because tp_hit.any() is true

            # If SL occurs before TP, it's NOT class 1.
            # If SL occurs on the same day as TP:
            #   - if sl_same_day_counts_as_first=True -> NOT class 1
            #   - else -> allow TP (treat as TP-first)
            if sl_same_day_counts_as_first:
                sl_before_or_same = bool(sl_hit[: first_tp_idx + 1].any())
            else:
                sl_before_or_same = bool(sl_hit[: first_tp_idx].any())

            return 0.0 if sl_before_or_same else 1.0
        except Exception:
            return np.nan

    # parallelize by symbol (keeps your prior structure)
    by_sym = {sym: df[df["symbol"] == sym].index.tolist() for sym in df["symbol"].unique()}

    def _task(sym: str, idxs: List[int]) -> Dict[int, float]:
        s = get_thread_session()
        out: Dict[int, float] = {}
        for i in idxs:
            out[i] = _label_up_row(df.loc[i], session=s)
        return out

    with ThreadPoolExecutor(max_workers=max(1, int(dl_workers))) as ex:
        futs = [ex.submit(_task, sym, idxs) for sym, idxs in by_sym.items()]
        for fut in futs:
            res = fut.result()
            for i, v in res.items():
                df.at[i, "label"] = v

    # ---- Downside label (unchanged) ----
    df = build_downside_labels_inplace(
        df,
        store_dir=store_dir,
        drop_pct=drop_pct,
        drop_horizon_days=drop_horizon_days,
        entry_lag_days=entry_lag_days,
        dl_workers=dl_workers,
        offline_cache_only=False,
    )

    return df



def build_base_feature_table(
    hits: List[OversoldHit],
    *,
    entry_lag_days: int,
    min_history_hits: int,
) -> pd.DataFrame:
    """
    Build DF with:
      file_date, symbol, and ALL features
    BUT NO labels (label, label_down10).
    This is built ONCE per stage (e.g. first 10 files).
    """

    by_symbol: Dict[str, List[OversoldHit]] = defaultdict(list)
    symbols_in_file: Dict[date, set] = defaultdict(set)

    for h in hits:
        by_symbol[h.symbol].append(h)
        symbols_in_file[h.file_date].add(h.symbol)

    file_dates = sorted({h.file_date for h in hits})
    rows = []

    for fday in file_dates:
        syms = symbols_in_file.get(fday, set())
        for sym in sorted(syms):
            hist = by_symbol.get(sym, [])
            hist_up_to = [x for x in hist if x.file_date <= fday]

            if len(hist_up_to) < min_history_hits:
                continue

            feats = build_features_for_symbol(hist, fday)
            if not feats:
                continue

            row = {"file_date": fday.isoformat(), "symbol": sym}
            row.update(feats)
            rows.append(row)

    df_base = pd.DataFrame(rows).sort_values(["file_date", "symbol"]).reset_index(drop=True)
    return df_base


def ensure_symbol_days_maybe_offline(
    symbol: str,
    need_start: date,
    need_end: date,
    *,
    session: requests.Session,
    store_dir: Path,
    offline_cache_only: bool,
) -> pd.DataFrame:
    """
    If offline_cache_only=True, do not fetch; just return whatever exists on disk.
    Otherwise use your existing ensure_symbol_days() (fetch+fill tombstones).
    """
    if offline_cache_only:
        # Return what we have on disk. No tombstones expansion needed for offline mode.
        return _read_symbol_store(store_dir, symbol)
    return ensure_symbol_days(symbol, need_start, need_end, session=session, store_dir=store_dir)


def get_klines_window_maybe_offline(
    symbol: str,
    start_day: date,
    end_day: date,
    *,
    session: requests.Session,
    store_dir: Path,
    offline_cache_only: bool,
) -> pd.DataFrame:
    """
    Unified accessor used everywhere.
    - offline_cache_only=True: disk only
    - otherwise: ensure (fetch if needed) then slice
    """
    if offline_cache_only:
        return get_klines_window_cached_only(symbol, start_day, end_day, store_dir=store_dir)
    return get_klines_window(symbol, start_day, end_day, session=session, store_dir=store_dir)


def _missing_days_in_window(df_window: pd.DataFrame, start_day: date, end_day: date) -> List[str]:
    """
    Given a returned window df (with 'day' as YYYY-MM-DD strings), compute missing dates.
    """
    if end_day < start_day:
        return []
    have = set(pd.to_datetime(df_window["day"], errors="coerce").dt.date.dropna().tolist()) if df_window is not None and not df_window.empty else set()
    missing = []
    d = start_day
    while d <= end_day:
        if d not in have:
            missing.append(d.isoformat())
        d += timedelta(days=1)
    return missing

def send_email_with_attachments(
    subject: str,
    body: str,
    attachments: List[Path],
    *,
    sender: str = SENDER_EMAIL,
    recipient: str = RECIPIENT_EMAIL,
    password: Optional[str] = SENDER_PASSWORD,
) -> None:
    if not password:
        print("NOTE: OVERSOLD_SENDER_PASSWORD not set; skipping email.")
        return

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    for p in attachments:
        try:
            p = Path(p)
            if not p.exists():
                print(f"NOTE: attachment missing, skipping: {p}")
                continue

            part = MIMEBase("application", "octet-stream")
            part.set_payload(p.read_bytes())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{p.name}"')
            msg.attach(part)
        except Exception as e:
            print(f"NOTE: failed attaching {p}: {e}")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        print(f"Email sent to {recipient} with {len(attachments)} attachment(s).")
    except Exception as e:
        print(f"ERROR: email send failed: {e}")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()

    # ---------- SWEEP ARGS (unchanged) ----------
    ap.add_argument("--sweep-enforce-sl-not-first", action="store_true",
                    help="In sweep: label=1 only if TP hit and SL did NOT happen before (or same day as) TP.")

    ap.add_argument("--sweep", action="store_true",
                    help="Run progressive parameter sweep on a small subset of files.")
    ap.add_argument("--sweep-files", default="10,20,40,80",
                    help="Comma-separated progressive file counts to test.")
    ap.add_argument("--sweep-topk", type=int, default=10,
                    help="Show top-K configs each stage.")
    ap.add_argument("--sweep-max-combos", type=int, default=0,
                    help="Optional cap on number of parameter combos (0=all).")
    ap.add_argument("--sweep-fast", action="store_true",
                    help="Use faster RF settings during sweep (recommended).")

    # ---------- NORMAL ARGS ----------
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--cache-dir", default="kline_store")
    ap.add_argument("--target-pct", type=float, default=0.20)
    ap.add_argument("--dl-workers", type=int, default=16)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip-oof", action="store_true")
    ap.add_argument("--dir", default="oversold_analysis")
    ap.add_argument(
        "--offline-cache-only",
        action="store_true",
        help="Never fetch klines from Binance; use cached parquet only.",
    )

    args = ap.parse_args()
    OFFLINE_CACHE_ONLY = bool(args.offline_cache_only)

    folder = Path(args.dir)
    store_dir = Path(args.cache_dir)

    # ============================================================
    # LOAD HITS (common for sweep + normal)
    # ============================================================
    hits = load_oversold_hits(folder)
    print(f"Loaded hits: {len(hits)}")

    file_dates = sorted({h.file_date for h in hits})
    if not file_dates:
        print("ERROR: No file dates found.", file=sys.stderr)
        return 2

    latest_file_date = file_dates[-1]
    print(f"Latest file date: {latest_file_date}")

    # ============================================================
    # OPTIONAL PREFETCH (common)
    # ============================================================
    if OFFLINE_CACHE_ONLY:
        print("\nPrefetch skipped: --offline-cache-only enabled.")
    else:
        try:
            print(f"\nPrefetching klines into: {store_dir} ...")
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

    # ============================================================
    # ===================== SWEEP MODE ===========================
    # ============================================================
    if args.sweep:
        print("\n=== PROGRESSIVE REALISTIC SWEEP (FORWARD TIME-SERIES) ===")

        sweep_file_counts = [int(x) for x in args.sweep_files.split(",") if x.strip().isdigit()]
        all_results = []

        for n_files in sweep_file_counts:
            print(f"\n=== SWEEP STAGE: last {n_files} files ===")

            # Select most recent N files
            stage_hits = [h for h in hits if h.file_date >= file_dates[-n_files]]

            # Build BASE table once (features only)
            df_base = build_base_feature_table(
                stage_hits,
                entry_lag_days=ENTRY_LAG_DAYS,
                min_history_hits=MIN_HISTORY_HITS,
            )
            print(f"[SWEEP] Base table rows: {len(df_base)}")

            # Determine realistic time split (not random!)
            stage_dates = sorted({h.file_date for h in stage_hits})
            holdout_days = max(3, int(len(stage_dates) * 0.25))  # ~25% forward holdout
            train_dates = stage_dates[:-holdout_days]
            valid_dates = stage_dates[-holdout_days:]

            print(
                f"[SWEEP] Time split: "
                f"train={train_dates[0]}..{train_dates[-1]} "
                f"valid={valid_dates[0]}..{valid_dates[-1]}"
            )

            stage_results = []
            combo_count = 0

            for H in HORIZONS:
                for DH in DROP_HORIZONS:
                    for TP in TPS:
                        for DP in DROPS:

                            combo_count += 1
                            if args.sweep_max_combos and combo_count > args.sweep_max_combos:
                                break

                            print(f"\n[SWEEP] H={H} DH={DH} TP={TP} DP={DP}")

                            # ---- RELABEL BASE TABLE FOR THIS COMBO ----
                            df_labeled = relabel_for_combo(
                                df_base,
                                target_pct=TP,
                                horizon_days=H,
                                drop_pct=DP,
                                drop_horizon_days=DH,
                                entry_lag_days=ENTRY_LAG_DAYS,
                                store_dir=store_dir,
                                dl_workers=args.dl_workers,
                                enforce_sl_not_first=bool(args.sweep_enforce_sl_not_first),
                            )

                            # Split by TIME (not random)
                            df_train = df_labeled[
                                df_labeled["file_date"].isin([d.isoformat() for d in train_dates])
                            ].dropna(subset=["label"]).copy()

                            df_valid = df_labeled[
                                df_labeled["file_date"].isin([d.isoformat() for d in valid_dates])
                            ].dropna(subset=["label"]).copy()

                            if df_train.empty or df_train["label"].nunique() < 2:
                                print("  -> Skipping (not enough training classes)")
                                continue

                            if df_valid.empty:
                                print("  -> Skipping (no forward validation rows)")
                                continue

                            feature_cols = _get_feature_cols(df_train, ("label", "label_down10"))

                            # Train on PAST only
                            X_tr = df_train[feature_cols].astype(float).values
                            y_tr = df_train["label"].astype(int).values

                            if args.sweep_fast:
                                clf = RandomForestClassifier(
                                    n_estimators=120,
                                    random_state=42,
                                    n_jobs=-1,
                                    class_weight="balanced_subsample",
                                )
                            else:
                                clf = RandomForestClassifier(
                                    n_estimators=300,
                                    random_state=42,
                                    n_jobs=-1,
                                    class_weight="balanced_subsample",
                                )

                            clf.fit(X_tr, y_tr)

                            # Validate on FUTURE only
                            X_va = df_valid[feature_cols].astype(float).values
                            y_va = df_valid["label"].astype(int).values

                            p_va = clf.predict_proba(X_va)[:, 1]
                            y_hat = (p_va >= 0.61).astype(int)

                            df_pred_hold = df_valid[["file_date", "symbol"]].copy()
                            df_preds = df_valid[["file_date", "symbol"]].copy()
                            df_preds["prob"] = p_va
                            df_preds["pred"] = y_hat
                            df_preds["true"] = y_va

                            # Print predictions by day (sorted by prob desc)
                            print("\n=== PREDICTIONS BY DAY (HOLDOUT) ===")
                            for day, g in df_preds.groupby("file_date", sort=True):
                                gg = g.sort_values("prob", ascending=False)
                                print(f"\n--- {day} (rows={len(gg)}) ---")
                                print(gg[["symbol", "prob", "pred", "true"]].to_string(index=False))

                            # Save predictions to CSV (overwrite each stage/combo; your lists are single-value anyway)
                            pred_out = Path("sweep_predictions_by_day.csv")
                            df_preds.sort_values(["file_date", "prob"], ascending=[True, False]).to_csv(pred_out,
                                                                                                        index=False)
                            print(f"\nSaved predictions to: {pred_out}")

                            # Forward (realistic) metrics
                            tp = int(((y_hat == 1) & (y_va == 1)).sum())
                            fp = int(((y_hat == 1) & (y_va == 0)).sum())
                            fn = int(((y_hat == 0) & (y_va == 1)).sum())

                            prec1 = tp / (tp + fp) if (tp + fp) else float("nan")
                            rec1 = tp / (tp + fn) if (tp + fn) else float("nan")
                            f1 = (
                                2 * prec1 * rec1 / (prec1 + rec1)
                                if (prec1 + rec1) and not math.isnan(prec1)
                                else float("nan")
                            )

                            print(
                                f"  REALISTIC -> prec1={prec1:.3f}  rec1={rec1:.3f}  "
                                f"f1={f1:.3f}  buys={tp + fp}  winners={tp}"
                            )

                            stage_results.append({
                                "n_files": n_files,
                                "H": H,
                                "DH": DH,
                                "TP": TP,
                                "DP": DP,
                                "prec1": prec1,
                                "rec1": rec1,
                                "f1": f1,
                                "tp": tp,
                                "fp": fp,
                                "fn": fn,
                                "train_rows": len(df_train),
                                "valid_rows": len(df_valid),
                            })

            df_stage = pd.DataFrame(stage_results).sort_values("f1", ascending=False)
            print(f"\n=== TOP {args.sweep_topk} FOR {n_files} FILES ===")
            print(df_stage.head(args.sweep_topk).to_string(index=False))

            all_results.extend(stage_results)

        df_all = pd.DataFrame(all_results).sort_values("f1", ascending=False)
        out = Path("sweep_results_realistic.csv")
        df_all.to_csv(out, index=False)
        print(f"\nSaved realistic sweep results to: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
