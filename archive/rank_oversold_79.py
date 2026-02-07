#!/usr/bin/env python3
"""
rank_oversold.py  (AI-ready-to-buy classifier + SIDEWAYS FILTER + MODEL PERSISTENCE)
+ Rolling split: trains only on files older than N days (default 4), predicts on the newest N days
+ Adds "run-up to highest HIGH since entry_date until today (UTC)" to output
+ ALWAYS emails the output

What this version does (dynamic / rolling):
- Discovers the latest file_date in oversold_analysis/
- Sets:
    train_end_date      = latest_file_date - holdout_days
    predict_start_date  = train_end_date + 1 day
- Labels (learning) computed ONLY for <= train_end_date
- Predicts ONLY for >= predict_start_date

Usage:
  python3.9 rank_oversold.py --threshold 0.55 --sideways-only --sideways-days 5
  python3.9 rank_oversold.py --threshold 0.55 --sideways-only --sideways-days 5 --holdout-days 4
  python3.9 rank_oversold.py --retrain --threshold 0.55 --sideways-only --sideways-days 5

Email:
- Set env var (recommended):
    export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import smtplib
import joblib
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
from sklearn.model_selection import train_test_split
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

# Recommended: export GMAIL_APP_PASSWORD="...."
sender_password = "qhvi syra bbad gylu"


def send_email_with_analysis(body: str, df: Optional[pd.DataFrame] = None, directory: str = ".") -> None:
    """
    Always attempts to send email. If env var missing, prints an error and returns.
    Attaches filtered CSV if df has rows.
    """
    if not sender_password:
        print("Email not sent: env var GMAIL_APP_PASSWORD is not set.", file=sys.stderr)
        return

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = "Oversold Analysis Alert"
    msg.attach(MIMEText(body, "plain"))

    temp_csv = None
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
def save_model_bundle(path: Path, clf: RandomForestClassifier, feature_cols: List[str], meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": clf, "feature_cols": feature_cols, "meta": meta}
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        obj = joblib.load(path)
        if not isinstance(obj, dict):
            return None
        if "model" not in obj or "feature_cols" not in obj or "meta" not in obj:
            return None
        return obj
    except Exception:
        return None


# -----------------------------
# Binance klines + robust cache
# -----------------------------
def fetch_daily_klines(
    symbol: str,
    start_day: date,
    end_day: date,
    session: requests.Session,
    cache_dir: Path,
) -> pd.DataFrame:
    """
    Fetch DAILY klines inclusive start_day..end_day (UTC).
    Robust cache:
      - If cache is empty/corrupt -> delete and refetch
      - Atomic write (tmp -> rename)

    Returns DataFrame with:
      day (YYYY-MM-DD), open, high, low, close, open_time, close_time
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{symbol}_{start_day.isoformat()}_{end_day.isoformat()}_1d.json"
    cache_path = cache_dir / cache_key

    def _parse_cached_json(p: Path):
        try:
            txt = p.read_text().strip()
            if not txt:
                return None
            return json.loads(txt)
        except Exception:
            return None

    data = None
    if cache_path.exists():
        data = _parse_cached_json(cache_path)
        if data is None:
            try:
                cache_path.unlink()
            except Exception:
                pass

    if data is None:
        start_dt = datetime.combine(start_day, datetime.min.time())
        end_dt = datetime.combine(end_day + timedelta(days=1), datetime.min.time())  # end exclusive

        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": int(start_dt.timestamp() * 1000),
            "endTime": int(end_dt.timestamp() * 1000),
            "limit": 1000,
        }

        r = session.get(BINANCE_BASE + KLINES_ENDPOINT, params=params, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()

        try:
            data = r.json()
        except Exception:
            return pd.DataFrame()

        if not isinstance(data, list):
            return pd.DataFrame()

        tmp_path = cache_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(data))
            os.replace(tmp_path, cache_path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows = []
    for k in data:
        try:
            open_time_ms = int(k[0])
            day = datetime.utcfromtimestamp(open_time_ms / 1000.0).date()
            rows.append(
                {
                    "day": day.isoformat(),
                    "open_time": open_time_ms,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "close_time": int(k[6]),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def label_plus_target_within_horizon(
    symbol: str,
    file_day: date,
    *,
    session: requests.Session,
    cache_dir: Path,
    target_pct: float,
    horizon_days: int,
    entry_lag_days: int,
) -> Optional[int]:
    """
    label = 1 if within horizon_days AFTER ENTRY_DAY, max(close) >= entry_open*(1+target_pct)
    ENTRY_DAY = file_day + entry_lag_days
    """
    entry_day = file_day + timedelta(days=entry_lag_days)
    start_day = entry_day
    end_day = entry_day + timedelta(days=horizon_days)

    df = fetch_daily_klines(symbol, start_day, end_day, session, cache_dir)
    if df.empty:
        return None

    entry_open = float(df.iloc[0]["open"])
    max_close = float(df["close"].max())
    target = entry_open * (1.0 + target_pct)

    return 1 if max_close >= target else 0


def runup_to_high_until_today(
    symbol: str,
    entry_day: date,
    *,
    session: requests.Session,
    cache_dir: Path,
) -> Optional[Dict[str, float]]:
    """
    Computes run-up from entry open to the highest HIGH observed from entry_day to today (UTC).
    Returns:
      entry_open, max_high_to_today, max_high_date, runup_to_high_pct
    """
    today_utc = datetime.utcnow().date()
    if entry_day > today_utc:
        return None

    df = fetch_daily_klines(symbol, entry_day, today_utc, session, cache_dir)
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


# -----------------------------
# SIDEWAYS FILTER (pre-entry consolidation)
# -----------------------------
def sideways_stats_pre_entry(
    symbol: str,
    entry_day: date,
    *,
    session: requests.Session,
    cache_dir: Path,
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

    df = fetch_daily_klines(symbol, start, end, session, cache_dir)
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
    cache_dir: Path,
    min_history_hits: int = 3,
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

    session = requests.Session()
    rows = []

    for fday in file_dates:
        for sym in sorted(symbols_in_file[fday]):
            hist = by_symbol[sym]
            hist_up_to = [x for x in hist if x.file_date <= fday]
            if len(hist_up_to) < min_history_hits:
                continue

            feats = build_features_for_symbol(hist, fday)
            if not feats:
                continue

            y = None
            if fday in train_dates:
                y = label_plus_target_within_horizon(
                    sym,
                    fday,
                    session=session,
                    cache_dir=cache_dir,
                    target_pct=target_pct,
                    horizon_days=horizon_days,
                    entry_lag_days=entry_lag_days,
                )
                if y is None:
                    continue

            row = {"file_date": fday.isoformat(), "symbol": sym}
            row.update(feats)
            row["label"] = y
            rows.append(row)

    df = pd.DataFrame(rows)
    return df, train_dates, holdout_dates


def train_model(df_train: pd.DataFrame, seed: int = 42) -> Tuple[RandomForestClassifier, List[str]]:
    df_train = df_train.dropna(subset=["label"]).copy()
    y = df_train["label"].astype(int).values

    feature_cols = [c for c in df_train.columns if c not in ("file_date", "symbol", "label")]
    X = df_train[feature_cols].astype(float).values

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=2,
    )
    clf.fit(X_tr, y_tr)

    p_va = clf.predict_proba(X_va)[:, 1]
    y_hat = (p_va >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_va, p_va)
    except Exception:
        auc = float("nan")

    print("\n=== Validation (random split sanity) ===")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_va, y_hat, digits=4))

    return clf, feature_cols


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
            extra = ""
            if args.sideways_only:
                rng = r.get("sideways_range_pct", np.nan)
                mac = r.get("sideways_max_abs_close_change_pct", np.nan)
                extra = f"  sideways_range={rng:.3f}  sideways_maxAbsClose={mac:.3f}"

            runup = r.get("runup_to_high_pct", np.nan)
            maxh = r.get("max_high_to_today", np.nan)
            maxd = r.get("max_high_date", "")

            lines.append(
                f"BUY  {r['symbol']:12s}  p={r['prob_up_target_h']:.3f}  "
                f"last_drop={r['last_drop']:.2f}  last_cur_rsi={r['last_cur_rsi']:.2f}  hits30={int(r['hits_30d'])}"
                f"{extra}"
                f"  runupToHigh={runup:.1f}%  maxHigh={maxh:.6g}  maxHighDate={maxd}"
            )

    return "\n".join(lines)


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
    ap.add_argument("--cache-dir", default="kline_cache", help="Where to cache Binance klines")
    ap.add_argument("--out", default="ai_predictions.csv", help="Output CSV path for predictions")
    ap.add_argument("--threshold", type=float, default=0.55, help="BUY threshold on probability")
    ap.add_argument("--topk", type=int, default=40, help="Top-K predictions per day to print")

    # Rolling split knob
    ap.add_argument(
        "--holdout-days",
        type=int,
        default=4,
        help="Number of most recent file-days to NOT train on; predict only (rolling split).",
    )

    # Sideways filter knobs
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

    # Model persistence
    ap.add_argument("--model-path", default="models/rf_oversold.joblib", help="Where to save/load the trained model")
    ap.add_argument("--retrain", action="store_true", help="Force retrain even if a saved model exists")

    args = ap.parse_args()

    folder = Path(args.dir)
    cache_dir = Path(args.cache_dir)

    # -----------------------------
    # Load data first (to discover latest file date)
    # -----------------------------
    hits = load_oversold_hits(folder)
    print(f"Loaded hits: {len(hits)}")

    file_dates = sorted({h.file_date for h in hits})
    if not file_dates:
        print("ERROR: No file dates found.", file=sys.stderr)
        return 2

    latest_file_date = file_dates[-1]
    holdout_days = max(1, int(args.holdout_days))

    # Rolling split:
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

    # -----------------------------
    # Build ML table with strict no-leak guarantee
    # -----------------------------
    df_all, train_dates, predict_dates_list = build_ml_table(
        hits,
        train_end_date=train_end_date,
        predict_start_date=predict_start_date,
        target_pct=args.target_pct,
        horizon_days=args.horizon_days,
        entry_lag_days=args.entry_lag_days,
        cache_dir=cache_dir,
        min_history_hits=args.min_history_hits,
    )

    print(f"\nTotal samples (train + predict): {len(df_all)}")
    print(f"Train dates: {train_dates[0]} .. {train_dates[-1]}  ({len(train_dates)} days)")
    if predict_dates_list:
        print(
            f"Predict dates (classify only): {predict_dates_list[0]} .. {predict_dates_list[-1]}  ({len(predict_dates_list)} days)"
        )
    else:
        print("Predict dates (classify only): (none found yet)")

    model_path = Path(args.model_path)

    # Build df_train strictly from <= train_end_date
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

    # Load existing model unless retrain
    bundle = None if args.retrain else load_model_bundle(model_path)
    model_meta: Optional[Dict] = None

    if bundle is not None:
        clf = bundle["model"]
        feature_cols = bundle["feature_cols"]
        model_meta = bundle.get("meta", {})

        missing = [c for c in feature_cols if c not in df_train.columns]
        if missing:
            print(f"\nSaved model feature mismatch; missing columns in current data: {missing}")
            print("Retraining...")
            bundle = None
        else:
            # IMPORTANT: rolling split changes over time -> retrain when cutoff moved
            saved_train_end = model_meta.get("train_end_date")
            if saved_train_end != train_end_date.isoformat():
                print(
                    f"\nSaved model was trained_through={saved_train_end} but rolling train_end_date={train_end_date}. "
                    "Retraining..."
                )
                bundle = None
            else:
                print(
                    f"\nLoaded saved model: {model_path} "
                    f"(trained_through={saved_train_end} sklearn={model_meta.get('sklearn_version')})"
                )

    if bundle is None:
        clf, feature_cols = train_model(df_train)

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
        }
        save_model_bundle(model_path, clf, feature_cols, model_meta)
        print(f"\nSaved model: {model_path}")

    # Predict ONLY in predict window (>= predict_start_date)
    df_pred = predict_dates(clf, feature_cols, df_all, predict_dates_list, threshold=args.threshold)
    df_pred = df_pred[df_pred["pred_buy"] == 1].copy()

    # Add entry_date column
    if not df_pred.empty:
        df_pred["entry_date"] = df_pred["file_date"].apply(
            lambda s: (datetime.strptime(s, "%Y-%m-%d").date() + timedelta(days=args.entry_lag_days)).isoformat()
        )

    # One shared session for everything below
    session = requests.Session()

    # Add run-up columns (entry open -> highest HIGH until today UTC)
    if not df_pred.empty:
        df_pred["entry_open"] = np.nan
        df_pred["max_high_to_today"] = np.nan
        df_pred["max_high_date"] = ""
        df_pred["runup_to_high_pct"] = np.nan

        for idx, r in df_pred.iterrows():
            sym = r["symbol"]
            entry_day = datetime.strptime(r["entry_date"], "%Y-%m-%d").date()
            info = runup_to_high_until_today(sym, entry_day, session=session, cache_dir=cache_dir)
            if info is None:
                continue
            df_pred.at[idx, "entry_open"] = info["entry_open"]
            df_pred.at[idx, "max_high_to_today"] = info["max_high_to_today"]
            df_pred.at[idx, "max_high_date"] = info["max_high_date"]
            df_pred.at[idx, "runup_to_high_pct"] = info["runup_to_high_pct"]

    # SIDEWAYS FILTER applied ONLY to BUY candidates
    if args.sideways_only and not df_pred.empty:
        sideways_rows = []

        for idx, r in df_pred.iterrows():
            sym = r["symbol"]
            entry_day = datetime.strptime(r["entry_date"], "%Y-%m-%d").date()

            stats = sideways_stats_pre_entry(
                sym,
                entry_day,
                session=session,
                cache_dir=cache_dir,
                sideways_days=args.sideways_days,
            )
            ok = passes_sideways_filter(
                stats,
                max_range_pct=args.sideways_max_range_pct,
                max_abs_close_change_pct=args.sideways_max_abs_change_pct,
            )

            sideways_rows.append((idx, ok, stats))

        df_pred["sideways_pass"] = 0
        df_pred["sideways_range_pct"] = np.nan
        df_pred["sideways_max_abs_close_change_pct"] = np.nan

        for idx, ok, stats in sideways_rows:
            df_pred.at[idx, "sideways_pass"] = 1 if ok else 0
            if stats is not None:
                df_pred.at[idx, "sideways_range_pct"] = stats["sideways_range_pct"]
                df_pred.at[idx, "sideways_max_abs_close_change_pct"] = stats["sideways_max_abs_close_change_pct"]

        df_pred = df_pred[(df_pred["pred_buy"] == 1) & (df_pred["sideways_pass"] == 1)].copy()
        df_pred = df_pred.sort_values(["file_date", "prob_up_target_h"], ascending=[True, False])

    # Save CSV (filtered)
    out_path = Path(args.out)
    df_pred.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")

    # Print to console
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
                extra = ""
                if args.sideways_only:
                    rng = r.get("sideways_range_pct", np.nan)
                    mac = r.get("sideways_max_abs_close_change_pct", np.nan)
                    extra = f"  sideways_range={rng:.3f}  sideways_maxAbsClose={mac:.3f}"

                runup = r.get("runup_to_high_pct", np.nan)
                maxh = r.get("max_high_to_today", np.nan)
                maxd = r.get("max_high_date", "")

                print(
                    f"BUY  {r['symbol']:12s}  p={r['prob_up_target_h']:.3f}  "
                    f"last_drop={r['last_drop']:.2f}  last_cur_rsi={r['last_cur_rsi']:.2f}  hits30={int(r['hits_30d'])}"
                    f"{extra}"
                    f"  runupToHigh={runup:.1f}%  maxHigh={maxh:.6g}  maxHighDate={maxd}"
                )

    # ALWAYS email the output + attach CSV (if non-empty)
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

