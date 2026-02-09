#!/usr/bin/env python3
"""
rank_oversold_predict_only.py
============================

CLEAN "predict-only" version with a SMALL cache.

What it does (only):
1) Parse oversold_analysis/*_oversold.txt into hits
2) Build features ONLY for the last HOLDOUT_DAYS file_dates (no training, no labels, no Binance)
3) Cache that small prediction-feature table using a stable hash of the oversold folder contents + params
4) Load existing model bundle (joblib)
5) Predict on those rows, filter by threshold
6) Save ai_predictions.csv
7) Email results (optional) + keeps password default as requested

Removed:
- ETF mode / positions / replay
- BTC mood
- runup-to-high
- Binance kline fetching + parquet store
- labeling + training + gating + downside model training
- full 80k ML table build

Assumptions:
- Your saved model bundle expects the SAME feature columns produced by build_features_for_symbol().
- The bundle is a dict and contains:
    - model: bundle["main_model"] OR bundle["clf_main"] OR bundle["model"]
    - feature cols: bundle["main_feature_cols"] OR bundle["feat_cols_main"] OR bundle["feature_cols"]
    - optional meta: bundle["meta"]

Usage:
  python rank_oversold_predict_only.py --dir oversold_analysis --threshold 0.5
  python rank_oversold_predict_only.py --no-email
"""

from __future__ import annotations

import argparse
import json
import os
import re
import smtplib
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# -----------------------------
# Defaults
# -----------------------------
HOLDOUT_DAYS = 7
MIN_HISTORY_HITS = 3
FEATURE_SCHEMA_VERSION = "v1"

# ============================================================
# EMAIL CONFIG (keeps password default as requested)
# ============================================================
RECIPIENT_EMAIL = os.getenv("OVERSOLD_RECIPIENT_EMAIL", "mina.moussa@hotmail.com")
SENDER_EMAIL = os.getenv("OVERSOLD_SENDER_EMAIL", "minamoussa903@gmail.com")
SENDER_PASSWORD = os.getenv("OVERSOLD_SENDER_PASSWORD", "qhvi syra bbad gylu")  # kept per your request
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def send_email_with_analysis(body: str, df: Optional[pd.DataFrame] = None, directory: str = ".") -> None:
    """
    Sends email with body. Attaches filtered_output.csv if df has rows.
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
# OVERSOLD FILE PARSING
# ============================================================
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class OversoldHit:
    symbol: str
    file_date: date  # scan day from filename
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


def last_holdout_file_dates_from_hits(hits: List[OversoldHit], holdout_days: int) -> List[date]:
    file_dates = sorted({h.file_date for h in hits})
    if not file_dates:
        return []
    return file_dates[-int(holdout_days):]


# ============================================================
# FEATURE ENGINEERING (history-only, from hits)
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


def build_predict_feature_table(
    hits: List[OversoldHit],
    *,
    predict_dates_list: List[date],
    min_history_hits: int,
) -> pd.DataFrame:
    """
    Builds features ONLY for predict_dates_list. No labels.
    Uses full hit history to compute windowed stats.
    """
    by_symbol: Dict[str, List[OversoldHit]] = defaultdict(list)
    symbols_in_file: Dict[date, set] = defaultdict(set)

    for h in hits:
        by_symbol[h.symbol].append(h)
        symbols_in_file[h.file_date].add(h.symbol)

    rows: List[Dict] = []
    predict_set = set(predict_dates_list)

    for fday in sorted(predict_set):
        syms = symbols_in_file.get(fday, set())
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

            row = {"file_date": fday.isoformat(), "symbol": sym}
            row.update(feats)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["file_date", "symbol"]).reset_index(drop=True)
    return df


# ============================================================
# SMALL PREDICT-FEATURE CACHE (hash)
# ============================================================
def _stable_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _oversold_folder_fingerprint(folder: Path) -> dict:
    folder = Path(folder)
    files = sorted(folder.glob("*_oversold.txt"))
    items = []
    for p in files:
        try:
            st = p.stat()
            items.append((p.name, _stable_int(st.st_size), _stable_int(st.st_mtime)))
        except Exception:
            items.append((p.name, 0, 0))

    import hashlib
    blob = json.dumps(items, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    sig = hashlib.sha1(blob).hexdigest()
    return {"count": len(files), "sig": sig}


def _predict_feat_cache_key(
    *,
    folder: Path,
    holdout_days: int,
    min_history_hits: int,
    feature_schema_version: str,
) -> str:
    fp = _oversold_folder_fingerprint(folder)
    import hashlib
    payload = {
        "kind": "predict_features",
        "feature_schema_version": str(feature_schema_version),
        "oversold_folder": str(Path(folder).resolve()),
        "folder_sig": fp["sig"],
        "folder_count": fp["count"],
        "holdout_days": int(holdout_days),
        "min_history_hits": int(min_history_hits),
    }
    s = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def _cache_paths(cache_root: Path, key: str) -> dict:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    return {
        "meta": cache_root / f"pred_feats_{key}.meta.json",
        "df": cache_root / f"pred_feats_{key}.parquet",
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_cached_pred_feats(cache_root: Path, key: str) -> Optional[Tuple[pd.DataFrame, dict]]:
    paths = _cache_paths(cache_root, key)
    if (not paths["meta"].exists()) or (not paths["df"].exists()):
        return None
    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        df = pd.read_parquet(paths["df"])
        if df is None or df.empty:
            return None
        return df, meta
    except Exception:
        return None


def save_cached_pred_feats(cache_root: Path, key: str, df: pd.DataFrame, meta: dict) -> None:
    paths = _cache_paths(cache_root, key)
    tmp_df = paths["df"].with_suffix(".tmp.parquet")
    df.to_parquet(tmp_df, index=False)
    os.replace(tmp_df, paths["df"])
    _atomic_write_text(paths["meta"], json.dumps(meta, indent=2, sort_keys=True))


def get_or_build_pred_feats_cached(
    hits: List[OversoldHit],
    *,
    folder: Path,
    cache_root: Path,
    holdout_days: int,
    min_history_hits: int,
    feature_schema_version: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[date], dict]:
    predict_dates_list = last_holdout_file_dates_from_hits(hits, holdout_days)
    if not predict_dates_list:
        raise RuntimeError("No predict dates found from oversold hits.")

    key = _predict_feat_cache_key(
        folder=folder,
        holdout_days=holdout_days,
        min_history_hits=min_history_hits,
        feature_schema_version=feature_schema_version,
    )

    cached = load_cached_pred_feats(cache_root, key)
    if cached is not None:
        df, meta = cached
        if verbose:
            print(f"[CACHE] Predict features hit: {cache_root}  key={key}  rows={len(df)}", flush=True)
        return df, predict_dates_list, meta

    if verbose:
        print(f"[CACHE] Predict features miss: building...  key={key}", flush=True)

    df_pred_feats = build_predict_feature_table(
        hits,
        predict_dates_list=predict_dates_list,
        min_history_hits=min_history_hits,
    )
    if df_pred_feats.empty:
        raise RuntimeError("Predict feature table is empty.")

    meta = {
        "predict_dates_list": [d.isoformat() for d in predict_dates_list],
        "saved_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "feature_schema_version": str(feature_schema_version),
        "holdout_days": int(holdout_days),
        "min_history_hits": int(min_history_hits),
        "folder_fingerprint": _oversold_folder_fingerprint(folder),
    }
    save_cached_pred_feats(cache_root, key, df_pred_feats, meta)
    if verbose:
        print(f"[CACHE] Predict features saved: {cache_root}  key={key}  rows={len(df_pred_feats)}", flush=True)

    return df_pred_feats, predict_dates_list, meta


# ============================================================
# MODEL LOAD + PREDICT
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_BUNDLE_PATH = BASE_DIR / "models" / "default_model_76.joblib"
ROLLING_BUNDLE_PATH = BASE_DIR / "models" / "rolling_bundle.joblib"



print(DEFAULT_MODEL_BUNDLE_PATH,ROLLING_BUNDLE_PATH)

def load_model_bundle(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        obj = joblib.load(path)
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        print(f"[MODEL] Failed to load {path}: {e}", file=sys.stderr)
        return None


def load_bundle_models() -> tuple[object, list[str], object, list[str], dict]:
    """
    Loads the *default* bundle (as per your debug output) and returns:
      (main_model, main_feature_cols, down_model, down_feature_cols, meta)

    Expects bundle keys:
      - 'main_model'
      - 'main_feature_cols'
      - 'down_model'
      - 'down_feature_cols'
      - 'meta'

    This is predict-only safe (no training).
    """
    bundle = load_model_bundle(DEFAULT_MODEL_BUNDLE_PATH)
    if bundle is None:
        raise RuntimeError(f"No model bundle found at: {DEFAULT_MODEL_BUNDLE_PATH}")

    missing_keys = [k for k in ["main_model", "main_feature_cols", "down_model", "down_feature_cols", "meta"] if k not in bundle]
    if missing_keys:
        raise RuntimeError(f"Model bundle missing keys: {missing_keys}. Found keys={list(bundle.keys())}")

    main_model = bundle["main_model"]
    main_feature_cols = bundle["main_feature_cols"]
    down_model = bundle["down_model"]
    down_feature_cols = bundle["down_feature_cols"]
    meta = bundle.get("meta", {}) if isinstance(bundle.get("meta", {}), dict) else {}

    if not hasattr(main_model, "predict_proba"):
        raise RuntimeError(f"bundle['main_model'] has no predict_proba(): {type(main_model)}")
    if not hasattr(down_model, "predict_proba"):
        raise RuntimeError(f"bundle['down_model'] has no predict_proba(): {type(down_model)}")

    if not isinstance(main_feature_cols, list) or not main_feature_cols or not all(isinstance(x, str) for x in main_feature_cols):
        raise RuntimeError("bundle['main_feature_cols'] is not a non-empty list[str].")
    if not isinstance(down_feature_cols, list) or not down_feature_cols or not all(isinstance(x, str) for x in down_feature_cols):
        raise RuntimeError("bundle['down_feature_cols'] is not a non-empty list[str].")

    return main_model, list(main_feature_cols), down_model, list(down_feature_cols), meta


def predict_on_pred_feats(
    *,
    main_model,
    main_feature_cols: list[str],
    down_model,
    down_feature_cols: list[str],
    df_pred_feats: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Predict-only pipeline using BOTH models:

      1) Compute feat_prob_drop10 using down_model on down_feature_cols
      2) Predict prob_up_target_h using main_model on main_feature_cols (which includes feat_prob_drop10)
      3) Add pred_buy by threshold
      4) Return full df (sorted), caller can filter pred_buy==1

    Raises if any required feature columns are missing (prevents silent "rubbish data").
    """
    if df_pred_feats is None or df_pred_feats.empty:
        return pd.DataFrame()

    df = df_pred_feats.copy()

    # ---- Step 1: downside helper feature ----
    miss_down = [c for c in down_feature_cols if c not in df.columns]
    if miss_down:
        raise RuntimeError(f"Prediction failed: missing feature columns for down_model: {miss_down}")

    X_down = df[down_feature_cols].astype(float).values
    df["feat_prob_drop10"] = down_model.predict_proba(X_down)[:, 1]

    # ---- Step 2: main model prediction ----
    miss_main = [c for c in main_feature_cols if c not in df.columns]
    if miss_main:
        raise RuntimeError(f"Prediction failed: missing feature columns for main_model: {miss_main}")

    X_main = df[main_feature_cols].astype(float).values
    df["prob_up_target_h"] = main_model.predict_proba(X_main)[:, 1]
    df["pred_buy"] = (df["prob_up_target_h"] >= float(threshold)).astype(int)

    # nice ordering
    df = df.sort_values(["file_date", "prob_up_target_h"], ascending=[True, False]).reset_index(drop=True)
    return df



# ============================================================
# EMAIL BODY
# ============================================================
def build_email_body(
    *,
    loaded_hits: int,
    pred_feat_rows: int,
    predict_dates_list: List[date],
    df_pred: pd.DataFrame,
    threshold: float,
    holdout_days: int,
    cache_meta: Optional[dict] = None,
    model_meta: Optional[dict] = None,
    topk: int = 40,
) -> str:
    lines: List[str] = []
    lines.append("Oversold Analysis Alert")
    lines.append("=" * 60)
    lines.append(f"Loaded hits: {loaded_hits}")
    lines.append(f"Prediction feature rows: {pred_feat_rows}")

    if predict_dates_list:
        lines.append(
            f"Predict dates: {predict_dates_list[0]} .. {predict_dates_list[-1]}  ({len(predict_dates_list)} days)"
        )
    else:
        lines.append("Predict dates: (none)")

    lines.append(f"holdout_days: {holdout_days}")
    lines.append(f"threshold:    {threshold}")

    if cache_meta:
        lines.append("")
        lines.append(f"pred_feats_cache_saved_utc: {cache_meta.get('saved_utc', '')}")

    if model_meta:
        lines.append("")
        for k in ["created_utc", "sklearn_version", "train_end_date", "predict_start_date"]:
            if k in model_meta:
                lines.append(f"{k}: {model_meta.get(k)}")

    lines.append("")
    lines.append("=== TOP CANDIDATES (predict window) ===")

    if df_pred is None or df_pred.empty:
        lines.append("(none)")
        return "\n".join(lines)

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

        lines.append(f"\nfile={d_str}  (top {len(dd)})  threshold={threshold}")
        for _, r in dd.iterrows():
            lines.append(
                f"BUY  {str(r['symbol']):12s}  p={_fmt_prob(r.get('prob_up_target_h', np.nan))}  "
                f"hits30d={int(r.get('hits_30d', 0) or 0)}"
            )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
    ap.add_argument("--min-history-hits", type=int, default=MIN_HISTORY_HITS)
    ap.add_argument("--dir", default="oversold_analysis", help="Folder containing *_oversold.txt")
    ap.add_argument("--cache-dir", default="kline_store", help="Cache folder root")
    ap.add_argument("--no-email", action="store_true", help="Do not send email")
    args = ap.parse_args()

    folder = Path(args.dir)
    cache_root = Path(args.cache_dir) / "_pred_feat_cache"

    # 1) Load hits
    hits = load_oversold_hits(folder)
    print(f"Loaded hits: {len(hits)}")

    # 2) Build/cache prediction features ONLY for last HOLDOUT_DAYS
    df_pred_feats, predict_dates_list, cache_meta = get_or_build_pred_feats_cached(
        hits,
        folder=folder,
        cache_root=cache_root,
        holdout_days=int(args.holdout_days),
        min_history_hits=int(args.min_history_hits),
        feature_schema_version=str(FEATURE_SCHEMA_VERSION),
        verbose=True,
    )
    print(f"Predict feature rows: {len(df_pred_feats)}")
    print(
        f"Predict dates: {predict_dates_list[0]} .. {predict_dates_list[-1]}  ({len(predict_dates_list)} days)"
    )

    # 3) Load model
    main_model, main_feature_cols, down_model, down_feature_cols, model_meta = load_bundle_models()

    # 4) Predict on those rows
    df_pred_all = predict_on_pred_feats(
        main_model=main_model,
        main_feature_cols=main_feature_cols,
        down_model=down_model,
        down_feature_cols=down_feature_cols,
        df_pred_feats=df_pred_feats,
        threshold=float(args.threshold),
    )
    df_pred = df_pred_all[df_pred_all["pred_buy"] == 1].copy()

    out_path = Path("ai_predictions.csv")
    df_pred.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}  rows={len(df_pred)}")

    # 5) Email / console output
    body = build_email_body(
        loaded_hits=len(hits),
        pred_feat_rows=len(df_pred_feats),
        predict_dates_list=predict_dates_list,
        df_pred=df_pred,
        threshold=float(args.threshold),
        holdout_days=int(args.holdout_days),
        cache_meta=cache_meta,
        model_meta=model_meta,
        topk=40,
    )
    print("\n" + body)

    if not args.no_email:
        send_email_with_analysis(body, df=df_pred, directory=str(folder))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
