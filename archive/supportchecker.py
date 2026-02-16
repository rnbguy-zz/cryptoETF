#!/usr/bin/env python3
"""
grid_gate_sup1_label_only.py
--------------------------------
Grid search gate_sup1 (support of class 1 in validation split) WITHOUT model training.

Varies:
  - horizon_days
  - target_pct
  - min_history_hits
  - test_size
  - offline_cache_only

Uses:
  - oversold_analysis/*_oversold.txt for (file_date, symbol) hits
  - cached klines in <cache_dir>/<SYMBOL>_1d.parquet (same store format as your script)

Optionally can fetch missing candles from Binance when offline_cache_only=False (set --allow-fetch),
but the fast path is: run your normal script once to fill the cache, then grid search offline.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BINANCE_BASE = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"


# -------------------------
# OVERSOLD PARSING
# -------------------------
@dataclass(frozen=True)
class OversoldHit:
    symbol: str
    file_date: date


def parse_file_date(path: Path) -> date:
    return datetime.strptime(path.name[:10], "%Y-%m-%d").date()


def parse_oversold_line(line: str) -> Optional[str]:
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
        symbol = parts[5].upper()
    except Exception:
        return None

    return symbol


def load_hits(folder: Path) -> List[OversoldHit]:
    files = sorted(folder.glob("*_oversold.txt"))
    if not files:
        raise SystemExit(f"No *_oversold.txt files found in {folder}")

    hits: List[OversoldHit] = []
    for fp in files:
        fday = parse_file_date(fp)
        for line in fp.read_text(errors="ignore").splitlines():
            sym = parse_oversold_line(line)
            if not sym:
                continue
            hits.append(OversoldHit(symbol=sym, file_date=fday))
    return hits


# -------------------------
# KLINE CACHE HELPERS
# -------------------------
def _read_symbol_parquet(store_dir: Path, symbol: str) -> pd.DataFrame:
    p = store_dir / f"{symbol}_1d.parquet"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_parquet(p)
    if df.empty:
        return df

    df = df.copy()
    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
    if "no_data" in df.columns:
        df = df[~df["no_data"].astype(bool)]
    # keep only what we need
    cols = [c for c in ["day", "open", "high", "low", "close"] if c in df.columns]
    df = df[cols].dropna(subset=["day"]).sort_values("day").drop_duplicates("day", keep="last")
    return df


def _fetch_1d(symbol: str, start_day: date, end_day: date, session: requests.Session) -> pd.DataFrame:
    if end_day < start_day:
        return pd.DataFrame()

    # Binance allows up to 1000 rows; for 1d this is enough for ~1000 days.
    start_dt = datetime.combine(start_day, datetime.min.time())
    end_dt = datetime.combine(end_day + timedelta(days=1), datetime.min.time())

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

    data = r.json()
    rows = []
    for k in data or []:
        try:
            d = datetime.utcfromtimestamp(int(k[0]) / 1000.0).date()
            rows.append(
                {
                    "day": d,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                }
            )
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.date
        df = df.dropna(subset=["day"]).sort_values("day").drop_duplicates("day", keep="last")
    return df


def _ensure_range(
    symbol: str,
    start_day: date,
    end_day: date,
    *,
    store_dir: Path,
    sym_cache: Dict[str, pd.DataFrame],
    offline_only: bool,
    allow_fetch: bool,
    session: Optional[requests.Session],
) -> pd.DataFrame:
    df = sym_cache.get(symbol)
    if df is None:
        df = _read_symbol_parquet(store_dir, symbol)
        sym_cache[symbol] = df

    if df.empty:
        if offline_only or (not allow_fetch):
            return df
        fetched = _fetch_1d(symbol, start_day, end_day, session=session)  # type: ignore[arg-type]
        if fetched.empty:
            return df
        df = fetched
    else:
        have_min = df["day"].min()
        have_max = df["day"].max()
        need_fetch = (start_day < have_min) or (end_day > have_max)
        if need_fetch and (not offline_only) and allow_fetch:
            fetched = _fetch_1d(symbol, start_day, end_day, session=session)  # type: ignore[arg-type]
            if not fetched.empty:
                df = pd.concat([df, fetched], ignore_index=True)
                df = df.sort_values("day").drop_duplicates("day", keep="last")

    # write back if we fetched anything and we're allowed
    if (not offline_only) and allow_fetch and (not df.empty):
        outp = store_dir / f"{symbol}_1d.parquet"
        store_dir.mkdir(parents=True, exist_ok=True)
        df.assign(day=df["day"].astype(str)).to_parquet(outp, index=False)

    sym_cache[symbol] = df
    return df


# -------------------------
# LABEL ONLY
# -------------------------
def label_hit_target(
    *,
    symbol: str,
    file_day: date,
    entry_lag_days: int,
    horizon_days: int,
    target_pct: float,
    last_closed: date,
    store_dir: Path,
    sym_cache: Dict[str, pd.DataFrame],
    offline_only: bool,
    allow_fetch: bool,
    session: Optional[requests.Session],
) -> Optional[int]:
    entry_day = file_day + timedelta(days=entry_lag_days)
    end_day = min(entry_day + timedelta(days=horizon_days), last_closed)
    if end_day < entry_day:
        return None

    df = _ensure_range(
        symbol, entry_day, end_day,
        store_dir=store_dir,
        sym_cache=sym_cache,
        offline_only=offline_only,
        allow_fetch=allow_fetch,
        session=session,
    )

    if df is None or df.empty:
        return None

    w = df[(df["day"] >= entry_day) & (df["day"] <= end_day)]
    if w.empty:
        return None

    try:
        entry_open = float(w.iloc[0]["open"])
        max_close = float(pd.to_numeric(w["close"], errors="coerce").max())
        if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(max_close):
            return None
        return 1 if (max_close >= entry_open * (1.0 + target_pct)) else 0
    except Exception:
        return None


# -------------------------
# BUILD EXAMPLES FOR A min_history_hits
# -------------------------
def build_examples(
    *,
    file_dates_sorted: List[date],
    syms_in_file: Dict[date, List[str]],
    by_sym_file_dates: Dict[str, List[date]],
    train_end_date: date,
    min_history_hits: int,
) -> List[Tuple[date, str]]:
    train_dates = [d for d in file_dates_sorted if d <= train_end_date]
    examples: List[Tuple[date, str]] = []

    # fast counting via bisect on sorted per-symbol file_dates
    import bisect
    for fday in train_dates:
        for sym in syms_in_file.get(fday, []):
            fds = by_sym_file_dates.get(sym, [])
            # count hits <= fday
            n = bisect.bisect_right(fds, fday)
            if n >= min_history_hits:
                examples.append((fday, sym))
    return examples


# -------------------------
# MAIN GRID
# -------------------------
def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_float(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="oversold_analysis")
    ap.add_argument("--cache-dir", default="kline_store")
    ap.add_argument("--train-end-date", default="", help="YYYY-MM-DD. If blank, uses latest_file_date - 7 (like your rolling split).")
    ap.add_argument("--holdout-days", type=int, default=7, help="Used only if --train-end-date is blank.")
    ap.add_argument("--entry-lag-days", type=int, default=1)

    ap.add_argument("--horizons", default="5,7,10")
    ap.add_argument("--targets", default="0.20,0.15")
    ap.add_argument("--min-history-hits", default="3,2")
    ap.add_argument("--test-sizes", default="0.2")

    ap.add_argument("--offline-modes", default="0,1", help="0=offline_cache_only False, 1=True")
    ap.add_argument("--allow-fetch", action="store_true", help="If offline=0, fetch missing candles from Binance.")
    ap.add_argument("--dl-workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--top", type=int, default=20, help="Show top N by gate_sup1")
    args = ap.parse_args()

    folder = Path(args.dir)
    store_dir = Path(args.cache_dir)

    hits = load_hits(folder)
    file_dates = sorted({h.file_date for h in hits})
    if not file_dates:
        raise SystemExit("No file dates found.")
    latest_file_date = file_dates[-1]

    if args.train_end_date:
        train_end_date = datetime.strptime(args.train_end_date, "%Y-%m-%d").date()
    else:
        train_end_date = latest_file_date - timedelta(days=int(args.holdout_days))

    last_closed = datetime.utcnow().date() - timedelta(days=1)

    # precompute maps
    syms_in_file: Dict[date, List[str]] = defaultdict(list)
    by_sym_file_dates: Dict[str, List[date]] = defaultdict(list)
    for h in hits:
        syms_in_file[h.file_date].append(h.symbol)
        by_sym_file_dates[h.symbol].append(h.file_date)

    # sort unique and per symbol
    for d in list(syms_in_file.keys()):
        syms_in_file[d] = sorted(set(syms_in_file[d]))
    for sym in list(by_sym_file_dates.keys()):
        by_sym_file_dates[sym] = sorted(by_sym_file_dates[sym])

    horizons = parse_list_int(args.horizons)
    targets = parse_list_float(args.targets)
    mhs_list = parse_list_int(args.min_history_hits)
    test_sizes = parse_list_float(args.test_sizes)
    offline_modes = [bool(int(x.strip())) for x in args.offline_modes.split(",") if x.strip()]

    print(f"latest_file_date={latest_file_date}  train_end_date={train_end_date}  last_closed_utc={last_closed}")
    print(f"horizons={horizons} targets={targets} min_history_hits={mhs_list} test_sizes={test_sizes} offline_modes={offline_modes}")
    print(f"allow_fetch={bool(args.allow_fetch)}  dl_workers={args.dl_workers}\n")

    # build examples once per min_history_hits (huge speed win)
    examples_by_mh: Dict[int, List[Tuple[date, str]]] = {}
    for mh in mhs_list:
        exs = build_examples(
            file_dates_sorted=file_dates,
            syms_in_file=syms_in_file,
            by_sym_file_dates=by_sym_file_dates,
            train_end_date=train_end_date,
            min_history_hits=mh,
        )
        examples_by_mh[mh] = exs
        print(f"examples built: min_history_hits={mh} -> {len(exs)} examples")

    print("")

    results = []
    for offline_only in offline_modes:
        session = requests.Session() if (not offline_only and args.allow_fetch) else None

        # share symbol parquet cache across combos for this offline mode
        sym_cache: Dict[str, pd.DataFrame] = {}

        for mh in mhs_list:
            examples = examples_by_mh[mh]

            for horizon_days in horizons:
                for target_pct in targets:
                    # label once per (offline, mh, horizon, target)
                    y = []
                    missing = 0

                    def _job(pair: Tuple[date, str]) -> Optional[int]:
                        fday, sym = pair
                        return label_hit_target(
                            symbol=sym,
                            file_day=fday,
                            entry_lag_days=int(args.entry_lag_days),
                            horizon_days=int(horizon_days),
                            target_pct=float(target_pct),
                            last_closed=last_closed,
                            store_dir=store_dir,
                            sym_cache=sym_cache,
                            offline_only=offline_only,
                            allow_fetch=bool(args.allow_fetch),
                            session=session,
                        )

                    with ThreadPoolExecutor(max_workers=max(1, int(args.dl_workers))) as ex:
                        futs = [ex.submit(_job, pair) for pair in examples]
                        for fut in as_completed(futs):
                            lab = fut.result()
                            if lab is None:
                                missing += 1
                            else:
                                y.append(int(lab))

                    if not y:
                        for ts in test_sizes:
                            results.append({
                                "offline": offline_only,
                                "min_history_hits": mh,
                                "horizon_days": horizon_days,
                                "target_pct": target_pct,
                                "test_size": ts,
                                "evaluable": 0,
                                "missing": missing,
                                "pos_train": 0,
                                "gate_sup1": 0,
                                "pos_rate": 0.0,
                            })
                        continue

                    y = np.array(y, dtype=int)
                    pos_train = int((y == 1).sum())
                    evaluable = int(len(y))
                    pos_rate = float(pos_train / max(1, evaluable))

                    for ts in test_sizes:
                        # same mechanic as your training report uses (stratified split)
                        idx = np.arange(evaluable)
                        _, _, _, y_va = train_test_split(
                            idx, y,
                            test_size=float(ts),
                            random_state=int(args.seed),
                            stratify=y if (pos_train > 0 and pos_train < evaluable) else None,
                        )
                        gate_sup1 = int((y_va == 1).sum())

                        results.append({
                            "offline": offline_only,
                            "min_history_hits": mh,
                            "horizon_days": horizon_days,
                            "target_pct": target_pct,
                            "test_size": ts,
                            "evaluable": evaluable,
                            "missing": missing,
                            "pos_train": pos_train,
                            "gate_sup1": gate_sup1,
                            "pos_rate": pos_rate,
                        })

                        print(
                            f"offline={int(offline_only)} mh={mh} H={horizon_days} tgt={target_pct:.2f} ts={ts:.2f}  "
                            f"evaluable={evaluable} pos_train={pos_train} gate_sup1={gate_sup1} pos_rate={pos_rate:.4f} missing={missing}"
                        )

    # show top results by gate_sup1
    results_sorted = sorted(results, key=lambda r: r["gate_sup1"], reverse=True)
    topn = int(args.top)
    print("\n=== TOP COMBOS BY gate_sup1 ===")
    for r in results_sorted[:topn]:
        print(
            f"gate_sup1={r['gate_sup1']:5d}  offline={int(r['offline'])}  mh={r['min_history_hits']}  "
            f"H={r['horizon_days']:2d}  tgt={r['target_pct']:.2f}  ts={r['test_size']:.2f}  "
            f"pos_train={r['pos_train']:5d}  evaluable={r['evaluable']:6d}  pos_rate={r['pos_rate']:.4f}  missing={r['missing']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
