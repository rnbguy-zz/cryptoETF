#!/usr/bin/env python3
"""
gate_from_signals.py

Goal
----
Given a pasted "signals" text (same format as your rank_oversold output),
this script will:

1) Parse model meta (predict_from) + each file=YYYY-MM-DD block
   and extract: symbol, p, hits30d, featDropP.

2) "Calibration" (known-now) test:
   For the FIRST few predicted days (starting at predict_from),
   it will simulate entries (entry_day = file_date + entry_lag_days, at OPEN),
   then determine whether TP or SL hit FIRST using:
     - 1D candles
     - intraday candles ONLY when TP+SL hit the same day
   -> label each signal as TP-first, SL-first, or OTHER.

3) Grid-search gates (p_min, drop_max, hits_min) on that calibration subset,
   optimizing for HIGH precision (avoid SL-first).

4) Apply the chosen gates to the remaining "future" predicted blocks and print
   what to buy per day (highest p first).

Usage
-----
# Put your pasted text into signals.txt (exactly what you showed)
python3.9 gate_from_signals.py --signals-file signals.txt \
  --tp-pct 0.20 --sl-pct 0.12 --entry-lag-days 1 \
  --calibration-days 5 --intraday-interval 1m \
  --ambiguous-fallback sl-first

Notes
-----
- "Calibration days" are the first N file= blocks starting at model predict_from.
- If some calibration trades can't be evaluated (missing candles), they're ignored.
- This is *not* changing your model. It's creating a practical "gate" filter:
    BUY if (p >= p_min) and (featDropP <= drop_max) and (hits30d >= hits_min)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# =============================================================================
# Parsing (signals text)
# =============================================================================

FILE_RE = re.compile(r"^\s*file=(\d{4}-\d{2}-\d{2})\b")
META_PREDICT_FROM_RE = re.compile(r"^\s*Model\s+predict_from:\s*(\d{4}-\d{2}-\d{2})\s*$", re.I)

BUY_LINE_RE = re.compile(r"^\s*BUY\s+([A-Z0-9]+)\b")
P_RE = re.compile(r"\bp\s*=\s*([0-9]*\.?[0-9]+)\b")
HITS_RE = re.compile(r"\bhits30d\s*=\s*(\d+)\b")
DROP_RE = re.compile(r"\bfeatDropP\s*=\s*([0-9]*\.?[0-9]+)\b")
META_TRAINED_THROUGH_RE = re.compile(r"^\s*Model\s+trained_through:\s*(\d{4}-\d{2}-\d{2})\s*$", re.I)


@dataclass(frozen=True)
class SignalRow:
    file_date: date
    symbol: str
    p: float
    hits30d: int
    drop_p: float
    raw_line: str


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_signals_text(text: str) -> Tuple[Optional[date], Optional[date], List[SignalRow], Dict[date, List[str]]]:
    """
    Returns:
      trained_through (from header if present, else None)
      predict_from    (from header if present, else None)
      signals         (one row per BUY line)
      raw_blocks      (file_date -> list of original lines in that block)
    """
    trained_through: Optional[date] = None
    predict_from: Optional[date] = None
    cur_file_date: Optional[date] = None

    raw_blocks: Dict[date, List[str]] = {}
    out: List[SignalRow] = []

    for line in text.splitlines():
        mtrain = META_TRAINED_THROUGH_RE.match(line)
        if mtrain and trained_through is None:
            trained_through = _parse_date(mtrain.group(1))

        mpred = META_PREDICT_FROM_RE.match(line)
        if mpred and predict_from is None:
            predict_from = _parse_date(mpred.group(1))

        mfile = FILE_RE.match(line)
        if mfile:
            cur_file_date = _parse_date(mfile.group(1))
            raw_blocks.setdefault(cur_file_date, []).append(line)
            continue

        if cur_file_date is not None:
            if line.strip():
                raw_blocks.setdefault(cur_file_date, []).append(line)

            mbuy = BUY_LINE_RE.match(line)
            if mbuy:
                sym = mbuy.group(1).upper()

                mp = P_RE.search(line)
                mh = HITS_RE.search(line)
                md = DROP_RE.search(line)

                p = float(mp.group(1)) if mp else 0.0
                hits = int(mh.group(1)) if mh else 0
                drop_p = float(md.group(1)) if md else 1.0

                out.append(SignalRow(
                    file_date=cur_file_date,
                    symbol=sym,
                    p=p,
                    hits30d=hits,
                    drop_p=drop_p,
                    raw_line=line,
                ))

    out.sort(key=lambda r: (r.file_date, r.symbol))
    return trained_through, predict_from, out, raw_blocks



def load_signals_file(path: Path) -> Tuple[Optional[date], Optional[date], List[SignalRow], Dict[date, List[str]]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return parse_signals_text(text)



# =============================================================================
# Binance klines + caching
# =============================================================================

BINANCE_BASE = "https://api.binance.com"
KLINES_EP = "/api/v3/klines"


@dataclass
class Candle:
    day: date
    o: float
    h: float
    l: float
    c: float


@dataclass
class IntraCandle:
    ts_ms: int
    o: float
    h: float
    l: float
    c: float


def _utc_ms(d: datetime) -> int:
    return int(d.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(2.0, 0.25 * (2 ** (attempt - 1))))


def _session_get_json(session: requests.Session, url: str, params: dict, timeout: int = 30):
    last_err = None
    for attempt in range(1, 7):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 418):
                _sleep_backoff(attempt)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            _sleep_backoff(attempt)
    raise RuntimeError(f"HTTP failed after retries: {last_err}")


def fetch_1d_klines(symbol: str, start_day: date, end_day: date, session: requests.Session) -> List[Candle]:
    if end_day < start_day:
        return []

    out: List[Candle] = []
    cur = start_day

    while cur <= end_day:
        chunk_end = min(end_day, cur + timedelta(days=900))

        start_dt = datetime(cur.year, cur.month, cur.day, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = datetime(chunk_end.year, chunk_end.month, chunk_end.day, 0, 0, 0, tzinfo=timezone.utc) + timedelta(days=1)

        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": _utc_ms(start_dt),
            "endTime": _utc_ms(end_dt),
            "limit": 1000,
        }

        data = _session_get_json(session, BINANCE_BASE + KLINES_EP, params=params, timeout=30)
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected response for {symbol}: {type(data)}")

        for k in data:
            try:
                open_time_ms = int(k[0])
                d = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc).date()
                out.append(Candle(day=d, o=float(k[1]), h=float(k[2]), l=float(k[3]), c=float(k[4])))
            except Exception:
                continue

        cur = chunk_end + timedelta(days=1)

    by_day: Dict[date, Candle] = {}
    for c in out:
        by_day[c.day] = c

    candles = [by_day[d] for d in sorted(by_day.keys())]
    candles = [c for c in candles if start_day <= c.day <= end_day]
    return candles


def fetch_intraday_klines_for_day(
    symbol: str,
    day: date,
    interval: str,
    session: requests.Session,
) -> List[IntraCandle]:
    start_dt = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)

    start_ms = _utc_ms(start_dt)
    end_ms = _utc_ms(end_dt)

    out: List[IntraCandle] = []
    cur_start = start_ms

    while cur_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        data = _session_get_json(session, BINANCE_BASE + KLINES_EP, params=params, timeout=30)
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected intraday response for {symbol}: {type(data)}")

        if not data:
            break

        last_open_ms = None
        for k in data:
            try:
                open_time_ms = int(k[0])
                last_open_ms = open_time_ms
                out.append(IntraCandle(
                    ts_ms=open_time_ms,
                    o=float(k[1]), h=float(k[2]), l=float(k[3]), c=float(k[4])
                ))
            except Exception:
                continue

        if last_open_ms is None:
            break

        cur_start = last_open_ms + 1
        time.sleep(0.05)

    by_ts: Dict[int, IntraCandle] = {c.ts_ms: c for c in out}
    return [by_ts[t] for t in sorted(by_ts.keys())]


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache_path: Path, obj: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, cache_path)


def get_candles_cached(
    symbol: str,
    start_day: date,
    end_day: date,
    *,
    session: requests.Session,
    cache: dict,
    cache_path: Path,
) -> Dict[date, Candle]:
    sym = symbol.upper()
    cache.setdefault(sym, {}).setdefault("candles_1d", {})
    have = cache[sym]["candles_1d"]

    need_days = [start_day + timedelta(days=i) for i in range((end_day - start_day).days + 1)]
    missing = [d for d in need_days if d.isoformat() not in have]

    if missing:
        d0, d1 = min(missing), max(missing)
        fetched = fetch_1d_klines(sym, d0, d1, session=session)
        for c in fetched:
            have[c.day.isoformat()] = {"o": c.o, "h": c.h, "l": c.l, "c": c.c}
        save_cache(cache_path, cache)

    out: Dict[date, Candle] = {}
    for d in need_days:
        rec = have.get(d.isoformat())
        if rec:
            out[d] = Candle(day=d, o=float(rec["o"]), h=float(rec["h"]), l=float(rec["l"]), c=float(rec["c"]))
    return out


def get_intraday_cached_for_day(
    symbol: str,
    day: date,
    *,
    interval: str,
    session: requests.Session,
    cache: dict,
    cache_path: Path,
) -> List[IntraCandle]:
    sym = symbol.upper()
    cache.setdefault(sym, {}).setdefault("intraday", {})
    cache[sym]["intraday"].setdefault(interval, {})
    have = cache[sym]["intraday"][interval]

    key = day.isoformat()
    if key not in have:
        candles = fetch_intraday_klines_for_day(sym, day, interval=interval, session=session)
        have[key] = [[c.ts_ms, c.o, c.h, c.l, c.c] for c in candles]
        save_cache(cache_path, cache)

    raw = have.get(key, [])
    out: List[IntraCandle] = []
    for row in raw:
        try:
            out.append(IntraCandle(ts_ms=int(row[0]), o=float(row[1]), h=float(row[2]), l=float(row[3]), c=float(row[4])))
        except Exception:
            continue
    out.sort(key=lambda x: x.ts_ms)
    return out


# =============================================================================
# TP/SL evaluation (SL-before-TP logic)
# =============================================================================

def evaluate_tp_sl_first(
    *,
    symbol: str,
    entry_open: float,
    day: date,
    day_candle: Candle,
    tp_pct: float,
    sl_pct: float,
    session: requests.Session,
    cache: dict,
    cache_path: Path,
    intraday_interval: str,
    ambiguous_fallback: str,   # "none"|"tp-first"|"sl-first"
    resolve_ambiguous_intraday: bool,
) -> Optional[str]:
    """
    For a given day candle AFTER entry, determine if TP or SL hit first on that day.
    Returns "TP" or "SL" (or None if neither hit).
    """
    tp_price = entry_open * (1.0 + tp_pct)
    sl_price = entry_open * (1.0 - sl_pct)

    hit_tp_1d = day_candle.h >= tp_price
    hit_sl_1d = day_candle.l <= sl_price

    if hit_tp_1d and not hit_sl_1d:
        return "TP"
    if hit_sl_1d and not hit_tp_1d:
        return "SL"
    if not hit_tp_1d and not hit_sl_1d:
        return None

    # BOTH hit same day
    intra: List[IntraCandle] = []
    if resolve_ambiguous_intraday:
        intra = get_intraday_cached_for_day(
            symbol, day,
            interval=intraday_interval,
            session=session, cache=cache, cache_path=cache_path,
        )

    if not intra:
        if ambiguous_fallback == "tp-first":
            return "TP"
        if ambiguous_fallback == "sl-first":
            return "SL"
        return None

    for c in intra:
        hit_tp = c.h >= tp_price
        hit_sl = c.l <= sl_price

        if hit_tp and not hit_sl:
            return "TP"
        if hit_sl and not hit_tp:
            return "SL"
        if hit_tp and hit_sl:
            # can't resolve inside this intraday candle
            if ambiguous_fallback == "tp-first":
                return "TP"
            if ambiguous_fallback == "sl-first":
                return "SL"
            return None

    return None


def simulate_trade_outcome(
    *,
    symbol: str,
    entry_day: date,
    entry_open: float,
    candles_by_day: Dict[date, Candle],
    tp_pct: float,
    sl_pct: float,
    max_hold_days: int,
    session: requests.Session,
    cache: dict,
    cache_path: Path,
    intraday_interval: str,
    ambiguous_fallback: str,
    resolve_ambiguous_intraday: bool,
) -> str:
    """
    Walk forward from entry_day inclusive until TP or SL hit first; else OTHER.
    Returns: "TP", "SL", or "OTHER"
    """
    last_day = entry_day + timedelta(days=max_hold_days)
    d = entry_day
    while d <= last_day:
        c = candles_by_day.get(d)
        if c is None:
            d += timedelta(days=1)
            continue

        hit = evaluate_tp_sl_first(
            symbol=symbol,
            entry_open=entry_open,
            day=d,
            day_candle=c,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            session=session,
            cache=cache,
            cache_path=cache_path,
            intraday_interval=intraday_interval,
            ambiguous_fallback=ambiguous_fallback,
            resolve_ambiguous_intraday=resolve_ambiguous_intraday,
        )
        if hit in ("TP", "SL"):
            return hit
        d += timedelta(days=1)

    return "OTHER"


# =============================================================================
# Gate search
# =============================================================================

@dataclass(frozen=True)
class Gate:
    p_min: float
    drop_max: float
    hits_min: int


def passes_gate(sig: SignalRow, gate: Gate) -> bool:
    return (sig.p >= gate.p_min) and (sig.drop_p <= gate.drop_max) and (sig.hits30d >= gate.hits_min)


@dataclass
class CalibRow:
    sig: SignalRow
    outcome: str  # "TP" | "SL" | "OTHER"


def score_gate(rows: List[CalibRow], gate: Gate) -> Tuple[float, int, int, int]:
    """
    Returns: (precision, taken, tp, sl)
    Precision is computed on TP/(TP+SL) for taken rows where outcome in {TP,SL}.
    OTHER outcomes are counted in taken but do not affect precision denominator.
    """
    taken = 0
    tp = 0
    sl = 0
    for r in rows:
        if passes_gate(r.sig, gate):
            taken += 1
            if r.outcome == "TP":
                tp += 1
            elif r.outcome == "SL":
                sl += 1

    denom = tp + sl
    prec = (tp / denom) if denom > 0 else 0.0
    return prec, taken, tp, sl


def pick_best_gate(rows: List[CalibRow]) -> Tuple[Gate, Dict[str, float]]:
    # Coarse-but-safe grid (fast + avoids overfit)
    p_grid = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75]
    drop_grid = [0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
    hits_grid = [1, 2, 3, 4, 5, 8, 10, 12, 15]

    best_gate = Gate(0.60, 0.60, 3)
    best = (-1.0, -1, -1, 10**9)  # (precision, tp, taken, sl) with sl as tie-breaker (lower better)

    # Require at least some decided outcomes exist in calibration
    decided = [r for r in rows if r.outcome in ("TP", "SL")]
    if len(decided) == 0:
        # fallback gate (your current empiric)
        return best_gate, {"precision": 0.0, "taken": 0, "tp": 0, "sl": 0}

    for pmin in p_grid:
        for dmax in drop_grid:
            for hmin in hits_grid:
                gate = Gate(pmin, dmax, hmin)
                prec, taken, tp, sl = score_gate(rows, gate)

                # Hard constraints to avoid silly "precision=1.0 with 1 trade"
                if (tp + sl) < 2:
                    continue
                if tp < 1:
                    continue

                cand = (prec, tp, taken, sl)
                if cand[0] > best[0]:
                    best = cand
                    best_gate = gate
                elif cand[0] == best[0]:
                    # tie-break: prefer more TP
                    if cand[1] > best[1]:
                        best = cand
                        best_gate = gate
                    elif cand[1] == best[1]:
                        # then prefer fewer SL
                        if cand[3] < best[3]:
                            best = cand
                            best_gate = gate
                        elif cand[3] == best[3]:
                            # then prefer more taken (coverage)
                            if cand[2] > best[2]:
                                best = cand
                                best_gate = gate

    prec, taken, tp, sl = score_gate(rows, best_gate)
    return best_gate, {"precision": prec, "taken": taken, "tp": tp, "sl": sl}

# =============================================================================
# Replay (single-position) + compounding (ETF Algo style)
# =============================================================================

@dataclass
class ReplayTrade:
    symbol: str
    file_date: date          # signal day (file=...)
    entry_day: date
    entry_open: float
    exit_day: date
    exit_price: float
    exit_reason: str         # TP/SL/TP_AMBIG/SL_AMBIG/MAX_HOLD
    gain_pct: float


def _gain_pct(entry: float, exit_px: float) -> float:
    return (exit_px / max(entry, 1e-12) - 1.0) * 100.0


def _fmt_d(d: date) -> str:
    # Windows-friendly day/month formatting
    return d.strftime("%Y-%m-%d")


def evaluate_exit_reason_and_price(
    *,
    symbol: str,
    entry_open: float,
    day: date,
    day_candle: Candle,
    tp_pct: float,
    sl_pct: float,
    session: requests.Session,
    cache: dict,
    cache_path: Path,
    intraday_interval: str,
    ambiguous_fallback: str,
    resolve_ambiguous_intraday: bool,
) -> Optional[Tuple[str, float]]:
    """
    Returns (reason, exit_price) or None.
    Uses SAME logic as your ETF Algo: TP if hit, SL if hit, if both then intraday resolution else fallback.
    """
    tp_price = entry_open * (1.0 + tp_pct)
    sl_price = entry_open * (1.0 - sl_pct)

    hit_tp_1d = day_candle.h >= tp_price
    hit_sl_1d = day_candle.l <= sl_price

    if hit_tp_1d and not hit_sl_1d:
        return ("TP", tp_price)
    if hit_sl_1d and not hit_tp_1d:
        return ("SL", sl_price)
    if not hit_tp_1d and not hit_sl_1d:
        return None

    # BOTH hit same day
    intra: List[IntraCandle] = []
    if resolve_ambiguous_intraday:
        intra = get_intraday_cached_for_day(
            symbol, day,
            interval=intraday_interval,
            session=session, cache=cache, cache_path=cache_path,
        )

    if not intra:
        if ambiguous_fallback == "tp-first":
            return ("TP_AMBIG", tp_price)
        if ambiguous_fallback == "sl-first":
            return ("SL_AMBIG", sl_price)
        return None  # "none" -> unknown

    # scan intraday candles in time order
    for c in intra:
        hit_tp = c.h >= tp_price
        hit_sl = c.l <= sl_price

        if hit_tp and not hit_sl:
            return ("TP", tp_price)
        if hit_sl and not hit_tp:
            return ("SL", sl_price)
        if hit_tp and hit_sl:
            # still ambiguous within this intraday candle
            if ambiguous_fallback == "tp-first":
                return ("TP_AMBIG", tp_price)
            if ambiguous_fallback == "sl-first":
                return ("SL_AMBIG", sl_price)
            return None

    return None


def replay_single_position_from_gated_picks(
    *,
    gated_picks_by_file_date: Dict[date, SignalRow],  # one pick per file day (#01 per day)
    entry_lag_days: int,
    tp_pct: float,
    sl_pct: float,
    max_hold_days: int,
    cache_path: Path,
    intraday_interval: str,
    ambiguous_fallback: str,
    resolve_ambiguous_intraday: bool,
    end_day: Optional[date] = None,  # default today UTC
) -> Tuple[List["ReplayTrade"], float, Optional[dict]]:
    """
    Single-position replay (NO ROLLING):
      - If FLAT: enter the pick for file_date on entry_day = file_date + entry_lag_days (at OPEN).
      - If HOLDING: ignore ALL new picks until exit.
      - Exit ONLY on TP/SL (intraday resolve when both hit same day) OR max_hold_days at CLOSE.

    Returns:
      trades            List[ReplayTrade] for CLOSED trades only
      compounded_factor Product of (1 + gain) for CLOSED trades only
      open_pos          dict describing OPEN position at end_day (or None)
    """
    if not gated_picks_by_file_date:
        return [], 1.0, None

    session = requests.Session()
    cache = load_cache(cache_path)

    file_days = sorted(gated_picks_by_file_date.keys())
    start_entry_day = min(fd + timedelta(days=entry_lag_days) for fd in file_days)

    if end_day is None:
        end_day = datetime.utcnow().date()

    # All symbols that could be traded (from the gated picks)
    symbols = sorted({s.symbol for s in gated_picks_by_file_date.values()})

    # Fetch candles for each symbol from the first possible entry day through end_day
    candles_by_symbol: Dict[str, Dict[date, Candle]] = {}
    for sym in symbols:
        candles_by_symbol[sym] = get_candles_cached(
            sym,
            start_entry_day,
            end_day,
            session=session,
            cache=cache,
            cache_path=cache_path,
        )

    trades: List[ReplayTrade] = []
    compounded = 1.0

    holding_symbol: Optional[str] = None
    holding_entry_open: float = 0.0
    holding_entry_day: Optional[date] = None
    holding_file_date: Optional[date] = None

    d = start_entry_day
    while d <= end_day:
        # 1) If holding, check exit (TP/SL first; else MAX_HOLD at close)
        if holding_symbol is not None and holding_entry_day is not None and holding_file_date is not None:
            c = candles_by_symbol.get(holding_symbol, {}).get(d)
            if c is not None and d >= holding_entry_day:
                hit = evaluate_exit_reason_and_price(
                    symbol=holding_symbol,
                    entry_open=holding_entry_open,
                    day=d,
                    day_candle=c,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    session=session,
                    cache=cache,
                    cache_path=cache_path,
                    intraday_interval=intraday_interval,
                    ambiguous_fallback=ambiguous_fallback,
                    resolve_ambiguous_intraday=resolve_ambiguous_intraday,
                )

                reason: Optional[str] = None
                exit_px: Optional[float] = None

                if hit is not None:
                    reason, exit_px = hit
                else:
                    if max_hold_days > 0 and d >= (holding_entry_day + timedelta(days=max_hold_days)):
                        reason = "MAX_HOLD"
                        exit_px = c.c

                if reason is not None and exit_px is not None:
                    g = _gain_pct(holding_entry_open, float(exit_px))
                    compounded *= (1.0 + g / 100.0)

                    trades.append(
                        ReplayTrade(
                            symbol=holding_symbol,
                            file_date=holding_file_date,
                            entry_day=holding_entry_day,
                            entry_open=holding_entry_open,
                            exit_day=d,
                            exit_price=float(exit_px),
                            exit_reason=reason,
                            gain_pct=g,
                        )
                    )

                    # go FLAT
                    holding_symbol = None
                    holding_entry_open = 0.0
                    holding_entry_day = None
                    holding_file_date = None

        # 2) If FLAT, see if a pick becomes active today (entry_day == d)
        if holding_symbol is None:
            candidates: List[SignalRow] = []
            for fd, sig in gated_picks_by_file_date.items():
                if fd + timedelta(days=entry_lag_days) == d:
                    candidates.append(sig)

            if candidates:
                # If multiple map to same entry day, take highest p (tie-break lower drop, higher hits)
                candidates.sort(key=lambda s: (-s.p, s.drop_p, -s.hits30d, s.symbol))
                chosen = candidates[0]

                c_entry = candles_by_symbol.get(chosen.symbol, {}).get(d)
                if c_entry is not None:
                    holding_symbol = chosen.symbol
                    holding_entry_open = c_entry.o
                    holding_entry_day = d
                    holding_file_date = chosen.file_date

        d += timedelta(days=1)

    # Build open position summary (if still holding at end_day)
    open_pos: Optional[dict] = None
    if holding_symbol is not None and holding_entry_day is not None and holding_file_date is not None:
        c_last = candles_by_symbol.get(holding_symbol, {}).get(end_day)
        last_close = c_last.c if c_last else holding_entry_open
        unreal = (last_close / max(holding_entry_open, 1e-12) - 1.0) * 100.0

        open_pos = {
            "symbol": holding_symbol,
            "file_date": holding_file_date,
            "entry_day": holding_entry_day,
            "entry_open": float(holding_entry_open),
            "last_day": end_day,
            "last_close": float(last_close),
            "unrealized_pct": float(unreal),
        }

    return trades, compounded, open_pos

# =============================================================================
# Binance LIVE trading (Spot) + OCO exits
# Drop these functions/classes into your script (gate_from_signals.py)
# =============================================================================

import hashlib
import hmac
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlencode

BINANCE_BASE = "https://api.binance.com"


class BinanceHTTPError(RuntimeError):
    pass


class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = BINANCE_BASE) -> None:
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

        self._time_offset_ms = 0
        self._sync_time()

    def _sync_time(self) -> None:
        try:
            j = self.session.get(self.base_url + "/api/v3/time", timeout=10).json()
            server_ms = int(j["serverTime"])
            local_ms = int(time.time() * 1000)
            self._time_offset_ms = server_ms - local_ms
        except Exception:
            self._time_offset_ms = 0

    def _now_ms(self) -> int:
        return int(time.time() * 1000) + int(self._time_offset_ms)

    def _sign_query(self, params: dict) -> str:
        qs = urlencode(params, doseq=True)
        sig = hmac.new(self.api_secret, qs.encode("utf-8"), hashlib.sha256).hexdigest()
        return qs + "&signature=" + sig

    def _raise_binance(self, r: requests.Response) -> None:
        try:
            body = r.text
        except Exception:
            body = "<no body>"
        raise BinanceHTTPError(f"HTTP {r.status_code} {r.reason} for {r.url}\nResponse: {body}")

    def _get(self, path: str, params: dict, signed: bool = False, timeout: int = 30):
        url = self.base_url + path
        if not signed:
            r = self.session.get(url, params=params, timeout=timeout)
            if not r.ok:
                self._raise_binance(r)
            return r.json()

        p = dict(params)
        p["timestamp"] = self._now_ms()
        p.setdefault("recvWindow", 20000)

        qs = self._sign_query(p)
        full_url = url + "?" + qs
        r = self.session.get(full_url, timeout=timeout)

        # auto re-sync time on -1021
        if r.status_code == 400:
            try:
                j = r.json()
                if isinstance(j, dict) and j.get("code") == -1021:
                    self._sync_time()
                    p["timestamp"] = self._now_ms()
                    qs = self._sign_query(p)
                    full_url = url + "?" + qs
                    r = self.session.get(full_url, timeout=timeout)
            except Exception:
                pass

        if not r.ok:
            self._raise_binance(r)
        return r.json()

    def _post(self, path: str, params: dict, signed: bool = True, timeout: int = 30):
        url = self.base_url + path
        if not signed:
            r = self.session.post(url, params=params, timeout=timeout)
            if not r.ok:
                self._raise_binance(r)
            return r.json()

        p = dict(params)
        p["timestamp"] = self._now_ms()
        p.setdefault("recvWindow", 20000)

        qs = self._sign_query(p)
        full_url = url + "?" + qs
        r = self.session.post(full_url, timeout=timeout)

        if r.status_code == 400:
            try:
                j = r.json()
                if isinstance(j, dict) and j.get("code") == -1021:
                    self._sync_time()
                    p["timestamp"] = self._now_ms()
                    qs = self._sign_query(p)
                    full_url = url + "?" + qs
                    r = self.session.post(full_url, timeout=timeout)
            except Exception:
                pass

        if not r.ok:
            self._raise_binance(r)
        return r.json()

    # -------- endpoints --------

    def exchange_info(self, symbol: str) -> dict:
        return self._get("/api/v3/exchangeInfo", {"symbol": symbol}, signed=False)

    def account(self) -> dict:
        return self._get("/api/v3/account", {}, signed=True)

    def open_orders(self, symbol: str) -> list:
        return self._get("/api/v3/openOrders", {"symbol": symbol}, signed=True)

    def ticker_price(self, symbol: str) -> float:
        j = self._get("/api/v3/ticker/price", {"symbol": symbol}, signed=False)
        return float(j["price"])

    def market_buy_quote_qty(self, symbol: str, quote_qty: Decimal) -> dict:
        return self._post(
            "/api/v3/order",
            {
                "symbol": symbol,
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": f"{quote_qty:f}",
                "newOrderRespType": "FULL",
            },
            signed=True,
        )

    def oco_sell(self, symbol: str, quantity: Decimal, tp_price: Decimal, sl_stop: Decimal, sl_limit: Decimal) -> dict:
        return self._post(
            "/api/v3/order/oco",
            {
                "symbol": symbol,
                "side": "SELL",
                "quantity": f"{quantity:f}",
                "price": f"{tp_price:f}",
                "stopPrice": f"{sl_stop:f}",
                "stopLimitPrice": f"{sl_limit:f}",
                "stopLimitTimeInForce": "GTC",
            },
            signed=True,
        )


def _dec(x: float) -> Decimal:
    return Decimal(str(x))


def _round_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _extract_symbol_filters(exchange_info_json: dict) -> Tuple[Decimal, Decimal]:
    symbols = exchange_info_json.get("symbols") or []
    if not symbols:
        raise RuntimeError("exchangeInfo missing symbols[]")
    filters = symbols[0].get("filters") or []

    tick_size = None
    step_size = None
    for flt in filters:
        if flt.get("filterType") == "PRICE_FILTER":
            tick_size = Decimal(str(flt.get("tickSize", "0")))
        elif flt.get("filterType") == "LOT_SIZE":
            step_size = Decimal(str(flt.get("stepSize", "0")))

    if tick_size is None or step_size is None:
        raise RuntimeError("Could not find PRICE_FILTER / LOT_SIZE in exchangeInfo")
    return tick_size, step_size


def _base_asset_from_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol.endswith("USDT"):
        return symbol[:-4]
    raise ValueError("This live logic assumes USDT quote pairs only (symbol endswith USDT).")


def _get_free_usdt(acct_json: dict) -> Decimal:
    for b in acct_json.get("balances", []):
        if b.get("asset") == "USDT":
            try:
                return Decimal(str(b.get("free", "0")))
            except Exception:
                return Decimal("0")
    return Decimal("0")


def _get_free_asset(acct_json: dict, asset: str) -> Decimal:
    for b in acct_json.get("balances", []):
        if b.get("asset") == asset:
            try:
                return Decimal(str(b.get("free", "0")))
            except Exception:
                return Decimal("0")
    return Decimal("0")


def _get_total_asset(acct_json: dict, asset: str) -> Decimal:
    for b in acct_json.get("balances", []):
        if b.get("asset") == asset:
            try:
                free = Decimal(str(b.get("free", "0")))
                locked = Decimal(str(b.get("locked", "0")))
                return free + locked
            except Exception:
                return Decimal("0")
    return Decimal("0")


def _find_exit_orders(open_orders: list) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Heuristic:
      - TP typically shows as LIMIT
      - SL leg shows as STOP_LOSS_LIMIT (but OCO legs may appear differently across accounts)
    This is “good enough” to avoid stacking extra exits.
    """
    tp = None
    sl = None
    for o in open_orders:
        t = (o.get("type") or "").upper()
        if tp is None and t in ("LIMIT", "LIMIT_MAKER"):
            tp = o
        if sl is None and t in ("STOP_LOSS", "STOP_LOSS_LIMIT"):
            sl = o
    return tp, sl


def live_manage_or_buy_with_oco(
    *,
    symbol: str,
    ref_entry_open: float,   # IMPORTANT: you asked TP/SL derived from ENTRY DATE OPEN
    tp_pct: float,           # e.g. 0.20
    sl_pct: float,           # e.g. 0.12
    live_trade: bool,        # False => dry-run
) -> bool:
    """
    If you already HOLD the symbol (spot) and you have NO exit orders open for it:
      -> place OCO exits based on ref_entry_open (+20% / -12% etc.)

    Else if you do NOT hold it:
      -> ask to market buy with (almost) all free USDT, then place OCO exits based on ref_entry_open.

    If exit orders already exist:
      -> do nothing (return True).

    Returns True if we handled an action path (including dry-run) else False.
    """
    symbol = symbol.upper()
    base_asset = _base_asset_from_symbol(symbol)

    api_key = os.getenv("BINANCE_API_KEY", "tE06bWu6VfzgIB1wlyZfZzaZwPe0F6RyVQrp0Fh7B8fvTzNyhxe8UZSrJV3y0Iu0").strip()
    api_secret = os.getenv("BINANCE_API_SECRET",
                           "bFBUdNU7c8HBW3pNt2CnT1m7RlASUw6ReFsWYRqPFWLyj7NIjVFLK7j2BIaFTGLf").strip()
    if not api_key or not api_secret:
        print("ERROR: Set BINANCE_API_KEY and BINANCE_API_SECRET env vars.")
        return False

    client = BinanceClient(api_key=api_key, api_secret=api_secret)

    tp_price = ref_entry_open * (1.0 + tp_pct)
    sl_price = ref_entry_open * (1.0 - sl_pct)

    ex = client.exchange_info(symbol)
    tick_size, step_size = _extract_symbol_filters(ex)

    tp_dec = _round_step(_dec(tp_price), tick_size)
    sl_stop_dec = _round_step(_dec(sl_price), tick_size)
    sl_limit_dec = _round_step(sl_stop_dec * Decimal("0.999"), tick_size)  # slightly below stop

    print("\n" + "=" * 90)
    print(f"BINANCE LIVE MANAGER: {symbol}")
    print(f"ref_entry_open={ref_entry_open:.8g} -> TP={tp_dec}  SL(stop)={sl_stop_dec}  SL(limit)={sl_limit_dec}")
    print("=" * 90)

    acct = client.account()
    free_usdt = _get_free_usdt(acct)
    total_base = _get_total_asset(acct, base_asset)

    open_orders = client.open_orders(symbol)
    tp_order, sl_order = _find_exit_orders(open_orders)

    if tp_order or sl_order:
        print("Existing exit orders detected — not placing another OCO.")
        if tp_order:
            print(f"  TP: type={tp_order.get('type')} price={tp_order.get('price')} orderId={tp_order.get('orderId')}")
        if sl_order:
            print(f"  SL: type={sl_order.get('type')} stopPrice={sl_order.get('stopPrice')} orderId={sl_order.get('orderId')}")
        return True

    holding_threshold = step_size  # “>= 1 step” means we consider it a holding

    # ---- If holding: just place exits for FREE qty ----
    if total_base >= holding_threshold:
        free_base = _get_free_asset(acct, base_asset)
        qty = _round_step(free_base, step_size)

        print(f"Holding detected: total={total_base} {base_asset}, free={free_base} {base_asset}, qty_for_exits={qty}")
        if qty <= 0:
            print("No FREE qty available (it may be locked). Check open orders/balance on Binance.")
            return False

        ans = input("Place OCO exits now? (yes/no): ").strip().lower()
        if ans not in ("y", "yes"):
            print("Cancelled.")
            return False

        if not live_trade:
            print(f"[DRY-RUN] Would place OCO SELL qty={qty} on {symbol} at TP={tp_dec}, SL={sl_stop_dec}/{sl_limit_dec}")
            return True

        resp = client.oco_sell(symbol=symbol, quantity=qty, tp_price=tp_dec, sl_stop=sl_stop_dec, sl_limit=sl_limit_dec)
        print("OCO placed:")
        print(json.dumps(resp, indent=2))
        return True

    # ---- Not holding: buy then place exits ----
    print(f"Not holding {base_asset}. Free USDT={free_usdt}")
    if free_usdt <= Decimal("0"):
        print("No free USDT to buy.")
        return False

    ans = input("Buy now (market) and place OCO exits? (yes/no): ").strip().lower()
    if ans not in ("y", "yes"):
        print("Cancelled.")
        return False

    spend_usdt = (free_usdt * Decimal("0.999")).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    if spend_usdt <= Decimal("0"):
        print("Spend calc resulted in 0.")
        return False

    if not live_trade:
        print(f"[DRY-RUN] Would MARKET BUY {symbol} using quoteOrderQty={spend_usdt} then place OCO exits.")
        return True

    buy = client.market_buy_quote_qty(symbol, spend_usdt)
    print("BUY response:")
    print(json.dumps(buy, indent=2))

    acct2 = client.account()
    free_base2 = _get_free_asset(acct2, base_asset)
    qty2 = _round_step(free_base2, step_size)

    print(f"Post-buy free {base_asset}={free_base2} -> qty_for_exits={qty2}")
    if qty2 <= 0:
        print("After buy, free qty rounds to 0 (stepSize). Place exits manually.")
        return True

    resp = client.oco_sell(symbol=symbol, quantity=qty2, tp_price=tp_dec, sl_stop=sl_stop_dec, sl_limit=sl_limit_dec)
    print("OCO placed:")
    print(json.dumps(resp, indent=2))
    return True

def run_live_prompt_after_gate(
    *,
    open_pos: Optional[dict],
    gated_picks_by_file_date: Dict[date, SignalRow],  # your #01-per-day picks (future blocks)
    entry_lag_days: int,
    tp_pct: float,
    sl_pct: float,
    cache_path: Path,
    live_trade: bool,
) -> None:
    """
    Live behavior you asked for:

    1) If we have an OPEN position from the replay (open_pos != None),
       we assume that's the position to manage NOW:
         - Do NOT buy anything new
         - Just add OCO exits for it (if you’re holding it and no exits exist)

    2) Else (no open_pos), take the latest available gated pick (max file_date),
       compute its ENTRY OPEN from Binance 1D candle on entry_day = file_date + lag,
       then:
         - If already holding it -> add OCO exits
         - Else -> optionally buy then add OCO exits

    TP/SL levels are always based on that entry_open (your requirement).
    """
    # Case 1: manage existing open position
    if open_pos is not None:
        sym = str(open_pos["symbol"]).upper()
        ref_entry_open = float(open_pos["entry_open"])
        print("\n" + "=" * 90)
        print("LIVE: managing OPEN position from replay (no new buys)")
        print("=" * 90)
        live_manage_or_buy_with_oco(
            symbol=sym,
            ref_entry_open=ref_entry_open,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            live_trade=live_trade,
        )
        return

    # Case 2: no open position -> use latest gated pick
    if not gated_picks_by_file_date:
        print("\nLIVE: no gated picks available to buy/manage.")
        return

    latest_fd = max(gated_picks_by_file_date.keys())
    pick = gated_picks_by_file_date[latest_fd]
    entry_day = latest_fd + timedelta(days=int(entry_lag_days))

    # fetch entry day's OPEN for that symbol (this is your TP/SL anchor)
    session = requests.Session()
    cache = load_cache(cache_path)
    candles = get_candles_cached(
        pick.symbol,
        entry_day,
        entry_day,
        session=session,
        cache=cache,
        cache_path=cache_path,
    )
    c = candles.get(entry_day)
    if c is None:
        print(f"\nLIVE: missing entry candle for {pick.symbol} on {entry_day.isoformat()} -> cannot compute entry_open anchor.")
        return

    print("\n" + "=" * 90)
    print(f"LIVE: latest gated pick is file={latest_fd.isoformat()} -> entry_day={entry_day.isoformat()}")
    print(f"  #01 BUY {pick.symbol}  p={pick.p:.3f} hits30d={pick.hits30d} featDropP={pick.drop_p:.3f}")
    print("=" * 90)

    live_manage_or_buy_with_oco(
        symbol=pick.symbol,
        ref_entry_open=float(c.o),
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        live_trade=live_trade,
    )


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-file", required=True)
    ap.add_argument("--cache-path", default="binance_cache.json")

    ap.add_argument("--entry-lag-days", type=int, default=1)
    ap.add_argument("--tp-pct", type=float, default=0.20)
    ap.add_argument("--sl-pct", type=float, default=0.12)
    ap.add_argument("--max-hold-days", type=int, default=21)

    ap.add_argument("--calibration-days", type=int, default=5,
                    help="How many file= blocks from predict_from to use as calibration.")
    ap.add_argument("--intraday-interval", default="1m",
                    choices=["1m", "3m", "5m", "15m", "30m", "1h"])
    ap.add_argument("--ambiguous-fallback", default="sl-first",
                    choices=["none", "tp-first", "sl-first"])
    ap.add_argument("--no-intraday-resolve", action="store_true",
                    help="Disable intraday disambiguation (faster, less accurate).")

    # ---- MISSING FLAGS (your crash) ----
    ap.add_argument("--ask-to-buy-today", action="store_true",
                    help="After gate+replay, manage open position (or latest gated pick) on Binance.")
    ap.add_argument("--live-trade", action="store_true",
                    help="Actually place orders on Binance (default is DRY-RUN).")

    args = ap.parse_args()

    signals_path = Path(args.signals_file)
    if not signals_path.exists():
        print(f"ERROR: signals file not found: {signals_path}")
        return 2

    trained_through, predict_from, signals, raw_blocks = load_signals_file(signals_path)
    if not signals:
        print("ERROR: No BUY lines found.")
        return 2

    file_dates = sorted(raw_blocks.keys())
    if not file_dates:
        print("ERROR: No file= blocks found.")
        return 2

    if trained_through is None:
        trained_through = max(file_dates)
        print(f"[warn] No 'Model trained_through' found in header. Using latest file date as trained_through: {trained_through.isoformat()}")

    # Calibration = known data (<= trained_through)
    calib_file_dates = [d for d in file_dates if d <= trained_through]
    if not calib_file_dates:
        print(f"ERROR: No calibration blocks <= trained_through ({trained_through.isoformat()}).")
        return 2
    calib_end = max(calib_file_dates)

    # Future = predicted window (> trained_through)
    future_file_dates = [d for d in file_dates if d > calib_end]

    # Group signals by file_date
    sig_by_file: Dict[date, List[SignalRow]] = {}
    for s in signals:
        sig_by_file.setdefault(s.file_date, []).append(s)

    lag = int(args.entry_lag_days)
    max_hold = int(args.max_hold_days)

    # Candle range needed for calibration simulation (entry day through hold window)
    calib_min_entry = min(d + timedelta(days=lag) for d in calib_file_dates)
    calib_max_last = max(d + timedelta(days=lag + max_hold) for d in calib_file_dates)

    today_utc = datetime.utcnow().date()
    fetch_end = min(calib_max_last, today_utc)

    session = requests.Session()
    cache_path = Path(args.cache_path)
    cache = load_cache(cache_path)

    symbols_calib = sorted({s.symbol for d in calib_file_dates for s in sig_by_file.get(d, [])})
    candles_by_symbol: Dict[str, Dict[date, Candle]] = {}
    for sym in symbols_calib:
        candles_by_symbol[sym] = get_candles_cached(
            sym,
            calib_min_entry,
            fetch_end,
            session=session,
            cache=cache,
            cache_path=cache_path,
        )

    resolve_intraday = not bool(args.no_intraday_resolve)

    print("\n" + "=" * 90)
    print("CALIBRATION (simulate TP vs SL first)")
    print("=" * 90)
    print(f"predict_from:        {(predict_from.isoformat() if predict_from else '(missing)')}")
    print(f"calibration blocks:  {calib_file_dates[0].isoformat()} .. {calib_end.isoformat()}  (N={len(calib_file_dates)})")
    print(f"entry_lag_days:      {lag}")
    print(f"tp_pct / sl_pct:     {args.tp_pct:.2f} / {args.sl_pct:.2f}")
    print(f"max_hold_days:       {max_hold}")
    print(f"intraday_interval:   {args.intraday_interval}  (resolve={resolve_intraday}, fallback={args.ambiguous_fallback})")
    print(f"candle fetch range:  {calib_min_entry.isoformat()} .. {fetch_end.isoformat()}  (today_utc={today_utc.isoformat()})")
    print("=" * 90)

    calib_rows: List[CalibRow] = []
    decided_tp = 0
    decided_sl = 0
    undecided = 0
    missing = 0

    for fd in calib_file_dates:
        rows = sig_by_file.get(fd, [])
        if not rows:
            continue

        print(f"\nfile={fd.isoformat()}  (rows={len(rows)})")

        for r in sorted(rows, key=lambda x: (-x.p, x.symbol)):
            entry_day = fd + timedelta(days=lag)
            c_entry = candles_by_symbol.get(r.symbol, {}).get(entry_day)
            if c_entry is None:
                missing += 1
                print(f"  {r.symbol:10s} p={r.p:.3f} drop={r.drop_p:.3f} hits={r.hits30d:2d}  -> SKIP (missing entry candle {entry_day.isoformat()})")
                continue

            outcome = simulate_trade_outcome(
                symbol=r.symbol,
                entry_day=entry_day,
                entry_open=c_entry.o,
                candles_by_day=candles_by_symbol.get(r.symbol, {}),
                tp_pct=float(args.tp_pct),
                sl_pct=float(args.sl_pct),
                max_hold_days=max_hold,
                session=session,
                cache=cache,
                cache_path=cache_path,
                intraday_interval=str(args.intraday_interval),
                ambiguous_fallback=str(args.ambiguous_fallback),
                resolve_ambiguous_intraday=resolve_intraday,
            )

            calib_rows.append(CalibRow(sig=r, outcome=outcome))
            if outcome == "TP":
                decided_tp += 1
            elif outcome == "SL":
                decided_sl += 1
            else:
                undecided += 1

            print(f"  {r.symbol:10s} p={r.p:.3f} drop={r.drop_p:.3f} hits={r.hits30d:2d}  -> {outcome}")

    print("\n" + "-" * 90)
    print(f"Calibration outcomes: TP={decided_tp}  SL={decided_sl}  OTHER={undecided}  missing_entry={missing}")
    print("-" * 90)

    gate, metrics = pick_best_gate(calib_rows)

    print("\n" + "=" * 90)
    print("CHOSEN GATE (fit on calibration)")
    print("=" * 90)
    print(f"BUY if: p >= {gate.p_min:.2f}  AND  featDropP <= {gate.drop_max:.2f}  AND  hits30d >= {gate.hits_min}")
    print(f"Calibration precision (TP/(TP+SL)): {metrics['precision']:.3f}   taken={int(metrics['taken'])}  TP={int(metrics['tp'])}  SL={int(metrics['sl'])}")
    print("=" * 90)

    if not future_file_dates:
        print("\n(no future blocks after calibration end — nothing to recommend)")
        return 0

    print("\n" + "=" * 90)
    print("RECOMMENDED BUYS (apply gate to remaining predicted days)")
    print("=" * 90)

    gated_picks_by_file_date: Dict[date, SignalRow] = {}

    for fd in future_file_dates:
        rows = sig_by_file.get(fd, [])
        if not rows:
            continue

        passed = [r for r in rows if passes_gate(r, gate)]
        passed.sort(key=lambda x: (-x.p, x.drop_p, -x.hits30d, x.symbol))

        print(f"\nfile={fd.isoformat()}  (passed {len(passed)}/{len(rows)})")
        if not passed:
            print("  (none)")
            continue

        for i, r in enumerate(passed, start=1):
            print(f"  #{i:02d} BUY {r.symbol:10s}  p={r.p:.3f}  hits30d={r.hits30d:2d}  featDropP={r.drop_p:.3f}")

        gated_picks_by_file_date[fd] = passed[0]  # #01 per day

    trades, compounded_factor, open_pos = replay_single_position_from_gated_picks(
        gated_picks_by_file_date=gated_picks_by_file_date,
        entry_lag_days=int(args.entry_lag_days),
        tp_pct=float(args.tp_pct),
        sl_pct=float(args.sl_pct),
        max_hold_days=int(args.max_hold_days),
        cache_path=Path(args.cache_path),
        intraday_interval=str(args.intraday_interval),
        ambiguous_fallback=str(args.ambiguous_fallback),
        resolve_ambiguous_intraday=not bool(args.no_intraday_resolve),
        end_day=datetime.utcnow().date(),
    )

    print("\n" + "=" * 90)
    print("REPLAY (SINGLE POSITION) ON GATED PICKS  (#01 per day)")
    print("=" * 90)

    if not trades:
        print("No CLOSED trades in replay window (maybe still holding / missing candles).")
    else:
        for i, t in enumerate(trades, start=1):
            print(
                f"{i:03d}. {t.symbol:10s}  signal_file={t.file_date.isoformat()}  "
                f"entry={t.entry_day.isoformat()} open={t.entry_open:.8g}  "
                f"exit={t.exit_day.isoformat()} px={t.exit_price:.8g}  "
                f"reason={t.exit_reason:9s}  gain={t.gain_pct:+.2f}%"
            )

    if open_pos is not None:
        print("\nOPEN POSITION (not closed yet):")
        print(
            f"  {open_pos['symbol']}  "
            f"signal_file={open_pos['file_date'].isoformat()}  "
            f"entry={open_pos['entry_day'].isoformat()}  "
            f"open={open_pos['entry_open']:.8g}  "
            f"last_day={open_pos['last_day'].isoformat()}  "
            f"last_close={open_pos['last_close']:.8g}  "
            f"unrealized={open_pos['unrealized_pct']:+.2f}%"
        )

    compounded_pct = (compounded_factor - 1.0) * 100.0
    print("\n" + "-" * 90)
    print(f"Compounded factor: {compounded_factor:.6f}")
    print(f"Compounded gain:   {compounded_pct:+.2f}%")
    print("-" * 90)

    # ---- LIVE prompt (optional) ----
    if args.ask_to_buy_today:
        run_live_prompt_after_gate(
            open_pos=open_pos,
            gated_picks_by_file_date=gated_picks_by_file_date,
            entry_lag_days=int(args.entry_lag_days),
            tp_pct=float(args.tp_pct),
            sl_pct=float(args.sl_pct),
            cache_path=Path(args.cache_path),
            live_trade=bool(args.live_trade),
        )

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
