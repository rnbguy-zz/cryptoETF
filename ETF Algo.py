#!/usr/bin/env python3
"""
etf_replay_from_signals_single_position.py

ALIGNMENT + REALISM + SINGLE-POSITION ROTATION

What’s different vs your current replay:
1) Ambiguous TP+SL same day:
   - Default fallback is SL-first (matches your label logic).
   - Still can resolve via intraday candles if enabled.

2) Horizon-bounded exits (aligns with labeling):
   - --max-hold-days (default 21): if still OPEN on/after entry_day+max_hold_days => exit at that day's CLOSE.

3) Optional strategy exits (same knobs you were using elsewhere):
   - --trail-pct (default 0.12): exit if CLOSE <= peak_close*(1-trail_pct)
   - --time-stop-days (default 14) and --time-stop-min-gain (default 0.05):
        if age>=days and CLOSE < entry*(1+min_gain) => exit at CLOSE

4) SINGLE POSITION at a time (your new requirement):
   - If a new entry signal arrives while holding something, we ROTATE:
       - Exit current at PREVIOUS DAY close (d-1 close) right before buying the new coin on day d open.
       - Then enter the chosen new coin on day d open.
   - If multiple signals arrive for the same entry day, we pick the one with the highest parsed p=... if present,
     otherwise first encountered.

Signals format supported:
  file=YYYY-MM-DD
  BUY SYMBOL ... (optional: p=0.781)

Usage:
  python etf_replay_from_signals_single_position.py --signals-file signals.txt

Common knobs:
  --tp-pct 0.20 --sl-pct 0.10 --max-hold-days 21 --ambiguous-fallback sl-first --intraday-interval 1m
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# ----------------------------
# Parsing signals
# ----------------------------

FILE_RE = re.compile(r"^\s*file=(\d{4}-\d{2}-\d{2})\b")
BUY_RE = re.compile(r"^\s*BUY\s+([A-Z0-9]+)\b")
P_RE = re.compile(r"\bp=([0-9]*\.?[0-9]+)\b")


@dataclass(frozen=True)
class Signal:
    file_date: date
    symbol: str
    raw_line: str
    p: float  # optional; 0.0 if missing


def parse_signals_text(text: str) -> Tuple[List[Signal], Dict[date, List[str]]]:
    cur_file_date: Optional[date] = None
    raw_block: Dict[date, List[str]] = {}
    signals: List[Signal] = []

    for line in text.splitlines():
        m = FILE_RE.match(line)
        if m:
            cur_file_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            raw_block.setdefault(cur_file_date, []).append(line.rstrip("\n"))
            continue

        if cur_file_date is not None:
            if line.strip():
                raw_block.setdefault(cur_file_date, []).append(line.rstrip("\n"))

            b = BUY_RE.match(line)
            if b:
                sym = b.group(1).upper()
                mp = P_RE.search(line)
                p = float(mp.group(1)) if mp else 0.0
                signals.append(Signal(file_date=cur_file_date, symbol=sym, raw_line=line.rstrip("\n"), p=p))

    signals.sort(key=lambda s: (s.file_date, s.symbol))
    return signals, raw_block


def load_signals_file(path: Path) -> Tuple[List[Signal], Dict[date, List[str]]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return parse_signals_text(text)


# ----------------------------
# Binance klines + caching
# ----------------------------

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
    ts_ms: int   # open time in ms UTC
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

    # dedupe by day
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


# ----------------------------
# Strategy
# ----------------------------

@dataclass
class Position:
    symbol: str
    entry_day: date
    entry_open: float
    status: str  # OPEN/CLOSED
    exit_day: Optional[date] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    peak_close: float = float("nan")     # for trailing stop
    last_close: float = float("nan")     # for reporting


def evaluate_exit_realistic(
    *,
    pos: Position,
    day_candle: Candle,
    tp_pct: float,
    sl_pct: float,
    session: requests.Session,
    cache: dict,
    cache_path: Path,
    intraday_interval: str,
    ambiguous_fallback: str,     # "none"|"tp-first"|"sl-first"
    resolve_ambiguous_intraday: bool,
) -> Optional[Tuple[str, float]]:
    """
    Returns (reason, exit_price) or None.

    - If only TP hit in 1D: exit TP
    - If only SL hit in 1D: exit SL
    - If both hit in 1D:
        - If resolve_ambiguous_intraday: use intraday candles to determine which hit first
        - If cannot resolve or resolve_ambiguous_intraday=False: use ambiguous_fallback
    """
    tp_price = pos.entry_open * (1.0 + tp_pct)
    sl_price = pos.entry_open * (1.0 - sl_pct)

    hit_tp_1d = day_candle.h >= tp_price
    hit_sl_1d = day_candle.l <= sl_price

    if hit_tp_1d and not hit_sl_1d:
        return ("TP", tp_price)
    if hit_sl_1d and not hit_tp_1d:
        return ("SL", sl_price)
    if not hit_tp_1d and not hit_sl_1d:
        return None

    # BOTH hit on 1D
    if resolve_ambiguous_intraday:
        intra = get_intraday_cached_for_day(
            pos.symbol, day_candle.day,
            interval=intraday_interval,
            session=session, cache=cache, cache_path=cache_path,
        )
    else:
        intra = []

    if not intra:
        if ambiguous_fallback == "tp-first":
            return ("TP_AMBIG", tp_price)
        if ambiguous_fallback == "sl-first":
            return ("SL_AMBIG", sl_price)
        return None

    for c in intra:
        hit_tp = c.h >= tp_price
        hit_sl = c.l <= sl_price

        if hit_tp and not hit_sl:
            return ("TP", tp_price)
        if hit_sl and not hit_tp:
            return ("SL", sl_price)
        if hit_tp and hit_sl:
            if ambiguous_fallback == "tp-first":
                return ("TP_AMBIG", tp_price)
            if ambiguous_fallback == "sl-first":
                return ("SL_AMBIG", sl_price)
            return None

    return None


# ----------------------------
# Engine (SINGLE POSITION)
# ----------------------------

def run_replay_single_position(
    *,
    signals: List[Signal],
    raw_blocks: Dict[date, List[str]],
    start_day: date,
    end_day: date,
    entry_lag_days: int,
    tp_pct: float,
    sl_pct: float,
    cache_path: Path,
    intraday_interval: str,
    ambiguous_fallback: str,
    resolve_ambiguous_intraday: bool,
    max_hold_days: int,
    trail_pct: float,
    time_stop_days: int,
    time_stop_min_gain: float,
) -> None:
    session = requests.Session()
    cache = load_cache(cache_path)

    # Build signals by ENTRY day (file_date + lag)
    sig_by_entry_day: Dict[date, List[Signal]] = {}
    for s in signals:
        e = s.file_date + timedelta(days=entry_lag_days)
        sig_by_entry_day.setdefault(e, []).append(s)

    # For any day, we might need candles for:
    # - current position symbol
    # - candidate signal symbols
    symbols_all = sorted({s.symbol for s in signals})

    candles_by_symbol: Dict[str, Dict[date, Candle]] = {}
    for sym in symbols_all:
        candles_by_symbol[sym] = get_candles_cached(
            sym, start_day, end_day,
            session=session, cache=cache, cache_path=cache_path,
        )

    def fmt_d(d: date) -> str:
        return d.strftime("%-d-%-m-%Y") if os.name != "nt" else d.strftime("%#d-%#m-%Y")

    def gain_pct(entry: float, px: float) -> float:
        return (px / max(entry, 1e-12) - 1.0) * 100.0

    compounded_factor = 1.0
    total_realised_gain = 0.0
    num_closed_trades = 0

    pos: Optional[Position] = None
    trades: List[Position] = []

    print("=" * 80)
    print("ETF REPLAY (SINGLE POSITION ROTATION, horizon-bounded, ambiguity aligned)")
    print("=" * 80)
    print(f"Replay window (UTC days): {start_day.isoformat()} .. {end_day.isoformat()}")
    print(f"Entry lag days: {entry_lag_days}  TP: {tp_pct*100:.0f}%  SL: {sl_pct*100:.0f}%")
    print(f"Max hold days: {max_hold_days} (exit at CLOSE on day entry+max_hold)")
    print(f"Trail pct: {trail_pct*100:.1f}%  Time stop: {time_stop_days}d (min gain {time_stop_min_gain*100:.1f}%)")
    print(f"Intraday interval: {intraday_interval}  Resolve ambiguous intraday: {resolve_ambiguous_intraday}")
    print(f"Ambiguous fallback: {ambiguous_fallback}")
    print(f"Signals parsed: {len(signals)}  Symbols: {len(symbols_all)}")
    print(f"Cache: {cache_path}")
    print("=" * 80)

    d = start_day
    while d <= end_day:
        day_header_printed = False

        # -------------------------
        # 1) Update/exit existing position (strategy exits)
        # -------------------------
        if pos is not None and pos.status == "OPEN":
            day_candle = candles_by_symbol.get(pos.symbol, {}).get(d)

            if not day_header_printed:
                print(f"\n----\nDate {fmt_d(d)}")
                day_header_printed = True

            if day_candle is None:
                print(f"Holding {pos.symbol}: (missing 1D candle for {d.isoformat()})")
            else:
                pos.last_close = day_candle.c
                # init peak_close
                if not (pos.peak_close == pos.peak_close):
                    pos.peak_close = day_candle.c
                pos.peak_close = max(pos.peak_close, day_candle.c)

                # exits only valid from entry_day onward
                if d >= pos.entry_day:
                    # TP/SL (with ambiguity handling)
                    exit_hit = evaluate_exit_realistic(
                        pos=pos,
                        day_candle=day_candle,
                        tp_pct=tp_pct,
                        sl_pct=sl_pct,
                        session=session,
                        cache=cache,
                        cache_path=cache_path,
                        intraday_interval=intraday_interval,
                        ambiguous_fallback=ambiguous_fallback,
                        resolve_ambiguous_intraday=resolve_ambiguous_intraday,
                    )

                    # Trailing stop (CLOSE-based)
                    if exit_hit is None and trail_pct > 0:
                        trail_px = pos.peak_close * (1.0 - trail_pct)
                        if day_candle.c <= trail_px:
                            exit_hit = ("TRAIL", day_candle.c)

                    # Time stop (CLOSE-based)
                    if exit_hit is None and time_stop_days > 0:
                        age = (d - pos.entry_day).days
                        if age >= time_stop_days:
                            if day_candle.c < pos.entry_open * (1.0 + time_stop_min_gain):
                                exit_hit = ("TIME", day_candle.c)

                    # Max hold (CLOSE-based) — aligns with horizon-bounded label world
                    if exit_hit is None and max_hold_days > 0:
                        if d >= (pos.entry_day + timedelta(days=max_hold_days)):
                            exit_hit = ("MAX_HOLD", day_candle.c)

                    if exit_hit is not None:
                        reason, exit_price = exit_hit
                        pos.status = "CLOSED"
                        pos.exit_day = d
                        pos.exit_price = float(exit_price)
                        pos.exit_reason = reason

                        g = gain_pct(pos.entry_open, pos.exit_price)
                        total_realised_gain += g
                        num_closed_trades += 1
                        compounded_factor *= (1.0 + g / 100.0)

                        print("\n" + "=" * 80)
                        print(
                            f"Sold {pos.symbol}: close={day_candle.c:.8g}, high={day_candle.h:.8g}, low={day_candle.l:.8g}, "
                            f"exit={pos.exit_price:.8g} ({reason}), gain={g:.2f}%."
                        )
                        pos = None  # flat

        # -------------------------
        # 2) Entries + rotation (SINGLE POSITION)
        # -------------------------
        entry_signals = sig_by_entry_day.get(d, [])
        if entry_signals:
            if not day_header_printed:
                print(f"\n----\nDate {fmt_d(d)}")
                day_header_printed = True

            # print raw blocks for the *source file_date(s)* of signals
            src_file_dates = sorted({s.file_date for s in entry_signals})
            for fd in src_file_dates:
                blk = raw_blocks.get(fd, [])
                if blk:
                    print("")
                    for line in blk:
                        print(line)

            # pick best signal for this day (highest p if present, else first)
            entry_signals_sorted = sorted(entry_signals, key=lambda s: (s.p, s.symbol), reverse=True)
            chosen = entry_signals_sorted[0]
            chosen_sym = chosen.symbol

            # ROTATE if currently holding something else:
            # Exit current at previous day's CLOSE (d-1 close), then buy new at today's OPEN.
            if pos is not None and pos.status == "OPEN" and pos.symbol != chosen_sym:
                prev_day = d - timedelta(days=1)
                prev_candle = candles_by_symbol.get(pos.symbol, {}).get(prev_day)

                if prev_candle is None:
                    # If we cannot price the rotation exit, fallback to today's open if available.
                    today_candle_old = candles_by_symbol.get(pos.symbol, {}).get(d)
                    if today_candle_old is None:
                        print(f"\n(rotate failed) Cannot price exit for {pos.symbol} on {prev_day.isoformat()} or {d.isoformat()}. Keeping position.")
                    else:
                        exit_px = today_candle_old.o
                        g = gain_pct(pos.entry_open, exit_px)
                        total_realised_gain += g
                        num_closed_trades += 1
                        compounded_factor *= (1.0 + g / 100.0)
                        pos.status = "CLOSED"
                        pos.exit_day = d
                        pos.exit_price = float(exit_px)
                        pos.exit_reason = "ROTATE_OPEN_FALLBACK"
                        print(
                            f"\n[ROTATE] Sold {pos.symbol} at {d.isoformat()} OPEN={exit_px:.8g} "
                            f"(missing prev close). gain={g:.2f}%."
                        )
                        pos = None
                else:
                    exit_px = prev_candle.c
                    g = gain_pct(pos.entry_open, exit_px)
                    total_realised_gain += g
                    num_closed_trades += 1
                    compounded_factor *= (1.0 + g / 100.0)
                    pos.status = "CLOSED"
                    pos.exit_day = prev_day
                    pos.exit_price = float(exit_px)
                    pos.exit_reason = "ROTATE_PREV_CLOSE"
                    print(
                        f"\n[ROTATE] Sold {pos.symbol} at {prev_day.isoformat()} CLOSE={exit_px:.8g} "
                        f"to rotate into {chosen_sym}. gain={g:.2f}%."
                    )
                    pos = None

            # If flat, enter chosen on today's OPEN
            if pos is None:
                c_new = candles_by_symbol.get(chosen_sym, {}).get(d)
                if c_new is None:
                    print(f"\n(buy failed) {chosen_sym} entry_day={d.isoformat()} missing 1D candle data.")
                else:
                    pos = Position(
                        symbol=chosen_sym,
                        entry_day=d,
                        entry_open=c_new.o,
                        status="OPEN",
                        peak_close=c_new.c,
                        last_close=c_new.c,
                    )
                    trades.append(pos)

                    tp_price = pos.entry_open * (1.0 + tp_pct)
                    sl_price = pos.entry_open * (1.0 - sl_pct)

                    p_txt = f" p={chosen.p:.3f}" if chosen.p > 0 else ""
                    print(
                        f"\nBought {chosen_sym} on {fmt_d(d)} from file={chosen.file_date.isoformat()}{p_txt} "
                        f"at buy price (open={pos.entry_open:.8g}). "
                        f"Waiting for TP (>= {tp_price:.8g}) or SL (<= {sl_price:.8g})."
                    )
            else:
                # pos is open and same symbol as chosen OR we failed rotation
                if pos.symbol == chosen_sym:
                    print(f"\n(note) Signal for {chosen_sym} arrived but already holding it. No action.")
                else:
                    print(f"\n(note) Signal for {chosen_sym} arrived but still holding {pos.symbol}. No entry.")

        d += timedelta(days=1)

    # -------------------------
    # Final summaries
    # -------------------------
    print("\n" + "=" * 80)
    print("FINAL POSITIONS / TRADES")
    print("=" * 80)

    for i, p in enumerate(trades, start=1):
        if p.status == "OPEN":
            last_px = p.last_close if (p.last_close == p.last_close) else p.entry_open
            g = gain_pct(p.entry_open, last_px)
            print(
                f"{i:03d}. {p.symbol}: OPEN  entry_day={p.entry_day.isoformat()} entry_open={p.entry_open:.8g} "
                f"last_close={last_px:.8g} gain={g:.2f}%"
            )
        else:
            exit_px = float(p.exit_price or p.entry_open)
            g = gain_pct(p.entry_open, exit_px)
            print(
                f"{i:03d}. {p.symbol}: CLOSED entry_day={p.entry_day.isoformat()} exit_day={(p.exit_day.isoformat() if p.exit_day else '')} "
                f"exit={exit_px:.8g} reason={p.exit_reason} gain={g:.2f}%"
            )

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY (CLOSED TRADES)")
    print("=" * 80)
    if num_closed_trades > 0:
        avg_gain = total_realised_gain / num_closed_trades
        print(f"Closed trades:        {num_closed_trades}")
        print(f"Total realised gain:  {total_realised_gain:.2f}%")
        print(f"AVG gain per trade:   {avg_gain:.2f}%")
    else:
        print("No closed trades yet — cannot compute average.")

    compounded_pct = (compounded_factor - 1.0) * 100.0
    print(f"Compounded gain:      {compounded_pct:.2f}%")
    print("=" * 80)


# ----------------------------
# CLI
# ----------------------------

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-file", required=True)
    ap.add_argument("--entry-lag-days", type=int, default=1)

    ap.add_argument("--tp-pct", type=float, default=0.20)
    ap.add_argument("--sl-pct", type=float, default=0.10)

    ap.add_argument("--start", default="", help="Optional override YYYY-MM-DD")
    ap.add_argument("--end", default="", help="Optional override YYYY-MM-DD")
    ap.add_argument("--cache-path", default="binance_cache.json")

    ap.add_argument("--intraday-interval", default="1m", choices=["1m", "3m", "5m", "15m", "30m", "1h"])
    ap.add_argument(
        "--ambiguous-fallback",
        default="sl-first",  # align to your label (conservative)
        choices=["none", "tp-first", "sl-first"],
        help="If intraday cannot resolve (or disabled), what to do on TP+SL same day.",
    )
    ap.add_argument(
        "--no-intraday-resolve",
        action="store_true",
        help="Disable intraday resolution entirely (more like your daily-label world).",
    )

    # Alignment knobs
    ap.add_argument("--max-hold-days", type=int, default=21, help="Exit at CLOSE on/after entry_day + max_hold_days.")
    ap.add_argument("--trail-pct", type=float, default=0.12, help="Trailing stop pct off PEAK CLOSE (0 disables).")
    ap.add_argument("--time-stop-days", type=int, default=14, help="Time stop days (0 disables).")
    ap.add_argument("--time-stop-min-gain", type=float, default=0.05, help="Min gain required at time stop.")

    args = ap.parse_args()

    signals_path = Path(args.signals_file)
    if not signals_path.exists():
        print(f"ERROR: signals file not found: {signals_path}", file=sys.stderr)
        return 2

    signals, raw_blocks = load_signals_file(signals_path)
    if not signals:
        print("ERROR: No signals found. Need 'file=YYYY-MM-DD' and 'BUY SYMBOL' lines.", file=sys.stderr)
        return 2

    file_dates = sorted({s.file_date for s in signals})
    start_day = _parse_date(args.start) if args.start else min(file_dates) + timedelta(days=int(args.entry_lag_days))
    end_day = _parse_date(args.end) if args.end else datetime.utcnow().date()

    if end_day < start_day:
        print(f"ERROR: end < start ({end_day} < {start_day})", file=sys.stderr)
        return 2

    run_replay_single_position(
        signals=signals,
        raw_blocks=raw_blocks,
        start_day=start_day,
        end_day=end_day,
        entry_lag_days=int(args.entry_lag_days),
        tp_pct=float(args.tp_pct),
        sl_pct=float(args.sl_pct),
        cache_path=Path(args.cache_path),
        intraday_interval=str(args.intraday_interval),
        ambiguous_fallback=str(args.ambiguous_fallback),
        resolve_ambiguous_intraday=not bool(args.no_intraday_resolve),
        max_hold_days=int(args.max_hold_days),
        trail_pct=float(args.trail_pct),
        time_stop_days=int(args.time_stop_days),
        time_stop_min_gain=float(args.time_stop_min_gain),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
