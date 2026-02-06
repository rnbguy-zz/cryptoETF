#!/usr/bin/env python3
"""
etf_replay_from_signals.py

REALISTIC exit ordering:
- Uses Binance 1D candles for most checks.
- If BOTH TP and SL are hit in the same 1D day, it fetches intraday candles for that UTC day
  and determines which was hit first (no assumption).

Defaults:
- start_day = earliest file=YYYY-MM-DD found in the signals file
- end_day   = today UTC (datetime.utcnow().date())

Usage:
  python etf_replay_from_signals.py --signals-file signals.txt

Optional realism knobs:
  --intraday-interval 1m|5m|15m|1h   (default 5m)
  --ambiguous-fallback none|tp-first|sl-first  (default none)
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


@dataclass(frozen=True)
class Signal:
    file_date: date
    symbol: str
    raw_line: str


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
                signals.append(Signal(file_date=cur_file_date, symbol=sym, raw_line=line.rstrip("\n")))

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
    """
    Fetch intraday klines for one UTC day [day 00:00:00 .. next day 00:00:00).
    Binance limit is 1000 per call; for 1m you need multiple pages (~1440 mins).
    """
    start_dt = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)

    start_ms = _utc_ms(start_dt)
    end_ms = _utc_ms(end_dt)

    out: List[IntraCandle] = []
    cur_start = start_ms

    # paginate until we reach end_ms
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

        # safety: if we didn't advance, stop
        if last_open_ms is None:
            break

        # next page: start after last candle open time
        cur_start = last_open_ms + 1

        # light rate-limit friendliness
        time.sleep(0.05)

    # dedupe/sort by ts
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
    """
    Cache structure:
      cache[symbol]["intraday"][interval][YYYY-MM-DD] = [ [ts,o,h,l,c], ... ]
    """
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
# Strategy hook
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
    max_high_so_far: float = float("nan")
    last_close: float = float("nan")


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
    ambiguous_fallback: str,  # "none"|"tp-first"|"sl-first"
) -> Optional[Tuple[str, float]]:
    """
    Returns (reason, exit_price) or None.
    Realistic rule:
      - If only TP is hit in 1D => TP.
      - If only SL is hit in 1D => SL.
      - If both hit in 1D => fetch intraday candles for that day and find which triggers first.
        If intraday missing/empty:
           - none => no exit (hold)
           - tp-first/sl-first => fallback ordering
    """
    tp_price = pos.entry_open * (1.0 + tp_pct)
    sl_price = pos.entry_open * (1.0 - sl_pct)

    hit_tp_1d = day_candle.h >= tp_price
    hit_sl_1d = day_candle.l <= sl_price

    if hit_tp_1d and not hit_sl_1d:
        return ("TP_20", tp_price)
    if hit_sl_1d and not hit_tp_1d:
        return ("SL_10", sl_price)
    if not hit_tp_1d and not hit_sl_1d:
        return None

    # BOTH hit on 1D: resolve via intraday
    intra = get_intraday_cached_for_day(
        pos.symbol, day_candle.day,
        interval=intraday_interval,
        session=session, cache=cache, cache_path=cache_path,
    )

    if not intra:
        # cannot resolve without assuming
        if ambiguous_fallback == "tp-first":
            return ("TP_20", tp_price)
        if ambiguous_fallback == "sl-first":
            return ("SL_10", sl_price)
        # none: don't assume
        return None

    for c in intra:
        # On each intraday candle, check whether thresholds were crossed.
        # Still possible BOTH cross inside same intraday candle; for 1m this is rare but still possible.
        # If it happens, we again cannot know within that minute unless you go finer (not available reliably).
        hit_tp = c.h >= tp_price
        hit_sl = c.l <= sl_price

        if hit_tp and not hit_sl:
            return ("TP_20", tp_price)
        if hit_sl and not hit_tp:
            return ("SL_10", sl_price)
        if hit_tp and hit_sl:
            # still ambiguous even at this resolution
            if ambiguous_fallback == "tp-first":
                return ("TP_20", tp_price)
            if ambiguous_fallback == "sl-first":
                return ("SL_10", sl_price)
            return None

    return None


# ----------------------------
# Engine
# ----------------------------

def run_replay(
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
) -> None:
    total_realised_gain = 0.0
    num_closed_trades = 0

    sig_by_entry_day: Dict[date, List[Signal]] = {}
    for s in signals:
        e = s.file_date + timedelta(days=entry_lag_days)
        sig_by_entry_day.setdefault(e, []).append(s)

    symbols = sorted({s.symbol for s in signals})

    session = requests.Session()
    cache = load_cache(cache_path)

    candles_by_symbol: Dict[str, Dict[date, Candle]] = {}
    for sym in symbols:
        candles_by_symbol[sym] = get_candles_cached(
            sym, start_day, end_day,
            session=session, cache=cache, cache_path=cache_path,
        )

    positions: Dict[str, Position] = {}
    trades: List[Position] = []

    def fmt_d(d: date) -> str:
        return d.strftime("%-d-%-m-%Y") if os.name != "nt" else d.strftime("%#d-%#m-%Y")

    print("=" * 80)
    print("ETF REPLAY FROM SIGNALS (Binance 1D + intraday disambiguation, UTC days)")
    print("=" * 80)
    print(f"Replay window (UTC days): {start_day.isoformat()} .. {end_day.isoformat()}")
    print(f"Entry lag days: {entry_lag_days}  TP: {tp_pct*100:.0f}%  SL: {sl_pct*100:.0f}%")
    print(f"Intraday interval (only when TP+SL hit same day): {intraday_interval}")
    print(f"Ambiguous fallback: {ambiguous_fallback}")
    print(f"Signals parsed: {len(signals)}  Symbols: {len(symbols)}")
    print(f"Cache: {cache_path}")
    print("=" * 80)

    d = start_day
    while d <= end_day:
        day_header_printed = False

        # Entries
        entry_signals = sig_by_entry_day.get(d, [])
        if entry_signals:
            if not day_header_printed:
                print(f"\n----\nDate {fmt_d(d)}")
                day_header_printed = True

            src_file_dates = sorted({s.file_date for s in entry_signals})
            for fd in src_file_dates:
                blk = raw_blocks.get(fd, [])
                if blk:
                    print("")
                    for line in blk:
                        print(line)

            for s in entry_signals:
                sym = s.symbol
                if sym in positions and positions[sym].status == "OPEN":
                    print(
                        f"\n(skipped) {sym} signal from file={s.file_date.isoformat()} "
                        f"because position already OPEN since {positions[sym].entry_day.isoformat()}"
                    )
                    continue

                c = candles_by_symbol.get(sym, {}).get(d)
                if c is None:
                    print(f"\n(buy failed) {sym} entry_day={d.isoformat()} missing 1D candle data.")
                    continue

                pos = Position(
                    symbol=sym,
                    entry_day=d,
                    entry_open=c.o,
                    status="OPEN",
                    max_high_so_far=c.h,
                    last_close=c.c,
                )
                positions[sym] = pos
                trades.append(pos)

                tp_price = pos.entry_open * (1.0 + tp_pct)
                sl_price = pos.entry_open * (1.0 - sl_pct)

                print(
                    f"\nBought {sym} on {fmt_d(d)} from file={s.file_date.isoformat()} "
                    f"at buy price (open={pos.entry_open:.8g}). "
                    f"Waiting for sell target (>= {tp_price:.8g}) or stop loss (<= {sl_price:.8g})."
                )

        # Exits/Holds
        open_syms = [sym for sym, p in positions.items() if p.status == "OPEN"]
        if open_syms:
            if not day_header_printed:
                print(f"\n----\nDate {fmt_d(d)}")
                day_header_printed = True

            for sym in sorted(open_syms):
                pos = positions[sym]
                day_candle = candles_by_symbol.get(sym, {}).get(d)
                if day_candle is None:
                    print(f"Holding {sym}: (missing 1D candle for {d.isoformat()})")
                    continue

                # tracking
                if pos.max_high_so_far == pos.max_high_so_far:
                    pos.max_high_so_far = max(pos.max_high_so_far, day_candle.h)
                else:
                    pos.max_high_so_far = day_candle.h
                pos.last_close = day_candle.c

                if d < pos.entry_day:
                    continue

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
                )

                if exit_hit is not None:
                    reason, exit_price = exit_hit
                    pos.status = "CLOSED"
                    pos.exit_day = d
                    pos.exit_price = float(exit_price)
                    pos.exit_reason = reason

                    gain = (pos.exit_price / max(pos.entry_open, 1e-12) - 1.0) * 100.0
                    total_realised_gain += gain
                    num_closed_trades += 1

                    print(
                        f"Sold {sym}: close={day_candle.c:.8g}, high={day_candle.h:.8g}, low={day_candle.l:.8g}, "
                        f"exit={pos.exit_price:.8g} ({reason}), gain={gain:.2f}%. (Position CLOSED)"
                    )
                else:
                    tp_price = pos.entry_open * (1.0 + tp_pct)
                    sl_price = pos.entry_open * (1.0 - sl_pct)
                    both_hit = (day_candle.h >= tp_price) and (day_candle.l <= sl_price)

                    gain_so_far = (pos.last_close / max(pos.entry_open, 1e-12) - 1.0) * 100.0
                    max_gain = (pos.max_high_so_far / max(pos.entry_open, 1e-12) - 1.0) * 100.0
                    extra = " (AMBIGUOUS: TP+SL both hit; no assumption made)" if both_hit and ambiguous_fallback == "none" else ""
                    print(
                        f"Holding {sym}: close={day_candle.c:.8g}, high={day_candle.h:.8g}, "
                        f"maxHighSoFar={pos.max_high_so_far:.8g}, "
                        f"gainSoFar={gain_so_far:.2f}%, maxGainSoFar={max_gain:.2f}%. (Position HOLDING){extra}"
                    )

        d += timedelta(days=1)

    # Final summaries
    print("\n" + "=" * 80)
    print("FINAL POSITIONS (FULL HISTORY FROM THIS RUN)")
    print("=" * 80)

    def _gain_pct(entry: float, px: float) -> float:
        return (px / max(entry, 1e-12) - 1.0) * 100.0

    for i, p in enumerate(trades, start=1):
        if p.status == "OPEN":
            last_px = p.last_close if p.last_close == p.last_close else p.entry_open
            gain = _gain_pct(p.entry_open, last_px)
            print(
                f"{i:03d}. {p.symbol}: OPEN  "
                f"entry_day={p.entry_day.isoformat()} entry_open={p.entry_open:.8g} "
                f"last_close={last_px:.8g} gain={gain:.2f}% "
                f"max_high_so_far={p.max_high_so_far:.8g}"
            )
        else:
            exit_px = float(p.exit_price or p.entry_open)
            gain = _gain_pct(p.entry_open, exit_px)
            print(
                f"{i:03d}. {p.symbol}: CLOSED "
                f"entry_day={p.entry_day.isoformat()} exit_day={(p.exit_day.isoformat() if p.exit_day else '')} "
                f"exit={exit_px:.8g} reason={p.exit_reason} gain={gain:.2f}% "
                f"max_high_so_far={p.max_high_so_far:.8g}"
            )

    open_now = [p for p in positions.values() if p.status == "OPEN"]
    print("\n" + "=" * 80)
    print("OPEN POSITIONS NOW")
    print("=" * 80)
    if not open_now:
        print("(none)")
    else:
        for p in sorted(open_now, key=lambda x: (x.symbol, x.entry_day)):
            last_px = p.last_close if p.last_close == p.last_close else p.entry_open
            gain = _gain_pct(p.entry_open, last_px)
            print(
                f"{p.symbol}: OPEN entry_day={p.entry_day.isoformat()} entry_open={p.entry_open:.8g} "
                f"last_close={last_px:.8g} gain={gain:.2f}% max_high_so_far={p.max_high_so_far:.8g}"
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

    ap.add_argument("--intraday-interval", default="5m", choices=["1m", "3m", "5m", "15m", "30m", "1h"])
    ap.add_argument(
        "--ambiguous-fallback",
        default="none",
        choices=["none", "tp-first", "sl-first"],
        help="If intraday cannot resolve (missing data or both hit in same intraday candle), what to do.",
    )

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
    start_day = _parse_date(args.start) if args.start else min(file_dates)
    end_day = _parse_date(args.end) if args.end else datetime.utcnow().date()

    if end_day < start_day:
        print(f"ERROR: end < start ({end_day} < {start_day})", file=sys.stderr)
        return 2

    run_replay(
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
