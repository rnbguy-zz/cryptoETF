#!/usr/bin/env python3.9
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests


# =============================================================================
# CoinGecko Market Cap Gate (FINAL GATE)
#   - Input: Binance-style symbol like KITEUSDT
#   - Resolve to CoinGecko coin_id via /search (prefers market_cap_rank)
#   - Fetch market cap via /simple/price?include_market_cap=true
#   - Pass only if market cap >= MARKETCAP_MIN_USD
# =============================================================================

MARKETCAP_MIN_USD = 100_000_000.0  # 100M

# =============================================================================
# Market cap cache policy
#   - Persist ONLY if mc >= 100M OR mc <= 50M
#   - Always check persisted cache BEFORE any online lookup
# =============================================================================

MARKETCAP_PERSIST_HIGH_USD = 100_000_000.0
MARKETCAP_PERSIST_LOW_USD  = 50_000_000.0
MARKETCAP_CACHE_EXTREMES_KEY = "_marketcap_extremes"   # stored inside binance_cache.json

def _is_extreme_marketcap(mc: float) -> bool:
    mc = float(mc)
    return (mc >= float(MARKETCAP_PERSIST_HIGH_USD)) or (mc <= float(MARKETCAP_PERSIST_LOW_USD))

def _load_extreme_mc_map_once(mc_cache: dict, cache_path: Path) -> Dict[str, dict]:
    """
    Loads persisted extreme market caps into memory ONCE per run.
    Stored under cache[MARKETCAP_CACHE_EXTREMES_KEY][SYMBOL] = {mc,id,provider,ts_utc}
    """
    key = ("persisted_mc_extremes", "map")
    if key in mc_cache and isinstance(mc_cache[key], dict):
        return mc_cache[key]

    m: Dict[str, dict] = {}
    try:
        blob = load_cache(cache_path)
        raw = blob.get(MARKETCAP_CACHE_EXTREMES_KEY, {})
        if isinstance(raw, dict):
            for sym, rec in raw.items():
                if not isinstance(rec, dict):
                    continue
                mc = rec.get("mc")
                pid = rec.get("id", "")
                prov = rec.get("provider", "")
                if mc is None:
                    continue
                # persisted map should already contain only extremes,
                # but we still sanity check:
                try:
                    mc_f = float(mc)
                except Exception:
                    continue
                if _is_extreme_marketcap(mc_f):
                    m[str(sym).upper()] = {
                        "mc": mc_f,
                        "id": str(pid),
                        "provider": str(prov),
                        "ts_utc": str(rec.get("ts_utc", "")),
                    }
    except Exception:
        pass

    mc_cache[key] = m
    return m

def _persist_extreme_marketcap(
    *,
    cache_path: Path,
    symbol_usdt: str,
    mc: float,
    provider: str,
    pid: str,
) -> None:
    """
    Writes to disk ONLY for extreme market caps.
    """
    mc = float(mc)
    if not _is_extreme_marketcap(mc):
        return

    try:
        blob = load_cache(cache_path)
        blob.setdefault(MARKETCAP_CACHE_EXTREMES_KEY, {})
        blob[MARKETCAP_CACHE_EXTREMES_KEY][symbol_usdt.upper()] = {
            "mc": mc,
            "provider": str(provider),
            "id": str(pid),
            "ts_utc": datetime.utcnow().isoformat(),
        }
        save_cache(cache_path, blob)
    except Exception:
        # never fail the strategy due to caching
        pass

def _base_from_usdt_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s.endswith("USDT"):
        raise ValueError(f"Expected *USDT symbol (e.g. KITEUSDT), got: {symbol}")
    base = s[:-4]
    if not base:
        raise ValueError(f"Bad symbol: {symbol}")
    return base

def resolve_coingecko_id_from_usdt_symbol(symbol_usdt: str, session: requests.Session, *, debug: bool = False) -> str:
    """
    Resolve CoinGecko coin id for a Binance symbol like KITEUSDT.

    Uses /search and picks best candidate by:
      1) ranked coins first (market_cap_rank not None)
      2) lowest market_cap_rank number
    """
    base = _base_from_usdt_symbol(symbol_usdt)
    query = base.lower()

    r = session.get(
        "https://api.coingecko.com/api/v3/search",
        params={"query": query},
        timeout=15
    )
    r.raise_for_status()
    coins = r.json().get("coins", [])
    if not coins:
        raise ValueError(f"No CoinGecko search results for: {base}")

    def rank_key(c):
        rank = c.get("market_cap_rank")
        return (rank is None, rank if rank is not None else 10**9)

    coins.sort(key=rank_key)
    best = coins[0]

    if debug:
        print(f"[mc] {symbol_usdt} base='{base}' candidates(top5):")
        for c in coins[:5]:
            if coingecko_trades_on_binance(c["id"], session):
                print(f"     id={c.get('id')!r} sym={c.get('symbol')!r} name={c.get('name')!r} rank={c.get('market_cap_rank')}")

    return str(best["id"])

def fetch_market_cap_usd_by_coingecko_id(coin_id: str, session: requests.Session) -> float:
    r = session.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
        },
        timeout=15
    )
    r.raise_for_status()
    data = r.json()
    if coin_id not in data:
        raise RuntimeError(f"CoinGecko missing id in response: {coin_id}")
    mc = data[coin_id].get("usd_market_cap")
    if mc is None:
        raise RuntimeError(f"CoinGecko missing usd_market_cap for id: {coin_id}")
    return float(mc)

def get_market_cap_usd_for_usdt_symbol(
    symbol_usdt: str,
    *,
    session: requests.Session,
    mc_cache: dict,
    cache_path: Optional[Path] = None,
    debug: bool = False,
) -> Tuple[str, float]:
    """
    Returns (coin_id, market_cap_usd).
    Caches by symbol to avoid repeated calls.
    """
    symbol_usdt = symbol_usdt.upper()

    # in-memory cache first
    if symbol_usdt in mc_cache:
        rec = mc_cache[symbol_usdt]
        return str(rec["coin_id"]), float(rec["market_cap_usd"])

    coin_id = resolve_coingecko_id_from_usdt_symbol(symbol_usdt, session=session, debug=debug)
    mc = fetch_market_cap_usd_by_coingecko_id(coin_id, session=session)

    mc_cache[symbol_usdt] = {"coin_id": coin_id, "market_cap_usd": float(mc)}

    # optionally persist into your existing json cache file
    if cache_path is not None:
        try:
            cache = load_cache(cache_path)
            cache.setdefault("_coingecko_marketcap", {})
            cache["_coingecko_marketcap"][symbol_usdt] = {"coin_id": coin_id, "market_cap_usd": float(mc), "ts_utc": datetime.utcnow().isoformat()}
            save_cache(cache_path, cache)
        except Exception:
            pass

    return coin_id, float(mc)

def _human(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:,.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


# =============================================================================
# Parsing (signals text)
# =============================================================================

FILE_RE = re.compile(r"^\s*file\s*=?\s*((?:\d{4}|\d{2})-\d{2}-\d{2})\b", re.I)
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
    s = s.strip()
    if re.match(r"^\d{2}-\d{2}-\d{2}$", s):
        yy, mm, dd = s.split("-")
        s = f"20{yy}-{mm}-{dd}"
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_signals_text(text: str) -> Tuple[Optional[date], Optional[date], List[SignalRow], Dict[date, List[str]]]:
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


def fetch_live_price(symbol: str, session: requests.Session) -> float:
    j = _session_get_json(session, BINANCE_BASE + "/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
    return float(j["price"])


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

    today_utc = datetime.utcnow().date()

    need_days = [start_day + timedelta(days=i) for i in range((end_day - start_day).days + 1)]
    missing = [d for d in need_days if d.isoformat() not in have]

    if missing:
        d0, d1 = min(missing), max(missing)
        fetched = fetch_1d_klines(sym, d0, d1, session=session)

        changed = False
        for c in fetched:
            # ✅ Don't persist today's still-forming candle
            if c.day >= today_utc:
                continue
            have[c.day.isoformat()] = {"o": c.o, "h": c.h, "l": c.l, "c": c.c}
            changed = True

        if changed:
            save_cache(cache_path, cache)

    out: Dict[date, Candle] = {}
    for d in need_days:
        if d >= today_utc:
            # ✅ For today: always pull fresh from Binance instead of cache
            fresh = fetch_1d_klines(sym, d, d, session=session)
            if fresh:
                out[d] = fresh[0]
            continue

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
    p_grid = [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75]
    drop_grid = [0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
    hits_grid = [1, 2, 3, 4, 5, 8, 10, 12, 15]

    best_gate = Gate(0.60, 0.60, 3)
    best = (-1.0, -1, -1, 10**9)  # (precision, tp, taken, sl)

    decided = [r for r in rows if r.outcome in ("TP", "SL")]
    if len(decided) == 0:
        return best_gate, {"precision": 0.0, "taken": 0, "tp": 0, "sl": 0}

    for pmin in p_grid:
        for dmax in drop_grid:
            for hmin in hits_grid:
                gate = Gate(pmin, dmax, hmin)
                prec, taken, tp, sl = score_gate(rows, gate)

                if (tp + sl) < 2:
                    continue
                if tp < 1:
                    continue

                cand = (prec, tp, taken, sl)
                if cand[0] > best[0]:
                    best = cand
                    best_gate = gate
                elif cand[0] == best[0]:
                    if cand[1] > best[1]:
                        best = cand
                        best_gate = gate
                    elif cand[1] == best[1]:
                        if cand[3] < best[3]:
                            best = cand
                            best_gate = gate
                        elif cand[3] == best[3]:
                            if cand[2] > best[2]:
                                best = cand
                                best_gate = gate

    prec, taken, tp, sl = score_gate(rows, best_gate)
    return best_gate, {"precision": prec, "taken": taken, "tp": tp, "sl": sl}


# =============================================================================
# Replay (single-position) + compounding
# =============================================================================

@dataclass
class ReplayTrade:
    symbol: str
    file_date: date
    entry_day: date
    entry_open: float
    exit_day: date
    exit_price: float
    exit_reason: str
    gain_pct: float


def _gain_pct(entry: float, exit_px: float) -> float:
    return (exit_px / max(entry, 1e-12) - 1.0) * 100.0


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


def replay_single_position_from_gated_picks(
    *,
    gated_picks_by_file_date: Dict[date, SignalRow],
    entry_lag_days: int,
    tp_pct: float,
    sl_pct: float,
    max_hold_days: int,
    cache_path: Path,
    intraday_interval: str,
    ambiguous_fallback: str,
    resolve_ambiguous_intraday: bool,
    end_day: Optional[date] = None,
) -> Tuple[List["ReplayTrade"], float, Optional[dict]]:
    if not gated_picks_by_file_date:
        return [], 1.0, None

    session = requests.Session()
    cache = load_cache(cache_path)

    file_days = sorted(gated_picks_by_file_date.keys())
    start_entry_day = min(fd + timedelta(days=entry_lag_days) for fd in file_days)

    if end_day is None:
        end_day = datetime.utcnow().date()

    symbols = sorted({s.symbol for s in gated_picks_by_file_date.values()})

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

                    holding_symbol = None
                    holding_entry_open = 0.0
                    holding_entry_day = None
                    holding_file_date = None

        if holding_symbol is None:
            candidates: List[SignalRow] = []
            for fd, sig in gated_picks_by_file_date.items():
                if fd + timedelta(days=entry_lag_days) == d:
                    candidates.append(sig)

            if candidates:
                candidates.sort(key=lambda s: (-s.p, s.drop_p, -s.hits30d, s.symbol))
                chosen = candidates[0]

                c_entry = candles_by_symbol.get(chosen.symbol, {}).get(d)
                if c_entry is not None:
                    holding_symbol = chosen.symbol
                    holding_entry_open = c_entry.o
                    holding_entry_day = d
                    holding_file_date = chosen.file_date

                    hit0 = evaluate_exit_reason_and_price(
                        symbol=holding_symbol,
                        entry_open=holding_entry_open,
                        day=d,
                        day_candle=c_entry,
                        tp_pct=tp_pct,
                        sl_pct=sl_pct,
                        session=session,
                        cache=cache,
                        cache_path=cache_path,
                        intraday_interval=intraday_interval,
                        ambiguous_fallback=ambiguous_fallback,
                        resolve_ambiguous_intraday=resolve_ambiguous_intraday,
                    )

                    if hit0 is not None:
                        reason0, exit_px0 = hit0
                        g0 = _gain_pct(holding_entry_open, float(exit_px0))
                        compounded *= (1.0 + g0 / 100.0)

                        trades.append(
                            ReplayTrade(
                                symbol=holding_symbol,
                                file_date=holding_file_date,
                                entry_day=holding_entry_day,
                                entry_open=holding_entry_open,
                                exit_day=d,
                                exit_price=float(exit_px0),
                                exit_reason=reason0,
                                gain_pct=g0,
                            )
                        )

                        holding_symbol = None
                        holding_entry_open = 0.0
                        holding_entry_day = None
                        holding_file_date = None

        d += timedelta(days=1)

    open_pos: Optional[dict] = None
    if holding_symbol is not None and holding_entry_day is not None and holding_file_date is not None:
        c_last = candles_by_symbol.get(holding_symbol, {}).get(end_day)
        last_close = c_last.c if c_last else holding_entry_open

        if end_day == datetime.utcnow().date():
            try:
                live_px = fetch_live_price(holding_symbol, session=session)
                last_close = live_px
                tp_price = holding_entry_open * (1.0 + tp_pct)
                sl_price = holding_entry_open * (1.0 - sl_pct)

                if live_px >= tp_price:
                    g = _gain_pct(holding_entry_open, float(tp_price))
                    compounded *= (1.0 + g / 100.0)
                    trades.append(
                        ReplayTrade(
                            symbol=holding_symbol,
                            file_date=holding_file_date,
                            entry_day=holding_entry_day,
                            entry_open=holding_entry_open,
                            exit_day=end_day,
                            exit_price=float(tp_price),
                            exit_reason="TP_NOW",
                            gain_pct=g,
                        )
                    )
                    holding_symbol = None
                    holding_entry_open = 0.0
                    holding_entry_day = None
                    holding_file_date = None

                elif live_px <= sl_price:
                    g = _gain_pct(holding_entry_open, float(sl_price))
                    compounded *= (1.0 + g / 100.0)
                    trades.append(
                        ReplayTrade(
                            symbol=holding_symbol,
                            file_date=holding_file_date,
                            entry_day=holding_entry_day,
                            entry_open=holding_entry_open,
                            exit_day=end_day,
                            exit_price=float(sl_price),
                            exit_reason="SL_NOW",
                            gain_pct=g,
                        )
                    )
                    holding_symbol = None
                    holding_entry_open = 0.0
                    holding_entry_day = None
                    holding_file_date = None
            except Exception:
                pass

        if holding_symbol is not None and holding_entry_day is not None and holding_file_date is not None:
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
# =============================================================================

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
    ref_entry_open: float,
    tp_pct: float,
    sl_pct: float,
    live_trade: bool,
) -> bool:
    symbol = symbol.upper()
    base_asset = _base_asset_from_symbol(symbol)

    # SAFER: require env vars (do NOT hardcode keys in code)
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
    sl_limit_dec = _round_step(sl_stop_dec * Decimal("0.999"), tick_size)

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

    holding_threshold = step_size

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
    gated_picks_by_file_date: Dict[date, SignalRow],
    entry_lag_days: int,
    tp_pct: float,
    sl_pct: float,
    cache_path: Path,
    live_trade: bool,
) -> None:
    if open_pos is None:
        print("\nLIVE: replay is FLAT (no open position) -> not buying anything.")
        return

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

# =============================================================================
# Market Cap Gate (CoinGecko primary, CoinPaprika fallback) + caching + 429 backoff
# =============================================================================

MARKETCAP_MIN_USD = 100_000_000.0  # 100M

def _base_from_usdt_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s.endswith("USDT"):
        raise ValueError(f"Expected *USDT symbol (e.g. KITEUSDT), got: {symbol}")
    base = s[:-4]
    if not base:
        raise ValueError(f"Bad symbol: {symbol}")
    return base

def _get_json_with_429_backoff(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    timeout: int = 15,
    max_attempts: int = 6,
    base_sleep: float = 1.0,
):
    """
    Retries on 429 with exponential backoff.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                # backoff: 1s, 2s, 4s, 8s... (cap ~30s)
                sleep_s = min(30.0, base_sleep * (2 ** (attempt - 1)))
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(10.0, 0.5 * attempt))
    raise RuntimeError(f"HTTP failed after retries: {last_err}")

def coingecko_trades_on_binance(coin_id: str, session: requests.Session) -> bool:
    r = session.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/tickers",
        params={
            "exchange_ids": "binance",
            "include_exchange_logo": "false",
        },
        timeout=15,
    )
    r.raise_for_status()
    tickers = r.json().get("tickers", [])
    return any(t.get("target") == "USDT" for t in tickers)

# ---------- CoinGecko ----------
def resolve_coingecko_id_from_usdt_symbol(symbol_usdt: str, session: requests.Session, debug=False) -> str:
    base = symbol_usdt.replace("USDT", "").lower()

    r = session.get(
        "https://api.coingecko.com/api/v3/search",
        params={"query": base},
        timeout=15,
    )
    r.raise_for_status()
    coins = r.json().get("coins", [])

    if not coins:
        raise ValueError(f"No CoinGecko results for {symbol_usdt}")

    # Prefer ranked coins first
    coins.sort(key=lambda c: (c.get("market_cap_rank") is None,
                              c.get("market_cap_rank", 10**9)))

    for c in coins:
        cid = c["id"]

        try:
            if coingecko_trades_on_binance(cid, session):
                if debug:
                    print(f"[mc] {symbol_usdt} resolved to {cid} (Binance verified)")
                return cid
        except Exception:
            continue

    raise RuntimeError(f"No Binance-listed CoinGecko match for {symbol_usdt}")

def fetch_market_cap_usd_coingecko(coin_id: str, session: requests.Session) -> float:
    data = _get_json_with_429_backoff(
        session,
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": coin_id, "vs_currencies": "usd", "include_market_cap": "true"},
        timeout=15,
        max_attempts=6,
        base_sleep=1.0,
    )
    if coin_id not in data or data[coin_id].get("usd_market_cap") is None:
        raise RuntimeError(f"CoinGecko missing usd_market_cap for id={coin_id}")
    return float(data[coin_id]["usd_market_cap"])

def get_market_cap_usd_coingecko(
    symbol_usdt: str,
    *,
    session: requests.Session,
    mc_cache: dict,
    cache_path: Optional[Path],
    debug: bool = False,
) -> Tuple[str, float]:
    symbol_usdt = symbol_usdt.upper()
    key = ("coingecko", symbol_usdt)
    if key in mc_cache:
        rec = mc_cache[key]
        return rec["id"], float(rec["mc"])

    coin_id = resolve_coingecko_id_from_usdt_symbol(symbol_usdt, session=session, debug=debug)
    mc = fetch_market_cap_usd_coingecko(coin_id, session=session)

    mc_cache[key] = {"id": coin_id, "mc": float(mc)}

    # optional persistence inside your existing json cache
    if cache_path is not None:
        try:
            cache = load_cache(cache_path)
            cache.setdefault("_marketcap", {})
            cache["_marketcap"].setdefault("coingecko", {})
            cache["_marketcap"]["coingecko"][symbol_usdt] = {"id": coin_id, "mc": float(mc), "ts_utc": datetime.utcnow().isoformat()}
            save_cache(cache_path, cache)
        except Exception:
            pass

    return coin_id, float(mc)

# ---------- CoinPaprika fallback ----------
def _coinpaprika_symbol_map(session: requests.Session, mc_cache: dict) -> Dict[str, str]:
    """
    Build symbol->coinpaprika_id map once per run (cached).
    Note: symbol collisions exist; we pick the first occurrence.
    """
    key = ("coinpaprika", "symbol_map")
    if key in mc_cache:
        return mc_cache[key]

    coins = session.get("https://api.coinpaprika.com/v1/coins", timeout=30).json()
    m: Dict[str, str] = {}
    for c in coins:
        sym = (c.get("symbol") or "").upper()
        cid = c.get("id")
        is_active = c.get("is_active", True)
        if not sym or not cid or not is_active:
            continue
        # first wins (simple); good enough for most symbols
        m.setdefault(sym, cid)

    mc_cache[key] = m
    return m

def get_market_cap_usd_coinpaprika(
    symbol_usdt: str,
    *,
    session: requests.Session,
    mc_cache: dict,
    debug: bool = False,
) -> Tuple[str, float]:
    symbol_usdt = symbol_usdt.upper()
    base = _base_from_usdt_symbol(symbol_usdt)  # e.g. KITE
    key = ("coinpaprika", symbol_usdt)
    if key in mc_cache:
        rec = mc_cache[key]
        return rec["id"], float(rec["mc"])

    sym_map = _coinpaprika_symbol_map(session, mc_cache)
    coin_id = sym_map.get(base.upper())
    if not coin_id:
        raise ValueError(f"CoinPaprika: no id found for symbol {base}")

    data = session.get(f"https://api.coinpaprika.com/v1/tickers/{coin_id}", timeout=20).json()
    mc = (((data.get("quotes") or {}).get("USD") or {}).get("market_cap"))
    if mc is None:
        raise RuntimeError(f"CoinPaprika: missing USD market_cap for {coin_id}")

    if debug:
        print(f"[mc] CP {symbol_usdt} -> {coin_id} name={data.get('name')} mc={mc}")

    mc_cache[key] = {"id": coin_id, "mc": float(mc)}
    return coin_id, float(mc)

def passes_marketcap_gate(
    symbol_usdt: str,
    *,
    session: requests.Session,
    mc_cache: dict,
    cache_path: Path,
    min_usd: float = MARKETCAP_MIN_USD,
    debug: bool = False,
) -> Tuple[bool, float, str, str]:
    """
    Returns (passes, market_cap_usd, provider_id, provider_name)

    Policy:
      1) Check persisted EXTREMES cache first (disk -> loaded once per run).
      2) Then check per-run memory cache.
      3) Only if still missing, go online (CoinGecko then CoinPaprika).
      4) Persist result ONLY if mc >= 100M OR mc <= 50M.
    """
    symbol_usdt = symbol_usdt.upper()
    min_usd = float(min_usd)

    # 1) persisted extremes (NO internet)
    persisted_map = _load_extreme_mc_map_once(mc_cache, cache_path)
    if symbol_usdt in persisted_map:
        rec = persisted_map[symbol_usdt]
        mc = float(rec["mc"])
        pid = str(rec.get("id", ""))
        prov = str(rec.get("provider", ""))
        return (mc >= min_usd), mc, pid, prov

    # 2) per-run memory (fast)
    run_key = ("final", symbol_usdt)
    if run_key in mc_cache:
        rec = mc_cache[run_key]
        mc = float(rec["mc"])
        return (mc >= min_usd), mc, str(rec.get("id", "")), str(rec.get("provider", "none"))

    # 3) CoinGecko
    try:
        cid, mc = get_market_cap_usd_coingecko(
            symbol_usdt,
            session=session,
            mc_cache=mc_cache,
            cache_path=None,   # <— IMPORTANT: don't let the old function persist everything
            debug=debug,
        )
        mc_f = float(mc)
        mc_cache[run_key] = {"mc": mc_f, "id": cid, "provider": "coingecko"}

        # 4) persist only if extreme
        _persist_extreme_marketcap(cache_path=cache_path, symbol_usdt=symbol_usdt, mc=mc_f, provider="coingecko", pid=cid)

        # also update in-memory persisted map so later calls don't hit disk
        if _is_extreme_marketcap(mc_f):
            persisted_map[symbol_usdt] = {"mc": mc_f, "id": str(cid), "provider": "coingecko", "ts_utc": datetime.utcnow().isoformat()}

        return (mc_f >= min_usd), mc_f, str(cid), "coingecko"
    except Exception as e:
        if debug:
            print(f"[mc] CoinGecko failed for {symbol_usdt}: {e}")

    # 5) CoinPaprika fallback
    try:
        pid, mc = get_market_cap_usd_coinpaprika(
            symbol_usdt,
            session=session,
            mc_cache=mc_cache,
            debug=debug,
        )
        mc_f = float(mc)
        mc_cache[run_key] = {"mc": mc_f, "id": pid, "provider": "coinpaprika"}

        _persist_extreme_marketcap(cache_path=cache_path, symbol_usdt=symbol_usdt, mc=mc_f, provider="coinpaprika", pid=pid)

        if _is_extreme_marketcap(mc_f):
            persisted_map[symbol_usdt] = {"mc": mc_f, "id": str(pid), "provider": "coinpaprika", "ts_utc": datetime.utcnow().isoformat()}

        return (mc_f >= min_usd), mc_f, str(pid), "coinpaprika"
    except Exception as e:
        if debug:
            print(f"[mc] CoinPaprika failed for {symbol_usdt}: {e}")

    mc_cache[run_key] = {"mc": 0.0, "id": "", "provider": "none"}
    return (False, 0.0, "", "none")

# =============================================================================
# Main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-file", default="./signals1.txt")
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

    ap.add_argument("--ask-to-buy-today", action="store_true", default=True,
                    help="After gate+replay, manage open position on Binance.")
    ap.add_argument("--live-trade", action="store_true", default=True,
                    help="Actually place orders on Binance (default is DRY-RUN).")

    # Market cap gate flags (optional)
    ap.add_argument("--min-marketcap-usd", type=float, default=MARKETCAP_MIN_USD,
                    help="Final gate: require CoinGecko market cap >= this USD amount (default 100M).")
    ap.add_argument("--marketcap-debug", action="store_true",
                    help="Print CoinGecko candidate matches for each symbol (debug).")

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
        print(f"[warn] No 'Model trained_through' found. Using latest file date as trained_through: {trained_through.isoformat()}")

    calib_file_dates = [d for d in file_dates if d <= trained_through]
    if not calib_file_dates:
        print(f"ERROR: No calibration blocks <= trained_through ({trained_through.isoformat()}).")
        return 2
    calib_end = max(calib_file_dates)

    future_file_dates = [d for d in file_dates if d > calib_end]

    sig_by_file: Dict[date, List[SignalRow]] = {}
    for s in signals:
        sig_by_file.setdefault(s.file_date, []).append(s)

    lag = int(args.entry_lag_days)
    max_hold = int(args.max_hold_days)

    calib_min_entry = min(d + timedelta(days=lag) for d in calib_file_dates)
    calib_max_last = max(d + timedelta(days=lag + max_hold) for d in calib_file_dates)

    today_utc = datetime.utcnow().date()
    fetch_end = min(calib_max_last, today_utc)

    session = requests.Session()
    mc_cache: Dict[tuple, dict] = {}  # in-memory, per-run cache
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

    mc_cache: Dict[tuple, dict] = {}  # keep this (tuple keys used elsewhere)
    print("\n" + "=" * 90)
    print("RECOMMENDED BUYS (apply gate to remaining predicted days)")
    print("NOTE: FINAL GATE ADDED -> market cap must be >= "
          f"{_human(float(args.min_marketcap_usd))} USD (CoinGecko)")
    print("=" * 90)

    gated_picks_by_file_date: Dict[date, SignalRow] = {}

    for fd in future_file_dates:
        rows = sig_by_file.get(fd, [])
        if not rows:
            continue

        # first apply normal gate
        prelim = [r for r in rows if passes_gate(r, gate)]
        prelim.sort(key=lambda x: (-x.p, x.drop_p, -x.hits30d, x.symbol))

        print(f"\nfile={fd.isoformat()}  (prelim passed {len(prelim)}/{len(rows)})")
        if not prelim:
            print("  (none)")
            continue

        # now apply FINAL market cap gate
        final_passed: List[SignalRow] = []
        for r in prelim:
            ok, mc, pid, provider = passes_marketcap_gate(
                r.symbol,
                session=session,
                mc_cache=mc_cache,
                cache_path=cache_path,
                min_usd=float(args.min_marketcap_usd),
                debug=bool(args.marketcap_debug),
            )
            if ok:
                final_passed.append(r)
                print(
                    f"  PASS  BUY {r.symbol:10s}  p={r.p:.3f} hits30d={r.hits30d:2d} featDropP={r.drop_p:.3f}  mc={_human(mc)} USD  ({provider}:{pid})")
            else:
                if mc > 0:
                    print(
                        f"  FAIL  {r.symbol:10s}  (market cap {_human(mc)} USD < {_human(float(args.min_marketcap_usd))} USD)")
                else:
                    print(f"  FAIL  {r.symbol:10s}  (market cap lookup failed)")
        if not final_passed:
            print("  (none after market cap gate)")
            continue

        # If multiple remain, keep your ordering: highest p, lower drop, higher hits
        final_passed.sort(key=lambda x: (-x.p, x.drop_p, -x.hits30d, x.symbol))
        gated_picks_by_file_date[fd] = final_passed[0]  # #01 per day

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