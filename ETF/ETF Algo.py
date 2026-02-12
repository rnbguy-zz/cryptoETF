#!/usr/bin/env python3
"""
ETF Algo.py  (SINGLE POSITION ROTATION replay + OPTIONAL Binance Spot OCO prompt)

You chose Option B:
- If the REPLAY ends with an OPEN position, prompt to manage THAT coin on Binance:
    - If you already have exits open -> do nothing
    - Else if you already hold the coin -> place OCO exits on FREE qty (or fallback TP then monitor SL)
    - Else (not holding) -> ask to BUY it, then place OCO exits

Replay keeps ALL its existing history/summary/compound prints.

DRY RUN vs LIVE:
- default: DRY RUN (prints what it would do)
- to place real orders:
    --live-trade
    set env vars:
      BINANCE_API_KEY
      BINANCE_API_SECRET

Usage:
  python ETF Algo.py --signals-file signals.txt --tp-pct 0.20 --sl-pct 0.12 --max-hold-days 21 \
    --ambiguous-fallback sl-first --intraday-interval 1m --ask-to-buy-today --live-trade
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests


# =============================================================================
# Parsing signals
# =============================================================================

FILE_RE = re.compile(r"^\s*file=(\d{4}-\d{2}-\d{2})\b")
BUY_RE = re.compile(r"^\s*BUY\s+([A-Z0-9]+)\b")
P_RE = re.compile(r"\bp\s*=\s*([0-9]*\.?[0-9]+)\b")  # parses "p=0.563" if present


@dataclass(frozen=True)
class Signal:
    file_date: date
    symbol: str
    raw_line: str
    p: float  # 0.0 if missing


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


# =============================================================================
# Strategy
# =============================================================================

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


# =============================================================================
# Binance trading (signed REST) + rounding helpers
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

    def _delete(self, path: str, params: dict, signed: bool = True, timeout: int = 30):
        url = self.base_url + path

        if not signed:
            r = self.session.delete(url, params=params, timeout=timeout)
            if not r.ok:
                self._raise_binance(r)
            return r.json()

        p = dict(params)
        p["timestamp"] = self._now_ms()
        p.setdefault("recvWindow", 20000)

        qs = self._sign_query(p)
        full_url = url + "?" + qs
        r = self.session.delete(full_url, timeout=timeout)

        if r.status_code == 400:
            try:
                j = r.json()
                if isinstance(j, dict) and j.get("code") == -1021:
                    self._sync_time()
                    p["timestamp"] = self._now_ms()
                    qs = self._sign_query(p)
                    full_url = url + "?" + qs
                    r = self.session.delete(full_url, timeout=timeout)
            except Exception:
                pass

        if not r.ok:
            self._raise_binance(r)
        return r.json()

    # ----- endpoints -----

    def exchange_info(self, symbol: str) -> dict:
        return self._get("/api/v3/exchangeInfo", {"symbol": symbol}, signed=False)

    def account(self) -> dict:
        return self._get("/api/v3/account", {}, signed=True)

    def ticker_price(self, symbol: str) -> float:
        j = self._get("/api/v3/ticker/price", {"symbol": symbol}, signed=False)
        return float(j["price"])

    def open_orders(self, symbol: str) -> list:
        return self._get("/api/v3/openOrders", {"symbol": symbol}, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._delete("/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def market_buy_quote_qty(self, symbol: str, quote_qty: Decimal) -> dict:
        return self._post("/api/v3/order", {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": f"{quote_qty:f}",
            "newOrderRespType": "FULL",
        }, signed=True)

    def oco_sell(self, symbol: str, quantity: Decimal, tp_price: Decimal, sl_stop: Decimal, sl_limit: Decimal) -> dict:
        return self._post("/api/v3/order/oco", {
            "symbol": symbol,
            "side": "SELL",
            "quantity": f"{quantity:f}",
            "price": f"{tp_price:f}",
            "stopPrice": f"{sl_stop:f}",
            "stopLimitPrice": f"{sl_limit:f}",
            "stopLimitTimeInForce": "GTC",
        }, signed=True)

    def order_limit_sell(self, symbol: str, quantity: Decimal, price: Decimal) -> dict:
        return self._post("/api/v3/order", {
            "symbol": symbol,
            "side": "SELL",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": f"{quantity:f}",
            "price": f"{price:f}",
        }, signed=True)

    def order_stop_loss_limit_sell(self, symbol: str, quantity: Decimal, stop_price: Decimal, limit_price: Decimal) -> dict:
        return self._post("/api/v3/order", {
            "symbol": symbol,
            "side": "SELL",
            "type": "STOP_LOSS_LIMIT",
            "timeInForce": "GTC",
            "quantity": f"{quantity:f}",
            "stopPrice": f"{stop_price:f}",
            "price": f"{limit_price:f}",
        }, signed=True)


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

    f = symbols[0].get("filters") or []
    tick_size = None
    step_size = None
    for flt in f:
        if flt.get("filterType") == "PRICE_FILTER":
            tick_size = Decimal(str(flt.get("tickSize", "0")))
        if flt.get("filterType") == "LOT_SIZE":
            step_size = Decimal(str(flt.get("stepSize", "0")))
    if tick_size is None or step_size is None:
        raise RuntimeError("Could not find PRICE_FILTER / LOT_SIZE in exchangeInfo")
    return tick_size, step_size


def _get_free_usdt(acct_json: dict) -> Decimal:
    for b in acct_json.get("balances", []):
        if b.get("asset") == "USDT":
            try:
                return Decimal(str(b.get("free", "0")))
            except Exception:
                return Decimal("0")
    return Decimal("0")


def _base_asset_from_symbol(symbol: str) -> str:
    if symbol.endswith("USDT"):
        return symbol[:-4]
    raise ValueError("This script's live trading assumes USDT quote pairs only (endswith USDT).")


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


def _monitor_then_place_sl(
    *,
    client: BinanceClient,
    symbol: str,
    base_asset: str,
    sl_stop: Decimal,
    sl_limit: Decimal,
    step_size: Decimal,
    poll_sec: int = 5,
) -> None:
    print("\nMonitoring position for TP fill or SL trigger... (Ctrl+C to stop)")

    while True:
        try:
            open_orders = client.open_orders(symbol)
            tp_order, sl_order = _find_exit_orders(open_orders)

            acct = client.account()
            total_base = _get_total_asset(acct, base_asset)
            free_base = _get_free_asset(acct, base_asset)

            # Position gone (sold/filled) => done
            if _round_step(total_base, step_size) <= 0:
                print("Position likely closed (no remaining base). Done.")
                return

            # If SL already exists, keep waiting
            if sl_order is not None:
                time.sleep(poll_sec)
                continue

            # No SL order yet: check live price
            px = Decimal(str(client.ticker_price(symbol)))
            if px <= sl_stop:
                print(f"\nSL trigger hit: price={px} <= {sl_stop}")

                # Cancel TP so we can free up balance for SL order
                if tp_order is not None:
                    try:
                        oid = int(tp_order.get("orderId"))
                        print(f"Cancelling TP orderId={oid} ...")
                        client.cancel_order(symbol, oid)
                    except Exception as e:
                        print(f"WARNING: Failed to cancel TP order: {e}")

                # Re-check free qty and place SL
                acct2 = client.account()
                free_base2 = _get_free_asset(acct2, base_asset)
                qty = _round_step(free_base2, step_size)

                if qty <= 0:
                    print(f"No FREE {base_asset} available to place SL (free={free_base2}). Done.")
                    return

                sl_resp = client.order_stop_loss_limit_sell(symbol, qty, sl_stop, sl_limit)
                print("Placed SL STOP_LOSS_LIMIT sell:")
                print(json.dumps(sl_resp, indent=2))
                # after placing SL, continue monitoring
            time.sleep(poll_sec)

        except KeyboardInterrupt:
            print("\nStopped monitoring. Note: You may have open orders on Binance.")
            return
        except Exception as e:
            print(f"Monitor loop error: {e}")
            time.sleep(max(5, poll_sec))


def _prompt_manage_replay_open_position_spot(
    *,
    replay_pos: Position,
    tp_pct: float,
    sl_pct: float,
    live_trade: bool,
) -> None:
    """
    Option B prompt:
    - If replay ends OPEN on symbol S, prompt to manage S on Binance.
    - Uses replay_pos.entry_open to compute TP/SL levels (same as replay).
    - If not holding S -> ask to buy it (if you want), then place exits.
    """
    symbol = replay_pos.symbol.upper()
    base_asset = _base_asset_from_symbol(symbol)

    tp_price = replay_pos.entry_open * (1.0 + tp_pct)
    sl_price = replay_pos.entry_open * (1.0 - sl_pct)

    print("\n" + "=" * 80)
    print(f"BINANCE LIVE PROMPT (REPLAY OPEN POSITION): {symbol}")
    print(f"Replay entry_open={replay_pos.entry_open:.8g} => TP={tp_price:.8g}  SL={sl_price:.8g}")
    print("=" * 80)

    api_key = os.getenv("BINANCE_API_KEY", "tE06bWu6VfzgIB1wlyZfZzaZwPe0F6RyVQrp0Fh7B8fvTzNyhxe8UZSrJV3y0Iu0").strip()
    api_secret = os.getenv("BINANCE_API_SECRET", "bFBUdNU7c8HBW3pNt2CnT1m7RlASUw6ReFsWYRqPFWLyj7NIjVFLK7j2BIaFTGLf").strip()
    if not api_key or not api_secret:
        print("ERROR: BINANCE_API_KEY / BINANCE_API_SECRET env vars not set. Cannot trade.")
        return

    client = BinanceClient(api_key=api_key, api_secret=api_secret)

    # Rounding filters
    ex = client.exchange_info(symbol)
    tick_size, step_size = _extract_symbol_filters(ex)

    tp_dec = _round_step(_dec(tp_price), tick_size)
    sl_stop_dec = _round_step(_dec(sl_price), tick_size)
    sl_limit_dec = _round_step(sl_stop_dec * Decimal("0.999"), tick_size)  # slightly below stop

    if tp_dec <= 0 or sl_stop_dec <= 0 or sl_limit_dec <= 0:
        print("ERROR: Computed TP/SL invalid after rounding; aborting.")
        return

    acct = client.account()
    free_usdt = _get_free_usdt(acct)
    total_base = _get_total_asset(acct, base_asset)
    holding_threshold = step_size

    open_orders = client.open_orders(symbol)
    tp_order, sl_order = _find_exit_orders(open_orders)

    # If exits already exist, do nothing (prevents stacking multiple OCOs)
    if tp_order or sl_order:
        print("\nExisting exit orders detected — not placing another one.")
        if tp_order:
            print(f"  TP: orderId={tp_order.get('orderId')} type={tp_order.get('type')} price={tp_order.get('price')}")
        if sl_order:
            print(f"  SL: orderId={sl_order.get('orderId')} type={sl_order.get('type')} stopPrice={sl_order.get('stopPrice')} price={sl_order.get('price')}")
        return

    # If already holding, offer to place exits (no buy)
    if total_base >= holding_threshold:
        free_base = _get_free_asset(acct, base_asset)
        qty = _round_step(free_base, step_size)

        print(f"\nHolding {symbol} detected. total={total_base} {base_asset}, free={free_base} {base_asset}, step={step_size}")
        if qty <= 0:
            print(
                f"No FREE {base_asset} available to place exits (free={free_base}, total={total_base}).\n"
                "Your balance may be locked elsewhere. Check Binance open orders."
            )
            return

        print(f"OCO levels (rounded): TP={tp_dec}  SL(stop)={sl_stop_dec}  SL(limit)={sl_limit_dec}")
        ans = input(f"Do you want to PLACE OCO exits for {symbol} now? (yes/no): ").strip().lower()
        if ans not in ("y", "yes"):
            print("Cancelled.")
            return

        if not live_trade:
            print(f"\n[DRY-RUN] Would place OCO exits for qty={qty} {base_asset} on {symbol}")
            return

        # Try OCO
        try:
            oco_resp = client.oco_sell(symbol=symbol, quantity=qty, tp_price=tp_dec, sl_stop=sl_stop_dec, sl_limit=sl_limit_dec)
            print("\nOCO placed successfully:")
            print(json.dumps(oco_resp, indent=2))
            return
        except Exception as e:
            print(f"\nOCO failed; falling back to TP-first then conditional SL. Reason: {e}")

        # Fallback: TP now, SL later if needed
        tp_resp = client.order_limit_sell(symbol, qty, tp_dec)
        print("Placed TP LIMIT sell:")
        print(json.dumps(tp_resp, indent=2))
        _monitor_then_place_sl(
            client=client,
            symbol=symbol,
            base_asset=base_asset,
            sl_stop=sl_stop_dec,
            sl_limit=sl_limit_dec,
            step_size=step_size,
            poll_sec=5,
        )
        return

    # Not holding: ask to buy then set exits
    print(f"\nNot holding {base_asset}. USDT free={free_usdt}")

    if free_usdt <= Decimal("0"):
        print("WARNING: free USDT is 0; you cannot market-buy right now unless you free USDT.")
    print(f"OCO levels (rounded): TP={tp_dec}  SL(stop)={sl_stop_dec}  SL(limit)={sl_limit_dec}")

    ans = input(f"Do you want to BUY {symbol} now and place OCO exits? (yes/no): ").strip().lower()
    if ans not in ("y", "yes"):
        print("Trade cancelled.")
        return

    if not live_trade:
        print("\n[DRY-RUN] Would buy + place OCO exits for:", symbol)
        return

    spend_usdt = (free_usdt * Decimal("0.999")).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    if spend_usdt <= Decimal("0"):
        print(f"ERROR: No available USDT to trade (free={free_usdt}).")
        return

    print(f"\nUSDT available: {free_usdt}  -> spending: {spend_usdt}  (spot, market, no leverage)")
    buy_resp = client.market_buy_quote_qty(symbol, spend_usdt)

    executed_qty = Decimal(str(buy_resp.get("executedQty", "0")))
    cumm_quote = Decimal(str(buy_resp.get("cummulativeQuoteQty", "0")))

    if executed_qty <= 0:
        print("ERROR: Buy returned executedQty=0. Response:")
        print(json.dumps(buy_resp, indent=2))
        return

    # Use actual FREE after buy (fees may reduce it)
    acct2 = client.account()
    free_base2 = _get_free_asset(acct2, base_asset)
    qty_for_exit = _round_step(free_base2, step_size)

    print("\nBUY filled:")
    print(f"  executedQty:            {executed_qty}")
    print(f"  quote spent (USDT):     {cumm_quote}")
    print(f"  free {base_asset}:       {free_base2}")
    print(f"  qty for exits (rounded): {qty_for_exit}")

    if qty_for_exit <= 0:
        print(f"ERROR: After buy, free {base_asset} rounds to 0; cannot place exits automatically.")
        print("You may need to place exits manually in Binance Spot UI.")
        return

    # Try OCO
    try:
        oco_resp = client.oco_sell(symbol=symbol, quantity=qty_for_exit, tp_price=tp_dec, sl_stop=sl_stop_dec, sl_limit=sl_limit_dec)
        print("\nOCO placed successfully:")
        print(json.dumps(oco_resp, indent=2))
        return
    except Exception as e:
        print(f"\nOCO failed; falling back to TP-first then conditional SL. Reason: {e}")

    # Fallback: TP now, SL later if needed
    tp_resp = client.order_limit_sell(symbol, qty_for_exit, tp_dec)
    print("Placed TP LIMIT sell:")
    print(json.dumps(tp_resp, indent=2))
    _monitor_then_place_sl(
        client=client,
        symbol=symbol,
        base_asset=base_asset,
        sl_stop=sl_stop_dec,
        sl_limit=sl_limit_dec,
        step_size=step_size,
        poll_sec=5,
    )


# =============================================================================
# Engine (SINGLE POSITION)
# =============================================================================

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
    ask_to_buy_today: bool,
    live_trade: bool,
) -> None:
    session = requests.Session()
    cache = load_cache(cache_path)
    last_p_by_symbol: Dict[str, float] = {}

    # Build signals by ENTRY day (file_date + lag)
    sig_by_entry_day: Dict[date, List[Signal]] = {}
    for s in signals:
        e = s.file_date + timedelta(days=entry_lag_days)
        sig_by_entry_day.setdefault(e, []).append(s)

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

        # 1) Update/exit existing position
        if pos is not None and pos.status == "OPEN":
            day_candle = candles_by_symbol.get(pos.symbol, {}).get(d)

            if not day_header_printed:
                print(f"\n----\nDate {fmt_d(d)}")
                day_header_printed = True

            if day_candle is None:
                print(f"Holding {pos.symbol}: (missing 1D candle for {d.isoformat()})")
            else:
                pos.last_close = day_candle.c
                if not (pos.peak_close == pos.peak_close):
                    pos.peak_close = day_candle.c
                pos.peak_close = max(pos.peak_close, day_candle.c)

                if d >= pos.entry_day:
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

                    # Max hold (CLOSE-based)
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

        # 2) Entries + rotation
        entry_signals = sig_by_entry_day.get(d, [])
        if entry_signals:
            if not day_header_printed:
                print(f"\n----\nDate {fmt_d(d)}")
                day_header_printed = True

            # print raw blocks for the source file_date(s)
            src_file_dates = sorted({s.file_date for s in entry_signals})
            for fd in src_file_dates:
                blk = raw_blocks.get(fd, [])
                if blk:
                    print("")
                    for line in blk:
                        print(line)

            # best p per symbol for TODAY
            best_today_by_symbol: Dict[str, Signal] = {}
            for s in entry_signals:
                cur = best_today_by_symbol.get(s.symbol)
                if cur is None or s.p > cur.p:
                    best_today_by_symbol[s.symbol] = s

            # Entry filter: p > 0.61 OR prev_p < today_p (and today_p>0)
            # Entry filter (new rule):
            # - If symbol seen before: require today's p > last time's p (and still >= 0.60)
            eligible: List[Signal] = []
            for sym, s in best_today_by_symbol.items():
                prev_p = last_p_by_symbol.get(sym)

                not_seen_before_and_ok = (prev_p is None and s.p >= 0.60)
                higher_than_prev = (prev_p is not None and s.p > prev_p)  # rule 3 (covers rule 1 too)

                if not_seen_before_and_ok or higher_than_prev:
                    eligible.append(s)
            if not eligible:
                for sym, s in best_today_by_symbol.items():
                    last_p_by_symbol[sym] = s.p
                print("\n(note) Signals present but none passed entry filter: "
                      "new symbols need p>=0.60; previously-seen symbols need p>=0.60 AND p>prev_p.")
                d += timedelta(days=1)
                continue


            eligible.sort(key=lambda s: (s.p, s.symbol), reverse=True)
            chosen = eligible[0]
            chosen_sym = chosen.symbol

            # rotation: if holding different symbol, exit at prev close and enter new at today's open
            if pos is not None and pos.status == "OPEN" and pos.symbol != chosen_sym:
                prev_day = d - timedelta(days=1)
                prev_candle = candles_by_symbol.get(pos.symbol, {}).get(prev_day)

                if prev_candle is None:
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
                if pos.symbol == chosen_sym:
                    print(f"\n(note) Signal for {chosen_sym} arrived but already holding it. No action.")
                else:
                    print(f"\n(note) Signal for {chosen_sym} arrived but still holding {pos.symbol}. No entry.")

            # Update last seen p for symbols that had signals today
            for sym, s in best_today_by_symbol.items():
                last_p_by_symbol[sym] = s.p

        d += timedelta(days=1)

    # Final summaries
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

    # Option B live prompt: only if REPLAY ends OPEN
    if ask_to_buy_today:
        if pos is None or pos.status != "OPEN":
            print("\n" + "=" * 80)
            print("BINANCE LIVE PROMPT")
            print("=" * 80)
            print("Replay ended FLAT (no OPEN position). Option B does nothing.")
            return

        _prompt_manage_replay_open_position_spot(
            replay_pos=pos,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            live_trade=live_trade,
        )


# =============================================================================
# CLI
# =============================================================================

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-file", required=True)
    ap.add_argument("--entry-lag-days", type=int, default=1)

    ap.add_argument("--tp-pct", type=float, default=0.20)
    ap.add_argument("--sl-pct", type=float, default=0.12)

    ap.add_argument("--start", default="", help="Optional override YYYY-MM-DD")
    ap.add_argument("--end", default="", help="Optional override YYYY-MM-DD")
    ap.add_argument("--cache-path", default="binance_cache.json")

    ap.add_argument("--intraday-interval", default="1m", choices=["1m", "3m", "5m", "15m", "30m", "1h"])
    ap.add_argument(
        "--ambiguous-fallback",
        default="sl-first",
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

    # Live prompt + trading
    ap.add_argument("--ask-to-buy-today", action="store_true", help="(Option B) If replay ends OPEN, prompt to place OCO exits on that coin.")
    ap.add_argument("--live-trade", action="store_true", help="Actually place Binance Spot orders (else dry-run).")

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
        ask_to_buy_today=bool(args.ask_to_buy_today),
        live_trade=bool(args.live_trade),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
