#!/usr/bin/env python3
"""
ETF Algo_withbuy.py  (etf_replay_from_signals + OPTIONAL Binance Spot buy + TP/SL exits)

REALISTIC exit ordering (backtest/replay):
- Uses Binance 1D candles for most checks.
- If BOTH TP and SL are hit in the same 1D day, it fetches intraday candles for that UTC day
  and determines which was hit first (no assumption), unless --ambiguous-fallback is set.

Defaults:
- start_day = earliest file=YYYY-MM-DD found in the signals file
- end_day   = today UTC (datetime.utcnow().date())

Usage (replay only):
  python ETF Algo_withbuy.py --signals-file signals.txt

Optional realism knobs:
  --intraday-interval 1m|3m|5m|15m|30m|1h   (default 5m)
  --ambiguous-fallback none|tp-first|sl-first  (default none)

Optional "ask to buy today" (LIVE trading prompt at end):
  --ask-to-buy-today

DRY RUN vs LIVE:
- default: DRY RUN (prints what it would do)
- to place real orders:
    --live-trade
    set env vars:
      BINANCE_API_KEY
      BINANCE_API_SECRET

LIVE TRADING LOGIC (idempotent / safe-ish):
- At end, finds today's candidate (entry_day == end_day) and prompts:
    "Do you want to buy XXX?"
- If YES:
  1) If you already have exit orders (TP/SL) open -> DO NOTHING (prevents stacking).
  2) Else if you already hold the base asset (free+locked >= stepSize) -> DO NOT buy again;
     try to place exits on FREE qty:
       - Try OCO (TP + SL) first
       - If OCO fails: place TP LIMIT, then monitor, and place SL only if price hits SL trigger.
  3) Else (no holding and no exits): BUY MARKET using ~all available USDT (spot, no leverage),
     then place exits (OCO preferred, otherwise fallback as above).

NOTES:
- This script assumes USDT-quoted symbols (like KITEUSDT) for buy automation.
- OCO may fail if fee/rounding reduces free qty; fallback handles that.
- Monitoring loop runs in foreground; Ctrl+C to stop (you may have open orders on Binance).

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


def parse_p_from_raw_line(raw_line: str) -> Optional[float]:
    m = P_RE.search(raw_line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


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

    # dedupe by day
    by_day: Dict[date, Candle] = {}
    for c in out:
        by_day[c.day] = c

    candles = [by_day[d] for d in sorted(by_day.keys())]
    candles = [c for c in candles if start_day <= c.day <= end_day]
    return candles


def fetch_intraday_klines_for_day(symbol: str, day: date, interval: str, session: requests.Session) -> List[IntraCandle]:
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
# Replay / strategy
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
    ambiguous_fallback: str,
) -> Optional[Tuple[str, float]]:
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

    intra = get_intraday_cached_for_day(
        pos.symbol, day_candle.day,
        interval=intraday_interval,
        session=session, cache=cache, cache_path=cache_path,
    )

    if not intra:
        if ambiguous_fallback == "tp-first":
            return ("TP_20", tp_price)
        if ambiguous_fallback == "sl-first":
            return ("SL_10", sl_price)
        return None

    for c in intra:
        hit_tp = c.h >= tp_price
        hit_sl = c.l <= sl_price

        if hit_tp and not hit_sl:
            return ("TP_20", tp_price)
        if hit_sl and not hit_tp:
            return ("SL_10", sl_price)
        if hit_tp and hit_sl:
            if ambiguous_fallback == "tp-first":
                return ("TP_20", tp_price)
            if ambiguous_fallback == "sl-first":
                return ("SL_10", sl_price)
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


# =============================================================================
# Idempotent live-exit helpers
# =============================================================================

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


def _prompt_and_trade_today_spot(
    *,
    today_signal: Signal,
    entry_open: float,
    tp_pct: float,
    sl_pct: float,
    live_trade: bool,
) -> None:
    symbol = today_signal.symbol.upper()
    base_asset = _base_asset_from_symbol(symbol)

    tp_price = entry_open * (1.0 + tp_pct)
    sl_price = entry_open * (1.0 - sl_pct)

    print("\n" + "=" * 80)
    print(f"Today candidate detected (UTC day): {symbol}")
    print(f"Detected from candle open={entry_open:.8g} => TP={tp_price:.8g}  SL={sl_price:.8g}")
    print("=" * 80)

    ans = input(f"Do you want to buy {symbol}? (yes/no): ").strip().lower()
    if ans not in ("y", "yes"):
        print("Trade cancelled.")
        return

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

    if not live_trade:
        print("\n[DRY-RUN] Would do idempotent buy+exit handling for:", symbol)
        print(f"[DRY-RUN] TP={tp_dec}  SL(stop)={sl_stop_dec} SL(limit)={sl_limit_dec}")
        return

    # Fetch current state
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

    # If already holding, don't buy again; place exits for FREE qty
    if total_base >= holding_threshold:
        free_base = _get_free_asset(acct, base_asset)
        qty = _round_step(free_base, step_size)

        print(f"\nDetected existing holding (free+locked): {total_base} {base_asset} (>= {holding_threshold}). Will NOT buy again.")

        if qty <= 0:
            print(
                f"Your {base_asset} is likely locked elsewhere (free={free_base}, total={total_base}).\n"
                "No free quantity available to place exits. Check Binance open orders."
            )
            return

        print(f"Placing exits for existing position qty={qty} {base_asset}...")

        # Try OCO
        try:
            oco_resp = client.oco_sell(symbol=symbol, quantity=qty, tp_price=tp_dec, sl_stop=sl_stop_dec, sl_limit=sl_limit_dec)
            print("OCO placed successfully for existing position:")
            print(json.dumps(oco_resp, indent=2))
            return
        except Exception as e:
            print(f"OCO failed, falling back to TP-first then conditional SL. Reason: {e}")

        # Fallback: TP now, SL only if needed
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

    # Else: buy using almost all USDT
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
# Replay engine
# =============================================================================

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
    ask_to_buy_today: bool,
    live_trade: bool,
) -> None:
    total_realised_gain = 0.0
    num_closed_trades = 0
    compounded_factor = 1.0

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
                    compounded_factor *= (1.0 + gain / 100.0)

                    print("\n" + "=" * 80)
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

    # Summary
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
    compounded_pct = (compounded_factor - 1.0) * 100.0
    print(f"Compounded gain:      {compounded_pct:.2f}%")

    # LIVE prompt at end: today candidate
    if ask_to_buy_today:
        today = end_day
        today_entries = sig_by_entry_day.get(today, [])
        if not today_entries:
            print("\n(No entry candidates for today UTC; nothing to prompt.)")
            return

        # Choose “best” today entry: highest parsed p=... else first
        scored: List[Tuple[float, int, Signal]] = []
        for idx, s in enumerate(today_entries):
            p = parse_p_from_raw_line(s.raw_line)
            score = p if p is not None else -1.0
            scored.append((score, -idx, s))
        scored.sort(reverse=True)
        best = scored[0][2]

        c = candles_by_symbol.get(best.symbol, {}).get(today)
        if c is None:
            print("\nERROR: Cannot prompt to buy; missing today's 1D candle open for:", best.symbol)
            return

        _prompt_and_trade_today_spot(
            today_signal=best,
            entry_open=float(c.o),
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

    ap.add_argument("--intraday-interval", default="5m", choices=["1m", "3m", "5m", "15m", "30m", "1h"])
    ap.add_argument(
        "--ambiguous-fallback",
        default="none",
        choices=["none", "tp-first", "sl-first"],
        help="If intraday cannot resolve (missing data or both hit in same intraday candle), what to do.",
    )

    ap.add_argument("--ask-to-buy-today", action="store_true", help="At the end, prompt to buy today's candidate.")
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
        ask_to_buy_today=bool(args.ask_to_buy_today),
        live_trade=bool(args.live_trade),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
