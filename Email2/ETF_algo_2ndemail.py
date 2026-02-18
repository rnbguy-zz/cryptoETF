#!/usr/bin/env python3
"""
python3.9 ETF_algo_2ndemail.py --tp-pct 0.2 --sl-pct 0.12 --signals-file ../signals.txt --ask-to-buy-today --live-trade
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

FILE_RE = re.compile(r"^\s*\ufeff?\s*file\s*=?\s*(\d{4}-\d{2}-\d{2})\b", re.I)

BUY_RE = re.compile(r"^\s*BUY\s+([A-Z0-9]+)\b")
P_RE = re.compile(r"\bp\s*=\s*([0-9]*\.?[0-9]+)\b")  # parses "p=0.563"
RUNUP_RE = re.compile(
    r"\brunupToHigh\s*=\s*([0-9]*\.?[0-9]+)\s*%?",
    re.IGNORECASE
)
FEATDROP_RE = re.compile(r"\bfeatDropP\s*=\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE)



@dataclass(frozen=True)
class Signal:
    file_date: date
    symbol: str
    raw_line: str
    p: float  # 0.0 if missing
    runup_to_high_pct: Optional[float]  # None if missing; percent (e.g. 7.68)
    feat_drop_p: Optional[float]  # <-- add this


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

                mr = RUNUP_RE.search(line)
                runup = float(mr.group(1)) if mr else None

                mf = FEATDROP_RE.search(line)
                feat_drop_p = float(mf.group(1)) if mf else None

                signals.append(Signal(
                    file_date=cur_file_date,
                    symbol=sym,
                    raw_line=line.rstrip("\n"),
                    p=p,
                    runup_to_high_pct=runup,
                    feat_drop_p=feat_drop_p,
                ))

    signals.sort(key=lambda s: (s.file_date, s.symbol))
    return signals, raw_block


def load_signals_file(path: Path) -> Tuple[List[Signal], Dict[date, List[str]]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return parse_signals_text(text)


# =============================================================================
# Binance client + helpers (same as your existing script)
# =============================================================================

BINANCE_BASE = "https://api.binance.com"
KLINES_EP = "/api/v3/klines"


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

    # endpoints
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

            if _round_step(total_base, step_size) <= 0:
                print("Position likely closed (no remaining base). Done.")
                return

            if sl_order is not None:
                time.sleep(poll_sec)
                continue

            px = Decimal(str(client.ticker_price(symbol)))
            if px <= sl_stop:
                print(f"\nSL trigger hit: price={px} <= {sl_stop}")

                if tp_order is not None:
                    try:
                        oid = int(tp_order.get("orderId"))
                        print(f"Cancelling TP orderId={oid} ...")
                        client.cancel_order(symbol, oid)
                    except Exception as e:
                        print(f"WARNING: Failed to cancel TP order: {e}")

                acct2 = client.account()
                free_base2 = _get_free_asset(acct2, base_asset)
                qty = _round_step(free_base2, step_size)

                if qty <= 0:
                    print(f"No FREE {base_asset} available to place SL (free={free_base2}). Done.")
                    return

                sl_resp = client.order_stop_loss_limit_sell(symbol, qty, sl_stop, sl_limit)
                print("Placed SL STOP_LOSS_LIMIT sell:")
                print(json.dumps(sl_resp, indent=2))

            time.sleep(poll_sec)

        except KeyboardInterrupt:
            print("\nStopped monitoring. Note: You may have open orders on Binance.")
            return
        except Exception as e:
            print(f"Monitor loop error: {e}")
            time.sleep(max(5, poll_sec))


# =============================================================================
# The logic you asked for: latest block only, filter p>0.75 and runupToHigh < 10%
# =============================================================================

def _latest_block_shortlist(signals: List[Signal], *, p_min: float = 0.75, runup_max_pct: float = 10.0) -> Tuple[date, List[Signal]]:
    """
    Returns (latest_file_date, shortlist_signals) from the MOST RECENT file= block only.

    - best p per symbol within that block
    - keep only where p > p_min and runupToHigh exists and runupToHigh < runup_max_pct
    """
    latest = max(s.file_date for s in signals)
    latest_sigs = [s for s in signals if s.file_date == latest]

    best_by_symbol: Dict[str, Signal] = {}
    for s in latest_sigs:
        cur = best_by_symbol.get(s.symbol)
        if cur is None or s.p > cur.p:
            best_by_symbol[s.symbol] = s

    shortlist: List[Signal] = []
    for s in best_by_symbol.values():
        if not (s.p > p_min):
            continue
        if s.runup_to_high_pct is None:
            continue
        if not (s.runup_to_high_pct < runup_max_pct):
            continue
        shortlist.append(s)

    shortlist.sort(key=lambda x: (x.p, x.symbol), reverse=True)
    return latest, shortlist


def _prompt_pick_from_shortlist(shortlist: List[Signal], latest_day: date) -> Optional[Signal]:
    print("\n" + "=" * 80)
    print(f"LATEST SIGNALS ONLY: file={latest_day.isoformat()}")
    print("Filter: p > 0.75 AND runupToHigh < 10%")
    print("=" * 80)

    if not shortlist:
        print("No symbols matched in the latest file block.")
        return None

    for i, s in enumerate(shortlist, start=1):
        fd = "" if s.feat_drop_p is None else f"  featDropP={s.feat_drop_p:.3f}"
        print(f"{i:02d}) {s.symbol:12s}  p={s.p:.3f}  runupToHigh={s.runup_to_high_pct:.2f}%{fd}")

    if any((x.feat_drop_p or 0.0) > 0.2 for x in shortlist):
        print("\nWARNING: featDropP > 0.2 detected for at least one candidate — it may dip ~12% first.")

    if len(shortlist) == 1:
        s = shortlist[0]
        ans = input(f"\nOnly one match: {s.symbol}. Manage it now (place OCO if holding, else buy+OCO)? (yes/no): ").strip().lower()

        if ans in ("y", "yes"):
            return s
        print("Cancelled.")
        return None

    ans = input("\nPick a number to buy (or press Enter to cancel): ").strip()
    if not ans:
        print("Cancelled.")
        return None
    try:
        idx = int(ans)
    except ValueError:
        print("Invalid number. Cancelled.")
        return None

    if not (1 <= idx <= len(shortlist)):
        print("Out of range. Cancelled.")
        return None

    s = shortlist[idx - 1]
    ans2 = input(f"Manage {s.symbol} now (place OCO if holding, else buy+OCO)? (yes/no): ").strip().lower()

    if ans2 not in ("y", "yes"):
        print("Cancelled.")
        return None
    return s

def _utc_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_1d_open_for_day(symbol: str, day: date, session: Optional[requests.Session] = None) -> float:
    """
    Returns the 1D candle OPEN for `day` (UTC).
    Uses Binance /api/v3/klines interval=1d with start/end bounds for that day.
    """
    s = session or requests.Session()
    start_dt = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)

    params = {
        "symbol": symbol.upper(),
        "interval": "1d",
        "startTime": _utc_ms(start_dt),
        "endTime": _utc_ms(end_dt),
        "limit": 2,
    }

    r = s.get(BINANCE_BASE + KLINES_EP, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list) or not data:
        raise RuntimeError(f"No 1D kline returned for {symbol} on {day.isoformat()}")

    # kline: [openTime, open, high, low, close, ...]
    return float(data[0][1])


def _buy_and_place_exits(
    *,
    symbol: str,
    signal_file_date: date,
    feat_drop_p: Optional[float] = None,
    tp_pct: float,
    sl_pct: float,
    live_trade: bool,
    entry_lag_days: int = 1,
) -> None:
    """
    Uses entry-day OPEN as the reference:
      entry_day = signal_file_date + entry_lag_days   (UTC)
      entry_open = 1D OPEN for entry_day
      TP = entry_open * (1 + tp_pct)
      SL = entry_open * (1 - sl_pct)

    Prints:
      - entry_day open
      - current live price
      - % change from entry_open -> live
      - TP/SL levels (raw + rounded)

    Then:
      - If exits already exist -> do nothing
      - If holding (any positive total) -> prompt to place OCO exits (NO BUY)
      - Else -> prompt to buy then place exits
    """
    symbol = symbol.upper()
    base_asset = _base_asset_from_symbol(symbol)

    # Reference day from signals
    entry_day = signal_file_date + timedelta(days=int(entry_lag_days))

    # Fetch entry-day OPEN (UTC) + live price (public endpoint)
    pub_sess = requests.Session()
    try:
        entry_open = fetch_1d_open_for_day(symbol, entry_day, session=pub_sess)
    except Exception as e:
        print(f"ERROR: Could not fetch 1D OPEN for {symbol} on {entry_day.isoformat()}: {e}")
        return

    try:
        live_px = float(pub_sess.get(
            BINANCE_BASE + "/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=15
        ).json()["price"])
    except Exception as e:
        print(f"ERROR: Could not fetch live price for {symbol}: {e}")
        return

    # Compute levels off the entry OPEN (your rule)
    tp_price = entry_open * (1.0 + tp_pct)
    sl_price = entry_open * (1.0 - sl_pct)

    # % change from entry_open -> live price
    chg_pct = (live_px / max(entry_open, 1e-12) - 1.0) * 100.0

    print("\n" + "=" * 80)
    print(f"REFERENCE FROM SIGNALS: file={signal_file_date.isoformat()}  -> entry_day={entry_day.isoformat()} (UTC)")
    print(f"Entry-day OPEN:      {entry_open:.8g}")
    print(f"Live price now:      {live_px:.8g}")
    print(f"% change (open→now): {chg_pct:+.2f}%")
    print(f"Planned TP (raw):    {tp_price:.8g}   (+{tp_pct*100:.0f}%)")
    print(f"Planned SL (raw):    {sl_price:.8g}   (-{sl_pct*100:.0f}%)")
    print("=" * 80)

    api_key = os.getenv("BINANCE_API_KEY", "tE06bWu6VfzgIB1wlyZfZzaZwPe0F6RyVQrp0Fh7B8fvTzNyhxe8UZSrJV3y0Iu0").strip()
    api_secret = os.getenv("BINANCE_API_SECRET",
                           "bFBUdNU7c8HBW3pNt2CnT1m7RlASUw6ReFsWYRqPFWLyj7NIjVFLK7j2BIaFTGLf").strip()
    if not api_key or not api_secret:
        print("ERROR: BINANCE_API_KEY / BINANCE_API_SECRET env vars not set. Cannot trade.")
        return

    client = BinanceClient(api_key=api_key, api_secret=api_secret)

    # Rounding filters
    ex = client.exchange_info(symbol)
    tick_size, step_size = _extract_symbol_filters(ex)

    tp_dec = _round_step(_dec(tp_price), tick_size)
    sl_stop_dec = _round_step(_dec(sl_price), tick_size)
    sl_limit_dec = _round_step(sl_stop_dec * Decimal("0.999"), tick_size)

    print(f"Rounded levels:      TP={tp_dec}  SL(stop)={sl_stop_dec}  SL(limit)={sl_limit_dec}")
    print("=" * 80)

    if tp_dec <= 0 or sl_stop_dec <= 0 or sl_limit_dec <= 0:
        print("ERROR: Computed TP/SL invalid after rounding; aborting.")
        return

    # Account state
    acct = client.account()
    free_usdt = _get_free_usdt(acct)
    total_base = _get_total_asset(acct, base_asset)
    free_base = _get_free_asset(acct, base_asset)

    # Existing exits?
    open_orders = client.open_orders(symbol)
    tp_order, sl_order = _find_exit_orders(open_orders)
    if tp_order or sl_order:
        print("\nExisting exit orders detected — not placing another one.")
        if tp_order:
            print(f"  TP: orderId={tp_order.get('orderId')} type={tp_order.get('type')} price={tp_order.get('price')}")
        if sl_order:
            print(f"  SL: orderId={sl_order.get('orderId')} type={sl_order.get('type')} stopPrice={sl_order.get('stopPrice')} price={sl_order.get('price')}")
        return

    # ---- HOLDING PATH ----
    if total_base > Decimal("0"):
        qty = _round_step(free_base, step_size)

        print(f"\nHolding {symbol} detected. total={total_base} {base_asset}, free={free_base} {base_asset}, stepSize={step_size}")
        print(f"Rounded FREE qty for exits: {qty}")

        if qty <= 0:
            print(
                f"You're holding {base_asset}, but FREE qty rounds to 0 at stepSize={step_size}.\n"
                f"- total={total_base}, free={free_base}\n"
                "No buy prompt will be shown."
            )
            return

        ans = input(f"Place OCO exits for {symbol} now (based on entry_day OPEN)? (yes/no): ").strip().lower()
        if ans not in ("y", "yes"):
            print("Cancelled.")
            return

        if not live_trade:
            print(f"\n[DRY-RUN] Would place OCO exits for qty={qty} {base_asset} on {symbol}")
            return

        try:
            oco_resp = client.oco_sell(symbol=symbol, quantity=qty, tp_price=tp_dec, sl_stop=sl_stop_dec, sl_limit=sl_limit_dec)
            print("\nOCO placed successfully:")
            print(json.dumps(oco_resp, indent=2))
            return
        except Exception as e:
            print(f"\nOCO failed; falling back to TP-first then conditional SL. Reason: {e}")

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

    # ---- NOT HOLDING PATH ----
    print(f"\nNot holding {base_asset}. USDT free={free_usdt}")
    if free_usdt <= Decimal("0"):
        print("WARNING: free USDT is 0; you cannot market-buy right now unless you free USDT.")
        return

    ans = input(f"Buy {symbol} now and place exits based on entry_day OPEN? (yes/no): ").strip().lower()
    if ans not in ("y", "yes"):
        print("Trade cancelled.")
        return

    spend_usdt = (free_usdt * Decimal("0.999")).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    if spend_usdt <= Decimal("0"):
        print(f"ERROR: No available USDT to trade (free={free_usdt}).")
        return

    if not live_trade:
        print(f"\n[DRY-RUN] Would market BUY {symbol} spending {spend_usdt} USDT, then place exits.")
        return

    print(f"\nUSDT available: {free_usdt}  -> spending: {spend_usdt}")
    buy_resp = client.market_buy_quote_qty(symbol, spend_usdt)

    executed_qty = Decimal(str(buy_resp.get("executedQty", "0")))
    if executed_qty <= 0:
        print("ERROR: Buy returned executedQty=0. Response:")
        print(json.dumps(buy_resp, indent=2))
        return

    acct2 = client.account()
    free_base2 = _get_free_asset(acct2, base_asset)
    qty_for_exit = _round_step(free_base2, step_size)

    print("\nBUY filled:")
    print(f"  executedQty:         {executed_qty}")
    print(f"  free {base_asset}:    {free_base2}")
    print(f"  qty for exits:       {qty_for_exit}")

    if qty_for_exit <= 0:
        print(f"ERROR: After buy, free {base_asset} rounds to 0; cannot place exits automatically.")
        return

    try:
        oco_resp = client.oco_sell(symbol=symbol, quantity=qty_for_exit, tp_price=tp_dec, sl_stop=sl_stop_dec, sl_limit=sl_limit_dec)
        print("\nOCO placed successfully:")
        print(json.dumps(oco_resp, indent=2))
        return
    except Exception as e:
        print(f"\nOCO failed; falling back to TP-first then conditional SL. Reason: {e}")

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



def run_latest_block_buy_prompt(
    *,
    signals: List[Signal],
    tp_pct: float,
    sl_pct: float,
    live_trade: bool,
) -> None:
    latest_day, shortlist = _latest_block_shortlist(signals, p_min=0.75, runup_max_pct=10.0)

    print("\n" + "-" * 80)
    print(f"LATEST FILE BLOCK DETECTED: file={latest_day.isoformat()}")
    print(f"BUY signals in latest block: {sum(1 for s in signals if s.file_date == latest_day)}")
    print("-" * 80)

    chosen = _prompt_pick_from_shortlist(shortlist, latest_day)

    # MUST check this BEFORE using chosen.*
    if chosen is None:
        return

    # chosen-specific featDropP warning
    if chosen.feat_drop_p is not None and chosen.feat_drop_p > 0.2:
        print("\n" + "!" * 80)
        print(f"WARNING: {chosen.symbol} featDropP={chosen.feat_drop_p:.3f} > 0.2 — it may dip ~12% first.")
        print("!" * 80 + "\n")

    _buy_and_place_exits(
        symbol=chosen.symbol,
        signal_file_date=chosen.file_date,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        live_trade=live_trade,
        entry_lag_days=1,
    )



# =============================================================================
# Minimal "main" wrapper (keeps your replay intact in your real file)
# Here: only wires --ask-to-buy-today to the latest-block prompt you want.
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals-file", default="./signals2.txt")
    ap.add_argument("--tp-pct", type=float, default=0.20)
    ap.add_argument("--sl-pct", type=float, default=0.12)
    ap.add_argument("--ask-to-buy-today", action="store_true", default=True)
    ap.add_argument("--live-trade", action="store_true", default=True)
    args = ap.parse_args()

    signals_path = Path(args.signals_file)
    if not signals_path.exists():
        print(f"ERROR: signals file not found: {signals_path}", file=sys.stderr)
        return 2

    signals, _raw_blocks = load_signals_file(signals_path)
    print("\n" + "=" * 80)
    print("SIGNALS PARSE SUMMARY")
    print("=" * 80)
    print(f"Signals file: {signals_path}")
    print(f"Total BUY signals parsed: {len(signals)}")

    if signals:
        days = sorted({s.file_date for s in signals})
        print(f"File date range: {days[0]} -> {days[-1]}")
        print(f"Unique symbols: {len({s.symbol for s in signals})}")
    else:
        print("WARNING: No BUY lines parsed.")

    print("=" * 80 + "\n")

    print("\nFIRST 10 PARSED BUY LINES:")
    for s in signals[:10]:
        print(f"  {s.file_date}  BUY {s.symbol}  p={s.p}  runup={s.runup_to_high_pct}  featDropP={s.feat_drop_p}")
    print("=" * 80 + "\n")
    # ==========================================================

    if not signals:
        print("ERROR: No signals found.", file=sys.stderr)
        return 2

    if args.ask_to_buy_today:
        run_latest_block_buy_prompt(
            signals=signals,
            tp_pct=float(args.tp_pct),
            sl_pct=float(args.sl_pct),
            live_trade=bool(args.live_trade),
        )
        return 0

    print("No action (run with --ask-to-buy-today to use latest-block buy prompt).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

