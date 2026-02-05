import requests
import boto3
import json
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from typing import Optional, List, Tuple, Dict, Any

# ============================================================
# EMAIL CONFIG
# ============================================================
recipient_email = "mina.moussa@hotmail.com"
sender_email = "minamoussa903@gmail.com"
sender_password = "thjj eryc yzym dylb"
SENDER_PASSWORD_CLEAN = (sender_password or "").replace(" ", "")

# ============================================================
# S3 CONFIG
# ============================================================
S3_BUCKET = "my-crypto-alerts-config"
S3_CONFIG_KEY = "config.json"
LOCAL_STATE_FILE_TEMPLATE = "/tmp/{symbol}_state.json"

# ============================================================
# BINANCE ENDPOINTS
# ============================================================
BINANCE_TICKER_24HR_URL = "https://api.binance.com/api/v3/ticker/24hr"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

LOG_FILE = "/tmp/evr_since_volume_alert.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s %(message)s")

REQUEST_TIMEOUT_SECS = 15

# ============================================================
# STRATEGY SETTINGS
# ============================================================
SELL_PCT_ABOVE_OPEN = 0.07  # ✅ sell is ALWAYS 7% above since_open

# ============================================================
# EVR SETTINGS
# ============================================================
EVR_INTERVAL = "1h"
KLINES_PAGE_LIMIT = 1000
EVR_EXCLUDE_RECENT = 2

PIVOT_LEFT = 3
PIVOT_RIGHT = 3

BREAKOUT_BUFFER_PCT = 0.001
DEBUG_EVR = False


# ============================================================
# HELPERS
# ============================================================
def parse_since_date(s: str) -> datetime:
    """
    Accepts:
      - "22-12-2025" (DD-MM-YYYY)
      - "2025-12-22" (YYYY-MM-DD)
      - ISO (best effort)
    Returns UTC datetime at 00:00:00.
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("since date is empty")

    for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            pass

    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    except Exception:
        raise ValueError(f"Unsupported since date format: {s}")


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def calc_gain_pct(entry: float, sell: float) -> Optional[float]:
    if entry <= 0:
        return None
    return ((sell - entry) / entry) * 100.0


def _as_float(x: Any, field: str) -> float:
    try:
        return float(x)
    except Exception:
        raise ValueError(f"'{field}' must be a number")


# ============================================================
# STATE
# ============================================================
def load_state(symbol: str) -> Dict[str, Any]:
    path = LOCAL_STATE_FILE_TEMPLATE.format(symbol=symbol)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                state = json.load(f)
            return {
                "met": bool(state.get("met", False)),
                "last_checked": state.get("last_checked"),
                "last_resistance": state.get("last_resistance"),
                "since": state.get("since"),
                "since_open": state.get("since_open"),
                "sell": state.get("sell"),
            }
        except Exception as e:
            logging.error(f"Error loading state for {symbol}: {e}")
    return {
        "met": False,
        "last_checked": None,
        "last_resistance": None,
        "since": None,
        "since_open": None,
        "sell": None,
    }


def save_state(symbol: str, met: bool, evr: float, since: str, since_open: float, sell: float):
    payload = {
        "met": bool(met),
        "last_checked": datetime.now(timezone.utc).isoformat(),
        "last_resistance": evr,
        "since": since,
        "since_open": since_open,
        "sell": sell,
    }
    try:
        with open(LOCAL_STATE_FILE_TEMPLATE.format(symbol=symbol), "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logging.error(f"Error saving state for {symbol}: {e}")


# ============================================================
# CONFIG (S3 config.json)
# ============================================================
def load_remote_config() -> List[Dict[str, Any]]:
    """
    config.json per alert:
      - {"symbol":"MMTUSDT","volume":5500000,"since":"22-12-2025"}

    Backward compatible:
      - {"symbol":"MMTUSDT","volume_threshold":5500000,"since":"2025-12-22"}

    NOTE: 'sell' is NOT used anymore. Sell is derived = open_since * 1.07
    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_CONFIG_KEY)
    data = json.loads(obj["Body"].read().decode("utf-8"))

    alerts = data.get("alerts", [])
    if not isinstance(alerts, list) or not alerts:
        raise ValueError("config.json must contain a non-empty 'alerts' array.")

    normalized: List[Dict[str, Any]] = []
    for idx, a in enumerate(alerts):
        if not isinstance(a, dict):
            raise ValueError(f"alerts[{idx}] must be an object")

        if "symbol" not in a:
            raise ValueError(f"alerts[{idx}] must include 'symbol'")

        symbol = str(a["symbol"]).upper().strip()
        if not symbol:
            raise ValueError(f"alerts[{idx}].symbol must be non-empty")

        if "volume" in a:
            vol_thr = _as_float(a["volume"], f"alerts[{idx}].volume")
        elif "volume_threshold" in a:
            vol_thr = _as_float(a["volume_threshold"], f"alerts[{idx}].volume_threshold")
        else:
            raise ValueError(f"alerts[{idx}] must include 'volume' (or 'volume_threshold')")

        since = str(a.get("since", "")).strip()
        if not since:
            raise ValueError(f"alerts[{idx}] must include 'since' (e.g. '22-12-2025')")

        forbidden = {"price", "price_threshold", "volume_price", "volume_price_threshold"}
        bad = forbidden.intersection(set(a.keys()))
        if bad:
            raise ValueError(
                f"alerts[{idx}] contains unsupported field(s): {sorted(list(bad))}. "
                f"Price is auto-derived using EVR since 'since'."
            )

        normalized.append({"symbol": symbol, "volume_threshold": vol_thr, "since": since})

    return normalized


# ============================================================
# EMAIL
# ============================================================
def send_email(subject: str, body: str):
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, SENDER_PASSWORD_CLEAN)
            server.send_message(msg)
        logging.info(f"Email sent: {subject}")
        print(f"Email sent: {subject}")
    except Exception as e:
        logging.error(f"SMTP send failed: {e}")
        print(f"Error sending email: {e}")


# ============================================================
# BINANCE
# ============================================================
def fetch_ticker_24hr(symbol: str) -> Tuple[float, float]:
    r = requests.get(
        BINANCE_TICKER_24HR_URL, params={"symbol": symbol}, timeout=REQUEST_TIMEOUT_SECS
    )
    r.raise_for_status()
    d = r.json()
    return float(d["volume"]), float(d["lastPrice"])


def fetch_klines_paged(symbol: str, interval: str, start_time_ms: int, end_time_ms: int) -> List[dict]:
    out: List[dict] = []
    cur = start_time_ms

    while True:
        r = requests.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cur,
                "endTime": end_time_ms,
                "limit": KLINES_PAGE_LIMIT,
            },
            timeout=REQUEST_TIMEOUT_SECS,
        )
        r.raise_for_status()
        arr = r.json()
        if not arr:
            break

        for k in arr:
            out.append(
                {
                    "open_time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": int(k[6]),
                }
            )

        last_open = int(arr[-1][0])
        if last_open == cur:
            break
        cur = last_open + 1

        if len(arr) < KLINES_PAGE_LIMIT:
            break

    return out


def fetch_open_price_since(symbol: str, interval: str, since_str: str) -> Optional[float]:
    """
    Open price of the FIRST candle whose open_time >= since date (UTC).
    """
    since_dt = parse_since_date(since_str)
    start_ms = dt_to_ms(since_dt)

    r = requests.get(
        BINANCE_KLINES_URL,
        params={"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": 1},
        timeout=REQUEST_TIMEOUT_SECS,
    )
    r.raise_for_status()
    arr = r.json()
    if not arr:
        return None
    return float(arr[0][1])


# ============================================================
# PIVOT HELPERS
# ============================================================
def pivot_highs(highs: List[float], left: int, right: int) -> List[int]:
    idxs = []
    n = len(highs)
    for i in range(left, n - right):
        v = highs[i]
        if all(v > x for x in highs[i - left : i]) and all(v >= x for x in highs[i + 1 : i + right + 1]):
            idxs.append(i)
    return idxs


# ============================================================
# EVR SINCE DATE
#   Anchor low = lowest low since date
#   EVR = first pivot-high after anchor low
# ============================================================
def identify_evr_since(symbol: str, since_str: str) -> Optional[float]:
    since_dt = parse_since_date(since_str)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = dt_to_ms(since_dt)

    klines = fetch_klines_paged(symbol, EVR_INTERVAL, start_ms, now_ms)
    if not klines or len(klines) < (PIVOT_LEFT + PIVOT_RIGHT + 20):
        return None

    if EVR_EXCLUDE_RECENT and len(klines) > EVR_EXCLUDE_RECENT:
        klines = klines[:-EVR_EXCLUDE_RECENT]

    lows = [k["low"] for k in klines]
    highs = [k["high"] for k in klines]

    anchor_low_idx = min(range(len(lows)), key=lambda i: lows[i])

    hi_idxs = pivot_highs(highs, PIVOT_LEFT, PIVOT_RIGHT)
    post_hi = [i for i in hi_idxs if i > anchor_low_idx]
    if not post_hi:
        return None

    evr_idx = post_hi[0]
    evr = float(highs[evr_idx])

    if DEBUG_EVR:
        anchor_time = datetime.fromtimestamp(klines[anchor_low_idx]["open_time"] / 1000, tz=timezone.utc)
        evr_time = datetime.fromtimestamp(klines[evr_idx]["open_time"] / 1000, tz=timezone.utc)
        print(
            f"[{symbol}] EVR-since debug: since={since_str} "
            f"anchor_low={lows[anchor_low_idx]:.10f} @ {anchor_time.isoformat()} "
            f"evr={evr:.10f} @ {evr_time.isoformat()} "
            f"candles={len(klines)}"
        )

    return evr


# ============================================================
# PROCESS ONE ALERT
# ============================================================
def process_alert(alert: Dict[str, Any]):
    symbol = alert["symbol"]
    vol_thr = alert["volume_threshold"]
    since = alert["since"]

    try:
        volume, price = fetch_ticker_24hr(symbol)
    except Exception as e:
        logging.error(f"[{symbol}] Failed to fetch ticker: {e}")
        return

    try:
        since_open = fetch_open_price_since(symbol, EVR_INTERVAL, since)
    except Exception as e:
        logging.error(f"[{symbol}] Failed to fetch since-open price: {e}")
        since_open = None

    try:
        evr = identify_evr_since(symbol, since)
    except Exception as e:
        logging.error(f"[{symbol}] Failed to identify EVR since {since}: {e}")
        evr = None

    if evr is None or since_open is None:
        print(f"[{symbol}] missing EVR or since_open (evr={evr}, since_open={since_open})")
        return

    # Sell is ALWAYS 7% above since_open
    sell = since_open * (1.0 + SELL_PCT_ABOVE_OPEN)

    # Rule: if EVR > OPEN => buy at open (mark met immediately)
    evr_above_open = evr > since_open

    if evr_above_open:
        entry_price = since_open
        entry_label = "OPEN"
        met_now = True
        note = "EVR ABOVE OPEN"
        breakout_price = evr * (1.0 + BREAKOUT_BUFFER_PCT)  # still log it
    else:
        entry_price = evr
        entry_label = "EVR"
        breakout_price = evr * (1.0 + BREAKOUT_BUFFER_PCT)
        met_now = (volume >= vol_thr) and (price >= breakout_price)
        note = "WAITING FOR BREAKOUT"

    gain_pct = calc_gain_pct(entry_price, sell)
    gain_str = "N/A" if gain_pct is None else f"{gain_pct:.2f}%"

    state = load_state(symbol)
    was_met = state["met"]

    print(
        f"[VOL+EVR-SINCE] {symbol}: since={since} "
        f"entry={entry_label}({entry_price:.10f}) "
        f"sell={sell:.10f} gain={gain_str} | "
        f"thr vol={vol_thr:,.0f}, obs vol={volume:,.0f}, "
        f"price={price:.10f}, evr={evr:.10f}, breakout={breakout_price:.10f}, "
        f"open={since_open:.10f}, met_now={met_now}, was_met={was_met}, note={note}"
    )

    if met_now and not was_met:
        if evr_above_open:
            subject = f"{symbol} EVR ABOVE OPEN – buy at open"
            reason = "EVR ABOVE OPEN (buy at open)"
        else:
            subject = f"{symbol} EVR BREAKOUT – volume+price confirmed"
            reason = "BREAKOUT ABOVE EVR (volume+price)"

        body = (
            f"Signal: {reason}\n\n"
            f"Symbol: {symbol}\n"
            f"Since date (UTC): {since}\n\n"
            f"Open on since date ({EVR_INTERVAL}): {since_open:.10f}\n"
            f"EVR (since): {evr:.10f}\n"
            f"Breakout (+{BREAKOUT_BUFFER_PCT*100:.2f}%): {breakout_price:.10f}\n"
            f"Current price: {price:.10f}\n\n"
            f"Entry basis: {entry_label}\n"
            f"Entry price: {entry_price:.10f}\n"
            f"Sell (open + {SELL_PCT_ABOVE_OPEN*100:.2f}%): {sell:.10f}\n"
            f"Projected gain: {gain_str}\n\n"
            f"24h Volume: {volume:,.0f} (Threshold: {vol_thr:,.0f})\n"
        )

        logging.info(body)
        send_email(subject, body)

    save_state(symbol, met_now, evr, since, since_open, sell)


# ============================================================
# MAIN
# ============================================================
def check_conditions():
    try:
        alerts = load_remote_config()
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        print(f"Error loading config: {e}")
        return

    for alert in alerts:
        process_alert(alert)


if __name__ == "__main__":
    check_conditions()

