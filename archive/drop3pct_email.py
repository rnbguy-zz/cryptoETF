#!/usr/bin/env python3
"""
Per-symbol one-time email alert when LIVE price is >=3% BELOW since-date open.

- since_open = OPEN of the first candle (INTERVAL) at/after since-date (UTC)
- trigger_price = since_open * (1 - DROP_PCT/100)
- trigger when live_price <= trigger_price

Per-symbol toggle:
- When a symbol triggers, we write /tmp/crypto_alert_drop_<SYMBOL>.txt
- Next runs skip only that symbol, but keep checking others.

Config is read from S3_URI (same format you pasted).
"""

import os
import re
import json
import ssl
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import List, Optional, Tuple

import boto3
import requests

# ============================================================
# EMAIL CONFIG (already set on your box)
# ============================================================
recipient_email = "mina.moussa@hotmail.com"
sender_email = "minamoussa903@gmail.com"
sender_password = "thjj eryc yzym dylb"  # keep whatever you already pasted here

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

# ============================================================
# S3 CONFIG
# ============================================================
S3_URI = "s3://my-crypto-alerts-config/config.json"

# ============================================================
# RULE SETTINGS
# ============================================================
INTERVAL = "15m"
DROP_PCT = 3.0  # alert when live <= since_open * 0.97

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"

REQ_TIMEOUT = 12


# ============================================================
# HELPERS
# ============================================================
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"S3_URI must start with s3:// (got: {uri})")
    rest = uri[len("s3://"):]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parts[0], parts[1]

def parse_date_ms(d: str) -> int:
    d = (d or "").strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(d, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            pass
    raise ValueError(f"Bad date format: {d} (expected DD-MM-YYYY)")

def send_email(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.set_content(body)

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
        s.login(sender_email, sender_password)
        s.send_message(msg)

def read_s3_config() -> dict:
    bucket, key = parse_s3_uri(S3_URI)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())

def fetch_klines(symbol: str, limit: int, start_ms: Optional[int] = None) -> List[list]:
    params = {"symbol": symbol, "interval": INTERVAL, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_price(symbol: str) -> float:
    r = requests.get(BINANCE_PRICE_URL, params={"symbol": symbol}, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return float(r.json()["price"])

def safe_symbol_for_filename(symbol: str) -> str:
    # Keep only A-Z, 0-9, underscore
    sym = (symbol or "").upper().strip()
    sym = re.sub(r"[^A-Z0-9_]+", "_", sym)
    return sym

def toggle_path_for_symbol(symbol: str) -> str:
    sym = safe_symbol_for_filename(symbol)
    return f"/tmp/crypto_alert_drop_{sym}.txt"

@dataclass
class Alert:
    symbol: str
    since: str

def parse_alerts(cfg: dict) -> List[Alert]:
    alerts = cfg.get("alerts", [])
    if not isinstance(alerts, list):
        raise ValueError("Config must contain list field 'alerts'")
    out: List[Alert] = []
    for a in alerts:
        if not isinstance(a, dict):
            continue
        sym = str(a.get("symbol", "")).upper().strip()
        since = str(a.get("since", "")).strip()
        if sym and since:
            out.append(Alert(sym, since))
    return out

def since_open(symbol: str, since: str) -> float:
    start = parse_date_ms(since)
    kl = fetch_klines(symbol, 2, start)
    if not kl:
        raise RuntimeError(
            "No klines returned for since-open (symbol may be invalid/delisted or no data from since date)"
        )
    return float(kl[0][1])


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"=== RUN {utc_now()} ===")
    print(f"Rule: alert when live <= since_open * (1 - {DROP_PCT:.2f}%) ; per-symbol toggles in /tmp")

    cfg = read_s3_config()
    alerts = parse_alerts(cfg)

    triggered_this_run = []
    errors = []

    for a in alerts:
        tpath = toggle_path_for_symbol(a.symbol)
        if os.path.exists(tpath):
            print(f"[SKIP] {a.symbol} already alerted (toggle exists): {tpath}")
            continue

        try:
            so = since_open(a.symbol, a.since)
            live = get_price(a.symbol)
            trigger_price = so * (1.0 - DROP_PCT / 100.0)

            pct_from_open = ((live - so) / so) * 100.0 if so else 0.0
            is_trigger = live <= trigger_price  # ONLY if below (or equal) 3% from since-open

            print(
                f"{a.symbol} since={a.since} since_open={so:.8f} live={live:.8f} "
                f"from_open={pct_from_open:.2f}% trigger={trigger_price:.8f} hit={is_trigger}"
            )

            if is_trigger:
                # Email per symbol, and write toggle for that symbol only
                subject = f"Crypto Alert: {a.symbol} is down ≥{DROP_PCT:.0f}% from since-open"
                body_lines = [
                    f"Triggered at: {utc_now()} UTC",
                    f"Symbol: {a.symbol}",
                    f"Since date: {a.since}",
                    f"Interval: {INTERVAL}",
                    "",
                    f"Since-open: {so:.8f}",
                    f"Trigger price (since-open × {(1.0 - DROP_PCT/100.0):.4f}): {trigger_price:.8f}",
                    f"Live price: {live:.8f}",
                    f"From since-open: {pct_from_open:.2f}%",
                    "",
                    f"Toggle written: {tpath}",
                    f"Config: {S3_URI}",
                ]
                send_email(subject, "\n".join(body_lines) + "\n")

                with open(tpath, "w", encoding="utf-8") as f:
                    f.write(f"ALERT_SENT_UTC={utc_now()}\n")
                    f.write(f"SYMBOL={a.symbol}\n")
                    f.write(f"SINCE={a.since}\n")
                    f.write(f"INTERVAL={INTERVAL}\n")
                    f.write(f"DROP_PCT={DROP_PCT}\n")
                    f.write(f"SINCE_OPEN={so}\n")
                    f.write(f"TRIGGER_PRICE={trigger_price}\n")
                    f.write(f"LIVE_PRICE={live}\n")
                    f.write(f"CONFIG={S3_URI}\n")

                print(f"[ALERTED] {a.symbol} email sent + toggle written: {tpath}")
                triggered_this_run.append(a.symbol)

        except Exception as e:
            msg = f"{a.symbol} (since {a.since}) ERROR: {e}"
            errors.append(msg)
            print(msg)

    if triggered_this_run:
        print(f"[DONE] Alerted this run: {', '.join(triggered_this_run)}")
    else:
        print("[DONE] No symbols triggered this run.")

    if errors:
        print("\nErrors (non-fatal):")
        for e in errors[:50]:
            print(" -", e)


if __name__ == "__main__":
    main()

