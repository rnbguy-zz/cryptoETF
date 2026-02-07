#!/usr/bin/env python3
import os
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
sender_password = "qhvi syra bbad gylu"  # keep whatever you already pasted here

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

# ============================================================
# S3 CONFIG + TOGGLE
# ============================================================
S3_URI = "s3://my-crypto-alerts-config/config.json"
TOGGLE_FILE = "/tmp/crypto_alert_sent.txt"

# ============================================================
# RULE SETTINGS
# ============================================================
INTERVAL = "15m"
TARGET_PCT = 7.0

PAGE_LIMIT = 1000
MAX_PAGES = 24

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"


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
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=12)
    r.raise_for_status()
    return r.json()

def get_price(symbol: str) -> float:
    r = requests.get(BINANCE_PRICE_URL, params={"symbol": symbol}, timeout=8)
    r.raise_for_status()
    return float(r.json()["price"])

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
        raise RuntimeError("No klines returned for since-open (symbol may be invalid/delisted or no data from since date)")
    # [openTime, open, high, low, close, volume, closeTime, quoteVolume, ...]
    return float(kl[0][1])

def max_high_since(symbol: str, since: str) -> float:
    start = parse_date_ms(since)
    cur = start
    high = 0.0
    pages = 0

    while pages < MAX_PAGES:
        pages += 1
        kl = fetch_klines(symbol, PAGE_LIMIT, cur)
        if not kl:
            break

        for k in kl:
            h = float(k[2])
            if h > high:
                high = h

        # advance
        cur = int(kl[-1][0]) + 1

        if len(kl) < PAGE_LIMIT:
            break

    if high <= 0.0:
        raise RuntimeError("Could not compute max high (no data returned after paging)")
    return high


# ============================================================
# MAIN
# ============================================================
def main():
    if os.path.exists(TOGGLE_FILE):
        print(f"[SKIP] Toggle exists: {TOGGLE_FILE} (already alerted).")
        return

    print(f"=== RUN {utc_now()} ===")

    cfg = read_s3_config()
    alerts = parse_alerts(cfg)

    hits = []
    errors = []

    for a in alerts:
        try:
            so = since_open(a.symbol, a.since)
            target = so * (1 + TARGET_PCT / 100.0)
            mh = max_high_since(a.symbol, a.since)
            live = get_price(a.symbol)

            hit = mh >= target
            print(f"{a.symbol} since_open={so:.8f} target={target:.8f} max_high={mh:.8f} hit={hit}")

            if hit:
                gain = ((mh - so) / so) * 100.0 if so else 0.0
                hits.append(
                    f"{a.symbol} since={a.since} since_open={so:.8f} target={target:.8f} max_high={mh:.8f} ({gain:.2f}%) live={live:.8f}"
                )

        except Exception as e:
            msg = f"{a.symbol} (since {a.since}) ERROR: {e}"
            errors.append(msg)
            print(msg)

    if not hits:
        print("No hits.")
        if errors:
            print("\nErrors (non-fatal):")
            for e in errors[:50]:
                print(" -", e)
        return

    # Compose email
    body_lines = [
        f"Triggered at: {utc_now()} UTC",
        f"Rule: max HIGH since since-date >= since-open × 1.07 (+{TARGET_PCT:.2f}%)",
        f"Since-open = OPEN of first {INTERVAL} candle at/after since-date (UTC midnight).",
        "",
        "HITS:"
    ]
    body_lines.extend([f"- {h}" for h in hits])

    if errors:
        body_lines.append("")
        body_lines.append("ERRORS (non-fatal):")
        body_lines.extend([f"- {e}" for e in errors[:50]])

    subject = f"Crypto Alert: {len(hits)} symbol(s) hit +{TARGET_PCT:.0f}%"
    body = "\n".join(body_lines) + "\n"

    # Send email + set toggle
    send_email(subject, body)
    with open(TOGGLE_FILE, "w", encoding="utf-8") as f:
        f.write(f"ALERT_SENT_UTC={utc_now()}\n")
        f.write(f"CONFIG={S3_URI}\n")
        f.write(f"HITS_COUNT={len(hits)}\n")

    print(f"[DONE] Email sent. Toggle written: {TOGGLE_FILE}")

if __name__ == "__main__":
    main()
