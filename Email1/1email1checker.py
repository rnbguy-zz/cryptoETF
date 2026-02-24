#!/usr/bin/env python3.9
import os
import sys
import imaplib
import email
import datetime
import re
import quopri
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime
from email.message import Message
from typing import Optional, Tuple, List
from html import unescape

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
SUBJECT_TITLE = "Oversold Analysis Alert1"
MAILBOX = '"[Gmail]/Sent Mail"'
OUTPUT_FILE = "signals1.txt"
# ---------- Helpers for decoding / cleaning ----------

def write_out(lines):
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")

def best_decode(s: Optional[str]) -> str:
    if s is None:
        return ""
    try:
        return str(make_header(decode_header(s)))
    except Exception:
        return s

def qp_maybe_decode(s: str) -> str:
    """
    If text has quoted-printable artifacts (e.g., =3D, soft line breaks =\n),
    force-decode it. Otherwise return as-is.
    """
    if s is None:
        return ""
    # Heuristics: presence of =3D, =\r\n, or lines ending with '='
    if re.search(r"(=3D|=\r?\n|\r?\n=)", s):
        try:
            return quopri.decodestring(s).decode("utf-8", errors="replace")
        except Exception:
            # Try latin-1 fallback
            return quopri.decodestring(s).decode("latin-1", errors="replace")
    return s

def strip_tags_and_entities(text: str) -> str:
    """
    Remove all HTML tags and decode HTML entities. Basic whitespace cleanup.
    """
    if not text:
        return ""
    # Remove tags (your input uses &lt;...&gt; literal; handle both real and escaped)
    text = re.sub(r"(?is)<[^>]+>", "", text)          # real HTML tags
    text = re.sub(r"(?is)&lt;[^&gt;]+&gt;", "", text) # escaped tags
    # Entities
    text = unescape(text)
    # Normalize nbsp
    text = text.replace("\xa0", " ")
    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim per line
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()

def html_to_text(html: str) -> str:
    """
    Convert HTML to readable plain text using stdlib only.
    - Adds newlines for block-level elements
    - Converts <br> to newline; </td>/</th> to tabs; </tr> to newline
    - Removes script/style
    """
    if not html:
        return ""
    # Remove script/style (both real and escaped)
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", html)
    html = re.sub(r"(?is)&lt;(script|style).*?&gt;.*?&lt;/\1&gt;", "", html)

    # Common structural breaks
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)&lt;br\s*/?&gt;", "\n", html)

    for tag in [
        "p","div","section","article","header","footer","aside",
        "h1","h2","h3","h4","h5","h6",
        "ul","ol","li",
        "table","thead","tbody","tfoot","tr","blockquote","pre","hr"
    ]:
        html = re.sub(fr"(?i)</{tag}\s*>", "\n", html)
        html = re.sub(fr"(?i)&lt;/{tag}\s*&gt;", "\n", html)

    # Table cells -> tab
    html = re.sub(r"(?i)</t[dh]\s*>", "\t", html)
    html = re.sub(r"(?i)&lt;/t[dh]\s*&gt;", "\t", html)

    # Strip tags and entities, then cleanup
    return strip_tags_and_entities(html)

def looks_like_html(text: str) -> bool:
    """
    Detect if text is effectively HTML or HTML-in-text.
    """
    if not text:
        return False
    low = text.lower()
    if "<html" in low or "<body" in low or "&lt;html" in low or "&lt;body" in low:
        return True
    if re.search(r"</?(div|span|p|br|table|tr|td|th|h[1-6]|ul|ol|li|font)\b", text, re.I):
        return True
    if re.search(r"&[a-zA-Z]+;", text):
        return True
    if re.search(r"&amp;[a-zA-Z]+;", text):
        return True
    return False

def postprocess_readable(text: str) -> str:
    """
    Final pass to fix common artifacts:
    - quoted-printable leftovers (=3D, soft wraps)
    - stray <br>, &nbsp;, etc.
    - collapse whitespace
    """
    if not text:
        return ""
    # Try quoted-printable force-decode if artifacts present
    text2 = text
    # If still contains apparent HTML, convert to text
    if looks_like_html(text2):
        text2 = html_to_text(text2)
    else:
        # Even if deemed plain, unescape entities and normalize
        text2 = strip_tags_and_entities(text2)

    # Final tidy: collapse excessive blank lines
    text2 = re.sub(r"\n{3,}", "\n\n", text2).strip()
    text2 = strip_control_chars(text2)
    return text2

# ---------- NEW: Cutoff after the last BUY line ----------

def keep_until_last_buy(text: str) -> str:
    """
    Keep content up to and including the last line that starts with 'BUY'.
    - Case-insensitive.
    - Allows leading spaces and non-breaking spaces before BUY.
    - 'BUY' may be followed by word boundary (space, tab, punctuation).
    If no BUY line is found, return the original text.
    """
    if not text:
        return text

    # Normalize non-breaking spaces to regular spaces for matching
    norm = text.replace("\xa0", " ")
    lines = norm.splitlines()

    last_idx = -1
    buy_re = re.compile(r'^\s*BUY\b', re.IGNORECASE)
    for i, ln in enumerate(lines):
        if buy_re.match(ln):
            last_idx = i

    if last_idx == -1:
        # No BUY line found; return as-is
        return text.strip()

    # Slice up to and including the last BUY line (preserve original spacing in those lines)
    original_lines = text.splitlines()
    clipped = original_lines[: last_idx + 1]
    out = "\n".join(clipped).rstrip()
    return out

# ---------- IMAP helpers (robust parsing) ----------

def _iter_fetch_bytes(resp_obj) -> List[bytes]:
    """
    From imaplib.fetch() response, collect all bytes parts safely.
    """
    if not resp_obj:
        return []
    out = []
    for part in resp_obj:
        if isinstance(part, tuple):
            if isinstance(part[0], (bytes, bytearray)):
                out.append(bytes(part[0]))
            if len(part) > 1 and isinstance(part[1], (bytes, bytearray)):
                out.append(bytes(part[1]))
        elif isinstance(part, (bytes, bytearray)):
            out.append(bytes(part))
    return out

def fetch_internaldate(M: imaplib.IMAP4_SSL, msgid: bytes) -> Optional[datetime.datetime]:
    status, resp = M.fetch(msgid, "(INTERNALDATE)")
    if status != "OK":
        return None
    lines = _iter_fetch_bytes(resp)
    for line in lines:
        try:
            text = line.decode("utf-8", errors="ignore")
        except Exception:
            continue
        if "INTERNALDATE" in text:
            q1 = text.find('"')
            q2 = text.find('"', q1 + 1)
            if q1 != -1 and q2 != -1:
                idate_str = text[q1+1:q2]
                try:
                    return parsedate_to_datetime(idate_str)
                except Exception:
                    return None
    return None

def fetch_headers(M: imaplib.IMAP4_SSL, msgid: bytes) -> Tuple[str, str, str]:
    status, data = M.fetch(msgid, "(BODY.PEEK[HEADER.FIELDS (SUBJECT FROM DATE)])")
    if status != "OK":
        return ("", "", "")
    raw_headers = b"".join(_iter_fetch_bytes(data))
    msg = email.message_from_bytes(raw_headers)
    subj = best_decode(msg.get("Subject"))
    from_ = best_decode(msg.get("From"))
    date_ = best_decode(msg.get("Date"))
    return (subj, from_, date_)

def fetch_full_message(M: imaplib.IMAP4_SSL, msgid: bytes) -> Message:
    status, msg_data = M.fetch(msgid, "(RFC822)")
    if status != "OK" or not msg_data:
        raise RuntimeError("Failed to fetch the email body")

    chunks = []
    for part in msg_data:
        if isinstance(part, tuple) and len(part) > 1 and isinstance(part[1], (bytes, bytearray)):
            chunks.append(bytes(part[1]))  # ONLY the message bytes (not the metadata)
    raw = b"".join(chunks)
    return email.message_from_bytes(raw)

# ---------- Body extraction ----------

def strip_control_chars(s: str) -> str:
    # remove ASCII control chars except \n \r \t
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)


def extract_text_body(msg: Message) -> str:
    """
    Prefer text/plain (cleaned), otherwise convert text/html to text.
    Also handles cases where senders put HTML into the text/plain part or
    leave quoted-printable artifacts.
    """

    def decode_part(part: Message) -> str:
        payload = part.get_payload(decode=True)
        if payload is None:
            # Sometimes get_payload(decode=True) returns None; fallback to raw payload
            raw = part.get_payload()
            return raw if isinstance(raw, str) else ""

        # Try declared charset first
        charset = (part.get_content_charset() or "").strip().lower()
        candidates = []
        if charset:
            candidates.append(charset)

        # Common fallbacks (Gmail + marketing emails often hit these)
        candidates += ["utf-8", "windows-1252", "latin-1"]

        for cs in candidates:
            try:
                return payload.decode(cs, errors="replace")
            except Exception:
                continue

        # last resort
        return payload.decode("utf-8", errors="replace")

    if msg.is_multipart():
        plain_part = None
        html_part = None
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            if ctype == "text/plain" and plain_part is None:
                plain_part = part
            elif ctype == "text/html" and html_part is None:
                html_part = part

        if plain_part is not None:
            txt = decode_part(plain_part)
            # Clean quoted-printable artifacts and embedded HTML/entities
            return postprocess_readable(txt)

        if html_part is not None:
            html = decode_part(html_part)
            return postprocess_readable(html)

        return "[No displayable body found]"
    else:
        txt = decode_part(msg)
        return postprocess_readable(txt)

# ---------- Main ----------

def main():
    username = 'minamoussa903@gmail.com'#os.environ.get("GMAIL_USER")
    app_pass = 'qhvi syra bbad gylu'#os.environ.get("GMAIL_APP_PASS")
    if not username or not app_pass:
        print("ERROR: Set GMAIL_USER and GMAIL_APP_PASS env vars.", file=sys.stderr)
        sys.exit(2)

    try:
        M = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        M.login(username, app_pass)
        status, _ = M.select(MAILBOX)
        if status != "OK":
            print(f"ERROR: Unable to select mailbox {MAILBOX}", file=sys.stderr)
            sys.exit(1)

        # Find last 24h with the subject; robust for TZ
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)

        after_str = today.strftime("%Y/%m/%d")
        before_str = tomorrow.strftime("%Y/%m/%d")

        gm_query = (
            'X-GM-RAW '
            f'"after:{after_str} before:{before_str} subject:\\"{SUBJECT_TITLE}\\""'
        )

        status, data = M.search(None, gm_query)
        if status != "OK":
            print("ERROR: IMAP search failed", file=sys.stderr)
            sys.exit(1)

        ids = data[0].split()
        if not ids:
            print(f"No email found in the last 24h with subject: {SUBJECT_TITLE}")
            sys.exit(0)

        # Pick most recent by Date header, fallback to INTERNALDATE
        dated_ids: List[Tuple[datetime.datetime, bytes]] = []
        for msgid in ids:
            subj, from_, date_hdr = fetch_headers(M, msgid)
            dt = None
            if date_hdr:
                try:
                    dt = parsedate_to_datetime(date_hdr)
                except Exception:
                    dt = None
            if dt is None:
                dt = fetch_internaldate(M, msgid)
            if dt is None:
                dt = datetime.datetime.min.replace(tzinfo=None)
            dated_ids.append((dt, msgid))

        dated_ids.sort(key=lambda x: x[0])
        latest_id = dated_ids[-1][1]

        # Fetch and print nicely
        msg = fetch_full_message(M, latest_id)
        subj = best_decode(msg.get("Subject"))
        from_ = best_decode(msg.get("From"))
        date_ = best_decode(msg.get("Date"))
        body = extract_text_body(msg)

        # >>> Cut output at the last BUY line <<<
        body = keep_until_last_buy(body)

        out_lines = []
        out_lines.append(body)
        write_out(out_lines)
        print(body)

        M.close()
        M.logout()

    except imaplib.IMAP4.error as e:
        print(f"IMAP error: {e}", file=sys.stderr)
        print("Tips:", file=sys.stderr)
        print(" - Use a Gmail App Password (not your normal password).", file=sys.stderr)
        print(" - Ensure IMAP is enabled in Gmail settings.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

