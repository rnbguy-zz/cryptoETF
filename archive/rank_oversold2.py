#!/usr/bin/env python3
"""
rank_oversold_jasmy_virtual_class.py

Scans an input directory for files named like:
  YYYY-MM-DD_oversold.txt

Only includes files whose date is within the last N days (UTC window).

Parses lines like:
  2025-12-03   2025-12-09     64.031204    44.808087 19.223117   JASMYUSDT

Then, PER FILE (no cross-file comparison), computes:
- dist = |CurrentRSI - 45|
- drop_pct  = percentile rank of Drop within that file (highest drop => pct close to 1.0)
- prev_pct  = percentile rank of PreviousRSI within that file (highest prev => pct close to 1.0)

"JASMY/VIRTUAL class" gates (tunable):
- CurrentRSI <= 45.0
- dist <= 0.55
- drop_pct >= 0.90
- prev_pct >= 0.80

Score (0..100):
- reset_score (0..40): closer to 45 is better (0 dist => 40; dist==0.55 => 0)
- drop_score  (0..35): 35 * drop_pct
- mom_score   (0..25): 25 * prev_pct
TOTAL = sum

Prints only rows with TOTAL >= --min-score (default 90).

Usage:
  python3.9 rank_oversold_jasmy_virtual_class.py /path/to/oversold_analysis --days 7 --min-score 90

Example:
  python3.9 rank_oversold_jasmy_virtual_class.py oversold_analysis
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Row:
    prev_date: str
    cur_date: str
    prev_rsi: float
    cur_rsi: float
    drop: float
    symbol: str


@dataclass
class Scored:
    row: Row
    file_name: str
    file_date: str
    dist: float
    drop_pct: float
    prev_pct: float
    reset_score: float
    drop_score: float
    mom_score: float
    total: float


# -----------------------------
# Helpers
# -----------------------------
DATE_IN_NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_oversold\.txt$")

# Matches:
# 2025-12-03   2025-12-09     64.031204    44.808087 19.223117   JASMYUSDT
LINE_RE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2})\s+(\d{4}-\d{2}-\d{2})\s+"
    r"([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([A-Za-z0-9_]+)\s*$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", nargs="?", default="inputfiles", help="Directory containing *_oversold.txt files")
    p.add_argument("--days", type=int, default=7, help="Lookback window in days (UTC)")
    p.add_argument("--min-score", type=float, default=90.0, help="Print only symbols with TOTAL >= this")
    # Class gates (tune here)
    p.add_argument("--target-rsi", type=float, default=45.0, help="RSI target center (default 45)")
    p.add_argument("--max-dist", type=float, default=0.55, help="Max |CurrentRSI-target| to be considered class")
    p.add_argument("--drop-pct-min", type=float, default=0.90, help="Min drop percentile within file (0..1)")
    p.add_argument("--prev-pct-min", type=float, default=0.80, help="Min prevRSI percentile within file (0..1)")
    p.add_argument("--require-below-target", action="store_true", default=False,
                   help="Require CurrentRSI <= target (default ON)")
    p.add_argument("--allow-above-target", action="store_true",
                   help="If set, disables the CurrentRSI <= target requirement")
    p.add_argument("--debug", action="store_true", help="Print file window + parsing stats")
    return p.parse_args()


def utc_today_date() -> datetime:
    return datetime.now(timezone.utc)


def file_date_from_name(filename: str) -> Optional[str]:
    m = DATE_IN_NAME_RE.match(filename)
    return m.group(1) if m else None


def list_files_in_window(input_dir: str, days: int) -> Tuple[List[str], str, str]:
    now = utc_today_date()
    end = now.date()
    start = (now - timedelta(days=days)).date()
    start_s, end_s = start.isoformat(), end.isoformat()

    files = []
    for name in os.listdir(input_dir):
        d = file_date_from_name(name)
        if not d:
            continue
        if start_s <= d <= end_s:
            files.append(name)

    files.sort()
    return files, start_s, end_s


def parse_rows_from_file(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            prev_date, cur_date = m.group(1), m.group(2)
            prev_rsi, cur_rsi, drop = float(m.group(3)), float(m.group(4)), float(m.group(5))
            symbol = m.group(6).strip()
            rows.append(Row(prev_date, cur_date, prev_rsi, cur_rsi, drop, symbol))
    return rows


def percentile_desc(values: List[float]) -> List[float]:
    """
    Descending percentile rank: largest => 1.0, smallest => 0.0.
    Tie-safe (averaged ranks).
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # argsort ascending
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for r, idx in enumerate(order):
        ranks[idx] = float(r)

    # average ties on the sorted order
    sorted_vals = [values[i] for i in order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = sum(ranks[order[k]] for k in range(i, j + 1)) / (j - i + 1)
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1

    # scale 0..1 ascending, then we want descending in the sense "bigger is better"
    # BUT: we already ranked ascending (smallest rank 0). Convert to 0..1 then keep it
    # as "percentile of being large" by using ascending/(n-1) because larger values have larger ranks.
    pct = [r / (n - 1) for r in ranks]
    return pct


def score_components(dist: float, max_dist: float, drop_pct: float, prev_pct: float) -> Tuple[float, float, float, float]:
    # reset_score 0..40
    reset = max(0.0, 40.0 * (1.0 - (dist / max_dist if max_dist > 0 else 1.0)))
    # drop_score 0..35
    drop = 35.0 * drop_pct
    # mom_score 0..25
    mom = 25.0 * prev_pct
    total = reset + drop + mom
    return reset, drop, mom, total


def scan_file(
    file_name: str,
    file_date: str,
    rows: List[Row],
    target: float,
    max_dist: float,
    drop_pct_min: float,
    prev_pct_min: float,
    require_below_target: bool,
) -> List[Scored]:
    if not rows:
        return []

    prevs = [r.prev_rsi for r in rows]
    curs = [r.cur_rsi for r in rows]
    drops = [r.drop for r in rows]

    drop_pct = percentile_desc(drops)
    prev_pct = percentile_desc(prevs)

    out: List[Scored] = []
    for i, r in enumerate(rows):
        dist = abs(r.cur_rsi - target)

        if require_below_target and not (r.cur_rsi <= target):
            continue
        if dist > max_dist:
            continue
        if drop_pct[i] < drop_pct_min:
            continue
        if prev_pct[i] < prev_pct_min:
            continue

        reset_s, drop_s, mom_s, total = score_components(dist, max_dist, drop_pct[i], prev_pct[i])
        out.append(
            Scored(
                row=r,
                file_name=file_name,
                file_date=file_date,
                dist=dist,
                drop_pct=drop_pct[i],
                prev_pct=prev_pct[i],
                reset_score=reset_s,
                drop_score=drop_s,
                mom_score=mom_s,
                total=total,
            )
        )

    out.sort(key=lambda x: x.total, reverse=True)
    return out


def fmt_pct(x: float) -> str:
    return f"{x:.2f}"


def main() -> int:
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"ERROR: input_dir not found: {input_dir}")
        return 2

    now = utc_today_date()
    files, start_s, end_s = list_files_in_window(input_dir, args.days)

    # gates
    target = args.target_rsi
    max_dist = args.max_dist
    drop_pct_min = args.drop_pct_min
    prev_pct_min = args.prev_pct_min
    require_below = (not args.allow_above_target) and args.require_below_target

    if args.debug:
        print(f"Today(UTC)={now.strftime('%Y-%m-%d')}  Window={start_s}..{end_s}")
        print(f"Input dir: {input_dir}")
        print(f"Files in window: {len(files)}")
        for f in files:
            print(f"  - {f}")
        print()

        print("Class gates:")
        print(f"  target_rsi={target}")
        print(f"  require_below_target={require_below}")
        print(f"  max_dist={max_dist}")
        print(f"  drop_pct_min={drop_pct_min}")
        print(f"  prev_pct_min={prev_pct_min}")
        print()

    winners: List[Scored] = []

    for fname in files:
        fdate = file_date_from_name(fname) or "????-??-??"
        path = os.path.join(input_dir, fname)
        rows = parse_rows_from_file(path)
        scored = scan_file(
            file_name=fname,
            file_date=fdate,
            rows=rows,
            target=target,
            max_dist=max_dist,
            drop_pct_min=drop_pct_min,
            prev_pct_min=prev_pct_min,
            require_below_target=require_below,
        )
        for s in scored:
            if s.total >= args.min_score:
                winners.append(s)

    if not winners:
        print(f"No symbols scored >= {args.min_score:g} in the last {args.days} days.")
        return 0

    winners.sort(key=lambda x: (x.total, x.file_date, x.row.symbol), reverse=True)

    print(f"Winners (TOTAL >= {args.min_score:g}):\n")
    for s in winners:
        r = s.row
        print(
            f"{r.symbol:<16} TOTAL={s.total:6.2f}  "
            f"Reset={s.reset_score:5.1f}/40  Drop={s.drop_score:5.1f}/35  Mom={s.mom_score:5.1f}/25  "
            f"|RSI-{target:g}|={s.dist:5.3f}  "
            f"Drop={r.drop:6.2f} (pct={fmt_pct(s.drop_pct)})  "
            f"PrevRSI={r.prev_rsi:6.2f} (pct={fmt_pct(s.prev_pct)})  "
            f"CurRSI={r.cur_rsi:6.2f}  "
            f"Dates {r.prev_date}→{r.cur_date}  "
            f"File={s.file_name}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

