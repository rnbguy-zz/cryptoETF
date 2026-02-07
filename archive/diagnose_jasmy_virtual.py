#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

TARGETS = {"JASMYUSDT", "VIRTUALUSDT"}

def parse_filename_date(name: str):
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()

def read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("small") or line.lower().startswith("previous"):
                continue
            parts = line.split()
            if len(parts) < 6 or not parts[0].startswith("20"):
                continue
            try:
                prev_date, curr_date = parts[0], parts[1]
                prev_rsi = float(parts[2])
                curr_rsi = float(parts[3])
                drop = float(parts[4])
                sym = parts[5].upper()
            except Exception:
                continue
            rows.append((prev_date, curr_date, prev_rsi, curr_rsi, drop, sym))
    return rows

def percentile_desc(arr: np.ndarray):
    # largest value -> 1.0
    n = len(arr)
    if n == 1:
        return np.array([1.0])
    order = np.argsort(arr)  # ascending
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)

    # average ties
    s = arr[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j+1] == s[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j+1]].mean()
            ranks[order[i:j+1]] = avg
        i = j + 1

    pct = ranks / (n - 1)        # smallest -> 0, largest -> 1
    return pct

def rank_min(arr: np.ndarray):
    uniq = np.unique(arr)
    mapping = {v: i+1 for i, v in enumerate(np.sort(uniq))}
    return np.array([mapping[v] for v in arr], dtype=int)

def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("oversold_analysis")
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix == ".txt" and parse_filename_date(p.name)])

    if not files:
        print("No dated oversold txt files found.")
        return 1

    for p in files:
        rows = read_rows(p)
        if not rows:
            continue

        prev_rsi = np.array([r[2] for r in rows], dtype=float)
        curr_rsi = np.array([r[3] for r in rows], dtype=float)
        drops    = np.array([r[4] for r in rows], dtype=float)
        syms     = [r[5] for r in rows]

        dist = np.abs(curr_rsi - 45.0)
        dist_rank = rank_min(dist)  # 1 = closest to 45
        drop_pct  = percentile_desc(drops)
        prev_pct  = percentile_desc(prev_rsi)

        # "JASMY/VIRTUAL class" candidate count (within-file)
        # tweak these thresholds later after you see diagnostics
        big_drop = np.quantile(drops, 0.85)       # top 15% drops
        hi_prev  = np.median(prev_rsi)            # above median
        near45   = 0.25                           # within 0.25 of 45
        below45  = True

        class_mask = (dist <= near45) & (drops >= big_drop) & (prev_rsi >= hi_prev)
        if below45:
            class_mask = class_mask & (curr_rsi <= 45.0)

        class_count = int(class_mask.sum())

        # Only print files that actually contain our targets
        found = [t for t in TARGETS if t in syms]
        if not found:
            continue

        print(f"\n=== {p.name}  (rows={len(rows)})  class_count={class_count} "
              f"(near45<=±{near45}, drop>=p85={big_drop:.2f}, prev>=median={hi_prev:.2f}, below45={below45}) ===")

        for t in found:
            i = syms.index(t)
            print(
                f"{t:12} prev={prev_rsi[i]:6.2f} (pct={prev_pct[i]:.2f})  "
                f"cur={curr_rsi[i]:6.2f} |cur-45|={dist[i]:.3f} (rank={dist_rank[i]})  "
                f"drop={drops[i]:6.2f} (pct={drop_pct[i]:.2f})"
            )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

