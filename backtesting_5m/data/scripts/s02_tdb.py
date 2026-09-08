"""The 5 m book from the v1 recorder SQLite -- a second, independent capture.

Two databases, contiguous by `id`: the 81.5 GB archive on polydata
(2026-04-13 -> 09-05, scanned there by `_box_extract_book.py` over the era's id
range only) and the live one on this laptop (09-05 -> now). Both write
`exch_bid`/`exch_ask` as UP-token probabilities at 10 Hz.

    python s02_tdb.py            # scan the box, then merge with the live DB
    python s02_tdb.py --reuse    # box parts already in /tmp

⚠ `ticks` CARRIES NO MARKET ID. The v1 quoter records one namespace,
`poly_btc_5m`, and which market that was at a given moment is implicit. 5 m
windows are on exact 300 s boundaries, so the market is the window containing
the observation -- but that inference is only as good as the quoter's own
switching, and a quote published a moment after a window rolls may still be
the OLD market's book. Rather than assume either way, every row keeps
`t_ms` and the archive is used to measure the disagreement per bucket
(`results/gate_attribution.tsv`); the panel build then trims whatever edge the
measurement says is contaminated.
"""
import argparse
import gzip
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd

import common

HOST = "root@polydata"
DB = "/var/lib/trading/legacy_v1/trading_v1_20260413_20260905.db"
BOX_SCRIPT = "/tmp/_box_extract_book.py"
PART = "/tmp/book5m_w{}.csv.gz"
N_WORKER = 4
#: First `id` at or after ERA_START, found by binary search on `ts`.
ERA_ID_LO, ID_HI = 111_580_683, 130_580_056


def launch():
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(["scp", "-q", os.path.join(here, "_box_extract_book.py"),
                    f"{HOST}:{BOX_SCRIPT}"], check=True)
    span = (ID_HI - ERA_ID_LO + 1) // N_WORKER
    for i in range(N_WORKER):
        a = ERA_ID_LO + i * span
        b = ID_HI if i == N_WORKER - 1 else ERA_ID_LO + (i + 1) * span - 1
        subprocess.run(["ssh", HOST,
                        f"nohup python3 {BOX_SCRIPT} {DB} {a} {b} "
                        f"{PART.format(i)} > /tmp/book5m_w{i}.log 2>&1 &"],
                       check=True)
        print(f"  worker {i}: id {a:,}..{b:,}", flush=True)
    subprocess.run(["ssh", HOST,
                    "while pgrep -f _box_extract_book >/dev/null; do sleep 15; done"],
                   check=True)


def pull_box():
    parts = []
    for i in range(N_WORKER):
        local = os.path.join(common.DATA, f"book5m_w{i}.csv.gz")
        subprocess.run(["scp", "-q", f"{HOST}:{PART.format(i)}", local],
                       check=True)
        with gzip.open(local, "rt") as fh:
            parts.append(pd.read_csv(fh))
        print(f"  part {i}: {len(parts[-1]):,} rows", flush=True)
    return pd.concat(parts, ignore_index=True)


def read_live():
    con = sqlite3.connect(f"file:{common.TDB_LIVE}?mode=ro", uri=True)
    rows = con.execute(
        "select ts, exch_bid, exch_ask from ticks where ns = 'poly_btc_5m' "
        "and exch_bid is not null and exch_ask is not null and ts >= ?",
        (common.ERA_START,)).fetchall()
    return pd.DataFrame(rows, columns=["ts", "bid", "ask"])


def attribute(df, gamma):
    """Give each quote the market whose 300 s window contains it."""
    df = df[df.ts >= common.ERA_START].copy()
    df["open_ts"] = common.market_open(df.ts.values)
    m = gamma[["open_ts", "market_id"]].drop_duplicates("open_ts")
    out = df.merge(m, on="open_ts", how="inner")
    lost = len(df) - len(out)
    print(f"  {lost:,} quotes fell in a window gamma lists no market for "
          f"({lost/max(len(df),1):.3%})", flush=True)
    return out.drop(columns=["open_ts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="box parts already in /tmp; skip the scan")
    a = ap.parse_args()
    os.makedirs(common.DATA, exist_ok=True)

    gamma = pd.read_parquet(os.path.join(common.DATA, "gamma_5m.parquet"))

    if not a.reuse:
        launch()
    box = pull_box()
    print(f"tdb_old: {len(box):,} raw quotes", flush=True)
    common.write_quotes("tdb_old", attribute(box, gamma))

    live = read_live()
    print(f"tdb_live: {len(live):,} raw quotes", flush=True)
    common.write_quotes("tdb_live", attribute(live, gamma))


if __name__ == "__main__":
    sys.exit(main())
