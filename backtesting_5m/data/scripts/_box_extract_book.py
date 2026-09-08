"""Run ON polydata. Emit 5 m book L1 from the archived v1 recorder SQLite.

Stock python3 only on that box -- no duckdb, no pyarrow -- and the DB is
81.5 GB, so this streams gzipped CSV over an `id` range.

Unlike the Chainlink extractor this keeps EVERY sample, not just change-points.
The book is a state a backtest reads at a moment, so a repeated observation of
an unchanged book is a confirmation the book was still there, which is real
information. Only unobserved buckets are absent.
"""
import gzip
import sqlite3
import sys
import time


def main(db, lo, hi, out):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.execute("pragma cache_size=-200000")
    q = (f"select ts, exch_bid, exch_ask from ticks "
         f"where id between {lo} and {hi} and ns = 'poly_btc_5m' "
         f"and exch_bid is not null and exch_ask is not null")
    n, t0 = 0, time.time()
    with gzip.open(out, "wt", compresslevel=6) as fh:
        fh.write("ts,bid,ask\n")
        for ts, b, a in con.execute(q):
            n += 1
            fh.write(f"{ts!r},{b!r},{a!r}\n")
    print(f"{out}: {n:,} quotes in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
