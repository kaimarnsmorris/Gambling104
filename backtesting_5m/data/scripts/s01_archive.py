"""The 5 m book from polydata's quoter archive, read over Z: with DuckDB.

The archive republishes the quoter's own state at 10.0 Hz per market -- exactly
the panel's grid, measured at dt median 0.100 s over a full market -- so no
resampling happens here.

Which markets are the 5 m book comes from `s00_gamma.py`, not from this file.
⚠ Do not be tempted to classify by the archive's own columns. `tf` and `kind`
do not exist before 2026-08-17, and recovering the timeframe from each
market's observed SPAN -- the obvious fallback, and what the older
`export_books_polydata.py` does -- agrees with `tf` on only 66.6 % of the
markets that carry one: a partially captured 15 m market has a 5 m-sized span
and impersonates one. Coverage is the thing being measured, so it cannot also
be the classifier.

Both `kind='signal'` and `kind='tick'` rows carry `exch_bid`/`exch_ask` and
both are real observations of the book, so both are read. `kind` only governs
which OTHER columns a row carries, none of which are wanted here.
"""
import glob
import os
import sys
import time

import duckdb
import pandas as pd

import common


def day_dirs():
    ds = sorted(glob.glob(os.path.join(common.ARCHIVE, "date=*")))
    if not ds:
        raise SystemExit(f"no archive days under {common.ARCHIVE} -- is Z: mounted?")
    first = pd.to_datetime(common.ERA_START, unit="s").strftime("%Y-%m-%d")
    return [d for d in ds if os.path.basename(d)[5:] >= first]


def main():
    gpath = os.path.join(common.DATA, "gamma_5m.parquet")
    if not os.path.exists(gpath):
        raise SystemExit("run s00_gamma.py first -- it says which markets exist")
    ids = set(pd.read_parquet(gpath).market_id)
    print(f"archive: {len(ids):,} target markets", flush=True)

    con = duckdb.connect(config={"memory_limit": "3GB"})
    con.register("want", pd.DataFrame({"market_id": sorted(ids)}))

    parts, t0 = [], time.time()
    for d in day_dirs():
        g = os.path.join(d, "*.parquet").replace("\\", "/")
        q = (f"select b.market_id, b.ts, b.exch_bid bid, b.exch_ask ask "
             f"from read_parquet('{g}') b join want w using (market_id) "
             f"where b.exch_bid is not null and b.exch_ask is not null")
        df = con.execute(q).df()
        if len(df):
            parts.append(df)
        print(f"  {os.path.basename(d)}: {len(df):,} quotes "
              f"({time.time()-t0:.0f}s)", flush=True)

    common.write_quotes("archive", pd.concat(parts, ignore_index=True))


if __name__ == "__main__":
    sys.exit(main())
