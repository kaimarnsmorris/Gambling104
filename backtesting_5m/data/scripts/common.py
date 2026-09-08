"""Shared vocabulary for the BTC up/down 5 m book reconciliation.

Every capture is reduced to the same intermediate -- a QUOTES table, one row
per observation of the UP token's L1:

    src        which capture saw it
    market_id  the CLOB token id of the UP token
    ts         observation time, epoch seconds, on that capture's host clock
    bid, ask   the UP token's best bid / offer, in probability (0..1)

The panel is then one row per (market, 100 ms bucket) in which at least one
capture actually observed the book. Nothing is carried forward.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA = os.path.join(ROOT, "intermediate")
RESULTS = os.path.join(ROOT, "results")
PANEL = os.path.join(ROOT, "book_5m_100ms.parquet")
STRIKES = os.path.join(ROOT, "strikes_5m.parquet")

ARCHIVE = r"Z:\parquet\stream_poly_crypto_updown\base_asset=BTC"
TDB_LIVE = r"C:\Users\kaima\trading_data\trading.db"
GAMBLING102 = r"C:\Users\kaima\OneDrive\Documents\GitHub\Gambling102"

#: The 5 m book moved from the 30 s TWAP to the 60 s TWAP at the market opening
#: 2026-08-14 00:00:00 UTC -- established by binary-searching gamma's
#: `cryptoMarketConfig.twapLookbackSeconds`, which reads 30 on the market
#: opening 23:55:00 and 60 on this one. Markets before it settle on a
#: different feed and are a different instrument.
ERA_START = 1786665600

H = 300                      # the 5 m window, seconds
BUCKET_MS = 100              # the panel's grid
N_BUCKET = H * 1000 // BUCKET_MS

#: Which capture wins a bucket both saw. The archive is the quoter's own
#: republished state and carries the market id; the trading.dbs are a second
#: host watching the same CLOB and must have their market inferred.
PRIORITY = ["archive", "tdb_old", "tdb_live"]

QUOTES = {s: os.path.join(DATA, f"quotes_{s}.parquet") for s in PRIORITY}

#: A market is kept only if the capture saw both ends of its window and most
#: of the middle. Truncated markets are the ones this filter exists to remove.
END_S = 10
MIN_COVERAGE = 0.90

#: `ticks` carries no market id, so a trading.db quote is attributed to the
#: window containing it -- and the v1 quoter keeps publishing the PREVIOUS
#: market's book for a few seconds after a roll, because it emits no signal
#: until the new strike resolves. Measured against the archive, disagreement
#: over 5 c runs 100 % in the first second and decays to the baseline by 6 s.
#: So trading.db quotes before this point are attribution-unsafe and dropped;
#: the archive, which carries the id, is unaffected.
TDB_TRIM_MS = 6000


def market_open(ts):
    """The 5 m window containing `ts`. Windows are on exact 300 s boundaries."""
    return (np.asarray(ts, dtype="float64") // H * H).astype("int64")


def write_quotes(src, df):
    """Persist one capture's L1 observations, sorted and sanity-checked."""
    df = df.dropna(subset=["bid", "ask"])
    # A crossed or out-of-range book is not a quote, it is a decode error.
    bad = (df.bid > df.ask) | (df.bid < 0) | (df.ask > 1)
    if bad.any():
        print(f"  {src}: dropping {int(bad.sum()):,} crossed/out-of-range rows",
              flush=True)
        df = df[~bad]
    df = df.sort_values(["market_id", "ts"], kind="stable").reset_index(drop=True)
    os.makedirs(DATA, exist_ok=True)
    df.to_parquet(QUOTES[src], index=False)
    print(f"{src}: {len(df):,} quotes, {df.market_id.nunique():,} markets, "
          f"{iso(df.ts.min())} -> {iso(df.ts.max())}", flush=True)
    return df


def iso(ts):
    return pd.to_datetime(float(ts), unit="s").strftime("%Y-%m-%d %H:%M:%S")
