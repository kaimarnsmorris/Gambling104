"""Where the harness reads and writes. Mirrors data/scripts/common.py."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, os.pardir))

DATA = os.path.join(PROJECT, "data")
PANEL = os.path.join(DATA, "book_5m_100ms.parquet")
STRIKES = os.path.join(DATA, "strikes_5m.parquet")
RESULTS = os.path.join(DATA, "results")
SPOT = os.path.join(DATA, "spot_5m_100ms.parquet")
FAIR_DIR = os.path.join(DATA, "fair")
INVESTIGATIONS = os.path.join(PROJECT, "investigations")

#: Harness-owned scratch. Everything under `data/` is read-only input to this
#: harness, so anything the harness generates about a data file -- the cached
#: input fingerprints, for one -- belongs here and never beside the file it
#: describes.
CACHE = os.path.join(HERE, ".cache")
FINGERPRINTS = os.path.join(CACHE, "fingerprints")

VENUE_L1 = r"Z:\parquet\stream_venue_l1"

#: EXTERNAL, READ-ONLY. Chainlink Data Streams for BTC, 1 s cadence, from a
#: sibling repo (Gambling102) -- not copied into this repo's `data/` because
#: it belongs to that project's own capture, not this one's. `twap60` is the
#: 60 s Chainlink TWAP: the exact settlement variable for this era, i.e. what
#: `fair.py`'s E[A] is forecasting. Coverage is 2026-08-14 02:54 UTC through
#: 2026-08-21 01:59 UTC only -- it does not reach the back half of this
#: harness's 6-day evaluation sample (08-19..08-24), so only 08-19 and 08-20
#: have a Chainlink line available.
RTDS_BTC = (r"C:\Users\kaima\OneDrive\Documents\GitHub\Gambling102\research"
           r"\btc\backtesting\2026-08-23_backtesting_5m\data\london"
           r"\rtds_btc.parquet")

H = 300
BUCKET_MS = 100
N_BUCKET = H * 1000 // BUCKET_MS      # 3000
TICK = 0.01
