"""Where the harness reads and writes. Mirrors data/scripts/common.py."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, os.pardir))

DATA = os.path.join(PROJECT, "data")
PANEL = os.path.join(DATA, "book_5m_100ms.parquet")
STRIKES = os.path.join(DATA, "strikes_5m.parquet")
RESULTS = os.path.join(DATA, "results")
#: The spot panel, on a BTC/**USD** basis -- the venue's BTC/USDT mid less
#: the capture's `usdt_basis`. This is the DEFAULT because these markets
#: settle on Chainlink's BTC/USD 60 s TWAP, so the panel must be quoted in
#: the same currency as the thing being forecast.
#:
#: Measured against the Chainlink oracle over 1.65 M buckets: the legacy
#: BTC/USDT panel sat +$43.17 high (sd 16.36); this one sits +$4.50
#: (sd 7.44). The old bias was also TIME-VARYING -- day-means drifting
#: ~$40 to ~$10 across the sample -- so no fitted intercept could absorb
#: it. See harness/build/spot_5m_100ms.py.
SPOT = os.path.join(DATA, "spot_5m_100ms_usd.parquet")
SPOT_USD = SPOT                      # explicit alias; same file

#: The spot panel rebuilt over the ORACLE OVERLAP, 2026-08-17..08-21. Same
#: build, same USD basis correction, same columns as `SPOT`; only the window
#: differs. It exists because the three feeds this harness needs do not span
#: the same days:
#:
#:     book panel   2026-08-14 -> 09-08
#:     venue L1     2026-08-17 -> 09-09
#:     RTDS oracle  2026-08-14 -> 08-21 01:59
#:     ------------------------------------
#:     overlap      2026-08-17 -> 08-21 01:59
#:
#: `SPOT` covers 2026-08-19..24, of which only ~2.1 days have an oracle line,
#: and a basis-learning fair block returns NaN without one -- so a six-day
#: headline had collapsed to a two-day one. This window carries 3.90 days of
#: oracle-covered spot against that 2.08: 3,281,366 buckets against
#: 1,759,022, a factor of 1.87.
#:
#: Two partial days at the ends, both from the source captures and not from
#: this build: `stream_venue_l1` does not start until 2026-08-17 04:19 UTC
#: (236 markets that day, not 288), and the oracle stops at 2026-08-21 01:59,
#: so 08-21's spot is complete but only its first ~2 h can be scored against
#: Chainlink.
SPOT_ORACLE_WINDOW = os.path.join(DATA, "spot_5m_100ms_usd_0817_0821.parquet")

#: The superseded BTC/USDT panel. Kept only so a historical run folder can
#: be reproduced against the data it actually used. Do not build on it.
SPOT_LEGACY_USDT = os.path.join(DATA, "spot_5m_100ms.parquet")
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
