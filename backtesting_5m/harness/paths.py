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

VENUE_L1 = r"Z:\parquet\stream_venue_l1"

H = 300
BUCKET_MS = 100
N_BUCKET = H * 1000 // BUCKET_MS      # 3000
TICK = 0.01
