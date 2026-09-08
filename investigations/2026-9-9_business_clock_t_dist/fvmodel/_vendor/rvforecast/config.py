"""Global configuration: paths, date range, horizon grid, seeds.

Nothing fitted lives here — fitted numbers go to output/params.json.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
DATA_DIR = OUTPUT / "data"
DAYS_DIR = DATA_DIR / "days"
TABLES = OUTPUT / "tables"
REPORT = OUTPUT / "report"
IMG = REPORT / "img"
CACHE = OUTPUT / "cache"

for _p in (OUTPUT, DATA_DIR, DAYS_DIR, TABLES, REPORT, IMG, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

BARS_PARQUET = Path(os.environ.get("RVF_BARS",
                                   str(DATA_DIR / "btcusdt_1s.parquet")))
PARAMS_JSON = OUTPUT / "params.json"

SEED = 20260907

# ----------------------------------------------------------------------------- data
SYMBOL = "BTCUSDT"
MARKET = "futures/um"  # USDT-margined perpetual
# The 12 full calendar months ending at the last complete month as of 2026-09-07.
# Every date below can be overridden by an environment variable so that run_all.py
# --quick can exercise the whole pipeline on a short window.
START_DATE = os.environ.get("RVF_START_DATE", "2025-09-01")
END_DATE = os.environ.get("RVF_END_DATE", "2026-08-31")  # inclusive, UTC
BINANCE_BASE = "https://data.binance.vision/data"

# gap_flag bit field on the 1s bars
GAP_NO_TRADE = 1   # no print inside this second; close forward-filled
GAP_STALE = 2      # return into this second spans > MAX_STALE_S without a print
GAP_MISSING = 4    # source data absent for this second (missing file / outage)
MAX_STALE_S = 5

# ------------------------------------------------------------------------ modelling
SEC_PER_DAY = 86400.0
SEC_PER_YEAR = 31_536_000.0

HORIZONS = [10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400]
HORIZON_LABELS = ["10s", "30s", "1m", "2m", "5m", "10m", "15m", "30m", "1h", "2h", "4h"]

# regression sampling grid (seconds between forecast origins)
GRID_STEP_S = 10

# registers: K half-lives, geometric, in business days (1 unit = one average day)
N_REGISTERS = 8
HL_MIN_BDAYS = 30.0 / SEC_PER_DAY      # 30 business-seconds
HL_MAX_BDAYS = 14.0                    # 2 business-weeks

BURN_IN_DAYS = 14  # discarded from every fit

# binary-market task
MARKET_LENGTHS = [300, 900, 3600, 14400]
MARKET_LENGTH_LABELS = ["5m", "15m", "1h", "4h"]

# ------------------------------------------------------------------- split protocol
# 12 months: 1 = 2025-09 ... 12 = 2026-08
WF_FIRST_EVAL_MONTH = int(os.environ.get("RVF_WF_FIRST", 7))
TUNING_EVAL_MONTHS = tuple(int(x) for x in
                           os.environ.get("RVF_TUNING_MONTHS", "7,8,9,10").split(","))
N_MONTHS = int(os.environ.get("RVF_N_MONTHS", 12))
HOLDOUT_START = os.environ.get("RVF_HOLDOUT_START", "2026-07-20")
SEASONALITY_FREEZE_END = os.environ.get("RVF_FREEZE_END", "2026-03-01")


@dataclass
class Config:
    n_jobs: int = field(default_factory=lambda: min(12, os.cpu_count() or 4))
    quick: bool = False  # set by run_all.py --quick for smoke tests


CONFIG = Config()
